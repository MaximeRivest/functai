"""Where a generative student's weights answer calls.

    baked.on("transformers")          # in this process (the default for weights here)
    baked.on("vllm")                  # a vLLM server started here, for throughput
    baked.on("tinker")                # Tinker's sampler (the default for weights on Tinker)
    baked.on("http://host:8000/v1")   # any OpenAI-compatible server already serving them

Every runner is called with the student's exact prompt tokens (the ids its
chat template writes, as in training) wherever the server takes token ids:
in-process, vLLM's and SGLang's completions endpoint, Tinker's sampler. A
server that takes only chat messages gets messages and renders them with its
own template; the runner says so (``fidelity``).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import lm15

from .examples import BakeError
from .functions import chat_messages
from .template import prompt_ids


class Runner:
    """Generates replies for prompt token ids."""
    name = "runner"
    fidelity = "tokens"            # "tokens": called with the training tokens; "template": the server renders

    def __init__(self, baked):
        self.baked = baked

    def tokenizer(self):
        return self.baked.tokenizer()

    def generate(self, items: Sequence[Tuple[List[int], int]]) -> List[Tuple[str, int, bool]]:
        """For each (prompt ids, max new tokens): (text, tokens generated, stopped by the template's end)."""
        raise NotImplementedError

    def complete(self, request: Any) -> Any:
        meta = self.baked.meta
        msgs = chat_messages(request)
        tok = self.tokenizer()
        ids = prompt_ids(tok, msgs, self.baked.template.kwargs)
        limit = (request.config.max_tokens if request.config and request.config.max_tokens else None) \
            or meta.get("generation", {}).get("max_new_tokens") or 1024
        text, n_out, stopped = self.submit(ids, int(limit), msgs)
        return lm15.Response(id=None, model=self.baked.model, message=lm15.Message.assistant([lm15.TextPart(text)]),
                             finish_reason="stop" if stopped else "length",
                             usage=lm15.Usage(input_tokens=len(ids), output_tokens=n_out,
                                              total_tokens=len(ids) + n_out))

    def submit(self, ids: List[int], limit: int, messages: Optional[List[Dict[str, str]]] = None
               ) -> Tuple[str, int, bool]:
        return self.generate([(ids, limit)])[0]

    def close(self) -> None:
        pass


# ------------------------------------------------------------------ in this process


class InProcess(Runner):
    """transformers in this process: concurrent calls are batched (greedy, left padding)."""
    name = "transformers"

    def __init__(self, baked, device: Optional[str] = None):
        super().__init__(baked)
        from .baked import _Batcher
        self.device = device
        self.model = None
        self._lock = threading.Lock()
        # a call waits up to 10 ms for others to join its batch: generating takes hundreds of
        # milliseconds, and a batch of 16 costs about what one does
        self._batcher = _Batcher(self.generate, max_batch=32, max_wait=0.01)

    def submit(self, ids, limit, messages=None):
        return self._batcher.submit((ids, limit))

    def _ensure(self):
        if self.model is not None:
            return
        with self._lock:
            if self.model is None:
                self.model = load_model(self.baked, self.device or self.baked.device)

    def generate(self, items):
        from .heads import _torch
        torch = _torch()
        self._ensure()
        model, tok = self.model, self.tokenizer()
        stops = self.baked.template.stop_token_ids
        width = max(len(ids) for ids, _ in items)
        pad = tok.pad_token_id if tok.pad_token_id is not None else stops[0]
        input_ids = torch.full((len(items), width), pad, dtype=torch.long)
        mask = torch.zeros((len(items), width), dtype=torch.long)
        for i, (ids, _) in enumerate(items):
            input_ids[i, width - len(ids):] = torch.tensor(ids)
            mask[i, width - len(ids):] = 1
        device = next(model.parameters()).device
        with torch.inference_mode():
            out = model.generate(input_ids=input_ids.to(device), attention_mask=mask.to(device), do_sample=False,
                                 max_new_tokens=max(n for _, n in items), pad_token_id=pad, eos_token_id=stops,
                                 temperature=None, top_p=None, top_k=None)
        results = []
        for i, (_ids, n) in enumerate(items):
            new = out[i, width:width + n].tolist()
            stopped = False
            for k, t in enumerate(new):
                if t in stops:
                    new, stopped = new[:k], True
                    break
            results.append((tok.decode(new, skip_special_tokens=True), len(new) + int(stopped), stopped))
        return results

    def close(self):
        self.model = None


def load_model(baked, device: str):
    """The student's weights in this process: the merged folder, or the base with
    the adapter; bf16 on a GPU that has it, else fp32."""
    from .heads import _torch
    torch = _torch()
    from transformers import AutoModelForCausalLM
    from transformers.utils import logging as hf_logging
    w = baked.meta["weights"]
    dtype = torch.bfloat16 if device.startswith("cuda") and torch.cuda.is_bf16_supported() else \
        torch.float16 if device.startswith("cuda") else torch.float32
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
    try:
        if w["form"] == "merged":
            model = AutoModelForCausalLM.from_pretrained(str(baked.path / w["path"]), dtype=dtype)
        elif w["form"] == "lora":
            import peft
            base = AutoModelForCausalLM.from_pretrained(w["base"], dtype=dtype)
            model = peft.PeftModel.from_pretrained(base, str(baked.path / w["adapter"]))
        else:
            raise BakeError(f"{baked.name}'s weights are on {w.get('service')} ({w.get('uri')}): run them there "
                            f"(baked.on({w.get('service')!r})) or bring them here first (baked.download())")
    finally:
        hf_logging.enable_progress_bar()
    return model.to(device).eval()


# ------------------------------------------------------------------ OpenAI-compatible servers


def _post(url: str, body: Dict[str, Any], api_key: Optional[str], timeout: float) -> Dict[str, Any]:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": f"Bearer {api_key}"} if api_key else {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


class Endpoint(Runner):
    """An OpenAI-compatible server already serving the student. ``tokens``:
    ``True`` sends prompt token ids to ``/completions`` (vLLM, SGLang),
    ``False`` sends chat messages to ``/chat/completions``, ``"auto"`` tries
    ids once and keeps what works."""
    name = "endpoint"

    def __init__(self, baked, url: str, *, api_key: Optional[str] = None, model: Optional[str] = None,
                 tokens: Any = "auto", timeout: float = 600, concurrency: int = 64):
        super().__init__(baked)
        self.url = url.rstrip("/")
        self.api_key = api_key or os.environ.get("FUNCTAI_BAKED_API_KEY")
        self.served = model or baked.name
        self.tokens = tokens
        self.timeout = timeout
        self._sem = threading.Semaphore(concurrency)
        self.fidelity = "tokens" if tokens is True else "template" if tokens is False else "tokens"

    def generate(self, items):
        return [self._one(ids, n, None) for ids, n in items]

    def submit(self, ids, limit, messages=None):
        return self._one(ids, limit, messages)

    def _one(self, ids: List[int], limit: int, msgs: Optional[List[Dict[str, str]]]) -> Tuple[str, int, bool]:
        stops = self.baked.template.stop_token_ids
        with self._sem:
            if self.tokens is not False:
                try:
                    r = _post(f"{self.url}/completions", {
                        "model": self.served, "prompt": ids, "max_tokens": limit, "temperature": 0.0,
                        "stop_token_ids": stops, "skip_special_tokens": True}, self.api_key, self.timeout)
                    self.tokens, self.fidelity = True, "tokens"
                    c = r["choices"][0]
                    n = (r.get("usage") or {}).get("completion_tokens", 0)
                    return c.get("text", ""), n, c.get("finish_reason") != "length"
                except urllib.error.HTTPError as exc:
                    if self.tokens is True or exc.code not in (400, 404, 422) or msgs is None:
                        raise
                    self.tokens, self.fidelity = False, "template"
            if msgs is None:
                raise BakeError(f"{self.url} takes no token ids, and these calls carry no messages")
            r = _post(f"{self.url}/chat/completions", {
                "model": self.served, "messages": msgs, "max_tokens": limit, "temperature": 0.0,
                "chat_template_kwargs": self.baked.template.kwargs}, self.api_key, self.timeout)
            c = r["choices"][0]
            n = (r.get("usage") or {}).get("completion_tokens", 0)
            return c["message"].get("content") or "", n, c.get("finish_reason") != "length"


class VLLM(Endpoint):
    """A vLLM server started here (continuous batching: throughput). Its own
    process group; ``baked.stop()`` ends it."""
    name = "vllm"

    def __init__(self, baked, *, python: Optional[str] = None, port: Optional[int] = None,
                 gpu_memory: Optional[float] = None, device: Optional[str] = None, timeout: float = 900,
                 extra_args: Sequence[str] = ()):
        endpoint, proc = start_vllm(baked, python=python, port=port, gpu_memory=gpu_memory, device=device,
                                    timeout=timeout, extra_args=extra_args)
        super().__init__(baked, endpoint, tokens=True)
        self.proc = proc
        import atexit
        atexit.register(self.close)

    def close(self):
        if self.proc is None:
            return
        import signal
        try:
            os.killpg(self.proc.pid, signal.SIGTERM)
            self.proc.wait(timeout=60)
        except ProcessLookupError:
            pass
        except Exception:  # noqa: BLE001 — it did not stop in time
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        self.proc = None


def start_vllm(baked, *, python=None, port=None, gpu_memory=None, device=None, timeout=900, extra_args=()):
    from .hardware import devices, nixos_triton
    w = baked.meta["weights"]
    if w["form"] not in ("merged", "lora"):
        raise BakeError(f"{baked.name}'s weights are on {w.get('service')}: baked.download() brings them here first")
    python = python or sys.executable
    if port is None:
        import socket
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
    gpus = [d for d in devices() if d.kind == "cuda"]
    if not gpus:
        raise BakeError("vLLM serves on a CUDA GPU, and there is none here; run in-process (baked.on('transformers'))")
    dev = int(device.split(":")[1]) if device else gpus[0].index
    chosen = next(d for d in gpus if d.index == dev)
    if gpu_memory is None:
        gpu_memory = max(0.05, round(chosen.free_gb / chosen.total_gb - 0.05, 2))
    gen = baked.meta.get("generation", {})
    max_len = int(gen.get("max_model_len") or 8192)
    model_dir = str(baked.path / w["path"]) if w["form"] == "merged" else w["base"]
    cmd = [python, "-m", "vllm.entrypoints.openai.api_server", "--model", model_dir,
           "--served-model-name", baked.name, "--port", str(port), "--gpu-memory-utilization", str(gpu_memory),
           "--max-model-len", str(max_len), "--tokenizer", str(baked.path / "tokenizer"), *extra_args]
    if w["form"] == "lora":
        cmd += ["--enable-lora", "--lora-modules", f"{baked.name}={baked.path / w['adapter']}",
                "--max-lora-rank", str(baked.meta.get("lora_rank", 64))]
    log_path = baked.path.parent / f".{baked.path.name}.vllm.log"
    nixos_triton()
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(dev)}
    import shutil
    if "VLLM_USE_FLASHINFER_SAMPLER" not in env and shutil.which("nvcc") is None and \
            not Path(env.get("CUDA_HOME", "/usr/local/cuda")).exists():
        env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"     # FlashInfer's sampler compiles with nvcc at startup
    proc = subprocess.Popen(cmd, stdout=open(log_path, "w"), stderr=subprocess.STDOUT, env=env,
                            start_new_session=True)
    endpoint = f"http://127.0.0.1:{port}/v1"
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM exited ({proc.returncode}); its log: {log_path}\n"
                               + Path(log_path).read_text()[-2000:])
        try:
            urllib.request.urlopen(f"{endpoint}/models", timeout=2).read()
            return endpoint, proc
        except OSError:
            time.sleep(2)
    proc.terminate()
    raise TimeoutError(f"vLLM did not answer in {timeout:.0f} s; its log: {log_path}")


# ------------------------------------------------------------------ Tinker


class Tinker(Runner):
    """Tinker's sampler, with the student's exact prompt tokens."""
    name = "tinker"

    def __init__(self, baked, *, concurrency: int = 32):
        super().__init__(baked)
        try:
            import tinker
        except ImportError as err:
            raise ImportError('running on Tinker needs its SDK: pip install "functai[tinker]"') from err
        w = baked.meta["weights"]
        uri = w.get("sampler_uri") or w.get("uri")
        if not uri:
            raise BakeError(f"{baked.name} has no Tinker weights to sample from")
        self._tinker = tinker
        self.client = tinker.ServiceClient().create_sampling_client(model_path=uri)
        self._sem = threading.Semaphore(concurrency)

    def generate(self, items):
        t = self._tinker
        stops = self.baked.template.stop_token_ids
        futures = []
        for ids, n in items:
            futures.append(self.client.sample(prompt=t.types.ModelInput.from_ints(list(ids)), num_samples=1,
                                              sampling_params=t.types.SamplingParams(max_tokens=n, temperature=0.0,
                                                                                     stop=list(stops))))
        tok = self.tokenizer()
        out = []
        for (ids, n), f in zip(items, futures):
            seq = f.result().sequences[0]
            toks = list(seq.tokens)
            stopped = bool(toks) and toks[-1] in stops or getattr(seq, "stop_reason", None) == "stop"
            if toks and toks[-1] in stops:
                toks = toks[:-1]
            out.append((tok.decode(toks, skip_special_tokens=True), len(seq.tokens), stopped))
        return out

    def submit(self, ids, limit, messages=None):
        with self._sem:
            return self.generate([(ids, limit)])[0]


def make(baked, where: Any = None, **options) -> Runner:
    """The runner for ``where`` (see the module docstring); None: the default
    for where the weights are."""
    w = baked.meta["weights"]
    if where is None:
        where = "tinker" if w.get("service") == "tinker" and w["form"] == "remote" else "transformers"
    if isinstance(where, Runner):
        return where
    if where == "transformers":
        return InProcess(baked, **options)
    if where == "vllm":
        return VLLM(baked, **options)
    if where == "tinker":
        return Tinker(baked, **options)
    if isinstance(where, str) and where.startswith(("http://", "https://")):
        return Endpoint(baked, where, **options)
    raise BakeError(f"a baked model runs on 'transformers', 'vllm', 'tinker' or an http(s) URL, not {where!r}")


__all__ = ["Runner", "InProcess", "VLLM", "Endpoint", "Tinker", "make", "load_model"]
