"""Generative students: a small chat model trained to write an AI function's
answers, in the layout it will be called with.

    baked = summarize.bake(rows, method="sft", student="Qwen/Qwen3.5-0.8B", teacher="claude-opus-5.5")
    fast = summarize.using(lm=baked)

For functions whose outputs are open (text, numbers, structured values), or
when the answer must come with the function's own reasoning. The training
examples are the exact requests the function's lmcc layout writes, and the
target is the reply that layout writes for the right answer: the same writer
lays out the calls the student will get, so training and use cannot drift.
The student learns the answer only; a teacher's reasoning is not copied
unless the function asks for reasoning itself (module="cot").

The trained model is saved with that layout (``baked.layout``) and always
reads its calls through it, whatever the function's adapter says later.
Served in-process with transformers (batched greedy decoding), or by vLLM
(``baked.serve()``) for throughput.
"""

from __future__ import annotations

import json
import math
import os
import random
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import lm15
import lmcc

from .examples import BakeError


def _torch():
    from .heads import _torch as t
    return t()


# ------------------------------------------------------------------ chat conversion


def chat_messages(request: Dict[str, Any]) -> List[Dict[str, str]]:
    """An lm15 request (canonical JSON) as chat-template messages."""
    out: List[Dict[str, str]] = []
    system = request.get("system")
    if system:
        out.append({"role": "system", "content": system if isinstance(system, str) else _text(system)})
    for m in request.get("messages", []):
        role = m["role"]
        if role not in ("user", "assistant", "system", "developer"):
            raise BakeError(f"a generative student reads text chats; the request has a {role!r} message "
                            f"(tool calls are not trained yet)")
        out.append({"role": "system" if role == "developer" else role, "content": _text(m)})
    return out


def _text(message: Any) -> str:
    parts = message.get("parts", []) if isinstance(message, dict) else message
    texts = []
    for p in parts:
        if p.get("type") != "text":
            raise BakeError(f"a generative student reads text; the request carries a {p.get('type')} part")
        texts.append(p["text"])
    return "".join(texts)


def template_kwargs(tokenizer) -> Dict[str, Any]:
    """Qwen3-style templates think by default; the student answers directly."""
    return {"enable_thinking": False} if "enable_thinking" in (tokenizer.chat_template or "") else {}


def prompt_ids(tokenizer, messages: List[Dict[str, str]], kwargs: Dict[str, Any]) -> List[int]:
    return list(tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                              return_dict=False, **kwargs))


def example_ids(tokenizer, messages: List[Dict[str, str]], answer: str,
                kwargs: Dict[str, Any]) -> Tuple[List[int], int]:
    """Token ids of prompt + answer (+ the template's end of turn), and where the
    answer starts: the loss is on the answer only."""
    prompt = prompt_ids(tokenizer, messages, kwargs)
    full = list(tokenizer.apply_chat_template(messages + [{"role": "assistant", "content": answer}],
                                              tokenize=True, return_dict=False, **kwargs))
    if full[:len(prompt)] != prompt:        # a template that rewrites earlier turns: concatenate instead
        eos = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
        full = prompt + tokenizer(answer, add_special_tokens=False)["input_ids"] + eos
    return full, len(prompt)


# ------------------------------------------------------------------ training data


def _layout(fn, layout: Any):
    from .. import adapters
    settings = fn._effective()
    if layout is not None:
        return adapters.resolve_adapter(layout) if not isinstance(layout, (list, tuple)) else \
            adapters.template_adapter(layout)
    if fn._template is not None:
        return adapters.template_adapter(fn._template)
    return adapters.resolve_adapter(settings.get("adapter") or "xml")


def bake(fn, data: Any, *, student: str, teacher: Any = None, labels: str = "auto", test: Any = None,
         holdout: float = 0.1, validation: float = 0.1, epochs: Optional[int] = None, lr: Optional[float] = None,
         batch_size: int = 8, device: Optional[str] = None, seed: int = 0, num_threads: int = 16,
         path: Any = None, local_files_only: bool = False, log: Callable[[str], None] = lambda s: None,
         lora: Any = "auto", layout: Any = None, reasoning: bool = False, max_new_tokens: Optional[int] = None,
         accumulate: int = 4):
    """Train a chat model to answer ``fn`` (see ``functai.bake.bake`` for the shared options).

    - ``lora``: ``True`` trains a small adapter (merged into the weights when
      saved), ``False`` all weights; ``"auto"``: all weights when they fit in the
      free GPU memory, else LoRA.
    - ``layout``: the lmcc adapter (or a template) the student is trained and
      called with; default the function's own. A short layout is cheaper at scale.
    - ``reasoning``: keep the function's hidden reasoning output (module="cot") as
      part of the answer the student learns to write.
    """
    from . import heads
    from .baked import Baked, default_home, write_meta
    from ..core import FunctAIFunc
    from ..evaluation import parallel, rows_of
    from . import _gold_values, _split
    from .examples import row_inputs
    if fn._tools:
        raise BakeError(f"{fn.__name__} uses tools; training tool-calling students is not supported yet. "
                        f"Reinforcement learning on Prime (functai_verifiers) runs the tool loop")
    s = fn._effective()
    cot = reasoning and s.get("module") == "cot"
    spec = fn._variant_spec(reasoning=cot, tools=False)
    adapter = _layout(fn, layout)
    adapter = lmcc.adapter(name=adapter.name, messages=adapter.template, reader=adapter.reader,
                           transports=adapter.transports, formats=adapter.formats, extensions=adapter.extensions,
                           replay="values", strict=adapter.strict) if adapter.replay != "values" else adapter
    capabilities = {"instruct": True}
    from .. import adapters as _adapters
    plan = adapter.bind(spec.signature, capabilities, registry=_adapters.REGISTRY)
    outputs = [f.name for f in spec.signature.outputs if f.purpose != "tools.calls"]
    rows = rows_of(data)
    test_rows = rows_of(test) if test is not None else None
    notes: List[str] = []

    gold = [_gold_values(r, outputs) for r in rows]
    labeled = [i for i, g in enumerate(gold) if g is not None]
    if test_rows is None:
        _pool, held = _split(labeled, holdout, 20, 500, seed)
        test_rows = [rows[i] for i in held]
        train_idx = [i for i in range(len(rows)) if i not in set(held)]
    else:
        train_idx = list(range(len(rows)))
    if any(_gold_values(r, outputs) is None for r in test_rows):
        raise BakeError(f"every test row needs the outputs {outputs}")
    need_teacher = labels == "teacher" or (labels == "auto" and any(gold[i] is None for i in train_idx))
    if need_teacher and teacher is None:
        raise BakeError(f"rows lack the outputs {outputs} and no teacher was given: pass teacher=")
    targets: Dict[int, Dict[str, Any]] = {i: gold[i] for i in train_idx if gold[i] is not None and labels != "teacher"}
    labeling: Dict[str, Any] = {}
    to_label = [i for i in train_idx if i not in targets] if labels != "data" else []
    if labels == "data":
        train_idx = [i for i in train_idx if i in targets]
    if to_label:
        tfn = teacher if isinstance(teacher, FunctAIFunc) else fn.using(lm=teacher)
        log(f"labeling {len(to_label):,} rows with {getattr(teacher, '__name__', teacher)}")
        t0 = time.time()

        def one(i: int) -> Optional[Dict[str, Any]]:
            try:
                pred = tfn._invoke((), row_inputs(fn, rows[i]), full=True)
            except Exception:  # noqa: BLE001 — a failed row is dropped and counted
                return None
            return {k: pred[k] for k in outputs if k in pred}
        got = parallel(one, to_label, max(1, num_threads))
        for i, g in zip(to_label, got):
            if g is not None and set(g) == set(outputs):
                targets[i] = g
        train_idx = [i for i in train_idx if i in targets]
        labeling = {"teacher": str(getattr(teacher, "__name__", teacher)), "rows": sum(g is not None for g in got),
                    "failed": sum(g is None for g in got), "seconds": round(time.time() - t0, 1)}
    if len(train_idx) < 10:
        raise BakeError(f"only {len(train_idx)} training rows; a generative student needs at least 10")
    train_idx, val_idx = _split(train_idx, validation, 16, 500, seed + 1) if len(train_idx) >= 60 \
        else (train_idx, [])

    # ---- the exact requests and replies the layout writes
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(student, local_files_only=local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not tokenizer.chat_template:
        raise BakeError(f"{student} has no chat template; use an instruct/chat model")
    kwargs = template_kwargs(tokenizer)
    demos = fn._past(plan, spec, {**s, "stateful": False})

    def encode_row(i: int) -> Tuple[List[int], int]:
        from .. import engine
        inputs = engine.prepare_inputs(spec, row_inputs(fn, rows[i]))
        example = plan.example(inputs, targets[i])
        request = plan.render(plan.turn(inputs), turns=demos + [example]).request("student")
        msgs = chat_messages(request)
        # the last user message is the current call's; the example's reply is the one before it
        answer_at = max(k for k, m in enumerate(msgs) if m["role"] == "assistant")
        prompt = msgs[:answer_at]
        return example_ids(tokenizer, prompt, msgs[answer_at]["content"], kwargs)

    log("writing the training conversations")
    train_ex = [encode_row(i) for i in train_idx]
    val_ex = [encode_row(i) for i in val_idx]
    answer_lengths = sorted(len(ids) - start for ids, start in train_ex)
    new_tokens = max_new_tokens or int(min(4096, max(16, math.ceil(answer_lengths[int(0.99 * (len(answer_lengths) - 1))] * 1.5))))

    # ---- the model
    torch = _torch()
    from transformers import AutoModelForCausalLM
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
    log(f"loading {student}")
    model = AutoModelForCausalLM.from_pretrained(student, local_files_only=local_files_only, dtype=torch.float32)
    n_params = sum(p.numel() for p in model.parameters())
    full_need = heads.training_memory_gb(n_params)
    lora_need = n_params * 4 / 2 ** 30 * 1.2 + 1.5
    dev = heads.choose_device(device, lora_need)
    free = _free_gb(dev)
    use_lora = lora if isinstance(lora, bool) else (dev == "cpu" and n_params > 300e6) or (free is not None and free < full_need)
    if use_lora:
        try:
            import peft
        except ImportError as err:
            raise ImportError('LoRA training needs peft: pip install "functai[bake]"') from err
        model = peft.get_peft_model(model, peft.LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                                           target_modules="all-linear", task_type="CAUSAL_LM"))
        notes.append("trained a LoRA adapter (merged into the saved weights): the full model did not fit "
                     "in the free GPU memory" if lora == "auto" else "trained a LoRA adapter (merged when saved)")
    lr = lr or (2e-4 if use_lora else 2e-5)
    epochs = epochs or min(20, max(2, round(3 * 2000 / max(1, len(train_ex)))))
    log(f"training {'a LoRA adapter on ' if use_lora else ''}{n_params / 1e6:,.0f}M parameters on {dev}: "
        f"{len(train_ex):,} conversations, up to {epochs} passes, lr {lr:g}")
    result = _train(model, tokenizer, train_ex, val_ex, epochs=epochs, lr=lr, batch_size=batch_size,
                    accumulate=accumulate, device=dev, seed=seed, log=log)
    if use_lora:
        model = model.merge_and_unload()

    # ---- write it, then test through the function itself
    name = f"{fn.__name__}-{Path(student).name}-{time.strftime('%Y%m%d-%H%M%S')}"
    out = Path(path).expanduser().resolve() if path else default_home() / name
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} exists and is not empty")
    out.mkdir(parents=True, exist_ok=True)
    model.to("cpu").save_pretrained(str(out / "model"), safe_serialization=True)
    tokenizer.save_pretrained(str(out / "tokenizer"))
    tokenizer.save_pretrained(str(out / "model"))        # a standard model folder: vLLM, transformers, TGI
    meta = {"name": fn.__name__, "kind": "generative", "student": student, "layout": adapter.dump(),
            "signature": lmcc.signature_to_dict(spec.signature),
            "fingerprint": lmcc.signature_fingerprint(spec.signature), "reasoning": cot,
            "capabilities": capabilities, "chat_template_kwargs": kwargs, "max_new_tokens": new_tokens,
            "parameters": n_params, "lora": bool(use_lora)}
    write_meta(out, meta)
    baked = Baked(out, device=dev)
    baked._model, baked._tokenizer = _for_inference(model, tokenizer, dev)
    log(f"testing on {len(test_rows):,} rows")
    report = _report(fn, baked, test_rows, outputs, teacher=teacher, labeling=labeling, rows=(len(train_ex),
                     len(val_ex)), result=result, device=dev, n_params=n_params, notes=notes,
                     num_threads=num_threads, log=log)
    meta["report"] = report.to_dict()
    write_meta(out, meta)
    baked.meta = json.loads((out / "baked.json").read_text())
    baked._report = report
    log(f"saved to {out}")
    return baked


def _free_gb(device: str) -> Optional[float]:
    if not device.startswith("cuda"):
        return None
    torch = _torch()
    free, _total = torch.cuda.mem_get_info(int(device.split(":")[1]))
    return free / 2 ** 30


def _pad(batch: Sequence[Tuple[List[int], int]], pad: int, device: str):
    torch = _torch()
    width = max(len(ids) for ids, _ in batch)
    input_ids = torch.full((len(batch), width), pad, dtype=torch.long)
    mask = torch.zeros((len(batch), width), dtype=torch.long)
    labels = torch.full((len(batch), width), -100, dtype=torch.long)
    for i, (ids, start) in enumerate(batch):
        input_ids[i, :len(ids)] = torch.tensor(ids)
        mask[i, :len(ids)] = 1
        labels[i, start:len(ids)] = torch.tensor(ids[start:])
    return input_ids.to(device), mask.to(device), labels.to(device)


def _train(model, tokenizer, train: List[Tuple[List[int], int]], val: List[Tuple[List[int], int]], *, epochs: int,
           lr: float, batch_size: int, accumulate: int, device: str, seed: int, log) -> Dict[str, Any]:
    torch = _torch()
    random.seed(seed)
    torch.manual_seed(seed)
    model.to(device).train()
    if hasattr(model, "gradient_checkpointing_enable") and device.startswith("cuda"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    steps = math.ceil(len(train) / (batch_size * accumulate)) * epochs
    warm = max(1, int(0.05 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    use_amp = device.startswith("cuda") and torch.cuda.is_bf16_supported()
    pad = tokenizer.pad_token_id
    rng = random.Random(seed)
    best = (float("inf"), None, 0)
    history = []
    bad = 0
    t0 = time.time()
    for epoch in range(epochs):
        order = sorted(range(len(train)), key=lambda i: len(train[i][0]))       # similar lengths per batch
        chunks = [order[b:b + batch_size] for b in range(0, len(order), batch_size)]
        rng.shuffle(chunks)
        model.train()
        total, n = 0.0, 0
        optim.zero_grad(set_to_none=True)
        for k, chunk in enumerate(chunks, 1):
            ids, mask, labels = _pad([train[i] for i in chunk], pad, device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                loss = answer_loss(model, ids, mask, labels)
            (loss / accumulate).backward()
            total += loss.item()
            n += 1
            if k % accumulate == 0 or k == len(chunks):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
        val_loss = _val_loss(model, val, pad, device, batch_size) if val else total / max(1, n)
        history.append({"epoch": epoch + 1, "train_loss": total / max(1, n), "val_loss": val_loss,
                        "seconds": round(time.time() - t0, 1)})
        log(f"epoch {epoch + 1}/{epochs}: train loss {total / max(1, n):.4f}, validation loss {val_loss:.4f}")
        if val_loss < best[0] - 1e-4:
            state = model.state_dict()
            trained = [k for k in state if "lora_" in k] or list(state)      # LoRA: keep the adapter only
            best = (val_loss, {k: state[k].detach().to("cpu", copy=True) for k in trained}, epoch + 1)
            bad = 0
        else:
            bad += 1
            if bad >= 2:
                log("stopped early: no better validation loss for 2 passes")
                break
    if best[1] is not None:
        model.load_state_dict(best[1], strict=False)
    model.eval()
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    return {"history": history, "best_epoch": best[2], "passes_run": len(history),
            "seconds": round(time.time() - t0, 1), "mixed_precision": "bf16" if use_amp else "fp32", "lr": lr}


def answer_loss(model, ids, mask, labels):
    """Cross-entropy on the answer tokens only, computing vocabulary scores only
    where there is a label: the prompt is most of the tokens, and a 250k-word
    vocabulary over all of them would not fit on a shared GPU. Same loss as the
    model's own, a fraction of the memory."""
    torch = _torch()
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    decoder = base.get_decoder() if hasattr(base, "get_decoder") else None
    head = base.get_output_embeddings() if hasattr(base, "get_output_embeddings") else None
    if decoder is None or head is None:
        return model(input_ids=ids, attention_mask=mask, labels=labels).loss
    hidden = decoder(input_ids=ids, attention_mask=mask).last_hidden_state
    target = labels[:, 1:]
    keep = target != -100
    logits = head(hidden[:, :-1][keep])
    return torch.nn.functional.cross_entropy(logits.float(), target[keep])


def _val_loss(model, val, pad, device, batch_size) -> float:
    torch = _torch()
    model.eval()
    total, n = 0.0, 0
    use_amp = device.startswith("cuda") and torch.cuda.is_bf16_supported()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
        for b in range(0, len(val), batch_size):
            ids, mask, labels = _pad(val[b:b + batch_size], pad, device)
            total += answer_loss(model, ids, mask, labels).item()
            n += 1
    return total / max(1, n)


# ------------------------------------------------------------------ running


def _for_inference(model, tokenizer, device: str):
    torch = _torch()
    dtype = torch.bfloat16 if device.startswith("cuda") and torch.cuda.is_bf16_supported() else torch.float32
    tokenizer.padding_side = "left"
    return model.to(device=device, dtype=dtype).eval(), tokenizer


def load(path: str, device: str):
    torch = _torch()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
    try:
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(path, "tokenizer"))
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(os.path.join(path, "model"), dtype=torch.float32)
    finally:
        hf_logging.enable_progress_bar()
    return _for_inference(model, tokenizer, device)


def generate_batch(baked, items: Sequence[Tuple[List[int], int]]) -> List[str]:
    """Greedy replies for (prompt ids, max new tokens) items, batched with left padding."""
    torch = _torch()
    baked._ensure()
    model, tokenizer = baked._model, baked._tokenizer
    width = max(len(ids) for ids, _ in items)
    pad = tokenizer.pad_token_id
    input_ids = torch.full((len(items), width), pad, dtype=torch.long)
    mask = torch.zeros((len(items), width), dtype=torch.long)
    for i, (ids, _) in enumerate(items):
        input_ids[i, width - len(ids):] = torch.tensor(ids)
        mask[i, width - len(ids):] = 1
    device = next(model.parameters()).device
    with torch.inference_mode():
        out = model.generate(input_ids=input_ids.to(device), attention_mask=mask.to(device), do_sample=False,
                             max_new_tokens=max(n for _, n in items), pad_token_id=pad)
    texts = []
    for i, (_ids, n) in enumerate(items):
        new = out[i, width:width + n].tolist()
        texts.append(tokenizer.decode(new, skip_special_tokens=True))
    return texts


def complete(baked, request: Any) -> Any:
    """Answer an lm15 request with the student: in-process, or through the vLLM
    server ``serve()`` started."""
    if baked.endpoint is not None:
        lm = lm15.OpenAIChatLM(api_key="functai", base_url=baked.endpoint, compat="vllm")
        extensions = {"chat_template_kwargs": baked.meta.get("chat_template_kwargs") or {}}
        config = request.config or lm15.Config()
        import dataclasses
        # greedy unless asked otherwise: the same decoding as in-process (a server's default is sampling)
        config = dataclasses.replace(config, extensions={**(config.extensions or {}), **extensions},
                                     response_format=None, max_tokens=config.max_tokens or baked.meta["max_new_tokens"],
                                     temperature=0.0 if config.temperature is None else config.temperature)
        return lm.complete(dataclasses.replace(request, model=baked.name, config=config))
    baked._ensure()
    msgs = chat_messages(lm15.serde.request_to_dict(request))
    ids = prompt_ids(baked._tokenizer, msgs, baked.meta.get("chat_template_kwargs") or {})
    limit = (request.config.max_tokens if request.config and request.config.max_tokens else None) \
        or baked.meta["max_new_tokens"]
    text = baked._batcher.submit((ids, int(limit)))
    n_out = len(baked._tokenizer(text, add_special_tokens=False)["input_ids"])
    return lm15.Response(id=None, model=baked.model, message=lm15.Message.assistant([lm15.TextPart(text)]),
                         finish_reason="stop", usage=lm15.Usage(input_tokens=len(ids), output_tokens=n_out,
                                                                 total_tokens=len(ids) + n_out))


def serve(baked, *, python: Optional[str] = None, port: Optional[int] = None, gpu_memory: Optional[float] = None,
          timeout: float = 600, extra_args: Sequence[str] = ()) -> str:
    """Serve the student with vLLM (throughput: continuous batching) and send
    calls there. ``python``: an interpreter with vLLM installed (default this
    one). ``gpu_memory``: the share of the GPU vLLM may take (default: what is
    free now, minus a margin). Returns the endpoint; ``baked.stop()`` ends it."""
    python = python or sys.executable
    if port is None:
        import socket
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
    if gpu_memory is None:
        torch = _torch()
        dev = int(baked.device.split(":")[1]) if baked.device.startswith("cuda") else 0
        free, total = torch.cuda.mem_get_info(dev)
        gpu_memory = max(0.05, round(free / total - 0.05, 2))
        env_dev = str(dev)
    else:
        env_dev = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    baked._model = None                           # the server holds the weights, not this process
    cmd = [python, "-m", "vllm.entrypoints.openai.api_server", "--model", str(baked.path / "model"),
           "--served-model-name", baked.name, "--port", str(port),
           "--gpu-memory-utilization", str(gpu_memory), "--max-model-len", "4096", *extra_args]
    log_path = baked.path.parent / f".{baked.path.name}.vllm.log"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": env_dev}
    if "TRITON_LIBCUDA_PATH" not in env and not Path("/sbin/ldconfig").exists() and \
            Path("/run/opengl-driver/lib/libcuda.so").exists():
        env["TRITON_LIBCUDA_PATH"] = "/run/opengl-driver/lib"     # NixOS: no ldconfig for Triton to ask
    import shutil
    if "VLLM_USE_FLASHINFER_SAMPLER" not in env and shutil.which("nvcc") is None and \
            not Path(env.get("CUDA_HOME", "/usr/local/cuda")).exists():
        env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"     # FlashInfer's sampler compiles with nvcc at startup
    # its own process group: stop() ends vLLM's engine subprocess too
    proc = subprocess.Popen(cmd, stdout=open(log_path, "w"), stderr=subprocess.STDOUT, env=env, start_new_session=True)
    endpoint = f"http://127.0.0.1:{port}/v1"
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM exited ({proc.returncode}); its log: {log_path}\n"
                               + Path(log_path).read_text()[-2000:])
        try:
            urllib.request.urlopen(f"{endpoint}/models", timeout=2).read()
            baked.endpoint, baked._server = endpoint, proc
            import atexit
            atexit.register(baked.stop)          # a server this process started does not outlive it
            return endpoint
        except OSError:
            time.sleep(2)
    proc.terminate()
    raise TimeoutError(f"vLLM did not answer in {timeout:.0f} s; its log: {log_path}")


# ------------------------------------------------------------------ the report


def _report(fn, baked, test_rows, outputs, *, teacher, labeling, rows, result, device, n_params, notes,
            num_threads, log):
    from .metrics import wilson
    from .report import BakeReport, FieldScores
    from ..evaluation import parallel
    from .examples import row_inputs
    from ..evaluation import _norm
    student_fn = fn.using(lm=baked, retries=0)

    def answer(f, row):
        try:
            return f._invoke((), row_inputs(fn, row), full=True)
        except Exception as exc:  # noqa: BLE001 — an unreadable answer is a wrong one, counted
            return exc

    t0 = time.time()
    preds = parallel(lambda r: answer(student_fn, r), test_rows, max(1, min(num_threads, 64)))
    rps = len(test_rows) / max(1e-9, time.time() - t0)
    teacher_preds = None
    if teacher is not None:
        tfn = teacher if hasattr(teacher, "_fn") else fn.using(lm=teacher)
        teacher_preds = parallel(lambda r: answer(tfn, r), test_rows, max(1, num_threads))
    fields = []
    unreadable = sum(isinstance(p, Exception) for p in preds)
    for name in outputs:
        right = [not isinstance(p, Exception) and _norm(p.get(name)) == _norm(r.get(name))
                 for p, r in zip(preds, test_rows)]
        k = sum(right)
        fs = FieldScores(name=name, accuracy=k / len(right) if right else float("nan"),
                         interval=list(wilson(k, len(right))), top3=None, ece=float("nan"), ece_raw=float("nan"),
                         nll=float("nan"), temperature=1.0)
        if teacher_preds is not None:
            tr = [not isinstance(p, Exception) and _norm(p.get(name)) == _norm(r.get(name))
                  for p, r in zip(teacher_preds, test_rows)]
            fs.teacher_accuracy = sum(tr) / len(tr) if tr else None
            agree = [not isinstance(a, Exception) and not isinstance(b, Exception) and _norm(a.get(name)) == _norm(b.get(name))
                     for a, b in zip(preds, teacher_preds)]
            fs.agreement = sum(agree) / len(agree) if agree else None
        fields.append(fs)
    if unreadable:
        notes.append(f"{unreadable} of {len(test_rows)} test replies could not be read in the layout")
    lat = []
    for row in test_rows[:10]:
        t1 = time.perf_counter()
        answer(student_fn, row)
        lat.append((time.perf_counter() - t1) * 1000)
    correct = [all(not isinstance(p, Exception) and _norm(p.get(n)) == _norm(r.get(n)) for n in outputs)
               for p, r in zip(preds, test_rows)]
    import statistics
    return BakeReport(
        function=fn.__name__, student=baked.student, parameters=n_params, device=device, truth="labeled",
        label_source=("teacher" if labeling.get("rows") else "the data's labels") + " (generated answers)",
        teacher=None if teacher is None else str(getattr(teacher, "__name__", teacher)),
        rows={"train": rows[0], "validation": rows[1], "test": len(test_rows)}, training=result, fields=fields,
        coverage=[], confidence=[], correct=correct,
        speed={"device": device, "rows_per_second": rps, "latency_ms": statistics.median(lat) if lat else float("nan")},
        labeling=labeling, notes=notes, max_length=baked.meta["max_new_tokens"])
