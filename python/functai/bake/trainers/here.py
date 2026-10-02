"""Training on this machine: TRL's ``SFTTrainer`` on bake's own tokens, with
PEFT's LoRA (4-bit base weights when 16-bit ones do not fit), every free GPU
(data parallel, through ``torchrun``), and the precision the hardware has.

Batches are built by tokens, not rows. With a flash-attention kernel and a
model whose every layer is attention, examples are packed into rows of the
batch size with no padding; otherwise (no kernel, or a hybrid model whose
linear-attention layers would carry state across packed examples) batches
are examples of similar length, padded a little. The first batch of every
pass is the largest, so a batch that does not fit fails in seconds; the run
then retries with smaller batches (and 4-bit weights when even one example
does not fit) from where it was.
"""

from __future__ import annotations

import math
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from ..examples import BakeError
from ..hardware import (GB, attention_implementation, devices, fixed_training_bytes, kernels, memory_per_token,
                        micro_batch_tokens, training_seconds)
from ..recipe import recipe
from . import Estimate, Job, Stopped, Trainer

OOM_MARK, STOPPED_MARK, DONE_MARK = "OOM", "STOPPED", "DONE"
CPU_LIMIT = 6e8           # parameters a CPU trains in reasonable time


def _duration(seconds: float) -> str:
    from ..running import _duration as d
    return d(seconds)


class Here(Trainer):
    name = "here"

    def set_up(self) -> bool:
        try:
            import torch  # noqa: F401
            import trl  # noqa: F401
        except ImportError:
            return False
        return True

    def estimate(self, job: Job) -> Estimate:
        try:
            import torch  # noqa: F401
            import trl  # noqa: F401
        except ImportError:
            return Estimate(False, ['training here needs PyTorch and TRL: pip install "functai[bake]"'],
                            summary="not installed")
        r = recipe(job)
        st = job.student
        s = job.stats
        longest = r["longest"]
        lora = r["lora"]
        rank = r["lora_rank"]
        opts = job.options
        accel = [d for d in devices() if d.kind in ("cuda", "mps")]
        if opts.get("device"):
            want = str(opts["device"])
            accel = [d for d in devices() if d.id == want or d.kind == want]
        tokens = s.get("tokens", 0) * r["epochs"]
        context = s.get("tokens", 0) / max(1, s.get("rows", 1))
        if not accel or all(d.kind == "cpu" for d in accel):
            if st.parameters > CPU_LIMIT:
                return Estimate(False, [f"no GPU here, and {st.name} ({st.parameters / 1e9:.1f}B) would take days on "
                                        f"the CPU"], summary="no GPU")
            cpu = devices()[-1]
            secs = training_seconds(st, tokens, devices_=[cpu], lora=lora, context=context, padding=1.2,
                                    efficiency=0.5)
            mean = s.get("tokens", 0) / max(1, s.get("rows", 1))
            micro = max(longest, int(math.ceil(r["examples_per_step"] * mean / 64) * 64))
            settings = {**r, "devices": ["cpu"], "precision": "fp32", "quantize": None, "packing": False,
                        "attention": "sdpa", "micro_tokens": micro, "accumulate": 1, "liger": False, "full": not lora}
            return Estimate(True, [], secs, None, settings, ["training on the CPU: slow, for small students only"],
                            summary=f"the CPU · ≈ {_duration(secs)}")
        quantize = None
        usable = []
        for q in ([None, "4bit"] if accel[0].kind == "cuda" else [None]):
            if q == "4bit" and not kernels()["bitsandbytes"]:
                continue
            if opts.get("quantize") is False and q:
                continue
            if opts.get("quantize") == "4bit" and q is None:
                continue
            usable = [d for d in accel if micro_batch_tokens(st, d, quantize=q, lora_rank=rank if lora else None,
                                                             longest=longest) > 0]
            if usable:
                quantize = q
                break
        if not usable:
            best = max(accel, key=lambda d: d.free_gb)
            need = (fixed_training_bytes(st, quantize="4bit" if kernels()["bitsandbytes"] else None,
                                         lora_rank=rank if lora else None)
                    + longest * memory_per_token(st)) / GB
            return Estimate(False, [f"the longest example ({longest:,} tokens) needs about {need:.0f} GB; the "
                                    f"freest GPU here ({best.name}) has {best.free_gb:.1f} GB free"],
                            summary="does not fit")
        if opts.get("devices"):
            usable = [d for d in usable if d.id in opts["devices"] or str(d.index) in map(str, opts["devices"])]
        if usable[0].kind == "cuda":
            precision = "bf16" if all(d.bf16 for d in usable) else "fp16"
        else:
            precision = "bf16"
        attention = None if st.hybrid else attention_implementation(usable[0])
        if attention and not all(attention_implementation(d) for d in usable):
            attention = None
        packing = bool(attention) and opts.get("packing") is not False
        room = min(micro_batch_tokens(st, d, quantize=quantize, lora_rank=rank if lora else None, longest=longest)
                   for d in usable)
        mean = s.get("tokens", 0) / max(1, s.get("rows", 1))
        # a micro-batch no bigger than one optimizer step's share of examples (short examples would
        # otherwise make a step of hundreds, and a pass of a handful of steps), and never smaller than
        # the longest example
        want = math.ceil(r["examples_per_step"] * mean / len(usable))
        micro = min(room, max(longest, int(math.ceil(want / 64) * 64)))
        accumulate = max(1, math.ceil(r["examples_per_step"] * mean / (micro * len(usable))))
        padding = 1.0 if packing else 1.15
        secs = training_seconds(st, tokens, devices_=usable, lora=lora, context=context, padding=padding)
        # loading the weights, validation passes, and merging the adapter on the CPU at the end
        secs += 45 + st.parameters * 2 / GB * 12 + training_seconds(
            st, s.get("val_tokens", 0) * 20 / 3, devices_=usable, lora=lora, context=context)
        notes: List[str] = []
        if st.hybrid:
            notes.append("not packing: this student has linear-attention layers, whose state would cross from one "
                         "packed example into the next; batches are examples of similar length instead")
            k = kernels()
            if not k["flash_linear_attention"]:
                notes.append("flash-linear-attention is not installed: its layers run on PyTorch's slow reference "
                             'code (pip install "functai[fast]")')
        elif not attention:
            notes.append('not packing: no flash-attention kernel here (pip install "functai[fast]" on a CUDA GPU '
                         'of compute capability 8.0 or more); batches are examples of similar length')
        if quantize:
            why = "as asked" if opts.get("quantize") == "4bit" else "the 16-bit ones do not fit"
            notes.append(f"4-bit base weights (QLoRA, {why}); the adapter is merged into the 16-bit weights at "
                         f"the end, which answer very slightly differently from the 4-bit ones it was trained on")
        # Liger's fused loss only when asked: TRL's chunked loss already keeps the vocabulary's scores
        # off the prompt tokens, and Liger's patches are tested per model family
        liger = bool(kernels()["liger"]) and usable[0].kind == "cuda" and opts.get("liger") is True
        used = (fixed_training_bytes(st, quantize=quantize, lora_rank=rank if lora else None)
                + micro * memory_per_token(st)) / GB
        names = sorted({d.name.replace("NVIDIA ", "").replace("GeForce ", "") for d in usable})
        where = (f"{len(usable)}× " if len(usable) > 1 else "") + " + ".join(names)
        size = f"{micro / 1024:.0f}k" if micro >= 4096 else f"{micro:,}"
        batch = f"{'packed' if packing else 'length-grouped'} {size}-token batches"
        summary = (f"{where}, {batch} · ≈ {_duration(secs)} · fits ({used:.1f} / "
                   f"{min(d.total_gb for d in usable):.0f} GB)")
        settings = {**r, "devices": [d.id for d in usable], "precision": precision, "quantize": quantize,
                    "packing": packing, "attention": attention or "sdpa", "micro_tokens": micro,
                    "accumulate": accumulate, "liger": liger, "full": not lora}
        return Estimate(True, [], secs, None, settings, notes, summary=summary)

    # ---- running

    def supervise(self, run) -> None:
        plan = run.plan
        s = plan["settings"]
        inline = bool(plan.get("inline"))
        while True:
            for mark in (OOM_MARK, STOPPED_MARK, DONE_MARK):
                (run.folder / mark).unlink(missing_ok=True)
            if inline:
                from .here_worker import main as worker
                rc = worker(str(run.folder))
            else:
                rc = self._launch(run, s)
            if (run.folder / DONE_MARK).exists():
                self._finish(run)
                return
            if (run.folder / STOPPED_MARK).exists():
                raise Stopped()
            if (run.folder / OOM_MARK).exists():
                self._shrink(run, s)
                continue
            raise BakeError(f"training stopped with exit code {rc}; see {run.folder / 'log.txt'}")

    def _finish(self, run) -> None:
        """The trained adapter into a baked folder: merged into 16-bit weights on
        the CPU (a standard model folder), the adapter kept beside it."""
        import json
        from ..finish import training_summary, write_local
        plan = run.plan
        run.update(phase="merging the adapter into the weights")
        training = training_summary(run)
        extra = run.folder / "final" / "functai_training.json"
        if extra.exists():
            training.update(json.loads(extra.read_text()))
        training.update(where="here", settings={k: v for k, v in {**plan["settings"],
                                                                  **(run.status.get("overrides") or {})}.items()
                                                if k in ("precision", "quantize", "packing", "micro_tokens",
                                                         "accumulate", "devices", "lora_rank", "learning_rate",
                                                         "epochs", "attention")})
        write_local(plan, run.folder / "final", Path(plan["output"]), training=training, run_folder=str(run.folder),
                    merge_weights=bool(plan.get("merge", True)))
        run.update(state="done", phase=None)

    def _shrink(self, run, s: Dict[str, Any]) -> None:
        over = dict(run.status.get("overrides") or {})
        micro = int(over.get("micro_tokens") or s["micro_tokens"])
        longest = int(s["longest"])
        if micro > longest:
            new = max(longest, int(micro * 0.7))
            over.update(micro_tokens=new, accumulate=max(1, math.ceil(s["accumulate"] * micro / new)))
            run.update(overrides=over, phase=f"out of memory: retrying with {new:,}-token batches")
            return
        if not (over.get("quantize") or s.get("quantize")) and kernels()["bitsandbytes"] and \
                str(s["devices"][0]).startswith("cuda"):
            over["quantize"] = "4bit"
            run.update(overrides=over, phase="out of memory with one example a batch: retrying with 4-bit weights")
            return
        raise BakeError(f"out of memory with one example a batch (the longest has {longest:,} tokens): use a smaller "
                        f"student, leave the longest rows out, or train elsewhere (where='tinker')")

    def _launch(self, run, s: Dict[str, Any]) -> int:
        devs = [d for d in s["devices"] if str(d).startswith("cuda")]
        env = dict(os.environ)
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        from ..hardware import nixos_triton
        nixos_triton()
        if "TRITON_LIBCUDA_PATH" in os.environ:
            env["TRITON_LIBCUDA_PATH"] = os.environ["TRITON_LIBCUDA_PATH"]
        if devs:
            # device numbers are this process's: translate through its own CUDA_VISIBLE_DEVICES
            visible = [v.strip() for v in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if v.strip()]
            idx = [int(str(d).split(":")[1]) for d in devs]
            env["CUDA_VISIBLE_DEVICES"] = ",".join(visible[i] if i < len(visible) else str(i) for i in idx)
        if len(devs) > 1:
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            cmd = [sys.executable, "-m", "torch.distributed.run", f"--nproc_per_node={len(devs)}",
                   f"--master_port={port}", "-m", "functai.bake", "worker", str(run.folder)]
        else:
            cmd = [sys.executable, "-m", "functai.bake", "worker", str(run.folder)]
        log = open(run.folder / "log.txt", "a")
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env,
                                cwd=str(run.folder))
        run.update(worker_pid=proc.pid)
        try:
            return proc.wait()
        except BaseException:
            proc.terminate()
            raise

    def checkpoint_model(self, run, step: int):
        from ..baked import Baked
        from ..finish import write_local
        ck = run.folder / "checkpoints" / f"checkpoint-{step}"
        out = run.folder / "checkpoint-models" / f"step-{step}"
        if not (out / "baked.json").exists():
            plan = run.plan
            plan = {**plan, "name": f"{plan['name']}-step{step}"}
            write_local(plan, ck, out, run_folder=str(run.folder))
        return Baked(out)


__all__ = ["Here"]
