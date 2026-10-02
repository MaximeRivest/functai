"""The training process on this machine (one per GPU under ``torchrun``).

Reads the run's plan and examples, builds batches by tokens, and trains with
TRL's ``SFTTrainer``: its loss computes vocabulary scores only on answer
tokens (``chunked_nll``); Transformers' Trainer brings checkpoints, resuming,
data parallelism and the schedule. Writes metrics as it goes, a checkpoint
on schedule and when asked to stop, and marks how it ended (DONE, STOPPED,
OOM) for the supervisor.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from .here import DONE_MARK, OOM_MARK, STOPPED_MARK


# ------------------------------------------------------------------ the examples


class Tokens:
    """Every example's token ids, as one flat array and offsets (no Python int per token)."""

    def __init__(self, path: Path):
        import numpy as np
        import pyarrow.parquet as pq
        table = pq.read_table(path, columns=["input_ids", "answer_start", "weight", "split"])
        ids = table.column("input_ids").combine_chunks()
        self.values = np.asarray(ids.values.to_numpy(zero_copy_only=False), dtype=np.int64)
        self.offsets = np.asarray(ids.offsets.to_numpy(), dtype=np.int64)
        self.starts = np.asarray(table.column("answer_start").to_numpy(), dtype=np.int64)
        self.weights = [float(w if w is not None else 1.0) for w in table.column("weight").to_pylist()]
        self.splits = table.column("split").to_pylist()

    def length(self, i: int) -> int:
        return int(self.offsets[i + 1] - self.offsets[i])

    def ids(self, i: int):
        return self.values[self.offsets[i]:self.offsets[i + 1]]


def repeat_by_weight(index: Sequence[int], weights: Sequence[float], seed: int) -> List[int]:
    """Each example ``floor(w)`` times, and once more with probability
    ``w - floor(w)``: a weight is how many times it is seen, in expectation."""
    rng = random.Random(seed)
    out: List[int] = []
    for i in index:
        w = weights[i]
        n = int(math.floor(w)) + (1 if rng.random() < w - math.floor(w) else 0)
        out += [i] * n
    return out


def packed_bins(index: Sequence[int], lengths: Dict[int, int], capacity: int) -> List[List[int]]:
    """Best-fit decreasing: examples into bins of at most ``capacity`` tokens."""
    import bisect
    order = sorted(index, key=lambda i: -lengths[i])
    room: List[Tuple[int, int]] = []         # (room left, bin number), sorted
    bins: List[List[int]] = []
    for i in order:
        n = lengths[i]
        k = bisect.bisect_left(room, (n, -1))
        if k < len(room):
            left, b = room.pop(k)
            bins[b].append(i)
            bisect.insort(room, (left - n, b))
        else:
            bins.append([i])
            bisect.insort(room, (capacity - n, len(bins) - 1))
    return bins


def grouped_bins(index: Sequence[int], lengths: Dict[int, int], capacity: int) -> List[List[int]]:
    """Examples of similar length, as many as fit in ``capacity`` tokens with padding."""
    order = sorted(index, key=lambda i: -lengths[i])
    bins: List[List[int]] = []
    cur: List[int] = []
    width = 0
    for i in order:
        w = max(width, lengths[i])
        if cur and w * (len(cur) + 1) > capacity:
            bins.append(cur)
            cur, w = [], lengths[i]
        cur.append(i)
        width = w
    if cur:
        bins.append(cur)
    return bins


class Collate:
    """One micro-batch into model inputs: packed into one row with positions
    restarting at each example, or padded rows with a mask."""

    def __init__(self, tokens: Tokens, bins: List[List[int]], pad: int, packed: bool):
        self.tokens, self.bins, self.pad, self.packed = tokens, bins, pad, packed

    def __call__(self, features):
        examples = [(self.tokens.ids(i), int(self.tokens.starts[i])) for f in features for i in self.bins[f["batch"]]]
        return self.collate(examples)

    def collate(self, examples):
        import torch
        if self.packed:
            ids = torch.cat([torch.as_tensor(x, dtype=torch.long) for x, _ in examples])
            pos = torch.cat([torch.arange(len(x)) for x, _ in examples])
            labels = ids.clone()
            at = 0
            for x, start in examples:
                labels[at:at + start] = -100
                at += len(x)
            return {"input_ids": ids[None], "position_ids": pos[None], "labels": labels[None]}
        width = max(len(x) for x, _ in examples)
        ids = torch.full((len(examples), width), self.pad, dtype=torch.long)
        mask = torch.zeros((len(examples), width), dtype=torch.long)
        labels = torch.full((len(examples), width), -100, dtype=torch.long)
        for r, (x, start) in enumerate(examples):
            t = torch.as_tensor(x, dtype=torch.long)
            ids[r, :len(t)] = t
            mask[r, :len(t)] = 1
            labels[r, start:len(t)] = t[start:]
        return {"input_ids": ids, "attention_mask": mask, "labels": labels}


# ------------------------------------------------------------------ the process


def _rank() -> Tuple[int, int, int]:
    return int(os.environ.get("RANK", 0)), int(os.environ.get("LOCAL_RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))


def main(folder: str) -> int:
    import torch
    root = Path(folder)
    try:
        return _train(root)
    except torch.OutOfMemoryError:
        import traceback
        traceback.print_exc()
        (root / OOM_MARK).write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))
        return 3


def _train(root: Path) -> int:
    import torch
    from transformers import AutoModelForCausalLM, TrainerCallback
    from transformers.trainer_utils import get_last_checkpoint
    from transformers.utils import logging as hf_logging
    from trl import SFTConfig, SFTTrainer

    from ..hardware import nixos_triton
    from ..running import Run
    from ..template import load_tokenizer
    nixos_triton()
    hf_logging.set_verbosity_warning()
    rank, local_rank, world = _rank()
    run = Run(root)
    plan = run.plan
    s = {**plan["settings"], **(run.status.get("overrides") or {})}
    student, local = plan["student"], bool(plan.get("local_files_only"))
    seed = int(s.get("seed") or 0)
    torch.manual_seed(seed)

    # ---- batches by tokens
    tokens = Tokens(root / "examples.parquet")
    n = len(tokens.splits)
    train = [i for i in range(n) if tokens.splits[i] != "validation"]
    val = [i for i in range(n) if tokens.splits[i] == "validation"]
    train = repeat_by_weight(train, tokens.weights, seed)
    lengths = {i: tokens.length(i) for i in set(train) | set(val)}
    micro = int(s["micro_tokens"])
    packed = bool(s.get("packing"))
    make_bins = packed_bins if packed else grouped_bins
    exp_len = {k: lengths[i] for k, i in enumerate(train)}      # a repeated example is packed as another one
    bins = [[train[k] for k in b] for b in make_bins(list(range(len(train))), exp_len, micro)]
    val_bins = make_bins(val, lengths, micro) if val else []
    largest = max(bins, key=lambda b: (max(lengths[i] for i in b) * len(b)) if not packed
                  else sum(lengths[i] for i in b))
    tok = load_tokenizer(student, local_files_only=local)
    accumulate = int(s["accumulate"])
    epochs = int(s["epochs"])
    per_rank = math.ceil(len(bins) / world)
    total = max(1, math.ceil(per_rank / accumulate) * epochs)
    eval_steps = max(1, round(total * float(s.get("eval_every") or 0.05)))
    save_steps = eval_steps * max(1, round(float(s.get("save_every") or 0.1) / float(s.get("eval_every") or 0.05)))
    if rank == 0:
        run.update(steps=total, phase="loading the student", micro_batches=len(bins), micro_tokens=micro)

    # ---- the model
    precision = s.get("precision", "bf16")
    on_gpu = torch.cuda.is_available() and str(s["devices"][0]).startswith("cuda")
    full = not s.get("lora")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16 if not full else torch.float32,
             "fp32": torch.float32}[precision]
    kwargs: Dict[str, Any] = {"dtype": dtype, "local_files_only": local, "attn_implementation": s.get("attention")
                              if s.get("attention") not in (None, "sdpa") else "sdpa"}
    if on_gpu:
        kwargs["device_map"] = {"": local_rank}
    quantize = s.get("quantize")
    if quantize == "4bit":
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if precision == "bf16" else torch.float16)
    model = AutoModelForCausalLM.from_pretrained(student, **kwargs)
    model.config.use_cache = False
    if not full:
        import peft
        if quantize:
            model = peft.prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                                         gradient_checkpointing_kwargs={"use_reentrant": False})
        model = peft.get_peft_model(model, peft.LoraConfig(
            r=int(s["lora_rank"]), lora_alpha=int(s.get("lora_alpha") or 32), lora_dropout=float(s.get("lora_dropout")
                                                                                                  or 0.0),
            target_modules="all-linear", task_type="CAUSAL_LM"))
        model.enable_input_require_grads()

    # ---- the trainer
    out = root / "checkpoints"
    liger = bool(s.get("liger"))
    args = SFTConfig(
        output_dir=str(out), num_train_epochs=epochs, per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=accumulate, learning_rate=float(s["learning_rate"]),
        weight_decay=float(s.get("weight_decay") or 0.0), max_grad_norm=float(s.get("max_grad_norm") or 1.0),
        lr_scheduler_type="warmup_stable_decay", warmup_steps=max(1, round(total * s["schedule"]["warmup"])),
        lr_scheduler_kwargs={"num_decay_steps": max(1, round(total * s["schedule"]["decay"])),
                             "decay_type": s["schedule"].get("decay_shape", "1-sqrt"), "min_lr_ratio": 0.0},
        bf16=precision == "bf16" and on_gpu, fp16=precision == "fp16" and on_gpu,
        gradient_checkpointing=on_gpu, gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=max(1, min(10, eval_steps // 4 or 1)), logging_first_step=True,
        eval_strategy="steps" if val_bins else "no", eval_steps=eval_steps,
        save_strategy="steps", save_steps=save_steps, save_only_model=False,
        save_total_limit=None if not full else 2,
        load_best_model_at_end=bool(val_bins) and epochs > 1, metric_for_best_model="eval_loss",
        greater_is_better=False, seed=seed, data_seed=seed, report_to=list(s.get("report_to") or []),
        remove_unused_columns=False, dataset_kwargs={"skip_prepare_dataset": True}, max_length=None,
        loss_type="nll" if liger else "chunked_nll", use_liger_kernel=liger,
        ddp_find_unused_parameters=False, dataloader_num_workers=0, disable_tqdm=True,
        use_cpu=not on_gpu and str(s["devices"][0]) == "cpu", include_num_input_tokens_seen="non_padding",
    )
    t0 = time.time()

    class Watch(TrainerCallback):
        """Metrics to metrics.jsonl; a stop request honored on every rank at the same step."""
        stopped = False

        def on_log(self, a, state, control, logs=None, **kw):
            if rank != 0 or not logs:
                return
            rec = {"step": state.global_step, "epoch": round(state.epoch or 0, 4), "seconds": round(time.time() - t0, 1)}
            for k_in, k_out in (("loss", "loss"), ("eval_loss", "eval_loss"), ("learning_rate", "learning_rate"),
                                ("grad_norm", "grad_norm"), ("mean_token_accuracy", "token_accuracy"),
                                ("num_input_tokens_seen", "tokens")):
                if k_in in logs:
                    rec[k_out] = logs[k_in]
            if "eval_mean_token_accuracy" in logs:
                rec["eval_token_accuracy"] = logs["eval_mean_token_accuracy"]
            if len(rec) > 3:
                run.log_metric(**rec)

        def on_step_end(self, a, state, control, **kw):
            want = (root / "STOP").exists()
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                flag = torch.tensor([1 if want else 0], device=f"cuda:{local_rank}" if on_gpu else "cpu")
                torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
                want = bool(flag.item())
            if want:
                Watch.stopped = True
                control.should_save = True
                control.should_training_stop = True

        def on_train_begin(self, a, state, control, **kw):
            if rank == 0:
                run.update(phase="training", steps=state.max_steps)

    # TRL takes a datasets.Dataset: one row per micro-batch, holding only its number (the collator looks
    # the examples up); train and validation micro-batches share one numbering
    all_bins = bins + val_bins
    collate = Collate(tokens, all_bins, tok.pad_token_id, packed)
    import datasets
    train_ds = datasets.Dataset.from_dict({"batch": list(range(len(bins)))})
    val_ds = datasets.Dataset.from_dict({"batch": list(range(len(bins), len(all_bins)))}) if val_bins else None
    trainer = SFTTrainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds, processing_class=tok,
                         data_collator=collate, callbacks=[Watch()])

    # ---- the largest batch first: one that does not fit fails now, not hours in
    resume = get_last_checkpoint(str(out)) if out.exists() else None
    if rank == 0:
        run.update(phase="checking memory with the largest batch" if not resume else
                   f"resuming from {Path(resume).name}")
    probe = collate.collate([(tokens.ids(i), int(tokens.starts[i])) for i in largest])
    trainer.model.train()
    probe = {k: v.to(trainer.args.device) for k, v in probe.items()}
    with trainer.compute_loss_context_manager():
        loss = trainer.compute_loss(trainer.model, probe)
    trainer.accelerator.backward(loss)
    trainer.model.zero_grad(set_to_none=True)
    del probe, loss
    if on_gpu:
        torch.cuda.empty_cache()

    trainer.train(resume_from_checkpoint=resume)
    if Watch.stopped:
        if rank == 0:
            (root / STOPPED_MARK).write_text("stopped")
        return 0
    trainer.save_model(str(root / "final"))
    if rank == 0:
        info = {"global_step": trainer.state.global_step, "best_checkpoint": trainer.state.best_model_checkpoint,
                "best_eval_loss": trainer.state.best_metric, "seconds": round(time.time() - t0, 1)}
        (root / "final" / "functai_training.json").write_text(json.dumps(info, default=str))
        (root / DONE_MARK).write_text("done")
    return 0


__all__ = ["main", "packed_bins", "grouped_bins", "repeat_by_weight", "Collate"]
