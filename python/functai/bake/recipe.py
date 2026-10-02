"""The training settings bake starts from, read from the data and the student.
Every one can be set by hand (``bake(..., lr=, epochs=, lora_rank=, batch=)``);
these are the defaults and why.

- LoRA on every linear layer, alpha 32, no dropout, rank by how much there
  is to learn (``students.lora_rank``). "LoRA Without Regret" (Thinking
  Machines, 2025) measured it matching full fine-tuning on supervised sets
  of this size; every place trains it (Tinker trains nothing else); and the
  adapter is small enough to keep each checkpoint.
- The learning rate from Tinker's fitted rule (``students.learning_rate``).
- Batches of about 8 to 32 examples per optimizer step (about 100 steps a
  pass on small sets): LoRA loses more than full fine-tuning at large
  batches (same paper).
- Passes: one past 20M training tokens, two in between, three under 1,000
  rows; validation loss is watched, and with several passes the best
  checkpoint is kept.
- The schedule: warmup (3% of steps), constant, then a 20% "1-sqrt" decay
  to zero (WSD; Hägele et al. 2024 found it matches cosine, and its
  constant phase lets a run be extended or branched from any checkpoint).
- Lengths from the data: replies may run to 1.25x the longest answer seen.
"""

from __future__ import annotations

import math
from typing import Any, Dict

from .students import learning_rate, lora_rank


def recipe(job) -> Dict[str, Any]:
    o = job.options
    s = job.stats
    st = job.student
    lora = True if o.get("lora") in (None, "auto") else bool(o.get("lora"))
    answer_tokens = s.get("answer_tokens", 0)
    rank = int(o.get("lora_rank") or lora_rank(answer_tokens))
    lr = float(o.get("lr") or learning_rate(st, lora=lora))
    n = max(1, job.train_rows)
    tokens = s.get("tokens", 0)
    epochs = int(o.get("epochs") or (1 if tokens > 20e6 else 3 if n < 1000 else 2))
    per_step = int(o.get("batch") or max(1, min(n, max(8, min(32, round(n / 100))))))
    steps = math.ceil(n / per_step) * epochs
    longest = int(s.get("longest", 0))
    answer_max = int((s.get("answer") or {}).get("max", 0))
    prompt_max = int((s.get("prompt") or {}).get("max", 0))
    max_new = int(o.get("max_new_tokens") or max(64, math.ceil(answer_max * 1.25)))
    model_len = int(math.ceil((prompt_max + max_new) / 1024) * 1024)
    if st.context:
        model_len = min(model_len, st.context)
    return {
        "lora": lora, "lora_rank": rank if lora else None, "lora_alpha": 32 if lora else None, "lora_dropout": 0.0,
        "learning_rate": lr, "epochs": epochs, "examples_per_step": per_step, "steps": steps,
        "schedule": {"type": "wsd", "warmup": 0.03, "decay": 0.2, "decay_shape": "1-sqrt"},
        "eval_every": 0.05, "save_every": 0.1, "patience": 3 if epochs > 1 else None,
        "longest": longest, "max_new_tokens": max_new, "max_model_len": model_len,
        "seed": int(o.get("seed") or 0), "weight_decay": 0.0, "max_grad_norm": 1.0,
    }


__all__ = ["recipe"]
