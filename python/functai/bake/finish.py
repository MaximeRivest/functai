"""From trained weights to a baked model folder (shared by every trainer)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from .baked import write_meta


def meta_for(plan: Dict[str, Any], weights: Dict[str, Any], *, training: Optional[Dict[str, Any]] = None,
             run_folder: Optional[str] = None) -> Dict[str, Any]:
    s = plan["settings"]
    return {
        "kind": "generative", "name": plan["name"], "student": plan["student"], "functions": plan["functions"],
        "weights": weights, "template": plan["template"],
        "generation": {"max_new_tokens": s["max_new_tokens"], "max_model_len": s["max_model_len"]},
        "lora_rank": s.get("lora_rank"), "parameters": plan.get("student_info", {}).get("parameters"),
        "run": {"folder": run_folder, "where": plan["where"], "fingerprint": plan.get("fingerprint")},
        "training": training or {},
    }


def save_tokenizer(student: str, out: Path, *, local_files_only: bool = False) -> None:
    from .template import load_tokenizer
    tok = load_tokenizer(student, local_files_only=local_files_only)
    tok.save_pretrained(str(out / "tokenizer"))


def merge(student: str, adapter: Path, out: Path, *, dtype: str = "bfloat16", local_files_only: bool = False) -> None:
    """The base weights with the adapter merged in, as a standard Hugging Face
    folder (``out``), made on the CPU so it never competes with training for GPU
    memory."""
    from .heads import _torch
    torch = _torch()
    import peft
    from transformers import AutoModelForCausalLM
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
    base = AutoModelForCausalLM.from_pretrained(student, dtype=getattr(torch, dtype), low_cpu_mem_usage=True,
                                                local_files_only=local_files_only)
    model = peft.PeftModel.from_pretrained(base, str(adapter)).merge_and_unload()
    model.save_pretrained(str(out), safe_serialization=True)
    from .template import load_tokenizer
    load_tokenizer(student, local_files_only=local_files_only).save_pretrained(str(out))


def write_local(plan: Dict[str, Any], adapter: Path, out: Path, *, training: Optional[Dict[str, Any]] = None,
                run_folder: Optional[str] = None, merge_weights: bool = True) -> Path:
    """A baked folder from a LoRA adapter (or full weights) trained here or
    brought here: ``adapter/``, ``model/`` (merged), ``tokenizer/``, ``baked.json``."""
    out.mkdir(parents=True, exist_ok=True)
    local = bool(plan.get("local_files_only"))
    full = (adapter / "config.json").exists() and not (adapter / "adapter_config.json").exists()
    if full:                                     # full fine-tuning: the weights are the model
        if (out / "model").exists():
            shutil.rmtree(out / "model")
        shutil.copytree(adapter, out / "model")
        weights = {"form": "merged", "base": plan["student"], "path": "model"}
    else:
        if (out / "adapter").resolve() != adapter.resolve():
            if (out / "adapter").exists():
                shutil.rmtree(out / "adapter")
            shutil.copytree(adapter, out / "adapter",
                            ignore=shutil.ignore_patterns("optimizer.pt", "scheduler.pt", "rng_state*", "trainer_state.json",
                                                          "training_args.bin", "scaler.pt"))
        if merge_weights:
            merge(plan["student"], out / "adapter", out / "model", local_files_only=local)
            weights = {"form": "merged", "base": plan["student"], "path": "model", "adapter": "adapter"}
        else:
            weights = {"form": "lora", "base": plan["student"], "adapter": "adapter"}
    save_tokenizer(plan["student"], out, local_files_only=local)
    write_meta(out, meta_for(plan, weights, training=training, run_folder=run_folder))
    return out


def write_remote(plan: Dict[str, Any], out: Path, weights: Dict[str, Any], *, training: Optional[Dict[str, Any]] = None,
                 run_folder: Optional[str] = None) -> Path:
    """A baked folder for weights that live on a service (``tinker://``)."""
    out.mkdir(parents=True, exist_ok=True)
    try:
        save_tokenizer(plan["student"], out, local_files_only=bool(plan.get("local_files_only")))
    except Exception:  # noqa: BLE001 — the tokenizer is fetched again when the model runs
        pass
    write_meta(out, meta_for(plan, weights, training=training, run_folder=run_folder))
    return out


def training_summary(run) -> Dict[str, Any]:
    recs = run.records()
    train = [r for r in recs if r.get("loss") is not None]
    evals = [r for r in recs if r.get("eval_loss") is not None]
    out: Dict[str, Any] = {"steps": train[-1]["step"] if train else 0,
                           "seconds": train[-1].get("seconds") if train else None,
                           "train_loss": train[-1]["loss"] if train else None}
    if evals:
        best = min(evals, key=lambda r: r["eval_loss"])
        out.update(eval_loss=evals[-1]["eval_loss"], best_eval_loss=best["eval_loss"], best_step=best.get("step"))
    out["history"] = [{k: r.get(k) for k in ("step", "loss", "eval_loss", "learning_rate", "tokens", "seconds")
                       if r.get(k) is not None} for r in recs]
    return json.loads(json.dumps(out, default=float))


__all__ = ["write_local", "write_remote", "merge", "meta_for", "training_summary"]
