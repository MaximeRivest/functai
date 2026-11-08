"""Tinker (Thinking Machines) as a place in a plan: whether it is set up,
whether it trains a student, and what a bake there would cost; and bringing
its weights here (``baked.download()``). Training on Tinker itself is
``functai.bake.service`` (design 21): a student's tokens from its published
files, five methods, checks by the function's criteria.

Set up: ``TINKER_API_KEY`` in the environment (a key from the Tinker console).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from ..examples import BakeError
from ..recipe import recipe
from . import Estimate, Job, Trainer


def _sdk():
    try:
        import tinker
    except ImportError as err:
        raise ImportError('training on Tinker needs its SDK: pip install "functai[tinker]"') from err
    return tinker



class Tinker(Trainer):
    name = "tinker"
    paid = True

    def set_up(self) -> bool:
        if not os.environ.get("TINKER_API_KEY", "").startswith("tml-"):
            return False
        try:
            _sdk()
        except ImportError:
            return False
        return True

    def offers(self, student: str):
        from ..prices import tinker_models
        models = tinker_models()
        return (student in models) if models else None

    def estimate(self, job: Job) -> Estimate:
        from ..prices import tinker_models
        problems: List[str] = []
        key = os.environ.get("TINKER_API_KEY", "")
        if not key:
            problems.append("no TINKER_API_KEY (a key from https://tinker-console.thinkingmachines.ai)")
        elif not key.startswith("tml-"):
            problems.append("TINKER_API_KEY is not a Tinker key (they start with tml-)")
        try:
            _sdk()
        except ImportError:
            problems.append('the tinker SDK is not installed: pip install "functai[tinker]"')
        r = recipe(job)
        st = job.student
        models = tinker_models()
        info = models.get(st.name)
        if models and info is None:
            offered = sorted(k for k in models if ":" not in k)
            problems.append(f"Tinker does not train {st.name} (it trains {', '.join(offered[:8])}…)")
        if not r["lora"]:
            problems.append("Tinker trains LoRA adapters only (lora=False asks for all weights)")
        if info and info.get("context") and r["longest"] > info["context"]:
            problems.append(f"the longest example ({r['longest']:,} tokens) is longer than Tinker's context for "
                            f"{st.name} ({info['context']:,})")
        tokens = job.stats.get("tokens", 0) * r["epochs"]
        val_tokens = job.stats.get("val_tokens", 0) * max(1, round(1 / r["eval_every"])) * r["epochs"]
        dollars = None
        if info and info.get("train") is not None:
            dollars = (tokens + val_tokens) * info["train"] / 1e6
        settings = {**r, "precision": "service", "packing": False}
        price = f"≈ ${dollars:,.2f}" if dollars is not None else "price unknown"
        summary = f"{price} ({tokens / 1e6:,.1f}M training tokens at ${info['train']:.2f}/M)" \
            if dollars is not None else price
        notes = ["Tinker publishes no throughput: no time estimate"]
        return Estimate(not problems, problems, None, dollars, settings, notes, summary=summary)

    # ---- running: functai.bake.service (tokens from the student's published files, five methods)

    def supervise(self, run) -> None:
        raise BakeError("this run on Tinker was started by an older functai; bake again: training on Tinker now goes "
                        "through functai.bake.service (design 21)")

    def download(self, baked, path: Path):
        """The adapter from Tinker, converted and merged into a standard folder
        (needs ``tinker-cookbook``, which converts Tinker's adapter format)."""
        try:
            from tinker_cookbook import weights as tw
        except ImportError as err:
            raise ImportError("bringing Tinker weights here needs tinker-cookbook (its adapter converter) and "
                              "PyTorch: pip install tinker-cookbook") from err
        import shutil
        from ..baked import Baked, write_meta
        w = baked.meta["weights"]
        path.mkdir(parents=True, exist_ok=True)
        raw = path / "tinker-adapter"
        if raw.exists():
            shutil.rmtree(raw)
        tw.download(tinker_path=w["sampler_uri"], output_dir=str(raw))
        if (path / "model").exists():
            shutil.rmtree(path / "model")
        tw.build_hf_model(base_model=w["base"], adapter_path=str(raw), output_path=str(path / "model"))
        if (path / "adapter").exists():
            shutil.rmtree(path / "adapter")
        tw.build_lora_adapter(base_model=w["base"], adapter_path=str(raw), output_path=str(path / "adapter"))
        shutil.rmtree(raw)
        if path.resolve() != baked.path and (baked.path / "tokenizer").exists() and not (path / "tokenizer").exists():
            shutil.copytree(baked.path / "tokenizer", path / "tokenizer")
        meta = {k: v for k, v in baked.meta.items() if k not in ("hashes", "sizes", "functai_baked")}
        meta["weights"] = {"form": "merged", "base": w["base"], "path": "model", "adapter": "adapter",
                           "from": w["sampler_uri"]}
        write_meta(path, meta)
        return Baked(path)


__all__ = ["Tinker"]
