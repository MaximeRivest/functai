"""Training on Tinker (Thinking Machines): bake's own tokens, sent as they
are, with the answer-only weights (and each example's weight, exactly);
Tinker runs the forward and backward passes on its GPUs.

No PyTorch here: the run's process only sends batches and writes metrics.
Checkpoints (weights and optimizer state) are Tinker's; their addresses are
kept in run.json, and a stopped run resumes from the last one. The trained
model is a ``tinker://`` address: it runs on Tinker's sampler
(``baked.on("tinker")``), or comes here with ``baked.download()``.

Set up: ``TINKER_API_KEY`` in the environment (a key from the Tinker console).
"""

from __future__ import annotations

import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List

from ..recipe import recipe
from . import Estimate, Job, Stopped, Trainer


def _sdk():
    try:
        import tinker
    except ImportError as err:
        raise ImportError('training on Tinker needs its SDK: pip install "functai[tinker]"') from err
    return tinker


def wsd(step: int, total: int, warmup: float, decay: float) -> float:
    """The learning-rate multiplier at ``step``: linear warmup, constant, then
    a "1-sqrt" decay to zero over the last ``decay`` of the steps."""
    w = max(1, round(total * warmup))
    d = max(1, round(total * decay))
    if step < w:
        return (step + 1) / w
    if step < total - d:
        return 1.0
    return max(0.0, 1.0 - math.sqrt((step - (total - d) + 1) / d))


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

    # ---- running

    def supervise(self, run) -> None:
        tinker = _sdk()
        from .here_worker import Tokens
        plan = run.plan
        s = plan["settings"]
        seed = int(s.get("seed") or 0)
        tokens = Tokens(run.folder / "examples.parquet")
        n = len(tokens.splits)
        train = [i for i in range(n) if tokens.splits[i] != "validation"]
        val = [i for i in range(n) if tokens.splits[i] == "validation"]
        per_step = int(s["examples_per_step"])
        epochs = int(s["epochs"])
        steps_per_epoch = max(1, len(train) // per_step)     # an uneven last batch would change the step's scale
        total = steps_per_epoch * epochs
        eval_every = max(1, round(total * float(s.get("eval_every") or 0.05)))
        save_every = max(1, round(total * float(s.get("save_every") or 0.1)))
        service = tinker.ServiceClient()
        st = run.status
        ckpts = list(st.get("remote_checkpoints") or [])
        if ckpts:
            last = ckpts[-1]
            client = service.create_training_client_from_state_with_optimizer(last["state"])
            start = int(last["step"])
            run.update(phase=f"resuming from step {start}")
        else:
            client = service.create_lora_training_client(base_model=plan["student"], rank=int(s["lora_rank"]),
                                                         seed=seed)
            start = 0
        run.update(steps=total, phase="training", tinker_run=getattr(client, "model_id", None))

        def datum(i: int, scale: float = 1.0):
            ids = tokens.ids(i).tolist()
            start_at = int(tokens.starts[i])
            weights = [0.0] * (start_at - 1) + [scale] * (len(ids) - start_at)
            return tinker.types.Datum(model_input=tinker.types.ModelInput.from_ints(ids[:-1]),
                                      loss_fn_inputs={"target_tokens": ids[1:], "weights": weights})

        def mean_nll(out, data) -> float:
            num = den = 0.0
            for o, d in zip(out.loss_fn_outputs, data):
                lp = o["logprobs"].tolist() if hasattr(o["logprobs"], "tolist") else list(o["logprobs"].data)
                w = d.loss_fn_inputs["weights"].tolist() if hasattr(d.loss_fn_inputs["weights"], "tolist") else \
                    list(d.loss_fn_inputs["weights"].data)
                num -= sum(a * b for a, b in zip(lp, w))
                den += sum(w)
            return num / den if den else float("nan")

        val_data = [datum(i) for i in val]
        t0 = time.time()
        seen = 0
        orders: Dict[int, List[int]] = {}
        for step in range(start, total):
            epoch, k = divmod(step, steps_per_epoch)
            if epoch not in orders:          # weights go into the loss (exactly), not into repetitions
                orders.clear()
                orders[epoch] = list(train)
                random.Random(seed + epoch).shuffle(orders[epoch])
            order = orders[epoch]
            batch_idx = order[k * per_step:(k + 1) * per_step]
            batch = [datum(i, tokens.weights[i]) for i in batch_idx]
            lr = float(s["learning_rate"]) * wsd(step, total, s["schedule"]["warmup"], s["schedule"]["decay"])
            fb = client.forward_backward(batch, "cross_entropy")
            opt = client.optim_step(tinker.types.AdamParams(learning_rate=lr, beta1=0.9, beta2=0.95, eps=1e-8))
            out = fb.result()
            opt.result()
            seen += sum(tokens.length(i) for i in batch_idx)
            rec: Dict[str, Any] = {"step": step + 1, "epoch": round((step + 1) / steps_per_epoch, 4),
                                   "loss": mean_nll(out, batch), "learning_rate": lr, "tokens": seen,
                                   "seconds": round(time.time() - t0, 1)}
            if val_data and ((step + 1) % eval_every == 0 or step + 1 == total):
                losses = []
                for b in range(0, len(val_data), 64):
                    chunk = val_data[b:b + 64]
                    losses.append((mean_nll(client.forward(chunk, "cross_entropy").result(), chunk), len(chunk)))
                rec["eval_loss"] = sum(a * m for a, m in losses) / sum(m for _a, m in losses)
            run.log_metric(**rec)
            stop = run.stop_requested()
            if (step + 1) % save_every == 0 or stop:
                saved = client.save_state(f"step-{step + 1:06d}").result()
                ckpts.append({"step": step + 1, "state": saved.path})
                run.update(remote_checkpoints=ckpts)
            if stop:
                raise Stopped()
        run.update(phase="saving the weights for sampling")
        state = client.save_state("final").result().path
        sampler = client.save_weights_for_sampler("final").result().path
        from ..finish import training_summary, write_remote
        weights = {"form": "remote", "service": "tinker", "uri": sampler, "sampler_uri": sampler, "state_uri": state,
                   "base": plan["student"], "lora_rank": s["lora_rank"]}
        write_remote(plan, Path(plan["output"]), weights,
                     training={**training_summary(run), "where": "tinker"}, run_folder=str(run.folder))
        run.update(state="done", phase=None)

    def checkpoint_model(self, run, step: int):
        tinker = _sdk()
        from ..baked import Baked
        from ..finish import write_remote
        ck = next(c for c in run.status.get("remote_checkpoints", []) if int(c["step"]) == step)
        out = run.folder / "checkpoint-models" / f"step-{step}"
        if not (out / "baked.json").exists():
            client = tinker.ServiceClient().create_training_client_from_state(ck["state"])
            sampler = client.save_weights_for_sampler(f"step-{step:06d}-sampler").result().path
            plan = {**run.plan, "name": f"{run.plan['name']}-step{step}"}
            write_remote(plan, out, {"form": "remote", "service": "tinker", "uri": sampler, "sampler_uri": sampler,
                                     "state_uri": ck["state"], "base": plan["student"],
                                     "lora_rank": plan["settings"]["lora_rank"]}, run_folder=str(run.folder))
        return Baked(out)

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


__all__ = ["Tinker", "wsd"]
