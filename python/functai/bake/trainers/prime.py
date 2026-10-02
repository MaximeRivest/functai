"""Training on Prime Intellect's hosted SFT (``prime train``, prime CLI 0.9+).

Prime's hosted SFT (closed beta as of 2026-10-01, enabled per account) trains
all weights of a model from Prime's cluster cache, reading a dataset from a
Prime volume. Its trainer tokenizes the ``messages`` column with its own
renderer, so it cannot take bake's tokens: the run writes the messages and
asks for the renderer settings bake's tokens were made with (the plan says
so). The weights stay on the volume; ``baked.download()`` says how to bring
them here (Prime's CLI has no download command yet).

Set up: the prime CLI, logged in (``prime login``), with hosted SFT enabled.
Reinforcement learning on Prime is ``functai.bake.prime`` (environments).
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import List, Optional

from ..examples import BakeError
from ..recipe import recipe
from ..students import learning_rate
from . import Estimate, Job, Stopped, Trainer


def _cli(*args: str, timeout: float = 120) -> subprocess.CompletedProcess:
    exe = shutil.which("prime")
    if exe is None:
        raise BakeError("the prime CLI is not installed: uv tool install prime")
    return subprocess.run([exe, *args], capture_output=True, text=True, timeout=timeout,
                          env={**__import__("os").environ, "PRIME_DISABLE_VERSION_CHECK": "1"})


def _version() -> Optional[tuple]:
    try:
        out = _cli("--version", timeout=20).stdout
    except Exception:  # noqa: BLE001
        return None
    m = re.search(r"(\d+)\.(\d+)\.(\d+)", out)
    return tuple(int(x) for x in m.groups()) if m else None


def _logged_in() -> bool:
    cfg = Path.home() / ".prime" / "config.json"
    if not cfg.exists():
        return False
    try:
        return bool(json.loads(cfg.read_text()).get("api_key"))
    except ValueError:
        return False


class Prime(Trainer):
    name = "prime"
    paid = True

    def set_up(self) -> bool:
        return shutil.which("prime") is not None and _logged_in() and (_version() or (0,)) >= (0, 9, 0)

    def _models(self) -> Optional[List[str]]:
        try:
            r = _cli("train", "models", "--fft-only", "-o", "json", timeout=60)
            data = json.loads(r.stdout)
        except Exception:  # noqa: BLE001 — no access, an old CLI, offline
            return None
        items = data if isinstance(data, list) else data.get("models", [])
        return [m.get("name") or m.get("id") for m in items if isinstance(m, dict)]

    def offers(self, student: str):
        models = self._models()
        return None if models is None else student in models

    def estimate(self, job: Job) -> Estimate:
        problems: List[str] = []
        if shutil.which("prime") is None:
            problems.append("the prime CLI is not installed (uv tool install prime)")
        else:
            v = _version()
            if v is None or v < (0, 9, 0):
                problems.append(f"prime CLI {'.'.join(map(str, v)) if v else '?'} is too old for hosted SFT "
                                f"(0.9 or later: prime upgrade)")
            if not _logged_in():
                problems.append("not logged in to Prime (prime login)")
        r = recipe(job)
        st = job.student
        if not problems:
            models = self._models()
            if models is None:
                problems.append("hosted SFT is not available to this account (closed beta: Prime enables it per "
                                "account)")
            elif st.name not in models:
                problems.append(f"Prime's cluster cache has no {st.name} for hosted SFT")
        settings = {**r, "lora": False, "lora_rank": None, "learning_rate": float(job.options.get("lr") or
                                                                                  learning_rate(st, lora=False)),
                    "precision": "service", "packing": False}
        notes = ["Prime trains all weights (hosted SFT has no LoRA) and tokenizes with its own renderer, so the "
                 "tokens are Prime's, checked only by the renderer settings",
                 "Prime publishes no price or throughput for hosted SFT yet"]
        return Estimate(not problems, problems, None, None, settings, notes,
                        summary="full fine-tuning on a dedicated cluster · price not published")

    # ---- running

    def config(self, run) -> str:
        plan = run.plan
        s = plan["settings"]
        kwargs = (plan.get("template") or {}).get("kwargs") or {}
        lines = [f"# {plan['name']}: functai bake, hosted SFT", f"max_steps = {int(s['steps'])}", "", "[ckpt]", "",
                 "[model]", f"name = {json.dumps(plan['student'])}", "", "[data]",
                 f"name = {json.dumps('datasets/' + run.folder.name)}", 'type = "sft"',
                 f"seq_len = {int(s['max_model_len'])}", f"batch_size = {int(s['examples_per_step'])}", "",
                 "[optim]", f"lr = {float(s['learning_rate'])!r}", "", "[deployment]", 'type = "single_node"',
                 "num_train_gpus = 1", "gpus_per_node = 1", "", "[renderer]", 'name = "auto"']
        for k, v in kwargs.items():
            lines.append(f"{k} = {json.dumps(v)}")
        return "\n".join(lines) + "\n"

    def supervise(self, run) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq
        plan = run.plan
        volume = plan["settings"].get("volume") or "functai-bake"
        st = run.status
        prime_id = st.get("prime_run")
        if not prime_id:
            data = run.folder / "prime" / "dataset"
            data.mkdir(parents=True, exist_ok=True)
            table = pq.read_table(run.folder / "examples.parquet", columns=["messages", "split", "weight"])
            rows = [r for r in table.to_pylist() if r["split"] != "validation"]
            from .here_worker import repeat_by_weight
            keep = repeat_by_weight(range(len(rows)), [float(r["weight"] or 1.0) for r in rows],
                                    int(plan["settings"].get("seed") or 0))
            pq.write_table(pa.Table.from_pylist([{"messages": rows[i]["messages"]} for i in keep]),
                           data / "train.parquet")
            (run.folder / "prime" / "sft.toml").write_text(self.config(run))
            run.update(phase=f"uploading the examples to the Prime volume {volume}")
            if _cli("volumes", "list", "-o", "json").stdout.find(f'"{volume}"') < 0:
                r = _cli("volumes", "create", volume, "--size", "100Gi")
                if r.returncode:
                    raise BakeError(f"prime volumes create failed: {r.stderr or r.stdout}")
                time.sleep(15)
            r = _cli("volumes", "put", volume, str(data), f"datasets/{run.folder.name}", timeout=3600)
            if r.returncode:
                raise BakeError(f"prime volumes put failed: {r.stderr or r.stdout}")
            r = _cli("train", str(run.folder / "prime" / "sft.toml"), "--volume", volume, "--yes", "-o", "json",
                     timeout=600)
            m = re.search(r"Dispatched hosted run (\w+)", r.stdout) or re.search(r'"id"\s*:\s*"(\w+)"', r.stdout)
            if r.returncode or not m:
                raise BakeError(f"prime train failed: {(r.stderr or r.stdout)[-1500:]}")
            prime_id = m.group(1)
            run.update(prime_run=prime_id, volume=volume,
                       dashboard=f"https://app.primeintellect.ai/dashboard/training/{prime_id}")
        run.update(phase=f"training on Prime (run {prime_id})")
        seen = 0
        while True:
            if run.stop_requested():
                _cli("train", "stop", prime_id, "--yes")
                raise Stopped()
            info = _cli("train", "get", prime_id, "-o", "json").stdout
            state = (re.search(r'"status"\s*:\s*"(\w+)"', info) or re.search(r"Status:\s*(\w+)", info))
            state = state.group(1).upper() if state else "UNKNOWN"
            try:
                metrics = json.loads(_cli("train", "metrics", prime_id, "-o", "json").stdout)
                steps = metrics.get("step") or []
                losses = metrics.get("loss/mean") or []
                for step, loss in list(zip(steps, losses))[seen:]:
                    run.log_metric(step=int(step), loss=float(loss))
                seen = max(seen, min(len(steps), len(losses)))
            except Exception:  # noqa: BLE001 — metrics come and go during start-up and teardown
                pass
            if state == "COMPLETED":
                break
            if state in ("FAILED", "ERROR", "CANCELLED", "STOPPED"):
                raise BakeError(f"the Prime run {prime_id} ended {state}: prime train logs {prime_id}")
            time.sleep(30)
        from ..finish import training_summary, write_remote
        weights = {"form": "remote", "service": "prime", "uri": f"prime://{volume}/{prime_id}", "run": prime_id,
                   "volume": volume, "base": plan["student"]}
        write_remote(plan, Path(plan["output"]), weights, training={**training_summary(run), "where": "prime"},
                     run_folder=str(run.folder))
        run.update(state="done", phase=None)

    def download(self, baked, path):
        w = baked.meta["weights"]
        raise BakeError(f"Prime keeps the weights on the volume {w['volume']} (run {w['run']}); its CLI has no "
                        f"download yet. Copy the final checkpoint out (prime volumes ssh {w['volume']}), then "
                        f"functai.bake.adopt(<that folder>, <your function>, examples=...) checks and uses it")


__all__ = ["Prime"]
