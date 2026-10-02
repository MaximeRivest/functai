"""A training run: a folder, a process, and a model at the end.

    run = summarize.bake(rows, wait=False)   # returns at once
    run.metrics()                            # the loss curve so far (a dpyr table, live)
    run.wait()                               # → the Baked model
    run.stop(); run.resume()
    functai.bake.runs()                      # every run on this machine
    functai.bake.run(folder)                 # reattach after a restart

The folder:

    plan.json         every resolved setting (what makes resuming exact)
    examples.parquet  the training conversations
    run.json          state (running, stopped, done, failed), where, process, attempts
    metrics.jsonl     one line per logged step: step, tokens, loss, lr, eval_loss, seconds
    log.txt           the trainer's own output
    checkpoints/      adapter and optimizer state (here), or the service's checkpoint ids
    baked/            the model, when done (or the ``path=`` given to bake)

A run is its own process (``python -m functai.bake supervise <folder>``), so
it survives the notebook or terminal that started it. Running the same bake
again with the same plan finds the folder and resumes.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .examples import BakeError

STATES = ("planned", "starting", "running", "stopping", "stopped", "done", "failed", "exported")


def runs_home() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return Path(base) / "functai" / "bakes"


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False, default=str) + "\n")
    tmp.replace(path)


def _alive(pid: Optional[int]) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:       # a zombie child of this process is not running
        done, _status = os.waitpid(int(pid), os.WNOHANG)
        return done == 0
    except ChildProcessError:
        return True


def _duration(seconds: Optional[float]) -> str:
    if seconds is None or seconds != seconds:
        return "?"
    seconds = int(seconds)
    if seconds < 90:
        return f"{seconds} s"
    if seconds < 5400:
        return f"{round(seconds / 60)} min"
    h, m = divmod(round(seconds / 60), 60)
    return f"{h} h {m:02d} min" if h < 48 else f"{h / 24:.1f} days"


class Run:
    """A training run in its folder (see the module docstring)."""

    def __init__(self, folder: "str | os.PathLike[str]"):
        self.folder = Path(folder).expanduser().resolve()
        if not (self.folder / "plan.json").exists():
            raise BakeError(f"{self.folder} is not a bake run (no plan.json)")
        self._thread: Optional[threading.Thread] = None

    # ---- files

    @property
    def plan(self) -> Dict[str, Any]:
        return json.loads((self.folder / "plan.json").read_text())

    def _state_file(self) -> Dict[str, Any]:
        p = self.folder / "run.json"
        return json.loads(p.read_text()) if p.exists() else {"state": "planned"}

    def update(self, **fields: Any) -> Dict[str, Any]:
        """Change run.json (the supervisor's; also ``stop``)."""
        data = self._state_file()
        data.update(fields)
        data["updated"] = _now()
        _write_json(self.folder / "run.json", data)
        return data

    @property
    def status(self) -> Dict[str, Any]:
        """run.json, with a process that ended without saying so shown as such."""
        data = self._state_file()
        if data.get("state") in ("starting", "running", "stopping") and not _alive(data.get("pid")) and \
                not (self._thread is not None and self._thread.is_alive()):
            data["state"] = "stopped"
            data.setdefault("error", "the run's process ended without finishing (see log.txt); run.resume() "
                                     "continues from the last checkpoint")
        return data

    @property
    def state(self) -> str:
        return self.status.get("state", "planned")

    @property
    def where(self) -> str:
        return self.plan["where"]

    def stop_requested(self) -> bool:
        return (self.folder / "STOP").exists()

    # ---- metrics

    def records(self) -> List[Dict[str, Any]]:
        p = self.folder / "metrics.jsonl"
        if not p.exists():
            return []
        out = []
        for line in p.read_text().splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                continue          # a line being written
        return out

    def metrics(self):
        """The logged steps as a dpyr table (a list of dicts without dpyr)."""
        recs = self.records()
        try:
            import dpyr
        except ImportError:
            return recs
        if not recs:
            return dpyr.from_dict({})
        cols = sorted({k for r in recs for k in r}, key=lambda k: (k not in ("step", "epoch"), k))
        return dpyr.from_dict({c: [r.get(c) for r in recs] for c in cols})

    def log_metric(self, **fields: Any) -> None:
        """Append one line to metrics.jsonl (what trainers call)."""
        fields.setdefault("time", _now())
        with open(self.folder / "metrics.jsonl", "a") as f:
            f.write(json.dumps(fields, default=float) + "\n")

    def progress(self) -> Dict[str, Any]:
        recs = self.records()
        st = self.status
        out: Dict[str, Any] = {"state": st.get("state"), "steps": st.get("steps")}
        train = [r for r in recs if r.get("loss") is not None]
        evals = [r for r in recs if r.get("eval_loss") is not None]
        if train:
            out.update(step=train[-1].get("step"), loss=train[-1]["loss"], seconds=train[-1].get("seconds"))
        if evals:
            out["eval_loss"] = evals[-1]["eval_loss"]
        if out.get("step") and out.get("steps") and out.get("seconds"):
            out["eta"] = out["seconds"] / out["step"] * (out["steps"] - out["step"])
        return out

    def log(self, lines: int = 40) -> str:
        p = self.folder / "log.txt"
        return "\n".join(p.read_text(errors="replace").splitlines()[-lines:]) if p.exists() else ""

    # ---- checkpoints and the model

    def checkpoints(self) -> List[int]:
        """Steps with a checkpoint, oldest first."""
        out = []
        for p in (self.folder / "checkpoints").glob("checkpoint-*"):
            try:
                out.append(int(p.name.split("-")[1]))
            except (IndexError, ValueError):
                continue
        for c in self._state_file().get("remote_checkpoints", []):
            out.append(int(c["step"]))
        return sorted(set(out))

    def checkpoint(self, step: Optional[int] = None):
        """A checkpoint as a usable model (the last one by default). Models from
        the constant phase of the schedule are not decayed: compare them with
        each other, not with the final model."""
        from .trainers import trainer
        steps = self.checkpoints()
        if not steps:
            raise BakeError("this run has no checkpoint yet")
        step = steps[-1] if step is None else step
        if step not in steps:
            raise BakeError(f"no checkpoint at step {step}; there are {steps}")
        return trainer(self.where).checkpoint_model(self, step)

    def baked(self):
        """The model this run made (when done)."""
        from .baked import Baked
        out = Path(self.plan["output"])
        if not (out / "baked.json").exists():
            raise BakeError(f"the run is {self.state}; its model is written when it is done")
        return Baked(out)

    # ---- the process

    @classmethod
    def create(cls, folder: Path, plan: Dict[str, Any]) -> "Run":
        folder.mkdir(parents=True, exist_ok=True)
        _write_json(folder / "plan.json", plan)
        run = cls(folder)
        if not (folder / "run.json").exists():
            run.update(state="planned", created=_now())
        return run

    def start(self, *, process: bool = True) -> "Run":
        """Start (or resume) the run in its own process; ``process=False`` runs it
        in a thread of this one (ends with it)."""
        st = self.status
        if st.get("state") in ("starting", "running") and (_alive(st.get("pid")) or
                                                          (self._thread is not None and self._thread.is_alive())):
            return self
        if st.get("state") == "done":
            return self
        (self.folder / "STOP").unlink(missing_ok=True)
        attempts = st.get("attempts", 0) + 1
        self.update(state="starting", attempts=attempts, error=None)
        if not process:
            self.update(pid=os.getpid())
            self._thread = threading.Thread(target=supervise, args=(str(self.folder),), daemon=True,
                                            name="functai-bake-run")
            self._thread.start()
            return self
        log = open(self.folder / "log.txt", "a")
        log.write(f"\n==== {_now()} attempt {attempts}\n")
        log.flush()
        proc = subprocess.Popen([sys.executable, "-m", "functai.bake", "supervise", str(self.folder)],
                                stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                start_new_session=True, cwd=str(self.folder), env=_child_env())
        self.update(pid=proc.pid)
        return self

    def resume(self) -> "Run":
        """Continue a stopped or crashed run from its last checkpoint."""
        if self.state == "done":
            return self
        return self.start(process=self._thread is None)

    def stop(self, *, wait: bool = True, timeout: float = 600) -> "Run":
        """Ask the run to stop at its next step (it saves a checkpoint first)."""
        if self.state not in ("starting", "running"):
            return self
        (self.folder / "STOP").write_text(_now())
        self.update(state="stopping")
        if wait:
            t0 = time.time()
            while self.state == "stopping" and time.time() - t0 < timeout:
                time.sleep(1)
            if self.state == "stopping":       # it did not answer: end the process group
                pid = self.status.get("pid")
                if pid and pid != os.getpid():
                    try:
                        os.killpg(int(pid), signal.SIGTERM)
                    except (ProcessLookupError, PermissionError):
                        pass
                self.update(state="stopped")
        return self

    def wait(self, *, poll: float = 5.0, show: Any = True):
        """Wait for the end; show progress; return the model."""
        say = _printer(show)
        last = None
        while True:
            st = self.status
            state = st.get("state")
            p = self.progress()
            line = self._line(p)
            if line != last:
                say(line)
                last = line
            if state == "done":
                return self.baked()
            if state == "exported":
                return None
            if state in ("failed", "stopped"):
                raise BakeError(f"the run {state}: {st.get('error') or 'see its log'}\n{self.log(15)}")
            time.sleep(poll)

    def _line(self, p: Dict[str, Any]) -> str:
        if p.get("step") is None:
            return f"{p.get('state')}: {self.status.get('phase') or 'preparing'}"
        parts = [f"step {p['step']:,}" + (f"/{p['steps']:,}" if p.get("steps") else ""), f"loss {p['loss']:.4f}"]
        if p.get("eval_loss") is not None:
            parts.append(f"validation {p['eval_loss']:.4f}")
        if p.get("eta") is not None:
            parts.append(f"about {_duration(p['eta'])} left")
        return " · ".join(parts)

    def __repr__(self) -> str:
        st = self.status
        p = self.progress()
        plan = self.plan
        line = f"<Run {plan.get('name')} → {plan.get('student')} on {plan.get('where')}: {st.get('state')}"
        if p.get("step") is not None:
            line += f", {self._line(p)}"
        return line + f" | {self.folder}>"


def _child_env() -> Dict[str, str]:
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    return env


def _printer(show: Any) -> Callable[[str], None]:
    if show is False or show is None:
        return lambda s: None
    if callable(show):
        return show
    return lambda s: print(f"[bake] {s}", file=sys.stderr, flush=True)


def supervise(folder: str) -> None:
    """The run's own process: train to the end with the plan's trainer, and keep
    run.json true whatever happens."""
    from .trainers import Stopped, trainer
    run = Run(folder)
    plan = run.plan
    run.update(state="running", pid=os.getpid(), started=_now(), where=plan["where"])
    try:
        trainer(plan["where"]).supervise(run)
    except Stopped:
        run.update(state="stopped", error=None)
        return
    except BaseException as exc:  # noqa: BLE001 — written down for whoever waits
        import traceback
        traceback.print_exc()
        run.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        if not isinstance(exc, Exception):
            raise
        return
    if run.status.get("state") not in ("done", "exported"):
        run.update(state="done", finished=_now())


def runs(home: "str | os.PathLike[str] | None" = None) -> List[Run]:
    """Every run under ``home`` (default ``~/.cache/functai/bakes``), newest first."""
    root = Path(home).expanduser() if home else runs_home()
    out = [Run(p.parent) for p in root.glob("*/plan.json")]
    return sorted(out, key=lambda r: (r.folder / "plan.json").stat().st_mtime, reverse=True)


def run(folder: "str | os.PathLike[str]") -> Run:
    """The run in ``folder``."""
    return Run(folder)


__all__ = ["Run", "runs", "run", "runs_home", "supervise"]
