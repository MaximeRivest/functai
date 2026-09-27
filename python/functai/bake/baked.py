"""A baked model: weights trained for one AI function, the layout they read,
and the answers they give. It is also an lm15-style client, so functai calls
it like any provider:

    fast = classify.using(lm=baked)
    fast("my card never arrived")               # the Literal answer
    fast.predict("...").probabilities         # {"result": {"card_arrival": 0.93, ...}}

On disk it is a folder:

    baked.json          what it answers (fields and their keys), the input
                        layout (an lmcc artifact) and signature it was trained
                        for, the temperature per field, the training settings,
                        the data fingerprint, the report, file hashes
    model/              Hugging Face weights (safetensors)
    heads.safetensors   the answer layers, when the function has several outputs
    tokenizer/
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import lm15
import lmcc

from .examples import CAPABILITIES, BakeError, HeadField, input_plan, request_text

FORMAT = 1
PROVIDER = "functai-baked"          # a head model: answers judgments (like Jev)
PROVIDER_LM = "functai-baked-lm"    # a generative student: answers in its trained layout
METHOD = "provider_classification"  # lm15's word for a classifier's distribution over declared answers


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_folder(root: Path) -> Dict[str, str]:
    return {p.relative_to(root).as_posix(): _sha256_file(p)
            for p in sorted(root.rglob("*")) if p.is_file() and p.name != "baked.json"}


def default_home() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return Path(base) / "functai" / "baked"


class _Batcher:
    """Rows from concurrent calls, run together: a caller waits at most ``max_wait``
    seconds for others to join its batch."""

    def __init__(self, run, max_batch: int = 64, max_wait: float = 0.001):
        self.run, self.max_batch, self.max_wait = run, max_batch, max_wait
        self.cond = threading.Condition()
        self.pending: List[Dict[str, Any]] = []
        self.thread: Optional[threading.Thread] = None

    def submit(self, item: Any) -> Any:
        slot = {"item": item, "done": threading.Event(), "result": None, "error": None}
        with self.cond:
            self.pending.append(slot)
            if self.thread is None or not self.thread.is_alive():
                self.thread = threading.Thread(target=self._loop, name="functai-baked", daemon=True)
                self.thread.start()
            self.cond.notify()
        slot["done"].wait()
        if slot["error"] is not None:
            raise slot["error"]
        return slot["result"]

    def _loop(self) -> None:
        while True:
            with self.cond:
                while not self.pending:
                    if not self.cond.wait(timeout=30):
                        self.thread = None
                        return
                deadline = time.monotonic() + self.max_wait
                while len(self.pending) < self.max_batch:
                    left = deadline - time.monotonic()
                    if left <= 0 or not self.cond.wait(timeout=left):
                        break
                batch, self.pending = self.pending[:self.max_batch], self.pending[self.max_batch:]
            try:
                results = self.run([s["item"] for s in batch])
                for s, r in zip(batch, results):
                    s["result"] = r
            except BaseException as exc:  # noqa: BLE001 — every caller in the batch sees it
                for s in batch:
                    s["error"] = exc
            for s in batch:
                s["done"].set()


class Baked:
    """A model trained for one AI function (see the module docstring)."""

    __functai_baked__ = True

    def __init__(self, path: "str | os.PathLike[str]", *, device: Optional[str] = None, check: bool = True):
        self.path = Path(path).expanduser().resolve()
        try:
            self.meta = json.loads((self.path / "baked.json").read_text())
        except FileNotFoundError:
            raise BakeError(f"{self.path} is not a baked model (no baked.json)") from None
        if self.meta.get("functai_baked") != FORMAT:
            raise BakeError(f"{self.path}: unknown baked format {self.meta.get('functai_baked')!r}")
        if check:
            changed = [rel for rel, digest in self.meta.get("hashes", {}).items()
                       if not (self.path / rel).exists() or _sha256_file(self.path / rel) != digest]
            if changed:
                raise BakeError(f"{self.path}: files changed since baking: {changed[:5]}")
        self.kind: str = self.meta["kind"]
        self.fields = [HeadField.from_dict(d) for d in self.meta.get("fields", [])]
        self._device = device
        self._model = None
        self._tokenizer = None
        self._lock = threading.Lock()
        self._batcher = _Batcher(self._run_texts if self.kind == "head" else self._generate, max_batch=64)
        self._server = None
        self._report = None
        self.endpoint: Optional[str] = None       # generative students served elsewhere (serve())

    # ---- identity

    @property
    def name(self) -> str:
        return self.meta["name"]

    @property
    def provider(self) -> str:
        return PROVIDER if self.kind == "head" else PROVIDER_LM

    @property
    def model(self) -> str:
        return f"baked:{self.name}"

    @property
    def student(self) -> str:
        return self.meta["student"]

    @property
    def layout(self) -> Dict[str, Any]:
        """The lmcc adapter artifact the model reads its inputs through."""
        return self.meta["layout"]

    @property
    def signature(self) -> lmcc.SignatureCore:
        return lmcc.signature_from_dict(self.meta["signature"])

    @property
    def fingerprint(self) -> str:
        return self.meta["fingerprint"]

    @property
    def capabilities(self) -> Dict[str, bool]:
        return dict(CAPABILITIES) if self.kind == "head" else dict(self.meta.get("capabilities", {"instruct": True}))

    @property
    def report(self):
        from .report import BakeReport
        if self._report is None and self.meta.get("report"):
            self._report = BakeReport.from_dict(self.meta["report"])
        return self._report

    def __repr__(self) -> str:
        what = ", ".join(f"{f.name} ({len(f.keys)} answers)" for f in self.fields) or self.kind
        return f"<Baked {self.name}: {self.student} → {what} | {self.path}>"

    # ---- running

    @property
    def device(self) -> str:
        if self._device is None:
            from .heads import choose_device
            size = sum(p.stat().st_size for p in (self.path / "model").rglob("*.safetensors"))
            self._device = choose_device(None, need_gb=size / 2 ** 30 * 1.5 + 0.5)
        return self._device

    def _ensure(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            if self.kind == "head":
                from . import heads
                self._model, self._tokenizer = heads.load(
                    str(self.path), self.meta["architecture"], [len(f.keys) for f in self.fields],
                    decoder=self.meta.get("decoder", False), device=self.device)
            else:
                from . import sft
                self._model, self._tokenizer = sft.load(str(self.path), self.device)

    def _run_texts(self, texts: Sequence[str]) -> List[Any]:
        self._ensure()
        if self.kind != "head":
            from . import sft
            return sft.generate(self, texts)
        from . import heads
        from .metrics import softmax
        ids, _cut = heads.encode(self._tokenizer, texts, self.meta["max_length"])
        zs = heads.logits(self._model, self._tokenizer, ids, self.device)
        temps = self.meta["temperatures"]
        out = []
        for r in range(len(texts)):
            out.append({f.name: dict(zip(f.keys, softmax(zs[i][r], temps[i]))) for i, f in enumerate(self.fields)})
        return out

    def _generate(self, items: Sequence[Any]) -> List[str]:
        from . import sft
        return sft.generate_batch(self, items)

    def serve(self, **options) -> str:
        """Serve a generative student with vLLM and send calls there (see ``sft.serve``)."""
        if self.kind == "head":
            raise BakeError("serve() is for generative students; a head model runs in-process at full speed")
        from . import sft
        return sft.serve(self, **options)

    def stop(self) -> None:
        """Stop the vLLM server ``serve()`` started; calls run in-process again."""
        if self._server is not None:
            import signal
            try:
                os.killpg(self._server.pid, signal.SIGTERM)
                self._server.wait(timeout=60)
            except ProcessLookupError:
                pass
            except Exception:  # noqa: BLE001 — it did not stop in time
                try:
                    os.killpg(self._server.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        self._server, self.endpoint = None, None

    def probabilities(self, texts: Sequence[str]) -> List[Dict[str, Dict[str, float]]]:
        """Per text, the probability of every answer, per field (texts as the layout writes them)."""
        if self.kind != "head":
            raise BakeError("probabilities come from head models; a generative student writes text")
        return self._run_texts(list(texts))

    def texts(self, rows: Sequence[Dict[str, Any]]) -> List[str]:
        """The input text the model reads for each row of inputs (written by its layout)."""
        from .. import engine
        plan = input_plan(self.signature, lmcc.load(self.layout))
        out = []
        for row in rows:
            values = {}
            for f in self.signature.inputs:
                if f.name not in row:
                    raise BakeError(f"a row lacks the input {f.name!r}")
                v = row[f.name]
                if f.shape.get("type") == "string" and "enum" not in f.shape and not isinstance(v, str) and v is not None:
                    v = engine.to_text(v)
                values[f.name] = v
            msg = plan.render(plan.turn(values)).messages[-1]
            out.append("".join(p.get("text", "") for p in msg["parts"]))
        return out

    def predict(self, rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Answers for many rows at once (the fast path for big tables): per row,
        each field's answer, its probability, and the full distribution."""
        dists = self.probabilities(self.texts(rows))
        out = []
        for d in dists:
            row: Dict[str, Any] = {}
            for f in self.fields:
                key = max(d[f.name], key=d[f.name].get)
                row[f.name] = f.values[f.keys.index(key)]
                row[f"{f.name}__confidence"] = d[f.name][key]
                row[f"{f.name}__probs"] = d[f.name]
            out.append(row)
        return out

    # ---- the client face (what functai calls)

    def resolve(self, model: str):
        from ..models import _Route
        return _Route(self.provider, self.name)

    def complete(self, request: Any) -> Any:
        if self.kind != "head":
            from . import sft
            return sft.complete(self, request)
        from .examples import nest
        text = request_text(request)
        dist = self._batcher.submit(text)
        value = {}
        for f in self.fields:
            key = max(dist[f.name], key=dist[f.name].get)
            value[f.name] = f.values[f.keys.index(key)]
        part = lm15.DataPart(value=nest(value), probabilities=dist, method=METHOD)
        n = len(self._tokenizer(text)["input_ids"]) if self._tokenizer is not None else 0
        return lm15.Response(id=None, model=self.model, message=lm15.Message.assistant([part]), finish_reason="stop",
                             usage=lm15.Usage(input_tokens=n, output_tokens=0, total_tokens=n))

    # ---- files

    def save(self, path: "str | os.PathLike[str]", *, overwrite: bool = False) -> "Baked":
        """Copy the model to ``path``; returns it loaded from there."""
        target = Path(path).expanduser().resolve()
        if target.exists():
            if not overwrite:
                raise FileExistsError(f"{target} exists; save(..., overwrite=True) replaces it")
            shutil.rmtree(target)
        shutil.copytree(self.path, target)
        return Baked(target, device=self._device)

    def size(self) -> int:
        return sum(p.stat().st_size for p in self.path.rglob("*") if p.is_file())


def write_meta(path: Path, meta: Dict[str, Any]) -> None:
    meta = dict(meta)
    meta["functai_baked"] = FORMAT
    meta.setdefault("created", _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"))
    meta["hashes"] = hash_folder(path)
    (path / "baked.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False, default=str) + "\n")


def is_baked(obj: Any) -> bool:
    return getattr(type(obj), "__functai_baked__", False) is True


def load(path: "str | os.PathLike[str]", *, device: Optional[str] = None) -> Baked:
    """A baked model from its folder (its file hashes are checked)."""
    return Baked(path, device=device)


__all__ = ["Baked", "load", "is_baked", "PROVIDER", "PROVIDER_LM"]
