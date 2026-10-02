"""A baked model: weights trained to answer one or more AI functions, the
layout each function's calls are written in, and where the weights run.
It is also an lm15-style client, so functai calls it like any provider:

    fast = summarize.using(lm=baked)
    fast("...")                          # the function, on the weights
    baked.on("vllm")                     # serve them for throughput

On disk it is a folder:

    baked.json          what it answers: per function, its signature, layout,
                        and the inputs left out (fixed and derived); the
                        weights' form and base; the chat template; the run
                        that made it; the report; file hashes
    model/              Hugging Face weights (safetensors), merged when trained
                        with LoRA: a standard folder for vLLM, TGI, transformers
    adapter/            the LoRA adapter alone (when trained with LoRA)
    tokenizer/
    heads.safetensors   a head model's answer layers, when it has several outputs

Weights trained on a service and not yet brought here are a ``tinker://``
address in ``baked.json`` (``baked.download()`` brings them here).
"""

from __future__ import annotations

import dataclasses
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
from .functions import Entry

FORMAT = 2
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


QUICK = 64 << 20


def _changed(root: Path, meta: Dict[str, Any], *, full: bool) -> List[str]:
    """Files that differ from their recorded hashes. Quick (the default when
    loading): every small file hashed, large ones (weights) checked by size;
    ``full``: every byte hashed (``functai.verify``, and saved programs)."""
    sizes = meta.get("sizes", {})
    out = []
    for rel, digest in meta.get("hashes", {}).items():
        p = root / rel
        if not p.exists():
            out.append(rel)
            continue
        size = p.stat().st_size
        if not full and size > QUICK and rel in sizes:
            if size != sizes[rel]:
                out.append(rel)
        elif _sha256_file(p) != digest:
            out.append(rel)
    return out


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
    """A model trained for one or more AI functions (see the module docstring)."""

    __functai_baked__ = True

    def __init__(self, path: "str | os.PathLike[str]", *, device: Optional[str] = None, check: Any = True):
        self.path = Path(path).expanduser().resolve()
        try:
            self.meta = json.loads((self.path / "baked.json").read_text())
        except FileNotFoundError:
            raise BakeError(f"{self.path} is not a baked model (no baked.json)") from None
        fmt = self.meta.get("functai_baked")
        if fmt != FORMAT:
            raise BakeError(f"{self.path}: baked format {fmt!r}; this functai reads format {FORMAT} "
                            f"(bake it again)" if fmt == 1 else f"{self.path}: unknown baked format {fmt!r}")
        if check:
            changed = _changed(self.path, self.meta, full=check == "full")
            if changed:
                raise BakeError(f"{self.path}: files changed since baking: {changed[:5]}")
        self.kind: str = self.meta["kind"]
        self.entries: Dict[str, Entry] = {d["name"]: Entry.from_meta(d) for d in self.meta["functions"]}
        head = self.meta.get("head") or {}
        self.fields = [HeadField.from_dict(d) for d in head.get("fields", [])]
        self._device = device
        self._model = None
        self._tokenizer = None
        self._lock = threading.Lock()
        self._batcher = _Batcher(self._run_texts, max_batch=64) if self.kind == "head" else None
        self._runner = None
        self._report = None

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
    def functions(self) -> List[str]:
        return list(self.entries)

    def _only(self) -> Entry:
        if len(self.entries) != 1:
            raise BakeError(f"{self.name} answers several functions ({self.functions}); ask for one: "
                            f"baked.entries[name]")
        return next(iter(self.entries.values()))

    @property
    def layout(self) -> Dict[str, Any]:
        """The lmcc adapter artifact its function's calls are written in (one function)."""
        return self._only().layout

    @property
    def signature(self) -> lmcc.SignatureCore:
        """What the student reads (one function)."""
        return self._only().signature

    @property
    def fingerprint(self) -> str:
        return self._only().fingerprint

    @property
    def capabilities(self) -> Dict[str, bool]:
        if self.kind == "head":
            return dict(CAPABILITIES)
        return dict(next(iter(self.entries.values())).capabilities)

    @property
    def template(self):
        from .template import Template
        return Template.from_dict(self.meta["template"])

    @property
    def report(self):
        if self._report is None and self.meta.get("report"):
            from .report import load_report
            self._report = load_report(self.meta["report"])
        return self._report

    def __repr__(self) -> str:
        if self.kind == "head":
            what = ", ".join(f"{f.name} ({len(f.keys)} answers)" for f in self.fields)
        else:
            what = ", ".join(self.functions)
        w = self.meta.get("weights", {})
        where = w.get("uri") if w.get("form") == "remote" else str(self.path)
        return f"<Baked {self.name}: {self.student} → {what} | {where}>"

    # ---- which function, and how its calls are laid out

    def entry_for(self, fn, spec) -> Entry:
        """The entry for ``fn`` (refused when the model was not trained for it,
        or when it changed since)."""
        entry = self.entries.get(fn.__name__)
        if entry is None and len(self.entries) == 1:
            entry = next(iter(self.entries.values()))       # one function: its name may differ; its signature may not
        if entry is None:
            import lmcc as _lmcc
            fp = _lmcc.signature_fingerprint(spec.signature)
            entry = next((e for e in self.entries.values() if e.fingerprint == fp), None)
        if entry is None:
            raise BakeError(f"the baked model {self.name!r} was not trained for {fn.__name__} "
                            f"(it answers {self.functions})")
        if self.kind == "head":
            from .examples import head_fields, head_signature
            try:
                signature = head_signature(spec, head_fields(spec))
            except BakeError as exc:
                raise BakeError(f"{fn.__name__} cannot run on the baked model {self.name!r}: {exc}") from None
            entry.check(dataclasses.replace(spec, signature=signature))
        else:
            entry.check(spec)
        return entry

    def call_signature(self, fn, spec) -> lmcc.SignatureCore:
        """The signature ``fn``'s calls bind on this model: what the student reads."""
        entry = self.entry_for(fn, spec)
        if self.kind == "head":
            return spec.signature          # the judgment layout reads the function's own outputs
        return entry.student_spec(spec).signature

    def reduce(self, fn, spec, inputs, *, check: bool = True):
        """(spec, inputs) as the student reads them (fixed and derived inputs left
        out, after checking their values)."""
        if self.kind == "head":
            return spec, dict(inputs)
        return self.entry_for(fn, spec).reduce(spec, inputs, check=check)

    # ---- running

    @property
    def device(self) -> str:
        if self._device is None:
            from .heads import choose_device
            size = sum(p.stat().st_size for p in (self.path / "model").rglob("*.safetensors")) \
                if (self.path / "model").exists() else 0
            self._device = choose_device(None, need_gb=size / 2 ** 30 * 1.3 + 0.5)
        return self._device

    def tokenizer(self):
        if self._tokenizer is None:
            with self._lock:
                if self._tokenizer is None:
                    from .template import load_tokenizer
                    local = self.path / "tokenizer"
                    tok = load_tokenizer(str(local) if local.exists() else self.student)
                    if self.kind != "head":
                        import hashlib as _h
                        got = "sha256:" + _h.sha256((tok.chat_template or "").encode()).hexdigest()
                        if got != self.template.sha256:
                            raise BakeError(f"{self.name}: the tokenizer's chat template is not the one it was "
                                            f"trained with")
                    self._tokenizer = tok
        return self._tokenizer

    def on(self, where: Any = None, **options) -> "Baked":
        """Run on ``where``: ``"transformers"`` (in this process), ``"vllm"`` (a
        server started here), ``"tinker"``, or an OpenAI-compatible URL. Returns
        the model."""
        if self.kind == "head":
            raise BakeError("a head model runs in-process at full speed; on() is for generative students")
        from . import runners
        old, self._runner = self._runner, runners.make(self, where, **options)
        if old is not None and old is not self._runner:
            old.close()
        return self

    @property
    def runner(self):
        if self._runner is None:
            with self._lock:
                if self._runner is None:
                    from . import runners
                    self._runner = runners.make(self)
        return self._runner

    def serve(self, **options) -> str:
        """Serve with vLLM and send calls there; returns the endpoint."""
        self.on("vllm", **options)
        return self._runner.url

    @property
    def endpoint(self) -> Optional[str]:
        return getattr(self._runner, "url", None)

    def stop(self) -> None:
        """Stop a server this model started (``serve()``/``on("vllm")``); calls run in-process again."""
        if self._runner is not None:
            self._runner.close()
        self._runner = None

    def download(self, path: "str | os.PathLike[str] | None" = None) -> "Baked":
        """Bring weights trained on a service here (merged into a standard
        folder). Returns the model, loaded from its folder."""
        w = self.meta["weights"]
        if w.get("form") != "remote":
            return self
        from .trainers import trainer
        return trainer(w["service"]).download(self, Path(path).expanduser().resolve() if path else self.path)

    # ---- the head path

    def _ensure(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            from . import heads
            head = self.meta["head"]
            self._model, self._tokenizer = heads.load(
                str(self.path), head["architecture"], [len(f.keys) for f in self.fields],
                decoder=head.get("decoder", False), device=self.device)

    def _run_texts(self, texts: Sequence[str]) -> List[Any]:
        self._ensure()
        from . import heads
        from .metrics import softmax
        head = self.meta["head"]
        ids, _cut = heads.encode(self._tokenizer, texts, head["max_length"])
        zs = heads.logits(self._model, self._tokenizer, ids, self.device)
        temps = head["temperatures"]
        return [{f.name: dict(zip(f.keys, softmax(zs[i][r], temps[i]))) for i, f in enumerate(self.fields)}
                for r in range(len(texts))]

    def probabilities(self, texts: Sequence[str]) -> List[Dict[str, Dict[str, float]]]:
        """Per text, the probability of every answer, per field (texts as the layout writes them)."""
        if self.kind != "head":
            raise BakeError("probabilities come from head models; a generative student writes text")
        return self._run_texts(list(texts))

    def texts(self, rows: Sequence[Dict[str, Any]]) -> List[str]:
        """The input text the model reads for each row of inputs (written by its layout)."""
        from .. import engine
        entry = self._only()
        plan = input_plan(entry.signature, lmcc.load(entry.layout))
        out = []
        for row in rows:
            values = {}
            for f in entry.signature.inputs:
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
        """A head model's answers for many rows at once (the fast path for big
        tables): per row, each field's answer, its probability, and the full
        distribution."""
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
            return self.runner.complete(request)
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

    def requirements(self) -> List[str]:
        """The packages running it needs here."""
        w = self.meta.get("weights", {})
        if w.get("form") == "remote" and w.get("service") == "tinker":
            return ["tinker", "transformers"]
        return ["torch", "transformers", "safetensors"] + (["peft"] if w.get("form") == "lora" else [])

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
    meta["sizes"] = {rel: (path / rel).stat().st_size for rel in meta["hashes"]}
    tmp = path / "baked.json.tmp"
    tmp.write_text(json.dumps(meta, indent=1, ensure_ascii=False, default=str) + "\n")
    tmp.replace(path / "baked.json")


def is_baked(obj: Any) -> bool:
    return getattr(type(obj), "__functai_baked__", False) is True


def load(path: "str | os.PathLike[str]", *, device: Optional[str] = None) -> Baked:
    """A baked model from its folder (its file hashes are checked)."""
    return Baked(path, device=device)


__all__ = ["Baked", "load", "is_baked", "write_meta", "default_home", "PROVIDER", "PROVIDER_LM", "FORMAT"]
