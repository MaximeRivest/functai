"""One function a student answers: how its calls are laid out for the
student, and what the student may leave out.

An ``Entry`` is the contract between training and calling. Training writes
every example through it; ``fn.using(lm=baked)`` lays out every call through
the same entry, read back from ``baked.json``. So the tokens a student is
trained on are the tokens it is called with.

- ``layout``: the lmcc adapter (default the function's own), with replies
  written from values, so a training reply is the layout's canonical reply.
- ``fixed``: inputs given one value at bake time. They are left out of the
  student's prompt; ``baked.json`` keeps a hash of each value, and a call
  with another value is refused (the student never learned to read it).
- ``derived``: inputs decided by another input (``{"guidance": "section"}``):
  left out too; a call is refused when the pair was never seen in training.
- No worked examples (demos): a trained student does not need them, and they
  would cost tokens on every call. A student is trained and called without.
"""

from __future__ import annotations

import dataclasses
import hashlib
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import lmcc
from lmcc import core as lmcc_core

from .examples import BakeError, row_inputs

CAPABILITIES = {"instruct": True}


def value_hash(value: Any) -> str:
    """``sha256:`` of a value's canonical JSON (the bytes every language writes)."""
    from ..calllog import canonical
    try:
        plain = lmcc.turn.to_json(value)
    except lmcc.Refusal:
        plain = repr(value)
    return "sha256:" + hashlib.sha256(canonical(plain).encode()).hexdigest()


def _preview(value: Any, n: int = 60) -> str:
    text = value if isinstance(value, str) else repr(value)
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1] + "…"


def layout_of(fn, layout: Any = None) -> lmcc.Adapter:
    """The lmcc adapter a student is trained and called with: ``layout`` (an
    adapter name, artifact or template), else the function's own; with replies
    written from values."""
    from .. import adapters
    settings = fn._effective()
    if layout is not None:
        adapter = adapters.template_adapter(layout) if isinstance(layout, (list, tuple)) else \
            adapters.resolve_adapter(layout)
    elif fn._template is not None:
        adapter = adapters.template_adapter(fn._template)
    else:
        adapter = adapters.resolve_adapter(settings.get("adapter") or "xml")
    if adapter.replay == "values":
        return adapter
    return lmcc.adapter(name=adapter.name, messages=adapter.template, reader=adapter.reader,
                        transports=adapter.transports, formats=adapter.formats, extensions=adapter.extensions,
                        replay="values", strict=adapter.strict)


def reduced_signature(signature: lmcc.SignatureCore, leave_out: Sequence[str]) -> lmcc.SignatureCore:
    if not leave_out:
        return signature
    kept = [f for f in signature.fields if not (f.direction == "input" and f.name in leave_out)]
    return lmcc_core._validated(lmcc.SignatureCore(signature.instructions, kept))


@dataclasses.dataclass
class Entry:
    """One function a generative student answers (see the module docstring)."""
    name: str
    fingerprint: str                       # lmcc fingerprint of the function's full signature
    signature: lmcc.SignatureCore          # what the student reads: the full one minus fixed and derived inputs
    layout: Dict[str, Any]                 # the lmcc adapter artifact
    outputs: List[str]
    reasoning: bool = False
    fixed: Dict[str, str] = dataclasses.field(default_factory=dict)          # input → value hash
    derived: Dict[str, Dict[str, Any]] = dataclasses.field(default_factory=dict)  # input → {"from", "values"}
    capabilities: Dict[str, bool] = dataclasses.field(default_factory=lambda: dict(CAPABILITIES))
    fn: Any = dataclasses.field(default=None, repr=False, compare=False)
    spec: Any = dataclasses.field(default=None, repr=False, compare=False)        # the function's (full) spec
    fixed_values: Dict[str, Any] = dataclasses.field(default_factory=dict, repr=False, compare=False)
    _plan: Any = dataclasses.field(default=None, repr=False, compare=False)

    # ---- building

    @classmethod
    def build(cls, fn, *, layout: Any = None, reasoning: bool = False, fixed: Optional[Mapping[str, Any]] = None,
              derived: Optional[Mapping[str, str]] = None, rows: Sequence[Mapping[str, Any]] = ()) -> "Entry":
        """The entry for ``fn``. ``rows`` are checked against ``fixed`` and
        ``derived`` (a fixed input must not vary; a derived one must follow
        its source)."""
        if fn._tools:
            raise BakeError(f"{fn.__name__} uses tools; training tool-calling students is not supported yet. "
                            f"Reinforcement learning on Prime (functai.bake.prime) runs the tool loop")
        s = fn._effective()
        from ..core import _module_name
        cot = bool(reasoning) and _module_name(s.get("module")) == "cot"
        spec = fn._variant_spec(reasoning=cot, tools=False)
        names = [f.name for f in spec.signature.inputs if f.purpose == "plain"]
        fixed = dict(fixed or {})
        derived = dict(derived or {})
        for n in list(fixed) + list(derived):
            if n not in names:
                raise BakeError(f"{fn.__name__} has no input {n!r} to leave out (its inputs: {names})")
        both = set(fixed) & set(derived)
        if both:
            raise BakeError(f"{sorted(both)} cannot be both fixed and derived")
        for n, src in derived.items():
            if src not in names or src in fixed or src in derived:
                raise BakeError(f"derived={{{n!r}: {src!r}}}: {src!r} must be another input the student still reads")
        prepared = _prepare(spec, fixed)
        fixed_hashes = {n: value_hash(prepared[n]) for n in fixed}
        for i, row in enumerate(rows):
            for n, h in fixed_hashes.items():
                if n in row and value_hash(_prepare(spec, {n: row[n]})[n]) != h:
                    raise BakeError(f"fixed={{{n!r}: ...}}: row {i} gives {n} another value "
                                    f"({_preview(row[n])!r}); a fixed input has one value in every row. "
                                    f"Leave it out of fixed=, or make the rows agree")
        table: Dict[str, Dict[str, Any]] = {}
        for n, src in derived.items():
            values: Dict[str, str] = {}
            for i, row in enumerate(rows):
                if n not in row or src not in row:
                    raise BakeError(f"derived={{{n!r}: {src!r}}}: row {i} lacks {n!r} or {src!r}")
                p = _prepare(spec, {n: row[n], src: row[src]})
                k, v = value_hash(p[src]), value_hash(p[n])
                if values.setdefault(k, v) != v:
                    raise BakeError(f"derived={{{n!r}: {src!r}}}: two rows with the same {src} "
                                    f"({_preview(row[src])!r}) give {n} different values, so {src} does not "
                                    f"decide it. Keep {n} as an input")
            table[n] = {"from": src, "values": values}
        adapter = layout_of(fn, layout)
        sig = reduced_signature(spec.signature, list(fixed) + list(derived))
        entry = cls(name=fn.__name__, fingerprint=lmcc.signature_fingerprint(spec.signature), signature=sig,
                    layout=adapter.dump(), outputs=[f.name for f in spec.signature.outputs
                                                    if f.purpose not in ("tools.calls",)],
                    reasoning=cot, fixed=fixed_hashes, derived=table, fn=fn, spec=spec,
                    fixed_values={n: prepared[n] for n in fixed})
        return entry

    # ---- the plan the student is called through

    @property
    def plan(self) -> lmcc.Plan:
        if self._plan is None:
            from .. import adapters
            self._plan = lmcc.load(self.layout).bind(self.signature, self.capabilities, registry=adapters.REGISTRY)
        return self._plan

    @property
    def left_out(self) -> List[str]:
        return list(self.fixed) + list(self.derived)

    def student_spec(self, spec):
        """``spec`` (the function's) as the student reads it."""
        if not self.left_out:
            return spec
        return dataclasses.replace(spec, signature=reduced_signature(spec.signature, self.left_out),
                                   params=tuple(p for p in spec.params if p not in self.left_out))

    def check(self, spec) -> None:
        """Refuse a function whose signature changed since baking."""
        if lmcc.signature_fingerprint(spec.signature) == self.fingerprint:
            return
        was = {f.name: (f.direction, f.type) for f in self.signature.fields}
        now = {f.name: (f.direction, f.type) for f in spec.signature.fields if f.name not in self.left_out}
        diff = sorted(set(was.items()) ^ set(now.items()))
        what = f"inputs, outputs or their types differ: {diff[:6]}" if diff else "its instruction changed"
        raise BakeError(f"{self.name} has changed since it was baked ({what}); bake it again")

    def reduce(self, spec, inputs: Mapping[str, Any], *, check: bool = True) -> Tuple[Any, Dict[str, Any]]:
        """``(the student's spec, the inputs it reads)`` for a call, after
        checking that fixed inputs have their baked values and derived ones
        follow their source (``check=False``: synthetic inputs, for probes)."""
        if not self.left_out:
            return spec, dict(inputs)
        if check:
            p = _prepare(spec, {k: v for k, v in inputs.items() if k in self.fixed or k in self.derived
                                or any(d["from"] == k for d in self.derived.values())})
            for n, h in self.fixed.items():
                if n in p and value_hash(p[n]) != h:
                    raise BakeError(f"{self.name}: this baked model was trained with {n} fixed to one value, and "
                                    f"this call gives another ({_preview(inputs[n])!r}). The student never learned "
                                    f"to read {n}: call it with the baked value, or bake again with this one")
            for n, d in self.derived.items():
                src = d["from"]
                if src not in p:
                    continue
                want = d["values"].get(value_hash(p[src]))
                if want is None:
                    raise BakeError(f"{self.name}: this baked model never saw {src}={_preview(inputs[src])!r} in "
                                    f"training, so it does not know the {n} that goes with it. Bake again with "
                                    f"rows that have it, or call the function on another model")
                if n in p and value_hash(p[n]) != want:
                    raise BakeError(f"{self.name}: {n} is not the one the baked model learned for "
                                    f"{src}={_preview(inputs[src])!r}; bake again with this pair")
        return self.student_spec(spec), {k: v for k, v in inputs.items() if k not in self.left_out}

    # ---- training text

    def messages(self, inputs: Mapping[str, Any], outputs: Optional[Mapping[str, Any]] = None
                 ) -> Tuple[List[Dict[str, str]], Optional[str]]:
        """``(the chat messages of the call, the reply the layout writes for
        outputs)`` for one row, as the student sees it. The prompt is a fresh
        render of the call (exactly what a call sends); the reply is the
        layout's canonical reply for the outputs."""
        from .. import engine
        spec = self.student_spec(self.spec) if self.spec is not None else None
        if spec is None:
            raise BakeError("an entry read from baked.json lays out calls, not training examples")
        values = engine.prepare_inputs(spec, {k: v for k, v in inputs.items() if k not in self.left_out})
        plan = self.plan
        request = plan.render(plan.turn(values)).request("student")
        msgs = chat_messages(request)
        if outputs is None:
            return msgs, None
        example = plan.example(values, {k: outputs[k] for k in self.outputs if k in outputs})
        rendered = plan.render(plan.turn(values), turns=[example]).request("student")
        both = chat_messages(rendered)
        at = max(k for k, m in enumerate(both) if m["role"] == "assistant")
        return msgs, both[at]["content"]

    def row_inputs(self, row: Mapping[str, Any]) -> Dict[str, Any]:
        """A row's inputs for the function, fixed values filled in where the
        row leaves them out."""
        filled = dict(row)
        for n, v in self.fixed_values.items():
            filled.setdefault(n, v)
        return row_inputs(self.fn, filled)

    # ---- baked.json

    def to_meta(self) -> Dict[str, Any]:
        return {"name": self.name, "fingerprint": self.fingerprint, "signature": lmcc.signature_to_dict(self.signature),
                "layout": self.layout, "outputs": list(self.outputs), "reasoning": self.reasoning,
                "fixed": dict(self.fixed), "derived": dict(self.derived), "capabilities": dict(self.capabilities)}

    @classmethod
    def from_meta(cls, d: Mapping[str, Any]) -> "Entry":
        return cls(name=d["name"], fingerprint=d["fingerprint"], signature=lmcc.signature_from_dict(d["signature"]),
                   layout=d["layout"], outputs=list(d.get("outputs") or []), reasoning=bool(d.get("reasoning")),
                   fixed=dict(d.get("fixed") or {}), derived=dict(d.get("derived") or {}),
                   capabilities=dict(d.get("capabilities") or CAPABILITIES))


def _prepare(spec, values: Mapping[str, Any]) -> Dict[str, Any]:
    from .. import engine
    return engine.prepare_inputs(spec, dict(values))


# ------------------------------------------------------------------ chat messages


def chat_messages(request: Any) -> List[Dict[str, str]]:
    """An lm15 request (object or canonical JSON) as chat-template messages."""
    if not isinstance(request, dict):
        import lm15
        request = lm15.serde.request_to_dict(request)
    out: List[Dict[str, str]] = []
    system = request.get("system")
    if system:
        out.append({"role": "system", "content": system if isinstance(system, str) else _text(system)})
    for m in request.get("messages", []):
        role = m["role"]
        if role not in ("user", "assistant", "system", "developer"):
            raise BakeError(f"a generative student reads text chats; the request has a {role!r} message "
                            f"(tool calls are not trained yet)")
        out.append({"role": "system" if role == "developer" else role, "content": _text(m)})
    return out


def _text(message: Any) -> str:
    parts = message.get("parts", []) if isinstance(message, dict) else message
    texts = []
    for p in parts:
        if p.get("type") != "text":
            raise BakeError(f"a generative student reads text; the request carries a {p.get('type')} part "
                            f"(images and files are not trained yet)")
        texts.append(p["text"])
    return "".join(texts)


__all__ = ["Entry", "chat_messages", "value_hash", "layout_of", "CAPABILITIES"]
