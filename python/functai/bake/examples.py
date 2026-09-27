"""From an AI function and rows of data to training examples.

The input a trained model reads is written by lmcc, with the same layout the
function uses when the trained model answers a call (``Baked`` pins it). So
training text and call-time text come from one writer and cannot drift.

A head model answers *finite* outputs: ``Literal``, ``Enum`` and ``bool``.
Each becomes a list of answer keys, spelled as lmcc spells them in a reply
("true", an enum's value), which are also the keys of the probabilities
functai returns (``prediction.probabilities[field][key]``).
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
from typing import Any, Dict, List, Sequence, Tuple

import lmcc
from lmcc import core as lmcc_core

from .. import adapters, engine


class BakeError(ValueError):
    """The function or the data cannot be baked as asked; the message says what to do."""


def answer_key(value: Any) -> str:
    """How an answer is keyed in probabilities: "true"/"false", an enum's value as text."""
    if isinstance(value, enum.Enum):
        value = value.value
    if isinstance(value, bool):
        return "true" if value else "false"
    return value if isinstance(value, str) else json.dumps(value)


@dataclasses.dataclass(frozen=True)
class HeadField:
    """One finite output: its answer keys in a fixed order, and their JSON values."""
    name: str
    keys: Tuple[str, ...]
    values: Tuple[Any, ...]

    def index(self, label: Any) -> int:
        key = answer_key(label)
        try:
            return self.keys.index(key)
        except ValueError:
            raise KeyError(key) from None

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "keys": list(self.keys), "values": list(self.values)}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HeadField":
        return cls(d["name"], tuple(d["keys"]), tuple(d["values"]))


def _finite(shape: Dict[str, Any]) -> Any:
    if "enum" in shape:
        return tuple(shape["enum"])
    if shape.get("type") == "boolean":
        return (True, False)
    return None


def field_value(container: Any, name: str) -> Any:
    """A (possibly dotted: ``result.sentiment``) field's value in a row or a prediction."""
    head, _, rest = name.partition(".")
    try:
        value = container[head]
    except (KeyError, TypeError):
        value = getattr(container, head, None)
    if not rest or value is None:
        return value
    return field_value(value, rest)


def nest(values: Dict[str, Any]) -> Dict[str, Any]:
    """``{"result.sentiment": "pos", "result.spam": False}`` → ``{"result": {...}}``."""
    out: Dict[str, Any] = {}
    for name, v in values.items():
        head, _, rest = name.partition(".")
        if rest:
            out.setdefault(head, {})[rest] = v
        else:
            out[head] = v
    return out


def head_fields(spec) -> List[HeadField]:
    """The function's outputs as head fields; refuses outputs a classifier cannot give."""
    fields: List[HeadField] = []
    open_ended: List[str] = []
    for f in spec.signature.outputs:
        if f.purpose in ("reasoning", "tools.calls"):
            continue                         # hidden work a head does not do
        base, nullable = lmcc_core.nullable_base(f.shape)
        if nullable:
            raise BakeError(f"output {f.name!r} may be None; a head model picks one answer from a fixed list. "
                            f"Make it a Literal/Enum with an explicit 'none' answer")
        values = _finite(base)
        if values is not None:
            fields.append(HeadField(f.name, tuple(answer_key(v) for v in values), values))
            continue
        props = base.get("properties") if base.get("type") == "object" else None
        subs = {k: _finite(lmcc_core.nullable_base(v)[0]) if not lmcc_core.nullable_base(v)[1] else None
                for k, v in (props or {}).items()}
        if props and all(v is not None for v in subs.values()) and set(base.get("required", props)) == set(props):
            # a record of finite answers (a dataclass of Literals, Enums, bools): one answer layer each
            for k, vals in subs.items():
                fields.append(HeadField(f"{f.name}.{k}", tuple(answer_key(v) for v in vals), vals))
            continue
        open_ended.append(f"{f.name} ({f.type or 'text'})")
    if open_ended:
        raise BakeError(f"a head model answers finite outputs only (Literal, Enum, bool), and "
                        f"{', '.join(open_ended)} is open-ended. Bake a generative student instead: "
                        f"bake(..., method='sft'), or drop the output from the function")
    if not fields:
        raise BakeError("the function has no output to learn")
    return fields


def head_signature(spec, fields: Sequence[HeadField]) -> lmcc.SignatureCore:
    """The signature a head model serves: the plain inputs and the finite outputs."""
    names = {f.name.split(".")[0] for f in fields}
    kept = [f for f in spec.signature.fields
            if (f.direction == "input" and f.purpose == "plain") or (f.direction == "output" and f.name in names)]
    return lmcc_core._validated(lmcc.SignatureCore(spec.signature.instructions, kept))


def head_layout(signature: lmcc.SignatureCore) -> lmcc.Adapter:
    """What a head model reads: the inputs alone (one input bare, several in tags);
    no instruction. The reply is a JSON object the model's answers fill."""
    return adapters.judgment_adapter(signature)


CAPABILITIES = {"native_structured_output": True}


def input_plan(signature: lmcc.SignatureCore, layout: lmcc.Adapter) -> lmcc.Plan:
    return layout.bind(signature, CAPABILITIES, registry=adapters.REGISTRY)


def request_text(request: Any) -> str:
    """The text a trained model reads from an lm15 request: its last user message."""
    msg = request.messages[-1]
    parts = []
    for p in msg.parts:
        kind = getattr(p, "type", None) or type(p).__name__
        if kind in ("text", "TextPart"):
            parts.append(p.text)
        else:
            raise BakeError(f"a baked model reads text; the request carries a {kind} part")
    return "".join(parts)


def row_inputs(fn, row: Dict[str, Any]) -> Dict[str, Any]:
    names = [n for n in fn._sig.parameters if n in row]
    bound = fn._sig.bind_partial(**{n: row[n] for n in names})
    bound.apply_defaults()
    missing = [n for n in fn._sig.parameters if n not in bound.arguments]
    if missing:
        raise BakeError(f"rows lack the input column(s) {missing}")
    return dict(bound.arguments)


def render_input(plan: lmcc.Plan, spec, inputs: Dict[str, Any]) -> str:
    values = engine.prepare_inputs(spec, inputs)
    request = plan.render(plan.turn({k: v for k, v in values.items() if plan.signature.field_named(k)}))
    msg = request.messages[-1]
    out = []
    for p in msg["parts"]:
        if p.get("type") != "text":
            raise BakeError(f"a head model reads text; input writes a {p.get('type')} part")
        out.append(p["text"])
    return "".join(out)


# ------------------------------------------------------------------ targets


def distribution(field: HeadField, probs: Dict[str, float]) -> List[float]:
    """A teacher's distribution over this field's keys, renormalized over the keys
    it knows (mass on other text is dropped, as the records' logprob targets did)."""
    raw = [max(0.0, float(probs.get(k, 0.0))) for k in field.keys]
    total = sum(raw)
    if total <= 0:
        raise ValueError("no probability on any declared answer")
    return [p / total for p in raw]


def one_hot(field: HeadField, label: Any) -> List[float]:
    i = field.index(label)
    return [1.0 if j == i else 0.0 for j in range(len(field.keys))]


def fingerprint(texts: Sequence[str], targets: Sequence[Sequence[Sequence[float]]]) -> str:
    """A hash of the training data (inputs and targets), recorded with the model."""
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode())
        h.update(b"\0")
    h.update(json.dumps([[[round(p, 6) for p in row] for row in f] for f in targets]).encode())
    return "sha256:" + h.hexdigest()


__all__ = ["BakeError", "HeadField", "answer_key", "head_fields", "head_signature", "head_layout",
           "input_plan", "render_input", "request_text", "row_inputs", "distribution", "one_hot"]
