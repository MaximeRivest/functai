"""Prediction: what one call produced.

A read-only mapping with attribute access, so ``dict(pred)``,
``pred.result`` and ``pred["result"]`` all work. (Data for evaluation and
optimization is plain rows: dicts, or a table; see ``functai.evaluation``.)
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any, Dict, Iterable, Iterator, Optional


class _Record(Mapping):
    __slots__ = ("_store",)

    def __init__(self, store: Dict[str, Any]):
        object.__setattr__(self, "_store", dict(store))

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return self._store[name]
        except KeyError:
            raise AttributeError(f"{type(self).__name__} has no field {name!r}; "
                                 f"fields: {list(self._store)}") from None

    def __getitem__(self, key: str) -> Any:
        return self._store[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._store)

    def __len__(self) -> int:
        return len(self._store)

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            self._store[name] = value

    def __repr__(self) -> str:
        inner = ", ".join(f"{k}={v!r}" for k, v in self._store.items())
        return f"{type(self).__name__}({inner})"

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._store)


def _answer_key(value: Any) -> str:
    """How an answer is keyed in probabilities ("true", an enum's value as text)."""
    import enum
    import json
    if isinstance(value, enum.Enum):
        value = value.value
    if isinstance(value, bool):
        return "true" if value else "false"
    return value if isinstance(value, str) else json.dumps(value)


class Prediction(_Record):
    """Everything one call produced.

    - the outputs, by name (``pred.result``, ``pred.reasoning``, ``dict(pred)``)
    - ``pred.turn``: the lmcc turn (inputs, every model and tool step, outputs)
    - ``pred.response`` / ``pred.responses``: the lm15 responses
    - ``pred.usage``: tokens summed over every model call
    - ``pred.repairs``: what the reader forgave in the reply
    - ``pred.call_id``: the call's id, for ``functai.rate`` and the call log
    """

    __slots__ = ("turn", "response", "responses", "repairs", "attempts", "probabilities",
                 "measured_by", "escalated", "first", "refusal", "call_id")

    def __init__(self, values: Dict[str, Any], *, turn=None, response=None, responses: Iterable = (),
                 repairs: Iterable = (), attempts: int = 1, probabilities=None, measured_by=None):
        super().__init__(values)
        object.__setattr__(self, "escalated", False)   # True: a first model was unsure; .first is its answer
        object.__setattr__(self, "first", None)
        object.__setattr__(self, "refusal", None)      # on_unreadable="record": why the reply had no values
        object.__setattr__(self, "call_id", None)      # set by the call that produced it
        object.__setattr__(self, "turn", turn)
        object.__setattr__(self, "response", response)
        object.__setattr__(self, "responses", list(responses))
        object.__setattr__(self, "repairs", list(repairs))
        object.__setattr__(self, "attempts", attempts)
        object.__setattr__(self, "probabilities", dict(probabilities or {}))
        object.__setattr__(self, "measured_by", dict(measured_by or {}))

    @property
    def confidence(self) -> Optional[float]:
        """How sure the model was: the probability it gave its own answer, the lowest
        over the outputs it measured; None when it measured none."""
        if not self.probabilities:
            return None
        values = []
        for field, dist in self.probabilities.items():
            if not dist:
                continue
            p = dist.get(_answer_key(self._store.get(field)))
            values.append(p if p is not None else max(dist.values()))
        return min(values) if values else None

    @property
    def usage(self) -> Dict[str, int]:
        total: Dict[str, int] = {}
        for r in self.responses:
            usage = getattr(r, "usage", None)
            if usage is None:
                continue
            for k, v in dataclasses.asdict(usage).items():
                if isinstance(v, int):
                    total[k] = total.get(k, 0) + v
        return total
