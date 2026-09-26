"""Example (training data) and Prediction (what one call produced).

Both are read-only mappings with attribute access, so ``dict(pred)``,
``pred.result`` and ``pred["result"]`` all work, as they did with DSPy.
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

    def toDict(self) -> Dict[str, Any]:  # noqa: N802 — DSPy's spelling, kept for migrants
        return dict(self._store)

    to_dict = toDict


class Example(_Record):
    """One training example: ``Example(question="2+2", result="4").with_inputs("question")``.

    Keys named in ``with_inputs`` are inputs; the rest are labels. When no
    inputs are marked, functai treats the keys that match the function's
    parameters as inputs.
    """

    __slots__ = ("_input_keys",)

    def __init__(self, base: Optional[Mapping] = None, **fields: Any):
        store = dict(base or {})
        store.update(fields)
        super().__init__(store)
        keys = getattr(base, "_input_keys", None) if base is not None else None
        object.__setattr__(self, "_input_keys", frozenset(keys) if keys else None)

    def with_inputs(self, *keys: str) -> "Example":
        ex = Example(self._store)
        object.__setattr__(ex, "_input_keys", frozenset(keys))
        return ex

    @property
    def input_keys(self) -> Optional[frozenset]:
        return self._input_keys

    def inputs(self) -> "Example":
        if self._input_keys is None:
            raise ValueError("inputs() needs the input keys: call .with_inputs(...) first")
        return Example({k: v for k, v in self._store.items() if k in self._input_keys}).with_inputs(
            *self._input_keys)

    def labels(self) -> "Example":
        keys = self._input_keys or frozenset()
        return Example({k: v for k, v in self._store.items() if k not in keys})

    def copy(self, **fields: Any) -> "Example":
        ex = Example({**self._store, **fields})
        object.__setattr__(ex, "_input_keys", self._input_keys)
        return ex

    def without(self, *keys: str) -> "Example":
        ex = Example({k: v for k, v in self._store.items() if k not in keys})
        object.__setattr__(ex, "_input_keys", self._input_keys)
        return ex

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Example) and self._store == other._store

    __hash__ = None  # mutable mapping semantics


class Prediction(_Record):
    """Everything one call produced.

    - the outputs, by name (``pred.result``, ``pred.reasoning``, ``dict(pred)``)
    - ``pred.turn``: the lmcc turn (inputs, every model and tool step, outputs)
    - ``pred.response`` / ``pred.responses``: the lm15 responses
    - ``pred.usage``: tokens summed over every model call
    - ``pred.repairs``: what the reader forgave in the reply
    """

    __slots__ = ("turn", "response", "responses", "repairs", "attempts", "probabilities",
                 "measured_by")

    def __init__(self, values: Dict[str, Any], *, turn=None, response=None, responses: Iterable = (),
                 repairs: Iterable = (), attempts: int = 1, probabilities=None, measured_by=None):
        super().__init__(values)
        object.__setattr__(self, "turn", turn)
        object.__setattr__(self, "response", response)
        object.__setattr__(self, "responses", list(responses))
        object.__setattr__(self, "repairs", list(repairs))
        object.__setattr__(self, "attempts", attempts)
        object.__setattr__(self, "probabilities", dict(probabilities or {}))
        object.__setattr__(self, "measured_by", dict(measured_by or {}))

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


def as_example(item: Any, input_names: Iterable[str]) -> Example:
    """An Example from what users pass as training data: an Example, a
    DSPy Example (duck-typed, no DSPy import), a dict, or an ``(inputs,
    outputs)`` pair of dicts."""
    names = list(input_names)
    if isinstance(item, Example):
        return item if item.input_keys is not None else item.with_inputs(*[k for k in item if k in names])
    keys = getattr(item, "_input_keys", None)
    to_dict = getattr(item, "toDict", None)
    if callable(to_dict):                                   # dspy.Example
        data = dict(to_dict())
        return Example(data).with_inputs(*(keys or [k for k in data if k in names]))
    if isinstance(item, Mapping):
        if set(item) == {"inputs", "outputs"} and isinstance(item["inputs"], Mapping):
            return Example({**item["inputs"], **item["outputs"]}).with_inputs(*item["inputs"])
        return Example(dict(item)).with_inputs(*[k for k in item if k in names])
    if isinstance(item, (list, tuple)) and len(item) == 2 and all(isinstance(x, Mapping) for x in item):
        ins, outs = item
        return Example({**ins, **outs}).with_inputs(*ins)
    raise TypeError(f"a training example is an Example, a dict, or an (inputs, outputs) pair of dicts, "
                    f"not {type(item).__name__}")
