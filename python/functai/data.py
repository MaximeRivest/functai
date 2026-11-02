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


def usage_of(response: Any) -> Dict[str, int]:
    """The token counts one lm15 response reported, by name (``input_tokens``,
    ``output_tokens``, ``reasoning_tokens``, ``total_tokens``, ...). lm15 2
    gives each counter one meaning on every provider: input includes cached
    tokens, output includes reasoning. A counter the provider did not report
    exactly is absent, never a guess."""
    usage = getattr(response, "usage", None)
    if usage is None or not dataclasses.is_dataclass(usage):
        return {}
    out: Dict[str, int] = {}
    for f in dataclasses.fields(usage):
        v = getattr(usage, f.name)
        if isinstance(v, int) and not isinstance(v, bool):
            out[f.name] = v
    return out


def sum_usage(parts: Iterable[Mapping[str, int]]) -> Dict[str, int]:
    """Token counts summed over model calls (contract/calls.md, ``usage``):
    a counter is in the sum only when every call reported it, so a sum is
    exact or absent, never a part of the total. No calls: ``{}``."""
    total: Optional[Dict[str, int]] = None
    for part in parts:
        if total is None:
            total = dict(part)
        else:
            total = {k: v + part[k] for k, v in total.items() if k in part}
    return total or {}


def usage_read(usage: Optional[Mapping[str, Any]], lm15_version: Optional[str]) -> Dict[str, int]:
    """Token counts kept by an earlier run, with the meaning lm15 2 gives them
    (contract/calls.md, *Token counts written with lm15 1*).

    ``lm15_version`` is the lm15 that wrote them (a call record's
    ``process.lm15``); None when the writer did not say. lm15 1 kept each
    provider's own counting: Anthropic's input left the cached tokens out,
    Gemini's and xAI's output left the reasoning out. Those counts are kept
    only when they cannot differ from lm15 2's: ``total_tokens`` is
    ``input_tokens + output_tokens`` (no reasoning or tool tokens outside
    them) and no cache count is above 0. Otherwise every count is unknown
    (``{}``): a number that may mean something else is worse than none."""
    counts = {k: v for k, v in (usage or {}).items() if isinstance(v, int) and not isinstance(v, bool)}
    major = str(lm15_version or "").split(".", 1)[0]
    if major.isdigit() and int(major) >= 2:
        return counts
    i, o, t = counts.get("input_tokens"), counts.get("output_tokens"), counts.get("total_tokens")
    if i is None or o is None or t is None or t != i + o:
        return {}
    if counts.get("cache_read_tokens", 0) > 0 or counts.get("cache_write_tokens", 0) > 0:
        return {}
    return counts


class Prediction(_Record):
    """Everything one call produced.

    - the outputs, by name (``pred.result``, ``pred.reasoning``, ``dict(pred)``)
    - ``pred.turn``: the lmcc turn (inputs, every model and tool step, outputs)
    - ``pred.response`` / ``pred.responses``: the lm15 responses
    - ``pred.usage``: tokens summed over every model call (a counter one of
      them did not report is absent)
    - ``pred.repairs``: what the reader forgave in the reply
    - ``pred.call_id``: the call's id, for ``functai.rate`` and the call log
    """

    __slots__ = ("turn", "response", "responses", "repairs", "attempts", "probabilities",
                 "measured_by", "escalated", "first", "refusal", "call_id", "tool_calls")

    def __init__(self, values: Dict[str, Any], *, turn=None, response=None, responses: Iterable = (),
                 repairs: Iterable = (), attempts: int = 1, probabilities=None, measured_by=None):
        super().__init__(values)
        object.__setattr__(self, "escalated", False)   # True: a first model was unsure; .first is its answer
        object.__setattr__(self, "first", None)
        object.__setattr__(self, "refusal", None)      # on_unreadable="record": why the reply had no values
        object.__setattr__(self, "call_id", None)      # set by the call that produced it
        object.__setattr__(self, "tool_calls", None)   # with tools: every tool call asked for, across steps
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
        """Tokens summed over this call's model calls (a count one did not report is absent)."""
        return sum_usage(usage_of(r) for r in self.responses)
