"""Measuring programs: cases, results, ratings and honest numbers.

The rules are the contract's (contract/measures.md, design/20): a **case**
is the inputs a program can be asked again, named by its key; a **result**
is one call of one setup on one case (a call record, carrying ``case``); a
**rating** is a judgement of a result, or of an item of it, or of two
results side by side, by a person, an account, a judge program or code.
Everything below is statistics over them, the same in every language that
measures: every range is over cases (items of one answer and repeats of one
case are not independent), right-or-wrong comparisons use the MOVER-Wilson
interval (the paired t interval says "no difference" with no uncertainty
when no case changed), and a judge is combined with a random draw of
people's ratings by prediction-powered inference (PPI++).

    functai.scoreboard(study, [real_problem])        # per setup: people, people drawn, each judge, combined
    functai.compare_setups(study, old, new, real_problem, source=professor, margin=0.05)
    functai.rank(study, preferred)                   # Bradley–Terry over every setup, with fragility
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import math
import random as _random
from collections.abc import Mapping
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .calllog import canonical

Z = 1.959964
CONTEXT = ("earlier", "sections", "helpers")
SPLITS = ("train", "held_out")
KINDS = ("verdict", "label", "score", "choice")
RANK_STEPS = 20


class MeasureError(ValueError):
    """A measure refused what it was given (``code``: ``case-split``, ...)."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _sha(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def _hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _mean(xs: Sequence[float]) -> float:
    s = 0.0
    for x in xs:
        s += x
    return s / len(xs)


def _var(xs: Sequence[float]) -> float:
    m = _mean(xs)
    return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)


# ------------------------------------------------------------------ criteria


@dataclasses.dataclass(frozen=True)
class Criterion:
    '''A question an answer is judged by: its name, its kind, and how its values count.

    Parameters
    ----------
    name : str
        Letters, digits and ``_``: ``"real_problem"``. ``"correct"`` (a
        verdict on the whole answer) is what format-1 ratings mean.
    kind : "verdict", "label", "score" or "choice"
        A verdict is ``"right"`` or ``"wrong"``; a label one of ``answers``;
        a score a number (on ``scale``, when given); a choice a pick between
        two answers (``"first"``, ``"second"`` or ``"tie"`` from a judge).
    answers : dict, optional
        A label's answers and the number each counts for, from 0 to 1, or
        None (it counts for nothing: "unsure").
    scale : (lowest, highest), optional
        A score's range; it counts as ``(value - lowest) / (highest - lowest)``.
    items : str, optional
        A JSON Pointer to a list in the answer (``"/insights"``): each item
        of it is judged.
    aggregate : "mean" or "sum"
        How a result's items add up: the share of good items, or how many.
    question : str, optional
        The words people are asked.
    judge : AI function, module or function, optional
        Who judges automatically (``evaluate(criteria=...)``, ``judge``). It
        is given by name what it asks for: the case's inputs, ``answer``,
        ``outputs``, ``expected``, ``item``, ``inputs``, and for a pair
        ``first`` and ``second``. Its answer (or its answer's ``value``
        field) is the judgement: True or False for a verdict, an answer for a
        label, a number for a score, ``"first"``, ``"second"`` or ``"tie"``
        for a choice.

    Examples
    --------
    ```python
    real_problem = functai.Criterion("real_problem", "label", items="/insights",
                                     answers={"yes": 1, "no": 0, "unsure": None},
                                     question="Is this a real problem?")
    real_problem.bounded
    ```
    '''
    name: str
    kind: str = "verdict"
    answers: Optional[Mapping[str, Optional[float]]] = None
    scale: Optional[Tuple[float, float]] = None
    items: Optional[str] = None
    aggregate: str = "mean"
    question: Optional[str] = None
    judge: Any = dataclasses.field(default=None, compare=False)

    def __post_init__(self) -> None:
        import re
        if not isinstance(self.name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.name):
            raise ValueError(f"a criterion's name is letters, digits and _, not {self.name!r}")
        if self.kind not in KINDS:
            raise ValueError(f"a criterion's kind is one of {', '.join(KINDS)}, not {self.kind!r}")
        if self.kind == "label":
            if not self.answers:
                raise ValueError(f"{self.name}: a label needs its answers, each the number it counts for (or None)")
            for k, v in self.answers.items():
                if v is not None and not (isinstance(v, (int, float)) and 0 <= v <= 1):
                    raise ValueError(f"{self.name}: answer {k!r} counts for a number from 0 to 1, or None")
            object.__setattr__(self, "answers", dict(self.answers))
        elif self.answers is not None:
            raise ValueError(f"{self.name}: only a label has answers")
        if self.scale is not None:
            if self.kind != "score" or len(self.scale) != 2 or not self.scale[0] < self.scale[1]:
                raise ValueError(f"{self.name}: a scale is a score's (lowest, highest), lowest below highest")
            object.__setattr__(self, "scale", (float(self.scale[0]), float(self.scale[1])))
        if self.aggregate not in ("mean", "sum"):
            raise ValueError(f"{self.name}: aggregate is 'mean' or 'sum'")
        if self.items is not None:
            if self.kind == "choice":
                raise ValueError(f"{self.name}: a choice is between two answers, not items")
            if not (isinstance(self.items, str) and self.items.startswith("/")):
                raise ValueError(f"{self.name}: items is a JSON Pointer to a list in the answer, like '/insights'")

    def data(self) -> Dict[str, Any]:
        """The criterion as the contract writes it."""
        out: Dict[str, Any] = {"name": self.name, "kind": self.kind}
        if self.items is not None:
            out["items"] = self.items
            out["aggregate"] = self.aggregate
        if self.answers is not None:
            out["answers"] = dict(self.answers)
        if self.scale is not None:
            out["scale"] = list(self.scale)
        if self.question:
            out["question"] = self.question
        return out

    @property
    def bounded(self) -> bool:
        """Its values lie between 0 and 1 (not a count of items, nor a score without a scale)."""
        return bounded(self.data())

    def with_judge(self, judge: Any) -> "Criterion":
        return dataclasses.replace(self, judge=judge)


correct = Criterion("correct", "verdict")
preferred = Criterion("preferred", "choice", question="Which answer is better?")


def criterion_of(c: Any) -> Dict[str, Any]:
    if isinstance(c, Criterion):
        return c.data()
    if isinstance(c, Mapping):
        return Criterion(**{k: v for k, v in c.items() if k in {f.name for f in dataclasses.fields(Criterion)}}).data()
    if isinstance(c, str):
        return {"name": c, "kind": "verdict"}
    raise TypeError(f"a criterion is a functai.Criterion, a dict or a name, not {type(c).__name__}")


def bounded(criterion: Mapping[str, Any]) -> bool:
    if criterion.get("items") is not None and criterion.get("aggregate", "mean") == "sum":
        return False
    return criterion["kind"] != "score" or criterion.get("scale") is not None


# ------------------------------------------------------------------ case keys


def case_key_of(inputs: Mapping[str, Any], context: Optional[Mapping[str, Any]] = None) -> str:
    """The key of inputs given as JSON (and a conversation's context)."""
    doc: Dict[str, Any] = {"inputs": dict(inputs)}
    for k in CONTEXT:
        v = (context or {}).get(k)
        if v not in (None, [], {}):
            doc[k] = v
    return _sha(doc)


def row_key(row: Mapping[str, Any], inputs: Sequence[str]) -> str:
    """A row's case key, its values written as a call record writes them."""
    from .calllog import to_json
    return case_key_of({n: to_json(row[n])[0] for n in inputs if n in row},
                       {k: to_json(row[k])[0] for k in CONTEXT if k in row and k not in inputs})


def _inputs_kept(call: Mapping[str, Any]) -> bool:
    if (call.get("described") or {}).get("inputs"):
        return False
    if call.get("content") is True:
        return "inputs" in call
    om = call.get("omitted")
    return call.get("functai_call") == 2 and isinstance(om, Mapping) and om.get("inputs") == []


def call_key(call: Mapping[str, Any]) -> Optional[str]:
    """A call record's case: its ``case``, else its recorded inputs' key when it saw nothing; else None."""
    if isinstance(call.get("case"), str):
        return call["case"]
    if call.get("functai_call") != 2 or call.get("saw") or not _inputs_kept(call):
        return None
    return case_key_of(call.get("inputs") or {})


def _input_names(program: Any) -> List[str]:
    from .evaluation import _Target
    return list(_Target(program).input_names)


def case_key(program: Any, row: Mapping[str, Any]) -> str:
    '''The key of the case a row is for a program.

    Its values of the program's inputs (and a conversation's earlier turns,
    when the row keeps them) decide it; nothing else of the row does, so
    correcting its answer keeps its key. A call ``evaluate`` makes on the row
    carries it (``case`` in its record).

    Parameters
    ----------
    program : AI function or module, or a list of input names
    row : dict

    Returns
    -------
    str
        ``"sha256:…"``.

    Examples
    --------
    ```python
    functai.case_key(["message"], {"message": "Charged twice.", "result": "billing"})
    ```
    '''
    names = list(program) if isinstance(program, (list, tuple)) else _input_names(program)
    return row_key(row, names)


# ------------------------------------------------------------------ held out


def split_of(row: Mapping[str, Any], program: Any = None, *, share: float = 0.2, salt: str = "") -> str:
    '''Whether a row is for training or held out, the same in every language.

    The row's own ``split`` (``"train"`` or ``"held_out"``) when it has one;
    else held out when the hash of its key (its conversation's id, so every
    turn of a conversation is on one side) falls below ``share``. Adding rows
    never moves an old one.

    Parameters
    ----------
    row : dict
    program : AI function, module, or list of input names
        Whose inputs make the key (required unless the row has a ``split``).
    share : float
        The share held out, from 0 to 1.
    salt : str
        Another salt, another split of the same rows.

    Returns
    -------
    "train" or "held_out"

    Examples
    --------
    ```python
    [functai.split_of({"message": m}, ["message"]) for m in ("a", "b", "c", "d", "e")]
    ```
    '''
    names = [] if program is None else list(program) if isinstance(program, (list, tuple)) else _input_names(program)
    if "split" in row and "split" not in names:
        s = row["split"]
        if s not in SPLITS:
            raise MeasureError("case-split", f"split is 'train' or 'held_out', not {s!r}")
        return s
    if not (0 <= share <= 1):
        raise MeasureError("case-split", f"the held-out share is between 0 and 1, not {share!r}")
    if program is None:
        raise TypeError("split_of needs the program (or its input names) for a row without a split")
    conv = row.get("conversation") if "conversation" not in names else None
    group = conv if isinstance(conv, str) and conv else row_key(row, names)
    h = _hex(f"functai-split:{salt}:{group}")
    return "held_out" if int(h[:13], 16) / 2 ** 52 < share else "train"


def held_out(rows: Iterable[Mapping[str, Any]]) -> int:
    """How many rows say they are held out (``split`` ``"held_out"``)."""
    return sum(1 for r in rows if isinstance(r, Mapping) and r.get("split") == "held_out")


# ------------------------------------------------------------------ random draws


def draw(ids: Iterable[Any], n: int, seed: str) -> List[str]:
    '''A random draw of ``n`` ids, the same in every language.

    Ranked by the hash of the seed and each id: uniform, reproducible, and
    stable (drawing from more ids keeps each earlier pick that still ranks).
    Rate the drawn calls with ``sample=seed``: only ratings of a random draw
    measure a program without the bias of choosing what to judge.

    Parameters
    ----------
    ids : list of str (or predictions, or call records)
    n : int
    seed : str
        Names the draw; becomes the ratings' ``sample``.

    Returns
    -------
    list of str

    Examples
    --------
    ```python
    functai.draw([f"call-{i}" for i in range(100)], 5, seed="review-2026-11-05")
    ```
    '''
    plain = [_id_of(i) for i in ids]
    ranked = sorted(dict.fromkeys(plain), key=lambda i: (_hex(f"functai-draw:{seed}:{i}"), i))
    return ranked[:max(0, int(n))]


# ------------------------------------------------------------------ reading ratings


def source_of(r: Mapping[str, Any]) -> Tuple[str, str]:
    if isinstance(r.get("by"), str) and r["by"]:
        return ("person", r["by"])
    if isinstance(r.get("judge"), Mapping):
        j = r["judge"]
        return ("judge", j.get("setup") or j.get("code"))
    return ("account", r["id"])


def judgement(r: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """A rating of format 1 or 2 as one shape; None for another format."""
    fmt = r.get("functai_rating")
    if fmt == 1:
        return {"id": r["id"], "at": r["at"], "calls": [r["call"]], "item": "", "criterion": "correct",
                "kind": "verdict", "value": r.get("verdict"), "source": source_of(r),
                "blind": r.get("blind") is True, "sample": r.get("sample")}
    if fmt == 2:
        try:
            calls = list(r["calls"]) if "calls" in r else [r["call"]]
            out = {"id": r["id"], "at": r["at"], "calls": calls, "item": r.get("item", ""),
                   "criterion": r["criterion"], "kind": r["kind"], "value": r.get("value"), "source": source_of(r),
                   "blind": r.get("blind") is True, "sample": r.get("sample")}
        except (KeyError, TypeError):
            return None                                   # not a rating of this format: skipped
        if r["kind"] == "label":
            out["answers"] = dict(r.get("answers") or {})
        if r["kind"] == "score" and "scale" in r:
            out["scale"] = list(r["scale"])
        if isinstance(r.get("judge"), Mapping):
            out["judge"] = r["judge"].get("name")
        return out
    return None


def counting(ratings: Iterable[Mapping[str, Any]], criterion: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """The judgements that count for a criterion (measures.md, *What counts*), by (at, id)."""
    latest: Dict[Any, Dict[str, Any]] = {}
    for r in ratings:
        j = judgement(r)
        if j is None or j["criterion"] != criterion["name"] or j["kind"] != criterion["kind"]:
            continue
        kind, who = j["source"]
        calls = tuple(sorted(j["calls"])) if kind == "person" else tuple(j["calls"])
        key = (kind, who) if kind == "account" else (kind, who, calls, j["item"])
        had = latest.get(key)
        if had is None or (j["at"], j["id"]) > (had["at"], had["id"]):
            latest[key] = j
    kept = [j for j in latest.values() if j["value"] is not None]
    return sorted(kept, key=lambda j: (j["at"], j["id"]))


def number_of(j: Mapping[str, Any]) -> Optional[float]:
    k, v = j["kind"], j["value"]
    if k == "verdict":
        return 1.0 if v == "right" else 0.0 if v == "wrong" else None
    if k == "label":
        x = (j.get("answers") or {}).get(v) if isinstance(v, str) else None
        return None if x is None else float(x)
    if k == "score":
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            return None
        if j.get("scale"):
            lo, hi = j["scale"]
            return (v - lo) / (hi - lo)
        return float(v)
    return None


def _matches(j: Mapping[str, Any], source: str, drawn: bool) -> bool:
    kind, who = j["source"]
    if source == "people":
        return kind in ("person", "account") and (not drawn or bool(j["sample"]))
    return kind == "judge" and who == source


# ------------------------------------------------------------------ results and values


def program_calls(records: Iterable[Mapping[str, Any]], program: Mapping[str, Any]) -> List[Dict[str, Any]]:
    best: Dict[str, Dict[str, Any]] = {}
    for r in records:
        if r.get("functai_call") not in (1, 2):
            continue
        p = r.get("program") or {}
        if p.get("name") != program["name"]:
            continue
        if program.get("module") is not None and p.get("module") != program["module"]:
            continue
        if program.get("file") is not None and p.get("file") != program["file"]:
            continue
        if r["id"] not in best or int(r.get("writer") or 1) > int(best[r["id"]].get("writer") or 1):
            best[r["id"]] = dict(r)
    return sorted(best.values(), key=lambda c: (c.get("started", ""), c["id"]))


def results(records: Iterable[Mapping[str, Any]], program: Mapping[str, Any], by: str = "setup") -> List[Dict[str, Any]]:
    out = []
    for c in program_calls(records, program):
        k = call_key(c)
        if k is None:
            continue
        out.append({"call": c, "group": (c.get("program") or {}).get(by), "case": k,
                    "failed": c.get("error") is not None})
    return out


_MISSING = object()


def pointer(value: Any, path: str) -> Any:
    """RFC 6901 into JSON data; ``_MISSING`` when a step is not there."""
    if path == "":
        return value
    for raw in path.split("/")[1:]:
        tok = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(value, Mapping) and tok in value:
            value = value[tok]
        elif isinstance(value, list) and tok.isdigit() and (tok == "0" or not tok.startswith("0")) and int(tok) < len(value):
            value = value[int(tok)]
        else:
            return _MISSING
    return value


def _result_value(res: Mapping[str, Any], criterion: Mapping[str, Any], by_target: Mapping[Any, List[float]]):
    items, agg = criterion.get("items"), criterion.get("aggregate", "mean")
    cid = res["call"]["id"]
    if res["failed"]:
        return 0.0 if items is None or agg == "sum" else None
    if items is None:
        vs = by_target.get(((cid,), ""))
        return _mean(vs) if vs else None
    lst = pointer(res["call"].get("outputs"), items)
    if not isinstance(lst, list):
        return 0.0 if agg == "sum" else None
    per = [_mean(by_target[((cid,), f"{items}/{i}")]) for i in range(len(lst)) if by_target.get(((cid,), f"{items}/{i}"))]
    if not per:
        return 0.0 if agg == "sum" and not lst else None
    share = _mean(per)
    return share * len(lst) if agg == "sum" else share


def _judged(ratings, criterion, source, drawn) -> Dict[Any, List[float]]:
    out: Dict[Any, List[float]] = {}
    for j in counting(ratings, criterion):
        if len(j["calls"]) != 1 or not _matches(j, source, drawn):
            continue
        x = number_of(j)
        if x is not None:
            out.setdefault((tuple(j["calls"]), j["item"]), []).append(x)
    return out


def _split_records(records: Iterable[Mapping[str, Any]]):
    calls, ratings = [], []
    for r in records:
        if "functai_rating" in r:
            ratings.append(r)
        elif "functai_call" in r:
            calls.append(r)
    return calls, ratings


def values_of(records: Sequence[Mapping[str, Any]], program: Mapping[str, Any], criterion: Mapping[str, Any],
              source: str, *, by: str = "setup", drawn: bool = False) -> Dict[str, Any]:
    """Per result its value, and per group {case: value} (measures.md, *Values*)."""
    calls, ratings = _split_records(records)
    by_target = _judged(ratings, criterion, source, drawn)
    per_result, table = [], {}
    for res in results(calls, program, by):
        v = _result_value(res, criterion, by_target)
        per_result.append({"call": res["call"]["id"], "group": res["group"], "case": res["case"], "value": v})
        if v is not None:
            table.setdefault(res["group"], {}).setdefault(res["case"], []).append(v)
    return {"results": per_result, "cases": {g: {k: _mean(vs) for k, vs in cs.items()} for g, cs in table.items()}}


def preferences_of(records, program, criterion, source, a, b, *, by="setup", drawn=False) -> Dict[str, float]:
    """{case: B's share} side by side (measures.md, *Side by side*), in key order."""
    calls, ratings = _split_records(records)
    picks: Dict[frozenset, List[Dict[str, Any]]] = {}
    for j in counting(ratings, criterion):
        if len(j["calls"]) == 2 and _matches(j, source, drawn):
            picks.setdefault(frozenset(j["calls"]), []).append(j)
    res = results(calls, program, by)
    ra = [r for r in res if r["group"] == a]
    rb = [r for r in res if r["group"] == b]
    out: Dict[str, List[float]] = {}
    for x in ra:
        for y in rb:
            if x["case"] != y["case"]:
                continue
            xid, yid = x["call"]["id"], y["call"]["id"]
            if x["failed"] or y["failed"]:
                v = 0.5 if x["failed"] and y["failed"] else (1.0 if x["failed"] else 0.0)
            else:
                js = picks.get(frozenset((xid, yid)))
                if not js:
                    continue
                v = _mean([1.0 if j["value"] == yid else 0.0 if j["value"] == xid else 0.5 for j in js])
            out.setdefault(x["case"], []).append(v)
    return {k: _mean(out[k]) for k in sorted(out)}


# ------------------------------------------------------------------ the statistics


def _t975(df: int) -> float:
    from .evaluation import _t975 as t
    return t(df)


def interval(values: Sequence[float]):
    from .evaluation import interval as iv
    return iv(values)


def wilson(x: float, n: int) -> Tuple[float, float]:
    p = x / n
    z2 = Z * Z
    center = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = Z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (0.0 if p == 0 else max(0.0, center - half), 1.0 if p == 1 else min(1.0, center + half))


def outcome(low: Optional[float], high: Optional[float], margin: Optional[float] = None) -> str:
    """What a range of a difference says: better, worse, same, not_worse, not_better or unclear."""
    if low is None:
        return "unclear"
    if low > 0:
        return "better"
    if high < 0:
        return "worse"
    if margin:
        if low > -margin and high < margin:
            return "same"
        if low > -margin:
            return "not_worse"
        if high < margin:
            return "not_better"
    return "unclear"


def fragility(d: Sequence[float]) -> int:
    """The fewest values whose removal takes their mean to 0 or across it."""
    if not d:
        return 0
    m = _mean(d)
    if m == 0:
        return 0
    s = 1.0 if m > 0 else -1.0
    order = sorted(range(len(d)), key=lambda i: (-s * d[i], i))
    left = [s * d[i] for i in order]
    total = sum(left)
    for k in range(1, len(d) + 1):
        total -= left[k - 1]
        rest = len(d) - k
        if rest == 0 or total / rest <= 0:
            return k
    return len(d)


def paired(a: Sequence[float], b: Sequence[float], margin: Optional[float] = None, bounded_: bool = False
           ) -> Dict[str, Any]:
    '''Is B better than A, case by case: the mean difference with its 95% range.

    When every value is 0 or 1, the MOVER-Wilson interval with the
    continuity-corrected correlation (exact coverage stays near 95% from 10
    cases, where the paired t interval can hold the truth 2% of the time);
    otherwise the paired t interval.

    Parameters
    ----------
    a, b : list of float
        The two versions' values on the same cases, in the same order.
    margin : float, optional
        How much worse is still acceptable: gives ``same``, ``not_worse`` and
        ``not_better`` outcomes.
    bounded_ : bool
        The values are between 0 and 1: the range is kept within -1 and 1.

    Returns
    -------
    dict
        ``diff``, ``low``, ``high``, ``n``, ``method``, ``outcome``,
        ``better``, ``worse``, ``same``, ``fragility``.

    Examples
    --------
    ```python
    functai.measures.paired([1] * 30, [1] * 30)["low"]      # not 0: thirty cases cannot show "no difference"
    ```
    '''
    a, b = [float(x) for x in a], [float(x) for x in b]
    if len(a) != len(b):
        raise ValueError(f"paired needs one value of each per case: {len(a)} against {len(b)}")
    n = len(a)
    if n == 0:
        return {"diff": None, "low": None, "high": None, "n": 0, "method": None, "outcome": "unclear",
                "better": 0, "worse": 0, "same": 0, "fragility": 0}
    d = [y - x for x, y in zip(a, b)]
    diff = _mean(d)
    low = high = method = None
    if n >= 2:
        if all(v in (0.0, 1.0) for v in a + b):
            method = "mover-wilson"
            n11 = sum(1 for x, y in zip(a, b) if x == 1 and y == 1)
            n10 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
            n01 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
            n00 = n - n11 - n10 - n01
            pa, pb = (n11 + n10) / n, (n11 + n01) / n
            la, ua = wilson(n11 + n10, n)
            lb, ub = wilson(n11 + n01, n)
            num: float = n11 * n00 - n10 * n01
            if num > 0:
                num = max(num - n / 2, 0.0)
            den = (n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)
            phi = 0.0 if den == 0 else num / math.sqrt(den)
            dl = math.sqrt(max(0.0, (pb - lb) ** 2 - 2 * phi * (pb - lb) * (ua - pa) + (ua - pa) ** 2))
            du = math.sqrt(max(0.0, (ub - pb) ** 2 - 2 * phi * (ub - pb) * (pa - la) + (pa - la) ** 2))
            low, high = max(-1.0, diff - dl), min(1.0, diff + du)
        else:
            method = "paired-t"
            ss = sum((v - diff) ** 2 for v in d)
            half = _t975(n - 1) * math.sqrt(ss / (n - 1)) / math.sqrt(n)
            low, high = diff - half, diff + half
            if bounded_:
                low, high = max(-1.0, low), min(1.0, high)
    return {"diff": diff, "low": low, "high": high, "n": n, "method": method, "outcome": outcome(low, high, margin),
            "better": sum(1 for v in d if v > 0), "worse": sum(1 for v in d if v < 0),
            "same": sum(1 for v in d if v == 0), "fragility": fragility(d)}


def win_rate(v: Sequence[float], margin: Optional[float] = None) -> Dict[str, Any]:
    """B's share of each case's picks: the mean, Wilson's range on it, and what it says against one half."""
    v = [float(x) for x in v]
    n = len(v)
    if n == 0:
        return {"rate": None, "low": None, "high": None, "n": 0, "outcome": "unclear", "fragility": 0}
    low, high = wilson(sum(v), n) if n >= 2 else (None, None)
    o = outcome(None if low is None else low - 0.5, None if high is None else high - 0.5, margin)
    return {"rate": _mean(v), "low": low, "high": high, "n": n, "outcome": o, "fragility": fragility([x - 0.5 for x in v])}


def ppi(judge: Mapping[str, float], people: Mapping[str, float], bounded_: bool = True, lo: float = 0.0,
        hi: float = 1.0) -> Optional[Dict[str, Any]]:
    '''A judge's values on every case combined with people's on a random few (PPI++).

    The judge's values are corrected by how they differ from people's where
    both judged; λ, how much the judge is trusted, is fitted to the data
    (0: people alone). Two pseudo-observations keep the range honest with few
    people's judgements. Needs at least three cases people judged; people's
    values must come from a random draw.

    Parameters
    ----------
    judge, people : dict
        ``{case: value}``.
    bounded_ : bool
        Values between ``lo`` and ``hi`` (0 and 1).

    Returns
    -------
    dict or None
        ``mean``, ``low``, ``high``, ``n`` (cases both judged), ``unlabeled``
        (cases only the judge did), ``lambda``.
    '''
    labeled = sorted(k for k in people if k in judge)
    unlabeled = sorted(k for k in judge if k not in people)
    n, big = len(labeled), len(unlabeled)
    if n < 3:
        return None
    y = [float(people[k]) for k in labeled]
    f = [float(judge[k]) for k in labeled]
    fu = [float(judge[k]) for k in unlabeled]
    every = f + fu
    vf = _var(every) if len(every) >= 2 else 0.0
    if big == 0 or vf == 0:
        lam = 0.0
    else:
        my, mf = _mean(y), _mean(f)
        c = sum((p - my) * (q - mf) for p, q in zip(y, f)) / (n - 1)
        lam = min(1.0, max(0.0, c / ((1 + n / big) * vf)))
    if bounded_:
        h = (hi - lo) / 2
    else:
        seen = y + every
        h = (max(seen) - min(seen)) / 2
    r = [p - lam * q for p, q in zip(y, f)]
    mr = _mean(r)
    est = (lam * _mean(fu) if big else 0.0) + mr
    vr = (sum((x - mr) ** 2 for x in r) + 2 * h * h) / (n + 1)
    se = math.sqrt((lam * lam * vf / big if big else 0.0) + vr / n)
    q975 = _t975(n - 2)
    low, high = est - q975 * se, est + q975 * se
    if bounded_:
        low, high = max(lo, low), min(hi, high)
    return {"mean": est, "low": low, "high": high, "n": n, "unlabeled": big, "lambda": lam}


def ppi_paired(judge_a, judge_b, people_a, people_b, margin=None, bounded_: bool = True) -> Optional[Dict[str, Any]]:
    """PPI++ on the differences B - A (cases both setups have), with an outcome."""
    jd = {k: judge_b[k] - judge_a[k] for k in judge_a if k in judge_b}
    pd = {k: people_b[k] - people_a[k] for k in people_a if k in people_b}
    out = ppi(jd, pd, bounded_, -1.0, 1.0)
    if out is None:
        return None
    return {"diff": out["mean"], "low": out["low"], "high": out["high"], "n": out["n"], "unlabeled": out["unlabeled"],
            "lambda": out["lambda"], "outcome": outcome(out["low"], out["high"], margin)}


def _kappa(xs, ys) -> Optional[float]:
    cats = sorted(set(xs) | set(ys), key=str)
    n = len(xs)
    po = sum(1 for a, b in zip(xs, ys) if a == b) / n
    pe = sum((xs.count(c) / n) * (ys.count(c) / n) for c in cats)
    return None if pe == 1 else (po - pe) / (1 - pe)


def _category(v: float) -> str:
    return "second" if v > 0.5 else "first" if v < 0.5 else "tie"


def agreement_of(a: Mapping[Any, float], b: Mapping[Any, float], choice: bool = False) -> Dict[str, Any]:
    """Two sources on the targets both judged: agreement and kappa, or mean absolute difference and r."""
    keys = sorted((k for k in a if k in b), key=str)
    n = len(keys)
    if n == 0:
        return {"n": 0, "agree": None, "low": None, "high": None, "kappa": None, "mad": None, "r": None}
    xs, ys = [float(a[k]) for k in keys], [float(b[k]) for k in keys]
    if choice or all(v in (0.0, 1.0) for v in xs + ys):
        cx = [_category(v) for v in xs] if choice else xs
        cy = [_category(v) for v in ys] if choice else ys
        same = sum(1 for p, q in zip(cx, cy) if p == q)
        low, high = wilson(same, n) if n >= 2 else (None, None)
        return {"n": n, "agree": same / n, "low": low, "high": high, "kappa": _kappa(cx, cy), "mad": None, "r": None}
    mx, my = _mean(xs), _mean(ys)
    sx = math.sqrt(sum((p - mx) ** 2 for p in xs))
    sy = math.sqrt(sum((q - my) ** 2 for q in ys))
    r = None if sx == 0 or sy == 0 else sum((p - mx) * (q - my) for p, q in zip(xs, ys)) / (sx * sy)
    return {"n": n, "agree": None, "low": None, "high": None, "kappa": None,
            "mad": _mean([abs(p - q) for p, q in zip(xs, ys)]), "r": r}


def positions_of(ratings, criterion, judge_id: str) -> Dict[str, Any]:
    js = [j for j in counting(ratings, criterion) if len(j["calls"]) == 2 and j["source"] == ("judge", judge_id)]
    by_pair: Dict[frozenset, Dict[tuple, Any]] = {}
    for j in js:
        by_pair.setdefault(frozenset(j["calls"]), {})[tuple(j["calls"])] = j["value"]
    both = [p for p in sorted(by_pair, key=lambda s: sorted(s)) if len(by_pair[p]) == 2]
    same = sum(1 for p in both if len(set(by_pair[p].values())) == 1)
    decided = [j for j in js if j["value"] != "tie"]
    first = sum(1 for j in decided if j["value"] == j["calls"][0])
    cl, ch = wilson(same, len(both)) if len(both) >= 2 else (None, None)
    fl, fh = wilson(first, len(decided)) if len(decided) >= 2 else (None, None)
    return {"pairs": len(both), "consistent": same / len(both) if both else None, "consistent_low": cl,
            "consistent_high": ch, "picks": len(decided), "first_wins": first / len(decided) if decided else None,
            "first_low": fl, "first_high": fh}


def self_agreement_of(per_case: Mapping[str, Sequence[float]]) -> Dict[str, Any]:
    cases = [k for k in sorted(per_case) if len(per_case[k]) >= 2]
    if not cases:
        return {"cases": 0, "mean": None, "low": None, "high": None, "spread": None}
    agree, within = [], []
    for k in cases:
        vs = [float(v) for v in per_case[k]]
        pairs = [(vs[i], vs[j]) for i in range(len(vs)) for j in range(i + 1, len(vs))]
        agree.append(_mean([1 - abs(p - q) for p, q in pairs]))
        m = _mean(vs)
        within.append(sum((v - m) ** 2 for v in vs) / len(vs))
    mean, low, high = interval(agree)
    if low is not None:
        low, high = min(1.0, max(0.0, low)), min(1.0, max(0.0, high))
    return {"cases": len(cases), "mean": mean, "low": low, "high": high, "spread": math.sqrt(_mean(within))}


# ------------------------------------------------------------------ Bradley–Terry


def _bt_fit(k: int, outcomes: Sequence[Tuple[int, int, float]]):
    met = {}
    for i, j, _s in outcomes:
        if i != j:
            met[(min(i, j), max(i, j))] = True
    allo = list(outcomes) + [(i, j, 0.5) for (i, j) in sorted(met)]
    adj: Dict[int, set] = {i: set() for i in range(k)}
    for (i, j) in met:
        adj[i].add(j)
        adj[j].add(i)
    ref = next((i for i in range(k) if adj[i]), None)
    if ref is None:
        return [None] * k, None, [], None, None
    comp, todo = {ref}, [ref]
    while todo:
        x = todo.pop()
        for y in sorted(adj[x]):
            if y not in comp:
                comp.add(y)
                todo.append(y)
    comp_l = sorted(comp)
    wins = [0.0] * k
    count = [[0.0] * k for _ in range(k)]
    for i, j, s in allo:
        if i == j:
            continue
        wins[i] += s
        wins[j] += 1 - s
        count[i][j] += 1
        count[j][i] += 1
    p = [1.0] * k
    for _ in range(10000):
        new = list(p)
        for i in comp_l:
            denom = 0.0
            for j in comp_l:
                if j != i and count[i][j]:
                    denom += count[i][j] / (p[i] + p[j])
            new[i] = wins[i] / denom
        r = new[ref]
        new = [x / r if idx in comp else 1.0 for idx, x in enumerate(new)]
        delta = max(abs(math.log(new[i]) - math.log(p[i])) for i in comp_l)
        p = new
        if delta < 1e-12:
            break
    s = [math.log(p[i]) if i in comp else None for i in range(k)]
    return s, ref, comp_l, p, count


def _invert(m: List[List[float]]) -> List[List[float]]:
    n = len(m)
    a = [list(row) + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(m)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: (abs(a[r][c]), -r))
        a[c], a[piv] = a[piv], a[c]
        pv = a[c][c]
        a[c] = [x / pv for x in a[c]]
        for r in range(n):
            if r != c and a[r][c] != 0:
                f = a[r][c]
                a[r] = [x - f * y for x, y in zip(a[r], a[c])]
    return [row[n:] for row in a]


def _strengths(versions, outcomes):
    idx = {v: n for n, v in enumerate(versions)}
    return _bt_fit(len(versions), [(idx[i], idx[j], float(s)) for _c, i, j, s in outcomes])


def bradley_terry(versions: Sequence[Any], outcomes: Sequence[Tuple[str, Any, Any, float]]) -> Dict[str, Any]:
    '''Rank versions from side-by-side outcomes (Bradley–Terry, Hunter's MM).

    Parameters
    ----------
    versions : list
        The first that met any other is the reference (strength 0).
    outcomes : list of (case, first, second, first's share)
        One per case and pair (a share: 1 the first won, 0 lost, 0.5 a tie).

    Returns
    -------
    dict
        ``versions`` (each ``strength``, ``low``, ``high``, ``vs_reference``),
        ``order``, ``reference``, and ``fragility``: at most how many cases,
        removed, would change which version leads (None: not within 20).
    '''
    versions = list(versions)
    s, ref, comp, p, count = _strengths(versions, outcomes)
    if ref is None:
        return {"versions": [{"version": v, "strength": None, "low": None, "high": None, "vs_reference": None}
                             for v in versions], "order": [], "reference": None, "fragility": None}
    others = [i for i in comp if i != ref]
    info = [[0.0] * len(others) for _ in others]
    for a_, i in enumerate(others):
        for j in comp:
            if j == i or not count[i][j]:
                continue
            w = count[i][j] * (p[i] / (p[i] + p[j])) * (p[j] / (p[i] + p[j]))
            info[a_][a_] += w
            if j != ref:
                info[a_][others.index(j)] -= w
    inv = _invert(info) if others else []
    rows = []
    for n, v in enumerate(versions):
        if s[n] is None:
            rows.append({"version": v, "strength": None, "low": None, "high": None, "vs_reference": None})
        elif n == ref:
            rows.append({"version": v, "strength": 0.0, "low": 0.0, "high": 0.0, "vs_reference": 0.5})
        else:
            se = math.sqrt(max(0.0, inv[others.index(n)][others.index(n)]))
            rows.append({"version": v, "strength": s[n], "low": s[n] - Z * se, "high": s[n] + Z * se,
                         "vs_reference": p[n] / (p[n] + 1.0)})
    order = sorted([n for n in range(len(versions)) if s[n] is not None], key=lambda n: (-s[n], n))
    return {"versions": rows, "order": [versions[n] for n in order], "reference": versions[ref],
            "fragility": _rank_fragility(versions, list(outcomes), order)}


def _rank_fragility(versions, outcomes, order, limit: int = RANK_STEPS) -> Optional[int]:
    if len(order) < 2:
        return None
    leader = order[0]
    kept = list(outcomes)
    for step in range(1, limit + 1):
        cases = sorted({o[0] for o in kept})
        if not cases:
            return None
        best = None
        for c in cases:
            s = _strengths(versions, [o for o in kept if o[0] != c])[0]
            live = [n for n in range(len(versions)) if s[n] is not None]
            if len(live) < 2 or s[leader] is None:
                continue
            top = sorted(live, key=lambda n: (-s[n], n))[0]
            if top != leader:
                return step
            gap = s[leader] - max(s[n] for n in live if n != leader)
            if gap <= 0:
                return step
            if best is None or gap < best[0] - 1e-9:
                best = (gap, c)
        if best is None:
            return None
        kept = [o for o in kept if o[0] != best[1]]
    return None


# ------------------------------------------------------------------ scoreboard


def _trees(calls: Sequence[Mapping[str, Any]], roots: Sequence[Mapping[str, Any]]) -> Dict[str, List[Mapping[str, Any]]]:
    best: Dict[str, Mapping[str, Any]] = {}
    for r in calls:
        if r.get("functai_call") in (1, 2) and (r["id"] not in best or int(r.get("writer") or 1) > int(best[r["id"]].get("writer") or 1)):
            best[r["id"]] = r
    kids: Dict[str, List[Mapping[str, Any]]] = {}
    for r in best.values():
        if r.get("parent"):
            kids.setdefault(r["parent"], []).append(r)
    out = {}
    for c in roots:
        tree, todo = [], [c]
        while todo:
            x = todo.pop()
            tree.append(x)
            todo.extend(kids.get(x["id"], []))
        out[c["id"]] = tree
    return out


def _usage(trees: Sequence[Sequence[Mapping[str, Any]]]) -> Dict[str, Any]:
    calls = [c for t in trees for c in t]
    out: Dict[str, Any] = {}
    asked = [c for c in calls if c.get("model") is not None]
    for key in ("input_tokens", "output_tokens"):
        vs = [(c.get("usage") or {}).get(key) for c in asked]
        out[key] = sum(vs) if all(isinstance(v, int) and not isinstance(v, bool) for v in vs) else None
    out["spent_usd"] = sum(float((c.get("spent") or {}).get("usd") or 0) for c in calls)
    return out


def _estimate(per_case: Mapping[str, float], bounded_: bool) -> Dict[str, Any]:
    vs = [per_case[k] for k in sorted(per_case)]
    mean, low, high = interval(vs)
    if bounded_ and low is not None:
        low, high = max(0.0, low), min(1.0, high)
    return {"mean": mean, "low": low, "high": high, "cases": len(vs)}


def scoreboard_of(records: Sequence[Mapping[str, Any]], program: Mapping[str, Any],
                  criteria: Sequence[Mapping[str, Any]], *, by: str = "setup") -> Dict[str, Any]:
    """The contract's scoreboard (measures.md, *Scoreboard*)."""
    calls, ratings = _split_records(records)
    res = results(calls, program, by)
    groups: List[Any] = []
    for r in res:
        if r["group"] not in groups:
            groups.append(r["group"])
    tree = _trees(calls, [r["call"] for r in res])
    rows = []
    for g in groups:
        mine = [r["call"] for r in res if r["group"] == g]
        row: Dict[str, Any] = {"group": g, "results": len(mine), "cases": len({r["case"] for r in res if r["group"] == g}),
                               "failed": sum(1 for c in mine if c.get("error") is not None),
                               "seconds": _mean([float(c.get("seconds") or 0) for c in mine]),
                               **_usage([tree[c["id"]] for c in mine]), "measures": []}
        for crit in criteria:
            agg = crit.get("aggregate", "mean") if crit.get("items") is not None else None
            judges: List[Tuple[str, Any]] = []
            for j in counting(ratings, crit):
                if j["source"][0] == "judge" and j["source"][1] not in [x[0] for x in judges]:
                    judges.append((j["source"][1], j.get("judge")))
            got = {}
            for src, name, drawn in [("people", None, False), ("people", None, True)] + [(jid, nm, False) for jid, nm in judges]:
                cs = values_of(records, program, crit, src, by=by, drawn=drawn)["cases"].get(g, {})
                got[(src, drawn)] = cs
                if cs:
                    row["measures"].append({"criterion": crit["name"], "aggregate": agg,
                                            "source": "people drawn" if drawn else "people" if src == "people" else "judge",
                                            "judge": name, **_estimate(cs, bounded(crit))})
            for jid, name in judges:
                comb = ppi(got[(jid, False)], got[("people", True)], bounded(crit))
                if comb is not None:
                    row["measures"].append({"criterion": crit["name"], "aggregate": agg, "source": "combined",
                                            "judge": name, "mean": comb["mean"], "low": comb["low"],
                                            "high": comb["high"], "cases": comb["n"] + comb["unlabeled"]})
        rows.append(row)
    return {"groups": rows}


def compare_of(records, program, criterion, source, a, b, *, by="setup", drawn=False, margin=None) -> Dict[str, Any]:
    cs = values_of(records, program, criterion, source, by=by, drawn=drawn)["cases"]
    ca, cb = cs.get(a, {}), cs.get(b, {})
    shared = sorted(k for k in ca if k in cb)
    out = paired([ca[k] for k in shared], [cb[k] for k in shared], margin, bounded(criterion))
    out["only_a"] = sum(1 for k in ca if k not in cb)
    out["only_b"] = sum(1 for k in cb if k not in ca)
    return out


def disagreements_of(records, program, criterion, source, *, by="setup") -> List[Dict[str, Any]]:
    v = values_of(records, program, criterion, source, by=by)
    cases = v["cases"]
    keys = sorted({k for cs in cases.values() for k in cs})
    out = []
    for k in keys:
        per = [{"group": g, "value": cs[k]} for g, cs in cases.items() if k in cs]
        vs = [x["value"] for x in per]
        if len(per) >= 2 and max(vs) - min(vs) > 0:
            out.append({"case": k, "kind": "groups", "spread": max(vs) - min(vs), "values": per})
    rep: Dict[Tuple[Any, str], List[float]] = {}
    for r in v["results"]:
        if r["value"] is not None:
            rep.setdefault((r["group"], r["case"]), []).append(r["value"])
    for (g, k), vs in sorted(rep.items(), key=lambda x: ("" if x[0][0] is None else str(x[0][0]), x[0][1])):
        if len(vs) >= 2 and max(vs) - min(vs) > 0:
            out.append({"case": k, "kind": "repeats", "spread": max(vs) - min(vs),
                        "values": [{"group": g, "value": x} for x in vs]})
    return sorted(out, key=lambda x: (x["kind"] != "groups", -x["spread"], x["case"],
                                      "" if x["kind"] == "groups" or x["values"][0]["group"] is None
                                      else str(x["values"][0]["group"])))


# ------------------------------------------------------------------ the public functions over a log


def _records(folder: Any = None, records: Any = None, since: Any = None) -> List[Dict[str, Any]]:
    """Call and rating records: given, or read from a folder or store (the call log by default)."""
    if records is not None:
        if hasattr(records, "read") and callable(records.read):
            return [dict(r) for r in records.read() if isinstance(r, Mapping)]
        return [dict(r) for r in records]
    from .calllog import read
    calls, ratings = read(folder, since=since)
    return calls + ratings


def _program_of(program: Any) -> Dict[str, Any]:
    if isinstance(program, Mapping):
        return {"name": program["name"], "module": program.get("module"), "file": program.get("file")}
    if isinstance(program, str):
        return {"name": program}
    from .calllog import _program_key
    name, module, _sig, _iface = _program_key(program)
    return {"name": name, "module": module}


def source_id(source: Any) -> str:
    """``"people"``, a judge's id (its setup, or its code's hash), or that of a program or function given."""
    if source is None or source == "people":
        return "people"
    if isinstance(source, str):
        return source
    if isinstance(source, Criterion):
        source = source.judge
    from .core import FunctAIFunc
    from .module import FunctAIModule
    if isinstance(source, (FunctAIFunc, FunctAIModule)):
        return source.setup
    if callable(source):
        from .calllog import code_hash
        return code_hash(source)
    raise TypeError(f"a source is 'people', a judge (a program or a function), or a judge's id; not {source!r}")


def _group_id(g: Any, by: str) -> Any:
    from .core import FunctAIFunc
    from .module import FunctAIModule
    if isinstance(g, (FunctAIFunc, FunctAIModule)):
        return g.setup if by == "setup" else g.version
    return g


class Scoreboard:
    """Every group of a program's results (its setups, or versions), and each
    criterion's estimate by each source. ``rows`` is one dict per (group,
    criterion, source); ``groups`` the contract's form; ``table`` a dpyr
    dataframe (needs dpyr)."""

    def __init__(self, program: str, data: Dict[str, Any], by: str):
        self.program, self.by = program, by
        self.groups: List[Dict[str, Any]] = data["groups"]

    @property
    def rows(self) -> List[Dict[str, Any]]:
        out = []
        for g in self.groups:
            base = {k: v for k, v in g.items() if k != "measures"}
            if not g["measures"]:
                out.append({**base, "criterion": None, "aggregate": None, "source": None, "judge": None,
                            "mean": None, "low": None, "high": None, "measured_cases": None})
            for m in g["measures"]:
                out.append({**base, "criterion": m["criterion"], "aggregate": m["aggregate"], "source": m["source"],
                            "judge": m["judge"], "mean": m["mean"], "low": m["low"], "high": m["high"],
                            "measured_cases": m["cases"]})
        return out

    @property
    def table(self):
        from .evaluation import _dpyr
        return _dpyr().read(self.rows)

    def __len__(self) -> int:
        return len(self.groups)

    def __repr__(self) -> str:
        def f(x):
            return "—" if x is None else f"{x:.2f}"
        lines = [f"Scoreboard({self.program}, {len(self.groups)} {self.by}{'s' if len(self.groups) != 1 else ''})"]
        for g in self.groups:
            name = str(g["group"])[:19] if g["group"] else "(none)"
            tokens = "" if g["input_tokens"] is None else f", {g['input_tokens'] + (g['output_tokens'] or 0):,} tokens"
            lines.append(f"  {name}: {g['results']} results on {g['cases']} cases, {g['failed']} failed, "
                         f"{g['seconds']:.1f}s each{tokens}")
            for m in g["measures"]:
                who = m["source"] + (f" ({m['judge']})" if m["judge"] else "")
                agg = f" [{m['aggregate']}]" if m["aggregate"] else ""
                lines.append(f"      {m['criterion']}{agg} by {who}: {f(m['mean'])} "
                             f"(95% range {f(m['low'])} to {f(m['high'])}, {m['cases']} cases)")
        return "\n".join(lines)


def scoreboard(program: Any, criteria: Sequence[Any] = (correct,), *, by: str = "setup", folder: Any = None,
               records: Any = None, since: Any = None) -> Scoreboard:
    '''How good every setup of a program is, by every source of judgement.

    Reads the call log (or ``records``): the program's results (calls on a
    case), and ratings of them. For each setup (or version), in order of its
    first result: how many results and cases, failures, time, tokens of
    whole trees (a pipeline's steps included) and what code spent; then each
    criterion's estimate with its 95% range over cases, by people (all their
    ratings), by people drawn (a random draw's: the fair one), by each judge,
    and each judge combined with people drawn (prediction-powered inference).

    Parameters
    ----------
    program : AI function, module, or name
    criteria : list of Criterion
        Single-answer criteria (a choice is between setups: ``win_rate``, ``rank``).
    by : "setup" or "version"
    folder, records, since
        Where the records are (the call log by default).

    Returns
    -------
    Scoreboard

    See Also
    --------
    compare_setups : two setups, case by case.
    agreement : whether a judge can stand in for people.
    '''
    crits = [criterion_of(c) for c in criteria]
    if any(c["kind"] == "choice" for c in crits):
        raise ValueError("a choice is between two setups: use functai.win_rate or functai.rank for it")
    prog = _program_of(program)
    data = scoreboard_of(_records(folder, records, since), prog, crits, by=by)
    return Scoreboard(prog["name"], data, by)


def compare_setups(program: Any, a: Any, b: Any, criterion: Any = correct, *, source: Any = "people",
                   drawn: bool = False, margin: Optional[float] = None, by: str = "setup", folder: Any = None,
                   records: Any = None, since: Any = None) -> Dict[str, Any]:
    '''Is setup ``b`` better than ``a``, on the cases both answered.

    Parameters
    ----------
    program : AI function, module, or name
    a, b : setup ids (or versions with ``by="version"``), or programs (their setup)
    criterion : Criterion
    source : "people", a judge (program or function), or a judge's id
    drawn : bool
        Only people's ratings of a random draw.
    margin : float, optional
        How much worse is still acceptable.

    Returns
    -------
    dict
        ``diff`` (b - a), ``low``, ``high``, ``method``, ``outcome``, ``n``,
        ``better``, ``worse``, ``same``, ``fragility``, ``only_a``, ``only_b``.
    '''
    crit = criterion_of(criterion)
    return compare_of(_records(folder, records, since), _program_of(program), crit, source_id(source),
                      _group_id(a, by), _group_id(b, by), by=by, drawn=drawn, margin=margin)


def win_rate_of_setups(program: Any, a: Any, b: Any, criterion: Any = preferred, *, source: Any = "people",
                       drawn: bool = False, margin: Optional[float] = None, by: str = "setup", folder: Any = None,
                       records: Any = None, since: Any = None) -> Dict[str, Any]:
    '''How often setup ``b`` wins side by side against ``a`` (ties count half), with its 95% range.

    A failed answer loses to any answer. Returns ``rate``, ``low``, ``high``,
    ``n`` (cases), ``outcome`` (against one half), ``fragility``.
    '''
    crit = criterion_of(criterion)
    per = preferences_of(_records(folder, records, since), _program_of(program), crit, source_id(source),
                         _group_id(a, by), _group_id(b, by), by=by, drawn=drawn)
    return win_rate(list(per.values()), margin)


def rank(program: Any, criterion: Any = preferred, *, source: Any = "people", drawn: bool = False, by: str = "setup",
         groups: Optional[Sequence[Any]] = None, folder: Any = None, records: Any = None, since: Any = None
         ) -> Dict[str, Any]:
    '''Rank every setup of a program from side-by-side picks (Bradley–Terry).

    Each case and pair of setups that met is one outcome (the later setup's
    share of the picks; a failed answer loses). Returns each setup's
    strength (the first's 0) with its range, the order, and ``fragility``: at
    most how many cases would change which setup leads.
    '''
    recs = _records(folder, records, since)
    prog, crit, src = _program_of(program), criterion_of(criterion), source_id(source)
    calls, _r = _split_records(recs)
    if groups is None:
        groups_l: List[Any] = []
        for r in results(calls, prog, by):
            if r["group"] not in groups_l:
                groups_l.append(r["group"])
    else:
        groups_l = [_group_id(g, by) for g in groups]
    outcomes = []
    for i, ga in enumerate(groups_l):
        for gb in groups_l[i + 1:]:
            for case, share_b in preferences_of(recs, prog, crit, src, ga, gb, by=by, drawn=drawn).items():
                outcomes.append((case, ga, gb, 1.0 - share_b))
    return bradley_terry(groups_l, outcomes)


def agreement(program: Any, criterion: Any, a: Any = "people", b: Any = None, *, drawn: bool = False,
              folder: Any = None, records: Any = None, since: Any = None) -> Dict[str, Any]:
    '''Can a judge stand in for people: two sources on the targets both judged.

    Right-or-wrong values and labels: ``agree`` (with its range) and Cohen's
    ``kappa``; scores: ``mad`` (mean absolute difference) and ``r``; picks:
    the same winner. Items of one answer are counted as targets of their own:
    this describes the judge, it is not a test.
    '''
    if b is None:
        raise TypeError("agreement of which two sources? a='people', b=the judge")
    recs = _records(folder, records, since)
    crit, prog = criterion_of(criterion), _program_of(program)
    calls, ratings = _split_records(recs)
    mine = {c["id"] for c in program_calls(calls, prog)}

    def values(src: str, only_drawn: bool) -> Dict[Any, float]:
        per: Dict[Any, List[float]] = {}
        for j in counting(ratings, crit):
            if not _matches(j, src, only_drawn) or not set(j["calls"]) <= mine:
                continue
            if crit["kind"] == "choice":
                pair = tuple(sorted(j["calls"]))
                x = 1.0 if j["value"] == pair[1] else 0.0 if j["value"] == pair[0] else 0.5
                per.setdefault((pair, ""), []).append(x)
            else:
                x = number_of(j)
                if x is not None:
                    per.setdefault((tuple(j["calls"]), j["item"]), []).append(x)
        return {k: _mean(v) for k, v in per.items()}

    sa, sb = source_id(a), source_id(b)
    return agreement_of(values(sa, drawn and sa == "people"), values(sb, drawn and sb == "people"),
                        crit["kind"] == "choice")


def positions(judge: Any, criterion: Any = preferred, *, folder: Any = None, records: Any = None,
              since: Any = None) -> Dict[str, Any]:
    '''Whether a judge favours a position: of the pairs it was shown both ways,
    how often it picked the same answer both times (``consistent``), and how
    often the first shown won (``first_wins``), each with its range.'''
    _c, ratings = _split_records(_records(folder, records, since))
    return positions_of(ratings, criterion_of(criterion), source_id(judge))


def self_agreement(program: Any, criterion: Any = correct, *, source: Any = "people", by: str = "setup",
                   folder: Any = None, records: Any = None, since: Any = None) -> Dict[Any, Dict[str, Any]]:
    '''How much each setup agrees with itself, over cases it answered two or
    more times: the mean pairwise agreement with its range, and ``spread``
    (the typical difference between repeats). ``{group: {...}}``.'''
    v = values_of(_records(folder, records, since), _program_of(program), criterion_of(criterion),
                  source_id(source), by=by)
    per: Dict[Any, Dict[str, List[float]]] = {}
    for r in v["results"]:
        if r["value"] is not None:
            per.setdefault(r["group"], {}).setdefault(r["case"], []).append(r["value"])
    return {g: self_agreement_of(cs) for g, cs in per.items()}


def disagreements(program: Any, criterion: Any = correct, *, source: Any = "people", by: str = "setup",
                  folder: Any = None, records: Any = None, since: Any = None) -> List[Dict[str, Any]]:
    '''Cases worth a look, or worth adding to the examples: those on which
    setups' values differ, then those on which one setup's repeats differ,
    the largest spread first. Each ``{"case", "kind", "spread", "values"}``.'''
    return disagreements_of(_records(folder, records, since), _program_of(program), criterion_of(criterion),
                            source_id(source), by=by)


def measure(program: Any, criterion: Any = correct, *, source: Any = "people", drawn: bool = False, by: str = "setup",
            folder: Any = None, records: Any = None, since: Any = None) -> Dict[str, Any]:
    '''Every result's value under a criterion and source, and each group's
    value by case: ``{"results": [...], "cases": {group: {case: value}}}``.'''
    return values_of(_records(folder, records, since), _program_of(program), criterion_of(criterion),
                     source_id(source), by=by, drawn=drawn)


# ------------------------------------------------------------------ judging


def judge_identity(judge: Any) -> Dict[str, Any]:
    """A rating's ``judge``: a program by its setup (and version), code by its hash."""
    from .core import FunctAIFunc
    from .module import FunctAIModule
    if isinstance(judge, (FunctAIFunc, FunctAIModule)):
        return {"name": judge.__name__, "setup": judge.setup, "version": judge.version}
    if callable(judge):
        from .calllog import code_hash
        return {"name": getattr(judge, "__name__", "judge"), "code": code_hash(judge)}
    raise TypeError(f"a judge is an AI function, a module or a function, not {type(judge).__name__}")


def _judge_inputs(judge: Any) -> Optional[List[str]]:
    from .core import FunctAIFunc
    from .module import FunctAIModule
    if isinstance(judge, (FunctAIFunc, FunctAIModule)):
        return _input_names(judge)
    try:
        params = inspect.signature(judge).parameters.values()
    except (TypeError, ValueError):
        return None
    if any(p.kind == p.VAR_KEYWORD for p in params):
        return None
    return [p.name for p in params if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)]


def ask_judge(judge: Any, available: Mapping[str, Any]) -> Tuple[Any, Optional[str]]:
    """Call a judge with what it asks for, by name; (its answer, its call id)."""
    names = _judge_inputs(judge)
    from .core import FunctAIFunc
    if names is None:
        kwargs = dict(available)
    else:
        missing = [n for n in names if n not in available]
        if missing and isinstance(judge, FunctAIFunc):
            req = {n for n, required in judge._named_inputs() if required}
            missing = [n for n in missing if n in req]
        elif missing:
            sig = inspect.signature(judge)
            missing = [n for n in missing if sig.parameters[n].default is inspect.Parameter.empty]
        if missing:
            raise TypeError(f"judge {getattr(judge, '__name__', judge)} asks for {missing}; a judge is given by name: "
                            f"{', '.join(sorted(available))}")
        kwargs = {n: available[n] for n in names if n in available}
    if isinstance(judge, FunctAIFunc):
        pred = judge.predict(**kwargs)
        return pred[judge._spec().main], getattr(pred, "call_id", None)
    from .calllog import capturing
    with capturing() as seen:
        out = judge(**kwargs)
    return out, seen.get("call")


def judged_value(criterion: Mapping[str, Any], out: Any) -> Any:
    """A judge's answer as a rating's value for the criterion's kind."""
    v = out
    if not isinstance(v, (str, int, float, bool)) and v is not None:
        from .evaluation import _cell
        data = _cell(v)
        if isinstance(data, Mapping):
            if "value" in data:
                v = data["value"]
            elif criterion["name"] in data:
                v = data[criterion["name"]]
            else:
                raise ValueError(f"the judge's answer for {criterion['name']} is a record without 'value' or "
                                 f"{criterion['name']!r}")
    k = criterion["kind"]
    if k == "verdict":
        if v in (True, "right"):
            return "right"
        if v in (False, "wrong"):
            return "wrong"
        raise ValueError(f"a verdict judge answers True, False, 'right' or 'wrong'; not {v!r}")
    if k == "label":
        if isinstance(v, str) and v in criterion["answers"]:
            return v
        raise ValueError(f"a label judge answers one of {list(criterion['answers'])}; not {v!r}")
    if k == "score":
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"a score judge answers a number; not {v!r}")
        if criterion.get("scale") and not (criterion["scale"][0] <= v <= criterion["scale"][1]):
            raise ValueError(f"a score of {v} is outside the scale {criterion['scale']}")
        return v
    if v in ("first", "second", "tie"):
        return v
    raise ValueError(f"a choice judge answers 'first', 'second' or 'tie'; not {v!r}")


def rating_record(target: Any, criterion: Mapping[str, Any], value: Any, *, who: Mapping[str, Any],
                  item: Optional[str] = None, blind: Optional[bool] = None, sample: Optional[str] = None,
                  note: Optional[str] = None, reasons: Iterable[str] = (), origin: Optional[str] = None,
                  answer: Any = None, outputs: Optional[Mapping[str, Any]] = None, judge_call: Optional[str] = None,
                  corrected: bool = False) -> Dict[str, Any]:
    """A rating record: format 1 when format 1 can say it (a person's or an account's verdict on a whole call
    under ``correct``), else format 2."""
    from .calllog import RATING_FORMAT, _iso, new_id, to_json
    import time as _time
    pair = isinstance(target, (list, tuple))
    one_format = (criterion["name"] == "correct" and criterion["kind"] == "verdict" and item is None and not pair
                  and "judge" not in who and judge_call is None)
    if one_format:
        rec: Dict[str, Any] = {"functai_rating": RATING_FORMAT, "id": new_id(), "call": target,
                               "at": _iso(_time.time()), **who, "verdict": value}
    else:
        rec = {"functai_rating": 2, "id": new_id(), "at": _iso(_time.time())}
        if pair:
            rec["calls"] = [str(target[0]), str(target[1])]
        else:
            rec["call"] = str(target)
        if item is not None:
            rec["item"] = item
        rec["criterion"] = criterion["name"]
        if criterion.get("question"):
            rec["question"] = criterion["question"]
        rec["kind"] = criterion["kind"]
        if criterion["kind"] == "label":
            rec["answers"] = dict(criterion["answers"])
        if criterion["kind"] == "score" and criterion.get("scale"):
            rec["scale"] = list(criterion["scale"])
        rec["value"] = value
        rec.update(who)
        if judge_call:
            rec["judge_call"] = judge_call
    if corrected:
        if answer is not _NONE:
            rec["answer"] = to_json(answer)[0]
        if outputs:
            rec["outputs"] = {str(k): to_json(v)[0] for k, v in outputs.items()}
    if reasons:
        rec["reasons"] = [str(r) for r in reasons]
    if note:
        rec["note"] = str(note)
    if origin:
        rec["origin"] = str(origin)
    if sample:
        rec["sample"] = str(sample)
    if blind is not None:
        rec["blind"] = bool(blind)
    return rec


_NONE = object()


def _item_values(outputs: Any, items: str) -> List[Any]:
    lst = pointer(outputs, items)
    return lst if isinstance(lst, list) else []


def judge(judge_: Any, calls: Any, criterion: Any, *, folder: Any = None, records: Any = None,
          sample: Optional[str] = None, expected: Optional[Mapping[str, Any]] = None, threads: int = 1
          ) -> List[Dict[str, Any]]:
    '''Have a judge rate logged calls, and keep its ratings in the log.

    Each rating names the judge by what it runs on (a program's setup, a
    function's code), so a changed judge is another judge, and the
    scoreboard tells people's ratings from it. An item criterion has every
    item judged.

    Parameters
    ----------
    judge_ : AI function, module or function
        Given by name what it asks for: the call's inputs, ``answer``,
        ``outputs``, ``inputs``, ``item``, ``expected``.
    calls : predictions, call ids, or call records
    criterion : Criterion
    folder, records : where the calls are, and where ratings go (the call log by default)
    sample : str, optional
        The random draw these calls are (``draw``).
    threads : int

    Returns
    -------
    list of dict
        The ratings written.
    '''
    crit = criterion_of(criterion)
    if crit["kind"] == "choice":
        raise ValueError("a choice is between two answers: judge_pairs")
    recs = _call_records(calls, folder, records)
    who = {"judge": judge_identity(judge_)}
    from .evaluation import parallel

    def one(call: Mapping[str, Any]) -> List[Dict[str, Any]]:
        if call.get("error") is not None or call.get("outputs") is None:
            return []                                     # a failed call is the worst answer without a judge
        outputs = call["outputs"]
        answer_name = (call.get("program") or {}).get("answer") or "result"
        base = {**(call.get("inputs") or {}), "inputs": call.get("inputs") or {}, "outputs": outputs,
                "answer": outputs.get(answer_name), "expected": (expected or {}).get(call["id"])}
        targets = [(None, None)] if crit.get("items") is None else [
            (f"{crit['items']}/{i}", it) for i, it in enumerate(_item_values(outputs, crit["items"]))]
        out = []
        for item, it in targets:
            got, jc = ask_judge(judge_, {**base, "item": it} if item is not None else base)
            out.append(rating_record(call["id"], crit, judged_value(crit, got), who=who, item=item, sample=sample,
                                     origin="judge", judge_call=jc))
        return out

    written = [r for rs in parallel(one, recs, max(1, threads)) for r in rs]
    _keep(written, folder, records)
    return written


def judge_pairs(judge_: Any, pairs: Sequence[Tuple[Any, Any]], criterion: Any = preferred, *, both_orders: bool = True,
                folder: Any = None, records: Any = None, sample: Optional[str] = None, threads: int = 1
                ) -> List[Dict[str, Any]]:
    '''Have a judge pick the better of two answers to the same case, in both orders.

    Judges favour a position; asked both ways, a judge's preference for the
    first (or second) shown cancels out, and ``positions`` measures it. The
    judge is given ``first`` and ``second`` (the answers), the case's inputs,
    and ``first_outputs``, ``second_outputs``; it answers ``"first"``,
    ``"second"`` or ``"tie"``.

    Returns
    -------
    list of dict
        The ratings written (``calls`` in the order shown).
    '''
    crit = criterion_of(criterion)
    if crit["kind"] != "choice":
        raise ValueError(f"judge_pairs judges a choice; {crit['name']} is a {crit['kind']}")
    flat = _call_records([c for p in pairs for c in p], folder, records)
    by_id = {c["id"]: c for c in flat}
    ids = [(_id_of(a), _id_of(b)) for a, b in pairs]
    who = {"judge": judge_identity(judge_)}
    jobs = [(x, y) for x, y in ids] + ([(y, x) for x, y in ids] if both_orders else [])
    from .evaluation import parallel

    def one(job: Tuple[str, str]) -> Optional[Dict[str, Any]]:
        first, second = by_id[job[0]], by_id[job[1]]
        if first.get("error") is not None or second.get("error") is not None:
            return None                                   # a failed answer loses without a judge
        name = (first.get("program") or {}).get("answer") or "result"
        avail = {**(first.get("inputs") or {}), "inputs": first.get("inputs") or {},
                 "first": (first.get("outputs") or {}).get(name), "second": (second.get("outputs") or {}).get(name),
                 "first_outputs": first.get("outputs"), "second_outputs": second.get("outputs")}
        got, jc = ask_judge(judge_, avail)
        v = judged_value(crit, got)
        value = job[0] if v == "first" else job[1] if v == "second" else "tie"
        return rating_record([job[0], job[1]], crit, value, who=who, sample=sample, origin="judge", judge_call=jc)

    written = [r for r in parallel(one, jobs, max(1, threads)) if r is not None]
    _keep(written, folder, records)
    return written


def _id_of(c: Any) -> str:
    from .data import Prediction
    if isinstance(c, Prediction):
        if getattr(c, "call_id", None) is None:
            raise ValueError("this prediction has no call id (it was not made by calling a program)")
        return str(c.call_id)
    if isinstance(c, Mapping):
        return str(c.get("id") or c.get("call"))
    return str(c)


def _call_records(calls: Any, folder: Any, records: Any) -> List[Dict[str, Any]]:
    wanted = [c for c in (calls if isinstance(calls, (list, tuple)) else [calls])]
    if all(isinstance(c, Mapping) and "functai_call" in c for c in wanted):
        return [dict(c) for c in wanted]
    known = {r["id"]: r for r in _records(folder, records) if "functai_call" in r}
    out = []
    for c in wanted:
        if isinstance(c, Mapping) and "functai_call" in c:
            out.append(dict(c))
            continue
        cid = _id_of(c)
        if cid not in known:
            raise ValueError(f"no logged call {cid}: is the call log on (configure(log_calls=...)), and flushed?")
        out.append(known[cid])
    return out


def _keep(ratings: Sequence[Mapping[str, Any]], folder: Any, records: Any) -> None:
    """Write ratings where the calls are: a store given as records, a folder, or the call log."""
    if not ratings:
        return
    from .calllog import write_rating
    target = records if records is not None and hasattr(records, "append") and callable(records.append) else folder
    if isinstance(records, list):
        records.extend(dict(r) for r in ratings)
        return
    for r in ratings:
        write_rating(r, target)


def rate_item(call: Any, value: Any, *, criterion: Any, item: Optional[Any] = None, by: Optional[str] = None,
              blind: Optional[bool] = None, sample: Optional[str] = None, note: Optional[str] = None,
              reasons: Iterable[str] = (), origin: Optional[str] = None, folder: Any = None,
              answer: Any = _NONE, outputs: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """``functai.rate`` with a criterion: a person's judgement (format 2 unless format 1 can say it)."""
    from .calllog import _call_id, caller_of, _process_info
    from .config import effective
    crit = criterion_of(criterion)
    if crit["kind"] == "choice":
        raise ValueError("a choice is between two answers: functai.prefer")
    if isinstance(item, int):
        if crit.get("items") is None:
            raise ValueError(f"{crit['name']} judges whole answers; item {item} of what?")
        item = f"{crit['items']}/{item}"
    elif item is not None and not (isinstance(item, str) and item.startswith("/")):
        raise ValueError("item is the item's index in the criterion's list, or a JSON Pointer ('/insights/3')")
    if item is None and crit.get("items") is not None:
        raise ValueError(f"{crit['name']} judges each item of {crit['items']}: say which (item=3)")
    if value is not None:
        value = judged_value(crit, value)
    corrected = answer is not _NONE or bool(outputs)
    if corrected and value != "wrong":
        raise ValueError("a correction goes with a verdict of 'wrong'")
    settings = effective()
    who_by = by or caller_of(settings).get("user")
    who = {"by": str(who_by)} if who_by else {"account": str(_process_info().get("user") or "unknown")}
    rec = rating_record(_call_id(call), crit, value, who=who, item=item, blind=blind, sample=sample, note=note,
                        reasons=reasons, origin=origin, answer=answer, outputs=outputs, corrected=corrected)
    from .calllog import write_rating
    write_rating(rec, folder)
    return rec


def prefer(a: Any, b: Any, winner: Any, *, criterion: Any = preferred, by: Optional[str] = None,
           blind: Optional[bool] = None, sample: Optional[str] = None, note: Optional[str] = None,
           shown: Optional[Tuple[Any, Any]] = None, folder: Any = None) -> Dict[str, Any]:
    '''Say which of two answers to the same case is better.

    Parameters
    ----------
    a, b : predictions, call ids or call records
        In the order they were shown (or give ``shown``).
    winner : one of them, ``"tie"``, or None (withdraw)
    blind : bool
        You were not told which version made which (show them in a random
        order: ``functai.shown_order``).
    sample : str, optional
        The random draw this pair is part of.

    Returns
    -------
    dict
        The rating written.
    '''
    from .calllog import caller_of, _process_info, write_rating
    from .config import effective
    crit = criterion_of(criterion)
    if crit["kind"] != "choice":
        raise ValueError(f"{crit['name']} is not a choice")
    ia, ib = _id_of(a), _id_of(b)
    if ia == ib:
        raise ValueError("two answers are compared, not one with itself")
    if shown is not None:
        ia, ib = _id_of(shown[0]), _id_of(shown[1])
    if winner is None or winner == "tie":
        value = winner
    else:
        value = _id_of(winner)
        if value not in (ia, ib):
            raise ValueError("the winner is one of the two answers, 'tie', or None to withdraw")
    settings = effective()
    who_by = by or caller_of(settings).get("user")
    who = {"by": str(who_by)} if who_by else {"account": str(_process_info().get("user") or "unknown")}
    rec = rating_record([ia, ib], crit, value, who=who, blind=blind, sample=sample, note=note)
    write_rating(rec, folder)
    return rec


def shown_order(a: Any, b: Any, seed: str = "") -> Tuple[str, str]:
    """Two answers in the order to show them blind: by the hash of the seed and the pair, the same everywhere."""
    ia, ib = sorted((_id_of(a), _id_of(b)))
    flip = int(_hex(f"functai-shown:{seed}:{ia}:{ib}")[:2], 16) % 2 == 1
    return (ib, ia) if flip else (ia, ib)


def as_case(call: Any, *, rating: Optional[Mapping[str, Any]] = None, share: float = 0.2, salt: str = "",
            folder: Any = None, records: Any = None) -> Dict[str, Any]:
    '''A logged call as a row to add to a program's examples.

    The call's inputs; its answer when it was right, or the correction of a
    ``wrong`` rating (given, or the latest one of the call's); and its
    ``split``, held out by the rule every language shares (``split_of``).

    Returns
    -------
    dict
        A row: inputs, the answer when known, ``split``.
    '''
    rec = _call_records([call], folder, records)[0]
    if rec.get("content") is not True or rec.get("saw"):
        raise ValueError("only a call whose inputs were all recorded, and that saw no earlier calls, makes a case")
    row = dict(rec.get("inputs") or {})
    answer = (rec.get("program") or {}).get("answer") or "result"
    r = rating
    if r is None:
        mine = [x for x in _records(folder, records) if "functai_rating" in x and x.get("call") == rec["id"]]
        js = [x for x in mine if (judgement(x) or {}).get("criterion") == "correct" and not x.get("judge")]
        r = max(js, key=lambda x: (x["at"], x["id"])) if js else None
    if r is not None:
        v = r.get("verdict", r.get("value"))
        if v == "right" and rec.get("outputs") and answer in rec["outputs"]:
            row[answer] = rec["outputs"][answer]
        elif v == "wrong" and "answer" in r:
            row[answer] = r["answer"]
        for k, val in (r.get("outputs") or {}).items():
            row.setdefault(k, val)
    names = list((rec.get("inputs") or {}).keys())
    if "split" in names:
        raise ValueError("the program has an input named split: its rows cannot also say their side")
    row["split"] = split_of(row, names, share=share, salt=salt)
    return row


__all__ = ["Criterion", "correct", "preferred", "case_key", "split_of", "draw", "scoreboard", "compare_setups",
           "win_rate_of_setups", "rank", "agreement", "positions", "self_agreement", "disagreements", "measure",
           "judge", "judge_pairs", "prefer", "shown_order", "as_case", "paired", "win_rate", "ppi", "ppi_paired",
           "bradley_terry", "fragility", "outcome", "Scoreboard", "MeasureError"]
