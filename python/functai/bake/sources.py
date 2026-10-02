"""Where training examples come from: rows of data, a teacher's answers, and
the calls a whole program makes while a teacher runs it.

Every source gives ``Item``s: one function call (its entry, its inputs, its
outputs when known), with a tag, a weight and where it came from. Rows
without outputs are answered by the teacher (``label``); what that costs is
estimated before (``teacher_estimate``) and counted after.
"""

from __future__ import annotations

import dataclasses
import random
import statistics
import threading
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .examples import BakeError
from .functions import Entry


@dataclasses.dataclass
class Item:
    entry: Entry
    inputs: Dict[str, Any]
    outputs: Optional[Dict[str, Any]]
    tag: Optional[str] = None
    weight: float = 1.0
    row_id: str = ""
    source: str = "data"             # "data" | "teacher" | "program"


def _outputs(row: Mapping[str, Any], names: Sequence[str]) -> Optional[Dict[str, Any]]:
    if any(row.get(n) is None for n in names):
        return None
    return {n: row[n] for n in names}


def _weight(row: Mapping[str, Any], column: Optional[str]) -> float:
    if not column:
        return 1.0
    w = row.get(column, 1.0)
    try:
        w = float(1.0 if w is None else w)
    except (TypeError, ValueError):
        raise BakeError(f"weight column {column!r}: {w!r} is not a number") from None
    if w < 0:
        raise BakeError(f"weight column {column!r}: weights are 0 or more, not {w}")
    return w


def from_rows(entry: Entry, rows: Sequence[Mapping[str, Any]], *, tags: Optional[str] = None,
              weights: Optional[str] = None, labels: str = "auto") -> List[Item]:
    """Items for one function from rows: inputs under the parameters' names,
    outputs under the outputs' names (``result`` for the return value)."""
    out = []
    names = [n for n in entry.outputs if n != "reasoning" or entry.reasoning]
    for i, row in enumerate(rows):
        inputs = entry.row_inputs(row)
        outputs = None if labels == "teacher" else _outputs(row, names)
        rid = row.get("id", row.get("row_id"))
        out.append(Item(entry, inputs, outputs, tag=None if not tags else
                        (None if row.get(tags) is None else str(row.get(tags))),
                        weight=_weight(row, weights), row_id=f"{entry.name}:{rid if rid is not None else i}"))
    return out


# ------------------------------------------------------------------ the teacher


def teacher_for(fn, teacher: Any):
    """The function that answers for the student: ``teacher`` (a model, an AI
    function with the same inputs, or ``{function: teacher}``), else the
    function itself on its own model."""
    from ..core import FunctAIFunc
    if teacher is None:
        return fn
    if isinstance(teacher, FunctAIFunc):
        return teacher
    if isinstance(teacher, Mapping):
        t = teacher.get(fn, teacher.get(fn.__name__))
        return teacher_for(fn, t) if t is not None else fn
    return fn.using(lm=teacher)


def teacher_name(fn) -> str:
    lm = fn._effective().get("lm")
    if lm is None:
        return "the configured model"
    from .. import models
    return lm if isinstance(lm, str) else getattr(lm, "name", None) or models.model_string(lm)


def _price(fn) -> Optional[Tuple[float, float]]:
    from .. import models
    from .prices import model_price
    s = fn._effective()
    try:
        _router, model, route = models.resolve(s)
    except Exception:  # noqa: BLE001 — no login yet: the price is unknown
        lm = s.get("lm")
        return model_price(None, lm) if isinstance(lm, str) else None
    return model_price(route.provider, route.model)


def _chars_to_tokens(n: float) -> float:
    return n / 3.6           # English prose and markup in modern tokenizers: about 3.6 characters a token


def teacher_estimate(items: Sequence[Item], teacher: Any, *, answer_tokens: Optional[float] = None
                     ) -> Dict[str, Any]:
    """What answering ``items`` with the teacher will cost, before any call:
    input tokens from the requests it will send (characters / 3.6), output
    tokens from ``answer_tokens`` (the rows that have answers) when known."""
    by_fn: Dict[int, List[Item]] = {}
    for it in items:
        by_fn.setdefault(id(it.entry), []).append(it)
    total_in = total_out = 0.0
    dollars: Optional[float] = 0.0
    names = []
    for group in by_fn.values():
        tfn = teacher_for(group[0].entry.fn, teacher)
        names.append(teacher_name(tfn))
        sample = group if len(group) <= 50 else random.Random(0).sample(group, 50)
        chars = []
        for it in sample:
            try:
                req = tfn.render(**it.inputs)
                chars.append(len(req.system or "") + sum(len(getattr(p, "text", "") or "")
                                                            for m in req.messages for p in m.parts))
            except Exception:  # noqa: BLE001 — an input the renderer refuses fails later, counted
                continue
        per_in = _chars_to_tokens(statistics.mean(chars)) if chars else 0.0
        n_in = per_in * len(group)
        n_out = answer_tokens * len(group) if answer_tokens is not None else None
        total_in += n_in
        total_out = None if n_out is None or total_out is None else total_out + n_out
        price = _price(tfn)
        if price is None or dollars is None:
            dollars = None
        else:
            dollars += n_in * price[0] / 1e6 + (n_out or 0) * price[1] / 1e6
    return {"rows": len(items), "teacher": ", ".join(sorted(set(names))), "input_tokens": total_in,
            "output_tokens": total_out, "dollars": dollars, "output_known": answer_tokens is not None}


def _answer_key(it: Item, teacher_name: str) -> str:
    from .functions import value_hash
    return value_hash({"function": it.entry.name, "teacher": teacher_name, "inputs": it.inputs})


def label(items: List[Item], teacher: Any, *, num_threads: int = 16, log: Callable[[str], None] = lambda s: None,
          stop: Optional[threading.Event] = None, keep: Any = None) -> Dict[str, Any]:
    """Ask the teacher for every item without outputs (in place). Returns what
    it took: rows answered and failed, seconds, tokens, dollars. ``keep``: a
    JSON-lines file where each answer is written as it comes, and read back
    first, so a bake interrupted while labeling never pays twice."""
    import json
    from pathlib import Path
    from ..evaluation import parallel
    todo = [it for it in items if it.outputs is None]
    if not todo:
        return {"rows": 0}
    kept: Dict[str, Dict[str, Any]] = {}
    keep_path = Path(keep) if keep else None
    if keep_path is not None and keep_path.exists():
        for line in keep_path.read_text().splitlines():
            try:
                rec = json.loads(line)
                kept[rec["key"]] = rec["outputs"]
            except (ValueError, KeyError):
                continue
    reused = 0
    if kept:
        for it in todo:
            k = _answer_key(it, teacher_name(teacher_for(it.entry.fn, teacher)))
            if k in kept:
                it.outputs, it.source = kept[k], "teacher"
                reused += 1
        todo = [it for it in todo if it.outputs is None]
        if reused:
            log(f"teacher: {reused:,} answers kept from before")
    if not todo:
        return {"rows": reused, "reused": reused}
    out_file = open(keep_path, "a") if keep_path is not None else None
    lock = threading.Lock()
    usage = {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}
    failures: List[str] = []
    done = [0]
    t0 = time.time()
    fns = {id(it.entry): teacher_for(it.entry.fn, teacher) for it in todo}
    prices = {k: _price(f) for k, f in fns.items()}
    spent = [0.0]
    every = max(10, len(todo) // 20)

    def one(it: Item) -> None:
        if stop is not None and stop.is_set():
            return
        tfn = fns[id(it.entry)]
        try:
            pred = tfn._invoke((), it.inputs, full=True)
        except Exception as exc:  # noqa: BLE001 — a failed row is dropped and counted
            with lock:
                failures.append(f"{type(exc).__name__}: {exc}")
            return
        names = [n for n in it.entry.outputs if n != "reasoning" or it.entry.reasoning]
        got = {n: pred[n] for n in names if n in pred}
        if set(got) != set(names):
            with lock:
                failures.append(f"the teacher's answer lacks {sorted(set(names) - set(got))}")
            return
        it.outputs, it.source = got, "teacher"
        u = pred.usage
        with lock:
            if out_file is not None:
                from lmcc.turn import to_json
                out_file.write(json.dumps({"key": _answer_key(it, teacher_name(tfn)), "outputs": to_json(got)},
                                          ensure_ascii=False) + "\n")
                out_file.flush()
            for k in usage:
                usage[k] += int(u.get(k, 0) or 0)
            p = prices[id(it.entry)]
            if p is not None:
                spent[0] += (u.get("input_tokens", 0) * p[0] + (u.get("total_tokens", 0) - u.get("input_tokens", 0))
                             * p[1]) / 1e6
            done[0] += 1
            n = done[0]
        if n % every == 0:
            log(f"teacher: {n:,}/{len(todo):,} rows" + (f", ${spent[0]:.2f} so far" if spent[0] else ""))

    try:
        parallel(one, todo, max(1, num_threads))
    finally:
        if out_file is not None:
            out_file.close()
    info: Dict[str, Any] = {"rows": done[0] + reused, "reused": reused, "failed": len(failures),
                            "seconds": round(time.time() - t0, 1),
                            "teacher": ", ".join(sorted({teacher_name(f) for f in fns.values()})), **usage}
    if all(p is not None for p in prices.values()):
        info["dollars"] = round(spent[0], 4)
    if failures:
        info["first_failure"] = failures[0]
        log(f"teacher: {len(failures):,} rows failed (left out); first: {failures[0]}")
    return info


# ------------------------------------------------------------------ a whole program


def from_program(program: Any, rows: Sequence[Mapping[str, Any]], *, teacher: Any = None,
                 functions: Optional[Sequence[Any]] = None, num_threads: int = 8,
                 log: Callable[[str], None] = lambda s: None) -> Tuple[Dict[Any, Tuple[Any, List[Tuple]]], Dict[str, Any]]:
    """Run ``program`` on each row (its inputs), with ``teacher`` (a model)
    answering every AI call inside it (``None``: each function's own model),
    and keep every call. Returns ``{function: (the function, [(inputs,
    outputs, row id), ...])}`` and what the runs cost."""
    from ..config import forced
    from ..core import FunctAIFunc
    from ..evaluation import _Target, parallel, run_row
    if isinstance(teacher, FunctAIFunc):
        raise BakeError("a program's teacher is a model (it answers every AI function inside); an AI function "
                        "teaches one function: bake it with {function: rows}")
    target = _Target(program)
    keep = None if functions is None else {getattr(f, "_fn", f) for f in functions}
    lock = threading.Lock()
    calls: Dict[Any, Tuple[Any, List[Tuple]]] = {}
    usage = {"input_tokens": 0, "output_tokens": 0}
    failed: List[str] = []
    done = [0]
    t0 = time.time()
    every = max(10, len(rows) // 20)

    def one(pair):
        i, row = pair
        if teacher is not None:
            with forced(lm=teacher):
                run = run_row(target, row)
        else:
            run = run_row(target, row)
        with lock:
            done[0] += 1
            n = done[0]
        if n % every == 0:
            log(f"program: {n:,}/{len(rows):,} rows")
        if run.error:
            with lock:
                failed.append(run.error)
            return
        for k, (fn, pred) in enumerate(run.trace):
            key = getattr(fn, "_fn", fn)
            if keep is not None and key not in keep:
                continue
            with lock:
                for u in usage:
                    usage[u] += int(pred.usage.get(u, 0) or 0)
                calls.setdefault(key, (fn, []))[1].append((dict(pred.turn.inputs), dict(pred), f"{i}.{k}"))

    parallel(one, list(enumerate(rows)), max(1, num_threads))
    if failed:
        log(f"the program failed on {len(failed):,} of {len(rows):,} rows (left out); first: {failed[0]}")
    if not calls:
        raise BakeError("the program made no AI calls on these rows" + (f" (first failure: {failed[0]})" if failed
                                                                         else ""))
    return calls, {"rows": len(rows), "failed": len(failed), "seconds": round(time.time() - t0, 1), **usage}


__all__ = ["Item", "from_rows", "from_program", "label", "teacher_for", "teacher_estimate"]
