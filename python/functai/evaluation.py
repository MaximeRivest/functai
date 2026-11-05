"""Evaluation: run a program over a dataset, get a table back.

Data goes in as rows: a list of dicts (one per row, values kept as the
Python objects they are), or anything ``dpyr.read()`` takes: a parquet/CSV/
JSON path, a pandas/polars/arrow table, a Hugging Face dataset, a dpyr
dataframe. Columns named like the function's parameters are its inputs; the
other columns are the expected outputs and anything else you want to keep.

Results come out as one row per example (``Evaluation.table``, a dpyr
dataframe): the data, the predictions (``pred_<output>``), one column per
metric, and ``error``, ``seconds``, ``input_tokens``, ``output_tokens``,
``reasoning_tokens``, ``total_tokens``, ``model``, ``run``. Tables need dpyr (``pip install "functai[data]"``); the
score and its interval do not.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import contextvars
import dataclasses
import datetime as _dt
import difflib
import enum
import inspect
import json
import math
import os
import secrets
import threading
import time
import typing
import warnings
from collections.abc import Mapping
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

from . import calllog
from .core import _STATE_OVERRIDE, _TRACE, FunctAIFunc, ProgramState
from .data import Prediction, sum_usage, usage_of
from .engine import LoginRequired

States = Dict[FunctAIFunc, ProgramState]

_DPYR_HINT = 'tables need dpyr: pip install "functai[data]"'
# columns every run table has besides the data, the predictions and the metrics
# (lm15 2: on every provider input_tokens includes cached tokens and
# output_tokens includes reasoning; a count not reported exactly is null)
_RUN_COLUMNS = ("error", "seconds", "input_tokens", "output_tokens", "reasoning_tokens", "total_tokens", "model", "run",
                "case_key", "repeat", "call_id", "reused")


def _dpyr():
    try:
        import dpyr
    except ImportError as err:
        raise ImportError(_DPYR_HINT) from err
    return dpyr


def _is_expr(obj: Any) -> bool:
    return type(obj).__module__.startswith("dpyr.")


# ------------------------------------------------------------------ data in


def rows_of(data: Any) -> List[Dict[str, Any]]:
    """The rows of a dataset, as dicts.

    A list of dicts is used as is, so inputs keep their Python types (images,
    dataclasses, ...). Anything else goes through ``dpyr.read()``: a path, a
    pandas/polars/arrow table, a Hugging Face dataset, a dpyr dataframe, a
    dict of columns, or a list of dataclasses / pydantic models."""
    if data is None:
        raise ValueError("no data: pass rows (a list of dicts) or a table")
    if isinstance(data, (list, tuple)) and all(isinstance(r, Mapping) for r in data):
        return [dict(r) for r in data]
    dpyr = _dpyr()
    frame = data if isinstance(data, dpyr.DFrame) else dpyr.read(data)
    if not isinstance(frame, dpyr.DFrame):
        raise TypeError(f"{type(frame).__name__} holds several tables; pick one: "
                        f"read(path, 'table_name')")
    return frame.collect().to_dicts()


# ------------------------------------------------------------------ programs


class _Target:
    """What gets run: one AI function, or a @module calling several."""

    def __init__(self, program: Any, *, call_defaults: Optional[Dict[str, Any]] = None):
        from .module import FunctAIModule
        self.program = program
        self.call_defaults = dict(call_defaults or {})
        kinds = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        if isinstance(program, FunctAIFunc):
            self.predictors: List[FunctAIFunc] = [program]
            named = program._named_inputs()               # a function loaded from data: its interface's
            self.single = True
        elif isinstance(program, FunctAIModule):
            self.predictors = program.ai_functions()
            if getattr(program, "_declared", False):       # its interface, given as data: its inputs by name
                named = [(f["name"], not f.get("optional")) for f in program.interface["inputs"]]
            else:
                named = [(n, p.default is inspect.Parameter.empty)
                         for n, p in inspect.signature(program._fn).parameters.items() if p.kind not in kinds]
            self.single = False
            self.call_defaults = {**program._opt_call_defaults, **self.call_defaults}
            if not self.predictors and not hasattr(program, "_described"):     # a remote program answers itself
                raise ValueError(f"@module {program.__name__} calls no @ai function")
        else:
            raise TypeError(f"expected an @ai function or a @module, not {type(program).__name__}")
        self.name = program.__name__
        self.input_names = [n for n, _required in named]
        self.required = [n for n, required in named if required and n not in self.call_defaults]

    @property
    def output_names(self) -> List[str]:
        if not self.single:
            return ["result"]
        return list(self.predictors[0]._spec().outputs)

    @property
    def main_output(self) -> str:
        return self.predictors[0]._spec().main if self.single else "result"

    @property
    def answer_fields(self) -> Optional[List[str]]:
        """The field names when the answer is a record (a dataclass, a pydantic
        model, a TypedDict), else None."""
        if self.single:
            spec = self.predictors[0]._spec()
            field = next((f for f in spec.signature.outputs if f.name == spec.main), None)
            typ = getattr(field, "annotation", None)
        else:
            from .columns import _return_type
            typ = _return_type(self.program._fn, None)
        return record_fields(typ)

    def inputs_of(self, row: Mapping[str, Any]) -> Dict[str, Any]:
        return {k: row[k] for k in self.input_names if k in row}

    def check(self, rows: Sequence[Mapping[str, Any]]) -> None:
        """Every row has every required input, before any model is called."""
        if not rows:
            raise ValueError("the dataset has no rows")
        columns = list(dict.fromkeys(k for r in rows for k in r))
        for name in self.required:
            if name not in columns:
                close = difflib.get_close_matches(name, columns, n=1)
                hint = f" (is {close[0]!r} a typo for it?)" if close else ""
                raise ValueError(f"{self.name} takes {name!r}, but the data has no column {name!r}{hint}. "
                                 f"Columns: {columns}")
            missing = [i for i, r in enumerate(rows) if name not in r]
            if missing:
                raise ValueError(f"row {missing[0]} has no {name!r}, which {self.name} needs")

    def run(self, row: Mapping[str, Any]) -> Prediction:
        from .conversations import replaying
        inputs = self.inputs_of(row)
        with replaying(row, self.program):        # a row from a conversation: asked with its earlier turns
            if self.single:
                return self.program._invoke((), inputs, full=True)
            return Prediction({"result": self.program(**{**self.call_defaults, **inputs})})


@contextlib.contextmanager
def with_states(states: Optional[States]) -> Iterator[None]:
    """Run with candidate states in place of the functions' own (this context only)."""
    if not states:
        yield
        return
    token = _STATE_OVERRIDE.set({**_STATE_OVERRIDE.get(), **{id(fn): st for fn, st in states.items()}})
    try:
        yield
    finally:
        _STATE_OVERRIDE.reset(token)


@contextlib.contextmanager
def tracing() -> Iterator[List[Tuple[FunctAIFunc, Prediction]]]:
    """Record every AI function call made in this context: ``(fn, prediction)``."""
    trace: List[Tuple[FunctAIFunc, Prediction]] = []
    token = _TRACE.set(trace)
    try:
        yield trace
    finally:
        _TRACE.reset(token)


class Progress:
    """A line on stderr, updated in place: rows done, errors, tokens, time
    left (``map(..., progress=True)``)."""

    def __init__(self, total: int, name: str, *, stream: Any = None):
        import sys
        self.total = total
        self.name = name
        self.out = stream or sys.stderr
        self.done = 0
        self.errors = 0
        self.tokens = 0
        self.t0 = time.perf_counter()
        self.shown = 0.0
        self.lock = threading.Lock()

    def step(self, run: "RowRun") -> None:
        with self.lock:
            self.done += 1
            self.errors += run.error is not None
            self.tokens += int(run.usage.get("total_tokens") or 0)
            now = time.perf_counter()
            if now - self.shown >= 0.2 or self.done == self.total:
                self.shown = now
                self._write()

    def line(self) -> str:
        elapsed = time.perf_counter() - self.t0
        left = elapsed / self.done * (self.total - self.done) if self.done else None
        parts = [f"{self.name}: {self.done}/{self.total} rows"]
        if self.errors:
            parts.append(f"{self.errors} error{'s' if self.errors != 1 else ''}")
        parts.append(f"{self.tokens:,} tokens")
        parts.append(f"{_duration(elapsed)} so far" if self.done == self.total or left is None
                     else f"about {_duration(left)} left")
        return " · ".join(parts)

    def _write(self) -> None:
        try:
            self.out.write("\r" + self.line() + ("\n" if self.done == self.total else ""))
            self.out.flush()
        except Exception:  # noqa: BLE001 — a progress line never fails a run
            pass


def _duration(seconds: float) -> str:
    s = int(round(seconds))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


def show_progress(setting: Any) -> bool:
    """``progress=None``: on when stderr is a terminal, or in a notebook."""
    if setting is not None:
        return bool(setting)
    import sys
    try:
        if sys.stderr.isatty():
            return True
    except Exception:  # noqa: BLE001
        pass
    return "ipykernel" in sys.modules


def parallel(fn: Callable, items: Sequence[Any], num_threads: int) -> List[Any]:
    """``fn`` over items, in order; each task runs in a copy of the caller's
    context, so ``with configure(...)`` and candidate states reach the threads."""
    if num_threads <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as pool:
        futures = [pool.submit(contextvars.copy_context().run, fn, x) for x in items]
        return [f.result() for f in futures]


@dataclasses.dataclass
class RowRun:
    """One row run (one repeat of one case): the prediction (None when the
    program raised), the error, what it cost, and the call that answered (its
    id; ``reused`` when it was taken from the log instead of asked)."""
    pred: Optional[Prediction]
    error: Optional[str]
    seconds: float
    trace: List[Tuple[FunctAIFunc, Prediction]]
    call_id: Optional[str] = None
    reused: bool = False
    recorded: Optional[Dict[str, Any]] = None        # the reused call's record

    @property
    def usage(self) -> Dict[str, int]:
        """Tokens summed over the row's model calls; a counter one of them did
        not report is absent (``functai.data.sum_usage``). A reused call's are
        its record's."""
        if self.recorded is not None:
            return {k: v for k, v in (self.recorded.get("usage") or {}).items() if isinstance(v, int)}
        return sum_usage(usage_of(r) for _fn, p in self.trace for r in p.responses)

    @property
    def model(self) -> Optional[str]:
        if self.recorded is not None:
            return self.recorded.get("model")
        names = sorted({r.model for _fn, p in self.trace for r in p.responses if getattr(r, "model", None)})
        return ", ".join(names) or None


def run_row(target: _Target, row: Mapping[str, Any], *, case: Optional[str] = None, repeat: int = 1) -> RowRun:
    """Run the program on one row. A failure is recorded, not raised, except a
    missing login, which would fail every row the same way. With ``case``,
    the call says which case it answered, and which repeat; a repeat after
    the first is another replicate (contract/replies.md), so a reply cache
    gives it its own answer."""
    from .config import scoped
    start = time.perf_counter()
    with tracing() as trace, calllog.marking(case, repeat) as mark, \
            (scoped(replicate=repeat - 1) if repeat > 1 else contextlib.nullcontext()):
        try:
            pred, error = target.run(row), None
        except LoginRequired:
            raise
        except Exception as exc:  # noqa: BLE001 — recorded in the row's error
            pred, error = None, f"{type(exc).__name__}: {exc}"
    call_id = getattr(pred, "call_id", None) or mark.get("call")
    return RowRun(pred, error, time.perf_counter() - start, list(trace), call_id=call_id)


# ------------------------------------------------------------------ metrics


def _norm(v: Any) -> Any:
    if isinstance(v, str):
        return " ".join(v.split()).casefold()
    if isinstance(v, enum.Enum):
        return _norm(v.value)
    return v


def exact_match(row: Mapping[str, Any], pred: Mapping[str, Any]) -> float:
    """The default metric: every expected output equals the prediction.

    1.0 when every output the data has a column for equals the prediction's
    (strings compared ignoring case and repeated whitespace), else 0.0."""
    keys = [k for k in pred if k in row]
    if not keys:
        raise ValueError(f"exact_match: the data has no column for any output ({list(pred)})")
    return float(all(_norm(row[k]) == _norm(pred[k]) for k in keys))


def record_fields(typ: Any) -> Optional[List[str]]:
    """The fields of a record type (dataclass, pydantic model, TypedDict), else None."""
    if typ is None or not isinstance(typ, type):
        return None
    if dataclasses.is_dataclass(typ):
        return [f.name for f in dataclasses.fields(typ)]
    fields = getattr(typ, "model_fields", None)
    if isinstance(fields, Mapping):
        return list(fields)
    if typing.is_typeddict(typ):
        return list(typing.get_type_hints(typ))
    return None


def answer_part(pred: Mapping[str, Any], key: str, main: str) -> Any:
    """An output of the prediction, or a field of its answer when the answer is a record."""
    if key in pred:
        return pred[key]
    ans = pred.get(main)
    if isinstance(ans, Mapping):
        return ans.get(key)
    return getattr(ans, key, None)


def expected_columns(target: "_Target", expected: Any, rows: Sequence[Mapping[str, Any]]) -> Dict[str, str]:
    """Where the right answers are, as ``{output or answer field: column}``.

    ``expected="category"``: the answer is in that column. A list of names:
    outputs, or fields of a record answer, each in the column of the same
    name. A dict: ``{output or field: column}``. None: the columns named like
    an output; or, for a record answer, like its fields."""
    outs, main = target.output_names, target.main_output
    fields = target.answer_fields or []
    columns = {k for r in rows for k in r}
    if expected is None:
        if any(o in columns for o in outs):
            return {}                                        # the classic case: exact_match on outputs
        return {f: f for f in fields if f in columns}
    if isinstance(expected, str):
        mapping = {main: expected}
    elif isinstance(expected, Mapping):
        mapping = {str(k): str(v) for k, v in expected.items()}
    elif isinstance(expected, (list, tuple)):
        mapping = {str(k): str(k) for k in expected}
    else:
        raise TypeError("expected= is a column name (the answer's), a list of names, or a dict {name: column}")
    unknown = [k for k in mapping if k not in outs and k not in fields]
    if unknown:
        known = outs + [f"{main}.{f}" for f in fields]
        raise ValueError(f"expected=: {target.name} has no output or answer field {unknown[0]!r} "
                         f"(it has: {', '.join(known)})")
    missing = [c for c in mapping.values() if c not in columns]
    if missing:
        raise ValueError(f"expected={missing[0]!r}: the data has no such column "
                         f"(columns: {sorted(columns)})")
    return mapping


def with_expected(rows: Sequence[Mapping[str, Any]], mapping: Mapping[str, str], target: "_Target"
                  ) -> List[Dict[str, Any]]:
    """The rows with the right answers also under the outputs' names (answer
    fields gathered into one record), which is how optimizers read labels."""
    main, outs = target.main_output, target.output_names
    out = []
    for r in rows:
        r2 = dict(r)
        record = {}
        for key, c in mapping.items():
            if c not in r:
                continue
            if key in outs:
                r2[key] = r[c]
            else:
                record[key] = r[c]
        if record:
            r2[main] = record
        out.append(r2)
    return out


def expected_metrics(mapping: Mapping[str, str], target: "_Target") -> List["Metric"]:
    """Exact match on every expected column (``exact_match``), and, when there
    are several, one metric per column (``<name>_match``)."""
    main = target.main_output

    def on(keys: Sequence[str]) -> Callable:
        def exact_match(row: Mapping[str, Any], pred: Mapping[str, Any]) -> float:
            return float(all(_norm(row.get(mapping[k])) == _norm(answer_part(pred, k, main)) for k in keys))
        return exact_match

    keys = list(mapping)
    metrics = [Metric("exact_match", fn=on(keys))]
    if len(keys) > 1:
        metrics += [Metric(f"{k}_match", fn=on([k])) for k in keys]
    return metrics


@dataclasses.dataclass
class Metric:
    """A resolved metric: a Python callable ``metric(row, prediction)``, run
    per row, or a dpyr expression over the run table's columns."""
    name: str
    fn: Optional[Callable] = None
    expr: Any = None

    def score(self, target: _Target, row: Mapping[str, Any], run: RowRun) -> Optional[float]:
        """This metric for one row run (None when the run failed)."""
        if run.pred is None:
            return None
        if self.fn is not None:
            if _is_program(self.fn):
                # an AI judge gets plain data, so typed parameters (row: dict,
                # prediction: dict) can be written into its prompt as JSON
                return as_score(self.fn(_cell(dict(row)), _cell(dict(run.pred))))
            return as_score(self.fn(dict(row), run.pred))
        return expr_scores(self, target, [row], [run])[0]


def _is_program(fn: Any) -> bool:
    from .module import FunctAIModule
    return isinstance(fn, (FunctAIFunc, FunctAIModule))


def as_score(value: Any) -> float:
    if isinstance(value, Prediction):                 # an AI judge's full prediction
        value = next(iter(value.values()), None)
    if value is None:
        raise ValueError("the metric returned None")
    return float(value)


def _check_arity(fn: Callable, name: str) -> None:
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return
    positional = [p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    required = [p for p in positional if p.default is p.empty]
    if any(p.kind is p.VAR_POSITIONAL for p in params):
        return
    if len(required) > 2 or len(positional) < 2:
        raise TypeError(f"metric {name} must take (row, prediction); its signature is "
                        f"{inspect.signature(fn)}")


def resolve_metrics(metric: Any, target: _Target, rows: Sequence[Mapping[str, Any]],
                    expected: Optional[Mapping[str, str]] = None) -> List[Metric]:
    """``metric`` as a list of named metrics. None: exact match against the
    ``expected`` columns, or the columns named like an output; else no metric.
    Accepts one metric, a list, or a dict of name → metric; a metric is a
    callable or a dpyr expression."""
    if metric is None and expected:
        return expected_metrics(expected, target)
    if metric is None:
        outs = set(target.output_names)
        labeled = any(k in outs for r in rows for k in r)
        return [Metric("exact_match", fn=exact_match)] if labeled else []
    if isinstance(metric, Mapping):
        items = list(metric.items())
    elif isinstance(metric, (list, tuple)):
        items = [(None, m) for m in metric]
    else:
        items = [(None, metric)]
    out: List[Metric] = []
    taken: set = set()
    for name, m in items:
        if _is_expr(m):
            base = name or "score"
        elif callable(m):
            base = name or getattr(m, "__name__", None) or "score"
            base = "score" if base == "<lambda>" else base
            _check_arity(m, base)
        else:
            raise TypeError(f"a metric is a function (row, prediction) -> number, or a dpyr expression; "
                            f"got {type(m).__name__}")
        unique, k = base, 2
        while unique in taken:
            unique, k = f"{base}_{k}", k + 1
        taken.add(unique)
        out.append(Metric(unique, expr=m) if _is_expr(m) else Metric(unique, fn=m))
    if any(m.fn is exact_match for m in out):
        outs = set(target.output_names)
        if not any(k in outs for r in rows for k in r):
            raise ValueError(f"exact_match compares outputs with the data's columns, but the data has no "
                             f"column named {sorted(outs)}")
    return out


def expr_scores(metric: Metric, target: _Target, rows: Sequence[Mapping[str, Any]],
                runs: Sequence[RowRun]) -> List[Optional[float]]:
    """A dpyr expression metric over the run table of these rows."""
    dpyr = _dpyr()
    names, records = _records(target, rows, runs, [], {}, run_id="")
    frame = dpyr.read(_tabular(names, records))
    try:
        values = frame.mutate(**{"functai_metric_value": metric.expr}).pull()
    except dpyr.DpyrError as err:
        raise type(err)(f"metric {metric.name}: {err}") from None
    out: List[Optional[float]] = []
    for v, run in zip(values, runs):
        if run.pred is None or v is None:
            out.append(None)
        elif isinstance(v, (bool, int, float)):
            out.append(float(v))
        else:
            raise TypeError(f"metric {metric.name} must give numbers or booleans, got {type(v).__name__}")
    return out


# ------------------------------------------------------------------ table cells


def _cell(v: Any) -> Any:
    """A value as a table cell: numbers, strings, dates, and lists/dicts of
    those. Enums become their values, dataclasses and pydantic models dicts;
    anything else becomes its text."""
    if v is None or isinstance(v, (bool, int, float, str, _dt.date)):
        return v
    if isinstance(v, enum.Enum):
        return _cell(v.value)
    if isinstance(v, Mapping):
        return {str(k): _cell(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set, frozenset)):
        return [_cell(x) for x in v]
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        return _cell({f.name: getattr(v, f.name) for f in dataclasses.fields(v)})
    dump = getattr(v, "model_dump", None)
    if callable(dump):
        return _cell(dump())
    return str(v)


def spread_fields(target: "_Target") -> List[str]:
    """A record answer's fields, when each gets its own ``pred_<field>`` column
    (none of them named like another output); else []."""
    fields = target.answer_fields or []
    others = set(target.output_names) - {target.main_output}
    return [] if any(f in others for f in fields) else fields


def _records(target: _Target, rows: Sequence[Mapping[str, Any]], runs: Sequence[RowRun],
             metrics: Sequence[Metric], values: Mapping[str, Sequence[Optional[float]]],
             run_id: str, meta: Optional[Sequence[Mapping[str, Any]]] = None) -> Tuple[List[str], List[Dict[str, Any]]]:
    """The run table's columns and records: one per row run (a row asked several times is several records).
    ``meta``: each run's ``example`` (its row's position), ``case_key`` and ``repeat``."""
    data_cols = list(dict.fromkeys(k for r in rows for k in r))
    outs = list(target.output_names)
    for run in runs:
        for k in (run.pred or {}):
            if k not in outs:
                outs.append(k)
    main, fields = target.main_output, spread_fields(target)
    pred_cols = [c for k in outs for c in ([f"pred_{f}" for f in fields] if k == main and fields
                                            else [f"pred_{k}"])]
    names = ["example", *data_cols, *pred_cols, *(m.name for m in metrics), *_RUN_COLUMNS]
    records = []
    for i, (row, run) in enumerate(zip(rows, runs)):
        usage = run.usage
        m = meta[i] if meta is not None else {"example": i, "case_key": None, "repeat": 1}
        rec = {"example": m["example"], **{k: _cell(row.get(k)) for k in data_cols}}
        for k in outs:
            if k == main and fields:
                rec.update({f"pred_{f}": _cell(answer_part(run.pred, f, main)) if run.pred is not None else None
                            for f in fields})
            else:
                rec[f"pred_{k}"] = _cell(run.pred.get(k)) if run.pred is not None else None
        rec.update({mt.name: values[mt.name][i] for mt in metrics})
        rec.update(error=run.error, seconds=run.seconds, input_tokens=usage.get("input_tokens"),
                   output_tokens=usage.get("output_tokens"), reasoning_tokens=usage.get("reasoning_tokens"),
                   total_tokens=usage.get("total_tokens"), model=run.model, run=run_id,
                   case_key=m["case_key"], repeat=m["repeat"], call_id=run.call_id, reused=run.reused)
        records.append(rec)
    return names, records


def _tabular(names: List[str], records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Records every row of which has every column in order, with any column
    whose values mix types (a model that answers 4 on one row and "4" on the
    next) turned into JSON text, with a warning, rather than lost."""
    dpyr = _dpyr()
    out = [{k: rec.get(k) for k in names} for rec in records]
    try:
        dpyr.read(out)
        return out
    except dpyr.DpyrError:
        pass
    for name in names:
        try:
            dpyr.read([{name: rec[name]} for rec in out])
        except dpyr.DpyrError as err:
            warnings.warn(f"[functai] column {name!r} mixes types ({err}); stored as JSON text", stacklevel=3)
            for rec in out:
                if rec[name] is not None:
                    rec[name] = json.dumps(rec[name], ensure_ascii=False, default=str)
    return out


# ------------------------------------------------------------------ statistics

# two-sided 95% Student t quantiles, by degrees of freedom
_T975 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160,
         2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056,
         2.052, 2.048, 2.045, 2.042]
_Z975 = 1.959964


def _t975(df: int) -> float:
    if df <= len(_T975):
        return _T975[df - 1]
    z = _Z975                        # Cornish-Fisher expansion: within 1e-4 for df > 30
    return z + (z ** 3 + z) / (4 * df) + (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * df ** 2)


def interval(values: Sequence[float]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """(mean, low, high) with a 95% interval: Wilson's for 0/1 scores (right
    near 0% and 100%, where the textbook ± interval is not), Student's t for
    the others. The interval is None with fewer than two values."""
    n = len(values)
    if n == 0:
        return None, None, None
    mean = sum(values) / n
    if n < 2:
        return mean, None, None
    if all(v in (0.0, 1.0) for v in values):
        z2 = _Z975 ** 2
        center = (mean + z2 / (2 * n)) / (1 + z2 / n)
        half = _Z975 * math.sqrt(mean * (1 - mean) / n + z2 / (4 * n * n)) / (1 + z2 / n)
        low = 0.0 if mean == 0.0 else max(0.0, center - half)     # exact at the ends
        high = 1.0 if mean == 1.0 else min(1.0, center + half)
        return mean, low, high
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
    half = _t975(n - 1) * sd / math.sqrt(n)
    return mean, mean - half, mean + half


def _fmt(x: Optional[float]) -> str:
    return "—" if x is None else f"{x:.3g}" if abs(x) >= 10 else f"{x:.2f}"


# ------------------------------------------------------------------ the result


class Evaluation:
    """The result of ``evaluate``: a score, its uncertainty, and every answer.

    ``ev.scores(metric)`` gives each row's value for a metric (the first by
    default; a row asked several times, the mean of its repeats);
    ``ev.write("run.parquet")`` saves the table.

    Attributes
    ----------
    score : float
        The first metric's mean over cases, from 0 to 1. A failed row counts 0.
    summary : dpyr dataframe
        One row per metric: ``mean``, the 95% interval ``low`` to ``high``
        (over cases), ``n`` (cases), and how many runs ``failed``.
    table : dpyr dataframe
        One row per run (``repeats`` runs per example): the data,
        ``pred_<output>`` for each output, each metric, ``error``,
        ``seconds``, ``input_tokens``, ``output_tokens``,
        ``reasoning_tokens``, ``total_tokens``, ``model``, ``run``,
        ``case_key``, ``repeat``, ``call_id`` (the call that answered: its
        trace in the call log) and ``reused`` (taken from the log).
    predictions : list
        Each run's ``Prediction`` (None where it failed), with its turns,
        tokens and repairs.
    errors : list
        ``(row number, message)`` for each run that failed.
    cases : list of str
        Each row's case key (``functai.case_key``).

    See Also
    --------
    evaluate : what produces it.
    compare : two of them, paired by case.
    """

    def __init__(self, *, target: _Target, rows: List[Dict[str, Any]], runs: List[RowRun],
                 metrics: List[Metric], values: Dict[str, List[Optional[float]]], run_id: str,
                 jobs: Optional[List[Tuple[int, int]]] = None, keys: Optional[List[str]] = None,
                 kinds: Optional[Dict[str, Any]] = None):
        self.program = target.name
        self.run = run_id
        self.metrics = [m.name for m in metrics]
        self.predictions = [r.pred for r in runs]
        self._target, self._rows, self._runs, self._metrics, self._values = target, rows, runs, metrics, values
        self._jobs = jobs if jobs is not None else [(i, 1) for i in range(len(rows))]
        self.cases = keys if keys is not None else [None] * len(rows)
        # how each metric's values count: "legacy" (a failed run is 0), or a criterion's data (none is no value)
        self._kinds = kinds or {}
        self._table = None

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def errors(self) -> List[Tuple[int, str]]:
        return [(self._jobs[i][0], r.error) for i, r in enumerate(self._runs) if r.error]

    @property
    def calls(self) -> List[Optional[str]]:
        """Each run's call id (its trace in the call log), in the table's order."""
        return [r.call_id for r in self._runs]

    def _per_row(self, name: str) -> List[Optional[float]]:
        """A metric's value per row: the mean of its runs' (a legacy metric's failed run counts 0; a
        criterion's run without a value is left out, and a row with none has none)."""
        legacy = self._kinds.get(name) is None
        acc: List[List[float]] = [[] for _ in self._rows]
        for (i, _rep), v in zip(self._jobs, self._values[name]):
            if v is None and legacy:
                v = 0.0
            if v is not None:
                acc[i].append(v)
        return [sum(a) / len(a) if a else None for a in acc]

    def scores(self, metric: Optional[str] = None) -> List[float]:
        """One metric's value per row (a failed row counting 0; the mean of a row's repeats)."""
        if not self.metrics:
            raise ValueError("this evaluation has no metric")
        name = metric or self.metrics[0]
        return [0.0 if v is None else v for v in self._per_row(name)]

    def by_case(self, metric: Optional[str] = None) -> Dict[str, float]:
        """``{case key: value}``: a metric's value per case (rows of one case, and repeats, averaged)."""
        name = metric or self.metrics[0]
        acc: Dict[str, List[float]] = {}
        for k, v in zip(self.cases, self._per_row(name)):
            if v is not None:
                acc.setdefault(k, []).append(v)
        return {k: sum(v) / len(v) for k, v in acc.items()}

    def _bounded(self, name: str) -> bool:
        kind = self._kinds.get(name)
        if kind is None:
            return name == "exact_match" or name.endswith("_match")
        from .measures import bounded
        return bounded(kind)

    @property
    def score(self) -> Optional[float]:
        if not self.metrics or not self._rows:
            return None
        return interval(list(self.by_case(self.metrics[0]).values()))[0]

    def __float__(self) -> float:
        if self.score is None:
            raise ValueError("this evaluation has no metric")
        return self.score

    def _summary_records(self) -> List[Dict[str, Any]]:
        out = []
        n_err = len(self.errors)
        for name in self.metrics:
            per = self.by_case(name)
            mean, low, high = interval([per[k] for k in sorted(per)])
            if low is not None and self._kinds.get(name) is not None and self._bounded(name):
                low, high = max(0.0, low), min(1.0, high)
            failed = sum(1 for r, v in zip(self._runs, self._values[name]) if v is None and r.error)
            out.append({"metric": name, "mean": mean, "low": low, "high": high, "n": len(per), "failed": failed})
        if not out:
            out.append({"metric": None, "mean": None, "low": None, "high": None, "n": len(self._rows),
                        "failed": n_err})
        return out

    @property
    def summary(self):
        """One row per metric: mean, 95% interval (low, high) over cases, n (cases), and how many runs
        failed (counted as 0 in the mean)."""
        return _dpyr().read(self._summary_records())

    @property
    def table(self):
        """One row per run: ``example`` (the row's position in the data), the data's columns,
        ``pred_<output>`` for each output, one column per metric (null where the run failed), then
        ``error``, ``seconds``, ``input_tokens``, ``output_tokens``, ``reasoning_tokens``, ``total_tokens``,
        ``model``, ``run``, ``case_key``, ``repeat``, ``call_id`` and ``reused``."""
        if self._table is None:
            names, records = _records(self._target, [self._rows[i] for i, _r in self._jobs], self._runs,
                                      self._metrics, self._values, self.run, self._meta())
            self._table = _dpyr().read(_tabular(names, records))
        return self._table

    def _meta(self) -> List[Dict[str, Any]]:
        return [{"example": i, "case_key": self.cases[i], "repeat": rep} for i, rep in self._jobs]

    def write(self, path: "str | os.PathLike") -> None:
        """Save the table (``.parquet`` keeps every type; also .jsonl, .arrow, ...)."""
        self.table.write(os.fspath(path))

    def __repr__(self) -> str:
        n = len(self._rows)
        head = f"Evaluation({self.program}, {n} example{'s' if n != 1 else ''}"
        reps = len(self._jobs) // n if n and len(self._jobs) % n == 0 else None
        if reps and reps > 1:
            head += f" × {reps} repeats"
        parts = []
        for rec in self._summary_records():
            if rec["metric"] is None:
                continue
            ci = f" [{_fmt(rec['low'])}, {_fmt(rec['high'])}]" if rec["low"] is not None else ""
            parts.append(f"{rec['metric']} {_fmt(rec['mean'])}{ci}")
        n_err = len(self.errors)
        tail = f", {n_err} failed" if n_err else ""
        reused = sum(1 for r in self._runs if r.reused)
        if reused:
            tail += f", {reused} reused"
        return head + (": " + "; ".join(parts) if parts else "") + tail + ")"


# ------------------------------------------------------------------ evaluate


def _new_run_id(name: str) -> str:
    return f"{name}-{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(2)}"


def _check_names(target: _Target, rows: Sequence[Mapping[str, Any]], metrics: Sequence[Metric]) -> None:
    columns = set(k for r in rows for k in r)
    added = {"example", *_RUN_COLUMNS, *(m.name for m in metrics),
             *(f"pred_{k}" for k in target.output_names), *(f"pred_{f}" for f in spread_fields(target))}
    clash = sorted(columns & added)
    if clash:
        raise ValueError(f"the data has column(s) {clash}, which the run table adds; rename them first")


def _reuse_records(reuse: Any) -> Tuple[List[Dict[str, Any]], Any]:
    """The records to reuse results from, and where this evaluation's calls and ratings then go."""
    if reuse is True:
        found, ratings = calllog.read(None)
        return found + ratings, None
    if isinstance(reuse, (list, tuple)):
        return [dict(r) for r in reuse], None
    if calllog.is_store(reuse):
        return [dict(r) for r in reuse.read() if isinstance(r, Mapping)], reuse
    found, ratings = calllog.read(reuse)
    return found + ratings, reuse


def _typed(target: _Target, outputs: Mapping[str, Any]) -> Dict[str, Any]:
    """A logged call's outputs (JSON) as the program's types, as far as they build."""
    from .signature import from_json
    if target.single:
        spec = target.predictors[0]._spec()
        return {k: from_json(spec.annotations.get(k), v) for k, v in outputs.items()}
    from .columns import _return_type
    try:
        typ = _return_type(target.program._fn, None)
    except Exception:  # noqa: BLE001
        typ = None
    return {k: from_json(typ, v) if k == "result" else v for k, v in outputs.items()}


def _reused_run(target: _Target, rec: Mapping[str, Any]) -> RowRun:
    pred = Prediction(_typed(target, rec.get("outputs") or {}))
    object.__setattr__(pred, "call_id", rec["id"])
    return RowRun(pred, None, float(rec.get("seconds") or 0.0), [], call_id=rec["id"], reused=True, recorded=dict(rec))


def _criteria(criteria: Any) -> List[Any]:
    from .measures import Criterion
    out = []
    for c in criteria or ():
        if not isinstance(c, Criterion):
            raise TypeError(f"criteria are functai.Criterion, each with a judge; not {type(c).__name__}")
        if c.kind == "choice":
            raise ValueError(f"{c.name} is a choice between two answers: functai.judge_pairs")
        if c.judge is None:
            raise ValueError(f"{c.name} has no judge: evaluate scores with judges (Criterion(..., judge=...)); "
                             "people's ratings are read by functai.scoreboard")
        out.append(c)
    return out


def _judge_run(target: _Target, row: Mapping[str, Any], run: RowRun, crit: Any, expected_cols: Mapping[str, str],
               known: Mapping[Any, Any], keep: List[Dict[str, Any]]) -> Optional[float]:
    """A criterion's value for one run: its judge asked (or its rating reused from the log), its ratings kept."""
    from . import measures
    data = crit.data()
    if run.error is not None or run.pred is None:
        return 0.0 if data.get("items") is None or crit.aggregate == "sum" else None
    outputs = calllog.to_json(dict(run.pred))[0]
    inputs = {k: calllog.to_json(row[k])[0] for k in target.input_names if k in row}
    answer = outputs.get(target.main_output) if isinstance(outputs, Mapping) else None
    exp = {o: calllog.to_json(row[c])[0] for o, c in expected_cols.items() if c in row} or \
        {o: calllog.to_json(row[o])[0] for o in target.output_names if o in row}
    base = {**inputs, "inputs": inputs, "outputs": outputs, "answer": answer,
            "expected": exp.get(target.main_output) if len(exp) <= 1 else exp}
    who = {"judge": measures.judge_identity(crit.judge)}
    jid = who["judge"].get("setup") or who["judge"].get("code")
    targets = [(None, None)] if data.get("items") is None else [
        (f"{crit.items}/{i}", it) for i, it in enumerate(measures._item_values(outputs, crit.items))]
    values = []
    for item, it in targets:
        prior = known.get((run.call_id, item or "", jid))
        if prior is not None:
            value = prior
        else:
            got, jc = measures.ask_judge(crit.judge, {**base, "item": it} if item is not None else base)
            value = measures.judged_value(data, got)
            if run.call_id is not None:
                keep.append(measures.rating_record(run.call_id, data, value, who=who, item=item, origin="judge",
                                                   judge_call=jc))
        x = measures.number_of({**data, "value": value, "answers": data.get("answers"), "scale": data.get("scale")})
        if x is not None:
            values.append(x)
    if data.get("items") is None:
        return values[0] if values else None
    if not values:
        return 0.0 if crit.aggregate == "sum" and not targets else None
    share = sum(values) / len(values)
    return share * len(targets) if crit.aggregate == "sum" else share


def evaluate(program: Any, data: Any, metric: Any = None, *, expected: Any = None, num_threads: int = 1,
             max_errors: Optional[int] = None, log: "str | os.PathLike | None" = None,
             call_defaults: Optional[Dict[str, Any]] = None, states: Optional[States] = None,
             threads: Optional[int] = None, progress: Optional[bool] = False, repeats: int = 1,
             reuse: Any = None, criteria: Optional[Sequence[Any]] = None) -> Evaluation:
    '''Run a program on rows with known answers, and score it.

    Every row runs (in parallel with ``num_threads``); a row that fails
    keeps its error and counts 0. The score comes with a 95% interval over
    cases, and every answer is kept as a row of a table you can filter and
    group. Each call says which case it answered (its record's ``case``), so
    later evaluations can reuse it and comparisons pair by case.

    Parameters
    ----------
    program : AI function or module
        What to evaluate.
    data : list of dict, or a table
        The rows: a list of dicts, or anything ``dpyr.read()`` takes (a
        parquet or CSV path, a pandas or polars dataframe, a Hugging Face
        dataset). Columns named like the parameters are the inputs; a column
        named like an output (``result`` for the return value) is its
        expected answer, unless ``expected=`` names another; other columns
        are kept.
    expected : str or dict, optional
        The column holding the right answers, when it isn't named like the
        output: ``expected="category"``. A dict names a column per output:
        ``{"result": "category", "order_id": "order"}``. The default metric is
        then exact match against these columns.
    metric : function, dpyr expression, AI function, list or dict
        How to score a row: ``metric(row, prediction)`` returning a number or a bool, a
        dpyr expression over the table (``col.pred_result == col.result``),
        an AI function acting as a judge, or several of these in a list or a
        dict ``{name: metric}``. Default: exact match on the outputs the data
        has columns for (case and spacing ignored).
    criteria : list of functai.Criterion, optional
        Questions with a judge (an AI function, a module or a function): each
        becomes a metric named like it, its ratings are kept in the call log
        (naming the judge by its setup), and a judgement already there for
        the same call and judge is reused. An item criterion judges each
        item of a list in the answer.
    repeats : int
        How many times each row is asked. A row's value is the mean of its
        repeats, and every range stays over cases; a repeat after the first
        is another replicate, so a reply cache gives it its own answer.
    reuse : folder, store, list of records, or True
        Take results already in this log (True: the call log): calls of the
        same program on the same setup and case that ended without an error,
        oldest first, count as the first repeats; only the rest are asked.
        This evaluation's calls and ratings go to the same place.
    num_threads, threads : int
        How many rows run at once (``threads`` is the same, by the name
        ``map`` and ``vectorize`` use).
    progress : bool, optional
        A line on stderr, updated as rows finish (None: when stderr is a
        terminal or in a notebook). Off by default here.
    max_errors : int, optional
        Stop and raise when more runs than this fail.
    log : folder, optional
        Write the run's table to ``<log>/<run>.parquet``; ``runs(log)`` reads
        every logged run back.
    call_defaults : dict, optional
        Arguments the rows don't have, for every call (a module's options).

    Returns
    -------
    Evaluation
        ``.score`` (the first metric's mean), ``.summary`` (each metric with
        its interval), ``.table`` (one row per run).

    See Also
    --------
    compare : two evaluations, paired by case.
    Evaluation : what this returns.
    functai.scoreboard : every setup of a program, from the call log.

    Examples
    --------
    ```python
    from typing import Literal
    from dpyr import col

    @ai
    def team(message: str) -> Literal["shipping", "billing", "product", "account"]:
        """Which team should answer this customer message?"""
        ...

    tickets = functai.datasets.tickets().slice_head(n=20)
    ev = evaluate(team, tickets, expected="category", num_threads=8)
    ev
    ```

    ```python
    ev.table.filter(col.exact_match == 0).select(col.message, col.category, col.pred_result)
    ```
    '''
    if threads is not None:
        num_threads = threads
    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats < 1:
        raise ValueError(f"repeats is a whole number of at least 1, not {repeats!r}")
    if reuse is not None and reuse is not False and states:
        raise ValueError("reuse takes results of the program's own setup; candidate states run another")
    target = _Target(program, call_defaults=call_defaults)
    rows = rows_of(data)
    target.check(rows)
    expected_cols = expected_columns(target, expected, rows)
    crits = _criteria(criteria)
    metrics = resolve_metrics(metric, target, rows, expected_cols) if (metric is not None or not crits or expected_cols
                                                                        or any(o in r for r in rows for o in target.output_names)) else []
    from .measures import row_key, call_key
    taken = {m.name for m in metrics}
    for c in crits:
        if c.name in taken:
            raise ValueError(f"two metrics are named {c.name}: rename the criterion or the metric")
        taken.add(c.name)
    _check_names(target, rows, [*metrics, *(Metric(c.name) for c in crits)])
    if log is not None or any(m.expr is not None for m in metrics):
        _dpyr()                                            # fail now, not after the model calls
    row_metrics = [m for m in metrics if m.fn is not None]
    keys = [row_key(r, target.input_names) for r in rows]

    # results already in the log: the program's own setup, on the same case, ended well, oldest first
    reused: Dict[str, List[Dict[str, Any]]] = {}
    known: Dict[Any, Any] = {}
    store = None
    if reuse is not None and reuse is not False:
        found, store = _reuse_records(reuse)
        from .measures import counting, program_calls, _program_of, judge_identity
        setup = getattr(program, "setup", None)
        mine = [c for c in program_calls(found, _program_of(program))
                if (c.get("program") or {}).get("setup") == setup and c.get("error") is None
                and c.get("content") is True and isinstance(c.get("outputs"), Mapping)]
        for c in mine:
            k = call_key(c)
            if k is not None:
                reused.setdefault(k, []).append(c)
        ratings = [r for r in found if "functai_rating" in r]
        for c in crits:
            ident = judge_identity(c.judge)
            jid = ident.get("setup") or ident.get("code")
            for j in counting(ratings, c.data()):
                if j["source"] == ("judge", jid) and len(j["calls"]) == 1:
                    known[(j["calls"][0], j["item"], jid)] = j["value"]
    jobs: List[Tuple[int, int]] = []
    plan: List[Optional[Dict[str, Any]]] = []
    for i, k in enumerate(keys):
        have = reused.get(k, [])
        for rep in range(1, repeats + 1):
            jobs.append((i, rep))
            plan.append(have[rep - 1] if rep - 1 < len(have) else None)

    run_id = _new_run_id(target.name)
    meter = Progress(sum(1 for p in plan if p is None), target.name) if show_progress(progress) else None
    kept: List[Dict[str, Any]] = []
    kept_lock = threading.Lock()

    def one(n: int) -> Tuple[RowRun, Dict[str, Optional[float]]]:
        i, rep = jobs[n]
        row = rows[i]
        if plan[n] is not None:
            run = _reused_run(target, plan[n])
        else:
            with with_states(states):
                run = run_row(target, row, case=keys[i], repeat=rep)
            if meter is not None:
                meter.step(run)
        vals: Dict[str, Optional[float]] = {}
        for m in row_metrics:
            try:
                vals[m.name] = m.score(target, row, run)
            except LoginRequired:
                raise
            except Exception as exc:  # noqa: BLE001 — a metric that fails on a row is that row's error
                vals[m.name] = None
                note = f"metric {m.name}: {type(exc).__name__}: {exc}"
                run.error = f"{run.error}; {note}" if run.error else note
        for c in crits:
            mine_kept: List[Dict[str, Any]] = []
            try:
                vals[c.name] = _judge_run(target, row, run, c, expected_cols, known, mine_kept)
            except LoginRequired:
                raise
            except Exception as exc:  # noqa: BLE001 — a judge that fails on a row is that row's error
                vals[c.name] = None
                note = f"criterion {c.name}: {type(exc).__name__}: {exc}"
                run.error = f"{run.error}; {note}" if run.error else note
            with kept_lock:
                kept.extend(mine_kept)
        return run, vals

    from .config import scoped
    logging_to = scoped(log_calls=store) if store is not None else contextlib.nullcontext()
    with logging_to, calllog.part_of("evaluation", run_id):   # logged calls say they answered known questions
        results = parallel(one, list(range(len(jobs))), max(1, num_threads))
        if kept:
            try:
                for r in kept:
                    calllog.write_rating(r, store)
            except ValueError:
                pass                                       # no call log here: the values stand, unkept
    runs = [r for r, _v in results]
    names = [m.name for m in row_metrics] + [c.name for c in crits]
    values: Dict[str, List[Optional[float]]] = {nm: [v.get(nm) for _r, v in results] for nm in names}
    for m in metrics:
        if m.expr is not None:
            values[m.name] = expr_scores(m, target, [rows[i] for i, _r in jobs], runs)
    all_metrics = [*metrics, *(Metric(c.name) for c in crits)]
    values = {m.name: values[m.name] for m in all_metrics}          # metric order
    kinds = {c.name: c.data() for c in crits}
    ev = Evaluation(target=target, rows=rows, runs=runs, metrics=all_metrics, values=values, run_id=run_id,
                    jobs=jobs, keys=keys, kinds=kinds)
    failed = ev.errors
    if max_errors is not None and len(failed) > max_errors:
        what = "rows" if repeats == 1 else "runs"
        raise RuntimeError(f"evaluation: {len(failed)} {what} failed; first (row {failed[0][0]}): {failed[0][1]}")
    if log is not None:
        os.makedirs(log, exist_ok=True)
        ev.write(os.path.join(os.fspath(log), f"{ev.run}.parquet"))
    return ev


# ------------------------------------------------------------------ comparing runs


def compare(before: Evaluation, after: Evaluation, *, margin: Optional[float] = None):
    '''Compare two evaluations case by case: is ``after`` better than ``before``?

    The two are paired on the cases both answered (by case key: the rows'
    inputs, whatever their order), which detects a real change with far
    fewer cases than two separate scores: a case both versions got right says
    nothing, a case only one got right says a lot. For right-or-wrong
    metrics the range is the MOVER-Wilson interval, which keeps its 95%
    where the paired t interval falls to a few percent (with no case that
    changed it says "exactly no difference"); otherwise the paired t interval.

    Parameters
    ----------
    before, after : Evaluation
        Two evaluations (typically two versions of a prompt, or two models)
        on the same cases, or overlapping ones.
    margin : float, optional
        How much worse is still acceptable: the outcome may then be
        ``same``, ``not_worse`` or ``not_better`` as well as ``better``,
        ``worse`` or ``unclear``.

    Returns
    -------
    dpyr dataframe
        One row per metric both have: ``before`` and ``after`` (the means on
        the shared cases), ``diff`` with its 95% interval ``low`` to
        ``high``, how many cases got ``better``, ``worse`` or stayed the
        ``same``, ``n``, ``method``, ``outcome``, ``fragility`` (the fewest
        cases whose removal would erase the difference), ``only_before`` and
        ``only_after`` (cases only one has).

    See Also
    --------
    evaluate : produces the evaluations.
    functai.compare_setups : the same from the call log.

    Examples
    --------
    ```python
    from typing import Literal

    rows = [
        {"message": "The mug arrived in pieces.", "result": "shipping"},
        {"message": "Box was crushed and the lamp inside is cracked.", "result": "shipping"},
        {"message": "I want my money back for the toaster.", "result": "billing"},
        {"message": "Please refund the blender, it stopped working.", "result": "billing"},
        {"message": "Toaster burns one side of the bread.", "result": "product"},
    ]

    @ai
    def category(message: str) -> Literal["shipping", "billing", "product"]:
        """The support category of the message."""
        ...

    @ai
    def category_v2(message: str) -> Literal["shipping", "billing", "product"]:
        """The support category of the message. An item that arrived broken is
        shipping; any request for money back is billing."""
        ...

    compare(evaluate(category, rows, num_threads=5), evaluate(category_v2, rows, num_threads=5))
    ```
    '''
    return _dpyr().read(comparison(before, after, margin=margin))


def comparison(before: Evaluation, after: Evaluation, *, margin: Optional[float] = None) -> List[Dict[str, Any]]:
    """``compare``'s rows, as dicts (no dpyr needed)."""
    from .measures import paired
    shared = [m for m in before.metrics if m in after.metrics]
    if not shared:
        raise ValueError(f"no metric in common: {before.metrics} vs {after.metrics}")
    records = []
    for name in shared:
        a, b = before.by_case(name), after.by_case(name)
        keys = sorted(k for k in a if k in b)
        if not keys:
            raise ValueError(f"compare needs cases both answered; {name} has none in common "
                             f"({len(a)} cases before, {len(b)} after, none with the same inputs)")
        only = (sum(1 for k in a if k not in b), sum(1 for k in b if k not in a))
        if any(only) and name == shared[0]:
            warnings.warn(f"[functai] compare: {len(keys)} cases both answered; {only[0]} only before, {only[1]} only "
                          "after are left out (a case is a row's inputs)", stacklevel=3)
        xs, ys = [a[k] for k in keys], [b[k] for k in keys]
        p = paired(xs, ys, margin, before._bounded(name) and after._bounded(name))
        records.append({"metric": name, "before": sum(xs) / len(xs), "after": sum(ys) / len(ys),
                        "diff": p["diff"], "low": p["low"], "high": p["high"], "better": p["better"],
                        "worse": p["worse"], "same": p["same"], "n": p["n"], "method": p["method"],
                        "outcome": p["outcome"], "fragility": p["fragility"],
                        "only_before": sum(1 for k in a if k not in b), "only_after": sum(1 for k in b if k not in a)})
    return records


def runs(folder: "str | os.PathLike"):
    """Every evaluation logged in a folder, as one table.

    Parameters
    ----------
    folder : str or path
        The folder given to ``evaluate(..., log=folder)``.

    Returns
    -------
    dpyr dataframe
        The rows of every run, with a ``run`` column. Runs with different
        columns line up by name; missing values are null.

    See Also
    --------
    evaluate : ``log=`` writes the runs.
    """
    dpyr = _dpyr()
    import duckdb
    pattern = os.path.join(os.fspath(folder), "*.parquet").replace("'", "''")
    if not any(f.endswith(".parquet") for f in os.listdir(folder)):
        raise ValueError(f"no runs in {folder}")
    arrow = duckdb.sql(f"SELECT * FROM read_parquet('{pattern}', union_by_name = true) "
                       f"ORDER BY run, example").to_arrow_table()
    return dpyr.read(arrow)


__all__ = ["Evaluation", "evaluate", "compare", "comparison", "runs", "exact_match", "rows_of", "interval"]
