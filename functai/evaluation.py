"""Evaluation: run a program over a dataset, get a table back.

Data goes in as rows: a list of dicts (one per row, values kept as the
Python objects they are), or anything ``dpyr.read()`` takes: a parquet/CSV/
JSON path, a pandas/polars/arrow table, a Hugging Face dataset, a dpyr
dataframe. Columns named like the function's parameters are its inputs; the
other columns are the expected outputs and anything else you want to keep.

Results come out as one row per example (``Evaluation.table``, a dpyr
dataframe): the data, the predictions (``pred_<output>``), one column per
metric, and ``error``, ``seconds``, ``input_tokens``, ``output_tokens``,
``model``, ``run``. Tables need dpyr (``pip install "functai[data]"``); the
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
import time
import warnings
from collections.abc import Mapping
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

from .core import _STATE_OVERRIDE, _TRACE, FunctAIFunc, ProgramState
from .data import Prediction
from .engine import LoginRequired

States = Dict[FunctAIFunc, ProgramState]

_DPYR_HINT = 'tables need dpyr: pip install "functai[data]"'
# columns every run table has besides the data, the predictions and the metrics
_RUN_COLUMNS = ("error", "seconds", "input_tokens", "output_tokens", "model", "run")


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
        if isinstance(program, FunctAIFunc):
            self.predictors: List[FunctAIFunc] = [program]
            params = program._sig.parameters
            self.single = True
        elif isinstance(program, FunctAIModule):
            self.predictors = program.ai_functions()
            params = inspect.signature(program._fn).parameters
            self.single = False
            self.call_defaults = {**program._opt_call_defaults, **self.call_defaults}
            if not self.predictors:
                raise ValueError(f"@module {program.__name__} calls no @ai function")
        else:
            raise TypeError(f"expected an @ai function or a @module, not {type(program).__name__}")
        self.name = program.__name__
        kinds = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        self.input_names = [n for n, p in params.items() if p.kind not in kinds]
        self.required = [n for n, p in params.items()
                         if p.kind not in kinds and p.default is inspect.Parameter.empty
                         and n not in self.call_defaults]

    @property
    def output_names(self) -> List[str]:
        if not self.single:
            return ["result"]
        return list(self.predictors[0]._spec().outputs)

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
        inputs = self.inputs_of(row)
        if self.single:
            return self.program(**inputs, all=True)
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
    """One row run: the prediction (None when the program raised), the error,
    and what it cost."""
    pred: Optional[Prediction]
    error: Optional[str]
    seconds: float
    trace: List[Tuple[FunctAIFunc, Prediction]]

    @property
    def usage(self) -> Dict[str, int]:
        total: Dict[str, int] = {}
        for _fn, p in self.trace:
            for k, v in p.usage.items():
                total[k] = total.get(k, 0) + v
        return total

    @property
    def model(self) -> Optional[str]:
        names = sorted({r.model for _fn, p in self.trace for r in p.responses if getattr(r, "model", None)})
        return ", ".join(names) or None


def run_row(target: _Target, row: Mapping[str, Any]) -> RowRun:
    """Run the program on one row. A failure is recorded, not raised, except a
    missing login, which would fail every row the same way."""
    start = time.perf_counter()
    with tracing() as trace:
        try:
            pred, error = target.run(row), None
        except LoginRequired:
            raise
        except Exception as exc:  # noqa: BLE001 — recorded in the row's error
            pred, error = None, f"{type(exc).__name__}: {exc}"
    return RowRun(pred, error, time.perf_counter() - start, list(trace))


# ------------------------------------------------------------------ metrics


def _norm(v: Any) -> Any:
    if isinstance(v, str):
        return " ".join(v.split()).casefold()
    if isinstance(v, enum.Enum):
        return _norm(v.value)
    return v


def exact_match(row: Mapping[str, Any], pred: Mapping[str, Any]) -> float:
    """1.0 when every output the data has a column for equals the prediction's
    (strings compared ignoring case and repeated whitespace), else 0.0."""
    keys = [k for k in pred if k in row]
    if not keys:
        raise ValueError(f"exact_match: the data has no column for any output ({list(pred)})")
    return float(all(_norm(row[k]) == _norm(pred[k]) for k in keys))


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
            return as_score(self.fn(dict(row), run.pred))
        return expr_scores(self, target, [row], [run])[0]


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


def resolve_metrics(metric: Any, target: _Target, rows: Sequence[Mapping[str, Any]]) -> List[Metric]:
    """``metric`` as a list of named metrics. None: exact match when the data
    has a column for an output, else no metric. Accepts one metric, a list,
    or a dict of name → metric; a metric is a callable or a dpyr expression."""
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


def _records(target: _Target, rows: Sequence[Mapping[str, Any]], runs: Sequence[RowRun],
             metrics: Sequence[Metric], values: Mapping[str, Sequence[Optional[float]]],
             run_id: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    data_cols = list(dict.fromkeys(k for r in rows for k in r))
    outs = list(target.output_names)
    for run in runs:
        for k in (run.pred or {}):
            if k not in outs:
                outs.append(k)
    pred_cols = [f"pred_{k}" for k in outs]
    names = ["example", *data_cols, *pred_cols, *(m.name for m in metrics), *_RUN_COLUMNS]
    records = []
    for i, (row, run) in enumerate(zip(rows, runs)):
        usage = run.usage
        rec = {"example": i, **{k: _cell(row.get(k)) for k in data_cols}}
        rec.update({f"pred_{k}": _cell(run.pred.get(k)) if run.pred is not None else None for k in outs})
        rec.update({m.name: values[m.name][i] for m in metrics})
        rec.update(error=run.error, seconds=run.seconds, input_tokens=usage.get("input_tokens"),
                   output_tokens=usage.get("output_tokens"), model=run.model, run=run_id)
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
    """What ``evaluate`` returns.

    - ``score``: the first metric's mean, 0 to 1 (a failed row counts 0)
    - ``summary``: one row per metric, with a 95% interval (a dpyr dataframe)
    - ``table``: one row per example (a dpyr dataframe)
    - ``predictions``: the Prediction of each row (None where it failed),
      with its turns, tokens and repairs
    - ``errors``: ``(example, message)`` for each row that failed
    """

    def __init__(self, *, target: _Target, rows: List[Dict[str, Any]], runs: List[RowRun],
                 metrics: List[Metric], values: Dict[str, List[Optional[float]]], run_id: str):
        self.program = target.name
        self.run = run_id
        self.metrics = [m.name for m in metrics]
        self.predictions = [r.pred for r in runs]
        self._target, self._rows, self._runs, self._metrics, self._values = target, rows, runs, metrics, values
        self._table = None

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def errors(self) -> List[Tuple[int, str]]:
        return [(i, r.error) for i, r in enumerate(self._runs) if r.error]

    def scores(self, metric: Optional[str] = None) -> List[float]:
        """One metric's value per row, a failed row counting 0."""
        if not self.metrics:
            raise ValueError("this evaluation has no metric")
        name = metric or self.metrics[0]
        return [0.0 if v is None else v for v in self._values[name]]

    @property
    def score(self) -> Optional[float]:
        return interval(self.scores())[0] if self.metrics and self._rows else None

    def __float__(self) -> float:
        if self.score is None:
            raise ValueError("this evaluation has no metric")
        return self.score

    def _summary_records(self) -> List[Dict[str, Any]]:
        out = []
        n_err = len(self.errors)
        for name in self.metrics:
            mean, low, high = interval(self.scores(name))
            failed = sum(v is None for v in self._values[name])
            out.append({"metric": name, "mean": mean, "low": low, "high": high, "n": len(self._rows),
                        "failed": failed})
        if not out:
            out.append({"metric": None, "mean": None, "low": None, "high": None, "n": len(self._rows),
                        "failed": n_err})
        return out

    @property
    def summary(self):
        """One row per metric: mean, 95% interval (low, high), n, and how many
        rows failed (counted as 0 in the mean)."""
        return _dpyr().read(self._summary_records())

    @property
    def table(self):
        """One row per example: ``example`` (the row's position in the data),
        the data's columns, ``pred_<output>`` for each output, one column per
        metric (null where the row failed), then ``error``, ``seconds``,
        ``input_tokens``, ``output_tokens``, ``model`` and ``run``."""
        if self._table is None:
            names, records = _records(self._target, self._rows, self._runs, self._metrics, self._values,
                                      self.run)
            self._table = _dpyr().read(_tabular(names, records))
        return self._table

    def write(self, path: "str | os.PathLike") -> None:
        """Save the table (``.parquet`` keeps every type; also .jsonl, .arrow, ...)."""
        self.table.write(os.fspath(path))

    def __repr__(self) -> str:
        n = len(self._rows)
        head = f"Evaluation({self.program}, {n} example{'s' if n != 1 else ''}"
        parts = []
        for rec in self._summary_records():
            if rec["metric"] is None:
                continue
            ci = f" [{_fmt(rec['low'])}, {_fmt(rec['high'])}]" if rec["low"] is not None else ""
            parts.append(f"{rec['metric']} {_fmt(rec['mean'])}{ci}")
        n_err = len(self.errors)
        tail = f", {n_err} failed" if n_err else ""
        return head + (": " + "; ".join(parts) if parts else "") + tail + ")"


# ------------------------------------------------------------------ evaluate


def _new_run_id(name: str) -> str:
    return f"{name}-{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(2)}"


def _check_names(target: _Target, rows: Sequence[Mapping[str, Any]], metrics: Sequence[Metric]) -> None:
    columns = set(k for r in rows for k in r)
    added = {"example", *_RUN_COLUMNS, *(m.name for m in metrics),
             *(f"pred_{k}" for k in target.output_names)}
    clash = sorted(columns & added)
    if clash:
        raise ValueError(f"the data has column(s) {clash}, which the run table adds; rename them first")


def evaluate(program: Any, data: Any, metric: Any = None, *, num_threads: int = 1,
             max_errors: Optional[int] = None, log: "str | os.PathLike | None" = None,
             call_defaults: Optional[Dict[str, Any]] = None, states: Optional[States] = None) -> Evaluation:
    """Run ``program`` (an @ai function or a @module) on every row of ``data``
    and score it.

    - ``data``: a list of dicts, or anything ``dpyr.read()`` takes
    - ``metric``: ``metric(row, prediction) -> float | bool`` (it can be an
      @ai judge), a dpyr expression over the run table
      (``col.pred_result == col.result``), a list of these, or a dict
      name → metric. Default: exact match when the data has a column for an
      output
    - ``max_errors``: raise when more rows than this fail
    - ``log``: a folder; the run's table is written there as ``<run>.parquet``
      (``functai.runs(folder)`` reads every run back as one table)
    - ``call_defaults``: arguments for a @module that the rows lack
    """
    target = _Target(program, call_defaults=call_defaults)
    rows = rows_of(data)
    target.check(rows)
    metrics = resolve_metrics(metric, target, rows)
    _check_names(target, rows, metrics)
    if log is not None or any(m.expr is not None for m in metrics):
        _dpyr()                                            # fail now, not after the model calls
    row_metrics = [m for m in metrics if m.fn is not None]

    def one(row: Dict[str, Any]) -> Tuple[RowRun, Dict[str, Optional[float]]]:
        with with_states(states):
            run = run_row(target, row)
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
        return run, vals

    results = parallel(one, rows, max(1, num_threads))
    runs = [r for r, _v in results]
    values: Dict[str, List[Optional[float]]] = {m.name: [v[m.name] for _r, v in results] for m in row_metrics}
    for m in metrics:
        if m.expr is not None:
            values[m.name] = expr_scores(m, target, rows, runs)
    values = {m.name: values[m.name] for m in metrics}          # metric order
    ev = Evaluation(target=target, rows=rows, runs=runs, metrics=metrics, values=values,
                    run_id=_new_run_id(target.name))
    failed = ev.errors
    if max_errors is not None and len(failed) > max_errors:
        raise RuntimeError(f"evaluation: {len(failed)} rows failed; first (row {failed[0][0]}): {failed[0][1]}")
    if log is not None:
        os.makedirs(log, exist_ok=True)
        ev.write(os.path.join(os.fspath(log), f"{ev.run}.parquet"))
    return ev


# ------------------------------------------------------------------ comparing runs


def compare(before: Evaluation, after: Evaluation):
    """Two evaluations of the same examples, compared example by example.

    One row per metric both have: the two means, ``diff`` (after − before)
    with its 95% interval (paired t, so a real change shows up with far fewer
    examples than two separate intervals would need), and how many examples
    got ``better``, ``worse`` or stayed the ``same``. If the interval of
    ``diff`` excludes 0, the change is unlikely to be luck."""
    if len(before) != len(after):
        raise ValueError(f"compare needs the same examples: {len(before)} rows vs {len(after)}")
    for i, (a, b) in enumerate(zip(before._rows, after._rows)):
        if a != b:
            raise ValueError(f"compare needs the same examples in the same order; row {i} differs")
    shared = [m for m in before.metrics if m in after.metrics]
    if not shared:
        raise ValueError(f"no metric in common: {before.metrics} vs {after.metrics}")
    records = []
    for name in shared:
        a, b = before.scores(name), after.scores(name)
        d = [y - x for x, y in zip(a, b)]
        diff, low, high = _paired_t(d) if len(d) >= 2 else (d[0], None, None)
        records.append({"metric": name, "before": sum(a) / len(a), "after": sum(b) / len(b),
                        "diff": diff, "low": low, "high": high,
                        "better": sum(v > 0 for v in d), "worse": sum(v < 0 for v in d),
                        "same": sum(v == 0 for v in d), "n": len(d)})
    return _dpyr().read(records)


def _paired_t(d: Sequence[float]) -> Tuple[float, Optional[float], Optional[float]]:
    """Mean difference with a paired t interval (differences of 0/1 scores
    are -1, 0 or 1, so Wilson's interval does not apply)."""
    n = len(d)
    mean = sum(d) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in d) / (n - 1))
    half = _t975(n - 1) * sd / math.sqrt(n)
    return mean, mean - half, mean + half


def runs(folder: "str | os.PathLike"):
    """Every run logged with ``evaluate(..., log=folder)``, as one table.
    Runs with different columns line up by name (missing values are null)."""
    dpyr = _dpyr()
    import duckdb
    pattern = os.path.join(os.fspath(folder), "*.parquet").replace("'", "''")
    if not any(f.endswith(".parquet") for f in os.listdir(folder)):
        raise ValueError(f"no runs in {folder}")
    arrow = duckdb.sql(f"SELECT * FROM read_parquet('{pattern}', union_by_name = true) "
                       f"ORDER BY run, example").to_arrow_table()
    return dpyr.read(arrow)


__all__ = ["Evaluation", "evaluate", "compare", "runs", "exact_match", "rows_of", "interval"]
