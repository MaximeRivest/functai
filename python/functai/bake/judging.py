"""Judging a generative student: ``functai.evaluate`` on the function running
on the baked model, plus what only a student has (readability, speed, the
cost of a call next to the teacher's).

    report = functai.bake.judge(baked, summarize, test_rows, metric=my_judge)
    report = functai.bake.judge(baked, {extract: rows1, summarize: rows2})

``metric`` is anything ``evaluate`` takes (a function, an AI judge, a dict
per output). Without one, a function whose outputs are all finite (Literal,
Enum, bool) is scored by exact match; any other is not scored, because exact
match on open text measures nothing: the report then gives readability
(the share of replies its layout reads back into the output type) and
samples to read.
"""

from __future__ import annotations

import dataclasses
import statistics
import time
from typing import Any, Callable, Dict, List, Mapping, Optional

from .examples import BakeError


def _finite(fn) -> bool:
    from lmcc import core as lmcc_core
    spec = fn._spec()
    for f in spec.signature.outputs:
        if f.purpose in ("reasoning", "tools.calls"):
            continue
        base, _nullable = lmcc_core.nullable_base(f.shape)
        if "enum" not in base and base.get("type") != "boolean":
            return False
    return True


def _metric_name(metric: Any) -> str:
    name = getattr(metric, "__name__", None) or type(metric).__name__
    return "metric" if name in ("<lambda>", "function") else name


def _short(v: Any, n: int = 160) -> str:
    text = v if isinstance(v, str) else repr(v)
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1] + "…"


@dataclasses.dataclass
class FunctionScores:
    name: str
    rows: int
    readable: float                       # share of replies read back into the output type
    metric: Optional[str] = None
    score: Optional[float] = None
    interval: Optional[List[float]] = None
    teacher_score: Optional[float] = None
    teacher_interval: Optional[List[float]] = None
    samples: List[Dict[str, str]] = dataclasses.field(default_factory=list)
    first_error: Optional[str] = None
    groups: Dict[str, Dict[str, Any]] = dataclasses.field(default_factory=dict)   # by=: per value, rows and score


@dataclasses.dataclass
class StudentReport:
    """What a generative bake produced, measured (``baked.report``)."""
    name: str
    student: str
    parameters: Optional[int]
    where: str
    functions: List[FunctionScores]
    training: Dict[str, Any] = dataclasses.field(default_factory=dict)
    labeling: Dict[str, Any] = dataclasses.field(default_factory=dict)
    speed: Dict[str, Any] = dataclasses.field(default_factory=dict)
    notes: List[str] = dataclasses.field(default_factory=list)
    kind: str = "generative"

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StudentReport":
        d = dict(d)
        d["functions"] = [FunctionScores(**f) for f in d.get("functions", [])]
        return cls(**d)

    @property
    def score(self) -> Optional[float]:
        """The first function's score (None when it was not scored)."""
        return self.functions[0].score if self.functions else None

    def __repr__(self) -> str:
        from .report import _pct
        t = self.training
        from .planning import _count
        lines = [f"Baked {self.name}: {self.student}" + (f" ({_count(self.parameters)} parameters)"
                                                          if self.parameters else "") + (", trained here" if self.where == "here" else f", trained on {self.where}")]
        if t:
            parts = [f"{t.get('steps', '?'):,} steps" if isinstance(t.get("steps"), int) else ""]
            if t.get("seconds"):
                from .running import _duration
                parts.append(_duration(t["seconds"]))
            if t.get("train_loss") is not None:
                parts.append(f"final training loss {t['train_loss']:.4f}")
            if t.get("eval_loss") is not None:
                parts.append(f"validation loss {t['eval_loss']:.4f}" + (
                    f" (best {t['best_eval_loss']:.4f} at step {t.get('best_step')})"
                    if t.get("best_eval_loss") is not None and t.get("best_eval_loss") != t.get("eval_loss") else ""))
            lines.append("  training: " + ", ".join(p for p in parts if p))
        lab = self.labeling or {}
        if lab.get("rows"):
            cost = f", ${lab['dollars']:.2f}" if lab.get("dollars") is not None else ""
            lines.append(f"  teacher answers: {lab['rows']:,} rows from {lab.get('teacher')}{cost}"
                         + (f"; {lab['failed']:,} failed" if lab.get("failed") else ""))
        for f in self.functions:
            lines.append("")
            lines.append(f"  {f.name} on {f.rows:,} test rows")
            lines.append(f"    readable replies   {_pct(f.readable)}")
            if f.score is not None:
                ci = f" ({_pct(f.interval[0])}–{_pct(f.interval[1])})" if f.interval and f.interval[0] is not None \
                    else ""
                lines.append(f"    {f.metric or 'score':<19}{_pct(f.score)}{ci}"
                             + (f"   teacher {_pct(f.teacher_score)}" if f.teacher_score is not None else ""))
            else:
                lines.append("    not scored: pass metric= (a function or an AI judge) to measure open answers")
            for g, v in f.groups.items():
                lines.append(f"      {g:<17}{_pct(v.get('score'))} on {v['rows']:,} rows")
            if f.first_error:
                lines.append(f"    first unreadable reply: {_short(f.first_error, 120)}")
            for smp in f.samples[:3]:
                lines.append(f"    · in:  {smp['input']}")
                lines.append(f"      out: {smp['student']}")
                if smp.get("expected"):
                    lines.append(f"      was: {smp['expected']}")
        s = self.speed
        if s:
            lines += ["", f"  speed ({s.get('runner')}): {s.get('rows_per_second', 0):,.1f} rows/s with "
                          f"{s.get('concurrency')} at once; {s.get('latency_ms', float('nan')):,.0f} ms for one"]
        if self.notes:
            lines += [""] + [f"  note: {n}" for n in self.notes]
        return "\n".join(lines)

    __str__ = __repr__


def judge(baked, fn: Any, rows: Any = None, *, metric: Any = None, teacher: Any = None, compare_teacher: bool = False,
          by: Optional[str] = None, num_threads: int = 16, samples: int = 5, save: bool = False,
          labeling: Optional[Dict[str, Any]] = None, log: Callable[[str], None] = lambda s: None) -> StudentReport:
    """Measure ``baked`` on test rows (see the module docstring). ``fn``: an AI
    function and ``rows``, or ``{function: rows}``. ``compare_teacher``: run
    the teacher on the same rows too (it costs a teacher pass). ``by``: a
    column of the rows (a tag, a source) to score each of its values apart. ``save``:
    write the report into the model's ``baked.json``."""
    from ..evaluation import evaluate, interval, rows_of
    if baked.kind == "head":
        raise BakeError("a head model's report is made when it is baked (baked.report)")
    pairs = list(fn.items()) if isinstance(fn, Mapping) else [(fn, rows)]
    out: List[FunctionScores] = []
    notes: List[str] = []
    speed: Dict[str, Any] = {}
    for f, rs in pairs:
        rs = rows_of(rs)
        if not rs:
            continue
        student = f.using(lm=baked, retries=0)
        finite = _finite(f)
        metrics = metric if metric is not None else (None if finite else (lambda row, pred: 1.0))
        name = None if metric is None and not finite else ("exact match" if metric is None else
                                                           _metric_name(metric))
        log(f"judging {f.__name__} on {len(rs):,} test rows")
        t0 = time.time()
        ev = evaluate(student, rs, metrics, num_threads=num_threads)
        secs = time.time() - t0
        readable = 1 - len(ev.errors) / len(rs)
        fs = FunctionScores(name=f.__name__, rows=len(rs), readable=readable, metric=name,
                            first_error=ev.errors[0][1] if ev.errors else None)
        if metric is not None or finite:
            scores = ev.scores()
            mean, lo, hi = interval(scores)
            fs.score, fs.interval = mean, [lo, hi]
            if by:
                buckets: Dict[str, List[float]] = {}
                for row, v in zip(rs, scores):
                    buckets.setdefault(str(row.get(by)), []).append(v)
                for g, vs in sorted(buckets.items()):
                    m, glo, ghi = interval(vs)
                    fs.groups[g] = {"rows": len(vs), "score": m, "interval": [glo, ghi]}
            if compare_teacher:
                from .sources import teacher_for
                tfn = teacher_for(f, teacher)
                tev = evaluate(tfn, rs, metrics, num_threads=num_threads)
                tm, tl, th = interval(tev.scores())
                fs.teacher_score, fs.teacher_interval = tm, [tl, th]
        outs = [x.name for x in f._spec().signature.outputs if x.purpose == "plain"]
        ins = [n for n, _r in f._named_inputs()]
        for row, pred in list(zip(rs, ev.predictions))[:samples]:
            got = None if pred is None else (pred[outs[0]] if len(outs) == 1 else {k: pred[k] for k in outs if k in pred})
            want = (row.get(outs[0]) if len(outs) == 1 else {k: row.get(k) for k in outs}) if outs else None
            fs.samples.append({"input": _short({k: row.get(k) for k in ins if k in row}, 120),
                               "student": _short(got if pred is not None else "(unreadable)"),
                               "expected": _short(want) if want is not None else ""})
        if not speed:
            speed = {"runner": baked.runner.name, "rows_per_second": len(rs) / max(secs, 1e-9),
                     "concurrency": num_threads}
            lat = []
            for row in rs[:5]:
                t1 = time.perf_counter()
                try:
                    student._invoke((), {k: row[k] for k in ins if k in row}, full=True)
                except Exception:  # noqa: BLE001 — an unreadable reply still took its time
                    pass
                lat.append((time.perf_counter() - t1) * 1000)
            speed["latency_ms"] = statistics.median(lat) if lat else float("nan")
            if baked.runner.fidelity != "tokens":
                notes.append("the server renders the chat template itself: tokens may differ from training")
        out.append(fs)
        if readable < 0.9:
            notes.append(f"{f.__name__}: {1 - readable:.0%} of replies could not be read in its layout; a longer "
                         f"training run, a larger student, or a simpler layout usually helps")
    report = StudentReport(name=baked.name, student=baked.student, parameters=baked.meta.get("parameters"),
                           where=(baked.meta.get("run") or {}).get("where", "?"), functions=out,
                           training=baked.meta.get("training") or {}, labeling=labeling or {}, speed=speed,
                           notes=notes)
    if save:
        from .baked import write_meta
        meta = {k: v for k, v in baked.meta.items() if k not in ("hashes", "sizes", "functai_baked")}
        meta["report"] = report.to_dict()
        write_meta(baked.path, meta)
        baked.meta = __import__("json").loads((baked.path / "baked.json").read_text())
    baked._report = report
    return report


__all__ = ["judge", "StudentReport", "FunctionScores"]
