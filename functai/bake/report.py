"""What baking produced, measured: ``baked.report``."""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Dict, List, Optional

from . import metrics


def _pct(x: Optional[float], digits: int = 1) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.{digits}f}%"


def _num(x: Optional[float], fmt: str = ",.0f") -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else format(x, fmt)


@dataclasses.dataclass
class FieldScores:
    name: str
    accuracy: float
    interval: List[float]
    top3: Optional[float]
    ece: float
    ece_raw: float
    nll: float
    temperature: float
    teacher_accuracy: Optional[float] = None
    agreement: Optional[float] = None
    worst: List[Dict[str, Any]] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class BakeReport:
    function: str
    student: str
    parameters: int
    device: str
    truth: str                       # "labeled" (test rows with their labels) | "teacher" (agreement only)
    label_source: str                # "the data's labels" | "teacher (soft)" | "teacher (hard)" | mixed
    teacher: Optional[str]
    rows: Dict[str, int]             # train, validation, test, dropped
    training: Dict[str, Any]
    fields: List[FieldScores]
    coverage: List[Dict[str, float]]
    confidence: List[float] = dataclasses.field(default_factory=list, repr=False)
    correct: List[bool] = dataclasses.field(default_factory=list, repr=False)
    speed: Dict[str, float] = dataclasses.field(default_factory=dict)
    labeling: Dict[str, Any] = dataclasses.field(default_factory=dict)
    breakeven: Dict[str, Any] = dataclasses.field(default_factory=dict)
    notes: List[str] = dataclasses.field(default_factory=list)
    max_length: int = 0
    truncated: int = 0

    # ---- use

    @property
    def accuracy(self) -> float:
        """Share of test rows with every output right."""
        return sum(self.correct) / len(self.correct) if self.correct else float("nan")

    def threshold(self, accuracy: float = 0.95, min_rows: int = 20) -> Optional[Dict[str, float]]:
        """The confidence cut at which the model's own answers reach ``accuracy`` on
        the test rows: ``{"threshold", "share", "accuracy"}`` (use it as
        ``escalate_below``), or None when no cut reaches it."""
        return metrics.threshold_for(self.confidence, self.correct, accuracy, min_rows)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BakeReport":
        d = dict(d)
        d["fields"] = [FieldScores(**f) for f in d.get("fields", [])]
        return cls(**d)

    # ---- text

    def __repr__(self) -> str:
        r = self.rows
        truth = "labeled" if self.truth == "labeled" else "teacher-labeled (no labeled rows)"
        lines = [
            f"Baked {self.function}: {self.student} ({self.parameters / 1e6:,.1f}M parameters)",
            f"  trained on {r.get('train', 0):,} rows ({self.label_source}), validated on {r.get('validation', 0):,}, "
            f"tested on {r.get('test', 0):,} {truth} rows",
        ]
        t = self.training
        if t:
            lines.append(f"  training: {t.get('passes_run', '?')} passes (best {t.get('best_epoch', '?')}), "
                         f"{t.get('seconds', 0):.0f} s on {self.device} ({t.get('mixed_precision', 'fp32')}), "
                         f"inputs up to {self.max_length} tokens")
        lines.append("")
        head = "accuracy" if self.truth == "labeled" else "agrees with the teacher"
        teacher_col = self.teacher and any(f.teacher_accuracy is not None for f in self.fields)
        lines.append(f"  {'on the test rows':<28}{'student':<24}" + (f"teacher ({self.teacher})" if teacher_col else ""))
        for f in self.fields:
            label = f"{head}" + (f" · {f.name}" if len(self.fields) > 1 else "")
            ci = f" ({_pct(f.interval[0])}–{_pct(f.interval[1])})" if f.interval else ""
            lines.append(f"  {label:<28}{_pct(f.accuracy) + ci:<24}" +
                         (_pct(f.teacher_accuracy) if f.teacher_accuracy is not None else ""))
            if f.top3 is not None:
                lines.append(f"  {'top-3':<28}{_pct(f.top3)}")
            lines.append(f"  {'calibration error (ECE)':<28}{f.ece:.3f} (was {f.ece_raw:.3f}; temperature "
                         f"{f.temperature:.2f})")
            if f.agreement is not None:
                lines.append(f"  {'agrees with the teacher':<28}{_pct(f.agreement)}")
        if self.coverage:
            lines += ["", "  answering only when sure:  most confident share → accuracy (confidence at the cut)"]
            for c in self.coverage:
                lines.append(f"    {_pct(c['share'], 0):>5} → {_pct(c['accuracy'])}  (≥ {c['threshold']:.2f})")
            best = self.threshold(0.95)
            if best:
                lines.append(f"    for 95% accuracy: escalate_below={best['threshold']:.2f} keeps "
                             f"{_pct(best['share'], 0)} of rows")
        s = self.speed
        if s:
            lines += ["", f"  speed on {s.get('device', self.device)}: {_num(s.get('rows_per_second'))} rows/s batched "
                          f"(tokenizing included), {s.get('latency_ms', float('nan')):.1f} ms for one row"]
        lab = self.labeling
        if lab.get("rows"):
            cost = f", ${lab['dollars']:.2f}" if lab.get("dollars") is not None else ""
            lines.append(f"  teacher labels: {lab['rows']:,} rows from {lab.get('teacher')} in {lab.get('seconds', 0):.0f} s"
                         f" ({lab.get('tokens_per_row', 0):,.0f} tokens a row{cost}"
                         + (f"; {lab['failed']} failed" if lab.get("failed") else "") + ")")
        b = self.breakeven
        if b.get("rows_time") is not None:
            lines.append(f"  break-even in time against the teacher: after {_num(b['rows_time'])} rows")
        if b.get("rows_money") is not None:
            lines.append(f"  break-even in money: after {_num(b['rows_money'])} rows")
        if self.notes:
            lines += [""] + [f"  note: {n}" for n in self.notes]
        return "\n".join(lines)

    __str__ = __repr__
