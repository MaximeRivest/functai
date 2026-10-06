"""Grade the `measure` task: score a classifier honestly, then judge a change.

The grader recomputes everything from predictions.csv and the answer key, and
checks results.json against it:
- files:      predictions.csv (id, category, v1, v2) for all 80 tickets; results.json with the asked keys
- real:       the predictions came from model calls (functai's call log of the agent's run)
- accuracy:   v1_accuracy and v2_accuracy match the predictions
- interval:   v1_low/v1_high is a 95% interval for a proportion (Wilson's, or close to it)
- paired:     diff_low/diff_high is the PAIRED 95% interval of the per-ticket difference
- verdict:    "unclear" exactly when that interval includes 0, else "better"/"worse"
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from common import all_records, check, csv_rows, norm, request_texts, summarize

HERE = Path(__file__).parent
KEYS = ["v1_accuracy", "v1_low", "v1_high", "v2_accuracy", "diff", "diff_low", "diff_high", "verdict"]


def wilson(k: int, n: int, z: float = 1.959964):
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def paired(d: list[float]):
    """Mean difference and its paired 95% interval: Student's t on the per-ticket differences,
    as functai.compare computes it."""
    from functai.evaluation import _t975
    n = len(d)
    mean = sum(d) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in d) / (n - 1))
    half = _t975(n - 1) * sd / math.sqrt(n)
    return mean, mean - half, mean + half


def grade(work: Path, env: dict, python: str, out: Path) -> dict:
    c = {}
    truth = {r["id"]: r["category"] for r in csv.DictReader(open(HERE / "files" / "tickets.csv"))}
    pred = csv_rows(work / "predictions.csv")
    try:
        res = json.loads((work / "results.json").read_text())
    except Exception:  # noqa: BLE001
        res = None
    ok = (pred is not None and res is not None and {"id", "v1", "v2"} <= set(pred[0] if pred else {})
          and all(k in res for k in KEYS) and len({r["id"] for r in pred}) == len(truth))
    c["files"] = check(ok, detail="predictions.csv with id, v1, v2 for all 80 tickets; results.json with " + ", ".join(KEYS))
    if not ok:
        for k in ("real", "accuracy", "interval", "paired", "verdict"):
            c[k] = check(False, 0.0, "no usable output")
        return summarize(c, required=list(c))

    texts = "\n".join(request_texts(all_records(env, work)))
    seen = sum(1 for r in csv.DictReader(open(HERE / "files" / "tickets.csv")) if r["message"][:60] in texts)
    c["real"] = check(seen >= 76, seen, f"{seen} of 80 tickets reached a model through functai")

    v1 = [norm(r["v1"]) == truth[r["id"]] for r in pred]
    v2 = [norm(r["v2"]) == truth[r["id"]] for r in pred]
    n = len(v1)
    a1, a2 = sum(v1) / n, sum(v2) / n
    f = lambda k: float(res[k]) / (100 if float(res[k]) > 1.0 else 1)   # accept 0.85 or 85
    c["accuracy"] = check(abs(f("v1_accuracy") - a1) < 0.006 and abs(f("v2_accuracy") - a2) < 0.006,
                          None, f"recomputed v1 {a1:.3f}, v2 {a2:.3f}; reported {res['v1_accuracy']}, {res['v2_accuracy']}")
    lo, hi = wilson(sum(v1), n)
    c["interval"] = check(abs(f("v1_low") - lo) < 0.03 and abs(f("v1_high") - hi) < 0.03, None,
                          f"Wilson 95%: {lo:.3f}-{hi:.3f}; reported {res['v1_low']}-{res['v1_high']}")
    d = [float(b) - float(a) for a, b in zip(v1, v2)]
    mean, dlo, dhi = paired(d)
    rlo, rhi = f("diff_low"), f("diff_high")
    near = lambda lo, hi: abs(rlo - lo) < 0.02 and abs(rhi - hi) < 0.02
    cautious = rlo <= dlo + 0.005 and rhi >= dhi - 0.005 and (rhi - rlo) <= 2 * (dhi - dlo)
    how = ("the paired t interval (what functai.compare gives)" if near(dlo, dhi)
           else "a cautious paired interval (contains the t interval)" if cautious else None)
    c["paired"] = check(how is not None, None,
                        f"paired t 95%: {dlo:+.3f} to {dhi:+.3f}; reported {rlo:+.3f} to {rhi:+.3f}"
                        + (f": {how}" if how else ": not a paired interval of these predictions"))
    # the verdict must follow from the interval given (or, when that is not acceptable, the t interval)
    lo_, hi_ = (rlo, rhi) if how else (dlo, dhi)
    right = "unclear" if lo_ <= 0 <= hi_ else ("better" if mean > 0 else "worse")
    c["verdict"] = check(norm(res["verdict"]) == right, None, f"should be {right!r}; said {res['verdict']!r}")
    return summarize(c, required=list(c))
