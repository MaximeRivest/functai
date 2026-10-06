"""Grade the `photos` task: name the shape and colour in 12 pictures.

Checks
- files:   predictions.csv with file, shape, color for the 12 images
- images:  the model received each picture as an image part (functai's call log of
           the agent's run), not as base64 text pasted into the prompt
- shape, color: agreement with the answer key (should be near perfect when the
           model really sees the picture)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from common import all_records, check, csv_rows, norm, request_parts, summarize

HERE = Path(__file__).parent
TRUTH = json.loads((HERE / "truth.json").read_text())


def grade(work: Path, env: dict, python: str, out: Path) -> dict:
    c = {}
    pred = csv_rows(work / "predictions.csv")
    ok = pred is not None and {"file", "shape", "color"} <= set(pred[0] if pred else {})
    by = {Path(r["file"]).name: r for r in pred} if ok else {}
    ok = ok and set(TRUTH) <= set(by)
    c["files"] = check(ok, detail="predictions.csv with file, shape, color for every image")

    parts = request_parts(all_records(env, work))
    with_image = sum(any(p.get("type") == "image" for p in ps) for ps in parts)
    pasted = sum(any(p.get("type") == "text" and re.search(r"[A-Za-z0-9+/]{400,}", p.get("text", "")) for p in ps)
                 for ps in parts)
    c["images"] = check(with_image >= 12 and pasted == 0, None,
                        f"{with_image} functai requests carried an image part; {pasted} pasted base64 as text"
                        + ("; an image added outside functai's request is invisible to its log, cache key and replay"
                           if with_image == 0 and pasted == 0 else ""))
    if not ok:
        c["shape"] = check(False, 0.0, "no usable output")
        c["color"] = check(False, 0.0, "no usable output")
        return summarize(c, required=list(c))
    shape = sum(norm(by[f]["shape"]) == t["shape"] for f, t in TRUTH.items()) / len(TRUTH)
    color = sum(norm(by[f]["color"]) == t["color"] for f, t in TRUTH.items()) / len(TRUTH)
    c["shape"] = check(shape >= 11 / 12, round(shape, 2))
    c["color"] = check(color >= 11 / 12, round(color, 2))
    return summarize(c, required=list(c))
