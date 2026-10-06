"""Grade the `forms` task: fill a codebook's form for 10 study descriptions.

Checks
- files:     extract.py, out/designs.csv and out/analyses.csv exist with the asked columns
- rerun:     running extract.py again sends nothing to the provider (replies kept)
- guidance:  the requests the model receives contain the codebook's conventions
             (read from the recording proxy: exactly what was sent, whatever the library)
- design, units, methods (F1), pseudoreplicated, outcome: agreement with the answer key
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import time

from common import check, csv_rows, norm, recorded, run_script, summarize

HERE = Path(__file__).parent
TRUTH = json.loads((HERE / "truth.json").read_text())

# The codebook's conventions; each must reach the model in some wording.
# Each pattern is absent from papers.csv, so only the codebook can bring it.
CONVENTIONS = {
    "units counted per treatment group": r"per (treatment )?group",
    "outcome in lowercase, short": r"lower ?case",
    "Spearman/Kendall are rank tests": r"kendall",
    "correlation means Pearson only": r"pearson",
    "pseudoreplication: units averaged first are fine": r"averag",
    "randomized includes random numbers": r"random numbers",
    "rank tests listed (Kruskal-Wallis)": r"kruskal",
}


def _float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _bool(v):
    return norm(v) in ("true", "1", "yes", "t")


def grade(work: Path, env: dict, python: str, out: Path) -> dict:
    c = {}
    designs = csv_rows(work / "out" / "designs.csv")
    analyses = csv_rows(work / "out" / "analyses.csv")
    files_ok = ((work / "extract.py").exists() and designs is not None and analyses is not None
                and {"paper_id", "design", "independent_units"} <= set(designs[0] if designs else {})
                and {"paper_id", "outcome", "method", "pseudoreplicated"} <= set(analyses[0] if analyses else {}))
    c["files"] = check(files_ok, detail="extract.py + out/designs.csv + out/analyses.csv with the asked columns")
    if not files_ok:
        for k in ("rerun", "guidance", "design", "units", "methods", "pseudoreplicated", "outcome"):
            c[k] = check(False, 0.0, "no usable output")
        return summarize(c, required=list(c))

    # re-run: free, and the proxy shows exactly what the model is sent
    record = env.get("BENCH_RECORD")
    t0 = time.time()
    r = run_script(work, env, python, "extract.py", out / "grade_calls")
    rerun = recorded(record, since=t0)
    c["rerun"] = check(r["code"] == 0 and not rerun and (work / "out" / "designs.csv").exists(), None,
                       f"exit {r['code']}; {len(rerun)} requests reached the provider")
    requests = [x["text"] for x in rerun]
    source = "the re-run"
    if not requests:            # replies kept: the latest request the agent's runs sent for each paper
        source = "the agent's latest requests"
        papers = [row["methods"][:80] for row in csv_rows(HERE / "files" / "papers.csv")]
        latest = {}
        for x in recorded(record):
            for i, paper in enumerate(papers):
                if paper in x["text"]:
                    latest[i] = x["text"]
        requests = list(latest.values())
    texts = "\n".join(requests).lower()
    found = {name: bool(re.search(rx, texts)) for name, rx in CONVENTIONS.items()}
    share = sum(found.values()) / len(found)
    c["guidance"] = check(share >= 0.85 and bool(requests), round(share, 2),
                          f"from {source}: " + ("missing: " + ", ".join(k for k, v in found.items() if not v)
                                                if share < 1 else "all reached the model"))

    # agreement with the answer key
    by_id = {d["paper_id"]: d for d in designs}
    design_ok = [norm(by_id.get(p, {}).get("design")) == t["design"] for p, t in TRUTH.items()]
    units_ok = []
    for p, t in TRUTH.items():
        got = _float(by_id.get(p, {}).get("independent_units"))
        units_ok.append((got is None and t["independent_units"] is None) or (got is not None and got == t["independent_units"]))
    c["design"] = check(sum(design_ok) / 10 >= 0.9, sum(design_ok) / 10)
    c["units"] = check(sum(units_ok) / 10 >= 0.8, sum(units_ok) / 10,
                       "wrong: " + ", ".join(p for p, ok in zip(TRUTH, units_ok) if not ok))

    tp = fp = fn = 0
    pseudo, outcome = [], []
    for p, t in TRUTH.items():
        pred = [a for a in analyses if a["paper_id"] == p]
        used = set()
        for a in t["analyses"]:
            match = next((i for i, x in enumerate(pred) if i not in used and norm(x["method"]) == a["method"]), None)
            if match is None:
                fn += 1
                continue
            used.add(match)
            tp += 1
            pseudo.append(_bool(pred[match]["pseudoreplicated"]) == a["pseudoreplicated"])
            outcome.append(norm(pred[match]["outcome"]) in [norm(o) for o in a["outcome"]])
        fp += len(pred) - len(used)
    f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    c["methods"] = check(f1 >= 0.85, round(f1, 2), f"{tp} right, {fp} extra, {fn} missed")
    c["pseudoreplicated"] = check(bool(pseudo) and sum(pseudo) / len(pseudo) >= 0.9,
                                  round(sum(pseudo) / max(1, len(pseudo)), 2))
    c["outcome"] = check(bool(outcome) and sum(outcome) / len(outcome) >= 0.7,
                         round(sum(outcome) / max(1, len(outcome)), 2), "lowercase short names, as the codebook asks")
    return summarize(c, required=["files", "rerun", "guidance", "units", "methods"])
