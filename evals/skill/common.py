"""Helpers shared by the graders: run a script the way the agent left it, read
functai's call log, and build a check."""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path


def check(passed: bool, value=None, detail: str = "") -> dict:
    return {"pass": bool(passed), "value": value, "detail": detail}


def run_script(work: Path, env: dict, python: str, script: str, calls: Path, timeout: int = 600) -> dict:
    """Run `python script` in the workspace with its own call-log folder; keep the output."""
    calls.mkdir(parents=True, exist_ok=True)
    e = {**env, "FUNCTAI_LOG_CALLS": str(calls)}
    try:
        p = subprocess.run([python, script], cwd=work, env=e, capture_output=True, text=True, timeout=timeout)
        return {"code": p.returncode, "out": p.stdout[-3000:], "err": p.stderr[-3000:]}
    except subprocess.TimeoutExpired:
        return {"code": "timeout", "out": "", "err": ""}


def call_log(folder: Path) -> list[dict]:
    """Every call record functai wrote under `folder` (one JSON object per line)."""
    records = []
    if not folder.exists():
        return records
    for f in sorted(folder.rglob("*.jsonl")):
        for line in f.read_text().splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def all_records(env: dict, work: Path, since: float | None = None, extra: Path | None = None) -> list[dict]:
    """Every functai call record this task produced, wherever it was written: the
    folder $FUNCTAI_LOG_CALLS named, `extra`, and any call log the agent's own
    script chose with `log_calls=` inside the workspace. `since`: only calls that
    started after that time (seconds since the epoch)."""
    from datetime import datetime
    folders = [Path(env["FUNCTAI_LOG_CALLS"])] if env.get("FUNCTAI_LOG_CALLS") else []
    folders += [extra] if extra else []
    records, seen = [], set()
    files = [f for d in folders if d and d.exists() for f in d.rglob("*.jsonl")]
    files += [f for f in work.rglob("*.jsonl") if ".venv" not in f.parts]
    for f in files:
        try:
            lines = f.read_text().splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(r, dict) or "functai_call" not in r or r.get("id") in seen:
                continue
            if since is not None:
                started = datetime.fromisoformat(str(r.get("started", "")).replace("Z", "+00:00")).timestamp() \
                    if r.get("started") else 0
                if started < since:
                    continue
            seen.add(r.get("id"))
            records.append(r)
    return records


def summary_of(records: list[dict]) -> dict:
    ex = exchanges(records)
    live = [x for x in ex if not x.get("cached")]
    return {"exchanges": len(ex), "uncached": len(live),
            "uncached_tokens": sum((x.get("usage") or {}).get("total_tokens", 0) for x in live),
            "models": sorted({x.get("model", "?") for x in ex})}


def exchanges(records: list[dict]) -> list[dict]:
    return [x for r in records for x in (r.get("exchanges") or [])]


def calls_summary(folder: Path) -> dict:
    ex = exchanges(call_log(folder))
    live = [x for x in ex if not x.get("cached")]
    tokens = sum((x.get("usage") or {}).get("total_tokens", 0) for x in live)
    return {"exchanges": len(ex), "uncached": len(live), "uncached_tokens": tokens,
            "models": sorted({x.get("model", "?") for x in ex})}


def request_texts(records: list[dict]) -> list[str]:
    """The text of every request (system and messages), as the model saw it."""
    out = []
    for x in exchanges(records):
        req = x.get("request") or {}
        parts = [str(req.get("system") or "")]
        for m in req.get("messages") or []:
            for p in m.get("parts") or []:
                if isinstance(p, dict) and p.get("type") == "text":
                    parts.append(p.get("text", ""))
        out.append("\n".join(parts))
    return out


def request_parts(records: list[dict]) -> list[list[dict]]:
    out = []
    for x in exchanges(records):
        req = x.get("request") or {}
        out.append([p for m in req.get("messages") or [] for p in (m.get("parts") or []) if isinstance(p, dict)])
    return out


def norm(text) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def csv_rows(path: Path) -> list[dict] | None:
    import csv
    if not path.exists():
        return None
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def summarize(checks: dict, required: list[str], score_keys: list[str] | None = None) -> dict:
    keys = score_keys or list(checks)
    values = []
    for k in keys:
        c = checks[k]
        v = c["value"]
        values.append(float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and 0 <= v <= 1
                      else (1.0 if c["pass"] else 0.0))
    return {"checks": checks, "score": round(sum(values) / len(values), 3),
            "success": all(checks[k]["pass"] for k in required)}


def env_for_grading(env: dict) -> dict:
    return {k: v for k, v in env.items() if k != "FUNCTAI_LOG_CALLS"} | {"PYTHONDONTWRITEBYTECODE": "1"}


def exists(p: Path) -> bool:
    return p.exists() and p.stat().st_size > 0


__all__ = ["all_records", "summary_of", "check", "run_script", "call_log", "calls_summary", "request_texts", "request_parts", "norm",
           "csv_rows", "summarize", "env_for_grading", "exists", "os"]


# ---------------------------------------------------------------------------
# The neutral record: every request any library sent to a provider, as the
# recording proxy saw it (recorder.py). The graders read only this, so FunctAI
# and DSPy are judged by the same instrument.
# ---------------------------------------------------------------------------
IMAGE_TYPES = ("image_url", "input_image", "image")


def _walk(node, texts: list, images: list):
    if isinstance(node, dict):
        if node.get("type") in IMAGE_TYPES:
            images.append(1)
            return
        for k, v in node.items():
            if isinstance(v, str) and k in ("text", "content", "instructions", "system"):
                texts.append(v)
            else:
                _walk(v, texts, images)
    elif isinstance(node, list):
        for v in node:
            _walk(v, texts, images)


def recorded(path, since: float | None = None) -> list[dict]:
    """The exchanges in a recorder file: text the model was sent, image parts, tokens."""
    out = []
    path = Path(path) if path else None
    if not path or not path.exists():
        return out
    for line in path.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if since is not None and r["time"] < since:
            continue
        texts, images = [], []
        _walk(r.get("request"), texts, images)
        u = r.get("usage") or {}
        out.append({"time": r["time"], "seconds": r.get("seconds"), "status": r.get("status"), "text": "\n".join(texts),
                    "images": len(images),
                    "input_tokens": u.get("input_tokens", u.get("prompt_tokens", 0)) or 0,
                    "output_tokens": u.get("output_tokens", u.get("completion_tokens", 0)) or 0})
    return out


def record_summary(path, since: float | None = None) -> dict:
    rs = recorded(path, since)
    return {"requests": len(rs), "input_tokens": sum(r["input_tokens"] for r in rs),
            "output_tokens": sum(r["output_tokens"] for r in rs),
            "errors": sum(1 for r in rs if (r["status"] or 200) >= 400)}
