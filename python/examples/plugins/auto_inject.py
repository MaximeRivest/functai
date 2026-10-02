"""``@path`` in a message brings the file in (a port of Pi's `auto-inject`).

The files' text is added to the turn's input before the turn is recorded,
so the turn holds exactly what the model was given: it is asked again with
the same text even after the files changed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

import functai

REFERENCE = re.compile(r'(?:^|\s)@("(?:\\.|[^"\\])*"|[\w./-]+)')
TEXT = {".py", ".ts", ".js", ".md", ".txt", ".json", ".toml", ".yaml", ".yml", ".csv", ".sh", ".r", ".jl", ".rs"}


def plugin(root: "str | Path" = ".", *, field: str = "message", limit: int = 200_000) -> functai.Plugin:
    base = Path(root).resolve()
    p = functai.Plugin("auto-inject", version="1.0.0")

    def files(ref: str):
        path = (base / ref).resolve()
        if base not in (path, *path.parents):          # the model never chooses these, the person does; still
            return []
        if path.is_dir():
            return sorted(f for f in path.rglob("*") if f.is_file() and f.suffix.lower() in TEXT
                          and not any(part.startswith(".") for part in f.relative_to(base).parts))
        return [path] if path.is_file() else []

    @p.turn_start
    def inject(turn: Any) -> Optional[functai.Change]:
        text = turn.inputs.get(field)
        if not isinstance(text, str):
            return None
        refs = []
        for m in REFERENCE.finditer(text):
            ref = m.group(1)
            if ref.startswith('"'):
                try:
                    ref = json.loads(ref)
                except ValueError:
                    continue
            refs.append(ref)
        parts, used = [], 0
        for ref in dict.fromkeys(refs):
            for f in files(ref):
                body = f.read_text(errors="replace")
                if used + len(body) > limit:
                    break
                used += len(body)
                parts.append(f"--- {f.relative_to(base)} ---\n{body}")
        return functai.Change(inputs={field: text + "\n\n" + "\n\n".join(parts)}) if parts else None

    return p
