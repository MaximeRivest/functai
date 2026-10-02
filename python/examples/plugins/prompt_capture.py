"""Keep the last instruction and the last request sent, per conversation (a
port of Pi's `prompt-capture`): ``pending`` (the instruction as the plugins
before it left it) and ``wire`` (the system text of the request sent)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import functai


def plugin(folder: "str | Path") -> functai.Plugin:
    where = Path(folder)
    where.mkdir(parents=True, exist_ok=True)
    p = functai.Plugin("prompt-capture", version="1.0.0")

    def save(kind: str, key: str, text: str) -> None:
        (where / f"{key}-{kind}.md").write_text(text)

    @p.before_call
    def pending(call: Any) -> None:
        save("pending", call.conversation or call.function, "\n\n".join([call.instruction, *call.sections]))

    @p.request
    def wire(event: Any) -> None:
        system = event.request.system
        text = system if isinstance(system, str) else "\n\n".join(getattr(part, "text", "") or ""
                                                                   for m in (system or ()) for part in m.parts)
        save("wire", event.function, text)
        return None                                     # only reads: the call stays replayable

    return p


def last(folder: "str | Path", key: str) -> Dict[str, str]:
    where = Path(folder)
    return {k: (where / f"{key}-{k}.md").read_text() for k in ("pending", "wire") if (where / f"{key}-{k}.md").exists()}
