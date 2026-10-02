"""Modes: switch how a conversation's AI function works (a port of the Pi and
Chattering `modes` extension).

A mode adds to the instruction (``appendix``), puts a line before it
(``opener``), or replaces it (``system_prompt``), and may offer only some
tools. The mode in use is an entry of the conversation, so it follows the
branch, and a host switches it without touching the program:

    chat = assistant.conversation("work", plugins=[modes.plugin(MODES)])
    modes.switch(chat, "review")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import functai


@dataclass(frozen=True)
class Mode:
    key: str
    label: str
    opener: str = ""
    appendix: str = ""
    system_prompt: str = ""
    tools: Optional[Sequence[str]] = None


def plugin(modes: Sequence[Mode], *, default: Optional[str] = None) -> functai.Plugin:
    by_key: Dict[str, Mode] = {m.key: m for m in modes}
    p = functai.Plugin("modes", version="1.0.0", description="switch prompt profiles per conversation")

    @p.before_call
    def apply(call: Any) -> Optional[functai.Change]:
        chosen = call.entries("mode")
        key = chosen[-1]["data"]["key"] if chosen else default
        mode = by_key.get(key) if key else None
        if mode is None:
            return None
        instruction = mode.system_prompt or call.instruction
        if mode.opener:
            instruction = f"{mode.opener}\n\n{instruction}"
        return functai.Change(
            instruction=instruction if instruction != call.instruction else None,
            sections=[mode.appendix] if mode.appendix.strip() else None,
            tools=[t for t in call.all_tools if t in mode.tools] if mode.tools is not None else None)

    return p


def switch(chat: Any, key: str) -> None:
    """Use mode ``key`` from now on, on this conversation's branch."""
    head = chat.head
    chat.remember("modes", "mode", {"key": key}, turn=head.id if head is not None else None)
