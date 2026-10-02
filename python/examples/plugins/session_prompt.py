"""A per-conversation opener line (a port of Pi's `session-prompt`): the
opener is an entry of the conversation, put before the instruction.

    chat = assistant.conversation("c", plugins=[session_prompt.plugin])
    session_prompt.set(chat, "You are terse and precise.")
"""

from __future__ import annotations

from typing import Any, Optional

import functai

plugin = functai.Plugin("session-prompt", version="1.0.0")


@plugin.before_call
def opener(call: Any) -> Optional[functai.Change]:
    found = call.entries("prompt")
    text = found[-1]["data"]["prompt"].strip() if found else ""
    return functai.Change(instruction=f"{text}\n\n{call.instruction}") if text else None


def set(chat: Any, text: str) -> None:  # noqa: A001 — the command's own word
    head = chat.head
    chat.remember("session-prompt", "prompt", {"prompt": text}, turn=head.id if head is not None else None)
