"""Built-in plugins, made with the public hooks only (contract/plugins.md):
anyone can replace them with their own.

    chat = tutor.conversation("alex", store="tutoring/", plugins=[functai.compaction(keep=20)])

    researcher = functai.delegate(research, description="Look things up in the notes.")
    @ai(tools=[researcher])
    def assistant(request: str) -> str: ...
"""

from __future__ import annotations

import hashlib
import inspect
from typing import Any, Callable, Dict, List, Optional

from . import calllog
from .plugins import Change, Context, Plugin, TurnEnd


# ------------------------------------------------------------------ compaction


def _summarizer() -> Any:
    from .core import ai

    @ai
    def summarize_conversation(earlier_summary: str, new_turns: List[Dict[str, Any]]) -> str:
        """Write the summary of a conversation that whoever continues it needs:
        what was asked, what was answered and decided, names, numbers and
        facts that came up, and what is still open. Start from the earlier
        summary (empty when there is none) and fold the new turns into it. Be
        complete about facts and short about everything else."""
        ...

    return summarize_conversation


_default: List[Any] = []


def compaction(*, keep: int = 20, every: int = 10, summarize: Optional[Callable[..., str]] = None,
               lm: Optional[str] = None, name: str = "compaction") -> Plugin:
    '''Keep a long conversation short: older turns are folded into a summary.

    When a turn ends and more than ``keep + every`` turns of its branch are
    not yet summarized, every turn but the last ``keep`` is folded into the
    summary (the earlier summary and those turns, given to ``summarize``).
    The summary is kept in the conversation (an entry at that turn, so each
    branch has its own), and the next turns are shown it, as a section of the
    instruction, with only the turns after it. The call that wrote it is in
    the call log; what a turn was shown is in its record, so a rated turn is
    asked again with the same summary.

    Parameters
    ----------
    keep : int
        How many recent turns are always shown whole.
    every : int
        How many more turns may pile up before the next summary (summarizing
        at every turn would cost a call per turn).
    summarize : function, optional
        ``summarize(earlier_summary, new_turns) -> str``, given by position,
        ``new_turns`` a list of ``{input…, output…}`` rows. Default: an AI function of FunctAI's.
    lm : str, optional
        The model the default summarizer uses (default: the one configured).
    name : str
        The plugin's name (two compactions with different settings on one
        conversation need two names).

    Returns
    -------
    Plugin
        For ``fn.conversation(..., plugins=[...])`` or ``configure``.
    '''
    if not isinstance(keep, int) or keep < 0 or not isinstance(every, int) or every < 1:
        raise ValueError("compaction(keep=n, every=m): keep a whole number of at least 0, every at least 1")
    ext = Plugin(name, version="1.0.0", description=f"compaction: summarize all but the last {keep} turns")

    def latest(entries: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        return entries[-1]["data"] if entries else None

    @ext.turn_end
    def fold(event: TurnEnd) -> None:
        if event.state != "done":
            return
        turns = event.turns()
        summary = latest(event.entries("summary"))
        ids = [t.id for t in turns]
        start = ids.index(summary["through"]) + 1 if summary and summary["through"] in ids else 0
        open_turns = turns[start:]
        if len(open_turns) < keep + every:
            return
        folded = open_turns[:len(open_turns) - keep]
        rows = [{**t.inputs, **t.outputs} for t in folded]
        fn = summarize
        if fn is None:
            if not _default:
                _default.append(_summarizer())
            fn = _default[0].using(lm=lm) if lm else _default[0]
        with calllog.part_of("compaction", event.turn):
            text = fn(summary["text"] if summary else "", rows)
        if not isinstance(text, str) or not text.strip():
            raise TypeError("a summary is a text")
        event.remember("summary", {"through": folded[-1].id, "text": text,
                                   "turns": (summary["turns"] if summary else 0) + len(folded)})

    @ext.context
    def show(event: Context) -> Optional[Change]:
        summary = latest(event.entries("summary"))
        if summary is None:
            return None
        log = event.conversation._read()
        branch = [st.id for st in log.branch(event.parent)]
        if summary["through"] not in branch:
            return None
        covered = set(branch[:branch.index(summary["through"]) + 1])
        return Change(keep=[t.id for t in event.turns if t.id not in covered],
                      sections=[f"Earlier in this conversation ({summary['turns']} turns, summarized):\n"
                                f"{summary['text']}"])

    return ext


# ------------------------------------------------------------------ delegation


def delegate(program: Any, *, name: Optional[str] = None, description: Optional[str] = None,
             remember: bool = True, effects: Optional[str] = None) -> Any:
    '''Another program as a tool: an assistant hands part of its work to it.

    Asked inside a conversation's turn, the program answers in a
    conversation of its own (``<conversation>.<name>``, in the same store),
    which follows the branch of the turn that asked: asked again later on
    that branch, it remembers what it was asked before; on another branch,
    it does not. Its calls are in the asking turn's call tree, under the tool
    call. Outside a conversation, or with ``remember=False``, it is called
    plainly.

    Parameters
    ----------
    program : AI function or module
        What does the work. Its inputs are the tool's.
    name : str, optional
        The tool's name (default: the program's).
    description : str, optional
        What the model is told the tool does (default: the program's docstring).
    remember : bool
        Keep a conversation with it, per branch (default True).
    effects : "reads" or "changes", optional
        What it does to the world (default: "reads" when every tool it has
        only reads, else unknown, which counts as "changes").

    Returns
    -------
    Tool
        For ``@ai(tools=[...])``.
    '''
    from .core import FunctAIFunc
    from .module import FunctAIModule
    from .tools import Tool, effects_of
    if not isinstance(program, (FunctAIFunc, FunctAIModule)):
        raise TypeError("delegate takes an AI function or a module")
    tool_name = name or program.__name__
    original = program.__wrapped__ if isinstance(program, FunctAIFunc) else program._fn
    signature = inspect.signature(original)
    signature = signature.replace(return_annotation=inspect.Signature.empty)

    def run(**inputs: Any) -> Any:
        call = calllog.current()
        turn_run = getattr(call, "turn_run", None) if call is not None else None
        if not remember or turn_run is None:
            return program(**inputs)
        from .conversations import Conversation
        outer = turn_run.conv
        sub_id = f"{outer.id}.{tool_name}"
        if len(sub_id) > 200:
            sub_id = f"{outer.id[:150]}.{hashlib.sha256(sub_id.encode()).hexdigest()[:16]}"
        sub = Conversation(program, sub_id, store=outer.store, _delegated=True)
        last = outer.entries("delegate", tool_name, branch=turn_run.turn)
        if last:
            sub = sub.continue_from(last[-1]["data"]["turn"])
        else:
            sub._head = None                             # the first delegation on this branch starts afresh
            sub._exact = True
        stream = sub.stream(**inputs)
        value = stream.result
        outer.remember("delegate", tool_name, {"turn": stream.turn_id}, turn=turn_run.turn)
        return value

    run.__name__ = tool_name
    run.__qualname__ = tool_name
    run.__doc__ = description or inspect.cleandoc(program.__doc__ or "") or f"Ask {program.__name__}."
    run.__signature__ = signature                        # type: ignore[attr-defined]
    run.__module__ = getattr(original, "__module__", __name__)
    iface = program.interface

    def parameters() -> Dict[str, Any]:
        # the tool's inputs are the program's, as its interface states them (no host types to resolve again)
        return {"type": "object", "properties": {f["name"]: f["shape"] for f in iface["inputs"]},
                "required": [f["name"] for f in iface["inputs"] if not f.get("optional")]}

    wrapped = Tool(run, effects=effects if effects is not None else effects_of(program))
    wrapped._tool_parameters = parameters                # type: ignore[attr-defined]
    wrapped._delegate = program                          # type: ignore[attr-defined]
    return wrapped


__all__ = ["compaction", "delegate"]
