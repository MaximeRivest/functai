"""Built-in plugins, made with the public hooks only (contract/plugins.md):
anyone can replace them with their own.

    chat = tutor.conversation("alex", store="tutoring/", plugins=[functai.compaction(keep=20)])

    researcher = functai.delegate(research, description="Look things up in the notes.")
    @ai(tools=[researcher])
    def assistant(request: str) -> str: ...
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple

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


# ------------------------------------------------------------------ long tool loops (the step hook)


REACHED = ("This task has reached its limit ({what}). Do not ask for any more tools: give your answer now, "
           "from what you have found so far.")
LAST_STEP = ("This is the last step this task may take. Do not ask for any more tools: give your answer now, "
             "from what you have found so far.")
PICTURE_GONE = "(A picture was here. It is no longer shown, to save space.)"


def _number(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else f"{x:g}"


def reached(spent: Any, *, tokens: Optional[int] = None, cost: Optional[float] = None,
            seconds: Optional[float] = None, prices: Any = None) -> List[Tuple[str, Optional[str]]]:
    """Which limits a tree's spending reached (contract/plugins.md, *limits*):
    each ``(limit, why)``, the limit as the model is told it (``"cost 5"``)
    and why it counts as reached when its count is not known (else None);
    ``[]`` when none. A count that is not known cannot be checked, and counts
    as reached."""
    out: List[Tuple[str, Optional[str]]] = []
    if tokens is not None:
        got = spent.tokens
        if got is None:
            out.append((f"{_number(tokens)} tokens", "a request did not say how many it used"))
        elif got >= tokens:
            out.append((f"{_number(tokens)} tokens", None))
    if cost is not None:
        got = spent.cost(prices or {})
        if got is None:
            unknown = sorted({m for m, _ in spent.exchanges if _priced(prices, m) is None})
            out.append((f"cost {_number(cost)}", (
                f"no price is known for {', '.join(unknown)}" if unknown else
                "a request's cost is not known from its token counts (a price without cache_read_per_million "
                "and cache_write_per_million is exact only for requests that report their cache counts; give "
                "both, equal to input_per_million when the provider does not charge them apart)")))
        elif got >= cost:
            out.append((f"cost {_number(cost)}", None))
    if seconds is not None and spent.seconds >= seconds:
        out.append((f"{_number(seconds)} seconds", None))
    return out


def stopped_by(hit: List[Tuple[str, Optional[str]]]) -> str:
    """The reason a ``limits`` stop gives: each limit reached, and why when its count is not known."""
    return "limit reached: " + "; ".join(f"{label} ({why})" if why else label for label, why in hit)


def _priced(prices: Any, model: str) -> Any:
    from .plugins import pricing
    return pricing(prices or {}, model)


def limits(*, tokens: Optional[int] = None, cost: Optional[float] = None, seconds: Optional[float] = None,
           prices: Any = None, then: str = "answer", name: str = "limits") -> Plugin:
    '''Stop a call tree (an agent, a pipeline) when it has used enough: tokens,
    money or time. Nothing is limited by default; this plugin is opt-in.

    Counted over the whole tree a call starts (every AI function inside a
    module, every tool's own calls), from the requests a provider answered
    in this process (replies from the reply cache, and those a resumed turn
    reuses, cost nothing now and are not counted). Checked before each AI
    function's call and before each step of a tool loop, so one request may
    go past a limit; the next does not start.

    Parameters
    ----------
    tokens : int, optional
        Every token read and written (``total_tokens``), summed.
    cost : float, optional
        The list price of the requests, from ``prices`` (needed with it).
        Exact or nothing: a request to a model with no price, or whose
        counts do not settle its price, counts as reaching the limit.
    seconds : float, optional
        Since the tree's outermost call started.
    prices : dict, optional
        A model's name (as you ask for it) → an lm15 ``InferencePricing``
        (or a dict of its fields: ``{"input_per_million": 3,
        "output_per_million": 15, "cache_read_per_million": 0.3,
        "cache_write_per_million": 3.75}``). FunctAI ships no price list:
        prices change, and a wrong price would mislead.
    then : "answer" or "stop"
        ``"answer"`` (the default): a tool loop that reaches a limit is
        offered no more tools and told to answer with what it has; so is a
        loop at its last step (``max_steps``), instead of failing with
        ``StepLimit``. ``"stop"``: the call stops. Either way a call that
        would start past a limit stops (``PluginError``, code
        ``plugin-stopped``).

    Examples
    --------
    ```python
    from lm15.models import InferencePricing
    prices = {"claude-opus-5-5": InferencePricing(input_per_million=5, output_per_million=25)}
    with functai.configure(plugins=[functai.limits(cost=3.0, seconds=600, prices=prices)]):
        report = concept_test(url)
    ```
    '''
    if then not in ("answer", "stop"):
        raise ValueError(f'then is "answer" or "stop", not {then!r}')
    if cost is not None and not prices:
        raise ValueError("a cost limit needs prices: {model: InferencePricing(...)} (lm15.models), or dicts of its "
                         "fields; FunctAI ships none")
    for label, v in (("tokens", tokens), ("cost", cost), ("seconds", seconds)):
        if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float)) or v <= 0):
            raise ValueError(f"{label} is a number above 0")
    if tokens is None and cost is None and seconds is None:
        raise ValueError("limits needs at least one of tokens, cost, seconds")
    for model in (prices or {}):
        _priced(prices, model)                     # a malformed price is refused now, not mid-run
    config: Dict[str, Any] = {k: v for k, v in (("tokens", tokens), ("cost", cost), ("seconds", seconds)) if v is not None}
    if prices:
        config["prices"] = {m: _price_json(_priced(prices, m)) for m in prices}
    config["then"] = then
    plugin = Plugin(name, version="1.0.0", description="stop a call tree when it used enough tokens, money or time",
                    config=config)
    limit = dict(tokens=tokens, cost=cost, seconds=seconds, prices=prices)

    @plugin.before_call
    def before(call: Any) -> Optional[Change]:
        hit = reached(call.spent, **limit)
        if not hit and cost is not None and isinstance(call.lm, str) and _priced(prices, call.lm) is None:
            hit = [(f"cost {_number(cost)}", f"no price is known for {call.lm}")]
        return Change(stop=stopped_by(hit)) if hit else None

    @plugin.step
    def step(s: Any) -> Optional[Change]:
        hit = reached(s.spent, **limit)
        if hit and then == "stop":
            return Change(stop=stopped_by(hit))
        if then == "answer" and s.tools and (hit or s.last_step):
            return Change(tools=[], reminder=REACHED.format(what=", ".join(label for label, _ in hit))
                          if hit else LAST_STEP)
        return None

    return plugin


def pictures_to_drop(results: List[Any], *, last: int, at_once: int = 1) -> Dict[int, Any]:
    """Which earlier tool results ``keep_pictures`` shows otherwise, and as
    what (contract/plugins.md): the pictures before the newest ``last`` are
    replaced by a short text, once at least ``at_once`` of them are shown."""
    seen = 0
    drop: Dict[int, List[int]] = {}
    for r in reversed(results):
        if isinstance(r.output, str):
            continue
        for i in range(len(r.output) - 1, -1, -1):
            if r.output[i].get("type") == "image":
                seen += 1
                if seen > last:
                    drop.setdefault(r.invocation, []).append(i)
    if sum(len(v) for v in drop.values()) < at_once:
        return {}
    out: Dict[int, Any] = {}
    for r in results:
        if r.invocation in drop:
            gone = set(drop[r.invocation])
            out[r.invocation] = [{"type": "text", "text": PICTURE_GONE} if i in gone else dict(p)
                                 for i, p in enumerate(r.output)]
    return out


def _price_json(price: Any) -> Dict[str, Any]:
    """An lm15 ``InferencePricing`` as JSON (contract/plugins.md, *limits*): its fields by lm15's names, without
    those at lm15's defaults (an unset rate, currency USD, no tiers), so one price is written one way."""
    out: Dict[str, Any] = {}
    for f in dataclasses.fields(price):
        v = getattr(price, f.name)
        if v is None or (f.name == "currency" and v == "USD") or (f.name == "tiers" and not v):
            continue
        out[f.name] = [dataclasses.asdict(t) if dataclasses.is_dataclass(t) else t for t in v] if f.name == "tiers" else v
    return out


def keep_pictures(last: int = 3, *, at_once: int = 1, name: str = "keep-pictures") -> Plugin:
    '''Show a long tool loop only its newest pictures: before each step, the
    pictures tool results gave earlier than the newest ``last`` are replaced
    by a short text (the results' own text stays). A browsing agent that
    takes a screenshot at every click otherwise sends every screenshot again
    with every request.

    ``at_once``: drop old pictures only once this many are due, all
    together. A provider's prompt cache keeps the start of a request only
    while it stays the same, and every drop changes it: with caching on,
    ``at_once=5`` drops five pictures every fifth step instead of one at
    each. The turn and the call log keep every picture as the tool gave it;
    each request in the call log holds what it sent.

    Examples
    --------
    ```python
    agent = browse.using(plugins=[functai.keep_pictures(3)], max_steps=200)
    ```
    '''
    if isinstance(last, bool) or not isinstance(last, int) or last < 0:
        raise ValueError("last is how many of the newest pictures stay shown (0 or more)")
    if isinstance(at_once, bool) or not isinstance(at_once, int) or at_once < 1:
        raise ValueError("at_once is 1 or more")
    plugin = Plugin(name, version="1.0.0", description="show a tool loop only its newest pictures",
                    config={"last": last, "at_once": at_once})

    @plugin.step
    def step(s: Any) -> Optional[Change]:
        drop = pictures_to_drop(s.results, last=last, at_once=at_once)
        return Change(replace=drop) if drop else None

    return plugin
