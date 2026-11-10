"""The cases in agents/: long agents and pipelines, written from the rules
(never from an implementation's output). Six kinds, told apart by "kind":

- "result" (../tools.md, *What a tool returns*): {"returned": what a tool
  returned, "expect": {"result": the result as data, "event": {"output",
  "parts"?}}}. ``returned`` is a text, an lm15 part (``{"type": "image",
  ...}``), or a list of texts and parts; a harness gives its tool the
  language's own value for it (a string; lm15's part; a list). ``result``
  is what the conversation's ``tool`` record and a ``tool_result`` hook
  hold; ``event`` the ``tool_result`` event's ``output`` and ``parts``.
- "state" (../conversations.md, ../tools.md *A result that comes
  later*): {"records", "now", "expect": {"states": {turn: state}, "later":
  {turn: [keys it still waits for]}, "unfinished": {turn: [invocations]},
  "delivered": {key: output}, "resumes": {turn: true | {"refuses":
  "turn-state"}}}}. Only turns with something to say are listed in
  ``later`` and ``unfinished``; ``resumes`` is whether a waiting turn may
  go on now (every approval answered, every result delivered).
- "kept" (../streaming.md, *The kept form*): as in events/: {"events",
  "kept": {call: {"inputs", "outputs"}}, "expect": {"events"}}, with the
  kinds note, later and piece, and a tool_result holding parts.
- "limits" (../plugins.md, *limits*): {"limits": {"tokens"?, "cost"?,
  "seconds"?, "prices"?, "then"}, "spent": {"exchanges": [[model,
  usage]], "seconds"}, "at": {"hook": "before_call", "lm"} or {"hook":
  "step", "step", "max_steps", "tools"}, "expect": {"change": null or the
  change the plugin returns}}.
- "pictures" (../plugins.md, *keep_pictures*): {"results": [{"invocation",
  "output"}], "last", "at_once", "expect": {"replace": {invocation:
  output}}} (``{}`` when nothing changes).
- "spent" (../plugins.md, *What a tree spent*): {"exchanges": [[model,
  usage]], "prices", "reported"?: dollars code said it spent (``spend``),
  "expect": {"tokens", "cost", "usage"}}. A limits case's "spent" may have
  "reported" too.

    python contract/cases/make.py
"""

import copy

import events
from stages import NOW, FUTURE, R, V1, a_conversation, state_of, uid

PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
PICTURE = {"type": "image", "media_type": "image/png", "data": PNG}
SOUND = {"type": "audio", "media_type": "audio/wav", "url": "https://example.com/a.wav"}
RESULT_PARTS = ("text", "image", "audio", "video", "document")


# ------------------------------------------------------------------ what a tool returns


def is_part(x) -> bool:
    return isinstance(x, dict) and x.get("type") in RESULT_PARTS


def result_of(returned):
    """tools.md, *What a tool returns*: a text is text; a part, or a list holding one, is a list of parts
    (each text a text part)."""
    if is_part(returned):
        return [copy.deepcopy(returned)]
    if isinstance(returned, list) and any(is_part(x) for x in returned):
        return [copy.deepcopy(x) if is_part(x) else {"type": "text", "text": x} for x in returned]
    return returned


def event_of(result) -> dict:
    """streaming.md: a tool_result's output is the result's text; parts, the whole result when it holds parts."""
    if isinstance(result, str):
        return {"output": result}
    return {"output": "\n".join(p["text"] for p in result if p["type"] == "text"), "parts": result}


def result_cases() -> dict:
    out = {}
    for i, (name, why, returned) in enumerate([
            ("a-text", "A text is the result as it is, and the event's output.", "clicked Buy"),
            ("a-picture", "A picture alone: a list of one part; the event's output is empty, its parts the "
                          "picture.", PICTURE),
            ("a-picture-and-its-caption", "A list holding a picture: each text a text part, in order; the "
                                          "event's output is the texts, its parts the whole result.",
             ["page 2: pricing", PICTURE, "the price is at the top right"]),
            ("two-pictures-and-a-sound", "Several parts of several kinds, as they are.",
             [PICTURE, PICTURE, SOUND]),
            ("texts-around-a-picture", "Texts are joined by a new line in the event's output.",
             ["first", "second", PICTURE])], 1):
        result = result_of(returned)
        out[f"result-{i:02d}-{name}"] = {"description": why, "kind": "result", "returned": returned,
                                         "expect": {"result": result, "event": event_of(result)}}
    return out


# ------------------------------------------------------------------ a turn that waits for later results


def later_of(records: list, turn: str) -> list:
    """The tools of a turn whose last record is later, and whose key no delivered record names."""
    last = {}
    for r in records:
        if r["kind"] == "tool" and r["turn"] == turn:
            last[(r["site"], r["invocation"])] = r
    delivered = delivered_of(records)
    return [r["key"] for r in last.values() if r["state"] == "later" and r["key"] not in delivered]


def delivered_of(records: list) -> dict:
    """tools.md: the first delivered record of a key is the one kept."""
    out = {}
    for r in records:
        if r["kind"] == "delivered":
            out.setdefault(r["key"], r["output"])
    return out


def unfinished_of(records: list, turn: str) -> list:
    started = {}
    for r in records:
        if r["kind"] == "tool" and r["turn"] == turn:
            k = (r["site"], r["invocation"])
            if r["state"] == "started":
                started[k] = r["invocation"]
            else:
                started.pop(k, None)
    return list(started.values())


def unanswered_of(records: list, turn: str) -> list:
    waits = [r for r in records if r["kind"] == "waiting" and r["turn"] == turn]
    if not waits:
        return []
    w = waits[-1]
    answered = {(r["site"], r["invocation"]) for r in records
                if r["kind"] == "approval" and r["turn"] == turn and r["seq"] > w["seq"]}
    return [a["invocation"] for a in w["approvals"] if (a["site"], a["invocation"]) not in answered]


def state_case(name: str, why: str, records: list) -> dict:
    turns = [r["turn"] for r in records if r["kind"] == "turn"]
    states = {t: state_of(records, t, NOW) for t in turns}
    resumes = {}
    for t in turns:
        if states[t] == "waiting":
            ok = not unanswered_of(records, t) and not later_of(records, t)
            resumes[t] = True if ok else {"refuses": "turn-state"}
    return {"description": why, "kind": "state", "records": records, "now": NOW,
            "expect": {"states": states,
                       "later": {t: later_of(records, t) for t in turns if later_of(records, t)},
                       "unfinished": {t: unfinished_of(records, t) for t in turns if unfinished_of(records, t)},
                       "delivered": delivered_of(records), "resumes": resumes}}


def state_cases() -> dict:
    base = a_conversation()
    t = uid(8)
    start = [R("turn", 14, turn=t, parent=uid(4), program=V1, inputs={"message": "look at a.com"}),
             R("lease", 15, turn=t, holder="h:1:a", until=FUTURE, attempt=1)]
    harvest = [R("tool", 16, turn=t, site="tutor#1", invocation=1, id="c1", name="start_harvest",
                 input={"url": "a.com"}, effects="changes", state="started", attempt=1),
               R("tool", 17, turn=t, site="tutor#1", invocation=1, id="c1", name="start_harvest",
                 input={"url": "a.com"}, effects="changes", state="later", key="harvest:1", attempt=1),
               R("tool", 18, turn=t, site="tutor#1", invocation=2, id="c2", name="weather", state="done",
                 output="sunny", attempt=1)]
    wait = [R("waiting", 19, turn=t, approvals=[], attempt=1,
              later=[{"site": "tutor#1", "invocation": 1, "id": "c1", "name": "start_harvest", "key": "harvest:1",
                      "path": "tutor/start_harvest"}])]
    waiting = base + start + harvest + wait
    out = {}
    out["state-01-a-turn-waits-for-a-later-result"] = state_case(
        "state-01-a-turn-waits-for-a-later-result",
        "A tool returned later(key): it ran (not unfinished), the step's other tool ran, and the turn waits for "
        "the result; it cannot go on before it is delivered.", waiting)
    delivered = waiting + [R("delivered", 20, key="harvest:1", output=[{"type": "text", "text": "3 pages"}, PICTURE], by="webhook")]
    out["state-02-a-delivered-result-lets-it-go-on"] = state_case(
        "state-02-a-delivered-result-lets-it-go-on",
        "The result delivered (a conversation's record, with no turn): the turn still waits until a process "
        "resumes it, and it may now.", delivered)
    twice = delivered + [R("delivered", 21, key="harvest:1", output="a late duplicate")]
    out["state-03-the-first-delivery-is-kept"] = state_case(
        "state-03-the-first-delivery-is-kept", "A key delivered twice: the first delivery is the one kept.",
        twice)
    early = base + [R("delivered", 14, key="harvest:9", output="fast")] + [
        {**r, "seq": r["seq"] + 1} for r in start + harvest]
    early[-2] = {**early[-2], "key": "harvest:9"}
    out["state-04-a-result-delivered-before-the-turn-waits"] = state_case(
        "state-04-a-result-delivered-before-the-turn-waits",
        "Delivered before the tool even said it would come later: the turn has nothing to wait for.", early)
    both = base + start + harvest + [R("waiting", 19, turn=t, attempt=1, approvals=[
        {"call": t, "invocation": 3, "id": "c3", "name": "pay", "input": {}, "effects": "changes",
         "path": "tutor/pay", "site": "tutor#1"}], later=wait[0]["later"]),
        R("delivered", 20, key="harvest:1", output="done")]
    out["state-05-an-approval-still-waits"] = state_case(
        "state-05-an-approval-still-waits",
        "Its result delivered, the turn still waits for a person's answer: it cannot go on yet.", both)
    return out


# ------------------------------------------------------------------ the kept form of the new kinds


def kept_event(e: dict, keep: dict, info: dict):
    """streaming.md, *The kept form*, for note, later and a tool_result with parts; every other kind as
    events.py keeps it."""
    fields = {**keep["inputs"], **keep["outputs"]}
    whole = all(fields.values())
    if e["kind"] == "note":
        out = {k: copy.deepcopy(v) for k, v in e.items() if k in events.ENVELOPE or k in ("name", "data")}
        if not whole:
            del out["data"]
            out["content"] = False
        return out
    if e["kind"] == "later":
        out = {k: copy.deepcopy(v) for k, v in e.items()
               if k in events.ENVELOPE or k in ("id", "invocation", "name", "key")}
        if not whole:
            del out["key"]
            out["content"] = False
        return out
    if e["kind"] == "piece":                         # the model's stream: kept as it is, or not at all
        return {k: copy.deepcopy(v) for k, v in e.items() if k in events.ENVELOPE or k == "lm15"} if whole else None
    if e["kind"] == "tool_call":
        out = {k: copy.deepcopy(v) for k, v in e.items()
               if k in events.ENVELOPE or k in ("id", "name", "input", "invocation")}
        if not whole:
            del out["input"]
            out["content"] = False
        return out
    if e["kind"] == "tool_result":
        out = {k: copy.deepcopy(v) for k, v in e.items()
               if k in events.ENVELOPE or k in ("id", "name", "output", "parts", "invocation")}
        if not whole:
            out.pop("output", None)
            out.pop("parts", None)
            out["content"] = False
        return out
    return events.kept_event(e, keep, info)


def kept_cases() -> dict:
    tree = events.cid(1)
    prog = events.program("browse")
    t = "2026-11-02T10:00:00.{:06d}Z"

    def E(seq, kind, **kw):
        return {"functai_event": 2, "kind": kind, "tree": tree, "writer": 1, "seq": seq,
                "after": None if seq == 1 else events.at(1, seq - 1), "at": t.format(seq * 10000),
                "call": tree, "function": "browse", **kw}

    log = [E(1, "started", parent=None, root=tree, program=prog, inputs={"task": "find the price"}, content=True,
             saw=[]),
           E(2, "request", request=1, model=events.MODEL),
           E(3, "piece", lm15={"type": "start", "id": "msg_1", "model": events.MODEL}),
           E(4, "piece", lm15={"type": "delta", "delta": {"type": "tool_call", "part_index": 0, "input": "{\"page\": ",
                                                          "id": "c1", "name": "look"}}),
           E(5, "piece", lm15={"type": "delta", "delta": {"type": "tool_call", "part_index": 0, "input": "\"home\"}"}}),
           E(6, "piece", lm15={"type": "end", "finish_reason": "tool_call",
                               "usage": {"input_tokens": 120, "output_tokens": 9, "total_tokens": 129}}),
           E(7, "tool_call", id="c1", name="look", input={"page": "home"}, invocation=1),
           E(8, "note", name="screen captured", data={"screen": 1, "key": "s1.jpg"}),
           E(9, "tool_result", id="c1", name="look", output="page home", parts=[{"type": "text", "text": "page home"}, PICTURE],
             invocation=1),
           E(10, "request", request=2, model=events.MODEL),
           E(11, "tool_call", id="c2", name="start_harvest", input={"url": "a.com"}, invocation=2),
           E(12, "later", id="c2", invocation=2, name="start_harvest", key="harvest:1")]
    out = {}
    for name, why, keep in [
            ("kept-01-whole", "Content whole: the model's stream (piece), the note, the later and the picture "
                              "result are kept as they are.",
             {"inputs": {"task": True}, "outputs": {"result": True}}),
            ("kept-02-not-whole", "Content not whole: no piece of the model's stream is kept; a note keeps its "
                                  "name, not its data; a later its tool, not its key; a result neither its output "
                                  "nor its parts.",
             {"inputs": {"task": False}, "outputs": {"result": True}})]:
        info = {tree: prog}
        kept = events.relink([x for x in (kept_event(e, keep, info) for e in log) if x is not None])
        out[name] = {"description": why, "kind": "kept", "events": copy.deepcopy(log), "kept": {tree: keep},
                     "expect": {"events": kept}}
    return out


# ------------------------------------------------------------------ limits and keep_pictures


REACHED = ("This task has reached its limit ({what}). Do not ask for any more tools: give your answer now, "
           "from what you have found so far.")
LAST_STEP = ("This is the last step this task may take. Do not ask for any more tools: give your answer now, "
             "from what you have found so far.")
PICTURE_GONE = "(A picture was here. It is no longer shown, to save space.)"


def number(x) -> str:
    return str(int(x)) if float(x).is_integer() else f"{x:g}"


def priced(prices: dict, model: str):
    p = prices.get(model)
    if p is None and ":" in model:
        p = prices.get(model.split(":", 1)[1])
    return p


def estimate(price: dict, usage: dict):
    """lm15's InferencePricing.estimate (lm15 spec/types.md): exact or None. Tiers are not used here."""
    total_in, out = usage.get("input_tokens"), usage.get("output_tokens")
    if total_in is None or out is None:
        return None
    r_in, r_out = price.get("input_per_million"), price.get("output_per_million")
    r_read, r_write = price.get("cache_read_per_million"), price.get("cache_write_per_million")
    read, write = usage.get("cache_read_tokens"), usage.get("cache_write_tokens")
    if read is None:
        if r_read is None or r_read != r_in:
            return None
        read = 0
    if write is None:
        if r_write is None or r_write != r_in:
            return None
        write = 0
    buckets = [(total_in - read - write, r_in), (read, r_read), (write, r_write), (out, r_out)]
    if any(n and r is None for n, r in buckets):
        return None
    return sum(n * r / 1_000_000 for n, r in buckets if n)


def spent_of(exchanges: list, prices: dict, reported: float = 0.0) -> dict:
    """plugins.md, *What a tree spent*: what code reported (``spend``) is in the cost, and needs no price."""
    usages = [u for _, u in exchanges]
    names = set(usages[0]).intersection(*usages[1:]) if usages else set()
    usage = {k: sum(u[k] for u in usages) for k in sorted(names)}
    tokens = usage.get("total_tokens") if exchanges else 0
    cost = 0.0
    for model, u in exchanges:
        p = priced(prices, model)
        got = estimate(p, u) if p is not None else None
        if got is None:
            cost = None
            break
        cost += got
    if cost is not None:
        cost += reported
    return {"tokens": tokens, "cost": cost, "usage": usage}


def reached(limits: dict, spent: dict) -> list:
    got = spent_of(spent["exchanges"], limits.get("prices") or {}, spent.get("reported", 0.0))
    out = []
    if limits.get("tokens") is not None:
        if got["tokens"] is None:
            out.append((f"{number(limits['tokens'])} tokens", "a request did not say how many it used"))
        elif got["tokens"] >= limits["tokens"]:
            out.append((f"{number(limits['tokens'])} tokens", None))
    if limits.get("cost") is not None:
        if got["cost"] is None:
            unknown = sorted({m for m, _ in spent["exchanges"] if priced(limits.get("prices") or {}, m) is None})
            out.append((f"cost {number(limits['cost'])}", (
                f"no price is known for {', '.join(unknown)}" if unknown else
                "a request's cost is not known from its token counts (a price without cache_read_per_million "
                "and cache_write_per_million is exact only for requests that report their cache counts; give "
                "both, equal to input_per_million when the provider does not charge them apart)")))
        elif got["cost"] >= limits["cost"]:
            out.append((f"cost {number(limits['cost'])}", None))
    if limits.get("seconds") is not None and spent["seconds"] >= limits["seconds"]:
        out.append((f"{number(limits['seconds'])} seconds", None))
    return out


def stopped_by(hit: list) -> str:
    return "limit reached: " + "; ".join(f"{label} ({why})" if why else label for label, why in hit)


def limits_change(limits: dict, spent: dict, at: dict):
    """plugins.md, *limits*: what the plugin returns at a before_call or a step."""
    hit = reached(limits, spent)
    if at["hook"] == "before_call":
        lm = at.get("lm")
        if not hit and limits.get("cost") is not None and isinstance(lm, str) \
                and priced(limits.get("prices") or {}, lm) is None:
            hit = [(f"cost {number(limits['cost'])}", f"no price is known for {lm}")]
        return {"stop": stopped_by(hit)} if hit else None
    if hit and limits["then"] == "stop":
        return {"stop": stopped_by(hit)}
    if limits["then"] == "answer" and at["tools"] and (hit or at["step"] >= at["max_steps"]):
        return {"tools": [], "reminder": REACHED.format(what=", ".join(label for label, _ in hit)) if hit else LAST_STEP}
    return None


def limits_cases() -> dict:
    priced_ = {"input_per_million": 3, "output_per_million": 15, "cache_read_per_million": 0.3,
               "cache_write_per_million": 3.75}
    prices = {"claude-sonnet-5": priced_}
    u = {"input_tokens": 200_000, "output_tokens": 10_000, "total_tokens": 210_000, "cache_read_tokens": 150_000,
         "cache_write_tokens": 0}
    nocache = {"input_tokens": 1000, "output_tokens": 10, "total_tokens": 1010}
    step = {"hook": "step", "step": 4, "max_steps": 8, "tools": ["look", "click"]}
    rows = [
        ("01-under-every-limit", "Under every limit: no change.",
         {"tokens": 1_000_000, "cost": 10, "seconds": 600, "prices": prices, "then": "answer"},
         {"exchanges": [["claude-sonnet-5", u]], "seconds": 30}, step),
        ("02-tokens-reached-answer", "Tokens reached: the loop is offered no tools and told to answer.",
         {"tokens": 200_000, "then": "answer"}, {"exchanges": [["claude-sonnet-5", u]], "seconds": 30}, step),
        ("03-cost-reached-stop", "The cost reached (uncached input, cache reads and output at their rates), "
                                 "with then stop: the call stops.",
         {"cost": 1.0, "prices": prices, "then": "stop"},
         {"exchanges": [["claude-sonnet-5", u]] * 5, "seconds": 30}, step),
        ("04-the-last-step", "No limit reached, but this is the last step: answer now (then answer).",
         {"seconds": 600, "then": "answer"}, {"exchanges": [], "seconds": 1},
         {"hook": "step", "step": 8, "max_steps": 8, "tools": ["look"]}),
        ("05-the-last-step-stop", "The last step, with then stop: no change (the loop ends as max_steps says).",
         {"seconds": 600, "then": "stop"}, {"exchanges": [], "seconds": 1},
         {"hook": "step", "step": 8, "max_steps": 8, "tools": ["look"]}),
        ("06-already-wrapping-up", "Tools already taken away: nothing more to change.",
         {"tokens": 1, "then": "answer"}, {"exchanges": [["claude-sonnet-5", u]], "seconds": 1},
         {"hook": "step", "step": 5, "max_steps": 8, "tools": []}),
        ("07-a-call-past-a-limit", "A call that would start past a limit stops.",
         {"seconds": 60, "then": "answer"}, {"exchanges": [], "seconds": 61}, {"hook": "before_call", "lm": None}),
        ("08-no-price-for-the-model", "A cost limit and a call to a model with no price: stopped before it "
                                      "is asked (a limit that cannot be checked is reached).",
         {"cost": 5, "prices": prices, "then": "answer"}, {"exchanges": [], "seconds": 0},
         {"hook": "before_call", "lm": "gpt-6"}),
        ("09-a-model-named-with-its-provider", "A model named provider:model is priced by its own name.",
         {"cost": 5, "prices": prices, "then": "answer"}, {"exchanges": [], "seconds": 0},
         {"hook": "before_call", "lm": "anthropic:claude-sonnet-5"}),
        ("10-cache-counts-not-reported", "A request that reports no cache counts, priced with cache rates "
                                         "that differ from input's: its cost is not known, so the limit is "
                                         "reached.",
         {"cost": 5, "prices": prices, "then": "answer"},
         {"exchanges": [["claude-sonnet-5", nocache]], "seconds": 0}, step),
        ("11-several-reached", "Several limits reached: each is named, in order tokens, cost, seconds.",
         {"tokens": 10, "cost": 0.001, "seconds": 1, "prices": prices, "then": "answer"},
         {"exchanges": [["claude-sonnet-5", u]], "seconds": 5}, step),
    ]
    out = {}
    for name, why, limits, spent, at in rows:
        out[f"limits-{name}"] = {"description": why, "kind": "limits", "limits": limits, "spent": spent, "at": at,
                                 "expect": {"change": limits_change(limits, spent, at)}}
    for i, (exchanges, why) in enumerate([
            ([], "Nothing spent: no tokens, no cost, no usage."),
            ([["claude-sonnet-5", u], ["claude-sonnet-5", u]], "Two requests: counts summed, cost exact."),
            ([["claude-sonnet-5", u], ["claude-sonnet-5", {"output_tokens": 3}]],
             "A request that did not report every count: only the counts every request reported are summed; "
             "the tokens and the cost are not known.")], 1):
        out[f"spent-{i:02d}"] = {"description": why, "kind": "spent", "exchanges": exchanges, "prices": prices,
                                 "expect": spent_of(exchanges, prices)}
    for i, (exchanges, reported, why) in enumerate([
            ([["claude-sonnet-5", u], ["claude-sonnet-5", u]], 0.05,
             "Two requests and what a tool said it spent (a browsing service's charge): the cost is both."),
            ([], 0.0125, "Nothing asked of a model, only what code spent: the cost is what it reported, no price "
                         "needed."),
            ([["claude-sonnet-5", u], ["no-price", u]], 0.05,
             "A request whose price is not known: the cost is not known, whatever code reported.")], 4):
        out[f"spent-{i:02d}"] = {"description": why, "kind": "spent", "exchanges": exchanges, "prices": prices,
                                 "reported": reported, "expect": spent_of(exchanges, prices, reported)}
    reported_only = {"exchanges": [], "seconds": 1.0, "reported": 0.6}
    out["limits-cost-reported-by-code"] = {
        "description": "A cost limit reached by what code said it spent alone (no model was asked yet): the next "
                       "call stops before it starts.",
        "kind": "limits", "limits": {"cost": 0.5, "prices": prices, "then": "answer"}, "spent": reported_only,
        "at": {"hook": "before_call", "lm": "claude-sonnet-5"},
        "expect": {"change": limits_change({"cost": 0.5, "prices": prices, "then": "answer"}, reported_only,
                                           {"hook": "before_call", "lm": "claude-sonnet-5"})}}
    return out


def pictures_to_drop(results: list, last: int, at_once: int) -> dict:
    """plugins.md, *keep_pictures*."""
    seen, drop = 0, {}
    for r in reversed(results):
        if isinstance(r["output"], str):
            continue
        for i in range(len(r["output"]) - 1, -1, -1):
            if r["output"][i]["type"] == "image":
                seen += 1
                if seen > last:
                    drop.setdefault(r["invocation"], set()).add(i)
    if sum(len(v) for v in drop.values()) < at_once:
        return {}
    return {str(r["invocation"]): [{"type": "text", "text": PICTURE_GONE} if i in drop[r["invocation"]] else p
                                   for i, p in enumerate(r["output"])]
            for r in results if r["invocation"] in drop}


def picture_cases() -> dict:
    def shot(n, caption=True):
        return {"invocation": n, "output": ([{"type": "text", "text": f"page {n}"}] if caption else []) + [PICTURE]}

    rows = [
        ("01-few-enough", "No more pictures than last: nothing changes.", [shot(1), shot(2)], 3, 1),
        ("02-the-oldest-goes", "One picture too many: the oldest is replaced by the text, its caption stays.",
         [shot(1), shot(2), shot(3), shot(4)], 3, 1),
        ("03-text-results-are-left", "A text result holds no picture and is never changed.",
         [shot(1), {"invocation": 2, "output": "clicked"}, shot(3), shot(4)], 2, 1),
        ("04-in-a-batch", "Fewer old pictures than at_once: nothing yet.", [shot(1), shot(2), shot(3)], 1, 3),
        ("05-the-batch-is-due", "As many old pictures as at_once: all go at once.",
         [shot(1), shot(2), shot(3), shot(4)], 1, 3),
        ("06-two-in-one-result", "The newest pictures are counted from the end, inside a result too.",
         [{"invocation": 1, "output": [PICTURE, PICTURE]}, shot(2, caption=False)], 2, 1),
        ("07-none-kept", "last 0: every picture goes.", [shot(1)], 0, 1),
    ]
    out = {}
    for name, why, results, last, at_once in rows:
        out[f"pictures-{name}"] = {"description": why, "kind": "pictures", "results": results, "last": last,
                                   "at_once": at_once,
                                   "expect": {"replace": pictures_to_drop(results, last, at_once)}}
    return out


def cases() -> dict:
    return {**result_cases(), **state_cases(), **kept_cases(), **limits_cases(), **picture_cases()}
