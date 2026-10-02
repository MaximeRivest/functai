"""Cases for ../plugins.md, written from its rules:

    order-*    which plugins run around a call, in which order
    combine-*  how the changes of the handlers of one hook combine

A harness builds plugins whose handlers return the given changes (as data),
runs the hook, and compares what it got.
"""

import copy


def order(layers: list, veto: bool) -> list:
    """plugins.md, *Order*: the program's own first, then each layer from the innermost out, then the process's,
    each layer in the order it lists them; once each, in its outermost layer; a host layer's program_plugins false
    drops the program's own."""
    seen, kept = set(), []
    for layer in reversed(layers):                        # outermost first: a plugin runs in its outermost layer
        mine = []
        if not (veto and layer["where"] == "own"):
            for name in layer["plugins"]:
                if name not in seen:
                    seen.add(name)
                    mine.append(name)
        kept.append(mine)
    return [n for mine in reversed(kept) for n in mine]  # innermost first, each layer in the order it lists


def combine(hook: str, start: dict, changes: list) -> dict:
    """plugins.md, *Order*: sections accumulate; instruction, lm, tools and inputs are replaced by later
    changes; settings merge; keep replaces, without adds; the first block ends tool_call."""
    state = copy.deepcopy(start)
    ran = 0
    for c in changes:
        ran += 1
        if c is None:
            continue
        if "sections" in c:
            state["sections"] = state.get("sections", []) + c["sections"]
        for k in ("instruction", "lm", "tools", "output"):
            if k in c:
                state[k] = c[k]
        if "settings" in c:
            state["settings"] = {**state.get("settings", {}), **c["settings"]}
        if "inputs" in c:
            state["inputs"] = {**state.get("inputs", {}), **c["inputs"]}
        if "keep" in c:
            state["keep"] = list(c["keep"])
        if "without" in c:
            w = state.setdefault("without", {})
            targets = c["without"] if isinstance(c["without"], dict) else {t: c["without"] for t in state["keep"]}
            for t, names in targets.items():
                w[t] = sorted(set(w.get(t, [])) | set(names))
        if "block" in c:
            state["block"] = c["block"]
            break
    if "without" in state:                                # only the turns shown have fields left out
        state["without"] = {t: v for t, v in state["without"].items() if t in state["keep"]}
    state["ran"] = ran
    return state


def cases() -> dict:
    out = {}
    layered = [{"where": "own", "plugins": ["mine", "host-mode"]},
               {"where": "block", "plugins": ["conversation"]},
               {"where": "block", "plugins": ["outer-block"]},
               {"where": "configure", "plugins": ["host-mode", "audit"]}]
    for i, (veto, why) in enumerate([(False, "the program's own first, the process's last; a plugin set twice runs "
                                              "once, in its outermost layer"),
                                     (True, "a host layer's program_plugins false drops the program's own")], 1):
        out[f"order-{i:02d}"] = {"description": why, "kind": "order", "layers": layered, "program_plugins": not veto,
                                 "expect": {"order": order(layered, veto)}}
    combos = [
        ("before_call", "sections accumulate; the instruction, the model and the tools are the last change's; "
                        "settings merge",
         {"instruction": "Answer.", "lm": "gpt-4.1-mini", "tools": ["read", "write"], "sections": [], "settings": {}},
         [{"sections": ["A."], "lm": "m1"}, None, {"sections": ["B."], "settings": {"temperature": 0.2}},
          {"instruction": "Be a pirate.", "tools": ["read"], "settings": {"max_tokens": 50}}]),
        ("context", "keep replaces the turns shown; without adds fields to leave out",
         {"keep": ["t1", "t2", "t3"], "without": {}, "sections": []},
         [{"without": ["photo"]}, {"keep": ["t2", "t3"], "sections": ["Summary."]}, {"without": {"t3": ["notes"]}}]),
        ("tool_call", "inputs merge; the first block ends the hook: later handlers do not run",
         {"inputs": {"to": "ana", "text": "hi"}},
         [{"inputs": {"to": "team"}}, {"block": "not today"}, {"inputs": {"to": "everyone"}}]),
        ("turn_start", "inputs are replaced by name, in order",
         {"inputs": {"message": "  hi  ", "tone": "kind"}},
         [{"inputs": {"message": "hi"}}, {"inputs": {"message": "hi @notes"}}]),
        ("tool_result", "the output is the last change's",
         {"output": "raw"}, [{"output": "raw (checked)"}, None]),
    ]
    for i, (hook, why, start, changes) in enumerate(combos, 1):
        out[f"combine-{i:02d}-{hook.replace('_', '-')}"] = {
            "description": why, "kind": "combine", "hook": hook, "start": start, "changes": changes,
            "expect": combine(hook, start, changes)}
    return out
