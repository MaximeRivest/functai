"""The cases in setup/: what a program runs on (../calls.md, *Setups*),
written from the rules.

Each case: {"description", "definition": a functions.md definition (as in
functions/), "settings": the settings in force around the call (contract
names: "lm", "temperature", "max_steps", "wall", ...), "plugins": the
built-in plugins around it, in the order they run ([{"builtin": "limits" or
"keep_pictures", "args": {...}}]), "expect": {"document": the setup's
document, "setup": its hash} or {"refuses": code}}. A harness defines the
function, sets the settings and plugins around it (a block), and reads its
setup. Tools' code is a language's own and is not in these cases (each
language's tests pin how it reads it); everything else is.

Two steps are borrowed, as functions.py borrows lmcc's request: lm15 says
which provider a model name routes to, and lm15 writes its settings
(config) as canonical JSON. Everything FunctAI decides is derived here.
"""

import copy

import functions
from common import sha

CONFIG = ("temperature", "max_tokens", "top_p", "seed", "stop", "reasoning")
DEFAULT_SETTINGS = {"max_steps": 8, "retries": 1, "tool_errors": "report", "on_unreadable": "raise"}
PLUGINS = {"limits": ("limits", "1.0.0"), "keep_pictures": ("keep-pictures", "1.0.0")}


def model_of(lm):
    """calls.md, *Setups*: a model as lm15 routes it (borrowed: which provider a bare name goes to)."""
    import lm15
    router = lm15.LMRouter() if hasattr(lm15, "LMRouter") else None
    if ":" in lm and "/" not in lm.split(":", 1)[0]:
        provider, model = lm.split(":", 1)
        return {"provider": provider, "model": model}
    route = router.resolve(lm)
    return {"provider": route.provider, "model": route.model}


def config_of(settings: dict):
    import lm15
    import lm15.serde
    given = {k: settings[k] for k in CONFIG if settings.get(k) is not None}
    return lm15.serde.config_to_dict(lm15.Config(**given)) if given else None


def plugin_of(spec: dict) -> dict:
    """The built-ins' config (plugins.md, *Built-in plugins*)."""
    name, version = PLUGINS[spec["builtin"]]
    args = spec["args"]
    if spec["builtin"] == "limits":
        config = {k: args[k] for k in ("tokens", "cost", "seconds") if args.get(k) is not None}
        if args.get("prices"):
            config["prices"] = {m: {k: v for k, v in p.items() if v is not None} for m, p in args["prices"].items()}
        config["then"] = args.get("then", "answer")
    else:
        config = {"last": args.get("last", 3), "at_once": args.get("at_once", 1)}
    return {"name": name, "version": version, "config": config}


def reach_of(wall, needs=None) -> object:
    """calls.md, *Reach*: "open" with no wall; else what the wall lets through (or the needs), sorted, once."""
    if wall is None:
        return "open"
    given = wall if isinstance(wall, dict) else {"name": wall}
    needs = needs or {}
    out = {"domains": sorted(set(given.get("domains", needs.get("domains", [])))),
           "secrets": sorted(set(given.get("secrets", needs.get("secrets", []))))}
    if "limits" in given:
        out["limits"] = copy.deepcopy(given["limits"])
    return out


def document(d: dict, settings: dict, plugins: list) -> dict:
    doc = {"version": functions.expect(d)["version"], "model": model_of(settings["lm"])}
    config = config_of(settings)
    if config:
        doc["config"] = config
    doc["settings"] = {k: settings.get(k, v) for k, v in DEFAULT_SETTINGS.items()}
    if settings.get("capabilities") is not None:
        doc["settings"]["capabilities"] = settings["capabilities"]
    if settings.get("escalate_to") is not None:
        doc["escalate"] = {"to": model_of(settings["escalate_to"]), "below": settings.get("escalate_below", 0.9)}
    if plugins:
        doc["plugins"] = [plugin_of(p) for p in plugins]
    doc["reach"] = reach_of(settings.get("wall"))
    return doc


def case(description, key, settings, plugins=()):
    d = functions.DEFINITIONS[key][1]
    doc = document(d, settings, list(plugins))
    return {"description": description, "definition": d, "settings": settings, "plugins": list(plugins),
            "expect": {"document": doc, "setup": sha(doc)}}


LM = "openai:gpt-4.1-mini"
WALL = {"name": "srt", "domains": ["api.browser-use.com", "*.browser-use.com", "api.browser-use.com"],
        "secrets": ["BROWSER_USE_API_KEY"], "limits": {"memory_mb": 2048, "seconds": 600}}


def cases() -> dict:
    out = {
        "01-a-model-and-defaults": case(
            "The version, the model as lm15 routes it, and FunctAI's settings that change what the model is asked "
            "or what comes back, each with its default. No lm15 setting set: no config. No wall: reach is open.",
            "01-an-answer-from-a-list", {"lm": LM}),
        "02-a-bare-name-is-its-route": case(
            "gpt-4.1-mini routes to openai: the same model, the same setup as openai:gpt-4.1-mini.",
            "01-an-answer-from-a-list", {"lm": "gpt-4.1-mini"}),
        "03-another-model-another-setup": case(
            "The same function on another model: the same version, another setup.",
            "01-an-answer-from-a-list", {"lm": "anthropic:claude-haiku-4-5"}),
        "04-sampling-and-settings": case(
            "lm15's settings as its canonical JSON; FunctAI's as set (capabilities only when set).",
            "01-an-answer-from-a-list",
            {"lm": LM, "temperature": 0.2, "max_tokens": 300, "seed": 7, "max_steps": 20, "retries": 2,
             "tool_errors": "raise", "on_unreadable": "record", "capabilities": {"native_structured_output": False}}),
        "05-what-is-not-in-a-setup": case(
            "Where and how it is reached, the reply cache, the call log, network retries: none is in the setup, "
            "which is 01's.",
            "01-an-answer-from-a-list",
            {"lm": LM, "base_url": "https://gateway.example/v1", "api_retries": 9, "api_max_wait": 5,
             "cache_replies": True, "replicate": 2, "log_content": False}),
        "06-escalation": case(
            "A backup model when the first is unsure: its route and the probability below which it answers.",
            "01-an-answer-from-a-list", {"lm": LM, "escalate_to": "anthropic:claude-sonnet-4-5", "escalate_below": 0.8}),
        "07-plugins-in-the-order-they-run": case(
            "The built-in plugins with their config, in the order they run.",
            "01-an-answer-from-a-list", {"lm": LM},
            [{"builtin": "keep_pictures", "args": {"last": 2}},
             {"builtin": "limits", "args": {"tokens": 100000, "cost": 0.5, "then": "stop", "prices": {
                 "gpt-4.1-mini": {"input_per_million": 0.4, "output_per_million": 1.6}}}}]),
        "08-a-wall-made-from-the-needs": case(
            "Behind a wall made from the program's needs (here none): reach is what it needs, nothing open.",
            "01-an-answer-from-a-list", {"lm": LM, "wall": "srt"}),
        "09-a-wall-that-says-what-it-lets-through": case(
            "Behind a wall that says what it lets through and its limits: reach is that, each list sorted and once "
            "each. The wall's name is not in it (recorded on each call).",
            "01-an-answer-from-a-list", {"lm": LM, "wall": WALL}),
        "10-the-same-wall-by-another-name": case(
            "Another wall that lets the same through, with the same limits: the same setup as 09.",
            "01-an-answer-from-a-list", {"lm": LM, "wall": {**WALL, "name": "bubblewrap",
                                                             "domains": list(reversed(WALL["domains"]))}}),
        "11-a-function-with-reasoning": case(
            "Another function (module cot): its own version in its setup.", "07-reasoning-first", {"lm": LM}),
    }
    s = {k: v["expect"]["setup"] for k, v in out.items()}
    assert s["01-a-model-and-defaults"] == s["02-a-bare-name-is-its-route"] == s["05-what-is-not-in-a-setup"]
    assert s["01-a-model-and-defaults"] != s["03-another-model-another-setup"]
    assert s["09-a-wall-that-says-what-it-lets-through"] == s["10-the-same-wall-by-another-name"]
    assert len(set(s.values())) == len(s) - 3                  # every other pair differs
    return out
