"""The cases in partial/: the answer so far, read from a call's answer text by
the answer's shape, written from the rules in ../streaming.md, *The answer
so far*.

Each case: {"description", "shape": the answer's shape (its interface's
last output), "checks": [{"text", "expect": {"value": the answer so far}
or {"nothing": true}}]}. ``nothing`` is no value yet (a stream's partial is
then absent: Python ``None``, TypeScript ``undefined``); ``value`` may be
JSON ``null`` (an answer that is null so far).

The reader here is written from the rules alone, not from an
implementation: it walks the text once, by the shape, and gives up (nothing)
on any text that cannot be the start of JSON.
"""

SPACE = " \t\n\r"                                  # JSON's white space
NOTHING = object()


class NotJSON(Exception):
    pass


def resolve(shape, root):
    """A shape as the reader follows it: $ref followed, an anyOf of one shape and null read as that shape,
    any other anyOf (or what it cannot follow) {}."""
    seen = set()
    while isinstance(shape, dict) and isinstance(shape.get("$ref"), str):
        ref = shape["$ref"]
        if ref in seen or not ref.startswith("#/$defs/"):
            return {}
        seen.add(ref)
        shape = (root.get("$defs") or {}).get(ref[len("#/$defs/"):], {})
    if not isinstance(shape, dict):
        return {}
    if isinstance(shape.get("anyOf"), list):
        options = [o for o in shape["anyOf"] if o != {"type": "null"}]
        if len(options) == 1 and len(shape["anyOf"]) == 2:
            return resolve(options[0], root)
        return {}
    return shape


def is_choice(shape) -> bool:
    return "enum" in shape or "const" in shape


def member_shape(shape, key, root):
    props = shape.get("properties")
    if isinstance(props, dict) and key in props:
        return resolve(props[key], root)
    extra = shape.get("additionalProperties")
    return resolve(extra, root) if isinstance(extra, dict) else {}


def item_shape(shape, i, root):
    prefix = shape.get("prefixItems")
    if isinstance(prefix, list) and i < len(prefix):
        return resolve(prefix[i], root)
    items = shape.get("items")
    return resolve(items, root) if isinstance(items, dict) else {}


ESCAPES = {'"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}
HEX = set("0123456789abcdefABCDEF")


def read_string(s, i):
    """(text so far, index after, closed) for the string whose quote is s[i]."""
    out, i = [], i + 1
    while i < len(s):
        c = s[i]
        if c == '"':
            return "".join(out), i + 1, True
        if c != "\\":
            out.append(c)
            i += 1
            continue
        if i + 1 >= len(s):
            break                                    # an escape still being written
        e = s[i + 1]
        if e in ESCAPES:
            out.append(ESCAPES[e])
            i += 2
            continue
        if e != "u":
            raise NotJSON("bad escape")
        digits = s[i + 2:i + 6]
        if any(d not in HEX for d in digits):
            raise NotJSON("bad escape")
        if len(digits) < 4:
            break
        code = int(digits, 16)
        if 0xD800 <= code < 0xDC00:                  # a pair: its second half first
            rest = s[i + 6:i + 12]
            if len(rest) < 6:
                if rest and not "\\u".startswith(rest[:2]) or any(d not in HEX for d in rest[2:]):
                    raise NotJSON("bad escape")
                break
            if rest[:2] != "\\u" or any(d not in HEX for d in rest[2:]):
                raise NotJSON("bad escape")
            low = int(rest[2:], 16)
            if not 0xDC00 <= low < 0xE000:
                raise NotJSON("bad escape")
            out.append(chr(0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00)))
            i += 12
            continue
        out.append(chr(code))
        i += 6
    return "".join(out), len(s), False


def skip(s, i):
    while i < len(s) and s[i] in SPACE:
        i += 1
    return i


def read_value(s, i, shape, root):
    """(value or NOTHING, index after, complete)."""
    i = skip(s, i)
    if i >= len(s):
        return NOTHING, i, False
    c = s[i]
    if c == '"':
        text, i, closed = read_string(s, i)
        if not closed and is_choice(shape):
            return NOTHING, i, False                 # "med" is no value of a choice
        return text, i, closed
    if c == "{":
        obj, i = {}, i + 1
        while True:
            i = skip(s, i)
            if i >= len(s):
                return obj, i, False
            if s[i] == "}":
                return obj, i + 1, True
            if s[i] == ",":
                i += 1
                continue
            if s[i] != '"':
                raise NotJSON("a key is a string")
            key, i, closed = read_string(s, i)
            if not closed:
                return obj, i, False
            i = skip(s, i)
            if i >= len(s):
                return obj, i, False
            if s[i] != ":":
                raise NotJSON("expected :")
            value, i, done = read_value(s, i + 1, member_shape(shape, key, root), root)
            if value is not NOTHING:
                obj[key] = value
            if not done:
                return obj, i, False
    if c == "[":
        items, i = [], i + 1
        while True:
            i = skip(s, i)
            if i >= len(s):
                return items, i, False
            if s[i] == "]":
                return items, i + 1, True
            if s[i] == ",":
                i += 1
                continue
            value, i, done = read_value(s, i, item_shape(shape, len(items), root), root)
            if value is not NOTHING:
                items.append(value)
            if not done:
                return items, i, False
    j = i
    while j < len(s) and s[j] not in ",]}" + SPACE:
        j += 1
    if j >= len(s):
        return NOTHING, j, False                     # 1 may become 12
    word = s[i:j]
    if word in ("true", "false", "null"):
        return {"true": True, "false": False, "null": None}[word], j, True
    import re
    if not re.fullmatch(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?", word):
        raise NotJSON(f"not a literal: {word!r}")
    number = float(word) if any(x in word for x in ".eE") else int(word)
    return number, j, True


def kind(shape) -> str:
    s = resolve(shape, shape)
    t = s.get("type")
    types = set(t if isinstance(t, list) else [t]) - {"null"}
    if types == {"string"} and not is_choice(s):
        return "text"
    if types and types <= {"object", "array"}:
        return "json"
    return "none"


def so_far(shape, text) -> dict:
    k = kind(shape)
    if k == "text":
        return {"value": text}
    if k == "none":
        return {"nothing": True}
    s = text.strip(SPACE)
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else ""
        if s.rstrip(SPACE).endswith("```"):
            s = s.rstrip(SPACE)[:-3]
        s = s.strip(SPACE)
    top = resolve(shape, shape)
    t = top.get("type")
    starts = {"object": "{", "array": "["}
    opens = "".join(starts[x] for x in (t if isinstance(t, list) else [t]) if x in starts)
    if not s or s[0] not in opens:
        return {"nothing": True}                    # a record starts with {, a list with [
    try:
        value, _i, _done = read_value(s, 0, resolve(shape, shape), shape)
    except NotJSON:
        return {"nothing": True}
    return {"nothing": True} if value is NOTHING else {"value": value}


S, N = {"type": "string"}, {"type": "number"}
SEVERITY = {"enum": ["low", "medium", "high"]}
INSIGHT = {"type": "object", "properties": {"title": S, "severity": SEVERITY, "pages": {"type": "integer"},
                                            "fixed": {"type": "boolean"}},
           "required": ["title", "severity", "pages", "fixed"], "additionalProperties": False}
REPORT = {"type": "object", "properties": {"summary": S, "insights": {"type": "array", "items": INSIGHT}},
          "required": ["summary", "insights"], "additionalProperties": False}
REFS = {"type": "object", "properties": {"insights": {"type": "array", "items": {"$ref": "#/$defs/Insight"}},
                                         "best": {"anyOf": [{"$ref": "#/$defs/Insight"}, {"type": "null"}]}},
        "required": ["insights", "best"], "$defs": {"Insight": INSIGHT}}


def case(description, shape, texts):
    return {"description": description, "shape": shape,
            "checks": [{"text": t, "expect": so_far(shape, t)} for t in texts]}


def cases() -> dict:
    out = {
        "01-text-is-its-text": case(
            "A text answer (string, or string or null, no enum): the text so far, whatever it holds.",
            {"anyOf": [S, {"type": "null"}]}, ["", "Hel", "Hello, {\"not\": json"]),
        "02-other-answers-wait-for-done": case(
            "A number, a choice or a boolean: nothing until done, even when the text looks whole.",
            SEVERITY, ["med", "medium"]),
        "03-a-number-waits-for-done": case("A number answer: nothing until done.", N, ["1", "12"]),
        "04-a-record-member-by-member": case(
            "A record: each member whose key is complete and whose value has started, with its value so far. A "
            "string shows its text so far; a choice (enum) only once its closing quote came; a number, true, "
            "false or null once a character after it ends it.",
            REPORT,
            ["", "{", '{"summ', '{"summary"', '{"summary": ', '{"summary": "Che', '{"summary": "Checkout hides',
             '{"summary": "Checkout hides fees", "insights": [', '{"summary": "x", "insights": [{"title": "Fees',
             '{"summary": "x", "insights": [{"title": "Fees", "severity": "med',
             '{"summary": "x", "insights": [{"title": "Fees", "severity": "medium"',
             '{"summary": "x", "insights": [{"title": "Fees", "severity": "medium", "pages": 1',
             '{"summary": "x", "insights": [{"title": "Fees", "severity": "medium", "pages": 12,',
             '{"summary": "x", "insights": [{"title": "Fees", "severity": "medium", "pages": 12, "fixed": fal',
             '{"summary": "x", "insights": [{"title": "Fees", "severity": "medium", "pages": 12, "fixed": false}',
             '{"summary": "x", "insights": []}', '{"summary": "x", "insights": []} and then some words']),
        "05-a-list-of-choices": case(
            "A list whose items are choices: an item still being written is left out.",
            {"type": "array", "items": SEVERITY}, ["[", '["low", "hi', '["low", "high"', '["low", "high"]']),
        "06-references-and-null": case(
            "$ref is followed, and an anyOf of one shape and null is read as that shape: a choice inside an "
            "Insight reached through either waits for its closing quote.",
            REFS,
            ['{"insights": [{"title": "A", "severity": "lo', '{"insights": [], "best": {"severity": "hi',
             '{"insights": [], "best": null', '{"insights": [], "best": null}']),
        "07-escapes": case(
            "Escapes are read; an escape still being written is left out (a lone backslash, \\u with fewer than "
            "four digits, a surrogate's first half waiting for its second).",
            {"type": "object", "properties": {"t": S}},
            ['{"t": "a\\n', '{"t": "a\\', '{"t": "\\u00e9t', '{"t": "\\u00e', '{"t": "\\ud83d',
             '{"t": "\\ud83d\\ude00!"}', '{"t": "\\x"}']),
        "08-fences-and-what-is-not-json": case(
            "A text in a fence is read after its first line break, a closing fence left out. A text that cannot "
            "be the start of JSON gives nothing: prose first, a key that is not a string, a missing colon, a "
            "word that is no literal.",
            {"type": "object", "additionalProperties": SEVERITY},
            ['```json\n{"a": "low", "b": "hi', '```json\n{"a": "low"}\n```', "```", "Sure! {", "{a: 1}",
             '{"a" "low"}', '{"a": lo}', '[]']),
        "09-a-tuple-and-a-map": case(
            "Each item's shape through prefixItems then items; each member's through properties then "
            "additionalProperties. Where neither says, the shape is {}: a string shows as it is written.",
            {"type": "array", "prefixItems": [S, SEVERITY], "items": {"type": "object"}},
            ['["abc", "lo', '["abc", "low", {"x": "anything so', '["abc", "low", {"x": 1}, {"y": "z"}]']),
    }
    return out
