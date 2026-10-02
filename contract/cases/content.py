"""The cases in content/: which values a call record keeps, written from
../calls.md, section "Content".

A case file: {"description", "fields": {"inputs": [names], "outputs":
[names], "added": [names]}, "layers": [{"where": "own" | "block" |
"configure", "log_content": true | false | {key: bool}}, ...] (closest
first), "environment": the value of FUNCTAI_LOG_CONTENT or null (unset),
"record": the call's whole record, "expect": {"record": the record as
written} or {"refuses": "log-content-field", "field": key}}.

``fields`` are the call's fields: its interface's inputs and outputs, and
``added``, the outputs FunctAI adds to an AI function (``reasoning``,
``calls``); ``added`` are among ``outputs`` too, in the record's order.
The ``tools`` input FunctAI adds is not a field of a call.

An implementation passes a case when, with those settings, the record it
writes for that call is ``expect.record`` (compared as JSON), or when the
setting is refused as expected (a program's own setting when the program is
defined or loaded; a block's or configure's when it is set).
"""

import copy
import re

from common import canonical, sha

OFF = {"0", "false", "no", "off"}
NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")             # matched whole (fullmatch): nothing after it
ALWAYS_KEPT = ("functai_call", "id", "parent", "root", "program", "started", "seconds", "sizes", "model", "usage",
               "confidence", "caller", "process", "saw", "escalated", "truncated", "journal")
DROPPED_FROM_EXCHANGES = ("request", "response", "request_hash")
ERROR_KEPT = ("type", "code")        # of an error, when content is not whole: never a member this contract does not name


# ------------------------------------------------------------------ the rules


def refusal(fields: dict, layers: list):
    names = fields["inputs"] + fields["outputs"]
    for layer in layers:
        v = layer["log_content"]
        if not isinstance(v, dict):
            continue
        for key in v:
            if key != "*" and not NAME.fullmatch(key):
                return {"refuses": "log-content-field", "field": key}
            if layer["where"] == "own" and key != "*" and key not in names:
                return {"refuses": "log-content-field", "field": key}
    return None


def says_drop(v, name: str) -> bool:
    """Whether one layer's setting drops a field: false, or a map that names it
    false, or whose "*" is false and does not name it."""
    if isinstance(v, bool):
        return not v
    if name in v:
        return not v[name]
    return v.get("*") is False


def kept(fields: dict, layers: list, environment) -> dict:
    """For each field, whether its value is written: only when no layer drops it,
    and, for an added field, only when no input or output of the program is
    dropped (an added field can quote any input and anticipate any output).
    Dropping one added field drops only it (design/09, B8)."""
    env_off = environment is not None and environment.strip().lower() in OFF
    out = {}
    for name in fields["inputs"] + fields["outputs"]:
        out[name] = not env_off and not any(says_drop(layer["log_content"], name) for layer in layers)
    own = [n for n in fields["inputs"] + fields["outputs"] if n not in fields["added"]]
    if not all(out[n] for n in own):
        for n in fields["added"]:
            out[n] = False
    return out


def written(record: dict, fields: dict, keep: dict) -> dict:
    if all(keep.values()):
        return copy.deepcopy(record)
    out = {}
    for k, v in record.items():
        if k in ALWAYS_KEPT:
            out[k] = copy.deepcopy(v)
        if k == "content":
            out["content"] = False
            out["omitted"] = {"inputs": [n for n in fields["inputs"] if not keep[n]],
                              "outputs": [n for n in fields["outputs"] if not keep[n]]}
    answer = record["program"]["answer"]
    inputs = {k: v for k, v in record["inputs"].items() if keep[k]}
    if inputs:
        out["inputs"] = inputs
    if record["outputs"] is None:
        out["outputs"] = None
    else:
        outputs = {k: v for k, v in record["outputs"].items() if keep[k]}
        if outputs:
            out["outputs"] = outputs
    if "described" in record:
        described = {k: [n for n in v if keep[n]] for k, v in record["described"].items()}
        if any(described.values()):
            out["described"] = described
    # no "returned": code can put any value of the call into what it returns (calls.md, Content)
    if "probabilities" in record:
        probabilities = {k: v for k, v in record["probabilities"].items() if keep[k]}
        if probabilities:
            out["probabilities"] = probabilities
    error = record["error"]
    out["error"] = None if error is None else {k: v for k, v in error.items() if k in ERROR_KEPT}
    exchanges = []
    for ex in record["exchanges"]:
        ex = {k: copy.deepcopy(v) for k, v in ex.items() if k not in DROPPED_FROM_EXCHANGES}
        if "error" in ex:
            ex["error"] = {k: v for k, v in ex["error"].items() if k in ERROR_KEPT}
        exchanges.append(ex)
    out["exchanges"] = exchanges
    order = list(record)
    order.insert(order.index("content") + 1, "omitted")
    return {k: out[k] for k in sorted(out, key=order.index)}


# ------------------------------------------------------------------ a call


FIELDS = {"inputs": ["transcript", "question"], "outputs": ["reasoning", "summary", "result"], "added": ["reasoning"]}
TOOLS = {"inputs": ["transcript", "question"], "outputs": ["reasoning", "calls", "summary", "result"],
         "added": ["reasoning", "calls"]}
CALLS = [{"id": "call_1", "name": "ci_log", "input": {"job": 4412}}]
TRANSCRIPT = "Ana: the build is red again.\nBen: it is the flaky upload test.\n" * 3
QUESTION = "Is the build broken for a real reason?"
CALL = "01926a8e-0001-7000-8000-000000000000"


def size(value) -> int:
    return len(canonical(value))


def record(*, failed=False, tools=False) -> dict:
    inputs = {"transcript": TRANSCRIPT, "question": QUESTION}
    outputs = None if failed else {"reasoning": "Ben says the upload test is flaky, so the build is not broken.",
                                   "summary": "A flaky upload test fails the build.", "result": "no"}
    if tools and outputs:
        outputs = {"reasoning": outputs["reasoning"], "calls": CALLS, "summary": outputs["summary"],
                   "result": outputs["result"]}
    rec = {"functai_call": 2, "id": CALL, "parent": None, "root": CALL,
           "program": {"name": "triage_build", "kind": "ai", "module": "ci", "version": "sha256:" + "1" * 64,
                       "signature": "sha256:" + "5" * 64, "interface": "sha256:" + "6" * 64, "answer": "result"},
           "started": "2026-09-28T10:00:00.000000Z", "seconds": 1.6, "content": True,
           "inputs": inputs, "outputs": outputs}
    if not failed:
        rec["returned"] = "No"
        rec["probabilities"] = {"summary": {"A flaky upload test fails the build.": 0.61},
                                "result": {"no": 0.93, "yes": 0.07}}
    rec["sizes"] = {"inputs": {k: size(v) for k, v in inputs.items()},
                    "outputs": {k: size(v) for k, v in (outputs or {}).items()}}
    rec["error"] = ({"type": "Refusal", "message": "parse-value: 'maybe' is not one of yes, no", "code": "parse-value"}
                    if failed else None)
    request = {"messages": [{"role": "user", "parts": [
        {"type": "text", "text": f"<transcript>\n{TRANSCRIPT}\n</transcript>\n<question>\n{QUESTION}\n</question>"}]}]}
    first = {"model": "gpt-4.1-mini", "provider": "openai", "started": "2026-09-28T10:00:00.001000Z",
             "seconds": 0.79, "cached": False, "finish": "stop",
             "usage": {"input_tokens": 120, "output_tokens": 14, "total_tokens": 134},
             "error": {"type": "Refusal", "code": "parse-value",
                       "message": "parse-value: 'Ben says the upload test is flaky' is not one of yes, no"},
             "request": request, "request_hash": sha(request),
             "response": {"message": {"role": "assistant", "parts": [
                 {"type": "text", "text": "<result>\nBen says the upload test is flaky\n</result>"}]}}}
    second = {"model": "gpt-4.1-mini", "provider": "openai", "started": "2026-09-28T10:00:00.801000Z",
              "seconds": 0.8, "cached": False, "finish": "stop",
              "usage": {"input_tokens": 150, "output_tokens": 30, "total_tokens": 180},
              "request": request, "request_hash": sha(request),
              "response": {"message": {"role": "assistant", "parts": [
                  {"type": "text", "text": "<result>\nmaybe\n</result>" if failed
                   else "<summary>\nA flaky upload test fails the build.\n</summary>\n<result>\nno\n</result>"}]}}}
    rec.update(model="gpt-4.1-mini", usage={"input_tokens": 270, "output_tokens": 44, "total_tokens": 314},
               confidence=None if failed else 0.61, exchanges=[first, second],
               saw=[], caller={"kind": "script"},
               process={"host": "lambda", "pid": 1, "user": "maxime", "language": "python", "runtime": "3.13.1",
                        "functai": "1.2.0"})
    return rec


def case(description, layers=(), environment=None, *, failed=False, fields=FIELDS, error_extra=None):
    layers = [{"where": w, "log_content": v} for w, v in layers]
    rec = record(failed=failed, tools=fields is TOOLS)
    if error_extra:
        rec["error"].update(error_extra)
        for ex in rec["exchanges"]:
            if "error" in ex:
                ex["error"].update(error_extra)
    refused = refusal(fields, layers)
    expect = refused or {"record": written(rec, fields, kept(fields, layers, environment))}
    return {"description": description, "fields": fields, "layers": layers, "environment": environment,
            "record": rec, "expect": expect}


def cases() -> dict:
    return {
        "01-everything-by-default": case("No layer and no environment: every value is written."),
        "02-false-writes-no-value": case(
            "log_content false: sizes, times, tokens, ids; no value, no message (the failed attempt's included), "
            "no request, reply or request hash; omitted names every field.", [("own", False)]),
        "03-all-but-one-input": case(
            "A block's map drops the transcript: omitted names it, and the reasoning FunctAI added (it can quote "
            "any input); the exchanges' requests, replies, request hashes and error messages go too.",
            [("block", {"transcript": False})]),
        "04-a-failed-call-keeps-no-message": case(
            "A failed call with one input dropped: outputs stay null, the error keeps its type and code but not "
            "its message (it can quote the reply).",
            [("block", {"transcript": False})], failed=True),
        "05-a-host-false-beats-a-program-true": case(
            "A field is written only when no layer drops it: a block's false wins over the function's own "
            "true. true only says a layer does not object.",
            [("own", True), ("block", False)]),
        "06-true-never-keeps-what-another-layer-drops": case(
            "The function's own map says true for the transcript; the block's false still drops every field.",
            [("own", {"transcript": True}), ("block", False)]),
        "07-the-environment-is-absolute": case(
            "FUNCTAI_LOG_CONTENT=0 drops every field, whatever the settings say (configure's true for the "
            "question included).", [("configure", {"question": True})], environment="0"),
        "08-the-answer-dropped": case(
            "The answer is dropped, so neither what the code returned nor its probabilities are written; the "
            "other output's are. The added reasoning goes too.", [("configure", {"result": False})]),
        "21-what-the-code-returned-goes-with-any-field": case(
            "Another output (summary) is dropped and the answer kept: what the code returned is not written "
            "either, since code can return any value of the call (return summary, _ai). The reasoning goes "
            "too: it can anticipate any output.", [("own", {"summary": False})]),
        "09-every-field-named": case(
            "A map that drops every field writes the same record as false.",
            [("own", {"transcript": False, "question": False, "reasoning": False, "summary": False,
                      "result": False})]),
        "10-a-misspelt-name-refuses": case(
            "A function's own map naming a field it does not have refuses when the function is defined: "
            "the transcript would otherwise be written.", [("own", {"transcrpit": False})]),
        "11-a-block-names-other-fields": case(
            "A block's map applies to the calls that have the field and to no other: a name this function "
            "lacks changes nothing.", [("block", {"notes": False})]),
        "12-environment-words": case(
            "Only 0, false, no and off (any case, white space around) turn content off from the "
            "environment.", environment=" Off "),
        "13-other-environment-words": case(
            "Anything else in FUNCTAI_LOG_CONTENT, empty included, drops nothing.",
            environment=""),
        "14-keep-only-these": case(
            "\"*\": false drops every field the map does not name true: a host's list of what may be kept, "
            "which a misspelt name cannot widen.",
            [("configure", {"*": False, "question": True, "result": True})]),
        "15-the-reasoning-alone": case(
            "The function's own map drops the reasoning FunctAI added, by its name: every other value is "
            "written but what the code returned (it could hold anything); the replies go (they hold the "
            "reasoning).", [("own", {"reasoning": False})]),
        "16-a-key-that-is-not-a-name": case(
            "A map key that is neither a field name nor \"*\" refuses wherever it is set (keys of another form "
            "are kept for later).", [("block", {"#private": False})]),
        "17-the-tool-calls-dropped": case(
            "A function with tools and reasoning: FunctAI adds two fields, calls and reasoning. Dropping the "
            "calls drops only them: the reasoning is written (an added field goes with an input or output of "
            "the program, not with another added field).", [("own", {"calls": False})], fields=TOOLS),
        "18-the-reasoning-dropped-with-tools": case(
            "Dropping the reasoning drops only it: the calls are written, and every other value but what the "
            "code returned.", [("configure", {"reasoning": False})], fields=TOOLS),
        "22-an-input-dropped-with-tools": case(
            "Dropping an input drops both added fields: the reasoning and the tool calls routinely repeat an "
            "input word for word.", [("block", {"transcript": False})], fields=TOOLS),
        "19-the-tools-input-is-not-a-field": case(
            "The tools input FunctAI adds to a function's signature is the function's state (its tools, in its "
            "version), not a field of a call: no record holds it, and a function's own map that names it refuses.",
            [("own", {"tools": False})], fields=TOOLS),
        "20-an-error-member-this-contract-does-not-name": case(
            "A failed call whose errors carry a member this contract does not name (a later version's "
            "diagnostic, which may quote a value): with content not whole, an error keeps only its type and "
            "code, the call's and each exchange's alike.",
            [("own", False)], failed=True, error_extra={"detail": {"body": "Ana: the build is red again."}}),
    }
