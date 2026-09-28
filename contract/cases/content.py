"""The cases in content/: which values a call record keeps, written from
../calls.md, section "Content".

A case file: {"description", "interface": {"inputs": [names], "outputs":
[names]}, "layers": [{"where": "own" | "block" | "configure",
"log_content": true | false | {name: bool}}, ...] (closest first),
"environment": the value of FUNCTAI_LOG_CONTENT or null (unset),
"record": the call's whole record, "expect": {"record": the record as
written} or {"refuses": "log-content-field", "field": name}}.

An implementation passes a case when, with those settings, the record it
writes for that call is ``expect.record`` (compared as JSON), or when
defining the program with its own setting refuses as expected.
"""

import copy

from common import canonical

OFF = {"0", "false", "no", "off"}
ALWAYS_KEPT = ("functai_call", "id", "parent", "root", "program", "started", "seconds", "sizes", "model", "usage",
               "confidence", "caller", "process", "saw", "escalated", "truncated")


# ------------------------------------------------------------------ the rules


def refusal(interface: dict, layers: list):
    names = interface["inputs"] + interface["outputs"]
    for layer in layers:
        if layer["where"] == "own" and isinstance(layer["log_content"], dict):
            for name in layer["log_content"]:
                if name not in names:
                    return {"refuses": "log-content-field", "field": name}
    return None


def kept(interface: dict, layers: list, environment) -> dict:
    """For each field, whether its value is written: the first layer that answers decides."""
    env = None if environment is None or environment.strip().lower() not in OFF else False
    out = {}
    for name in interface["inputs"] + interface["outputs"]:
        decision = None
        for layer in layers:
            v = layer["log_content"]
            if isinstance(v, bool):
                decision = v
                break
            if isinstance(v, dict) and name in v:
                decision = v[name]
                break
        if decision is None:
            decision = env if env is not None else True
        out[name] = decision
    return out


def written(record: dict, interface: dict, keep: dict) -> dict:
    if all(keep.values()):
        return copy.deepcopy(record)
    some = any(keep.values())
    out = {k: copy.deepcopy(v) for k, v in record.items() if k in ALWAYS_KEPT}
    out["content"] = False
    answer = record["program"]["answer"]
    if some:
        out["omitted"] = {"inputs": [n for n in interface["inputs"] if not keep[n]],
                          "outputs": [n for n in interface["outputs"] if not keep[n]]}
        inputs = {k: v for k, v in record["inputs"].items() if keep[k]}
        if inputs:
            out["inputs"] = inputs
        if record["outputs"] is None:
            out["outputs"] = None
        else:
            outputs = {k: v for k, v in record["outputs"].items() if keep[k]}
            if outputs:
                out["outputs"] = outputs
        if "returned" in record and keep[answer]:
            out["returned"] = record["returned"]
        if "probabilities" in record:
            probabilities = {k: v for k, v in record["probabilities"].items() if keep[k]}
            if probabilities:
                out["probabilities"] = probabilities
    error = record["error"]
    out["error"] = None if error is None else {k: v for k, v in error.items() if k != "message"}
    out["exchanges"] = [{k: v for k, v in ex.items() if k not in ("request", "response")}
                        for ex in record["exchanges"]]
    order = list(record) + ["omitted"]
    return {k: out[k] for k in sorted(out, key=order.index)}


# ------------------------------------------------------------------ a call


INTERFACE = {"inputs": ["transcript", "question"], "outputs": ["summary", "result"]}
TRANSCRIPT = "Ana: the build is red again.\nBen: it is the flaky upload test.\n" * 3
QUESTION = "Is the build broken for a real reason?"


def size(value) -> int:
    return len(canonical(value))


def record(*, failed=False) -> dict:
    inputs = {"transcript": TRANSCRIPT, "question": QUESTION}
    outputs = None if failed else {"summary": "A flaky upload test fails the build.", "result": "no"}
    rec = {"functai_call": 1, "id": "01926a8e-0001-7000-8000-000000000000", "parent": None,
           "root": "01926a8e-0001-7000-8000-000000000000",
           "program": {"name": "triage_build", "kind": "ai", "module": "ci", "version": "sha256:" + "1" * 64,
                       "signature": "sha256:" + "5" * 64, "answer": "result"},
           "started": "2026-09-28T10:00:00.000000Z", "seconds": 0.8, "content": True,
           "inputs": inputs, "outputs": outputs}
    if not failed:
        rec["returned"] = "No"
        rec["probabilities"] = {"summary": {"A flaky upload test fails the build.": 0.61},
                                "result": {"no": 0.93, "yes": 0.07}}
    rec["sizes"] = {"inputs": {k: size(v) for k, v in inputs.items()},
                    "outputs": {k: size(v) for k, v in (outputs or {}).items()}}
    rec["error"] = ({"type": "Refusal", "message": "parse-value: 'maybe' is not one of yes, no", "code": "parse-value"}
                    if failed else None)
    rec.update(model="gpt-4.1-mini", usage={"input_tokens": 120, "output_tokens": 14, "total_tokens": 134},
               confidence=None if failed else 0.61,
               exchanges=[{"model": "gpt-4.1-mini", "provider": "openai", "started": "2026-09-28T10:00:00.001000Z",
                           "seconds": 0.79, "cached": False, "finish": "stop",
                           "usage": {"input_tokens": 120, "output_tokens": 14, "total_tokens": 134},
                           "request": {"messages": [{"role": "user", "parts": [
                               {"type": "text", "text": f"<transcript>\n{TRANSCRIPT}\n</transcript>"}]}]},
                           "response": {"message": {"role": "assistant", "parts": [
                               {"type": "text", "text": "<result>\nmaybe\n</result>" if failed
                                else "<summary>\nA flaky upload test fails the build.\n</summary>"}]}}}],
               saw=[], caller={"kind": "script"},
               process={"host": "lambda", "pid": 1, "user": "maxime", "language": "python", "runtime": "3.13.1",
                        "functai": "1.2.0"})
    return rec


def case(description, layers=(), environment=None, *, failed=False, interface=INTERFACE):
    layers = [{"where": w, "log_content": v} for w, v in layers]
    rec = record(failed=failed)
    refused = refusal(interface, layers)
    expect = refused or {"record": written(rec, interface, kept(interface, layers, environment))}
    return {"description": description, "interface": interface, "layers": layers, "environment": environment,
            "record": rec, "expect": expect}


def cases() -> dict:
    return {
        "01-everything-by-default": case("No layer and no environment: every value is written."),
        "02-false-writes-no-value": case(
            "log_content false: sizes, times, tokens, ids, and no value, message, request or reply "
            "(the form written before 2026-09-28: no omitted).", [("own", False)]),
        "03-all-but-one-input": case(
            "A block's map keeps every value but the transcript's: content false, omitted names it; the "
            "exchange's request and reply go too (they hold every input and could quote it).",
            [("block", {"transcript": False})]),
        "04-a-failed-call-keeps-no-message": case(
            "A failed call with one input kept as its size only: outputs stay null, the error keeps its type "
            "and code but not its message (it can quote the reply).",
            [("block", {"transcript": False})], failed=True),
        "05-own-true-beats-a-block": case(
            "The function's own true answers for every field before a block's false is read.",
            [("own", True), ("block", False)]),
        "06-field-by-field": case(
            "Each field is decided by the closest layer that answers for it: the function's own map keeps "
            "the transcript; the block's false decides every other field.",
            [("own", {"transcript": True}), ("block", False)]),
        "07-the-environment-is-the-last-layer": case(
            "FUNCTAI_LOG_CONTENT=0 answers only for fields no setting names: configure's map keeps the "
            "question.", [("configure", {"question": True})], environment="0"),
        "08-the-answer-kept-as-its-size": case(
            "The answer is not written, so neither is what the code returned, nor its probabilities; the "
            "other output's are.", [("configure", {"result": False})]),
        "09-every-field-named": case(
            "A map that leaves out every field writes the same record as false: no omitted.",
            [("own", {"transcript": False, "question": False, "summary": False, "result": False})]),
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
            "Anything else in FUNCTAI_LOG_CONTENT, empty included, leaves the default: every value.",
            environment=""),
    }
