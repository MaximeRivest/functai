"""Stage 3: views, serving a program, and using it from elsewhere. The contract is contract/serving.md and
contract/streaming.md, *Views*."""

import asyncio
import json
import time
from typing import Any

import lm15
import pytest

import functai
from functai import ai, calllog, module, views
from functai.serving import Service, parse_position

XML = "<result>\n{}\n</result>"


def sse(body):
    """The events of a Server-Sent Events body."""
    chunks = b"".join(body).decode()
    out = []
    for block in chunks.split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        if "data" in lines:
            out.append((lines["id"], json.loads(lines["data"])))
    return out


@ai
def topic(message: str) -> str:
    """What is the customer writing about?"""


@ai
def answer(message: str, topic: str) -> str:
    """Answer the customer kindly."""


@module(answer_from=answer)
def support(message: str) -> str:
    return answer(message, topic(message))


@pytest.fixture
def model(fake):
    def respond(req):
        if "writing about" in str(req.system):
            return [lm15.ThinkingPart("private thoughts"), lm15.TextPart(XML.format("shipping"))]
        return XML.format("Your parcel is in Leeds.")
    return fake(responder=respond)


# ------------------------------------------------------------------ views


def test_the_outside_view_shows_the_boundary_only(model):
    s = support.stream("where is B-2210?")
    assert s.result == "Your parcel is in Leeds."
    full = [e.to_dict() for e in s.events()]
    outside = list(s.events(view="outside"))
    kinds = [e["kind"] for e in outside]
    assert kinds[0] == "started" and kinds[-1] == "done" and "thinking" not in kinds
    assert {e["call"] for e in outside} == {full[0]["call"]}               # everything addressed to the boundary
    text = "".join(e["text"] for e in outside if e["kind"] == "text")
    assert text == "Your parcel is in Leeds." and "shipping" not in json.dumps(outside)
    assert "file" not in outside[0]["program"] and "line" not in outside[0]["program"]
    assert outside[0]["after"] is None and all(b["after"] == {"writer": a["writer"], "seq": a["seq"]}
                                               for a, b in zip(outside, outside[1:]))
    assert all(any(f["seq"] == e["seq"] for f in full) for e in outside)   # numbers kept, never invented


def test_the_outside_view_of_a_failure_names_its_type_only(model):
    @ai
    def breaks(message: str) -> str:
        """Break."""

    model.responder = lambda req: (_ for _ in ()).throw(ValueError("the carrier log says: SECRET"))
    s = breaks.stream("x")
    with pytest.raises(ValueError):
        s.result
    shown = []
    with pytest.raises(ValueError):                                     # a cursor ends with the call's error
        for e in s.events(view="outside"):
            shown.append(e)
    last = shown[-1]
    assert last["kind"] == "failed" and last["error"] == {"type": "ValueError"} and last["content"] is False


def test_views_are_checked():
    with pytest.raises(ValueError):
        views.View("everything")


# ------------------------------------------------------------------ the service


def test_a_service_describes_and_answers(model):
    svc = Service(support, keys=["k1"])
    assert svc.handle("GET", "/interface", {}).status == 401
    d = json.loads(svc.handle("GET", "/interface", {"authorization": "Bearer k1"}).body)
    assert d["functai_interface"] == 1 and d["name"] == "support" and d["kind"] == "module"
    assert d["interface"]["inputs"][0]["name"] == "message"
    assert svc.handle("POST", "/call", {}, b'{"inputs": {"message": "x"}}').status == 401
    r = svc.handle("POST", "/call", {"Authorization": "Bearer k1"}, b'{"inputs": {"message": "x"}}')
    got = json.loads(r.body)
    assert r.status == 200 and got["value"] == "Your parcel is in Leeds." and got["outputs"] == {"result": got["value"]}
    bad = svc.handle("POST", "/call", {"authorization": "Bearer k1"}, b'{"inputs": {"mesage": "x"}}')
    assert bad.status == 422 and json.loads(bad.body)["error"]["code"] == "interface-input"
    assert json.loads(svc.handle("GET", "/openapi.json", {"authorization": "Bearer k1"}).body)["openapi"] == "3.1.0"
    assert b"<form" in svc.handle("GET", "/", {}).body


def test_a_served_ai_function_refuses_wrong_inputs_before_anything_runs(model):
    svc = Service(topic)
    asked = len(model.requests)
    for body in (b'{"inputs": {}}', b'{"inputs": {"mesage": "x"}}', b'{"inputs": {"message": null}}'):
        for route in ("/call", "/stream", "/conversations/c/turns"):
            r = svc.handle("POST", route, {}, body)
            err = json.loads(r.body)["error"]
            assert (r.status, err["type"], err["code"]) == (422, "InterfaceError", "interface-input"), (route, body)
            assert err["field"] == "message" or "mesage" in err["message"]
    assert len(model.requests) == asked                                      # no model was asked
    assert json.loads(svc.handle("POST", "/call", {}, b'{"inputs": {"message": "x"}}').body)["value"] == "shipping"


def test_a_service_streams_the_outside_view(model):
    svc = Service(support)
    r = svc.handle("POST", "/stream", {}, b'{"inputs": {"message": "x"}}')
    events = sse(r.body)
    assert events[0][1]["kind"] == "started" and events[-1][1]["kind"] == "done"
    assert all(parse_position(i) == {"writer": e["writer"], "seq": e["seq"]} for i, e in events)
    assert "shipping" not in json.dumps([e for _i, e in events])


def test_a_served_conversation_is_read_again_after_a_reload(model, tmp_path):
    svc = Service(answer, store=tmp_path)
    r = svc.handle("POST", "/conversations/c1/turns", {}, json.dumps(
        {"inputs": {"message": "hi", "topic": "t"}, "request_id": "p1", "wait": True}).encode())
    turn = json.loads(r.body)
    assert r.status == 201 and turn["state"] == "done" and turn["value"] == "Your parcel is in Leeds."
    again = json.loads(svc.handle("POST", "/conversations/c1/turns", {}, json.dumps(
        {"inputs": {"message": "hi", "topic": "t"}, "request_id": "p1", "wait": True}).encode()).body)
    assert again["turn"] == turn["turn"]                                  # a double click: one turn
    events = sse(svc.handle("GET", f"/conversations/c1/turns/{turn['turn']}/events", {}).body)
    middle = events[len(events) // 2][0]
    rest = sse(svc.handle("GET", f"/conversations/c1/turns/{turn['turn']}/events", {"last-event-id": middle}).body)
    assert [e for _i, e in rest] == [e for _i, e in events][len(events) // 2 + 1:]
    listed = json.loads(svc.handle("GET", "/conversations/c1/turns", {}).body)
    assert [t["turn"] for t in listed["turns"]] == [turn["turn"]]
    assert svc.handle("GET", "/conversations/c1/turns/nope", {}).status == 404


def test_a_caller_may_answer_its_approvals(fake, tmp_path):
    done = []

    @functai.tool(effects="changes")
    def refund(order: str) -> str:
        """Refund an order."""
        done.append(order)
        return "refunded"

    @ai(tools=[refund], approve="changes")
    def clerk(message: str) -> str:
        """Help."""

    def respond(req):
        if not any(type(p).__name__ == "ToolResultPart" for m in req.messages for p in m.parts):
            return [lm15.ToolCallPart(id="c1", name="refund", input={"order": "B-1"})]
        return XML.format("all done")

    fake(responder=respond)
    svc = Service(clerk, store=tmp_path, approvals="caller")
    turn = json.loads(svc.handle("POST", "/conversations/a/turns", {}, json.dumps(
        {"inputs": {"message": "refund B-1"}, "wait": True}).encode()).body)
    assert turn["state"] == "waiting" and turn["waiting"][0]["name"] == "refund" and done == []
    events = [e for _i, e in sse(svc.handle("GET", f"/conversations/a/turns/{turn['turn']}/events", {}).body)]
    assert events[-1]["kind"] == "approval" and events[-1]["to"] == "caller" and events[-1]["call"] == turn["turn"]
    r = svc.handle("POST", f"/conversations/a/turns/{turn['turn']}/approvals/1", {}, b'{"verdict": "yes"}')
    assert r.status == 202
    for _ in range(100):
        state = json.loads(svc.handle("GET", f"/conversations/a/turns/{turn['turn']}", {}).body)
        if state["state"] == "done":
            break
        time.sleep(0.05)
    assert state["value"] == "all done" and done == ["B-1"]


def test_what_cannot_be_served_is_refused(model):
    @module
    def summarize(frame: Any) -> str:
        return "two rows"

    with pytest.raises(functai.ServeError) as err:
        Service(summarize)
    assert err.value.code == "serve-opaque"
    with pytest.raises(functai.ServeError) as err:
        Service(support).serve("0.0.0.0", 0)
    assert err.value.code == "serve-keys"


def test_the_asgi_app_answers_as_the_service(model):
    app = Service(support).asgi

    async def ask(method, path, body=b""):
        sent = []
        messages = [{"type": "http.request", "body": body, "more_body": False}]

        async def receive():
            return messages.pop(0)

        async def send(msg):
            sent.append(msg)

        await app({"type": "http", "method": method, "path": path, "headers": []}, receive, send)
        return sent

    got = asyncio.run(ask("POST", "/call", b'{"inputs": {"message": "x"}}'))
    assert got[0]["status"] == 200 and json.loads(got[1]["body"])["value"] == "Your parcel is in Leeds."
    streamed = asyncio.run(ask("POST", "/stream", b'{"inputs": {"message": "x"}}'))
    body = b"".join(m.get("body", b"") for m in streamed[1:])
    assert b"event: done" in body


# ------------------------------------------------------------------ remote


def test_a_remote_program_is_a_program_again(model, tmp_path):
    functai.configure(log_calls=tmp_path)
    server = functai.serve(support, port=0, keys=["secret"], block=False)
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}"
        team = functai.remote(url, key="secret")
        assert team("where?") == "Your parcel is in Leeds." and team.version == support.version
        with pytest.raises(functai.InterfaceError):
            team(mesage="x")                                             # checked here, before anything is sent
        from functai.remote import RemoteError
        with pytest.raises(RemoteError) as err:
            functai.remote(url, key="wrong")
        assert err.value.status == 401 or err.value.code == "remote-401"
        table = team.map([{"message": "a"}, {"message": "b"}], threads=2, progress=False)
        assert [r["pred_result"] for r in table.collect().to_dicts()] == ["Your parcel is in Leeds."] * 2
        pieces = list(team.stream("where?"))
        assert "".join(pieces) == "Your parcel is in Leeds."
    finally:
        server.shutdown()
    recs = calllog.read(tmp_path)[0]
    mine = [r for r in recs if r["program"]["kind"] == "remote" and r["error"] is None]
    assert mine and mine[0]["program"]["remote"] == url
    served = {r["parent"] for r in recs if r["program"]["name"] == "support" and r["program"]["kind"] == "module"}
    assert {r["id"] for r in mine} <= served                               # the server's call names the caller's


def test_a_served_ai_function_with_lmcc_keywords_is_remote_too(fake):
    from pydantic import BaseModel, Field

    class Order(BaseModel):
        code: str = Field(pattern=r"^B-\d+$")

    @ai
    def parse(text: str) -> Order:
        """The order the text names."""

    fake(responder=lambda req: XML.format('{"code": "B-1"}'))
    server = functai.serve(parse, port=0, block=False)
    try:
        remote = functai.remote(f"http://127.0.0.1:{server.server_address[1]}")
        assert remote("order B-1") == {"code": "B-1"}
    finally:
        server.shutdown()
