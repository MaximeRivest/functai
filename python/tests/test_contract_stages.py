"""The contract's cases for stages 1.2 to 5 (contract/cases/replies, conversations, tools, views, context),
run through this implementation."""

import pytest

from functai import calllog, conversations, replies, tools, views
from functai.errors import SawError

from contract_support import case_files, load


def cases(folder):
    return case_files(folder)


@pytest.mark.parametrize("path", cases("replies"), ids=lambda p: p.stem)
def test_reply_key(path):
    from lm15.serde import request_from_dict
    case = load(path)
    request = request_from_dict(case["request"])
    assert replies.key(request, case["replicate"]) == case["expect"]["key"]


def _log(records):
    log = conversations._Log()
    log.apply(records)
    return log


@pytest.mark.parametrize("path", [p for p in cases("conversations") if p.stem.startswith("state")],
                         ids=lambda p: p.stem)
def test_conversation_state(path, monkeypatch):
    case = load(path)
    monkeypatch.setattr(conversations, "_now", lambda: conversations._parse(case["now"]))
    log = _log(case["records"])
    expect = case["expect"]
    assert {t: st.state() for t, st in log.turns.items()} == expect["states"]
    assert log.head == expect["head"]
    head = log.turns.get(log.head)
    nxt = expect["next"]
    state = head.state() if head else None
    if "refuses" in nxt:
        assert state == "waiting"
    elif "waits" in nxt:
        assert state == "running" and log.head == nxt["waits"]
    else:
        assert log.done_on(log.head) == nxt["parent"]
    assert {t: [a["invocation"] for a in st.unanswered()] for t, st in log.turns.items() if st.unanswered()} \
        == expect["waiting"]
    assert {t: [x["invocation"] for x in st.unfinished()] for t, st in log.turns.items() if st.unfinished()} \
        == expect["unfinished"]


@pytest.mark.parametrize("path", [p for p in cases("conversations") if p.stem.startswith("context")],
                         ids=lambda p: p.stem)
def test_conversation_context(path, monkeypatch, fake):
    from functai import ai
    case = load(path)
    monkeypatch.setattr(conversations, "_now", lambda: conversations._parse(case["now"]))
    fake("x")

    @ai
    def tutor(message: str) -> str:
        """Tutor."""

    store = conversations.stores.MemoryConversations()
    store.append("c", [{k: v for k, v in r.items() if k != "seq"} for r in case["records"]])
    rule = case["rule"]
    chat = tutor.conversation("c", store=store, context=conversations.Context(rule.get("last"),
                                                                                tuple(rule.get("without") or ())))
    log = chat._read()
    signature = calllog.signature_id(tutor._spec().signature)
    for d in log.programs.values():                  # the case's program is this one
        d["signature"] = signature
    ctx = chat._context(log, case["parent"])
    ids = ctx["ids"]
    entries = []
    for t, cid in zip(ctx["turns"], ids):
        e = {"call": cid}
        if t.get("steps"):
            e["steps"] = True
        entries.append(e)
    assert ctx["finish"](entries) == case["expect"]["saw"]
    assert ctx["rows"] == case["expect"]["rows"]


@pytest.mark.parametrize("path", [p for p in cases("tools") if p.stem.startswith("asks")], ids=lambda p: p.stem)
def test_tool_rules(path):
    case = load(path)
    rule = case["rule"]
    if rule == "function":
        rule = lambda a: True  # noqa: E731
    got = [tools.asks(rule, tools.Approval("c", 1, "t", a["name"], {}, a["effects"], a["path"]))
           for a in case["approvals"]]
    assert got == case["expect"]["asks"]


@pytest.mark.parametrize("path", [p for p in cases("tools") if p.stem.startswith("denial")], ids=lambda p: p.stem)
def test_tool_denial(path):
    case = load(path)
    assert tools.denial(case["reason"]) == case["expect"]["output"]


@pytest.mark.parametrize("path", cases("views"), ids=lambda p: p.stem)
def test_views(path):
    case = load(path)
    assert views.outside(case["events"], answer_from=case["answer_from"]) == case["expect"]["events"]


@pytest.mark.parametrize("path", cases("context"), ids=lambda p: p.stem)
def test_rows_that_keep_their_context(path):
    case = load(path)
    expect = case["expect"]
    if "refuses" in expect:
        with pytest.raises(SawError) as err:
            calllog.earlier_of(case["call"], case["records"])
        assert err.value.code == expect["refuses"]
    else:
        got = calllog.earlier_of(case["call"], case["records"])
        assert {k: got[k] for k in ("earlier", "conversation")} == expect
