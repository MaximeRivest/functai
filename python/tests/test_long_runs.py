"""Stage 1.2: replies kept on disk, one flight per request, map's progress line, the evidence check,
pruning the log without losing ratings. The contract is contract/replies.md."""

import threading
import time

import pytest

import functai
from functai import ai, calllog, replies

XML = "<result>\n{}\n</result>"


@ai
def capital(country: str) -> str:
    """The country's capital city."""


def test_replies_kept_on_disk_survive_a_restart(fake, tmp_path):
    r = fake(responder=lambda req: XML.format("Oslo"))
    functai.configure(cache_replies=tmp_path / "cache")
    assert capital("Norway") == "Oslo" and len(r.requests) == 1
    replies._disks.clear()                                            # a new process opens the same file
    functai.clear_cache()
    assert capital("Norway") == "Oslo" and len(r.requests) == 1       # from disk: no model call
    store = replies.store_for(tmp_path / "cache")
    assert len(store) == 1 and (tmp_path / "cache" / "replies.sqlite").stat().st_mode & 0o077 == 0
    functai.clear_cache(tmp_path / "cache")
    assert len(store) == 0


def test_a_replicate_is_another_answer(fake, tmp_path):
    r = fake(responder=lambda req: XML.format(f"answer {len(r.requests)}"))
    functai.configure(cache_replies=tmp_path)
    answers = [capital.using(replicate=i)("Chile") for i in range(3)]
    assert answers == ["answer 1", "answer 2", "answer 3"]
    assert [capital.using(replicate=i)("Chile") for i in range(3)] == answers and len(r.requests) == 3
    with pytest.raises(ValueError):
        capital.using(replicate=-1)


def test_only_a_reply_that_was_read_is_kept(fake, tmp_path):
    replies_ = ["this is not the layout", XML.format("Santiago")]
    r = fake(responder=lambda req: replies_.pop(0))
    functai.configure(cache_replies=tmp_path, retries=0)
    with pytest.raises(Exception):
        capital("Chile")
    assert capital("Chile") == "Santiago" and len(r.requests) == 2    # the unreadable one was not kept
    assert len(replies.store_for(tmp_path)) == 1


def test_one_flight_per_request(fake, tmp_path):
    gate = threading.Event()
    r = fake(responder=lambda req: (gate.wait(5), XML.format("Accra"))[1])
    functai.configure(cache_replies=tmp_path)
    out = []
    threads = [threading.Thread(target=lambda: out.append(capital("Ghana"))) for _ in range(4)]
    for t in threads:
        t.start()
    time.sleep(0.3)
    gate.set()
    for t in threads:
        t.join(5)
    assert out == ["Accra"] * 4 and len(r.requests) == 1


def test_a_claim_whose_lease_ran_out_is_taken_over(fake, tmp_path):
    fake(responder=lambda req: XML.format("Lima"))
    store = replies.DiskReplies(tmp_path, lease=0.2)
    key = "sha256:" + "0" * 64
    db = store._db()
    db.execute("INSERT INTO claims (key, owner, until) VALUES (?, ?, ?)", (key, "another:1:x", time.time() + 0.3))
    t0 = time.monotonic()
    assert store.claim(key) is None                                   # waited for the other process's lease
    assert time.monotonic() - t0 >= 0.2
    store.unclaim(key)


def test_a_call_whose_log_content_drops_a_field_is_not_written_to_disk(fake, tmp_path):
    r = fake(responder=lambda req: XML.format("Rome"))
    functai.configure(cache_replies=tmp_path, log_content={"country": False})
    capital("Italy")
    capital("Italy")
    assert len(replies.store_for(tmp_path)) == 0 and len(r.requests) == 1   # the memory cache still serves it


def test_the_key_is_the_contract_s(fake):
    import lm15
    request = lm15.Request(model="m", messages=(lm15.Message.user("hi"),))
    from lm15.serde import request_to_dict
    expected = "sha256:" + __import__("hashlib").sha256(calllog.canonical(
        {"functai_reply": 1, "request": request_to_dict(request), "replicate": 0}).encode()).hexdigest()
    assert replies.key(request) == expected and replies.key(request, 1) != expected


def test_the_setting_is_checked():
    for bad in ("", 3, object()):
        with pytest.raises((TypeError, ValueError)):
            functai.configure(cache_replies=bad)


# ------------------------------------------------------------------ map


def test_map_shows_its_progress_and_resumes_from_kept_replies(fake, tmp_path, capsys):
    pytest.importorskip("dpyr")
    fail = {"Ghana"}

    def respond(req):
        text = str(req.messages[-1])
        if any(c in text for c in fail):
            raise RuntimeError("the provider fell over")
        return XML.format("a capital")

    r = fake(responder=respond)
    functai.configure(cache_replies=tmp_path, api_retries=0)
    rows = [{"country": c} for c in ("Norway", "Ghana", "Chile")]
    table = capital.map(rows, threads=2, progress=True)
    line = capsys.readouterr().err
    assert "capital: 3/3 rows · 1 error" in line and "tokens" in line
    assert table.collect().to_dicts()[1]["error"].startswith("RuntimeError")
    fail.clear()
    asked = len(r.requests)
    again = capital.map(rows, num_threads=2, progress=False)            # run again: only the failed row is sent
    assert len(r.requests) - asked == 1 and all(x["error"] is None for x in again.collect().to_dicts())
    assert capsys.readouterr().err == ""


# ------------------------------------------------------------------ judges


def test_quotes_found():
    source = "The parcel left Leeds on Monday.  It was delayed — by snow — and \u201cnot lost\u201d."
    assert functai.quotes_found(source, "It was delayed - by snow") is True
    assert functai.quotes_found(source, ["\u201cthe parcel left leeds on monday.\u201d", "It was lost", "",
                                         '"not lost"']) == [True, False, False, True]


# ------------------------------------------------------------------ ratings outlive a cleanup


def test_pruning_keeps_what_ratings_need(fake, tmp_path):
    fake(responder=lambda req: XML.format("ok"))
    functai.configure(log_calls=tmp_path)

    @ai
    def chat(message: str) -> str:
        """Answer."""

    c = chat.conversation()
    c("first")
    p = c.predict("second")
    chat("unrated")
    functai.rate(p.call_id, "right")
    day = next(d for d in tmp_path.iterdir() if d.is_dir())
    old = tmp_path / "2020-01-01"
    day.rename(old)
    before = functai.rated(chat).collect().to_dicts()
    got = functai.prune_calls("30d")
    assert got["days"] == 1 and got["kept"] == 2 and got["calls"] == 1   # the rated call and the turn it saw
    assert not old.exists() and list(tmp_path.glob("kept-*.jsonl"))
    assert functai.rated(chat).collect().to_dicts() == before
