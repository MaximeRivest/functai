"""Token counts with lm15 2's meaning (contract/calls.md, ``usage``; contract/replies.md,
*The file*): a sum is exact or absent; counts lm15 1 wrote are kept only where they
cannot mean something else."""

import json
import sqlite3

import lm15

import functai
from functai import ai, calllog, replies
from functai.data import sum_usage, usage_read

XML = "<result>\n{}\n</result>"


@ai
def capital(country: str) -> str:
    """The country's capital city."""


def test_a_sum_keeps_only_the_counts_every_call_reported():
    assert sum_usage([]) == {}
    assert sum_usage([{"input_tokens": 3, "output_tokens": 2}]) == {"input_tokens": 3, "output_tokens": 2}
    both = sum_usage([{"input_tokens": 3, "output_tokens": 2, "reasoning_tokens": 1},
                      {"input_tokens": 4, "output_tokens": 5}])
    assert both == {"input_tokens": 7, "output_tokens": 7}        # reasoning: one call did not say
    assert sum_usage([{"input_tokens": 3}, {}]) == {}             # a call that reported nothing


def test_counts_lm15_1_wrote_are_kept_only_when_they_mean_the_same():
    plain = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15, "reasoning_tokens": 2,
             "cache_read_tokens": 0}
    assert usage_read(plain, "1.5.0") == plain
    assert usage_read(plain, None) == plain                        # a writer that did not say: lm15 1
    # Gemini 1.x: thinking outside the output (total = input + output + thinking)
    assert usage_read({"input_tokens": 10, "output_tokens": 5, "reasoning_tokens": 7, "total_tokens": 22},
                      "1.2.0") == {}
    # Anthropic 1.x with a cache: the input left the cached tokens out
    assert usage_read({"input_tokens": 20, "output_tokens": 5, "total_tokens": 25, "cache_read_tokens": 4873},
                      "1.1.0") == {}
    assert usage_read({"input_tokens": 10, "output_tokens": 5}, "1.1.0") == {}     # no total: cannot tell
    cached = {"input_tokens": 4893, "output_tokens": 5, "total_tokens": 4898, "cache_read_tokens": 4873}
    assert usage_read(cached, "2.0.0") == cached                  # lm15 2 wrote it: as written
    assert usage_read({"input_tokens": 1, "x": True, "y": "2"}, "2.0.0") == {"input_tokens": 1}


def test_a_new_reply_is_kept_as_format_2(fake, tmp_path):
    fake(responder=lambda req: XML.format("Oslo"))
    functai.configure(cache_replies=tmp_path)
    assert capital("Norway") == "Oslo"
    db = sqlite3.connect(tmp_path / "replies.sqlite")
    assert db.execute("SELECT format FROM replies").fetchall() == [(2,)]


def _rewrite_as_lm15_1(path, usage):
    db = sqlite3.connect(path)
    key, text = db.execute("SELECT key, response FROM replies").fetchone()
    data = json.loads(text)
    data["usage"] = usage
    db.execute("UPDATE replies SET format = 1, response = ? WHERE key = ?", (json.dumps(data), key))
    db.commit()


def test_a_reply_lm15_1_kept_is_reused_with_counts_that_cannot_mislead(fake, tmp_path):
    r = fake(responder=lambda req: XML.format("Oslo"))
    functai.configure(cache_replies=tmp_path)
    capital("Norway")
    path = tmp_path / "replies.sqlite"

    _rewrite_as_lm15_1(path, {"input_tokens": 20, "output_tokens": 5, "total_tokens": 25, "cache_read_tokens": 4873})
    replies._disks.clear()
    functai.clear_cache()
    pred = capital.predict("Norway")
    assert pred.result == "Oslo" and len(r.requests) == 1          # the reply is reused: no model call
    assert pred.usage == {}                                        # its counts may mean something else

    _rewrite_as_lm15_1(path, {"input_tokens": 20, "output_tokens": 5, "total_tokens": 25, "cache_read_tokens": 0})
    replies._disks.clear()
    functai.clear_cache()
    pred = capital.predict("Norway")
    assert len(r.requests) == 1 and pred.usage == {"input_tokens": 20, "output_tokens": 5, "total_tokens": 25,
                                                    "cache_read_tokens": 0}


def test_a_call_whose_model_calls_did_not_all_report_a_count_has_none(fake, tmp_path):
    usages = [lm15.Usage(input_tokens=3, output_tokens=2, total_tokens=5, reasoning_tokens=1),
              lm15.Usage(input_tokens=4, output_tokens=2, total_tokens=6)]
    replies_ = ["not the layout", XML.format("Oslo")]

    def respond(req):
        return lm15.Response(id="r", model=req.model, message=lm15.Message.assistant([lm15.TextPart(replies_.pop(0))]),
                             finish_reason="stop", usage=usages.pop(0))

    fake(responder=respond)
    functai.configure(log_calls=tmp_path)
    pred = capital.predict("Norway")                               # read again after an unreadable reply
    assert pred.usage == {"input_tokens": 7, "output_tokens": 4, "total_tokens": 11}
    (rec,) = [r for r in calllog.read(tmp_path)[0]]
    assert rec["usage"] == pred.usage


def test_the_calls_table_shows_counts_lm15_1_wrote_only_when_they_mean_the_same(fake, tmp_path):
    fake(responder=lambda req: XML.format("Oslo"))
    functai.configure(log_calls=tmp_path)
    capital("Norway")
    capital("Peru")
    (log,) = list(tmp_path.rglob("*.jsonl"))
    lines = [json.loads(line) for line in log.read_text().splitlines()]
    lines[0]["process"]["lm15"] = "1.5.0"
    lines[0]["usage"] = {"input_tokens": 20, "output_tokens": 5, "total_tokens": 25, "cache_read_tokens": 4873}
    lines[1]["process"]["lm15"] = "1.5.0"
    lines[1]["usage"] = {"input_tokens": 20, "output_tokens": 5, "total_tokens": 25}
    log.write_text("".join(json.dumps(r) + "\n" for r in lines))
    rows = functai.calls(capital, folder=tmp_path).to_dicts()
    assert [r["input_tokens"] for r in rows] == [None, 20]
