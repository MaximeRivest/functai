"""contract/cases/training/: a student's tokens from its files, pictures, reading
a reply, training examples, advantages, the energy score, losses as weights,
choosing. The same cases TypeScript passes. Needs tokenizers and transformers
(functai[tinker]); no PyTorch."""

import base64
import json
import math

import pytest

pytest.importorskip("tokenizers")
pytest.importorskip("transformers")

from contract_support import CONTRACT  # noqa: E402
from functai.bake import training as T  # noqa: E402
from functai.bake.examples import BakeError  # noqa: E402
from functai.bake.tokens import (PictureRule, Student, grid_tokens, picture, picture_size, qwen_tokens,  # noqa: E402
                                   students_table)

DIR = CONTRACT / "cases" / "training"
FILES = sorted(p.name for p in DIR.glob("*.json") if p.name != "toy-tokenizer.json")


def read(name):
    return json.loads((DIR / name).read_text())


def close(actual, expected, where=""):
    if isinstance(expected, float) or (isinstance(expected, int) and not isinstance(expected, bool)
                                       and isinstance(actual, float)):
        assert math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9), f"{where}: {actual} != {expected}"
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), f"{where}: {actual} != {expected}"
        for i, (a, e) in enumerate(zip(actual, expected)):
            close(a, e, f"{where}[{i}]")
    elif isinstance(expected, dict):
        assert isinstance(actual, dict) and set(actual) == set(expected), f"{where}: {actual} != {expected}"
        for k in expected:
            close(actual[k], expected[k], f"{where}.{k}")
    else:
        assert actual == expected, f"{where}: {actual!r} != {expected!r}"


def student(template, rule=None, effort=0.0):
    r = PictureRule(**rule) if rule else None
    return Student.from_files("toy", (DIR / "toy-tokenizer.json").read_text(), (DIR / template).read_text(),
                              picture_rule=r, effort=effort)


def test_the_students_table_is_the_contracts():
    assert students_table() == json.loads((CONTRACT / "students.json").read_text())


def test_the_training_cases_are_all_read():
    assert len(FILES) >= 19


@pytest.mark.parametrize("name", [f for f in FILES if f.startswith("tokens-")])
def test_tokens(name):
    c = read(name)
    st = student(c["template"], c["picture"], c["effort"])
    pics = [picture(base64.b64decode(b)) for b in c["pictures"]]
    e = c["expect"]
    if "refuses" in e:
        with pytest.raises(BakeError):
            st.render(c["messages"], pics, c["reply"])
        return
    r = st.render(c["messages"], pics, c["reply"])
    assert st.kwargs == e["kwargs"]
    assert (st.lead, st.end, st.stop_token, st.end_head) == (e["lead"], e["end"], e["stop_token"], e["end_head"])
    assert r.prompt == e["prompt"] and r.full == e["full"] and r.reply_start == e["reply_start"]
    assert r.prefix_fallback == bool(e.get("prefix_fallback"))


def test_pictures():
    for p in read("pictures-01-sizes.json")["pictures"]:
        assert picture_size(base64.b64decode(p["data"])) == p["expect"]
    c = read("pictures-02-grid.json")
    for s in c["sizes"]:
        assert grid_tokens(s["width"], s["height"], c["patch"]) == s["expect"]
    c = read("pictures-03-qwen.json")
    for s in c["sizes"]:
        assert qwen_tokens(s["width"], s["height"], c["patch"], c["merge"], c["min_pixels"], c["max_pixels"]) == s["expect"], s


def test_reading_replies():
    c = read("reply-01-reading.json")
    for g in c["generated"]:
        assert student(g["template"], None, c["effort"]).read_reply(g["generated"]) == g["expect"]


def test_datums():
    for f in ("datums-01-sft.json", "datums-02-sft-picture.json"):
        c = read(f)
        close(T.sft_datum(c["chunks"], c["reply_start"], c["weight"]), c["expect"], f)
    c = read("datums-03-rl.json")
    close(T.rl_datum(c["prompt"], c["sampled"], c["logprobs"], c["advantage"]), c["expect"])


def test_groups_energy_weights_and_choosing():
    for g in read("advantages-01-groups.json")["groups"]:
        close(T.advantages(g["rewards"], g["normalize"]), g["expect"])
    c = read("advantages-02-kl.json")
    close(T.kl_adjusted(c["advantages"], c["deltas"], c["beta"]), c["expect"])
    for g in read("energy-01.json")["groups"]:
        close(T.energy(g["readable"], g["d_ref"], g["d_pair"]), g["expect"])
    c = read("weights-01-dpo.json")
    close(T.dpo(c["pairs"], c["beta"]), c["expect"])
    for d in read("weights-02-decisions.json")["decisions"]:
        close(T.decision_weights(d["logprobs"], d["answer"], d["target"]), d["expect"])
    c = read("choosing-01.json")
    close(T.choose(c["checks"]), c["expect"])
