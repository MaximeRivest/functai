"""Training on Tinker for real (costs a few cents). Not run by pytest.

    export TINKER_API_KEY=tml-...
    cd python && .venv/bin/python tests/tinker_live.py            # the plans and their prices, nothing spent
    cd python && .venv/bin/python tests/tinker_live.py --yes      # do it

A small function trained on Qwen3.5-4B (sft), called; one rl step from it; a
decision model; then every checkpoint these runs made is deleted. Each step
prints what it did and what it cost. The first live run of design 21.
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Literal

import functai
import functai.bake
from functai import ai

STUDENT = os.environ.get("TINKER_STUDENT", "Qwen/Qwen3.5-4B")


@ai
def capital(country: str) -> str:
    """The capital city of the country, in one word."""


@ai
def mood(text: str) -> Literal["happy", "sad"]:
    """The mood of the message."""


PAIRS = [("France", "Paris"), ("Japan", "Tokyo"), ("Peru", "Lima"), ("Kenya", "Nairobi"), ("Chile", "Santiago"),
         ("Canada", "Ottawa"), ("Egypt", "Cairo"), ("Spain", "Madrid")]
ROWS = [{"country": c, "result": r} for c, r in PAIRS] * 2
TEST = [{"country": "Peru", "result": "Lima"}, {"country": "Japan", "result": "Tokyo"}]
MOODS = [{"text": t, "result": m} for t, m in [("What a lovely day", "happy"), ("I lost my keys", "sad"),
                                             ("We won the match!", "happy"), ("It rained all week", "sad")]] * 3


def main(spend: bool) -> None:
    if not os.environ.get("TINKER_API_KEY"):
        sys.exit("set TINKER_API_KEY first")
    home = Path(tempfile.mkdtemp(prefix="functai-tinker-live-"))
    common = dict(where="tinker", student=STUDENT, log=True)
    exact = functai.Criterion("exact", "verdict", judge=lambda answer, expected: answer.strip() == expected)
    plan = functai.bake.bake(capital, ROWS, test=TEST, criteria=[exact], epochs=2, run_folder=home / "sft",
                             plan_only=True, **common)
    dplan = functai.bake.bake(mood, MOODS, method="decision", test=MOODS[:4], epochs=2, run_folder=home / "decision",
                              plan_only=True, **common)
    print(f"\nestimated: sft ≈ ${plan.dollars['total']:.3f}, decision ≈ ${dplan.dollars['total']:.3f} (rl adds a "
          f"few cents)")
    if not spend:
        print("nothing spent; run with --yes to do it")
        return
    s1 = plan.run()
    print("sft:", capital.using(lm=s1)("Peru"), "| best", s1.meta["training"]["best"], "| spent",
          s1.meta["training"]["spent"])
    rl = functai.bake.bake(capital, ROWS, method="rl", start=s1, reward=exact, test=TEST, epochs=1, batch=4, group=4,
                           temperatures=[1.0], run_folder=home / "rl", **{k: v for k, v in common.items() if k != "student"})
    print("rl:", capital.using(lm=rl)("Japan"), "| checks", rl.meta["training"]["checks"])
    d = dplan.run()
    print("decision:", mood.using(lm=d)("I lost my keys"), "| best", d.meta["training"]["best"])
    import tinker
    rest = tinker.RestClient()
    made = [m for m in (s1, rl, d) for m in [m.meta["weights"].get("uri"), m.meta["weights"].get("state_uri")] if m]
    for path in made:
        rest.delete_checkpoint_from_tinker_path(path).result()
    print(f"deleted {len(made)} checkpoints; run folders in {home}")


if __name__ == "__main__":
    main("--yes" in sys.argv)
