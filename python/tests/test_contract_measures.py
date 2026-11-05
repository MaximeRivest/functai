"""The contract's measures (contract/measures.md): every case in
contract/cases/measures/, through functai.measures (not through the rules the
cases were written from)."""

import json
import math
from pathlib import Path

import pytest

from functai import measures as m
from functai.calllog import rated_rows

CASES = Path(__file__).resolve().parents[2] / "contract" / "cases" / "measures"


def files():
    return sorted(CASES.glob("*.json"))


def close(got, want, where="") -> None:
    """JSON-equal, numbers within 1e-9."""
    if isinstance(want, bool) or want is None or isinstance(want, str):
        assert got == want, f"{where}: {got!r} != {want!r}"
    elif isinstance(want, (int, float)):
        assert isinstance(got, (int, float)) and not isinstance(got, bool), f"{where}: {got!r} is not a number"
        assert math.isclose(got, want, rel_tol=0, abs_tol=1e-9), f"{where}: {got!r} != {want!r}"
    elif isinstance(want, list):
        assert isinstance(got, (list, tuple)) and len(got) == len(want), f"{where}: {got!r} != {want!r}"
        for i, (g, w) in enumerate(zip(got, want)):
            close(g, w, f"{where}[{i}]")
    elif isinstance(want, dict):
        assert isinstance(got, dict), f"{where}: {got!r} is not an object"
        assert set(got) == set(want), f"{where}: keys {sorted(got)} != {sorted(want)}"
        for k in want:
            close(got[k], want[k], f"{where}.{k}")
    else:
        raise TypeError(type(want))


def cases_list(cases):
    return [{"group": g, "case": k, "value": v} for g, cs in cases.items() for k, v in cs.items()]


def run(case):
    kind = case["kind"]
    if kind == "key-row":
        return {"key": m.row_key(case["row"], case["inputs"])}
    if kind == "key-call":
        return {"key": m.call_key(case["record"])}
    if kind == "split":
        try:
            return {"splits": [m.split_of(r, case["inputs"], share=case["share"], salt=case["salt"]) for r in case["rows"]]}
        except m.MeasureError as err:
            return {"refuses": err.code}
    if kind == "draw":
        return {"ids": m.draw(case["ids"], case["n"], case["seed"])}
    if kind == "ratings":
        return {"counting": [{"id": j["id"], "calls": j["calls"], "item": j["item"], "value": j["value"],
                              "source": list(j["source"])} for j in m.counting(case["records"], case["criterion"])]}
    if kind == "values":
        v = m.values_of(case["records"], case["program"], case["criterion"], case["source"], by=case["by"],
                        drawn=case["drawn"])
        return {"results": v["results"], "cases": cases_list(v["cases"])}
    if kind == "preferences":
        got = m.preferences_of(case["records"], case["program"], case["criterion"], case["source"], case["a"], case["b"],
                               by=case["by"], drawn=case["drawn"])
        return {"cases": [{"case": k, "value": v} for k, v in got.items()]}
    if kind == "paired":
        return m.paired(case["a"], case["b"], case["margin"], case["bounded"])
    if kind == "winrate":
        return m.win_rate(case["values"], case["margin"])
    if kind == "fragility":
        return {"fragility": m.fragility(case["values"])}
    if kind == "ppi":
        return m.ppi(case["judge"], case["people"], case["bounded"])
    if kind == "ppi-paired":
        return m.ppi_paired(case["judge_a"], case["judge_b"], case["people_a"], case["people_b"], case["margin"],
                            case["bounded"])
    if kind == "agreement":
        return m.agreement_of(case["a"], case["b"], case["choice"])
    if kind == "self":
        return m.self_agreement_of(case["per_case"])
    if kind == "rank":
        return m.bradley_terry(case["versions"], [(o["case"], o["first"], o["second"], o["first_share"])
                                                  for o in case["outcomes"]])
    if kind == "positions":
        ratings = [r for r in case["records"] if "functai_rating" in r]
        return m.positions_of(ratings, case["criterion"], case["judge"])
    if kind == "scoreboard":
        return m.scoreboard_of(case["records"], case["program"], case["criteria"], by=case["by"])
    if kind == "compare":
        return m.compare_of(case["records"], case["program"], case["criterion"], case["source"], case["a"], case["b"],
                            by=case["by"], drawn=case["drawn"], margin=case["margin"])
    if kind == "disagreements":
        return {"cases": m.disagreements_of(case["records"], case["program"], case["criterion"], case["source"],
                                            by=case["by"])}
    if kind == "rated":
        calls = [r for r in case["records"] if "functai_call" in r]
        ratings = [r for r in case["records"] if "functai_rating" in r]
        rows, left = rated_rows(calls, ratings, **{k: v for k, v in case["rated"].items()})
        return {"rows": rows, "left_out": left}
    raise AssertionError(f"a kind this harness does not know: {kind}")


@pytest.mark.parametrize("path", files(), ids=lambda p: p.stem)
def test_measures_case(path):
    case = json.loads(path.read_text())
    close(run(case), case["expect"], path.stem)


def test_every_kind_has_cases():
    kinds = {json.loads(p.read_text())["kind"] for p in files()}
    assert kinds >= {"key-row", "key-call", "split", "draw", "ratings", "values", "preferences", "paired", "winrate",
                     "fragility", "ppi", "ppi-paired", "agreement", "self", "rank", "positions", "scoreboard",
                     "compare", "disagreements", "rated"}
