"""How good is the team classifier, and do the two rules help?

Run it:   python score.py
Results:  predictions.csv, results.json
"""
import json
from typing import Literal

import functai
from dpyr import read
from functai import ai

functai.configure(lm="gpt-4.1-mini", temperature=0, cache_replies="disk")
tickets = read("tickets.csv")

Team = Literal["shipping", "billing", "product", "account"]


@ai
def team(message: str) -> Team:
    """Which team should answer this customer message?"""
    ...


@ai
def team_v2(message: str) -> Team:
    """Which team should answer this customer message?
    An item that arrived broken or damaged is a shipping problem, even when the customer asks for a replacement.
    Any request for money back (a refund, a chargeback, "I want my money back") goes to billing."""
    ...


before = functai.evaluate(team, tickets, expected="category", num_threads=8)
after = functai.evaluate(team_v2, tickets, expected="category", num_threads=8)
change = functai.compare(before, after).to_pandas().iloc[0]
print(before, after, change, sep="\n")

t1, t2 = before.table.to_pandas(), after.table.to_pandas()
read({"id": t1["id"], "category": t1["category"], "v1": t1["pred_result"], "v2": t2["pred_result"]}).write_csv("predictions.csv")
summary = before.summary.to_pandas().iloc[0]
results = {"v1_accuracy": before.score, "v1_low": float(summary["low"]), "v1_high": float(summary["high"]),
           "v2_accuracy": after.score, "diff": float(change["diff"]), "diff_low": float(change["low"]),
           "diff_high": float(change["high"]),
           "verdict": "unclear" if change["low"] <= 0 <= change["high"] else ("better" if change["diff"] > 0 else "worse")}
json.dump(results, open("results.json", "w"), indent=1)
print(results)
