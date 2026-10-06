"""How good is the team classifier, and do the two rules help? (DSPy)

Run it:   python score.py
Results:  predictions.csv, results.json
"""
import json
import math
from typing import Literal

import dspy
import pandas as pd

dspy.configure(lm=dspy.LM("openai/gpt-4.1-mini", temperature=0))
tickets = pd.read_csv("tickets.csv")
devset = [dspy.Example(message=m, team=c).with_inputs("message") for m, c in zip(tickets["message"], tickets["category"])]

Team = Literal["shipping", "billing", "product", "account"]


class Route(dspy.Signature):
    """Which team should answer this customer message?"""
    message: str = dspy.InputField()
    team: Team = dspy.OutputField()


class RouteWithRules(dspy.Signature):
    """Which team should answer this customer message?
    An item that arrived broken or damaged is a shipping problem, even when the customer asks for a replacement.
    Any request for money back (a refund, a chargeback, "I want my money back") goes to billing."""
    message: str = dspy.InputField()
    team: Team = dspy.OutputField()


def right(example, prediction, trace=None):
    return float(prediction.team == example.team)


evaluate = dspy.Evaluate(devset=devset, metric=right, num_threads=8)
v1, v2 = evaluate(dspy.Predict(Route)), evaluate(dspy.Predict(RouteWithRules))
s1 = [score for _, _, score in v1.results]
s2 = [score for _, _, score in v2.results]


def wilson(k, n, z=1.959964):
    p = k / n
    centre, half = (p + z * z / (2 * n)) / (1 + z * z / n), z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def paired_t(d):   # 95% interval of the mean per-ticket difference
    from scipy.stats import t
    n, mean = len(d), sum(d) / len(d)
    sd = math.sqrt(sum((x - mean) ** 2 for x in d) / (n - 1))
    half = t.ppf(0.975, n - 1) * sd / math.sqrt(n)
    return mean, mean - half, mean + half


n = len(s1)
low, high = wilson(sum(s1), n)
diff, dlow, dhigh = paired_t([b - a for a, b in zip(s1, s2)])
pd.DataFrame({"id": tickets["id"], "category": tickets["category"],
              "v1": [p.team for _, p, _ in v1.results], "v2": [p.team for _, p, _ in v2.results]}).to_csv("predictions.csv", index=False)
results = {"v1_accuracy": sum(s1) / n, "v1_low": low, "v1_high": high, "v2_accuracy": sum(s2) / n,
           "diff": diff, "diff_low": dlow, "diff_high": dhigh,
           "verdict": "unclear" if dlow <= 0 <= dhigh else ("better" if diff > 0 else "worse")}
json.dump(results, open("results.json", "w"), indent=1)
print(results)
