"""Simulated coverage of the contract's prediction-powered range (measures.md,
`ppi`) against PPI++ without the pseudo-observations, and people alone.

    python3 design/20-evidence/ppi.py

Each trial: N cases, a true rate p, people's 0-or-1 verdicts y, a judge that
agrees with them with probability `acc`; people judge a random n of the cases.
"""
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "contract" / "cases"))
import measure_rules as m  # noqa: E402


def plain_ppi(judge, people):
    """PPI++ as published: no pseudo-observations, normal quantile."""
    L = sorted(k for k in people if k in judge); U = sorted(k for k in judge if k not in people)
    n, N = len(L), len(U)
    y = [people[k] for k in L]; f = [judge[k] for k in L]; fu = [judge[k] for k in U]
    vf = m.var(f + fu)
    my, mf = m.mean(y), m.mean(f)
    c = sum((a - my) * (b - mf) for a, b in zip(y, f)) / (n - 1)
    lam = min(1.0, max(0.0, c / ((1 + n / N) * vf))) if vf else 0.0
    r = [a - lam * b for a, b in zip(y, f)]
    est = lam * m.mean(fu) + m.mean(r)
    se = math.sqrt(lam * lam * vf / N + m.var(r) / n)
    return est - m.Z * se, est + m.Z * se


def people_alone(people):
    vs = list(people.values())
    lo, hi = m.wilson(sum(vs), len(vs))
    return lo, hi


if __name__ == "__main__":
    rng = random.Random(5)
    print("N    n   judge  contract (cover, width)   plain PPI++       people alone")
    for N, n in ((20, 6), (40, 8), (40, 15), (200, 15), (200, 30), (1000, 60)):
        for acc in (0.99, 0.9, 0.75, 0.6):
            hit = {"c": 0, "p": 0, "h": 0}; width = {"c": 0.0, "p": 0.0, "h": 0.0}; trials = 3000
            for _ in range(trials):
                p = rng.uniform(0.05, 0.95)
                ys = [1.0 if rng.random() < p else 0.0 for _ in range(N)]
                fs = [y if rng.random() < acc else 1 - y for y in ys]
                keys = [f"k{i:04d}" for i in range(N)]
                drawn = set(rng.sample(range(N), n))
                judge = dict(zip(keys, fs))
                people = {keys[i]: ys[i] for i in drawn}
                got = m.ppi(judge, people)
                for k, (lo, hi) in (("c", (got["low"], got["high"])), ("p", plain_ppi(judge, people)),
                                    ("h", people_alone(people))):
                    hit[k] += lo <= p <= hi; width[k] += hi - lo
            print(f"{N:<4} {n:<3} {acc:<5}  " + "   ".join(f"{hit[k] / trials:.3f} {width[k] / trials:.3f}"
                                                            for k in ("c", "p", "h")))
