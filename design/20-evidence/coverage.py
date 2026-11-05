"""Exact coverage of three 95% ranges for B - A on right-or-wrong cases, by
enumerating every possible table of n paired cases (no simulation).

    python3 design/20-evidence/coverage.py

"mover-wilson" is the contract's (measures.md, `paired`); "newcombe-10" the
same without the continuity correction of phi; "paired-t" what compare used
before 2026-11-05. 120 settings drawn as real evaluations look: most cases
right for both, a few discordant.
"""
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "contract" / "cases"))
import measure_rules as m  # noqa: E402


def tables(n):
    for a in range(n + 1):
        for b in range(n + 1 - a):
            for c in range(n + 1 - a - b):
                yield a, b, c, n - a - b - c


def vectors(a, b, c, d):
    """before, after: a both right, b before only, c after only, d neither."""
    return [1] * a + [1] * b + [0] * c + [0] * d, [1] * a + [0] * b + [1] * c + [0] * d


def newcombe10(a, b):
    n = len(a)
    n11 = sum(x == 1 and y == 1 for x, y in zip(a, b)); n10 = sum(x == 1 and y == 0 for x, y in zip(a, b))
    n01 = sum(x == 0 and y == 1 for x, y in zip(a, b)); n00 = n - n11 - n10 - n01
    pa, pb = (n11 + n10) / n, (n11 + n01) / n
    la, ua = m.wilson(n11 + n10, n); lb, ub = m.wilson(n11 + n01, n)
    den = (n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)
    phi = 0 if den == 0 else (n11 * n00 - n10 * n01) / math.sqrt(den)
    d = pb - pa
    return (d - math.sqrt(max(0, (pb - lb) ** 2 - 2 * phi * (pb - lb) * (ua - pa) + (ua - pa) ** 2)),
            d + math.sqrt(max(0, (ub - pb) ** 2 - 2 * phi * (ub - pb) * (pa - la) + (pa - la) ** 2)))


def paired_t(a, b):
    d = [y - x for x, y in zip(a, b)]
    n = len(d); mean = sum(d) / n
    half = m.t975(n - 1) * math.sqrt(sum((v - mean) ** 2 for v in d) / (n - 1)) / math.sqrt(n)
    return mean - half, mean + half


METHODS = {"mover-wilson": lambda a, b: (lambda r: (r["low"], r["high"]))(m.paired(a, b)),
           "newcombe-10": newcombe10, "paired-t": paired_t}


def coverage(n, p, method, memo):
    theta = p[2] - p[1]
    total = 0.0
    for t in tables(n):
        if t not in memo:
            memo[t] = method(*vectors(*t))
        lo, hi = memo[t]
        if lo - 1e-12 <= theta <= hi + 1e-12:
            lp = math.lgamma(n + 1)
            ok = True
            for k, pk in zip(t, p):
                lp -= math.lgamma(k + 1)
                if k:
                    if pk == 0:
                        ok = False
                        break
                    lp += k * math.log(pk)
            if ok:
                total += math.exp(lp)
    return total


def settings(count=120, seed=1):
    rng = random.Random(seed)
    out = []
    while len(out) < count:
        pa, disc, share = rng.uniform(0.3, 0.99), rng.uniform(0.0, 0.3), rng.uniform(0.1, 0.9)
        p10, p01 = disc * share, disc * (1 - share)
        p11 = max(0.0, pa - p10)
        p00 = 1 - p11 - p10 - p01
        if p00 >= 0:
            out.append((p11, p10, p01, p00))
    return out


if __name__ == "__main__":
    grid = settings()
    print("n   method         mean    5th pct  lowest")
    for n in (10, 20, 30, 50):
        for name, fn in METHODS.items():
            memo = {}
            cs = sorted(coverage(n, p, fn, memo) for p in grid)
            print(f"{n:<3} {name:<14} {sum(cs) / len(cs):.3f}   {cs[len(cs) // 20]:.3f}    {cs[0]:.3f}")
