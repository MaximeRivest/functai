"""Simulated coverage of the paired t range (measures.md, `paired`, values that
are not all 0 or 1) for case values that are means of repeats.

    python3 design/20-evidence/paired_frac.py
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "contract" / "cases"))
import measure_rules as m  # noqa: E402

if __name__ == "__main__":
    rng = random.Random(3)
    print("cases repeats  cover  width")
    for n in (5, 10, 20, 40):
        for k in (2, 3, 5):
            hit = width = 0.0; trials = 3000
            for _ in range(trials):
                base = [rng.betavariate(0.6, 0.4) for _ in range(n)]
                shift = rng.uniform(-0.15, 0.15)
                pb = [min(1, max(0, b + shift * rng.uniform(0, 2))) for b in base]
                truth = sum(pb) / n - sum(base) / n
                a = [sum(rng.random() < x for _ in range(k)) / k for x in base]
                b = [sum(rng.random() < x for _ in range(k)) / k for x in pb]
                if all(v in (0, 1) for v in a + b):
                    continue
                got = m.paired(a, b, None, True)
                hit += got["low"] <= truth <= got["high"]; width += got["high"] - got["low"]
            print(f"{n:<5} {k:<8} {hit / trials:.3f}  {width / trials:.3f}")
