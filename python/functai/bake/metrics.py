"""What a trained model's answers are worth, measured the way the records did:
accuracy with its uncertainty, top-3, calibration (ECE, NLL), and how accurate
the model is on the share of rows it is most sure about (for sending the rest
to a bigger model). Plain Python on lists of probabilities: no numpy needed."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """95% interval of a proportion k/n (Wilson; honest near 0 and 1 and for small n)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, center - half), min(1.0, center + half))


def argmax(p: Sequence[float]) -> int:
    return max(range(len(p)), key=p.__getitem__)


def accuracy(probs: Sequence[Sequence[float]], labels: Sequence[int]) -> float:
    return sum(argmax(p) == y for p, y in zip(probs, labels)) / len(labels) if labels else float("nan")


def top_k(probs: Sequence[Sequence[float]], labels: Sequence[int], k: int = 3) -> float:
    hits = 0
    for p, y in zip(probs, labels):
        top = sorted(range(len(p)), key=p.__getitem__, reverse=True)[:k]
        hits += y in top
    return hits / len(labels) if labels else float("nan")


def ece(probs: Sequence[Sequence[float]], labels: Sequence[int], bins: int = 15) -> float:
    """Expected calibration error: how far confidence is from accuracy, averaged over bins."""
    if not labels:
        return float("nan")
    totals = [[0, 0.0, 0.0] for _ in range(bins)]
    for p, y in zip(probs, labels):
        i = argmax(p)
        c = p[i]
        b = min(bins - 1, int(c * bins))
        totals[b][0] += 1
        totals[b][1] += c
        totals[b][2] += float(i == y)
    n = len(labels)
    return sum(t[0] / n * abs(t[1] / t[0] - t[2] / t[0]) for t in totals if t[0])


def nll(probs: Sequence[Sequence[float]], labels: Sequence[int]) -> float:
    if not labels:
        return float("nan")
    return -sum(math.log(max(p[y], 1e-12)) for p, y in zip(probs, labels)) / len(labels)


def confidence(p: Sequence[float]) -> float:
    return max(p)


def coverage(conf: Sequence[float], correct: Sequence[bool],
             shares: Sequence[float] = (0.5, 0.8, 0.9, 1.0)) -> List[Dict[str, float]]:
    """Accuracy on the most confident share of rows, and the confidence at that cut."""
    order = sorted(range(len(conf)), key=lambda i: conf[i], reverse=True)
    out = []
    for s in shares:
        n = max(1, round(s * len(order)))
        kept = order[:n]
        out.append({"share": s, "accuracy": sum(correct[i] for i in kept) / n, "threshold": conf[kept[-1]]})
    return out


def threshold_for(conf: Sequence[float], correct: Sequence[bool], target: float,
                  min_rows: int = 20) -> Optional[Dict[str, float]]:
    """The lowest confidence cut whose kept rows are at least ``target`` accurate
    (with at least ``min_rows`` kept); None when no cut reaches it."""
    order = sorted(range(len(conf)), key=lambda i: conf[i], reverse=True)
    best = None
    right = 0
    for n, i in enumerate(order, 1):
        right += correct[i]
        if n >= min_rows and right / n >= target:
            best = {"threshold": conf[i], "share": n / len(order), "accuracy": right / n}
    return best


def fit_temperature(logits: Sequence[Sequence[float]], targets: Sequence[Sequence[float]]) -> float:
    """The temperature T minimizing cross-entropy of softmax(logits / T) against the
    targets (hard or soft), found by golden-section search on log T."""
    if not logits:
        return 1.0

    def loss(log_t: float) -> float:
        t = math.exp(log_t)
        total = 0.0
        for z, q in zip(logits, targets):
            zs = [v / t for v in z]
            m = max(zs)
            lse = m + math.log(sum(math.exp(v - m) for v in zs))
            total -= sum(qi * (zi - lse) for zi, qi in zip(zs, q) if qi)
        return total / len(logits)

    lo, hi = math.log(0.05), math.log(20.0)
    g = (math.sqrt(5) - 1) / 2
    a, b = hi - g * (hi - lo), lo + g * (hi - lo)
    fa, fb = loss(a), loss(b)
    for _ in range(60):
        if fa < fb:
            hi, b, fb = b, a, fa
            a = hi - g * (hi - lo)
            fa = loss(a)
        else:
            lo, a, fa = a, b, fb
            b = lo + g * (hi - lo)
            fb = loss(b)
    return math.exp((lo + hi) / 2)


def softmax(z: Sequence[float], t: float = 1.0) -> List[float]:
    zs = [v / t for v in z]
    m = max(zs)
    e = [math.exp(v - m) for v in zs]
    s = sum(e)
    return [v / s for v in e]
