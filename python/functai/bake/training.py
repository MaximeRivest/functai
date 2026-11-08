"""The arithmetic of training on a service (contract/training.md §6 to §10):
training examples ("datums") for each method, advantages, the energy score,
losses written as per-token weights, and choosing the checkpoint and
temperature to keep. Pure functions; every language computes the same numbers
(contract/cases/training/).

Chunks are ``{"text": [ids]}`` or ``{"picture": k, "tokens": n}``; a datum is
``{"model_input": chunks, "loss_fn_inputs": {name: list}}``.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

from .tokens import chunks_length

Chunk = Dict[str, Any]
Datum = Dict[str, Any]


def positions(chunks: Sequence[Chunk]) -> List[Optional[int]]:
    """Positions of chunks: token ids, None where a picture is."""
    out: List[Optional[int]] = []
    for c in chunks:
        out += list(c["text"]) if "text" in c else [None] * c["tokens"]
    return out


def merged(chunks: Sequence[Chunk]) -> List[Chunk]:
    """Adjacent text chunks joined, empty ones dropped."""
    out: List[Chunk] = []
    for c in chunks:
        if "text" in c:
            if not c["text"]:
                continue
            if out and "text" in out[-1]:
                out[-1] = {"text": out[-1]["text"] + list(c["text"])}
            else:
                out.append({"text": list(c["text"])})
        else:
            out.append(dict(c))
    return out


def shift(chunks: Sequence[Chunk]):
    """(input, targets): the chunks without their last token (trailing pictures dropped first); the tokens from the second on, 0 at pictures."""
    cs = merged(chunks)
    while cs and "text" not in cs[-1]:
        cs.pop()
    if not cs or chunks_length(cs) < 2:
        raise ValueError("a training example needs at least two tokens, the last of them text")
    last = cs[-1]
    inp = cs[:-1] + ([{"text": last["text"][:-1]}] if len(last["text"]) > 1 else [])
    targets = [0 if t is None else t for t in positions(cs)[1:]]
    return inp, targets


def sft_datum(chunks: Sequence[Chunk], reply_start: int, weight: float = 1.0) -> Datum:
    """A supervised example: the weight on every text position from the reply on."""
    cs = merged(chunks)
    inp, targets = shift(cs)
    pos = positions(cs)
    w = [weight if (i + 1) >= reply_start and pos[i + 1] is not None else 0.0 for i in range(len(targets))]
    return {"model_input": inp, "loss_fn_inputs": {"target_tokens": targets, "weights": w}}


def rl_datum(prompt: Sequence[Chunk], sampled: Sequence[int], logprobs: Sequence[float],
             advantage: Union[float, Sequence[float]]) -> Datum:
    """A sampled answer's example: the sampler's log-probabilities and the advantage (one, or one per token) on the sampled positions only."""
    cs = merged(list(prompt) + [{"text": list(sampled)}])
    inp, targets = shift(cs)
    start = chunks_length(prompt)
    per = isinstance(advantage, (list, tuple))
    lp = [logprobs[i + 1 - start] if i + 1 >= start else 0.0 for i in range(len(targets))]
    adv = [(advantage[i + 1 - start] if per else advantage) if i + 1 >= start else 0.0 for i in range(len(targets))]
    return {"model_input": inp, "loss_fn_inputs": {"target_tokens": targets, "logprobs": lp, "advantages": adv}}


def weighted_datum(chunks: Sequence[Chunk], reply_start: int, weight: Callable[[int], float]) -> Datum:
    """A weight per position of the reply (``weight(k)`` for the k-th from its start): a loss written as weights."""
    cs = merged(chunks)
    inp, targets = shift(cs)
    pos = positions(cs)
    w = [weight(i + 1 - reply_start) if (i + 1) >= reply_start and pos[i + 1] is not None else 0.0
         for i in range(len(targets))]
    return {"model_input": inp, "loss_fn_inputs": {"target_tokens": targets, "weights": w}}


def decision_datum(chunks: Sequence[Chunk], choice_tokens: Sequence[int], weights: Sequence[float]) -> Datum:
    """A decision's example: the last row holds the K choices' tokens and their weights, every other row 0."""
    cs = merged(chunks)
    inp, _ = shift(cs)
    n = chunks_length(cs) - 1
    k = len(choice_tokens)
    targets = [list(choice_tokens) if i == n - 1 else [0] * k for i in range(n)]
    w = [list(weights) if i == n - 1 else [0.0] * k for i in range(n)]
    return {"model_input": inp, "loss_fn_inputs": {"target_tokens": targets, "weights": w}}


# ------------------------------------------------------------------ groups


def advantages(rewards: Sequence[float], normalize: bool = False) -> Optional[List[float]]:
    """Each reward minus its group's mean (over its population standard deviation + 1e-6 with ``normalize``); None for a group that teaches nothing."""
    if max(rewards) - min(rewards) < 1e-9:
        return None
    mu = sum(rewards) / len(rewards)
    a = [r - mu for r in rewards]
    if normalize:
        sd = math.sqrt(sum((r - mu) ** 2 for r in rewards) / len(rewards))
        a = [x / (sd + 1e-6) for x in a]
    return a


def kl_adjusted(adv: Sequence[Sequence[float]], deltas: Sequence[Sequence[float]], beta: float) -> List[List[float]]:
    """Each token's advantage plus β·(mean δ − δ), the mean over every sampled token of the step."""
    flat = [d for row in deltas for d in row]
    mean = sum(flat) / len(flat)
    return [[a + beta * (mean - d) for a, d in zip(arow, drow)] for arow, drow in zip(adv, deltas)]


def energy(readable: Sequence[bool], d_ref: Sequence[Sequence[float]], d_pair: Sequence[Sequence[float]]) -> Dict[str, Any]:
    """The energy score of a group of answers against references (§8)."""
    idx = [i for i, ok in enumerate(readable) if ok]
    rewards = [-1.0] * len(readable)
    if not idx:
        return {"rewards": rewards, "score": 1.0}
    a = {i: sum(d_ref[i]) / len(d_ref[i]) for i in idx}
    if len(idx) == 1:
        rewards[idx[0]] = -a[idx[0]]
        return {"rewards": rewards, "score": a[idx[0]]}
    for i in idx:
        rewards[i] = -a[i] + sum(d_pair[i][j] for j in idx if j != i) / (len(idx) - 1)
    pairs = [d_pair[i][j] for i in idx for j in idx if i != j]
    return {"rewards": rewards, "score": sum(a.values()) / len(idx) - 0.5 * sum(pairs) / len(pairs)}


# ------------------------------------------------------------------ losses as weights


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))


def dpo(pairs: Sequence[Mapping[str, float]], beta: float = 0.1) -> List[Dict[str, float]]:
    """DPO as weights (§9): per pair, z, the loss, and the weight of the chosen and rejected replies' trained tokens."""
    out = []
    for p in pairs:
        z = beta * ((p["pi_c"] - p["rho_c"]) - (p["pi_r"] - p["rho_r"]))
        w = beta * _sigmoid(-z) / len(pairs)
        out.append({"z": z, "loss": math.log1p(math.exp(-z)) if z > -30 else -z, "chosen_weight": w,
                    "rejected_weight": -w})
    return out


def softmax(xs: Sequence[float]) -> List[float]:
    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    s = sum(e)
    return [x / s for x in e]


def decision_weights(logprobs: Sequence[float], answer: Optional[int] = None,
                     target: Optional[Sequence[float]] = None) -> Dict[str, Any]:
    """A decision as weights over its choices' log-probabilities (§9): a known answer, or a target distribution (Brier)."""
    p = softmax(logprobs)
    if answer is not None:
        return {"p": p, "loss": -math.log(p[answer]), "weights": [(1.0 if k == answer else 0.0) - pk
                                                                  for k, pk in enumerate(p)]}
    s = sum(pj * (pj - tj) for pj, tj in zip(p, target))
    return {"p": p, "loss": sum((pk - tk) ** 2 for pk, tk in zip(p, target)),
            "weights": [-2 * pk * ((pk - tk) - s) for pk, tk in zip(p, target)]}


def choose(checks: Sequence[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """The best (check, temperature): the highest score; the earlier check, then the lower temperature, on a tie."""
    best = None
    for c in checks:
        for t in sorted(c["scores"], key=float):
            v = c["scores"][t]
            if v is None or v != v:
                continue
            if best is None or v > best["score"]:
                best = {"step": c["step"], "temperature": float(t), "score": v}
    return best


__all__ = ["positions", "merged", "shift", "sft_datum", "rl_datum", "weighted_datum", "decision_datum", "advantages",
           "kl_adjusted", "energy", "dpo", "softmax", "decision_weights", "choose"]
