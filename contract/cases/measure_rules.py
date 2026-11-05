"""The rules of ../measures.md, as code: what the cases in measures/ are
written from (measures.py). Standard library only; nothing here comes from
an implementation. Every function is the section of measures.md it names.
"""

import hashlib
import math

from common import canonical, sha
from scores import Z, interval, t975

CONTEXT = ("earlier", "sections", "helpers")
SPLITS = ("train", "held_out")


class Refused(Exception):
    def __init__(self, code: str, message: str = ""):
        super().__init__(message or code)
        self.code = code


# ------------------------------------------------------------------ Case keys


def case_key(inputs: dict, context: dict = None) -> str:
    """*Case keys*: sha256 of {"inputs", and each non-empty context key}."""
    doc = {"inputs": inputs}
    for k in CONTEXT:
        v = (context or {}).get(k)
        if v not in (None, [], {}):
            doc[k] = v
    return sha(doc)


def row_key(row: dict, inputs: list) -> str:
    """A row's key: its values of the program's inputs, and its context (a key that is not an input)."""
    return case_key({n: row[n] for n in inputs if n in row},
                    {k: row[k] for k in CONTEXT if k in row and k not in inputs})


def inputs_kept(call: dict) -> bool:
    """calls.md rule 3, no_content: every input recorded as data."""
    if (call.get("described") or {}).get("inputs"):
        return False
    if call.get("content") is True:
        return "inputs" in call
    om = call.get("omitted")
    return call.get("functai_call") == 2 and isinstance(om, dict) and om.get("inputs") == []


def call_key(call: dict):
    """*A call's case*: its `case`, else its recorded inputs' key when it saw nothing; else none."""
    if isinstance(call.get("case"), str):
        return call["case"]
    if call.get("functai_call") != 2 or call.get("saw") or not inputs_kept(call):
        return None
    return case_key(call.get("inputs") or {})


# ------------------------------------------------------------------ Held out


def split_of(row: dict, inputs: list, share: float = 0.2, salt: str = "") -> str:
    """*Held out*: the row's own `split`, else by the hash of its group."""
    if "split" in row and "split" not in inputs:
        s = row["split"]
        if s not in SPLITS:
            raise Refused("case-split", f"split is train or held_out, not {s!r}")
        return s
    if not (0 <= share <= 1):
        raise Refused("case-split", "the held-out share is between 0 and 1")
    conv = row.get("conversation") if "conversation" not in inputs else None
    group = conv if isinstance(conv, str) and conv else row_key(row, inputs)
    h = hashlib.sha256(f"functai-split:{salt}:{group}".encode("utf-8")).hexdigest()
    return "held_out" if int(h[:13], 16) / 2 ** 52 < share else "train"


# ------------------------------------------------------------------ Random draws


def draw(ids: list, n: int, seed: str) -> list:
    """*Random draws*: the n ids (each once) with the smallest sha256("functai-draw:" + seed + ":" + id)."""
    ranked = sorted(dict.fromkeys(ids),
                    key=lambda i: (hashlib.sha256(f"functai-draw:{seed}:{i}".encode("utf-8")).hexdigest(), i))
    return ranked[:max(0, n)]


# ------------------------------------------------------------------ Ratings


def source_of(r: dict):
    """(kind, id): a person by name, an account rating by its own id, a judge by its setup or code."""
    if isinstance(r.get("by"), str) and r["by"]:
        return ("person", r["by"])
    if isinstance(r.get("judge"), dict):
        j = r["judge"]
        return ("judge", j.get("setup") or j.get("code"))
    return ("account", r["id"])


def judgement(r: dict):
    """A rating record of format 1 or 2 as one shape (*Reading ratings*); None for another format."""
    fmt = r.get("functai_rating")
    if fmt == 1:
        return {"id": r["id"], "at": r["at"], "calls": [r["call"]], "item": "", "criterion": "correct",
                "kind": "verdict", "value": r["verdict"], "source": source_of(r),
                "blind": r.get("blind") is True, "sample": r.get("sample")}
    if fmt == 2:
        calls = list(r["calls"]) if "calls" in r else [r["call"]]
        out = {"id": r["id"], "at": r["at"], "calls": calls, "item": r.get("item", ""),
               "criterion": r["criterion"], "kind": r["kind"], "value": r["value"], "source": source_of(r),
               "blind": r.get("blind") is True, "sample": r.get("sample")}
        if r["kind"] == "label":
            out["answers"] = r["answers"]
        if r["kind"] == "score" and "scale" in r:
            out["scale"] = r["scale"]
        if isinstance(r.get("judge"), dict):
            out["judge"] = r["judge"]["name"]
        return out
    return None


def later(a: dict, b: dict) -> bool:
    return (a["at"], a["id"]) > (b["at"], b["id"])


def counting(ratings: list, criterion: dict) -> list:
    """*What counts*: per target and criterion, a person's latest (unordered pair), a judge's latest per
    order shown, every account rating on its own; null values withdraw (a person's, a judge's) or count
    for nothing (an account's). In the order of (at, id)."""
    latest = {}
    for r in ratings:
        j = judgement(r)
        if j is None or j["criterion"] != criterion["name"] or j["kind"] != criterion["kind"]:
            continue
        kind, who = j["source"]
        target = (tuple(sorted(j["calls"])) if kind == "person" else tuple(j["calls"]), j["item"])
        key = (kind, who, target) if kind != "account" else (kind, who)
        if key not in latest or later(j, latest[key]):
            latest[key] = j
    kept = [j for j in latest.values() if j["value"] is not None]
    return sorted(kept, key=lambda j: (j["at"], j["id"]))


def number_of(j: dict):
    """A single answer's judgement as a number (*Kinds*): verdict 1 or 0, a label its answer's number (None:
    counts for nothing), a score on its scale from 0 to 1 (as it is without one)."""
    k, v = j["kind"], j["value"]
    if k == "verdict":
        return 1.0 if v == "right" else 0.0
    if k == "label":
        return j["answers"].get(v)
    if k == "score":
        if "scale" in j:
            lo, hi = j["scale"]
            return (v - lo) / (hi - lo)
        return float(v)
    raise ValueError(k)


def source_matches(j: dict, source: str, drawn: bool) -> bool:
    """`people` (persons and accounts; `drawn` keeps only those of a random draw), or a judge's id."""
    kind, who = j["source"]
    if source == "people":
        return kind in ("person", "account") and (not drawn or bool(j["sample"]))
    return kind == "judge" and who == source


# ------------------------------------------------------------------ Results


def program_calls(records: list, program: dict) -> list:
    """The calls of a program (name, and module and file when given), latest writer only, by (started, id)."""
    best = {}
    for r in records:
        if r.get("functai_call") not in (1, 2):
            continue
        p = r.get("program") or {}
        if p.get("name") != program["name"]:
            continue
        if program.get("module") is not None and p.get("module") != program["module"]:
            continue
        if program.get("file") is not None and p.get("file") != program["file"]:
            continue
        w = r.get("writer", 1)
        if r["id"] not in best or w > best[r["id"]].get("writer", 1):
            best[r["id"]] = r
    return sorted(best.values(), key=lambda c: (c["started"], c["id"]))


def results(records: list, program: dict, by: str = "setup") -> list:
    """*Results*: each call with a case key, its group (`program.setup`, or `program.version` with by
    "version"; null when the record has none) and its key."""
    out = []
    for c in program_calls(records, program):
        k = call_key(c)
        if k is None:
            continue
        p = c["program"]
        out.append({"call": c, "group": p.get(by), "case": k, "failed": c.get("error") is not None})
    return out


def pointer(value, path: str):
    """RFC 6901 into a call's outputs; a missing step gives the marker MISSING."""
    if path == "":
        return value
    for raw in path.split("/")[1:]:
        tok = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(value, dict) and tok in value:
            value = value[tok]
        elif isinstance(value, list) and tok.isdigit() and (tok == "0" or not tok.startswith("0")) and int(tok) < len(value):
            value = value[int(tok)]
        else:
            return MISSING
    return value


MISSING = object()


def mean(xs):
    s = 0.0
    for x in xs:
        s += x
    return s / len(xs)


def result_value(res: dict, criterion: dict, by_target: dict):
    """*The value of a result*: a whole answer's mean judgement; an item criterion's share of good items
    ("mean") or that share times the number of items ("sum"); a failed call 0, or no share and a count
    of 0. None when nothing gives one."""
    items = criterion.get("items")
    agg = criterion.get("aggregate", "mean")
    cid = res["call"]["id"]
    if res["failed"]:
        if items is None:
            return 0.0
        return 0.0 if agg == "sum" else None
    if items is None:
        vs = by_target.get(((cid,), ""))
        return mean(vs) if vs else None
    lst = pointer(res["call"].get("outputs"), items)
    if not isinstance(lst, list):
        return 0.0 if agg == "sum" else None
    per_item = []
    for i in range(len(lst)):
        vs = by_target.get(((cid,), f"{items}/{i}"))
        if vs:
            per_item.append(mean(vs))
    if not per_item:
        return 0.0 if agg == "sum" and not lst else None
    share = mean(per_item)
    return share * len(lst) if agg == "sum" else share


def judged(ratings: list, criterion: dict, source: str, drawn: bool = False) -> dict:
    """{(calls, item): [numbers]} of the counting judgements of one source (single answers)."""
    out = {}
    for j in counting(ratings, criterion):
        if len(j["calls"]) != 1 or not source_matches(j, source, drawn):
            continue
        x = number_of(j)
        if x is None:
            continue
        out.setdefault((tuple(j["calls"]), j["item"]), []).append(x)
    return out


def values(records: list, program: dict, criterion: dict, source: str, *, by: str = "setup",
           drawn: bool = False) -> dict:
    """*Values*: per result, then {group: {case: mean over its results}}, groups and cases in order of
    first result."""
    ratings = [r for r in records if "functai_rating" in r]
    by_target = judged(ratings, criterion, source, drawn)
    per_result, table = [], {}
    for res in results(records, program, by):
        v = result_value(res, criterion, by_target)
        per_result.append({"call": res["call"]["id"], "group": res["group"], "case": res["case"], "value": v})
        if v is not None:
            table.setdefault(res["group"], {}).setdefault(res["case"], []).append(v)
    cases = {g: {k: mean(vs) for k, vs in cs.items()} for g, cs in table.items()}
    return {"results": per_result, "cases": cases}


def preferences(records: list, program: dict, criterion: dict, source: str, a, b, *, by: str = "setup",
                drawn: bool = False) -> dict:
    """*Side by side*: {case: B's share} over the pairs of one A result and one B result on a case: a
    failed answer loses (both failed: a tie), else the mean of the source's counting picks (B 1, A 0, tie
    one half). Cases in key order."""
    ratings = [r for r in records if "functai_rating" in r]
    picks = {}
    for j in counting(ratings, criterion):
        if len(j["calls"]) == 2 and source_matches(j, source, drawn):
            picks.setdefault(frozenset(j["calls"]), []).append(j)
    res = results(records, program, by)
    ra = [r for r in res if r["group"] == a]
    rb = [r for r in res if r["group"] == b]
    out = {}
    for x in ra:
        for y in rb:
            if x["case"] != y["case"]:
                continue
            xid, yid = x["call"]["id"], y["call"]["id"]
            if x["failed"] or y["failed"]:
                v = 0.5 if x["failed"] and y["failed"] else (1.0 if x["failed"] else 0.0)
            else:
                js = picks.get(frozenset((xid, yid)))
                if not js:
                    continue
                v = mean([1.0 if j["value"] == yid else 0.0 if j["value"] == xid else 0.5 for j in js])
            out.setdefault(x["case"], []).append(v)
    return {k: mean(out[k]) for k in sorted(out)}


# ------------------------------------------------------------------ Statistics


def wilson(x: float, n: int):
    p = x / n
    z2 = Z * Z
    center = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = Z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (0.0 if p == 0 else max(0.0, center - half), 1.0 if p == 1 else min(1.0, center + half))


def outcome(low, high, margin=None) -> str:
    """*Outcomes*."""
    if low is None:
        return "unclear"
    if low > 0:
        return "better"
    if high < 0:
        return "worse"
    if margin:
        if low > -margin and high < margin:
            return "same"
        if low > -margin:
            return "not_worse"
        if high < margin:
            return "not_better"
    return "unclear"


def fragility(d: list) -> int:
    """*Fragility*: the fewest values whose removal takes the mean to 0 or across it (largest first)."""
    if not d:
        return 0
    m = mean(d)
    if m == 0:
        return 0
    s = 1.0 if m > 0 else -1.0
    order = sorted(range(len(d)), key=lambda i: (-s * d[i], i))
    left = [s * d[i] for i in order]
    total = sum(left)
    for k in range(1, len(d) + 1):
        total -= left[k - 1]
        rest = len(d) - k
        if rest == 0 or total / rest <= 0:
            return k
    return len(d)


def bounded(criterion: dict) -> bool:
    """*Bounds*: a criterion's values lie between 0 and 1 (a verdict, a label, a score on a scale), except a
    count of items (aggregate "sum") and a score without a scale."""
    if criterion.get("items") is not None and criterion.get("aggregate", "mean") == "sum":
        return False
    return criterion["kind"] != "score" or "scale" in criterion


def paired(a: list, b: list, margin=None, bounded_: bool = False) -> dict:
    """*Is B better than A*: a and b the two setups' values on the shared cases, in the same order; a bounded
    difference's range is kept within -1 and 1."""
    n = len(a)
    if n == 0:
        return {"diff": None, "low": None, "high": None, "n": 0, "method": None, "outcome": "unclear",
                "better": 0, "worse": 0, "same": 0, "fragility": 0}
    d = [y - x for x, y in zip(a, b)]
    diff = mean(d)
    low = high = method = None
    if n >= 2:
        if all(v in (0, 1) for v in a + b):
            method = "mover-wilson"
            n11 = sum(1 for x, y in zip(a, b) if x == 1 and y == 1)
            n10 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
            n01 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
            n00 = n - n11 - n10 - n01
            pa, pb = (n11 + n10) / n, (n11 + n01) / n
            la, ua = wilson(n11 + n10, n)
            lb, ub = wilson(n11 + n01, n)
            num = n11 * n00 - n10 * n01
            if num > 0:
                num = max(num - n / 2, 0.0)
            den = (n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)
            phi = 0.0 if den == 0 else num / math.sqrt(den)
            dl = math.sqrt(max(0.0, (pb - lb) ** 2 - 2 * phi * (pb - lb) * (ua - pa) + (ua - pa) ** 2))
            du = math.sqrt(max(0.0, (ub - pb) ** 2 - 2 * phi * (ub - pb) * (pa - la) + (pa - la) ** 2))
            low, high = max(-1.0, diff - dl), min(1.0, diff + du)
        else:
            method = "paired-t"
            ss = sum((v - diff) ** 2 for v in d)
            half = t975(n - 1) * math.sqrt(ss / (n - 1)) / math.sqrt(n)
            low, high = diff - half, diff + half
            if bounded_:
                low, high = max(-1.0, low), min(1.0, high)
    return {"diff": diff, "low": low, "high": high, "n": n, "method": method, "outcome": outcome(low, high, margin),
            "better": sum(1 for v in d if v > 0), "worse": sum(1 for v in d if v < 0),
            "same": sum(1 for v in d if v == 0), "fragility": fragility(d)}


def win_rate(v: list, margin=None) -> dict:
    """*Does B win side by side*: the mean of B's shares, Wilson's interval on it; the outcome of its
    distance from one half."""
    n = len(v)
    if n == 0:
        return {"rate": None, "low": None, "high": None, "n": 0, "outcome": "unclear", "fragility": 0}
    rate = mean(v)
    low, high = wilson(sum(v), n) if n >= 2 else (None, None)
    o = outcome(None if low is None else low - 0.5, None if high is None else high - 0.5, margin)
    return {"rate": rate, "low": low, "high": high, "n": n, "outcome": o,
            "fragility": fragility([x - 0.5 for x in v])}


def var(xs):
    m = mean(xs)
    return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)


def ppi(judge: dict, people: dict, bounded_: bool = True, lo: float = 0.0, hi: float = 1.0):
    """*A judge on everything, people on a random few*: PPI++ over cases. judge, people: {case: value}. Bounded
    values (between lo and hi): pseudo-observations half the range either side, the range kept within; else
    half the range of the values seen, and no bounds."""
    labeled = sorted(k for k in people if k in judge)
    unlabeled = sorted(k for k in judge if k not in people)
    n, big_n = len(labeled), len(unlabeled)
    if n < 3:
        return None
    y = [people[k] for k in labeled]
    f = [judge[k] for k in labeled]
    fu = [judge[k] for k in unlabeled]
    every = f + fu
    vf = var(every) if len(every) >= 2 else 0.0
    if big_n == 0 or vf == 0:
        lam = 0.0
    else:
        my, mf = mean(y), mean(f)
        c = sum((a - my) * (b - mf) for a, b in zip(y, f)) / (n - 1)
        lam = min(1.0, max(0.0, c / ((1 + n / big_n) * vf)))
    if bounded_:
        half_range = (hi - lo) / 2
    else:
        seen = y + every
        half_range = (max(seen) - min(seen)) / 2
    r = [a - lam * b for a, b in zip(y, f)]
    mr = mean(r)
    est = (lam * mean(fu) if big_n else 0.0) + mr
    vr = (sum((x - mr) ** 2 for x in r) + 2 * half_range ** 2) / (n + 1)
    se = math.sqrt((lam * lam * vf / big_n if big_n else 0.0) + vr / n)
    q = t975(n - 2)
    low, high = est - q * se, est + q * se
    if bounded_:
        low, high = max(lo, low), min(hi, high)
    return {"mean": est, "low": low, "high": high, "n": n,
            "unlabeled": big_n, "lambda": lam}


def ppi_paired(judge_a: dict, judge_b: dict, people_a: dict, people_b: dict, margin=None, bounded_: bool = True):
    """PPI++ on the differences B - A: cases both setups have, for the judge and for the people."""
    jd = {k: judge_b[k] - judge_a[k] for k in judge_a if k in judge_b}
    pd = {k: people_b[k] - people_a[k] for k in people_a if k in people_b}
    out = ppi(jd, pd, bounded_, lo=-1.0, hi=1.0)
    if out is None:
        return None
    return {"diff": out["mean"], "low": out["low"], "high": out["high"], "n": out["n"],
            "unlabeled": out["unlabeled"], "lambda": out["lambda"], "outcome": outcome(out["low"], out["high"], margin)}


def kappa(xs: list, ys: list):
    cats = sorted(set(xs) | set(ys))
    n = len(xs)
    po = sum(1 for a, b in zip(xs, ys) if a == b) / n
    pe = sum((xs.count(c) / n) * (ys.count(c) / n) for c in cats)
    return None if pe == 1 else (po - pe) / (1 - pe)


def agreement(a: dict, b: dict, choice: bool = False) -> dict:
    """*Can this judge stand in for people*: a, b {target: value} of two sources (side by side: B's share).
    Values all 0 or 1, or a side-by-side criterion: categories (a share is "first", "tie" or "second"):
    the share that agree (Wilson) and Cohen's kappa; otherwise the mean absolute difference and Pearson's r."""
    keys = sorted(k for k in a if k in b)
    n = len(keys)
    if n == 0:
        return {"n": 0, "agree": None, "low": None, "high": None, "kappa": None, "mad": None, "r": None}
    return describe_agreement([a[k] for k in keys], [b[k] for k in keys], choice)


def category(v: float) -> str:
    return "second" if v > 0.5 else "first" if v < 0.5 else "tie"


def describe_agreement(xs, ys, choice=False):
    n = len(xs)
    if choice or all(v in (0, 1) for v in xs + ys):
        cx = [category(v) for v in xs] if choice else xs
        cy = [category(v) for v in ys] if choice else ys
        same = sum(1 for p, q in zip(cx, cy) if p == q)
        low, high = wilson(same, n) if n >= 2 else (None, None)
        return {"n": n, "agree": same / n, "low": low, "high": high, "kappa": kappa(cx, cy), "mad": None, "r": None}
    mad = mean([abs(p - q) for p, q in zip(xs, ys)])
    mx, my = mean(xs), mean(ys)
    sx = math.sqrt(sum((p - mx) ** 2 for p in xs))
    sy = math.sqrt(sum((q - my) ** 2 for q in ys))
    r = None if sx == 0 or sy == 0 else sum((p - mx) * (q - my) for p, q in zip(xs, ys)) / (sx * sy)
    return {"n": n, "agree": None, "low": None, "high": None, "kappa": None, "mad": mad, "r": r}


def positions(ratings: list, criterion: dict, judge_id: str) -> dict:
    """*Position*: a judge's picks shown in both orders: how often it picked the same answer both times, and
    of its picks that were not ties, how often the first shown won."""
    js = [j for j in counting(ratings, criterion) if len(j["calls"]) == 2 and j["source"] == ("judge", judge_id)]
    by_pair = {}
    for j in js:
        by_pair.setdefault(frozenset(j["calls"]), {})[tuple(j["calls"])] = j["value"]
    both = [p for p in sorted(by_pair, key=lambda s: sorted(s)) if len(by_pair[p]) == 2]
    same = sum(1 for p in both if len(set(by_pair[p].values())) == 1)
    decided = [j for j in js if j["value"] != "tie"]
    first = sum(1 for j in decided if j["value"] == j["calls"][0])
    cl, ch = wilson(same, len(both)) if len(both) >= 2 else (None, None)
    fl, fh = wilson(first, len(decided)) if len(decided) >= 2 else (None, None)
    return {"pairs": len(both), "consistent": same / len(both) if both else None, "consistent_low": cl,
            "consistent_high": ch, "picks": len(decided), "first_wins": first / len(decided) if decided else None,
            "first_low": fl, "first_high": fh}


def self_agreement(per_case: dict) -> dict:
    """*Does a setup agree with itself*: per_case {case: [values of its results]}; cases with two or more."""
    cases = [k for k in sorted(per_case) if len(per_case[k]) >= 2]
    if not cases:
        return {"cases": 0, "mean": None, "low": None, "high": None, "spread": None}
    agree, within = [], []
    for k in cases:
        vs = per_case[k]
        pairs = [(vs[i], vs[j]) for i in range(len(vs)) for j in range(i + 1, len(vs))]
        agree.append(mean([1 - abs(p - q) for p, q in pairs]))
        m = mean(vs)
        within.append(sum((v - m) ** 2 for v in vs) / len(vs))
    iv = interval(agree)
    clip = (lambda x: x) if iv["low"] is None else (lambda x: min(1.0, max(0.0, x)))
    return {"cases": len(cases), "mean": iv["mean"], "low": clip(iv["low"]), "high": clip(iv["high"]),
            "spread": math.sqrt(mean(within))}


# ------------------------------------------------------------------ Bradley–Terry


def bt_fit(versions: list, outcomes: list):
    """Hunter's MM for Bradley–Terry with fractional wins and one virtual tie per pair that met. outcomes:
    [(i, j, share of i)], i and j indices of versions. Returns (strengths or None per version, the
    reference index, the component)."""
    k = len(versions)
    met = {}
    for i, j, _s in outcomes:
        if i != j:
            met[(min(i, j), max(i, j))] = True
    allo = list(outcomes) + [(i, j, 0.5) for (i, j) in sorted(met)]
    adj = {i: set() for i in range(k)}
    for (i, j) in met:
        adj[i].add(j)
        adj[j].add(i)
    ref = next((i for i in range(k) if adj[i]), None)
    if ref is None:
        return [None] * k, None, [], None, None
    comp, todo = {ref}, [ref]
    while todo:
        x = todo.pop()
        for y in sorted(adj[x]):
            if y not in comp:
                comp.add(y)
                todo.append(y)
    comp = sorted(comp)
    wins = [0.0] * k
    count = [[0.0] * k for _ in range(k)]
    for i, j, s in allo:
        if i == j:
            continue
        wins[i] += s
        wins[j] += 1 - s
        count[i][j] += 1
        count[j][i] += 1
    p = [1.0] * k
    for _ in range(10000):
        new = list(p)
        for i in comp:
            denom = 0.0
            for j in comp:
                if j != i and count[i][j]:
                    denom += count[i][j] / (p[i] + p[j])
            new[i] = wins[i] / denom
        r = new[ref]
        new = [x / r if idx in comp else 1.0 for idx, x in enumerate(new)]
        delta = max(abs(math.log(new[i]) - math.log(p[i])) for i in comp)
        p = new
        if delta < 1e-12:
            break
    s = [math.log(p[i]) if i in comp else None for i in range(k)]
    return s, ref, comp, p, count


def invert(m: list) -> list:
    """Gauss–Jordan with partial pivoting (the largest absolute value, the first on ties)."""
    n = len(m)
    a = [list(row) + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(m)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: (abs(a[r][c]), -r))
        a[c], a[piv] = a[piv], a[c]
        pv = a[c][c]
        a[c] = [x / pv for x in a[c]]
        for r in range(n):
            if r != c and a[r][c] != 0:
                f = a[r][c]
                a[r] = [x - f * y for x, y in zip(a[r], a[c])]
    return [row[n:] for row in a]


def rank(versions: list, outcomes: list) -> dict:
    """*How do many versions rank*: per version its strength (the reference's 0), its range, and the expected
    share of wins against the reference; the order; the fewest cases that change the leader (at most).
    outcomes: [(case, version i, version j, share of i)]."""
    idx = {v: n for n, v in enumerate(versions)}
    oc = [(idx[i], idx[j], s) for _c, i, j, s in outcomes]
    fit = bt_fit(versions, oc)
    s, ref = fit[0], fit[1]
    rows = []
    if ref is None:
        return {"versions": [{"version": v, "strength": None, "low": None, "high": None, "vs_reference": None}
                             for v in versions], "order": [], "reference": None, "fragility": None}
    comp, p, count = fit[2], fit[3], fit[4]
    others = [i for i in comp if i != ref]
    info = [[0.0] * len(others) for _ in others]
    for a_, i in enumerate(others):
        for j in comp:
            if j == i or not count[i][j]:
                continue
            w = count[i][j] * (p[i] / (p[i] + p[j])) * (p[j] / (p[i] + p[j]))
            info[a_][a_] += w
            if j != ref:
                info[a_][others.index(j)] -= w
    inv = invert(info) if others else []
    for n, v in enumerate(versions):
        if s[n] is None:
            rows.append({"version": v, "strength": None, "low": None, "high": None, "vs_reference": None})
        elif n == ref:
            rows.append({"version": v, "strength": 0.0, "low": 0.0, "high": 0.0, "vs_reference": 0.5})
        else:
            se = math.sqrt(max(0.0, inv[others.index(n)][others.index(n)]))
            rows.append({"version": v, "strength": s[n], "low": s[n] - Z * se, "high": s[n] + Z * se,
                         "vs_reference": p[n] / (p[n] + 1.0)})
    order = sorted([n for n in range(len(versions)) if s[n] is not None], key=lambda n: (-s[n], n))
    return {"versions": rows, "order": [versions[n] for n in order], "reference": versions[ref],
            "fragility": rank_fragility(versions, outcomes, order)}


def leader_gap(versions, outcomes):
    idx = {v: n for n, v in enumerate(versions)}
    s = bt_fit(versions, [(idx[i], idx[j], x) for _c, i, j, x in outcomes])[0]
    live = [n for n in range(len(versions)) if s[n] is not None]
    if len(live) < 2:
        return None, None
    top = sorted(live, key=lambda n: (-s[n], n))[0]
    return top, s


def rank_fragility(versions, outcomes, order, limit: int = 20):
    """At most how many cases, removed one at a time (each time the one that most narrows the leader's lead,
    the first case on ties within 1e-9), change which version leads; None when `limit` removals do not."""
    if len(order) < 2:
        return None
    leader = order[0]
    kept = list(outcomes)
    for step in range(1, limit + 1):
        cases = sorted({c for c, *_ in kept})
        if not cases:
            return None
        best = None
        for c in cases:
            rest = [o for o in kept if o[0] != c]
            top, s = leader_gap(versions, rest)
            if top is None or s[leader] is None:
                continue                                # no ranking left, or the leader left it: not a change
            if top != leader:
                return step
            gap = s[leader] - max(s[n] for n in range(len(versions)) if n != leader and s[n] is not None)
            if gap <= 0:
                return step
            if best is None or gap < best[0] - 1e-9:
                best = (gap, c)
        if best is None:
            return None
        kept = [o for o in kept if o[0] != best[1]]
    return None


# ------------------------------------------------------------------ Scoreboard


def trees(records: list, roots: list) -> dict:
    """{call id: the calls of its tree} (itself and every call inside it, by `parent`), latest writers."""
    best = {}
    for r in records:
        if r.get("functai_call") in (1, 2) and (r["id"] not in best or r.get("writer", 1) > best[r["id"]].get("writer", 1)):
            best[r["id"]] = r
    kids = {}
    for r in best.values():
        if r.get("parent"):
            kids.setdefault(r["parent"], []).append(r)
    out = {}
    for c in roots:
        tree, todo = [], [c]
        while todo:
            x = todo.pop()
            tree.append(x)
            todo.extend(kids.get(x["id"], []))
        out[c["id"]] = tree
    return out


def usage_sum(trees_: list) -> dict:
    """Tokens of results' trees: the sum over every call that asked a model; null when one of them did not
    report the count. What code spent: the sum of `spent.usd` over the trees."""
    calls = [c for t in trees_ for c in t]
    out = {}
    for key in ("input_tokens", "output_tokens"):
        asked = [c for c in calls if c.get("model") is not None]
        vs = [(c.get("usage") or {}).get(key) for c in asked]
        out[key] = sum(vs) if all(isinstance(v, int) for v in vs) else None
    out["spent_usd"] = sum(((c.get("spent") or {}).get("usd") or 0) for c in calls)
    return out


def estimate(per_case: dict, bounded_: bool) -> dict:
    """One group's values over its cases (key order): scores.md's interval, kept within 0 and 1 when bounded."""
    vs = [per_case[k] for k in sorted(per_case)]
    iv = interval(vs)
    low, high = iv["low"], iv["high"]
    if bounded_ and low is not None:
        low, high = max(0.0, low), min(1.0, high)
    return {"mean": iv["mean"], "low": low, "high": high, "cases": len(vs)}


def scoreboard(records: list, program: dict, criteria: list, *, by: str = "setup") -> dict:
    """*Scoreboard*: per group, in order of first result: its results, failures, seconds, tokens and spending;
    then per criterion (single answers) the estimate of each source: people, people drawn, each judge, and
    each judge combined with people drawn (PPI++)."""
    res = results(records, program, by)
    groups = []
    for r in res:
        if r["group"] not in groups:
            groups.append(r["group"])
    ratings = [r for r in records if "functai_rating" in r]
    rows = []
    tree = trees(records, [r["call"] for r in res])
    for g in groups:
        calls = [r["call"] for r in res if r["group"] == g]
        row = {"group": g, "results": len(calls), "cases": len({r["case"] for r in res if r["group"] == g}),
               "failed": sum(1 for c in calls if c.get("error") is not None),
               "seconds": mean([c["seconds"] for c in calls]), **usage_sum([tree[c["id"]] for c in calls]),
               "measures": []}
        for crit in criteria:
            agg = crit.get("aggregate", "mean") if crit.get("items") is not None else None
            judges = []
            for j in counting(ratings, crit):
                if j["source"][0] == "judge" and j["source"][1] not in [x[0] for x in judges]:
                    judges.append((j["source"][1], j.get("judge")))
            sources = [("people", None, False), ("people", None, True)] + [(jid, name, False) for jid, name in judges]
            got = {}
            for src, name, drawn in sources:
                cs = values(records, program, crit, src, by=by, drawn=drawn)["cases"].get(g, {})
                got[(src, drawn)] = cs
                if cs:
                    row["measures"].append({"criterion": crit["name"], "aggregate": agg,
                                            "source": "people drawn" if drawn else "people" if src == "people"
                                            else "judge", "judge": name, **estimate(cs, bounded(crit))})
            for jid, name in judges:
                comb = ppi(got[(jid, False)], got[("people", True)], bounded(crit))
                if comb is not None:
                    row["measures"].append({"criterion": crit["name"], "aggregate": agg, "source": "combined",
                                            "judge": name,
                                            "mean": comb["mean"], "low": comb["low"], "high": comb["high"],
                                            "cases": comb["n"] + comb["unlabeled"]})
        rows.append(row)
    return {"groups": rows}


def compare_groups(records: list, program: dict, criterion: dict, source: str, a, b, *, by="setup",
                   drawn=False, margin=None) -> dict:
    """Two groups' values on the cases both have (key order), paired."""
    cs = values(records, program, criterion, source, by=by, drawn=drawn)["cases"]
    ca, cb = cs.get(a, {}), cs.get(b, {})
    shared = sorted(k for k in ca if k in cb)
    out = paired([ca[k] for k in shared], [cb[k] for k in shared], margin, bounded(criterion))
    out["only_a"] = sum(1 for k in ca if k not in cb)
    out["only_b"] = sum(1 for k in cb if k not in ca)
    return out


def disagreements(records: list, program: dict, criterion: dict, source: str, *, by="setup") -> list:
    """*Disagreements*: cases on which groups' values differ (`groups`), then cases on which one group's
    results differ (`repeats`); each by its spread (largest first), then key."""
    v = values(records, program, criterion, source, by=by)
    cases = v["cases"]
    keys = sorted({k for cs in cases.values() for k in cs})
    out = []
    for k in keys:
        per = [{"group": g, "value": cs[k]} for g, cs in cases.items() if k in cs]
        vs = [x["value"] for x in per]
        if len(per) >= 2 and max(vs) - min(vs) > 0:
            out.append({"case": k, "kind": "groups", "spread": max(vs) - min(vs), "values": per})
    rep = {}
    for r in v["results"]:
        if r["value"] is not None:
            rep.setdefault((r["group"], r["case"]), []).append(r["value"])
    for (g, k), vs in sorted(rep.items(), key=lambda x: ("" if x[0][0] is None else str(x[0][0]), x[0][1])):
        if len(vs) >= 2 and max(vs) - min(vs) > 0:
            out.append({"case": k, "kind": "repeats", "spread": max(vs) - min(vs),
                        "values": [{"group": g, "value": v} for v in vs]})
    return sorted(out, key=lambda x: (x["kind"] != "groups", -x["spread"], x["case"],
                                      "" if x["kind"] == "groups" or x["values"][0]["group"] is None
                                      else str(x["values"][0]["group"])))
