"""The cases in scores/, written from ../scores.md.

An "interval" case: {"description", "kind": "interval", "values": [...],
"expect": {"mean", "low", "high"}} (null where there is none; compare
within 1e-12). A "match" case: {"description", "kind": "match",
"answers": {output: right value}, "prediction": {output: value},
"expect": {metric: 0 or 1}}: `exact_match`, and `<name>_match` for each
compared output when there are several.
"""

import math

Z = 1.959964
T975 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145,
        2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048,
        2.045, 2.042]
WHITE = "\t\n\x0b\x0c\r\x1c\x1d\x1e\x1f \x85\xa0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008" \
        "\u2009\u200a\u2028\u2029\u202f\u205f\u3000"


def t975(df: int) -> float:
    if df <= 30:
        return T975[df - 1]
    return Z + (Z ** 3 + Z) / (4 * df) + (5 * Z ** 5 + 16 * Z ** 3 + 3 * Z) / (96 * df ** 2)


def interval(values):
    n = len(values)
    if n == 0:
        return {"mean": None, "low": None, "high": None}
    mean = 0.0
    for v in values:
        mean += v
    mean /= n
    if n < 2:
        return {"mean": mean, "low": None, "high": None}
    if all(v in (0, 1) for v in values):
        z2 = Z * Z
        center = (mean + z2 / (2 * n)) / (1 + z2 / n)
        half = Z * math.sqrt(mean * (1 - mean) / n + z2 / (4 * n * n)) / (1 + z2 / n)
        low = 0.0 if mean == 0 else max(0.0, center - half)
        high = 1.0 if mean == 1 else min(1.0, center + half)
        return {"mean": mean, "low": low, "high": high}
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
    half = t975(n - 1) * sd / math.sqrt(n)
    return {"mean": mean, "low": mean - half, "high": mean + half}


def split(text: str) -> list:
    pieces, word = [], ""
    for ch in text:
        if ch in WHITE:
            if word:
                pieces.append(word)
            word = ""
        else:
            word += ch
    return pieces + ([word] if word else [])


def norm(v):
    return " ".join(split(v)).casefold() if isinstance(v, str) else v


def match(answers: dict, prediction: dict) -> dict:
    keys = [k for k in prediction if k in answers]
    each = {k: int(norm(answers[k]) == norm(prediction[k])) for k in keys}
    out = {"exact_match": int(all(each.values()))}
    if len(keys) > 1:
        out.update({f"{k}_match": v for k, v in each.items()})
    return out


INTERVALS = {
    "01-all-right": ("Every row right: the mean is 1 and the top of the range exactly 1.", [1, 1, 1, 1, 1]),
    "02-all-wrong": ("Every row wrong: the bottom of the range is exactly 0.", [0, 0, 0]),
    "03-seven-of-ten": ("Right or wrong: Wilson's interval.", [1, 1, 0, 1, 1, 0, 1, 1, 0, 1]),
    "04-one-row": ("One row: a mean, no range.", [1]),
    "05-no-rows": ("No rows: nothing.", []),
    "06-a-judge": ("Scores between 0 and 1: Student's t with n - 1 degrees of freedom.", [0.2, 0.9, 0.5, 0.7]),
    "07-many-scores": ("Beyond 30 degrees of freedom: the Cornish-Fisher expansion.",
                       [((i * 37) % 11) / 10 for i in range(40)]),
    "08-floats-that-are-right-or-wrong": ("0.0 and 1.0 are right or wrong too.", [1.0, 0.0, 1.0, 1.0]),
    "09-counts": ("Counts are scores too.", [3, 5, 4, 8, 2, 6]),
}

MATCHES = {
    "10-space-and-case": ("Text: white space collapsed, case folded.",
                          {"result": "new york"}, {"result": "  New \t York\n"}),
    "11-full-case-folding": ("Case folding is full: ß is ss, and a final sigma is a sigma.",
                             {"result": "STRASSE ΟΔΟΣ"}, {"result": "straße οδος"}),
    "12-unicode-white-space": ("No-break and em spaces are white space.",
                               {"result": "paris"}, {"result": "\u00a0Paris\u2003"}),
    "13-not-white-space": ("U+FEFF is not white space (JavaScript's \\s says it is).",
                           {"result": "paris"}, {"result": "paris\ufeff"}),
    "14-numbers-by-value": ("Numbers compare by value.", {"result": 1}, {"result": 1.0}),
    "15-no-folding-inside": ("Text inside a list is compared as it is.", {"result": ["A"]}, {"result": ["a"]}),
    "16-several-outputs": ("Several outputs: all must match, and each is scored on its own.",
                           {"summary": "Charged twice", "result": "billing"},
                           {"summary": "charged twice", "result": "shipping", "extra": "not compared"}),
    "17-wrong-choice": ("A different answer.", {"result": "happy"}, {"result": "unhappy"}),
}


def cases() -> dict:
    out = {}
    for name, (text, values) in INTERVALS.items():
        out[name] = {"description": text, "kind": "interval", "values": values, "expect": interval(values)}
    for name, (text, answers, prediction) in MATCHES.items():
        out[name] = {"description": text, "kind": "match", "answers": answers, "prediction": prediction,
                     "expect": match(answers, prediction)}
    return out
