"""What the case scripts share: canonical JSON and hashes, written from lmcc
kernel §3a and §7a (not imported from any FunctAI implementation)."""

import hashlib
import json


def number(x: float) -> str:
    """ECMAScript's Number::toString of a finite double (lmcc §7a)."""
    if x != x or x in (float("inf"), float("-inf")):
        raise ValueError(f"{x} has no JSON form")
    if x == 0:
        return "0"
    if x < 0:
        return "-" + number(-x)
    mantissa, _, exp = repr(x).partition("e")
    whole, _, frac = mantissa.partition(".")
    e10 = int(exp) if exp else 0
    if whole.strip("0"):
        n = len(whole.lstrip("0")) + e10
    else:
        n = e10 - (len(frac) - len(frac.lstrip("0")))
    digits = ((whole + frac).lstrip("0")).rstrip("0") or "0"
    k = len(digits)
    if k <= n <= 21:
        return digits + "0" * (n - k)
    if 0 < n <= 21:
        return digits[:n] + "." + digits[n:]
    if -6 < n <= 0:
        return "0." + "0" * (-n) + digits
    e = n - 1
    return digits[0] + ("." + digits[1:] if k > 1 else "") + "e" + ("+" if e >= 0 else "-") + str(abs(e))


def canonical(value) -> str:
    """lmcc §3a: keys sorted by code point, no white space, UTF-8, numbers by §7a."""
    if value is None or value is True or value is False:
        return "null" if value is None else "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return number(value)
    if isinstance(value, dict):
        return "{" + ",".join(json.dumps(k, ensure_ascii=False) + ":" + canonical(value[k])
                              for k in sorted(value)) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(canonical(v) for v in value) + "]"
    raise TypeError(type(value).__name__)


def sha(value) -> str:
    """``"sha256:"`` and the hex SHA-256 of the canonical JSON of ``value``."""
    return "sha256:" + hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()
