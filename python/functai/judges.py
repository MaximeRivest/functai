"""Checking a judge's evidence.

A judge written as an ordinary AI function (a 0–10 score with the quotes it
rests on) is only as good as its quotes: a judge that invents evidence is
caught by checking that every quote is really in the text.

    @ai
    def judge(answer: str, source: str) -> Verdict:
        \"\"\"Score the answer from 0 to 10 for faithfulness to the source.
        Quote, word for word, the sentences of the source your score rests on.\"\"\"

    v = judge(answer, source)
    functai.quotes_found(source, v.quotes)          # [True, True, False]: the third is invented
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, List, Union

_SAME = {
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'", "\u2032": "'", "`": "'", "\u00b4": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"', "\u2033": '"', "\u00ab": '"', "\u00bb": '"',
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-", "\u2015": "-", "\u2212": "-",
    "\u2026": "...", "\u00a0": " ",
}
_SPACE = re.compile(r"\s+")
_EDGES = " \t\n\"'.,;:!?…“”‘’«»()[]"


def _plain(text: str) -> str:
    """Text as the check compares it: Unicode compatibility form, case
    folded, curly quotes straight, dashes one dash, white space one space."""
    text = unicodedata.normalize("NFKC", str(text))
    text = "".join(_SAME.get(ch, ch) for ch in text)
    return _SPACE.sub(" ", text).strip().casefold()


def quotes_found(text: str, quotes: Union[str, Iterable[str]]) -> Union[bool, List[bool]]:
    '''Whether each quote is in the text, word for word.

    White space, case, curly and straight quotes, dashes and a quote's own
    surrounding quotation marks and final punctuation do not count; any
    other difference does (a changed word, a paraphrase, an invented
    sentence). Deterministic, and costs nothing.

    Parameters
    ----------
    text : str
        The source the quotes should come from.
    quotes : str or list of str
        One quote, or several.

    Returns
    -------
    bool or list of bool
        For one quote, whether it is found; for a list, one answer per quote.

    Examples
    --------
    ```python
    source = "The parcel left Leeds on Monday. It was delayed by snow."
    functai.quotes_found(source, ["“It was delayed by snow”", "It was lost"])
    ```
    '''
    haystack = _plain(text)
    if isinstance(quotes, str):
        return _found(haystack, quotes)
    return [_found(haystack, q) for q in quotes]


def _found(haystack: str, quote: str) -> bool:
    needle = _plain(quote).strip(_EDGES)
    return bool(needle) and needle in haystack


__all__ = ["quotes_found"]
