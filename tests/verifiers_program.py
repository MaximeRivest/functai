"""A task for the verifiers tests: sentiment, with a second layout."""

from typing import Literal

import lmcc

from functai import ai


@ai
def sentiment(text: str) -> Literal["positive", "negative", "neutral"]:
    """The sentiment of the review."""


ADAPTERS = {
    "line": lmcc.adapter(name="sentiment_line", messages=[
        lmcc.system("{instruction}\n\nReply with one line:\nSentiment: {result}"), lmcc.turns(), lmcc.user("{text}")]),
}
