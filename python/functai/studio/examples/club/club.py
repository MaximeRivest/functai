"""A robotics club's inbox: sort each message, answer questions from the FAQ, reply.

Try it:  functai studio run  ·  functai studio open
"""

from typing import Literal

from functai import ai, module

FAQ = open("club_faq.md", encoding="utf-8").read()


@ai
def kind(message: str) -> Literal["question", "idea", "spam"]:
    """What kind of message did the club get?"""


@ai
def answer(question: str, faq: str) -> str:
    """Answer using only the club FAQ. Say "I don't know" if it isn't there."""


@ai
def reply(message: str, draft: str) -> str:
    """Write a short, friendly reply a club leader would send. Two sentences at most."""


@module
def club_inbox(message: str) -> str:
    """Sort a message sent to the club; answer questions from the FAQ; welcome ideas; ignore spam."""
    k = kind(message)
    if k == "spam":
        return ""
    draft = answer(message, FAQ) if k == "question" else "Love it! Bring it up at Thursday's meeting."
    return reply(message, draft)


if __name__ == "__main__":
    print(club_inbox("When is the next meeting?"))
