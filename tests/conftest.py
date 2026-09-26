"""Offline tests: a fake lm15 router stands in for every provider."""

import threading

import lm15
import pytest

import functai
from functai import config as _config


class Resolution:
    def __init__(self, provider, model):
        self.provider, self.model = provider, model


def _text(parts):
    return "".join(getattr(p, "text", "") or "" for p in parts)


class FakeRouter:
    """Answers each request with ``responder(request)`` (a string, a list of lm15
    parts, a ``(reply, finish_reason)`` pair or an lm15 Response), or with the
    scripted replies in order. Records every request."""

    def __init__(self, *replies, responder=None, provider="openai"):
        self.replies = list(replies)
        self.responder = responder
        self.provider = provider
        self.requests = []
        self._lock = threading.Lock()

    def resolve(self, model):
        if ":" in model:
            provider, _, rest = model.partition(":")
            return Resolution(provider, rest)
        return Resolution(self.provider, model)

    def complete(self, request):
        with self._lock:
            self.requests.append(request)
            reply = self.responder(request) if self.responder else self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, lm15.Response):
            return reply
        finish = "stop"
        if isinstance(reply, tuple):
            reply, finish = reply
        parts = reply if isinstance(reply, list) else [lm15.TextPart(reply)]
        return lm15.Response(id="r", model=request.model, message=lm15.Message.assistant(parts),
                             finish_reason=finish,
                             usage=lm15.Usage(input_tokens=3, output_tokens=2, total_tokens=5))

    # helpers for assertions
    def system(self, i=-1):
        return self.requests[i].system

    def user(self, i=-1):
        return _text(self.requests[i].messages[-1].parts)

    def roles(self, i=-1):
        return [m.role for m in self.requests[i].messages]


@pytest.fixture(autouse=True)
def clean():
    before = dict(_config._GLOBAL)
    _config._GLOBAL.clear()
    functai.clear_cache()
    functai.clear_history()
    yield
    _config._GLOBAL.clear()
    _config._GLOBAL.update(before)


@pytest.fixture
def fake():
    """``router = fake("<result>\\nhi\\n</result>")`` — configure a model and a fake client."""
    def make(*replies, responder=None, provider="openai", lm="gpt-4.1-mini"):
        r = FakeRouter(*replies, responder=responder, provider=provider)
        functai.configure(lm=lm, client=r)
        return r
    return make
