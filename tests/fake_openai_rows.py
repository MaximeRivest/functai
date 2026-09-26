"""A fake OpenAI Chat Completions server for the verifiers tests: answers
sentiment in whichever layout the request uses, thinks first (a reasoning
field the task must drop), and answers nonsense on some rows (unreadable).
Deterministic.

    python tests/fake_openai_rows.py PORT
"""

import hashlib
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

POS = ("great", "loved", "amazing")
NEG = ("awful", "hated", "terrible")


def _text(m):
    c = m.get("content")
    return c if isinstance(c, str) else "".join(p.get("text", "") for p in (c or []))


def answer(messages):
    system = next((_text(m) for m in messages if m["role"] == "system"), "")
    user = [_text(m) for m in messages if m["role"] == "user"][-1]
    label = "positive" if any(w in user for w in POS) else "negative" if any(w in user for w in NEG) else "neutral"
    if int(hashlib.sha256(user.encode()).hexdigest(), 16) % 5 == 0:
        return "I would rather not say."
    if "Sentiment:" in system:
        return f"Sentiment: {label}"
    return f"<result>\n{label}\n</result>"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body):
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._send({"object": "list", "data": [{"id": "fake", "object": "model"}]})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self._send({"id": "c1", "object": "chat.completion", "created": 0, "model": req.get("model", "fake"),
                    "choices": [{"index": 0, "finish_reason": "stop", "message": {
                        "role": "assistant", "content": answer(req["messages"]),
                        "reasoning_content": "TEACHER THINKING about the review."}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}})


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), Handler).serve_forever()
