"""mitmdump addon: one JSON line per HTTPS exchange with a model provider, written to
$BENCH_RECORD (a .jsonl file). The same record for every library: what was sent
(the request body as JSON) and what it cost (the response's usage)."""
import json
import os
import time

OUT = os.environ["BENCH_RECORD"]


def response(flow):
    host = flow.request.pretty_host
    if not any(h in host for h in ("openai.com", "anthropic.com", "googleapis.com", "groq.com", "openrouter.ai")):
        return
    def body(message):
        try:
            return json.loads(message.get_text(strict=False) or "null")
        except Exception:  # noqa: BLE001  streamed or binary: keep the text
            return (message.get_text(strict=False) or "")[:200000]
    reply = body(flow.response)
    usage = reply.get("usage") if isinstance(reply, dict) else None
    if usage is None and isinstance(reply, str) and '"usage"' in reply:   # a streamed reply: usage in its last events
        for line in reversed(reply.splitlines()):
            if '"usage"' in line:
                try:
                    usage = (json.loads(line.split("data:", 1)[-1]).get("response") or {}).get("usage") \
                        or json.loads(line.split("data:", 1)[-1]).get("usage")
                except Exception:  # noqa: BLE001
                    pass
                break
    record = {"time": flow.request.timestamp_start, "seconds": round(flow.response.timestamp_end - flow.request.timestamp_start, 3),
              "host": host, "path": flow.request.path, "status": flow.response.status_code,
              "request": body(flow.request), "usage": usage}
    with open(OUT, "a") as f:
        f.write(json.dumps(record) + "\n")
