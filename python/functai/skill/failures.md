# FunctAI: when calls fail

| symptom | what it really is | do |
|---|---|---|
| `lmcc.Refusal` code `parse-filtered` (before 1.3: "couldn't read the reply", or `text content blocks must be non-empty`) on sensitive topics (toxins, drugs, weapons-adjacent biology) | the provider's **safety filter** stopped the reply, or the model declined (finish `content_filter`) | switch provider for that task (Opus stopped on toxicology papers; GPT-6 Astra read all 50). Prompt changes won't fix it. `on_unreadable="record"` keeps a long run going |
| `Refusal` code `parse-truncated` | reply hit the token limit; the hint says how much went to thinking | lower `reasoning`, raise `max_tokens`, or ask for less per call |
| `RateLimitError` with a note "the provider asks to wait 3 h 7 min (until …)" | a subscription's usage window is spent | functai waits at most `api_max_wait` (60 s) and raises at once beyond it, saying until when; run again then (with `cache_replies="disk"` finished rows are kept), or set `api_max_wait` to wait. Before 1.3 it **slept the full wait, silently**: there use `api_retries=0` for probes |
| `ServerError: Overloaded` (HTTP 529) | provider busy | `api_retries` handles it; raise it (`api_retries=8`) for long unattended runs |
| `LoginRequired` | login expired | run the command it prints (`functai.login("claude")`); functai never falls back to a paid key |
| "Claude Code X does not support this model; version Y or newer is required" | lm15 claims an old Claude Code version | `export LM15_CLAUDE_CODE_VERSION=Y` (no code change) |
| `InvalidRequestError: text content blocks must be non-empty` (before 1.3) | an empty, filtered reply being asked again | the safety-filter row above |

Record failures as data (the `error` column of `map`), and report them
separately from accuracy: "how often the model failed" is a result.
