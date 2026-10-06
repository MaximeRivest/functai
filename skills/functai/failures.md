# FunctAI: when calls fail

| symptom | what it really is | do |
|---|---|---|
| `lmcc.Refusal` "couldn't read the reply" on sensitive topics (toxins, drugs, weapons-adjacent biology) | the provider's **safety filter** cut the reply (finish `content_filter`, empty text) | check `functai.phistory(1)`; switch provider for that task (Opus refused toxicology papers; GPT-6 Astra read all 50). Prompt changes won't fix it |
| `Refusal` code `parse-truncated` | reply hit the token limit; the hint says how much went to thinking | lower `reasoning`, raise `max_tokens`, or ask for less per call |
| `RateLimitError` with `retry-after` of hours (subscriptions) | the plan's usage window is spent | functai's `api_retries` (default 3) **sleeps the full retry-after, uncapped**: a run can hang for hours silently. For probes use `api_retries=0`; for big runs check the allowance first, or catch `lm15.RateLimitError` at the top and stop |
| `ServerError: Overloaded` (HTTP 529) | provider busy | `api_retries` handles it; raise it (`api_retries=8`) for long unattended runs |
| `LoginRequired` | login expired | run the command it prints (`functai.login("claude")`); functai never falls back to a paid key |
| "Claude Code X does not support this model; version Y or newer is required" | lm15 claims an old Claude Code version | `export LM15_CLAUDE_CODE_VERSION=Y` (no code change) |
| `InvalidRequestError: text content blocks must be non-empty` | an empty reply (often a filtered one) being re-sent | treat as the safety-filter row above |

Record failures as data (the `error` column of `map`), and report them
separately from accuracy: "how often the model failed" is a result.
