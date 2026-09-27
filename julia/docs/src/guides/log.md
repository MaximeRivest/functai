# The call log, ratings and versions

## Turning it on

The log is off until asked for:

```julia
FunctAI.configure!(log_calls = true)          # the default folder
FunctAI.configure!(log_calls = "~/calls")     # a folder
```

or `FUNCTAI_LOG_CALLS=1` (or a folder) in the environment. The default folder is `$XDG_DATA_HOME/functai/calls` (`~/.local/share/functai/calls` on Linux), the same one FunctAI uses in Python, TypeScript and R. `log_calls = false` on a function wins over everything. `log_content = false` records sizes, times and tokens, never values or messages: for functions that see private data.

Every call is one line of JSON, written when it ends, in a file of this process and day. A folder that can't be written warns once; the call goes on.

## Reading it

```julia
DataFrame(calls())                          # every call, oldest first
DataFrame(calls(mood))                      # one function's
DataFrame(calls(mood; since = Date(2026, 9, 1)))
```

Columns: `id`, `started`, `name`, `module_name`, `version`, `model`, `seconds`, `inputs`, `outputs`, `error`, the token counts, `parent`, `caller`, `language`. Cost is tokens times price: `total_tokens - input_tokens` is what the model wrote, its hidden reasoning included.

## Ratings

```julia
p = predict(mood, "Arrived broken, but support was great.")
rate(p, :right)                                   # or :wrong, true, false
rate(p; answer = mixed, note = "broken item, good help")   # a correction is :wrong with the right answer
rate(p, nothing)                                  # withdraw your rating
rate.(predictions, :right)                        # many at once
```

`by` defaults to the caller's `user`, else your account's name; each person's latest rating of a call counts. `origin = :edit` marks a correction made while using the output (a renamed title), noisier than a review.

`rated(f)` turns the ratings into rows with known answers, for `evaluate` and the optimizers: the inputs, the right answer under the output's name (typed), then `call`, `version`, `rating`, `rated_by`, `origin`, `sample`, `disputed`. Calls rated under another signature, logged without content, or rated wrong without a correction are left out, and counted in a message. Ratings made in any FunctAI language count.

## Versions

```julia
version(mood)          # "sha256:…": what the function sends besides its inputs
signature_id(mood)     # its inputs' and outputs' names and shapes
```

A version changes with the instruction, the layout, the worked examples and code of its own; not with the model or the sampling settings (they're where a version runs, and each call records them). The same function written in any FunctAI language has the same version, so its calls and ratings pool. Calls with the same `signature_id` can share data even when the instruction changed.
