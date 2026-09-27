# Changelog

## 0.1.0 (unreleased)

The first TypeScript implementation of FunctAI, held to the same contract
as the Python package (`../contract`): every function, score, rating and
saved-folder case, and a check against Python itself (`../tools/crosslang.py`).

- `ai({ name, description, inputs, output | outputs, ...settings })`: a
  typed function whose body a model writes. Shapes with `t`, zod 4 or JSON
  Schema. The layouts `xml` (default), `chat` and `json`, templates,
  `module: "cot"`, tools.
- The same function has the same version and signature as in Python.
- Unreadable replies are asked again (the contract's words); values are
  checked against their shapes; transient provider errors are re-sent.
- Streaming (`fn.stream`): the answer's text as it arrives, and every
  event of `contract/streaming.md`.
- The call log, `rate`, `rated`, `calls`: the same folder and records as
  Python.
- `evaluate`, `exactMatch`, `interval`: the same scores and ranges as Python.
- `labeledFewShot`, `bootstrapFewShot`.
- `load`/`fromManifest`: run an AI function Python saved; `save`/`toManifest`.
- `module(name, fn, { uses })`: code that calls AI functions, logged as one
  call with theirs as children.
