# Changelog

## 0.1.0 (unreleased)

- `gepa(fn, rows, { teacher })`: the instruction rewritten from the
  function's mistakes, as Python's `GEPA` and R's `gepa()` do
  (`design/04-gepa.md`); `trials(fn)` gives the search.
- `withSettings({ caller })` adds to the enclosing block's caller instead
  of replacing it, so an evaluation inside an optimization is logged as
  both.

The first TypeScript implementation of FunctAI, held to the same contract
as the Python package (`../contract`): every function, score, rating and
saved-folder case, and a check against Python itself (`../tools/crosslang.py`).

- A temperature or top_p a model does not take (GPT-6, the o-series and
  GPT-5, Claude 5) is left out of its requests, with one warning, as in
  Python (`contract/models.json`, `fixed_sampling`).

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
- The call log's `program.file` and `.line` are the line that called `ai()`,
  found by who called rather than by file path: right when bundled (with
  `--enable-source-maps`, the original file), on Windows paths, `file:`
  URLs with spaces, and in a project whose own folder is named `functai`.
- `labeledFewShot`, `bootstrapFewShot`.
- `load`/`fromManifest`: run an AI function Python saved; `save`/`toManifest`.
- `module(name, fn, { uses })`: code that calls AI functions, logged as one
  call with theirs as children.
