# Changelog

## 0.1.0 (unreleased)

Stage 1 of the contract (design/08-stage1-foundations.md):

- **`module(name, { description, input, output | outputs, uses }, run)`**
  (breaking; was `module(name, run, { uses })`): a module declares its
  interface, checked on every call (`InterfaceError`, logged); `run` gets
  its inputs by name and `{ signal, callId }`. Its version includes its
  interface; its code hash is now `sha256:`-prefixed (every module's
  version changed once). `.stream()`, `.using()`, `.interface`.
- **Interfaces**: every program has `.interface` and `.interfaceId` (the
  call log's `program.interface`); an AI function's is checked when it is
  defined; `checkInterface()`. An optional input's default is in its
  interface and left out of the signature (zod's `.default(x)` changes
  the version once). `t.withDefault`, `t.json`, `t.opaque`, and
  `{ shape, desc, optional, opaque }` for a field.
- **Call log format 2**: `program.interface`, `saw` (`[]`), `request_hash`,
  `described`, `journal`; `logContent` by field, only removing, over every
  layer (`SettingError` for a misspelled name); reading formats 1 and 2
  (`rated` by interface).
- **Stream events format 2**: `tree`, `writer`, `seq`, `after`, `at`, the
  `request` event (a field's text is its latest request's); a stream
  opened inside a tree shows the tree's numbers. Forms (`"kept"`), views,
  resuming (`after`, `read`), `Follower`, `replay`, `keptLog`.
- **Observers and journals** (`observers`, `journal` settings):
  `MemoryStore` (claims, appends, batches, reads), best-effort and
  required journals with barriers, `JournalError` (`journal-policy`,
  `journal-scope`, `journal-barrier`, `journal-end` with `outcome`,
  `event` and `settle()`).
- **Saved folders**: nodes carry their interface; loading takes optional
  inputs from it and checks it; `describeSaved()`; `save(module)` writes
  the module's node and the AI functions it uses. The manifest is checked
  against the contract's schema.
- `sawOf` and `keepsSaw` read what a call saw.
- An object argument is the inputs by name when every key is an input's or
  it holds the one required input's name (else, that input's value); input
  errors are `InterfaceError`s (a `TypeError`).

- The API follows TypeScript: `ai("mood", { description, input, output })`
  (the name first; `input`, as for `tool`); a call takes its inputs by
  name, typed, or a one-input function its value alone, and nothing else,
  so a missing, misspelled or mistyped input is a compile error.
  `tool("name", { description, input }, run)` types `run`'s input.
- A call's options: `fn(input, { signal, lm, temperature, ... })`, an
  `AbortSignal` that cancels it and settings for that call only.
- `fn.map(inputs, { concurrency })`: every answer, in order.
- Rows are typed in `evaluate`, `labeledFewShot`, `bootstrapFewShot` and
  `gepa`: a row missing an input, or an `expected` column the rows lack,
  is a compile error. The few-shot optimizers take `expected` too.
- `gepa` returns `{ fn, trials, calls, reflections }` (`trials(fn)` is gone).
- `calls()` gives typed `LoggedCall`s (camelCase; `record` is the line
  as written); `rated().leftOut` is camelCase.
- Any Standard Schema with JSON Schema (zod 4, valibot, arktype, ...) is
  a field, typed by its output type.
- Optional inputs: a field whose schema accepts a missing value may be
  left out: `t.optional(...)` and zod's `.optional()` are sent as null
  (the shape says so, so the version is Python's for `x: T | None = None`),
  `.default(x)` as `x`; `.nullable()` must still be given. Given values
  are checked (and parsed) by their Standard Schema before any call. With
  one required input, its value alone is the call; with none, no argument.
- The reply cache: `cacheReplies: true` (memory, the last 20,000) or any
  store with `get`/`set`/`delete`, given plain JSON; `clearCache()`. As
  Python's `cache_replies`, and like it now: an unreadable reply is not
  kept, and `gepa`'s teacher is never answered from it. A store that
  fails is skipped with one warning.
- `tests/types.ts` pins the types (`tsc` checks it; each
  `@ts-expect-error` must stay an error).

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
