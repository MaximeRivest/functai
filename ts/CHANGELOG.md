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
- Journals and observers never hold a call up nor reach it:
  - A journal setting takes `timeout` (ms, default 30,000: the longest an
    append is waited for, its `signal` aborted then, and the longest a
    barrier waits) and `backoff` (ms, default 50, doubling up to 2 s
    between resends). A store that throws (even before returning a
    promise), never answers, or answers anything but `"kept"`,
    `"duplicate"` or a refusal is no answer or a refusal: it can no longer
    spin the process, hang a call or crash it. A barrier waits for its own
    event (not for other calls' later events), and stops when the call is
    cancelled (start and tools; the end waits for its timeout). Each
    append gets fresh copies of its events. A failing journal is warned
    about once per outage, not once per process.
  - Observers are called off the call's turn, each with its own copy of
    every event, from a queue of at most 10,000 events per observer (then
    it loses events and sees the gap); an object with `postMessage` (a
    `Worker`, a `MessagePort`) is an observer. `flush()` waits for
    observers and journals. (Before, observers ran inside the call and
    could change what the journal and the stream held.)
  - `JournalError` has no type parameter; `outcome.done` is what the
    caller would have got: the answer from `fn(x)` and a stream's result,
    the `Prediction` from `fn.predict(x)` and `stream.prediction`.
  - `observers` and `journal` settings that could only fail later refuse
    where they are set (`TypeError`), as `logContent` does; a call's
    options are checked too, and a loaded function's own `logContent`
    (`SettingError`, `log-content-field`).
- Records and events keep only a program's fields: a value given to a
  module under another name is refused and never kept (not even under a
  host's `"*": false` allowlist); inputs are recorded as given when the
  call starts (not a schema's transform, nor what the code later does to
  them). A module's argument that cannot be bound is a recorded refusal.
  Interface errors name the field and the kind of value, never the value.
- An AI function's record holds every output the model gave, `calls`
  (the tool calls of its last step, `[]` once it answers) included, as
  Python's does. A re-ask's exchange keeps the `request_hash` of the
  rendered request it came from, as Python's does.
- Closing a stream (or aborting its signal) while a module's code runs
  ends the call `Cancelled`, whatever the code returns after; a finished
  stream lets go of its caller's signal, and so does a retry's wait.
- Declarations: a field that is not a shape refuses `interface-malformed`
  naming it; an output cannot be `optional`, nor an AI function's
  output opaque. One output, whatever its name, is the value, in the
  types too (`outputs: { count: t.integer() }` returns a number).
  `t.string({ default })` (and `integer`, `number`, `boolean`) is typed as
  an input that may be left out; `t.json`, `t.list`, `t.object` and
  `t.record` take extra keys as the other builders do (their words were
  dropped before).
- Names JavaScript treats specially are data: a JSON member `__proto__`,
  a required property `toString`, a field named `constructor`; an unknown
  shape keyword named `constructor` refuses. This now holds for AI
  functions too: binding, preparing and sampling their inputs, reading
  rows and worked examples, and checking a model's object answers look at
  own properties only. Before, a required input named `toString` or
  `constructor` that was left out was sent to the model as JavaScript
  function text, and one named `__proto__` lost its value. A Standard
  Schema's JSON Schema keeps a property named `__proto__`. With lmcc's
  D-58 (below), every name works as in Python, and the same requests are
  sent, saved and loaded both ways: an input or an output named
  `__proto__`; a JSON input whose value holds a member named `__proto__`
  (before, the member was dropped); a worked example that leaves out an
  input or output named like an Object member (before, the call failed);
  and a reply that leaves out an output named like an Object member
  (`toString`, `valueOf`, …) is asked again and refused
  `parse-missing-fields`, by every adapter (before, it was read as `""`,
  and the `json` adapter refused a correct reply `parse-ambiguous`).
  `bootstrapFewShot` keeps an input named `__proto__`, and a `__proto__`
  member of an output, in the worked examples it records (before, both
  were lost, in the saved folder too). `gepa`'s default feedback reads
  rows by own members.
- **Members keep their order**, as in Python: a value's members, even
  names like `"10"` that JavaScript lists first, keep the value's order in
  what a call sends (a text input given an object included), its record
  and its events (observers, journals, `MemoryStore`), the rows `rated`
  reads back, the worked examples labeled and bootstrapped, and a saved
  folder, written and read (`save`, `load`). Values are copied, read and
  written with lmcc's helpers, never `structuredClone`, `JSON.parse` or
  `JSON.stringify`. JSON lm15 writes (a `response_format` schema, a
  tool's parameters, a `config`) follows lm15: JavaScript's order.
- FunctAI gives lm15 plain data: a request, its `Config`, a saved
  `config` and a cached reply carry no record of member order (lm15
  refuses an object with a symbol key). Before, a `json`-adapter function
  whose output shape had an integer-like property after another (from
  `lmcc.parseJson`, or a saved folder, Python's included), or a tool with
  such parameters, threw a `TypeError` at every render and call; a cached
  reply read back that way was skipped with a warning.
- The JSON FunctAI writes (a text input given an object, call log lines,
  saved folders) visits every array index again, a hole written `null`:
  before, `[, "B"]` was written `[,"B"]` (a call log line and a saved
  folder no reader took), and `["A", ,]` and `new Array(2)` lost
  elements. A boxed primitive is written as its value (`new Number(42)`
  was `{}`; a `String` object given to a text input is its text), and
  recorded as its value in the call log; a value that holds itself is
  refused with `JSON.stringify`'s `TypeError` (was a stack overflow).
- An integer past 2^53 in an answer, or anywhere in a value, is recorded
  as its digits (before, as a description; a JSON answer holding one was
  recorded as `{"$type": "Object", ...}`). An array's hole, or
  `undefined`, in a recorded value is `null` (an `undefined` made the
  whole value a description).
- **functai needs lmcc with decision D-58** (commit `3492090` or later):
  it refuses to start on an lmcc without its helpers (lmcc 0.8.4 as
  published on npm). The workarounds for the older kernel are gone: values
  are ordinary objects again, and nothing is refused for its names.
- A function loaded from a saved folder keeps its fields' type names as
  saved (`dict`, `str`): its signature's fingerprint is the saved one, so
  a worked example recorded by another language (a Python bootstrap) is
  replayed as the model wrote it. Before, it was written again from its
  values, another request than the saving language's, and a folder whose
  recorded reply was spelled otherwise refused `saved-differs`.
- A call whose code rejects with no reason (`Promise.reject()`, `throw
  undefined`) fails as any other: its `failed` event is kept and shown, and
  its record says it failed. Before, it made no terminal event and no
  record, and its caller got an unrelated `TypeError`. A tool that throws
  `undefined` is reported to the model as an error.
- An AI function records its inputs as given even when a schema changes
  them in place (a nested object edited by a transform): they are written
  as JSON before the schema runs.
- A schema whose validation returns another realm's promise (a `vm`
  context, an iframe) is awaited.
- Observers each have their own queue and share of the delivery time: a
  slow observer falls behind and loses events alone, never the others.
- A journal writer whose round of resends gave up tries again later on
  its own (1, 2, 4, … 60 s apart, for as long as the process lives) and at
  the next event. It gives up on a log only for memory: when all writers
  hold more than 100,000 unconfirmed events, the one holding the oldest
  gives up its log (warned once per outage), and lets go at once: its
  append under way is aborted and no longer waited for (even with
  `timeout: Infinity`), and no resend follows. `flush()` sends what is not
  confirmed once more (again after a round that was under way when it
  began), and returns `false` while a journal still holds events it did
  not confirm, or once after a writer gave up on events. (Before, it said
  `true` while a tree's end was lost for good.)
- A journal's `timeout` and `backoff` above 2,147,483,647 ms (a timer's
  limit) refuse where they are set; `timeout: Infinity` still waits for
  ever. `JournalError.settle({ signal })` can be stopped, also by an abort
  the store's own read makes.
- A builder's extra keys never replace what it makes (`t.list(x, { items
  })` is a `TypeError`), and a builder's `default` is typed as its value.
  `ModuleResult` of outputs built at run time (`Record<string, …>`) is a
  record, not one value.
- `replay()` and `Follower.recover()` stop at an event of a format they do
  not know (`stopped`); `recover()` from a readable source follows again;
  `Follower.forget(tree)`.
- Validation started for a call's inputs never leaves a rejection
  unhandled.
- `npm test` has a time limit per test.
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
