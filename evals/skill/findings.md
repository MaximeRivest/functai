# What the transcripts and the evaluation revealed (2026-10-06)

Sources: Pi sessions of `statlm` (01a0fe70, 2026-10-02 → 10-06),
`scholarsreadinglist` (01a0e736, 01a0f6b0, 01a107b5, 01a10989, 01a10c50,
01a10e86; 2026-09-28 → 10-06) and the species pilot run from `~`
(01a0fcef, 2026-10-02), plus the scripts in those projects. Every FunctAI
behaviour below was reproduced on functai 1.2.0 before being written down.

## FunctAI issues worth fixing

1. **Guidance silently dropped from the prompt.** Comment harvesting skips
   comments on the choices of a multi-line `Literal[...]`, a comment after its
   closing `]`, `Enum` member comments, comments on fields of dataclasses
   *nested* in the result (`list[Analysis]`), dataclass docstrings and
   `Annotated[..., "text"]`. StatLM's `rows.py` has 65 field comments; 1 reaches
   the model (it survived only because the whole annotation guide is also an
   input). When the source is unavailable (`python -`, `python -c`, `exec`),
   *every* comment is dropped, with no warning, so a probe sends a different
   prompt than the script. Pydantic `Field(description=...)` works at every depth.
   Suggest: harvest nested and member comments like top-level ones, and warn
   once when an AI function's comments cannot be read.
2. **A non-type annotation is accepted.** `def f(image: {"media": "image"})`
   runs and sends the image as base64 *JSON text*; the model guesses (3/12 in
   the evaluation) and the agent reported success. Suggest: refuse
   annotations that are not types, and ship an image input type (the species
   pilot and the evaluation both had to register one with `lmcc.format`).
3. **`api_retries` sleeps the provider's full `retry-after`, uncapped.** A
   subscription 429 said 11,270 s (about 3 h); every thread of a `map` would
   sleep that long. The species pilot set `api_retries=0` to avoid it; the
   rewriting scripts wrapped every call in their own retry loop. Suggest: a cap
   (`max_wait=`), and a refusal that says "your plan's window resets at …".
4. **A safety-filtered reply is reported as an unreadable reply.** Claude Opus
   stopped mid-answer on toxicology papers (finish `content_filter`); FunctAI
   raised a parse `Refusal`, which hid the cause. The re-ask then sent the
   empty reply back, which Claude rejects (`text content blocks must be
   non-empty`, HTTP 400). Suggest: name the filter in the error and never re-send
   an empty turn.
5. **`compare`'s refusal could say why.** "compare needs the same examples in
   the same order; row 0 differs" cost 3 of 3 agents a debugging round in the
   first evaluation: they had passed the new rules as an extra input column.
   Saying "the input columns differ: rules" would make the fix obvious.
6. **Training export was done through internals.** `sciencemadereadable/training/prepare.py`
   rebuilds prompts with `fn._variant_spec`, `engine.prepare_inputs`,
   `adapters.REGISTRY` and an AST rewrite to add an input; the 2026-10-02 bake
   rebuild broke it and pinned the project to a copy in `/tmp/fx/functai-train`.
   `functai.bake.examples(fn, rows, fixed=..., derived=...)` now does this job
   publicly; that project could move to it.

## On this machine

- `/tmp/lmcc` is a symlink to `~/Projects/lmcc` (created 2026-09-28). A script
  saved directly in `/tmp` imports it instead of the library, giving
  `module 'lmcc' has no attribute 'default_registry'`. Every occurrence on
  record (StatLM and Scholar's Reading List, 2026-09-28 → 10-02) came from a
  script in `/tmp`; agents blamed version drift. The link was removed on 2026-10-06.

## Patterns that worked (kept in the skill)

- StatLM's scripts: typed forms, `cache_replies="disk"`, AI functions as dpyr
  columns (`mutate`, `unpack`, `unnest`), readable top-to-bottom files.
- The species pilot's `fn.using(...).map(rows, threads=8)` per model, with a
  per-run `cache_replies` and `log_calls` folder.
- Two annotators plus a merging function, failures recorded as data.

## Patterns that cost time (warned against in the skill)

- Hand-rolled thread pools, retry loops and JSON caches around calls (translator.py,
  pilot50.py, annotate.py; 2 of 3 "without" agents in the forms task).
- Probes run from heredocs and `exec`, and scripts in `/tmp`.
- `sleep N; tail log` polling loops (dozens of turns in the rewriting sessions).
- Own quote matchers (`statlm/evaluate.py`, `score.py`) where `functai.quotes_found` exists.
