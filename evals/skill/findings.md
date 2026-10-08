# What the transcripts and the evaluation revealed (2026-10-06)

Sources: Pi sessions of `statlm` (01a0fe70, 2026-10-02 → 10-06),
`scholarsreadinglist` (01a0e736, 01a0f6b0, 01a107b5, 01a10989, 01a10c50,
01a10e86; 2026-09-28 → 10-06) and the species pilot run from `~`
(01a0fcef, 2026-10-02), plus the scripts in those projects. Every FunctAI
behaviour below was reproduced on functai 1.2.0 before being written down.

## FunctAI issues worth fixing

Status 2026-10-06 (unreleased, after 1.2.0): 1 to 5, 7 to 9 fixed (`python/CHANGELOG.md`,
*Unreleased*); 6 is the users' to change (`functai.bake.examples` is the public way). lmcc's
side was filed as issues and closed by lmcc 0.8.6 (2026-10-08): lm15 media parts as types and values
([lmcc#3](https://github.com/MaximeRivest/lmcc/issues/3)), rebuilding a host type from its
JSON form ([lmcc#4](https://github.com/MaximeRivest/lmcc/issues/4)), and a `parse-filtered`
refusal in the kernel ([lmcc#5](https://github.com/MaximeRivest/lmcc/issues/5)); FunctAI
uses all three since. Open: a list of pictures written as base64 text
([lmcc#7](https://github.com/MaximeRivest/lmcc/issues/7); FunctAI warns meanwhile).

1. **Guidance silently dropped from the prompt.** Comment harvesting skips
   comments on the choices of a multi-line `Literal[...]`, a comment after its
   closing `]`, `Enum` member comments, comments on fields of dataclasses
   *nested* in the result (`list[Analysis]`), dataclass docstrings and
   `Annotated[..., "text"]`. StatLM's `rows.py` has 65 field comments; 1 reaches
   the model (it survived only because the whole annotation guide is also an
   input). When the source is unavailable (`python -`, `python -c`, `exec`),
   *every* comment is dropped, with no warning, so a probe sends a different
   prompt than the script. Pydantic `Field(description=...)` works at every depth.
   Fixed: `functai/guidance.py` reads them with Python's parser and
   tokenizer; StatLM's schema now has 73 of its 80 comments in the prompt
   (the 7 others are section banners and group labels).
2. **A picture given as `{"media": "image"}` is sent as text.** The run
   succeeds, the model guesses (3/12 in the evaluation) and the agent reported
   success. The cause is item 8: FunctAI rewrites that annotation as a string.
   Honour it as lmcc does (an image input with no registration), and ship a
   type for pictures and files (design note 13).
3. **`api_retries` sleeps the provider's full `retry-after`, uncapped.** A
   subscription 429 said 11,270 s (about 3 h); every thread of a `map` would
   sleep that long. The species pilot set `api_retries=0` to avoid it; the
   rewriting scripts wrapped every call in their own retry loop. Fixed in all
   four languages: `api_max_wait` (60 s) raises at once beyond it, saying until when.
4. **A safety-filtered reply is reported as an unreadable reply.** Claude Opus
   stopped mid-answer on toxicology papers (finish `content_filter`); FunctAI
   raised a parse `Refusal`, which hid the cause. The re-ask then sent the
   empty reply back, which Claude rejects (`text content blocks must be
   non-empty`, HTTP 400). Fixed in all four languages: `parse-filtered`, not
   asked again; lmcc#5 proposes the code in the kernel.
5. **`compare`'s refusal could say why.** "compare needs the same examples in
   the same order; row 0 differs" cost 3 of 3 agents a debugging round in the
   first evaluation: they had passed the new rules as an extra input column.
   Saying "the input columns differ: rules" would make the fix obvious. Fixed
   in all four languages (R's `compare` did not check the rows at all).
6. **Training export was done through internals.** `sciencemadereadable/training/prepare.py`
   rebuilds prompts with `fn._variant_spec`, `engine.prepare_inputs`,
   `adapters.REGISTRY` and an AST rewrite to add an input; the 2026-10-02 bake
   rebuild broke it and pinned the project to a copy in `/tmp/fx/functai-train`.
   `functai.bake.examples(fn, rows, fixed=..., derived=...)` now does this job
   publicly; that project could move to it.

7. **The default OpenAI endpoint halves the speed.** lm15 routes `gpt-4.1-mini`
   to the Responses API; DSPy uses Chat Completions. Same FunctAI function,
   8 threads, 40 calls each, alternated twice: median 1.01 s and 1.05 s
   (`gpt-4.1-mini`) against 0.53 s and 0.57 s (`openai-chat:gpt-4.1-mini`).
   The benchmark's evaluation took 26 s with FunctAI and 15 s with DSPy, with
   both keeping 8 requests in flight 95% of the time: FunctAI adds no
   overhead. The root cause was not the endpoint: OpenAI's Responses API
   keeps every request for 30 days unless asked not to, and kept requests
   were slower (median 0.88–1.36 s against 0.69–0.91 s with `store: false`,
   four models). Fixed in all four languages: `store: false` by default.
8. **A JSON-Schema annotation is rewritten as text.** `picture: {"media":
   "image"}` becomes the shape `{"type": "string"}` in FunctAI's interface, so
   the picture is pasted as JSON text. lmcc's own frontend (`@lmcc.fn`) keeps
   `{"media": "image"}`, which the kernel sends as a real image part (§7b).
   Honouring it would give FunctAI an image input with no registration.
9. **Agents guess `functai.history`** (4 of 66 FunctAI sessions in the
   evaluation); it is `phistory` / `inspect_history`.

10. **A picture given by its path was answered from the cache after the file
    changed** (the key held the path, lm15 read the file only when sending).
    Fixed in all four languages: a media value given by a path is read when
    the call is bound (contract/programs.md).

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
