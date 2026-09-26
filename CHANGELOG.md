# Changelog

## 1.0.1

- An AI function used as a metric (a judge) gets plain data: typed
  parameters (`row: dict, prediction: dict`) are written into its prompt as
  JSON. Before, the prediction arrived as a `Prediction` object and every
  row failed with "Prediction is not JSON data".
- AI functions on columns work in modules with `from __future__ import
  annotations`: the return type is resolved, not read as the text `'str'`.
- Documentation: a runnable tutorial (`docs/tutorial.md`), every example
  rewritten for 1.0 and rendered with real outputs, and `tests/docs_live.py`
  to check that the README's code runs and to re-render them.

## 1.0.0

FunctAI no longer depends on DSPy. It is built on lmcc (layout: how values are
written into prompts and read back) and lm15 (one wire for every provider).
The public API is kept: `@ai`, `_ai`, `configure`, `all=True`, `stateful`,
`tools`, `module="cot"`, `.opt`, `undo_opt`, `@module`, `phistory`, docments.

- Chat templates in the decorator: `@ai(template=[system(...), turns(), user(...)])`,
  with lmcc's template language. The reply pattern in a template is also its
  parser; a template with no pattern and one output reads the whole reply.
- Layouts by name: `adapter=None|"xml"` (tags), `"chat"` (DSPy's sections),
  `"json"` (provider-enforced JSON), or any `lmcc.Adapter`.
- `module="cot"` uses a model's own thinking channel where it has one, a
  written reasoning section otherwise.
- Tools run in a tool loop (native tool calls where the model has them, text
  calls otherwise) instead of switching the program to ReAct; `max_steps`,
  `tool_errors`, `StepLimit`.
- Memory is kept as lmcc turns (`fn.history`, `fn.reset()`).
- Own optimizers: `LabeledFewShot`, `BootstrapFewShot`,
  `BootstrapFewShotWithRandomSearch`, `InstructionSearch` (MIPRO-style).
  Optimizers change only instructions and demos; `fn.state()`,
  `fn.save()`/`fn.load()`, `@module` save/load.
- Data is rows, results are tables (no `Example` class). A dataset is a list of
  dicts or any table dpyr reads (parquet, CSV, pandas/polars, Hugging Face);
  columns named like the parameters are the inputs. `evaluate(program, data,
  metric)` returns an `Evaluation`: `.score` (0 to 1), `.summary` (per metric,
  with a 95% interval: Wilson for 0/1 scores, Student's t otherwise), `.table`
  (one row per example: data, `pred_*`, metrics, error, seconds, tokens,
  model, run). A metric is `metric(row, prediction)` or a dpyr expression;
  several go in a list or dict. Failed rows are null in the table and count 0
  in the score. `compare(before, after)` pairs examples; `log=` and
  `functai.runs(folder)` keep runs as parquet files; `fn.map(table)`.
  Tables need `pip install "functai[data]"` (dpyr); scores do not.
- AI functions on columns: `df.mutate(topic=classify(col.text))`, with any
  mix of columns and constants as arguments, in `mutate()` and `filter()`
  (dpyr's `vectorize`). One model call per distinct input, 8 at a time,
  remembered for the session; displays only run the shown rows; the column
  is pinned to the prompt in use when it was written. `fn.vectorize(threads=,
  errors=, dtype=)` for options; @modules too (their return annotation types
  the column).
- Settings resolve at call time with a thread-safe cascade (`using` >
  function > `with configure` block > `configure`); unknown settings raise;
  any lm15 `Config` field is a setting.
- Reliability: one re-ask after an unreadable reply (`retries=1`), backoff on
  transient provider errors (`api_retries=3`), an opt-in in-memory reply
  cache (`cache_replies=True`; off by default), misspelled layouts repaired and reported.
- Accounts: `functai.login("claude" | "chatgpt" | "copilot" | "grok" | "kimi" |
  "openrouter" | "<provider>", key=...)`, `logins()`, `logout()`, `LoginRequired`;
  model prefixes `claude:`, `chatgpt:`, `copilot:`, `kimi:`. Saved lm15 logins are
  used automatically (after an explicit `api_key`, before the environment);
  `configure(auth=False | path)`. Claude Code / Codex CLI logins are used in place.
  Subscription providers get their API's abilities (native tools, thinking).
- Settings a model refuses are left out with one warning: `temperature`/`top_p`
  for OpenAI reasoning models, those and `max_tokens` for the ChatGPT backend;
  no stop sequences for xAI.
- Model, connection and layout on the program: `fn.lm`, `fn.adapter`, `fn.template`
  setters and `fn.using(lm=, client=, adapter=, template=)`; an adapter replaces a
  template and back (a copy's `adapter=` was ignored when the function had a
  template); `None` in `using` means inherit. `client=` takes an lm15 router or one
  provider's LM (`OpenAILM(api_key=...)`, `ClaudeCodeLM(...)`); `lm=` takes a model
  name or an lm15 `BoundClient`. Bad layouts, templates, models and clients are
  refused where they are written (a DSPy adapter now at definition).
- Saving programs with their dependencies: `functai.check(program)` (the dependency
  graph and every problem with its fix), `save(program, folder, record=[...])` (code,
  settings, demos, data files, pinned requirements and a full lock; refuses while
  there are errors; all or nothing), `verify(folder, trust=True)` (a fresh uv
  environment from the lock alone: byte-identical rendered requests, and recorded
  recordings replayed to the same results, no model called), `load(folder, trust=True)`
  (hash, package and prompt checks first), `functai.file("data/x.txt")`,
  `@ai(requires=...)`, `@module(requires=...)`, `python -m functai verify <folder>`.
- Baking (`functai[bake]`): `fn.bake(rows, student=..., teacher=..., labels=...)` trains
  a head model (outputs with a fixed set of answers; calibrated probabilities) or
  a generative student (`method="sft"`, LoRA when needed) and reports accuracy with
  its interval, top-3, calibration, the confident-share curve, speed, the teacher's
  accuracy on the same rows and break-even. `fn.using(lm=baked)` runs the function on
  the weights (the model keeps its training layout; a changed function is refused;
  concurrent calls are batched). `escalate_to` / `escalate_below` send unsure answers
  to a bigger model; `prediction.confidence`, `.escalated`, `.first`. `baked.serve()`
  serves a generative student with vLLM. Saved programs carry their weights
  (`models/`), pin torch/transformers, and verify what the weights answer.
- Prime Intellect (`functai[prime]`): the `functai-verifiers` harness, the
  `functai-rows` taskset, `functai.bake.prime.env_package` and `.config`;
  `on_unreadable="record"`, `prediction.refusal`.
- A spent or forbidden key (HTTP 403) is no longer reported as a missing login.
- `@module` finds AI functions called under another name or through helper
  functions (it missed them before).
- Notebooks: a dataclass's field comments now reach the prompt (they were lost for
  classes defined in cells).
- Inspection: `phistory()`, `inspect_history()`, `fn.render()`, `fn.explain()`.
- Types: tuples, sets, TypedDicts, `Any`, `Annotated[T, "description"]`;
  JSON integers read into `float` fields become floats.
- Breaking: automatic instruction writing (`autoinstruct`) and refinement are
  now opt-in and run at the first call, not at definition; `to_dspy()` is
  removed; `fn.signature` is an lmcc signature; DSPy adapters, modules and
  optimizers are refused with a pointer to their replacement.
- Requires Python 3.11+ (lmcc's type registry fails on 3.10's generic aliases).
- Fixes: `_ai` declarations in functions defined inside other functions are
  found; a user `__post_init__` error in a `flexiclass` is no longer swallowed.

## 0.12.0

- Includes the Pydantic compatibility fix and structured output/input improvements introduced after 0.11.0.
- See 0.11.1 notes below for details.

## 0.11.1

- Fix: do not convert Pydantic `BaseModel` subclasses to dataclasses when building signatures; avoids corrupting Pydantic internals.
- Feature: coerce JSON/dict model outputs back into declared output types (Pydantic models, dataclasses, or lists thereof).
- Feature: accept Pydantic v1 inputs by using `.dict()` when `.model_dump()` is unavailable.

## 0.11.0

- Auto-instruction: generate a first-pass system instruction from function code and types when creating an AI function. Controlled by `autocompile`/`autoinstruct`; enabled by default.
- Instruction refinement: improve the instruction over the first N calls using recent inputs/outputs as noisy hints (not gold). Configure via `instruction_autorefine_calls` and `instruction_autorefine_max_examples`; disable by calling `.freeze()`.
- Teacher modes: support `teacher` and/or `teacher_lm` on `@ai(...)` and `.opt(...)` to synthesize examples with `n_synth` for optimization. Accepts teacher programs (`FunctAIFunc`) or bare LMs.
- Example ingestion: `.opt(...)` now accepts DSPy `Example`s, JSON-like dicts, or `(inputs, outputs)` pairs; auto-coerces into a trainset.
- Default metric: when an optimizer requires a `metric` and none is provided, fall back to a conservative exact-match metric over all output fields.
- Optimization logs: record `.opt(...)` runs, including example counts and whether synthesis was used; retrieve via `optimization_runs()`.
- Bespoke instruction override: runtime overrides are honored in signatures; docstring-derived appendix is skipped when an override is set.
- History helper: new `phistory(n=1)` to quickly print `dspy.inspect_history()` for recent calls.
- Docs and examples: README/specs updated to use `phistory`; added `examples/typing_and_extraction` showing type-directed extraction patterns.

## 0.10.0

- Type-directed outputs: preserve function return annotations and variable annotations (including typing generics like `list[int]`, `dict[str, int]`), and pass them intact to DSPy.
- Clear precedence: when returning a variable from `_ai`, prefer that variable's annotation; otherwise inherit the function's return annotation; mismatch raises a helpful error with actionable fixes.
- Sentinel returns: `return _ai` / `return ...` uses the function's return annotation for the primary `result` output; auxiliary `_ai` variables become extra outputs.
- Extras typing: unannotated extra outputs default to `str`; annotated ones are preserved.
- Instruction text: keep output and field docs but avoid naming base types in instructions; only include class/dataclass field docs when meaningful.
- Schema friendliness: auto-convert plain classes-with-annotations to dataclasses and ensure nested types are schema-friendly.

## 0.8.1

- Adapters: accept duck-typed adapters and classes; allow per-call `@ai(adapter=...)` by scoping DSPy settings during invocation.
- _ai returns: resolve bare `_ai` placeholders inside tuples/lists/dicts (e.g., `return (id, email)`) to concrete values using return-order mapping.

## 0.8.0

- Add `flexiclass` to convert classes with annotations into dataclasses in place.
- Introduce `UNSET` sentinel; defaults use schema-safe `None` with post-init flip to `UNSET`.
- Harvest inline comments (docments) for:
  - Function parameters and return annotations
  - `_ai` output declarations (including comments after `_ai`)
  - Class/dataclass fields, with robust source fallbacks
- Append harvested guidance to signature instructions:
  - Parameter guidance, Output guidance, Return guidance
  - Qualified field guidance (e.g., `Account.id: User ID`)
- Auto-convert return/extra output classes (and nested types) to dataclasses to satisfy Pydantic schema generation.
- Normalize inputs: accept Pydantic BaseModel and dataclass instances as structured inputs (converted to dicts).
