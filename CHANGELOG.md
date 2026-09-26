# Changelog

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
- Own evaluation and optimizers: `Example`, `Prediction`, `evaluate`/`Evaluate`,
  `LabeledFewShot`, `BootstrapFewShot`, `BootstrapFewShotWithRandomSearch`,
  `InstructionSearch` (MIPRO-style). Optimizers change only instructions and
  demos; `fn.state()`, `fn.save()`/`fn.load()`, `@module` save/load.
- Settings resolve at call time with a thread-safe cascade (`using` >
  function > `with configure` block > `configure`); unknown settings raise;
  any lm15 `Config` field is a setting.
- Reliability: one re-ask after an unreadable reply (`retries=1`), backoff on
  transient provider errors (`api_retries=3`), an in-memory reply cache
  (`cache_replies=True`), misspelled layouts repaired and reported.
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
