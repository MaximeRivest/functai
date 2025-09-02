# Changelog

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
