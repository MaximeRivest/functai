# Changelog

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
