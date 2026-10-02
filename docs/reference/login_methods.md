---
rat:
  project: ../../python
  python:
    dependencies: ["-e .[data]", "pandas"]
---

# login_methods { #functai.login_methods }

```{.python .no-run}
login_methods(provider)
```

The ways you can sign in to a provider, and how far each is proven.

## Parameters {.doc-section .doc-section-parameters}

| Name     | Type   | Description                                                                                                                      | Default    |
|----------|--------|----------------------------------------------------------------------------------------------------------------------------------|------------|
| provider | str    | A provider or subscription: ``"openai"``, ``"anthropic"``, ``"claude"`` (a Claude subscription), ``"chatgpt"``, ``"copilot"``... | _required_ |

## Returns {.doc-section .doc-section-returns}

| Name   | Type         | Description                                                                                                                                                                                                                                                                                                                                                                                                                                            |
|--------|--------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|        | list of dict | One per method: ``method`` (what ``login(provider, method=...)`` takes), ``kind`` (``api_key`` or ``account``), ``status`` and, when there is one, a ``note``. ``status`` is ``supported`` (the code exists and was seen working), ``unverified`` (the code exists, but no recorded working run does: only used when you ask for it by name) or ``unavailable`` (it cannot run here; the note says why). It never says whether your account qualifies. |

## See Also {.doc-section .doc-section-see-also}

- [`login`](login.md): sign in.
- [`logins`](logins.md): what you can use right now.

## Examples {.doc-section .doc-section-examples}

```python
import functai
from functai import *
```

```python
functai.login_methods("chatgpt")
```

```output
[{'method': 'browser', 'kind': 'account', 'status': 'unverified', 'note': 'Browser login, inference, persistence and early renewal observed 2026-09-23; provider permission and billing remain unverified'}, {'method': 'device', 'kind': 'account', 'status': 'unverified', 'note': 'Device login and inference observed 2026-09-23; provider permission and billing remain unverified'}, {'method': 'external:codex-cli', 'kind': 'account', 'status': 'supported'}]
```