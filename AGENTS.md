# FunctAI: working in this repository

FunctAI is typed functions whose body is a language model call, in several
languages that agree through one contract. Python exists; TypeScript is
next, then Julia and R. The plan is `design/01-many-languages.md`; read it
before adding a language or changing the contract.

## The map

| Path | What it is | Who it answers to |
|---|---|---|
| `contract/` | formats, JSON Schemas and cases every implementation must pass | nothing: it is the authority |
| `python/` | the Python package (`functai` on PyPI), self-contained: pyproject, uv.lock, .venv, tests, examples, README (PyPI's page), CHANGELOG | `contract/` |
| `ts/` | the TypeScript package (`functai` on npm, not published yet): src, tests, tools, README, CHANGELOG. `src/generated/contract.ts` is the contract's data, written by `node tools/generate.ts` | `contract/` |
| `docs/` | the website: MRMD notebooks with their outputs, each pinned to `python/` as its rat project | the code they show |
| `tools/docs.py` | generates reference/example/news pages, runs notebooks on rat, builds the site | |
| `tools/crosslang.py` | Python and TypeScript against each other: saved in one, loaded in the other; one call log | `contract/` |
| `design/` | numbered design notes | |
| `check` | the one command: the contract, then every implementation | |

Rules:

- A language folder never imports from another. They meet in `contract/`.
- When code and contract disagree, the code is wrong. To change a rule,
  change the contract first (a new format number when the meaning of
  existing data changes), then every implementation, in the same commit.
- Contract cases are written by scripts from the rules (`contract/cases/make.py`),
  never copied from an implementation's output. Data every language must
  reproduce (layouts, the model table, case folding) lives in the contract
  as JSON; each implementation carries a copy and a test that it is equal.
- Each language keeps its own version, changelog and release tag prefix
  (`python-v1.1.0`, `ts-v…`, `r-v…`, `julia-v…`).

## Commands

```bash
./check                                         # everything; green before any commit
cd python && uv sync --all-extras --all-groups  # the Python environment (python/.venv)
cd python && .venv/bin/python -m pytest -q      # Python tests only (offline, a fake provider)
python/.venv/bin/python tools/docs.py generate  # reference, examples, news pages
python/.venv/bin/python tools/docs.py run [PAGE ...]   # run notebooks (needs model keys; costs cents)
python/.venv/bin/python tools/docs.py site      # build the website into site/
cd ts && npm install && npm test                # TypeScript (needs ../lmcc checked out: lmcc is not on npm yet)
cd ts && node tools/generate.ts                 # after changing contract/layouts, models.json or unicode/
python/.venv/bin/python contract/cases/make.py  # after changing a rule: rewrite the cases
```

Live runs (real models, costs cents): `cd python && .venv/bin/python
tests/live.py`; `cd ts && node --conditions=functai-source
--conditions=lmcc-source tools/live.ts`.

Model keys for live runs: `set -a; source ~/Projects/lm15-dev/.env; set +a`
(never commit it).

## Releasing Python

Bump `python/pyproject.toml` and `python/CHANGELOG.md`, run `./check`,
push, then publish a GitHub Release tagged `python-v<version>`.
`.github/workflows/release.yml` builds `python/` and publishes to PyPI by
Trusted Publishing (bound to that file name and the `pypi` environment:
keep both). A bare `v<version>` tag is refused.
