# Saved programs (format 1)

`save` writes a program to a folder you can read, diff and put in git;
`load` reads it back. This document says what the folder's manifest,
`functai.json`, holds, and which part of it any language can run.

**What travels between languages is an AI function.** Its signature,
instruction, worked examples and layout are data, so a function saved in
Python loads in TypeScript and sends the same bytes. A *program* also
holds the ordinary code between its AI calls (Python writes it to
`code/*.py`); that code runs only in the language that wrote it. A
loader refuses, before any call and with a reason, what it cannot run:
it never guesses.

`schema/saved.schema.json` checks a manifest's form.
`cases/saved/` holds manifests and what loading each must do.

## The folder

```
functai.json        the manifest (below)
code/<module>.py    Python: the code the program reaches, one file per module
files/              data files the program reads (functai.file(...))
models/<name>/      baked models (weights) the program runs on
requirements.txt    Python: the packages the code needs, pinned
requirements.lock   Python: those and everything they pull in
```

Only `functai.json` is read by every language.

## The manifest

| key | what it is |
|---|---|
| `functai_saved` | the format, `1`. |
| `language` | the language that wrote the folder and its code (`"python"`). Absent in folders written before 2026-09-27: read it as `"python"`. |
| `entry` | the key of the program `load` returns. |
| `created` | when it was saved (RFC 3339). |
| `nodes` | every piece of the program, by key `<module>:<name>`. |
| `modules`, `requirements`, `allowed`, `warnings`, `data_files`, `models`, `hashes` | the language's own: where the code is, what it needs, what `check` found, file hashes. Another language may ignore them. |

A node is `{"kind", "module", "name", ...}`. `kind` is `"ai"` (an AI
function), `"module"` (code that calls AI functions), `"function"` or
`"class"` (plain code). An `"ai"` node has `ai`:

| key | what it is |
|---|---|
| `signature` | the lmcc signature (plain-data form: `instructions` and `fields`), as the function sends it: its instructions are the current ones (improved, if it was). |
| `settings` | the settings that shape the request (`adapter`, `module`, `include_fn_name_in_instructions`, as [functions.md](functions.md) reads them), the model (`lm`) and the rest of the function's own settings. A setting whose value is not JSON is `{"baked": <name in models>}` (a baked model) or `{"node": <key>}` (another AI function). |
| `config` | lm15 settings the function set (`temperature`, `max_tokens`, …), in lm15's canonical JSON. |
| `template` | the function's own template (lmcc messages), or null. |
| `tools` | its tools: a node key (code in this folder), `{"import": "module:name"}`, or `{"tool": {name, description, parameters}}` (a tool described by data). |
| `state` | `instructions` (text or null) and `demos` (worked examples, as functions.md reads them). |
| `probes` | inputs the function was rendered with when saved: the sample input first ([calls.md](calls.md), *Versions*), then up to three demos' inputs, then examples given to `save`. |
| `fingerprints` | `signature`: lmcc's fingerprint of the signature (host type names included); `requests`: for each probe, `"sha256:"` of the canonical JSON of the request it rendered under the probe facts (or `"refused:<code>"`); `answers`: a baked model's answers to the probes. |
| `body` | `null` when the model writes the whole body (functions.md); `{"code": C}` when code of the function's own runs beside the model, `C` its source hash in the folder's `language`. Absent in folders written before 2026-09-27: read it as code. |
| `version` | the function's version when saved. Absent in folders written before 2026-09-27. |

## Loading an AI function in another language

A loader in a language other than the folder's `language`:

1. Reads `functai.json` and the node it is asked for (`entry` by
   default). Refuses `saved-malformed` when the manifest does not pass
   the schema, and `saved-format` when `functai_saved` is not a format
   it knows.
2. Refuses `saved-not-ai` when the node is not `kind: "ai"` (a module is
   code in the folder's language), and `saved-code` when its `body` is
   not `null`: code of its own runs beside the model, in a language
   this loader cannot run. The refusal names the node and the language.
3. Refuses `saved-tools` when the function has tools: a tool is code.
   (A later format may let a loader bind tools by name.)
4. Refuses `saved-model` when a setting is `{"baked": ...}` or
   `{"node": ...}`.
5. Builds the function from `signature`, `settings`, `config`,
   `template` and `state`, with the node's `name`, and its `module` as
   the call log's `program.module`.
6. **Checks it sends what was saved**: renders each probe under the probe
   facts and compares the hash with `fingerprints.requests` (the same
   `"refused:<code>"` counts as equal). Any difference refuses
   `saved-differs` and names the probe. When `version` is present, the
   loaded function's version must equal it.

The loaded function then behaves as it did where it was saved: the same
requests, the same version and signature in the call log, so its calls
and ratings pool with the saving language's.

A loader in the folder's own language runs the saved code (Python:
`functai.load(path, trust=True)`), as that language documents.
