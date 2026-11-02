# 15 — TypeScript 1.0: pictures, sounds, documents and files

*Status, 2026-10-31: built on the branch `ts-1.0`, every language green
(`./check`: the contract, Python 887, TypeScript 474, R 4,022, Julia
2,056, the cross-language check). TypeScript 1.0.0 is published once
lmcc 0.8.7 and attachments-dev are on npm (*Releasing*, below). Asked
for: "full image and all normal modality support" in TypeScript before
1.0, and TypeScript ready for Attachments' TypeScript package, which is
being written beside this (`attachments/ts`, npm `attachments-dev`).*

## What was wrong (measured before anything changed)

A probe of every kind of media input, through a fake router, in
TypeScript 0.1.0 on lmcc 0.8.6 (the npm version), then the same in
Python:

1. **Every picture function had one version.** The sample input (the one a
   version renders) had no value for a media shape: it was given
   `"example text"`, which lmcc refuses, so the version was
   `{"request": "refused:value-invalid"}` for every function with a
   picture, sound or document input, in all four languages, whatever its
   instruction, layout or worked examples. Their ratings pooled, and a
   saved folder's version said nothing.
2. **Lists and optional pictures were sent as text in TypeScript.** lmcc
   0.8.6 wrote `list[ImagePart]` and an optional picture through the
   layouts' `*` JSON format: the bytes as base64 inside the prompt, the
   call succeeding, the model guessing. lmcc 0.8.7 (D-64) fixed it on
   2026-10-09, and reached PyPI, never npm.
3. **lm15's own `image({ path })` was refused** in TypeScript (`'mediaType'
   is not a member of lm15's image part`): lm15-ts keeps its parts in
   camelCase, lmcc wants lm15's data. And there was no `t.image()`: a
   picture input was a raw shape typed `never`.
4. **Files were Python's alone, and broken across languages there too.**
   A Python function reading files (`Artifacts`), loaded from data (a
   folder "another language" wrote, a served interface), was not bound to
   Attachments: lmcc's Python binds by the Python type, and a saved
   signature names it only as text. Its sample sent `[]` as JSON and the
   load refused `saved-differs`. `Optional[Artifacts]` sent the artifacts as
   JSON text, pictures as base64. A failed file given in its wire form (a
   served call) was not refused. TypeScript, R and Julia would have sent
   such a function's artifacts as JSON text.
5. **Nothing compared pictures across languages**: no contract case had a
   media input, and `tools/crosslang.py` sent none.

## What the contract says now

`contract/functions.md`, *Pictures, sounds, documents and files*:

- A media shape `{"media": kind}` takes one part; its data is lm15's part
  data, logged with `type` (as Python logs an `ImagePart`). Lists of one
  kind and optional ones are parts too; a record holding a picture is
  refused (`no-format`). An implementation on lmcc before 0.8.7 must not
  run.
- An input whose `type` is `Artifacts` takes Attachments' `att()`
  results. Its shape is a list of objects; its data is Attachments' wire
  form; the model sees Attachments' `parts()` (pinned by Attachments' own
  contract), or `(no content)`. A failed file, and what Attachments'
  `from_wire` refuses, refuse the call at binding (`interface-input`). The
  type name is the third place a type name is part of the contract (after
  the two tool fields): it is what tells files from any list of objects.
- A language without Attachments refuses such a function
  `files-unavailable` (a new code), and never writes it as text.
- `calls.md`, *Versions*: a media shape's sample is a part by web address
  (`https://example.com/sample.png`, by kind).

Cases: `functions/19`–`23` (a picture, several, an optional one in the
json layout, a sound and a document in the chat layout, a video), each
with `sends`: what the function sends for real inputs (a picture's bytes,
two pictures, none, null). `files/` (`files.py`): a report and a picture
through Attachments, with what it sends for a text file, a folder with a
paged PDF, nothing, two failed files, and two values `from_wire`
refuses; and the same function as a saved folder. The case script asks
Attachments' Python for `parts()` only, as it asks lmcc for the layout.
R and Julia pass `files/` by refusing each function `files-unavailable`.

## TypeScript

```ts
import { ai, t, image } from "functai";
import { artifacts } from "functai/files";
import { att } from "attachments-dev";

const count = ai("count_items", { description: "How many items are on the shelf?",
  input: { photo: t.image() }, output: t.integer() });
await count({ photo: image({ path: "shelf.jpg" }) });

const summarize = ai("summarize", { description: "The report in three sentences.",
  input: { report: artifacts() }, output: t.string() });
await summarize({ report: await att("report.pdf[images: true]") });
```

Decisions, with the alternative an expert would raise:

- **`t.image()` takes lm15's parts, and functai re-exports lm15's
  `image`, `audio`, `video`, `document`, `binary`.** A user then needs no
  second install, and the value is the one lm15's README teaches. A bare
  path string (`photo: "shelf.jpg"`) is not accepted: it cannot be told
  from a web address or text, and a guess would be the kind of quiet
  substitution the contract forbids.
- **Files through a second entry point, `functai/files`,** rather than
  loading `attachments-dev` on first use. A dynamic `import("attachments-dev")`
  in the main entry makes Vite and webpack try to resolve a package a
  program may not use (a build error), or, hidden from them, never bundle
  it where it is used. A subpath is explicit, bundles exactly when used,
  and keeps the binding synchronous. It costs one import line, and a
  function saved elsewhere that reads files needs `import "functai/files"`
  before it loads (until then: `files-unavailable`, saying so).
  `package.json` lists the module under `sideEffects`, so a bare import is
  never tree-shaken away. attachments-dev is an optional peer dependency
  (`>=0.1.0 <2`: FunctAI uses its frozen wire form, `fromWire`, `parts()`
  and `errors`).
- **The name is `artifacts()`**, Python's type name (`Artifacts`), by the
  family's rule of one word per concept.
- **Files are a whole input** in every language: not optional, not in a
  list or a record, never an output or a tool's input (refused when
  defined). Inside another shape no format writes them as parts, so a
  layout would write them as text. No files is `att([])`.
- **Inside FunctAI an input of files is its wire form**, from binding on
  (the record, the cache key, conversations, served calls): one form,
  the same as Python's logs. The lmcc format reads the wire form back
  through Attachments only to write the parts.
- **A text value given to a picture input is lmcc's to refuse**
  (`value-invalid`, before any request), not `interface-input`: the
  contract says FunctAI's binding never reads the `media` keyword, and an
  `interface-input` here would make TypeScript differ from Python, R and
  Julia. Bytes in hand (`Uint8Array`) have no JSON form, are refused at
  binding, and the message says how to give a picture.
- TypeScript names a field's type only where the contract needs it
  (`Artifacts`); media fields stay untyped in the lmcc signature, as every
  other TypeScript field is. (Recorded turns replay across languages only
  under one fingerprint, which Python's `str` names already keep apart.)

## Trade-offs, stated

- **Every picture function gets a new version once**, in every language
  (cached replies are keyed by the request, and are not lost).
- **In TypeScript, Attachments reads only text-like files itself**; PDFs,
  Office files, pictures and audio need an Attachments service
  (`configure({ serviceUrl })` or a key). Without one, such a file is an
  artifact with an error, which the call refuses before sending anything.
  For a picture, `t.image()` needs no service.
- **Pictures are logged whole, twice** (input and request), as in Python;
  media by reference in logs (design/13) is still not done.
- **A PDF read through Attachments is text and page pictures**, not the
  original PDF; `t.document()` sends the PDF itself to providers that
  read PDFs.
- **The branch's `package-lock.json` is stale** until lmcc 0.8.7 and
  attachments-dev are on npm: `package.json` already asks for them.
- **Python's `Artifacts` shape lost its description** (it named a file
  Attachments has moved): a file function's signature id changes once.
  Files in Python were never released, so no released id changes.
- lm15 1.0.0's own types use `BufferSource` (a DOM type): a project
  type-checking its dependencies (`skipLibCheck: false`) without the DOM
  library sees one error inside lm15. FunctAI's and Attachments' types
  pass that check (an installed tarball, checked 2026-10-31).

Found on the way: an R test (`test-calls.R`, a function's own server)
never stopped the server it started, which kept `r/check`'s lock, so the
next `r/check` (and `./check`) waited forever. Fixed: the server's
process id is read before withr deletes its file.

## Releasing (what is left, in order)

1. lmcc 0.8.7 to npm, from a clean checkout of commit 5b62be4 (the
   working tree of `../lmcc` holds unrelated, uncommitted TypeScript
   changes): the worktree `/mnt/fast/functai/lmcc-0.8.7-release`, whose
   packed tarball passed lmcc's TypeScript tests and FunctAI's whole suite.
   `cd /mnt/fast/functai/lmcc-0.8.7-release/ts && npm login && npm publish
   --access public` (lmcc's RELEASING.md: the second factor needs a
   terminal and a browser).
2. attachments-dev to npm (its own release).
3. Here: `cd ts && npm install` (the lock then names both), `./check`,
   merge `ts-1.0`, `npm publish --access public` from `ts/`, tag
   `ts-v1.0.0`. Then the home page's table (`docs/index.md`) and
   design/01's status say 1.0.0.

Not in 1.0, still Python's: training and baked classifiers, tables other
than arrays of objects, programs with code of their own across languages,
`verify`/`check`/`runs`, `autoinstruct`, teacher-written training rows,
`optimization_runs()`, and the coding-agent skill (the README's *Not here
yet*). None changes what 1.0 does; each can come in a 1.x.
