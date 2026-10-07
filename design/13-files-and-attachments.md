# 13 — Files as inputs: FunctAI × Attachments × lmcc

*Status, 2026-10-06: proposal; since then FunctAI honours `{"media": "image"}`
annotations (a picture input with no registration, 12 of 12 in the evaluation's
pictures), and lmcc's side is filed as issues
[#3](https://github.com/MaximeRivest/lmcc/issues/3) (lm15 media parts as types
and values) and [#4](https://github.com/MaximeRivest/lmcc/issues/4) (a host
type's JSON form, both ways). From a review of the three code bases and a
working prototype (`~/.cache/functai-skill-eval/att/`). Nothing built in the
repositories yet. Motivated by the FunctAI vs DSPy benchmark
(`evals/skill/README.md`): pictures were DSPy's one clear edge (a native
`dspy.Image`; FunctAI needed an `lmcc.format` registration, 27 lines against
15). DSPy reads images only; with Attachments, a FunctAI function would take
any file: PDF, Word, Excel, slides, web pages, folders, repositories, audio.*

## What each layer is, today

| layer | owns | media today |
|---|---|---|
| **lm15** | the wire to every provider | parts: `ImagePart`, `DocumentPart` (native PDF), `AudioPart`, `VideoPart`, with `lm15.image(path=...)` |
| **lmcc** | how a value is written into a prompt and read back | shape `{"media": kind}` takes an lm15 part's data (kernel §7b); a format may write several parts at its hole; Python maps a type to a shape only through `lmcc.format(Type, ...)`; no built-in media type |
| **FunctAI** | typed functions, calls, logs, cache, replay, evaluation, baking, four languages | every input must have a JSON form (log, cache key, replay); no media type; logs store values inline |
| **Attachments** | turning any source into `{text, images, audio, video, meta}` (the frozen Artifact IR, `spec/IR-CONTRACT.md`) | images as raw `bytes` in process, `bytes_b64` on the wire (server only) |

Measured gaps (functai 1.2.0, lmcc 0.8.5, attachments 1.0.0a1):

1. **No layer accepts a picture as a typed input.** `picture: lm15.ImagePart`
   is refused by lmcc (`unmapped-type` on its `path` field). Users write a
   dict subclass plus `lmcc.format`; agents without the skill sent base64 as
   prompt text (the model guessed: 3 of 12) or hid the image below FunctAI in
   a custom client (invisible to the log and the cache key).
2. **An Attachments result has no JSON form in process.** FunctAI refuses it
   (`has no JSON form, and does not bind`). The wire form exists
   (`bytes_b64`) but only `server.py` writes it, privately.
3. **Media is logged inline, twice.** A call record holds the input and the
   request, each with the image as base64: a 789-byte PNG made a 5.9 KB
   record. A 2-page PDF at Attachments' default 200 dpi is 880 KB of PNG,
   about 2.4 MB per record and again in the reply cache.
4. **lmcc's `lift` does not rebuild a list subclass** from JSON
   (`model_validate` is tried only for dict and str data), so a replayed or
   loaded call would hand the format plain dicts, not `Artifacts`.

## Where things go

**Attachments** keeps what it is: the converters and the Artifact. It gains
no dependency on lmcc or FunctAI. What it adds is public and neutral (see
its review, `attachments/docs/review-2026-10-06-functai.md`): the wire form
as a public API both ways, a provider-neutral list of content parts (text
and image parts, optionally interleaved by page), errors that can raise, and
size control on rendered pages.

**lmcc** gains nothing Attachments-specific. Two small, general fixes:
`lift` rebuilds any type that offers a `from_json`-style constructor for list
data too, and Python maps `lm15.ImagePart` (and the other media parts) to
`{"media": kind}` by default, so a plain picture needs no registration in any
FunctAI language.

**FunctAI** owns the bridge (decided 2026-10-06), in Python first:

- `functai.files` (loaded only when `attachments` is importable): registers
  `attachments.Artifacts` with lmcc as a structured input whose shape is
  Attachments' published wire schema (`spec/artifact.schema.json`, an array
  of artifacts) and whose format writes parts: each artifact's text (with its
  file name only when asked: `sources=True`), then its images, interleaved by
  page when the artifact has page segments. Tested in the prototype: the
  `xml`, `json` and `chat` layouts all render text + image parts, and the
  reply cache works (second call 0.0 s).
- **Refuse error artifacts before any request.** `att()` never raises; a
  missing file is an artifact with `meta.error` and empty text, which would
  reach the model as an empty document. The bridge raises `InterfaceError`
  naming the source and the error code.
- **Media by reference in logs and caches.** Store each image once, by its
  SHA-256, beside the log (`calls/media/<sha256>`), and write
  `{"$media": mimetype, "sha256": ...}` in records, the form the contract's
  `saw` cases already use. The request hash still covers the bytes, so the
  cache key is unchanged. This is a contract change (calls.md, replies.md),
  so it comes with cases and all four languages, or it waits.
- **A plain picture**: `functai.Image` is not needed once lmcc maps
  `lm15.ImagePart`; `att("photo.png")` covers the rest.
- **Other languages**: the input's shape is the Artifact wire schema, which
  is language-neutral; TypeScript, R and Julia accept the JSON the
  Attachments server returns, with the same parts writer, when demand pulls.
- **Baking**: files are not trained yet (`bake` refuses them); a text-only
  artifact is ordinary text and could be.

Not decided, needs Maxime:

- **File names shown to the model.** Attachments' `.text` starts every file
  with `## <name>` and notes images as `[image: <name>]`: useful for a folder
  of documents, a leak for classification (`tabby_cat.png`). Proposal: off
  by default in the bridge, `sources=True` to show them.
- **Native PDFs.** Providers read PDFs directly (lm15 `DocumentPart`). That
  can be cheaper and better than text + page images, but the Artifact keeps
  no original bytes. Later, not for 1.0.
- **A file name as input** (`summarize("report.pdf[pages: 1-4]")`): the
  cache key must be the content, not the name. After the rest works.

## Order of work

1. Attachments: the public wire form, `parts()`, `errors="raise"`, page
   image size control; the `service_url` blocker; release 1.0.
2. lmcc: `lift` for list types; default mapping of lm15 media parts.
3. FunctAI Python: `functai.files`, refusal of error artifacts, docs page,
   skill (`recipes.md` §Images becomes three lines).
4. FunctAI contract: media by reference in logs and the cache, then every
   language.
5. Benchmark: a `files` task (PDF and spreadsheet questions; DSPy has no
   native way), the photos task again, FunctAI vs DSPy rerun.
