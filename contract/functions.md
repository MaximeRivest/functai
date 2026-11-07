# AI functions (format 1)

What an AI function sends, in every language. Two implementations that
follow this document send the same bytes for the same function, so a
function has one version and one signature wherever it runs, and a
function saved in one language runs unchanged in another
([saved.md](saved.md)).

The layers underneath are not repeated here. **lmcc** (kernel 0.8)
turns a signature, a layout and the values into a request and reads the
reply back; its corpus makes that byte-exact in every language. **lm15**
sends the request. This document fixes what FunctAI decides on top of
them: the signature a definition becomes, the layout, the facts about the
model, the worked examples, and how a call behaves when a reply cannot be
read.

`cases/functions/*.json` pin these rules: each case is a definition and
the signature, sample input, request, version and signature id it must
give.

## A definition

However a language writes an AI function (Python: a typed function and
its docstring; TypeScript: `ai({...})`), it comes down to this data:

| part | what it is |
|---|---|
| `name` | the function's name. |
| `description` | what it does, in words (Python: the docstring). May be empty. |
| `inputs` | in order, each `{name, shape, desc?, optional?, default_code?}`: `shape` a JSON Schema (below), `desc` words about it, `optional` true when a caller may leave it out; an optional input's shape has a `default`, the value it is sent with then ([programs.md](programs.md)); `default_code`, when that default is written as an expression, is its text as the language normalizes it (it counts in the version instead of the value: [calls.md](calls.md), *Versions*, *Defaults*). |
| `outputs` | in order, each `{name, shape, desc?}`. The **last is the answer** (`program.answer` in the call log). One output is usually named `result`. |
| `settings` | the ones that shape the request: `adapter`, `template`, `module`, `include_fn_name_in_instructions`, `capabilities`, `tools`. |
| `state` | what improving changes: `instructions` (text that replaces the written instruction, or null) and `demos` (worked examples). |

**A choice's answers, described one by one.** Where a language lets each
answer of a choice say what it means (Python: a comment after each choice
of a `Literal[...]` written over several lines, or after each member of an
`Enum`; R: `choice(approve = "...")`; TypeScript: `t.choice(["approve",
"deny"], { approve: "..." })`; Julia: `OneOf("approve" => "...")`), those
words are part of
the field's `desc`, written one way in every language, so the same choice
described the same way sends the same bytes wherever it is written:

- the field's own words, or `one of:` when it has none;
- then, for each described answer, in the shape's order, a line
  `  - <answer>: <meaning>` (two spaces, a dash, a space);
- joined by `\n`.

A meaning is trimmed of white space at both ends, and each run of white
space inside it that holds a line break (what Python's `str.splitlines()`
splits on) becomes one space. White space is what [scores.md](scores.md)
says it is. An answer whose
meaning is empty is not described, and an answer left undescribed is not
listed (the shape lists every answer). A choice with no described answer
leaves the field's words as they are. (Until 2026-10-08 R wrote `<words>.
<answer>: <meaning>; ...` on one line: a semicolon or a colon inside a
meaning made that ambiguous, and six long meanings were one wall of text.)

A definition is refused when lmcc refuses its signature
(`signature-malformed`), and then when its interface (its inputs and
outputs) breaks the rules every interface keeps ([programs.md](programs.md),
*Interfaces that are refused*: `interface-malformed`), so that a function
one language defines and saves, every language can load.

**Shapes** are JSON Schema as lmcc reads it (kernel §1): `{"type":
"string"}`, `integer`, `number`, `boolean`; a choice `{"enum": [...],
"type": "string"}`; a list `{"type": "array", "items": S}`; a map
`{"type": "object", "additionalProperties": S}`; a record `{"type":
"object", "properties": {...}, "required": [...]}` (Python writes every
field of a dataclass as required, in declaration order); optional
`{"anyOf": [S, {"type": "null"}]}`. The same data type must have the same
shape in every language: a TypeScript record of a name and an age is the
shape a Python dataclass `Person(name: str, age: int)` has, with no
`title`, `$schema` or `additionalProperties: false` added. A record is
closed all the same ([programs.md](programs.md), *Checking values*): the
shape says it by having `properties` and no `additionalProperties`.

## The signature

A definition becomes an lmcc signature (`instructions` and `fields`).

**Fields**, in this order:

1. the inputs, each `direction: "input"`, `purpose: "plain"`, its `desc`
   when it has one, its shape without its `default` (a default is how a
   call is bound before the request, not what the model is told: an
   input left out is sent with its default, like any given value);
2. with tools, the input `tools` (purpose `tools`, type `list[Tool]`,
   lmcc_std's tool list shape);
3. with `module: "cot"`, the output `reasoning` (`{"type": "string"}`,
   purpose `reasoning`), unless an input or output is already
   named `reasoning`;
4. with tools, the output `calls` (purpose `tools.calls`, type
   `list[ToolCall]`); a call's value for it is every tool call the model
   asked for during the call, in the order asked, across every step (not
   the last step's list, which is empty when the model answered);
5. the outputs, each `direction: "output"`, `purpose: "plain"`, **no
   `desc`** (lmcc would show a description instead of a value's format
   hint; output descriptions go into the instruction instead).

`type` is the host's name for the type (Python `str`, `Literal['a',
'b']`). It is not part of this contract except for the two tool fields,
where lmcc_std finds its formats by it, and an input of files, whose
`type` is `Artifacts` (*Pictures, sounds, documents and files*, below):
implementations compare signatures with `type` left out, and a call's
`program.signature` is computed without it ([calls.md](calls.md)).

**Instructions.** When `state.instructions` is not null, they are that
text with white space trimmed at both ends, and nothing else. Otherwise
they are built from three parts:

- the head: `Function: <name>` when `include_fn_name_in_instructions`
  (the default, true), then the description trimmed, joined by a blank
  line (`\n\n`), leaving out an empty part;
- the guidance, lines joined by `\n`:
  - when an input has a desc: `Parameter guidance:`, then `- <name>:
    <desc>` for each such input, then an empty line;
  - when an output has a desc: `Output guidance:`, then `- <name>:
    <desc>` for each such output, in output order, then an empty line;
  - the whole trimmed at both ends;
- the instructions: the head, then (when both are not empty) `\n\n`,
  then the guidance.

White space means what Python's `str.strip()` removes: the characters
listed under *Scores* in [scores.md](scores.md).

## The layout

The layout is an lmcc adapter. `settings.adapter`:

| value | layout |
|---|---|
| absent, `"xml"`, `"default"`, `"tags"` | [`layouts/xml.json`](layouts/xml.json): each value in tags, the reply pattern in the system message |
| `"chat"` | [`layouts/chat.json`](layouts/chat.json): DSPy's `[[ ## name ## ]]` sections |
| `"json"` | [`layouts/json.json`](layouts/json.json): one JSON object the provider enforces |
| an lmcc artifact (JSON) | that artifact, loaded by lmcc |

Names are matched ignoring case, `-` and spaces. The three files are
lmcc artifacts; an implementation builds or loads the same artifact
(equal as JSON, apart from `versions.kernel`). They need lmcc's standard
pack (`lmcc_std`: the `json` format, the reasoning and tool transports,
the `json_object` reader).

**Reasoning.** The `reasoning` output travels by what the model can do.
With `native_reasoning`, the request asks for the model's own thinking
and its summary (`config.reasoning` `{"effort": "medium", "summary":
"auto"}`) and the output reads the thinking parts (`part:thinking`):
without `summary`, OpenAI's reasoning models, Gemini, Claude Opus 4.7,
4.8 and every Claude 5 think and send none of it, so the output would
be empty (lm15 MAP-7 rule 7; on Claude, lm15 2.2 asks with
`thinking.display: "summarized"`). Where the provider's account cannot
have a summary (OpenAI may ask an organization to be verified first), the
request fails and says so; `capabilities: {"native_reasoning": false}`
has the model write its reasoning instead. **The function's own
`reasoning` setting** (lm15 `Config.reasoning`) chooses the effort and
the thinking budget, and the summary when it names one: the layout's
`medium` and `auto` apply only where it says nothing (so `reasoning =
{"effort": "low"}` sends `{"effort": "low", "summary": "auto"}`, never a
conflict between the two). Effort `off` sends no summary (lm15 refuses
one with it) and the model's thinking, if any, is not asked for. Without `native_reasoning`, the
model is told to write it in its own section, before the others
(`prefix_cot`): Claude 5 models sometimes refuse that (`refusal`, read as
`parse-filtered`), which is why they are in `reasoning_prefixes`. This
transport is spelled out in the layouts rather than named, so its
settings are FunctAI's (a later `lmcc_std` does not change them).

**A template** (`settings.template`, a list of lmcc messages and
directives) is its own layout: formats `{"*": {"use": "json"}}`, the
transports of `layouts/xml.json`, reader `derived`. Without a `turns`
directive, one is inserted right before the last user message (at the
end when there is none). When lmcc refuses to derive a reader from it
(`not-readable` with fix path `template`) and the function has one
visible output, the reader is `functai_reply` (version 1.0.0): the
whole reply, trimmed by lmcc's text rules, is that output's value, and a
reply of several outputs is joined by `\n`. With several outputs it
refuses `not-readable`.

**Judgment-only providers** (`models.json`, `judgment_only`) get, when
no adapter or template is set, a layout of `turns`, then one user message
holding the single input's value (`{<name>}`) or, with several inputs,
the input tags of `xml`; reader `json_object`. Each output's question is
its desc, else the description (with several outputs, followed by
` (<name>)`).

## Capabilities

What a model can do is declared, never guessed:
[`models.json`](models.json) holds the table, by the provider lm15
routes the model to (`provider:model` or a name lm15 knows).

- `judgment_only` providers: `native_structured_output` only.
- `native` providers: `instruct`, `native_function_calling` and
  `native_structured_output`; `stop_sequences` unless the provider is in
  `no_stop_sequences`; `native_reasoning` when the model name starts with
  one of the provider's `reasoning_prefixes`; `assistant_prefill` when
  the provider is in `assistant_prefill` and `native_reasoning` is false.
- `speaks_as` providers: the facts of the provider they speak as.
- `chat_completions` providers: `instruct`, `native_function_calling`,
  `native_structured_output`; `stop_sequences` and `native_reasoning`
  false.
- any other provider: `instruct`; `native_function_calling` when it is a
  `native_tool_hosts` provider; `stop_sequences` false.
- An Anthropic model (`anthropic`, or a provider that speaks as it) with
  `native_reasoning` and a `temperature` set to anything but 1 gets
  `native_reasoning` false and `assistant_prefill` true: its thinking
  runs only at temperature 1.
- Then the function's `capabilities` setting replaces any fact it names.

**Sampling a model does not take.** A model in `fixed_sampling` (by the
provider it is routed to, or the one that provider speaks as, and a
model-name prefix) runs only at `temperature` 1 and `top_p` 1. A call
whose settings give either another value leaves it out of the request
(the model then samples as at 1), before the capabilities above are
worked out, and warns once per process and provider that it did. So one
`temperature = 0` set for a session works across models, and says where
it could not apply.

**Requests the provider would keep.** A provider in `no_storage` (by the
provider the call is routed to) keeps every request and its reply on its
side unless the request says `store: false` (OpenAI's Responses API: 30
days). A call whose settings give no `store` sends `store: false` there,
silently; a `store` the settings give (`True` or `False`) is sent as given.
So a program's data stays where its log says it is, by default. The
setting is part of the request, so it is part of the reply cache's key
([replies.md](replies.md)).

A version is computed under the fixed facts of `probe`, with provider
`"probe"` (a judgment-only layout never applies there).

## Worked examples

`state.demos` are worked examples, each `{"inputs": {...}, "outputs":
{...}}` of JSON values (or a recorded lmcc turn, which is used as is when
its signature is the plan's and read as its inputs and outputs
otherwise). Each becomes an lmcc example turn, in order, placed where the
layout's `turns` directive is:

- only inputs the signature has as plain inputs, and outputs it has as
  plain or reasoning outputs, are kept;
- a demo left with no output is skipped, and so is one lmcc refuses to
  make an example of;
- a value given to a text input that is not text is written as text:
  objects and lists as JSON indented by two spaces, anything else as the
  host writes it.

A stateful function's earlier turns follow the demos (the last
`state_window`, default 5); they are never part of a version, and the
calls they were are the call's `saw` ([calls.md](calls.md), *Saw*).

A function's interface (what a caller gives and gets) is its
definition's inputs and outputs ([programs.md](programs.md)).

## The request

For a call, the implementation first binds the call's inputs to their
fields ([programs.md](programs.md), *Binding a call's inputs*: a refusal
there sends nothing), then binds the layout to the signature under
the model's capabilities (every refusal fires here, before anything is
sent), makes the turn from the values (tools: the tool list under
`tools`), renders it after the worked examples, and hands the result to
lm15 with the model and its settings.

**The probe.** The request a version names ([calls.md](calls.md),
*Versions*) is rendered the same way, under the `probe` facts, with no
conversation memory, for the **sample input**: each plain input's
value from its shape, as calls.md says. lmcc's `request("probe")` gives
the request as JSON; its hash is `"sha256:"` + SHA-256 of its canonical
JSON.

## Pictures, sounds, documents and files

**Media.** An input whose shape is lmcc's media shape `{"media": kind}`,
`kind` one of `image`, `audio`, `video`, `document`, `binary` (kernel
§7b), takes one part of that kind, and the model is sent that part: a
picture as a picture, never as text. Its value, as data, is lm15's part
data: `media_type` and one of `data` (the bytes in standard base64),
`url`, `file_id` or `path`, and `detail` for a picture; `"type"` (the
kind) may be given too. Python's `ImagePart`, TypeScript's lm15 parts
(`image({ path })`, whose members are camelCase), R's and Julia's lm15
bridges are that data in the language's own form, and are logged as it,
with `type`. A picture given by `path` is read when the call is bound
([programs.md](programs.md), *Binding a call's inputs*).

lmcc (kernel 0.8.7, D-64) also writes, as parts:

- a list of one kind (`{"type": "array", "items": {"media": "image"}}`):
  each item's part at the input's place, in order; `[]` writes nothing;
- an optional one (`{"anyOf": [{"media": "image"}, {"type": "null"}]}`):
  `null` is written as the text `null`.

A record holding a picture (`{"properties": {"photo": {"media":
"image"}}}`) is refused when the layout is bound (`no-format`): no
layout writes it as parts, and none may write it as text. An
implementation on an lmcc before 0.8.7 would send such values as base64
text the model cannot see, and must not run.

**Files.** An input whose `type` is `Artifacts` takes the result of
Attachments' `att()` (a PDF, a Word or Excel file, slides, a web page,
a folder, a repository, a picture; Attachments' own contract,
`attachments/contract/`). Its shape is `{"type": "array", "items":
{"type": "object"}}`, and its value, as data, is Attachments' wire form
of the artifacts (`to_wire`: each `{text, images: [{name, mimetype,
bytes_b64, page?}], audio, video, meta}`). The type name is what tells a
list of artifacts from any other list of objects, so it is kept where a
saved node or a served interface names its inputs' types.

- **What the model sees** at the input's place: Attachments' `parts()`
  of the artifacts, as Attachments' contract writes them
  (`presenting.md`: each artifact's text, file names shown, each page's
  pictures after that page's text) — text parts and picture parts, in
  that order. When `parts()` gives none (`[]`: no artifacts), one text
  part `(no content)`.
- **A file that failed** (an artifact with `meta.error`: `att()` never
  raises) refuses the call when it is bound, `interface-input`, naming
  the input and each failed source with its error code, before any
  request: an empty document would reach the model otherwise. A value
  Attachments' `from_wire` refuses (not a list, an artifact whose `text`
  is not text: Attachments' contract, `cases/wire.json`) is refused the
  same way; members it fills in (a missing `meta`) are its to fill.
- **A language that cannot read files** (no Attachments for it, or the
  package not installed) refuses such a function `files-unavailable`,
  when it is defined or loaded, or at the latest when a call or its
  version is rendered, and never sends anything. It never writes the
  artifacts as JSON text instead, which is what a layout's `*` format
  would do without the binding.

## When the reply cannot be read

- **An unreadable reply** (an lmcc refusal whose code starts with
  `parse-`, or `format-read-error`, other than the stopped and interrupted
  ones below) is followed by up to `retries` more
  requests (default 1). Each sends the conversation so far plus the reply
  and a user message: `Your reply could not be read: <lmcc's hint>. Reply
  again, in exactly the form the instructions give.`
- **A stopped reply** (lmcc's `parse-filtered`, kernel 0.8.6 §4a: the
  response's finish reason `content_filter`, a provider's safety filter or
  the model declining, or a `refusal` part, a model declining in its own
  words) is never an answer, even when it reads, and is **not asked
  again**: the same request would be stopped the same way, and the re-ask
  would send the stopped reply back (an empty one is a request Claude
  refuses). The call raises lmcc's refusal as it is, its hint lmcc's.
  (Until 2026-10-06 it was asked again as "a reply that could not be
  read"; from then to lmcc 0.8.6, FunctAI checked the finish reason
  itself, with a hint of its own.)
- **An interrupted reply** (lmcc's `parse-interrupted`, kernel 0.8.6 §4a:
  the response ended in error, finish reason `error`, before an output
  ended) is **sent again as it was**: the same request, with no
  correction and no larger budget, up to `retries` more times, since the
  model did nothing wrong and the remedy is to ask again. It counts as a
  re-ask, and is an exchange like one. When the re-asks are spent the call
  raises lmcc's refusal. (Before, it would have been re-asked with "Your
  reply could not be read", sending the broken half reply back.)
- **A cut-off reply** (`parse-truncated`) is sent again instead, with
  twice the token budget, only when the request set one (`max_tokens`,
  the caller's or a previous re-send's) and the provider takes one. When
  none was set, the reply already had the most the call allows (lm15's
  default for the model, recorded on the response as a `config.max_tokens`
  `defaulted` adaptation, or the provider's own maximum): it is not sent
  again, since no larger budget can be given and a guessed number would
  shrink it. (Until 2026-09-30 the guess was 1024: the re-send asked for
  2048 after a reply that had 128000.)
- **The cut-off refusal says what happened.** Its hint is lmcc's with
  lmcc's closing advice (`; raise max_tokens or ask for less`) removed,
  then, in order:
  - `; the model spent <T> of its <O> output tokens thinking` when the
    response's `usage.reasoning_tokens` `<T>` is above 0 (`; the model
    spent <T> tokens thinking` when `usage.output_tokens` `<O>` is absent
    or smaller);
  - the budget and the advice that fits it, one of:
    `; raise max_tokens (it was <N>) or ask for less` (the request set
    `<N>`), `; no max_tokens was set, and lm15 sent <N>, the most it knows
    this model to allow: lower the reasoning effort or ask for less` (the
    `defaulted` record's `applied`), or `; no max_tokens was set, so the
    provider used its own maximum: lower the reasoning effort or ask for
    less`;
  - ` (lm15 adapted the request: <field> <action>: <reason>; ...)` for the
    response's other adaptations, in order, when there are any. A dropped
    thinking budget is the usual one: without it a caller cannot tell why
    `thinking_budget` did not bound the thinking.

  The name is the language's (`maxTokens` in TypeScript); the rest is the
  same text in every language. lm15's adaptations of every exchange are
  also in the call log, on the recorded response.
- **A value that does not fit its type** (a choice outside its list, a
  record missing a field or holding one it does not name) is an
  unreadable reply (`parse-value`).
- **Keeping an unreadable reply** (`on_unreadable`: `"raise"`, the
  default, or `"record"`). With `"record"`, a reply whose refusal's code
  starts with `parse-` (`parse-filtered` included) that would end the
  call, once its re-asks are spent, ends it well instead, with no values:
  the call returns no answer (Python `None`, TypeScript `undefined`, R
  `NA`, Julia `missing`), what it returns says why (its refusal), its
  record has `outputs: {}`, no `returned`, and the refusal as its `error`
  ([calls.md](calls.md)), and in a conversation its turn is `done` with no
  outputs, so the turns after it go on. It is for runs that must not stop
  at one bad reply (rollouts, long batches). Any other failure (a
  provider error, `format-read-error`, a step limit) still raises.
- **Each request is its own exchange**: a re-ask, a resend with a larger
  budget and a tool step are each recorded as an exchange, with the
  `request_hash` of the lmcc request it sent ([calls.md](calls.md)).
- **A transient provider error** (rate limit, server error, time-out:
  lm15's retryable errors) is re-sent up to `api_retries` times (default
  3), after the provider's `retry_after`, else `min(30, 2^attempt)`
  seconds times a random factor in [0.5, 1.5). A `retry_after` longer
  than `api_max_wait` seconds (default 60; a number of at least 0) is not
  waited: the error is raised at once, saying how long the provider asked
  to wait, until when (UTC), and the setting that would wait for it (a
  note on the error, or its message where a language's errors take no
  note). A subscription's spent window asks for hours, and a run that
  sleeps that long without a word looks hung; a per-minute limit asks
  for less than a minute, and is waited.
- **Tools**: while the reply asks for tool calls, each runs and its
  result (text, or pictures and other parts: [tools.md](tools.md), *What
  a tool returns*) is added to the turn, up to `max_steps` requests
  (default 8); then `StepLimit`. Before each request after the first, the
  `step` hook may change what the next request offers and shows
  ([plugins.md](plugins.md), *The step hook*); the `limits` plugin ends
  the loop with an answer instead of `StepLimit`. A tool's error is reported to the model as `error:
  <type>: <message>` unless `tool_errors` is `"raise"`.

The re-ask and the error text are the only words FunctAI itself writes
into a conversation. lmcc's hints are prose, and may differ between
languages; nothing hashes them.
