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
| `inputs` | in order, each `{name, shape, desc?}`: `shape` a JSON Schema (below), `desc` words about it. |
| `outputs` | in order, each `{name, shape, desc?}`. The **last is the answer** (`program.answer` in the call log). One output is usually named `result`. |
| `settings` | the ones that shape the request: `adapter`, `template`, `module`, `include_fn_name_in_instructions`, `capabilities`, `tools`. |
| `state` | what improving changes: `instructions` (text that replaces the written instruction, or null) and `demos` (worked examples). |

**Shapes** are JSON Schema as lmcc reads it (kernel §1): `{"type":
"string"}`, `integer`, `number`, `boolean`; a choice `{"enum": [...],
"type": "string"}`; a list `{"type": "array", "items": S}`; a map
`{"type": "object", "additionalProperties": S}`; a record `{"type":
"object", "properties": {...}, "required": [...]}` (Python writes every
field of a dataclass as required, in declaration order); optional
`{"anyOf": [S, {"type": "null"}]}`. The same data type must have the same
shape in every language: a TypeScript record of a name and an age is the
shape a Python dataclass `Person(name: str, age: int)` has, with no
`title`, `$schema` or `additionalProperties: false` added.

## The signature

A definition becomes an lmcc signature (`instructions` and `fields`).

**Fields**, in this order:

1. the inputs, each `direction: "input"`, `purpose: "plain"`, its `desc`
   when it has one;
2. with tools, the input `tools` (purpose `tools`, type `list[Tool]`,
   lmcc_std's tool list shape);
3. with `module: "cot"`, the output `reasoning` (`{"type": "string"}`,
   purpose `reasoning`), unless an input or output is already
   named `reasoning`;
4. with tools, the output `calls` (purpose `tools.calls`, type
   `list[ToolCall]`);
5. the outputs, each `direction: "output"`, `purpose: "plain"`, **no
   `desc`** (lmcc would show a description instead of a value's format
   hint; output descriptions go into the instruction instead).

`type` is the host's name for the type (Python `str`, `Literal['a',
'b']`). It is not part of this contract except for the two tool fields,
where lmcc_std finds its formats by it: implementations compare
signatures with `type` left out, and a call's `program.signature` is
computed without it ([calls.md](calls.md)).

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
`state_window`, default 5); they are never part of a version.

## The request

For a call, the implementation binds the layout to the signature under
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

## When the reply cannot be read

- **An unreadable reply** (an lmcc refusal whose code starts with
  `parse-`, or `format-read-error`) is followed by up to `retries` more
  requests (default 1). Each sends the conversation so far plus the reply
  and a user message: `Your reply could not be read: <lmcc's hint>. Reply
  again, in exactly the form the instructions give.`
- **A cut-off reply** (`parse-truncated`) is sent again instead, with
  twice the token budget (`max_tokens`, 1024 when none was set), unless
  the provider takes no budget.
- **A value that does not fit its type** (a choice outside its list, a
  record missing a field) is an unreadable reply (`parse-value`).
- **A transient provider error** (rate limit, server error, time-out:
  lm15's retryable errors) is re-sent up to `api_retries` times (default
  3), after the provider's `retry_after`, else `min(30, 2^attempt)`
  seconds times a random factor in [0.5, 1.5).
- **Tools**: while the reply asks for tool calls, each runs and its
  result is added to the turn, up to `max_steps` requests (default 8);
  then `StepLimit`. A tool's error is reported to the model as `error:
  <type>: <message>` unless `tool_errors` is `"raise"`.

The re-ask and the error text are the only words FunctAI itself writes
into a conversation. lmcc's hints are prose, and may differ between
languages; nothing hashes them.
