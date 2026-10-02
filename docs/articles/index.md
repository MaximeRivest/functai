---
rat:
  python:
    dependencies: ["-e .[data]", "pandas"]
---

# Learn

## Start from what you have

[A table of text](../get-started.md)
: Label, sort or score every row, check it against answers you trust, make it better. **Start here if unsure.**

[Turn notes into data](notes-to-data.md)
: Field notes, reports, emails: several facts per note, your protocol, a score per field.

[From a prompt you already have](from-a-prompt.md)
: Your OpenAI messages, sent exactly as they are; then types, tables and measurement.

[Coming from another tool](coming-from.md)
: The OpenAI SDK, DSPy, pandas and polars, Instructor and Pydantic AI, side by side.

## Is it good?

[Is it right?](accuracy.md)
: A score with its range, every answer in a table, fair comparisons, how many rows to label.

[Make it better](improving.md)
: From free to expensive: rules, types, examples, optimizers, bigger models.

[Make it cheaper](cheaper.md)
: Several models on your data, tokens into money, where tokens go.

[Big tables](tables.md)
: Any data frame in, AI functions as columns, pandas or polars out.

## Ship

[Ship it: save, verify, load](saving.md)
: A program and everything it depends on, in a folder, proven to run elsewhere.

[Serve it over HTTP](serving.md)
: A program as a web service; a served program used from Python like a local one.

[Bake it into a small model](baking.md)
: Train a small model to answer a function; send the unsure cases to a big one.

[Every call, on record](call-log.md)
: Keep each call, mark answers right or wrong, and learn from the corrections.

## Toolbox

[How a function becomes a prompt](anatomy.md)
: The mapping from Python to prompt, and what happens on each call.

[Types](types.md)
: Lists, records, choices, maybe-missing values; as answers and as inputs.

[Reasoning and several answers](outputs.md)
: Think first, return several values, get everything with `predict`.

[Watch it being written](streaming.md)
: The answer as the model writes it; what an outside caller may see; every event, kept as it happens.

[Tools](tools.md)
: Let the model call your Python functions; a person approves the ones that change things.

[Memory](memory.md)
: Conversations: turns that remember each other, branches, stopping, helpers that remember.

[Plugins](plugins.md)
: Change what programs do without changing them, every change on record.

[Multi-step programs](modules.md)
: Plain Python calling several AI functions, measured and saved as one.

[Prompt formats and chat templates](layouts.md)
: How values are written into the prompt, and writing the conversation yourself.

[Models and settings](models.md)
: Any provider, the default, and where settings come from.

[Signing in](logins.md)
: API keys and subscriptions (Claude, ChatGPT, Copilot, xAI).

## When things go wrong

[Inspecting calls](inspection.md)
: Exactly what was sent and what came back.

[When the model gets it wrong](reliability.md)
: Repairs, retries, provider errors, the reply cache.

[Upgrading](upgrading.md)
: From 1.1 to the next release, and from 0.x to 1.0.
