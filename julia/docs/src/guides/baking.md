# Baking: training data, and students trained elsewhere

A baked model is weights trained to answer one or more AI functions. The contract fixes what every language agrees on: what a generative student is trained on, and how it is called (contract/baked.md). Julia writes the training examples every trainer reads, byte for byte the ones Python writes, and calls a baked student wherever its weights are served. Training itself happens in Python's `functai.bake` (on this machine's GPUs, on Tinker, on Prime Intellect) or in any trainer that reads the exported examples.

```@setup baking
using FunctAI
```

## The examples

```@example baking
@ai function reply(message::String, guide::String)::String
    "Answer the customer."
end
rows = [(message = "Where is my order?", guide = "Be kind.", result = "On its way."),
        (message = "Can I return it?", guide = "Be kind.", result = "Yes, within 30 days.")]
table = FunctAI.bake_examples(reply, rows; fixed = Dict("guide" => "Be kind."))
table[1].messages
```

One row per conversation: `function`, `messages` (the prompt, then the reply the student learns: the loss belongs on the reply only), `tag`, `weight`, `row_id`, `split` (`"train"` or `"validation"`) and `source`.

- **What the student reads** is the function's signature without the inputs the bake leaves out, without `reasoning` unless the bake keeps it (`reasoning = true`), in the function's layout with replies written from values, and no worked examples: a student is trained and called without them.
- **`fixed`**: an input with one value in every row is left out of every example and every call; its hash is kept, and a call that gives it another value is refused (`baked-fixed`).
- **`derived`**: an input another input decides (`derived = Dict("guidance" => "section")`) is left out too; a table of hashes is kept, and a call with a pair the student never saw is refused (`baked-derived`). Rows that break either are refused before anything is written ([`FunctAI.BakeError`](@ref) `bake-rows`).
- Turning messages into tokens is the student's own chat template's rule, with thinking off.

```julia
FunctAI.export_examples("reply.jsonl", reply, rows; fixed = Dict("guide" => "Be kind."))
```

writes them as JSON lines with `reply.jsonl.meta.json` beside it (`{"functai_examples": 1, …}`: what made them), for TRL's `SFTTrainer` (`assistant_only_loss`), Axolotl's chat-template datasets, Unsloth, or a service's upload.

## A student trained elsewhere

```julia
student = FunctAI.baked("baked/reply"; url = "http://localhost:8000/v1")   # `vllm serve baked/reply/model`
fast = configure(reply; lm = student)
fast("Where is my order?", "Be kind.")
```

[`FunctAI.baked`](@ref) reads a baked folder's `baked.json` (format 2, written by Python's `functai.bake` or any trainer that writes it) and calls its weights through an OpenAI-compatible server (vLLM, SGLang, TGI, llama.cpp) serving them: laid out exactly as the student was trained (its signature and layout, no worked examples, its chat template with thinking off), so the tokens it is called with are the tokens it learned. A call is refused when the function changed since it was baked (`baked-changed`: its inputs, outputs or their types), and as above for fixed and derived inputs. A student also answers when another model is unsure: `escalate_to = student`, or the other way round.

What Julia does not do yet: train (a head or a student), plan a bake, or check a folder's file hashes. A model is trained by Python's `bake`, a service, or any trainer from the exported examples; the design (design/01-many-languages.md, *Training in every language*) is for Julia to train in its own ecosystem (Lux, CUDA.jl) once it passes the same gate: a held-out score inside the range the bake reported.
