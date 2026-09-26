# Examples

Each folder is one topic. Its `README.md` is a notebook (Markdown with
Python cells, each followed by its real output): read it here, or open it
in Chattering and run it. New to FunctAI? Start with the
[Get started](https://maximerivest.github.io/functai/get-started.html).

| example | what it shows |
|---|---|
| [typing_and_extraction](typing_and_extraction/) | types in and out: lists, dicts, `Enum`, `Literal`, dataclasses, pydantic models, several outputs, post-processing |
| [docments_flexiclass](docments_flexiclass/) | comments are prompts: on parameters, the return line, class fields and `_ai` outputs; plain classes as types |
| [claide_code](claide_code/) | a terminal assistant: a shell tool with an allow-list, and memory (`stateful=True`) |
| [local_simple_rag_agent](local_simple_rag_agent/) | an agent that reads the web, a fact checker in a `@module`, and the same agent on a local model |
| [graph_rag](graph_rag/) | a knowledge graph built chunk by chunk with pydantic models, drawn and queried |
| [modules](modules/) | a multi-hop fact checker as one `@module`: evaluated, optimized, run on a table |
| [optimizing_translator](optimizing_translator/) | an AI judge as the metric, `InstructionSearch`, and `compare` before/after |
| [tracking_and_osb](tracking_and_osb/) | observability: `phistory`, `inspect_history`, token usage, logged evaluation runs, the reply cache |

To run one yourself: open its `README.md` in Chattering and press Run
all, or, with a model key in the environment:

```bash
python tools/docs.py run examples/modules/README.md
```
