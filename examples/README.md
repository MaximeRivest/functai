# Examples

Each folder is one topic, written as a [Quarto](https://quarto.org)
document (`main.qmd`) and rendered with real model replies into the
folder's `README.md`. New to FunctAI? Start with the
[tutorial](../docs/tutorial.md).

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

To run one yourself, open `main.qmd` in any notebook tool that reads
Quarto, or copy its cells. To re-render all of them (and check that the
README's code runs), with a model key in the environment:

```bash
QUARTO_PYTHON=/path/to/python-with-functai[data]-and-ipykernel \
    python tests/docs_live.py --render
```
