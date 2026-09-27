---
rat:
  project: ../../python
  python:
    dependencies: ["-e .[data]", "pandas"]
---

# Coming from another tool

*What you already know, in functai: the OpenAI SDK, DSPy, pandas and polars, Instructor and Pydantic AI.*

```python
import functai
functai.configure(lm="gpt-4.1-mini", temperature=0)   # the model behind every output on this page
from functai import ai, _ai
```

## The OpenAI SDK (or Anthropic's, or any chat API)

Your messages work as they are: [From a prompt you already have](from-a-prompt.md)
shows the same request going out, then the steps from there.

## DSPy

functai grew from the same idea (the signature is the prompt, and
programs are optimized from data), with plain Python functions as the
signature.

| DSPy | functai |
|---|---|
| `class QA(dspy.Signature)` with `InputField`/`OutputField` | a function: parameters are inputs, the return type and `_ai` variables are outputs |
| field `desc=` | a comment on the parameter or the line |
| `dspy.Predict(QA)` | `@ai` |
| `dspy.ChainOfThought(QA)` | `@ai(module="cot")`, or a `reasoning: str = _ai` line |
| `dspy.ReAct(QA, tools=[...])` | `@ai(tools=[...])` |
| a `dspy.Module` with `forward` | a function with `@module` |
| `dspy.Example(...).with_inputs(...)` | a row: a dict, or a table (`expected=` names the answer column) |
| `metric(example, pred, trace=None)` | `metric(row, prediction)`, a dpyr expression, or an AI judge |
| `dspy.Evaluate(...)` | `functai.evaluate(...)`: a score with its range, and a table |
| `BootstrapFewShot`, `BootstrapFewShotWithRandomSearch` | the same names, used through `fn.opt(...)` |
| `MIPROv2` | `InstructionSearch` |
| `dspy.configure(lm=dspy.LM("openai/gpt-4o"))` | `functai.configure(lm="openai/gpt-4o")` (the same spelling works) |
| `dspy.inspect_history()` | `functai.phistory()` |

## pandas and polars

Your data frame goes in with `read`, and comes back with `.to_pandas()`
or `.to_polars()`. In between, [dpyr](https://github.com/MaximeRivest/dpyr)'s
verbs, where AI functions are columns:

```python
import pandas as pd
from dpyr import read, col
from typing import Literal

@ai
def language(text: str) -> Literal["English", "French", "Spanish", "other"]:
    """The language the text is written in."""

df = pd.DataFrame({"text": ["Merci beaucoup !", "Thanks a lot!", "¡Muchas gracias!"]})
read(df).mutate(language=language(col.text)).to_pandas()
```

```output
text language
0  Merci beaucoup !   French
1     Thanks a lot!  English
2  ¡Muchas gracias!  Spanish
```

| pandas | dpyr |
|---|---|
| `df["x"] = df["text"].apply(f)` | `.mutate(x=f(col.text))`: each distinct value once, in parallel, remembered |
| `df[df["text"].apply(is_spam)]` | `.filter(is_spam(col.text))` |
| `df.groupby("g").agg(...)` | `.group_by(col.g).summarize(...)` |
| `pd.read_csv(...)` | `read("file.csv")` (also parquet, Excel, JSON, databases, URLs) |

Why not `.apply`? It runs one row at a time, calls the model again for
repeated values, and loses everything if row 9,000 fails. See
[Big tables](tables.md).

## Instructor, Pydantic AI, structured outputs

| Instructor / Pydantic AI | functai |
|---|---|
| `response_model=MyModel` / `output_type=MyModel` | the return type: `-> MyModel` (pydantic, dataclass, TypedDict, `Literal`, lists) |
| field descriptions | `Field(description=...)`, or a comment on the field |
| validators and retries | pydantic validators run on the reply; an unreadable reply is asked again once |
| the prompt, as a string | the docstring, or a [chat template](layouts.md) |
| `Agent(tools=[...])` | `@ai(tools=[...])` |

What functai adds: running over tables, [measuring](accuracy.md),
[optimizing](improving.md), and [saving](saving.md) a program with
everything it needs.
