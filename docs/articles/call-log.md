---
rat:
  project: ../../python
  python:
    dependencies: ["-e .[data]", "pandas"]
---

# Every call, on record

*Keep each call on disk, mark answers right or wrong, and turn the corrections into rows that measure and improve the function.*

```python
import functai
functai.configure(lm="gpt-4.1-mini", temperature=0)   # the model behind every output on this page
from functai import ai, _ai
```

`phistory()` shows the last few calls, and forgets them when Python
stops. Once a function is in use (in a notebook you come back to, a
script, an agent), you want every call kept: what it was asked, what it
answered, what it cost. And, when an answer is wrong, what the right one
was. Those corrections are exactly the rows `evaluate` and `.opt` need.

## Turn it on

```python
import tempfile
functai.configure(log_calls=tempfile.mkdtemp())   # a scratch folder for this page
from dpyr import col, n
```

Usually it is `functai.configure(log_calls=True)`: every call is then
written to `~/.local/share/functai/calls`, one line per call, one file per
process and day. Nothing is kept unless you ask. Tools that start Python
for you (rat, Chattering) can ask for you by setting
`FUNCTAI_LOG_CALLS=1`.

```python
from typing import Literal

@ai
def team(message: str) -> Literal["shipping", "billing", "product"]:
    """Which team should answer this customer message?"""
    ...

messages = [
    "I was charged twice for one order.",
    "The chair arrived with a snapped leg.",
    "The kettle's handle came off on the first day.",
    "Tracking has said 'in transit' for two weeks.",
]
[team(m) for m in messages]
```

```output
['billing', 'product', 'product', 'shipping']
```

```python
functai.calls(team).select("message", "pred_result", "seconds", "input_tokens")
```

```output
# dpyr dataframe · source: polars · showing 4 of 4 rows
┌────────────────────────────────────────────────┬─────────────┬──────────┬──────────────┐
│ message                                        ┆ pred_result ┆ seconds  ┆ input_tokens │
│ ---                                            ┆ ---         ┆ ---      ┆ ---          │
│ str                                            ┆ str         ┆ f64      ┆ i64          │
╞════════════════════════════════════════════════╪═════════════╪══════════╪══════════════╡
│ I was charged twice for one order.             ┆ billing     ┆ 0.877846 ┆ 58           │
│ The chair arrived with a snapped leg.          ┆ product     ┆ 1.079286 ┆ 58           │
│ The kettle's handle came off on the first day. ┆ product     ┆ 0.750674 ┆ 61           │
│ Tracking has said 'in transit' for two weeks.  ┆ shipping    ┆ 0.796156 ┆ 61           │
└────────────────────────────────────────────────┴─────────────┴──────────┴──────────────┘
```

`functai.calls(team)` reads them back as a table: the inputs, the answers
(`pred_result`, as in `evaluate`'s tables), and for each call its time,
tokens, model, version and who called.

## Right or wrong

Get the call with `predict`, then say whether its answer is right:

```python
p = team.predict("The chair arrived with a snapped leg.")
p.result
```

```output
'product'
```

Here, anything broken on the way is the shipping team's (the carrier
pays for it), so this answer is wrong:

```python
functai.rate(p, "wrong", answer="shipping", note="Broken on the way is shipping: the carrier pays.")
```

```output
{'functai_rating': 1, 'id': '01a0fce6-6994-72d6-858a-dc9448b3d153', 'call': '01a0fce6-6685-7630-bf31-e59b200eba54', 'at': '2026-10-02T13:55:53.876946Z', 'account': 'maxime', 'verdict': 'wrong', 'answer': 'shipping', 'note': 'Broken on the way is shipping: the carrier pays.'}
```

**Right means correct for this input, not "nice".** A wrong answer can
carry the right one (`answer=`); then it is a row of data. Without it
(`functai.rate(p, "wrong")`) it still says the answer is wrong, but not
what it should be.

```python
for message in ["The kettle's handle came off on the first day.", "I was charged twice for one order."]:
    functai.rate(team.predict(message), "right")
```

A later rating by the same person replaces the earlier one, and
`functai.rate(p, None)` withdraws it. Ratings are written to the log,
next to the calls, so any tool that reads the log (a dashboard, another
notebook) sees them, and can add its own.

## Corrections are data

`functai.rated(team)` turns the rated calls into rows with known answers:
the inputs under their names and the right answer under the output's.
It is the table `evaluate` and `.opt` take:

```python
rows = functai.rated(team)
rows.select("message", "result", "rating")
```

```output
# dpyr dataframe · source: polars · showing 3 of 3 rows
┌────────────────────────────────────────────────┬──────────┬────────┐
│ message                                        ┆ result   ┆ rating │
│ ---                                            ┆ ---      ┆ ---    │
│ str                                            ┆ str      ┆ str    │
╞════════════════════════════════════════════════╪══════════╪════════╡
│ The chair arrived with a snapped leg.          ┆ shipping ┆ wrong  │
│ The kettle's handle came off on the first day. ┆ product  ┆ right  │
│ I was charged twice for one order.             ┆ billing  ┆ right  │
└────────────────────────────────────────────────┴──────────┴────────┘
```

```python
functai.evaluate(team, rows)
```

```output
Evaluation(team, 3 examples: exact_match 0.67 [0.21, 0.94])
```

It gets the chair wrong, as we said. Teach it with the corrections, and
try it on a message it has not seen, before and after:

```python
taught = team.opt(rows)                          # an improved copy
team("My order came but the screen is cracked."), taught("My order came but the screen is cracked.")
```

```output
('product', 'shipping')
```

(Three rows are enough to show the mechanics, not to measure anything:
[Make it better](improving.md) keeps the rows it learns from apart from
the rows it is judged on.)

**One honest limit.** The calls people choose to rate are not a fair
sample: people rate what surprised them. A score on them says how the
function does *on those*. To know how often it is right in use, rate a
random draw of calls made in use, and give the draw a name:

```python
draw = functai.calls(team).filter(col.purpose == "use").slice_sample(n=3, seed=4)
draw.select("message", "pred_result", "call")
```

```output
# dpyr dataframe · source: polars · showing 3 of 3 rows
┌──────────────────────────────────────────┬─────────────┬──────────────────────────────────────┐
│ message                                  ┆ pred_result ┆ call                                 │
│ ---                                      ┆ ---         ┆ ---                                  │
│ str                                      ┆ str         ┆ str                                  │
╞══════════════════════════════════════════╪═════════════╪══════════════════════════════════════╡
│ My order came but the screen is cracked. ┆ shipping    ┆ 01a0fce6-911d-75e4-bcdb-9b76c32a194a │
│ I was charged twice for one order.       ┆ billing     ┆ 01a0fce6-5737-725c-8ae6-d2336c5599fa │
│ The chair arrived with a snapped leg.    ┆ product     ┆ 01a0fce6-5aa5-7708-9eff-d433a6c316a9 │
└──────────────────────────────────────────┴─────────────┴──────────────────────────────────────┘
```

Read each answer, then rate that row of the draw:

```{.python .no-run}
drawn = draw.collect().to_dicts()
functai.rate(drawn[0], "right", sample="draw-1")
functai.rate(drawn[1], "right", sample="draw-1")
functai.rate(drawn[2], "wrong", answer="shipping", sample="draw-1")
```

The rows of that draw, `functai.rated(team).filter(col.sample ==
"draw-1")`, give a score you can trust, once there are enough of them
(see [Is it right?](accuracy.md) for how many).

## Versions

Every call records which version of the function made it:

```python
team.version
```

```output
'sha256:c73c5af7368ddd503405fb6b3fb07e725250305109d91ea347d469256fb14586'
```

A version names everything the function sends besides its inputs: the
instruction, the worked examples, the layout, the tools, and its code when
it has code of its own (`return round(_ai, 2)`). A function whose body the
model writes whole has no code in its version, so the same function written
in another language has the same version.
Optimizing it, as we just did, made a new one; choosing another model
does not (the model is recorded next to it). So the log can compare
versions answer by answer:

```python
functai.calls(team).group_by("version").summarise(calls=n())
```

```output
# dpyr dataframe · source: polars · showing 2 of 2 rows
┌─────────────────────────────────────────────────────────────────────────┬───────┐
│ version                                                                 ┆ calls │
│ ---                                                                     ┆ ---   │
│ str                                                                     ┆ i64   │
╞═════════════════════════════════════════════════════════════════════════╪═══════╡
│ sha256:24a0418930b1a78d4a32cbc53eecb700d2d8f1126d205bd774fb37e56d8cffed ┆ 1     │
│ sha256:c73c5af7368ddd503405fb6b3fb07e725250305109d91ea347d469256fb14586 ┆ 14    │
└─────────────────────────────────────────────────────────────────────────┴───────┘
```

A saved program has the version of the one that was saved, so a call
made by a loaded program says exactly which saved folder answered.

## Who called, and what is kept

Calls made by `evaluate` and `.opt` say so (their `purpose` is
`evaluation` or `optimization`, not `use`): they answer known questions,
so a draw of real use leaves them out, as above. Say who is calling with
`caller`:

```python
with functai.configure(caller={"kind": "script", "user": "maxime"}):
    team("Where is my order?")
functai.calls(team).select("message", "caller").slice_tail(n=1)
```

```output
# dpyr dataframe · source: polars · showing 1 of 1 rows
┌────────────────────┬───────────────────────────────────┐
│ message            ┆ caller                            │
│ ---                ┆ ---                               │
│ str                ┆ str                               │
╞════════════════════╪═══════════════════════════════════╡
│ Where is my order? ┆ {"kind":"script","user":"maxime"} │
└────────────────────┴───────────────────────────────────┘
```

Programs that may see secrets (a clipboard helper sees passwords) keep
only sizes, times and tokens:

```python
@ai(log_content=False)
def fix_grammar(text: str) -> str:
    """The text with its grammar and spelling fixed; nothing else changed."""
    ...

fix_grammar("their going too the store")
functai.calls(fix_grammar).select("seconds", "input_tokens", "output_tokens", "model")
```

```output
# dpyr dataframe · source: polars · showing 1 of 1 rows
┌──────────┬──────────────┬───────────────┬──────────────┐
│ seconds  ┆ input_tokens ┆ output_tokens ┆ model        │
│ ---      ┆ ---          ┆ ---           ┆ ---          │
│ f64      ┆ i64          ┆ i64           ┆ str          │
╞══════════╪══════════════╪═══════════════╪══════════════╡
│ 0.875936 ┆ 55           ┆ 13            ┆ gpt-4.1-mini │
└──────────┴──────────────┴───────────────┴──────────────┘
```

`log_calls=False` on a function keeps it out of the log altogether.

## What a line holds

Each line is one call, as JSON: the inputs and outputs typed as the
function says, every request and reply exactly as sent, the tokens, the
time, the version, and the call it ran in (a module's steps are its
children). The format is a written contract
([`contract/calls.md`](https://github.com/maximerivest/functai/blob/master/contract/calls.md)),
so other programs, and FunctAI in other languages, read and write the
same folder.

```python
import json
from pathlib import Path
folder = Path(functai.settings.log_calls)
last = json.loads(sorted(folder.rglob("*.jsonl"))[-1].read_text().splitlines()[-1])
{k: last[k] for k in ("program", "content", "sizes", "model", "usage")}
```

```output
{'program': {'name': 'fix_grammar', 'kind': 'ai', 'module': '__main__', 'version': 'sha256:48485b5298ac7491dcaa9bbbd5e792d640f9d829e6131b8e597a7682023d519c', 'signature': 'sha256:0553dbe4a0c5e2004a138dff3be56e6e622a792fb34c20493475c9c6c2044fa8', 'interface': 'sha256:0553dbe4a0c5e2004a138dff3be56e6e622a792fb34c20493475c9c6c2044fa8', 'answer': 'result', 'line': 1}, 'content': False, 'sizes': {'inputs': {'text': 27}, 'outputs': {'result': 29}}, 'model': 'gpt-4.1-mini', 'usage': {'input_tokens': 55, 'output_tokens': 13, 'total_tokens': 68, 'cache_read_tokens': 0, 'cache_write_tokens': 0, 'reasoning_tokens': 0}}
```

## Keeping it small

The log only grows, one folder per day. `prune_calls` deletes the day
folders older than a time, and first keeps what your ratings need: every
rated call, the calls of its tree (a module's steps), the earlier turns it
was shown, and their ratings are copied into one file at the top of the
log, which every reader reads. So `rated(...)` gives the same rows after
pruning as before.

```{.python .no-run}
functai.prune_calls("90d")                    # day folders older than 90 days go
functai.prune_calls("12w", keep_rated=False)  # rated calls go too
```

It returns how many day folders and calls it deleted, and how many calls
it kept (`{"days", "calls", "kept"}`). `folder=` prunes another log.
