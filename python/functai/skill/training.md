# FunctAI: training a small model on a function ("bake")

- Plan first, spend nothing: `print(fn.bake(rows, plan_only=True))`. For a function with a fixed set of
  answers (`Literal`, `Enum`, `bool`) it is the head's plan: answers never seen in the labels, the rows held
  out to test and to validate. With few rows, pass your own `test=` rows (by paper, by customer...), so the
  rest all train (StatLM, 168 rows: 86 trained by default, 136 with `test=`). Before 1.3, `plan_only`
  planned a generative student instead and refused an encoder.
- Training data in the exact format the function will send: SKILL.md §5
  (`functai.bake.examples(fn, rows, fixed=...)`, then `.save("train.jsonl")`).
  `derived=` leaves out an input decided by another one; `student="Qwen/..."`
  adds the exact token ids under that model's chat template.
- Train with any tool (TRL, Axolotl, Unsloth), then `functai.bake.adopt(folder, fn, examples=ex)`,
  or let `fn.bake(rows)` do all of it (`where="here"|"tinker"|"prime"|"export"`).
  Use the trained model: `fn.using(lm=baked)`.
- **Never rebuild prompts from functai internals** (`fn._variant_spec`,
  `engine.prepare_inputs`, `adapters.REGISTRY`, `bake.sft`...). One project did;
  a functai update broke it overnight and pinned it to a temporary copy. If the
  public API can't do something, say so to the user (it is their library).

## On Tinker (no GPU, no PyTorch): five ways to learn

```python
import functai, functai.bake
match = functai.Criterion("match", "score", scale=(0, 1), judge=lambda answer, expected: my_similarity(answer, expected))
plan = functai.bake.bake(fn, rows, where="tinker", student="Qwen/Qwen3.5-4B", criteria=[match],
                         held_out_by="screen", plan_only=True)
print(plan)                                   # cases, tokens, steps, checks, ≈ dollars, what is sent: show the user
s1 = functai.bake.bake(fn, rows, where="tinker", student="Qwen/Qwen3.5-4B", criteria=[match], held_out_by="screen")
s2 = functai.bake.bake(fn, rows, where="tinker", method="rl", start=s1, reward=match)   # then reinforce
fn.using(lm=s2)(...)
```

- Needs `TINKER_API_KEY` and `pip install "functai[tinker]"`. Show the plan's
  price before running; it spends real money.
- `method`: `"sft"` (rows' answers; `teacher=` answers rows without one),
  `"rl"` (`reward=` a Criterion with a judge, a function, or
  `functai.bake.rewards.energy(distance)` for answers spread like people's),
  `"prefer"` (`pairs=[{"inputs", "chosen", "rejected"}]`), `"distill"`
  (`teacher=` a Tinker model with the student's tokenizer), `"decision"`
  (a Literal/Enum/bool answer, with probabilities).
- `criteria=` decides which checkpoint and temperature are kept, on held-out
  cases (`test=`, or a split; `held_out_by=` a column such as a screen or a
  document, so near-duplicates stay on one side). Report the held-out
  score from `baked.meta["training"]`, never the training reward.
- After `rl`, look at `metrics.jsonl`'s `agreement` (how alike a case's
  answers are): a reward can collapse them to one answer.
- Inkling 2 Small needs `functai[inkling2]` until its files are published.
