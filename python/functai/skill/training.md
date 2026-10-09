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
