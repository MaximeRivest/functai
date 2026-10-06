# FunctAI: training a small model on a function ("bake")

- Plan first, spend nothing: `print(fn.bake(rows, plan_only=True))`.
- Training data in the **exact** format the function will later send:
  ```python
  import functai.bake
  ex = functai.bake.examples(fn, rows, fixed={"style_guide": GUIDE})  # rows carry the answer column (e.g. "result")
  ex.stats(); ex.save("train.jsonl")       # chat `messages` per row; .parquet works too
  ```
  `fixed=` leaves a long constant input out of every example (the student
  learns it); `derived=` an input decided by another one; `student="Qwen/..."`
  adds exact token ids. Needs at least 10 rows.
- Train with any tool (TRL, Axolotl, Unsloth), then `functai.bake.adopt(folder, fn, examples=ex)`,
  or let `fn.bake(rows)` do all of it (`where="here"|"tinker"|"prime"|"export"`).
  Use the trained model: `fn.using(lm=baked)`.
- **Never rebuild prompts from functai internals** (`fn._variant_spec`,
  `engine.prepare_inputs`, `adapters.REGISTRY`, `bake.sft`...). One project did;
  a functai update broke it overnight and pinned it to a temporary copy. If the
  public API can't do something, say so to the user (it is their library).
