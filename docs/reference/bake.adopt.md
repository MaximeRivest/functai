# bake.adopt { #functai.bake.adopt }

```{.python .no-run}
bake.adopt(
    folder,
    fn,
    *,
    examples=None,
    student=None,
    path=None,
    layout=None,
    fixed=None,
    derived=None,
    reasoning=False,
    name=None,
    max_new_tokens=None,
)
```

A model trained elsewhere (TRL, Axolotl, Unsloth, by hand), as a baked
model functai calls through the function's layout.

``folder``: a Hugging Face model folder, or a LoRA adapter folder (its base
named in ``adapter_config.json``, or ``student=``). ``fn``: the AI function
(or a list of them) it answers. ``examples``: the examples it was trained
on (``functai.bake.examples(..., student=...)``, or their file): their
layouts are used, and their tokens are checked against the folder's chat
template, so a model trained on other tokens is refused. Without examples,
the function's own layout is used and nothing can be checked.