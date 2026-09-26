# bake.bake { #functai.bake.bake }

```{.python .no-run}
bake.bake(
    fn,
    data,
    *,
    student='jhu-clsp/ettin-encoder-17m',
    method='head',
    teacher=None,
    labels='auto',
    test=None,
    holdout=0.1,
    validation=0.1,
    epochs=None,
    lr=None,
    batch_size=32,
    max_length=None,
    device=None,
    seed=0,
    num_threads=16,
    path=None,
    compare_teacher=True,
    prices=None,
    local_files_only=False,
    log=True,
    **method_options,
)
```

Train weights that answer ``fn``; returns the baked model (see ``Baked``).

- ``data``: rows (a list of dicts, or any table ``dpyr.read`` takes). Input
  columns are named like the parameters; a column named like an output is a
  label (a ``<output>__probs`` column of ``{answer: p}`` is a soft label).
- ``teacher``: a model name or AI function that labels rows without labels
  (all rows with ``labels="teacher"``). Its probabilities are used when it
  gives them (Jev does), else its answers.
- ``labels``: ``"auto"`` (the data's labels, the teacher's for the rest),
  ``"data"`` (only rows with labels), ``"teacher"`` (the teacher's for every
  training row; the data's labels are then used only to test).
- ``test``: rows with labels to measure on; else ``holdout`` of the labeled
  rows is set aside (at least 50, at most 2,000).
- ``validation``: share of the training rows kept to stop training at the
  best pass and to fit the temperature (at least 32, at most 1,000).
- ``student``: a Hugging Face model id or local path. Ettin-17M trained in
  46 s to 91.5% on banking77 with human labels (2026-09-25).
- ``epochs``, ``lr``, ``batch_size``, ``max_length``, ``seed``: training
  settings (defaults as measured best per model size).
- ``device``: default the GPU with the most free memory if it has room,
  else the CPU; memory held by other programs is never taken.
- ``path``: where the model is written (default under ~/.cache/functai/baked).
- ``compare_teacher``: also run the teacher on the test rows, to report its
  accuracy next to the student's (the "teacher-limited" check).
- ``prices``: ``{"teacher": (dollars per M input tokens, per M output tokens),
  "gpu_per_hour": dollars}`` to report money and break-even.