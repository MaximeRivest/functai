# bake.bake { #functai.bake.bake }

```{.python .no-run}
bake.bake(
    what,
    data=None,
    *,
    method='auto',
    student=None,
    teacher=None,
    labels='auto',
    where='auto',
    test=None,
    metric=None,
    report=True,
    compare_teacher=False,
    wait=True,
    plan_only=False,
    path=None,
    run_folder=None,
    fixed=None,
    derived=None,
    layout=None,
    reasoning=False,
    tags=None,
    weights=None,
    functions=None,
    name=None,
    validation=None,
    holdout=None,
    num_threads=16,
    local_files_only=False,
    log=True,
    seed=0,
    **training,
)
```

Train weights that answer an AI function (or several); returns the baked model.

``what`` and ``data``:

- ``fn, rows``: one function; rows are dicts (or any table ``dpyr.read``
  takes) with the inputs under the parameters' names and, when known, the
  answers under the outputs' names (``result`` for the return value).
- ``{fn: rows, fn2: rows2}``: one student for several functions, each
  called through its own layout.
- ``program, rows``: a ``@module``; it runs on each row (with ``teacher``
  answering every AI call inside it), and every call becomes an example
  of its function (``functions=`` keeps some).
- ``examples``: made by ``functai.bake.examples`` (or a file of them).

The main choices (all decided from the data when left out; ``plan_only=True``
prints the plan and spends nothing):

- ``method``: ``"head"``, ``"sft"``, or ``"auto"`` (a head when every
  output is finite and nothing asks for a generative student).
- ``student``: a Hugging Face model id or folder.
- ``teacher``: answers the rows without answers: a model name, an AI
  function, or ``{fn: teacher}``; default each function's own model.
  ``labels="teacher"`` asks it for every row; ``"data"`` uses only rows
  with answers.
- ``where``: ``"here"``, ``"tinker"``, ``"prime"``, ``"export"``, a list in
  order of preference, or a ``Trainer``; ``"auto"``: here when this machine
  can train it, else the cheapest service set up
  (``configure(bake_where=...)`` sets a preference once).
- ``fixed={"input": value}``: an input with one value in every row, left
  out of the student's prompt; a call with another value is refused.
  ``derived={"input": "other input"}``: an input decided by another, left
  out too.
- ``test``: rows to judge on (else a share of the rows with answers is set
  aside); ``metric``: how to score them (anything ``evaluate`` takes; an
  AI judge for open text); ``report=False`` skips judging.
- ``wait=False``: return the ``Run`` at once (a folder and a process that
  outlive this one); running the same bake again resumes it.
- training settings: ``lora`` (True/False), ``lora_rank``, ``lr``,
  ``epochs``, ``batch`` (examples per step), ``quantize="4bit"``,
  ``packing``, ``devices``, ``liger``, ``max_new_tokens``, ``merge``
  (merge the adapter into the weights; default True), ``report_to``
  (["wandb"], ...).
- for a head: ``epochs``, ``lr``, ``batch_size``, ``max_length``,
  ``device``, ``prices``, ``holdout``, ``validation`` as before.