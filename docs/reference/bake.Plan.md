# bake.Plan { #functai.bake.Plan }

```{.python .no-run}
bake.Plan(what, data, opts)
```

A generative bake, decided before anything is spent:
``functai.bake.plan(fn, rows)`` or ``fn.bake(rows, plan_only=True)``.

``print(plan)`` says the rows (and how many a teacher must answer, at
what price), the tokens per pass, the longest example against the
student's context, the student, each place it could train with its time
or price (or what that place is missing), the training settings, the
speed-up kernels found here, and the run's folder. ``plan.using(...)`` is
the same bake with some settings changed, decided again;
``plan.run()`` does it.

**Where it trains** (``where="auto"``): the place you prefer
(``where=``, a place or a list in order of preference, else
``functai.configure(bake_where=...)``); else here, when a GPU here can
train the student (the CPU only for small ones); else a service you
have set up (its key or login present, the student offered), the
cheapest when there are several; else none, and the plan says what each
place lacks.

**Which student** (``student=None``): from a short tested list, the
largest that trains where it is going within a day; on a service, the
largest that service offers.

## Attributes

| Name | Description |
| --- | --- |
| `folder` | The run's folder: ``run_folder=``, else one under ``~/.cache/functai/bakes`` named by the |
| `name` | The bake's name: ``name=``, else the function's (or the functions' joined by ``+``). |
| `output` | Where the model is written: ``path=``, else ``<folder>/baked`` (``<folder>/export`` for |

## Methods

| Name | Description |
| --- | --- |
| [decide](#functai.bake.Plan.decide) | Work out every choice left open (student, place, settings, estimates). Done when the |
| [run](#functai.bake.Plan.run) | Answer the rows that need the teacher, write the examples, and train. |
| [to_dict](#functai.bake.Plan.to_dict) | What plan.json holds (what the run's process reads). |
| [using](#functai.bake.Plan.using) | The same bake with some settings changed (decided again). |

### decide { #functai.bake.Plan.decide }

```{.python .no-run}
bake.Plan.decide()
```

Work out every choice left open (student, place, settings, estimates). Done when the
plan is made; returns the plan.

### run { #functai.bake.Plan.run }

```{.python .no-run}
bake.Plan.run(wait=None)
```

Answer the rows that need the teacher, write the examples, and train.
Returns the baked model (``wait=False``: the Run, at once).

### to_dict { #functai.bake.Plan.to_dict }

```{.python .no-run}
bake.Plan.to_dict()
```

What plan.json holds (what the run's process reads).

### using { #functai.bake.Plan.using }

```{.python .no-run}
bake.Plan.using(**changes)
```

The same bake with some settings changed (decided again).