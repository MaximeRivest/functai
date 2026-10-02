# bake.Run { #functai.bake.Run }

```{.python .no-run}
bake.Run(folder)
```

A training run: a folder, a process of its own, and a model at the end.

What ``bake(..., wait=False)`` returns at once (and ``Plan.run``,
``functai.bake.runs()``, ``functai.bake.run(folder)``). The run is its own
process, so it outlives the notebook or terminal that started it; running
the same bake again with the same plan finds its folder and resumes it (or
finds it done).

```{.python .no-run}
run = summarize.bake(rows, wait=False)   # returns at once
run.metrics()                            # the loss curve so far, as a table
run.stop(); run.resume()                 # from the last checkpoint
baked = run.wait()                       # the model, when it is done
run.checkpoint(1200)                     # any checkpoint as a model
```

Its ``folder`` holds ``plan.json`` (every resolved setting: what makes
resuming exact), ``examples.parquet`` (the training conversations),
``run.json`` (its state, where it trains, its process, its attempts),
``metrics.jsonl`` (one line per logged step: step, tokens, loss, learning
rate, validation loss, seconds), ``log.txt`` (the trainer's own output),
``checkpoints/`` and, when done, ``baked/`` (the model).

## Attributes

| Name | Description |
| --- | --- |
| `plan` | Every setting the run was decided with (its ``plan.json``). |
| `state` | ``planned``, ``starting``, ``running``, ``stopping``, ``stopped`` (by ``stop()``, or its |
| `status` | run.json, with a process that ended without saying so shown as such. |
| `where` | Where it trains: ``here``, ``tinker``, ``prime`` or ``export``. |

## Methods

| Name | Description |
| --- | --- |
| [baked](#functai.bake.Run.baked) | The model this run made (when done). |
| [checkpoint](#functai.bake.Run.checkpoint) | A checkpoint as a usable model (the last one by default). Models from |
| [checkpoints](#functai.bake.Run.checkpoints) | Steps with a checkpoint, oldest first. |
| [log](#functai.bake.Run.log) | The last ``lines`` lines of the trainer's own output (``log.txt``): read it when a run fails. |
| [log_metric](#functai.bake.Run.log_metric) | Append one line to metrics.jsonl (what trainers call). |
| [metrics](#functai.bake.Run.metrics) | The logged steps as a dpyr table (a list of dicts without dpyr). |
| [progress](#functai.bake.Run.progress) | Where it is now: ``state``, ``step`` of ``steps``, the last ``loss`` and ``eval_loss``, |
| [records](#functai.bake.Run.records) | The lines of ``metrics.jsonl`` so far, as dicts (``metrics()`` gives them as a table). |
| [resume](#functai.bake.Run.resume) | Continue a stopped or crashed run from its last checkpoint. |
| [start](#functai.bake.Run.start) | Start (or resume) the run in its own process; ``process=False`` runs it |
| [stop](#functai.bake.Run.stop) | Ask the run to stop at its next step (it saves a checkpoint first). |
| [stop_requested](#functai.bake.Run.stop_requested) | Whether ``stop()`` was asked (what the trainer checks at each step). |
| [update](#functai.bake.Run.update) | Change run.json (the supervisor's; also ``stop``). |
| [wait](#functai.bake.Run.wait) | Wait for the end; show progress; return the model. |

### baked { #functai.bake.Run.baked }

```{.python .no-run}
bake.Run.baked()
```

The model this run made (when done).

### checkpoint { #functai.bake.Run.checkpoint }

```{.python .no-run}
bake.Run.checkpoint(step=None)
```

A checkpoint as a usable model (the last one by default). Models from
the constant phase of the schedule are not decayed: compare them with
each other, not with the final model.

### checkpoints { #functai.bake.Run.checkpoints }

```{.python .no-run}
bake.Run.checkpoints()
```

Steps with a checkpoint, oldest first.

### log { #functai.bake.Run.log }

```{.python .no-run}
bake.Run.log(lines=40)
```

The last ``lines`` lines of the trainer's own output (``log.txt``): read it when a run fails.

### log_metric { #functai.bake.Run.log_metric }

```{.python .no-run}
bake.Run.log_metric(**fields)
```

Append one line to metrics.jsonl (what trainers call).

### metrics { #functai.bake.Run.metrics }

```{.python .no-run}
bake.Run.metrics()
```

The logged steps as a dpyr table (a list of dicts without dpyr).

### progress { #functai.bake.Run.progress }

```{.python .no-run}
bake.Run.progress()
```

Where it is now: ``state``, ``step`` of ``steps``, the last ``loss`` and ``eval_loss``,
``seconds`` so far and ``eta`` (seconds left, estimated).

### records { #functai.bake.Run.records }

```{.python .no-run}
bake.Run.records()
```

The lines of ``metrics.jsonl`` so far, as dicts (``metrics()`` gives them as a table).

### resume { #functai.bake.Run.resume }

```{.python .no-run}
bake.Run.resume()
```

Continue a stopped or crashed run from its last checkpoint.

### start { #functai.bake.Run.start }

```{.python .no-run}
bake.Run.start(process=True)
```

Start (or resume) the run in its own process; ``process=False`` runs it
in a thread of this one (ends with it).

### stop { #functai.bake.Run.stop }

```{.python .no-run}
bake.Run.stop(wait=True, timeout=600)
```

Ask the run to stop at its next step (it saves a checkpoint first).

### stop_requested { #functai.bake.Run.stop_requested }

```{.python .no-run}
bake.Run.stop_requested()
```

Whether ``stop()`` was asked (what the trainer checks at each step).

### update { #functai.bake.Run.update }

```{.python .no-run}
bake.Run.update(**fields)
```

Change run.json (the supervisor's; also ``stop``).

### wait { #functai.bake.Run.wait }

```{.python .no-run}
bake.Run.wait(poll=5.0, show=True)
```

Wait for the end; show progress; return the model.