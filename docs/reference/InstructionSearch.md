# InstructionSearch { #functai.InstructionSearch }

```{.python .no-run}
InstructionSearch(
    metric=None,
    *,
    num_candidates=6,
    num_trials=12,
    minibatch_size=20,
    max_bootstrapped_demos=4,
    max_labeled_demos=4,
    prompt_lm=None,
    num_threads=1,
    seed=0,
    full_eval_top=3,
    metric_threshold=None,
    max_errors=10,
    teacher=None,
)
```

Search instructions written by a model, with demo sets, and keep the best.

Instruction candidates (the current one plus proposals written by
``prompt_lm`` from the code, the signature and a few examples) × demo sets
(bootstrapped, unless both demo limits are 0), searched over ``num_trials``
minibatch evaluations; the top combinations are then scored on the whole
``valset`` and the best wins. ``trials`` holds every trial afterwards, as
rows (``dpyr.read(opt.trials)`` makes them a table).

MIPRO-style; the search is random with greedy refinement, not Bayesian.