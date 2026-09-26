# BootstrapFewShotWithRandomSearch { #functai.BootstrapFewShotWithRandomSearch }

```{.python .no-run}
BootstrapFewShotWithRandomSearch(
    metric=None,
    *,
    metric_threshold=None,
    max_bootstrapped_demos=4,
    max_labeled_demos=16,
    max_rounds=1,
    num_candidate_programs=8,
    num_threads=1,
    max_errors=10,
    teacher=None,
    seed=0,
    stop_at_score=None,
)
```

Try several sets of demos and keep the one that scores best on the validation rows.

Several candidate demo sets (none, labeled only, bootstrapped, bootstrapped
from shuffled examples), each scored on ``valset`` (default: the trainset);
the best one wins. ``candidates`` holds every candidate afterwards, as rows
(``{"candidate", "demos", "score"}``; ``dpyr.read(opt.candidates)`` makes
them a table).