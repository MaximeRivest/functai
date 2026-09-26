# BootstrapFewShot { #functai.BootstrapFewShot }

```{.python .no-run}
BootstrapFewShot(
    metric=None,
    *,
    metric_threshold=None,
    max_bootstrapped_demos=4,
    max_labeled_demos=16,
    max_rounds=1,
    max_errors=10,
    teacher=None,
    teacher_settings=None,
    num_threads=1,
    seed=0,
)
```

Run the program (or a ``teacher``: a stronger model name, or an AI function)
on training examples; the runs ``metric`` accepts become demos, whole turns
included. Labeled examples fill the rest, up to ``max_labeled_demos``.