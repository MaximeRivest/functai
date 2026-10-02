# LabeledFewShot { #functai.LabeledFewShot }

```{.python .no-run}
LabeledFewShot(k=16, *, sample=True, seed=0)
```

Rows with known answers become worked examples, sent before every call.

The cheapest optimizer: no model is called to optimize. Use it as
``fn.opt(rows, optimizer=LabeledFewShot(k=8))``, or by name:
``functai.labeled_few_shot(fn, rows, k=8)``. One AI function only (a
``@module`` needs ``BootstrapFewShot``, which knows which step each
example belongs to).

## Parameters {.doc-section .doc-section-parameters}

| Name   | Type   | Description                                                          | Default   |
|--------|--------|----------------------------------------------------------------------|-----------|
| k      | int    | At most this many examples (default 16).                             | `16`      |
| sample | bool   | True (default): a random sample of the rows; False: the first ``k``. | `True`    |
| seed   | int    | The sample's seed, so the same rows give the same examples.          | `0`       |