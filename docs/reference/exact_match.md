# exact_match { #functai.exact_match }

```{.python .no-run}
exact_match(row, pred)
```

The default metric: every expected output equals the prediction.

1.0 when every output the data has a column for equals the prediction's
(strings compared ignoring case and repeated whitespace), else 0.0.