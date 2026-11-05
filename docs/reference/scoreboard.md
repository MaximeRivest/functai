# scoreboard { #functai.scoreboard }

```{.python .no-run}
scoreboard(
    program,
    criteria=(correct,),
    *,
    by='setup',
    folder=None,
    records=None,
    since=None,
)
```

How good every setup of a program is, by every source of judgement.

Reads the call log (or ``records``): the program's results (calls on a
case), and ratings of them. For each setup (or version), in order of its
first result: how many results and cases, failures, time, tokens of
whole trees (a pipeline's steps included) and what code spent; then each
criterion's estimate with its 95% range over cases, by people (all their
ratings), by people drawn (a random draw's: the fair one), by each judge,
and each judge combined with people drawn (prediction-powered inference).

## Parameters {.doc-section .doc-section-parameters}

| Name     | Type                         | Description                                                                  | Default      |
|----------|------------------------------|------------------------------------------------------------------------------|--------------|
| program  | AI function, module, or name |                                                                              | _required_   |
| criteria | list of Criterion            | Single-answer criteria (a choice is between setups: ``win_rate``, ``rank``). | `(correct,)` |
| by       | \'setup\' or \'version\'     |                                                                              | `'setup'`    |
| folder   | Any                          | Where the records are (the call log by default).                             | `None`       |
| records  | Any                          | Where the records are (the call log by default).                             | `None`       |
| since    | Any                          | Where the records are (the call log by default).                             | `None`       |

## Returns {.doc-section .doc-section-returns}

| Name   | Type       | Description   |
|--------|------------|---------------|
|        | Scoreboard |               |

## See Also {.doc-section .doc-section-see-also}

compare_setups : two setups, case by case.
agreement : whether a judge can stand in for people.