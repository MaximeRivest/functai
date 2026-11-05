# judge { #functai.judge }

```{.python .no-run}
judge(
    judge_,
    calls,
    criterion,
    *,
    folder=None,
    records=None,
    sample=None,
    expected=None,
    threads=1,
)
```

Have a judge rate logged calls, and keep its ratings in the log.

Each rating names the judge by what it runs on (a program's setup, a
function's code), so a changed judge is another judge, and the
scoreboard tells people's ratings from it. An item criterion has every
item judged.

## Parameters {.doc-section .doc-section-parameters}

| Name      | Type                                                                | Description                                                                                                     | Default    |
|-----------|---------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|------------|
| judge_    | AI function, module or function                                     | Given by name what it asks for: the call's inputs, ``answer``, ``outputs``, ``inputs``, ``item``, ``expected``. | _required_ |
| calls     | predictions, call ids, or call records                              |                                                                                                                 | _required_ |
| criterion | Criterion                                                           |                                                                                                                 | _required_ |
| folder    | where the calls are, and where ratings go (the call log by default) |                                                                                                                 | `None`     |
| records   | where the calls are, and where ratings go (the call log by default) |                                                                                                                 | `None`     |
| sample    | str                                                                 | The random draw these calls are (``draw``).                                                                     | `None`     |
| threads   | int                                                                 |                                                                                                                 | `1`        |

## Returns {.doc-section .doc-section-returns}

| Name   | Type         | Description          |
|--------|--------------|----------------------|
|        | list of dict | The ratings written. |