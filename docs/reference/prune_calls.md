# prune_calls { #functai.prune_calls }

```{.python .no-run}
prune_calls(older_than='90d', *, folder=None, keep_rated=True)
```

Delete the call log's day folders older than a time, keeping what
ratings need.

Keeping a log small is deleting whole day folders (contract/calls.md,
*The folder*). Before a day goes, every rated call in it is kept: the
call, every call of its tree (a module's helpers), every call its
``saw`` names (the earlier turns a row replays) and their ratings are
copied into one file at the folder's top level
(``kept-<host>-<pid>-<hex>.jsonl``), which every reader reads. A row of
``rated`` made before pruning is made the same after.

## Parameters {.doc-section .doc-section-parameters}

| Name       | Type                      | Description                                                 | Default   |
|------------|---------------------------|-------------------------------------------------------------|-----------|
| older_than | (text, timedelta or date) | Day folders before this go: ``"90d"``, ``"12w"``, a date.   | `'90d'`   |
| folder     | str or path               | The log folder (default: the one calls are logged to here). | `None`    |
| keep_rated | bool                      | False deletes rated calls too.                              | `True`    |

## Returns {.doc-section .doc-section-returns}

| Name   | Type   | Description                                                                |
|--------|--------|----------------------------------------------------------------------------|
|        | dict   | ``{"days": folders deleted, "calls": calls deleted, "kept": calls kept}``. |