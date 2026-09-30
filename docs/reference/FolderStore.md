# FolderStore { #functai.FolderStore }

```{.python .no-run}
FolderStore(folder)
```

Conversations kept in a folder, shared by every process that opens it:

    <folder>/conversations/<id>.jsonl   the records, one per line
    <folder>/conversations/<id>.lock    taken while appending
    <folder>/trees/<tree>.jsonl         each turn's call tree log (kept form)

Each append is one step across processes (a lock on the conversation),
written and flushed to disk before it returns (``durability`` ``"disk"``).
Files are readable by their owner only.