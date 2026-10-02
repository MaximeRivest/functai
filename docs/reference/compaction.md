# compaction { #functai.compaction }

```{.python .no-run}
compaction(keep=20, every=10, summarize=None, lm=None, name='compaction')
```

Keep a long conversation short: older turns are folded into a summary.

When a turn ends and more than ``keep + every`` turns of its branch are
not yet summarized, every turn but the last ``keep`` is folded into the
summary (the earlier summary and those turns, given to ``summarize``).
The summary is kept in the conversation (an entry at that turn, so each
branch has its own), and the next turns are shown it, as a section of the
instruction, with only the turns after it. The call that wrote it is in
the call log; what a turn was shown is in its record, so a rated turn is
asked again with the same summary.

## Parameters {.doc-section .doc-section-parameters}

| Name      | Type     | Description                                                                                                                                                    | Default        |
|-----------|----------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------|
| keep      | int      | How many recent turns are always shown whole.                                                                                                                  | `20`           |
| every     | int      | How many more turns may pile up before the next summary (summarizing at every turn would cost a call per turn).                                                | `10`           |
| summarize | function | ``summarize(earlier_summary, new_turns) -> str``, given by position, ``new_turns`` a list of ``{input…, output…}`` rows. Default: an AI function of FunctAI's. | `None`         |
| lm        | str      | The model the default summarizer uses (default: the one configured).                                                                                           | `None`         |
| name      | str      | The plugin's name (two compactions with different settings on one conversation need two names).                                                                | `'compaction'` |

## Returns {.doc-section .doc-section-returns}

| Name   | Type   | Description                                                   |
|--------|--------|---------------------------------------------------------------|
|        | Plugin | For ``fn.conversation(..., plugins=[...])`` or ``configure``. |