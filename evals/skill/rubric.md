# Rubric: what "done well" means for each task

Every check is computed by code (`tasks/<task>/grade.py`), never by a model's
opinion, from the task's outputs and the recording proxy's log of every request
the task's code sent (the same instrument for FunctAI and DSPy). Each task comes from a mistake that cost a real project hours; the
"why" column says which.

A run **succeeds** when every required check passes. Its **score** is the mean
of all checks (fractions count as their value). Efficiency is measured beside
it: wall time, agent tokens and list-price cost, tool calls, failed tool calls,
and the model calls the task's own code made.

## forms: fill a codebook's form for 10 study descriptions

| check | required | passes when | why |
|---|---|---|---|
| files | ✓ | extract.py, out/designs.csv, out/analyses.csv with the asked columns | |
| rerun | ✓ | running extract.py again sends no request to the provider | long runs must resume for free (`cache_replies="disk"`; DSPy caches by default) |
| guidance | ✓ | ≥ 85% of the codebook's 7 conventions appear in what the model receives (the re-run's requests; when it sends none, the agent's latest request per paper) | StatLM wrote 65 schema comments; 64 never reached the model (nested dataclass and `Literal` member comments are dropped) |
| units | ✓ | ≥ 80% right; the codebook counts units **per treatment group**, which a model cannot guess | outcome effect of the guidance being lost |
| methods | ✓ | F1 ≥ 0.85 on analyses (Spearman counts as `rank_test` by the codebook) | |
| design | | ≥ 90% right | |
| pseudoreplicated | | ≥ 90% right on matched analyses | |
| outcome | | ≥ 70% written the codebook's way (lowercase, short) | |

## measure: score a classifier honestly, then judge a change

| check | required | passes when | why |
|---|---|---|---|
| files | ✓ | predictions.csv (id, category, v1, v2) for all 80 tickets; results.json with the 8 keys | |
| real | ✓ | ≥ 76 of the 80 tickets reached the model | the numbers must come from real predictions |
| accuracy | ✓ | reported accuracies equal the ones recomputed from predictions.csv | |
| interval | ✓ | v1_low–v1_high within 0.03 of Wilson's 95% interval | a score without its range invites over-reading 80 rows |
| paired | ✓ | diff_low–diff_high is a paired 95% interval of the per-ticket differences: within 0.02 of the paired t interval (what `functai.compare` gives), or a cautious (exact, Bonferroni...) interval that contains the t interval and is at most twice as wide | pairing is what lets 80 rows tell a real change from luck |
| verdict | ✓ | "unclear" exactly when the reported (accepted) interval includes 0 | |

## export: training conversations for a small model

| check | required | passes when | why |
|---|---|---|---|
| files | ✓ | train.jsonl, a `messages` list per line | |
| rows | ✓ | 24 conversations | |
| exact | ✓ | ≥ 95% of conversations identical to the library's own format: `functai.bake.examples(simplify, rows, fixed={"style_guide": ...})`, or DSPy's `ChatAdapter().format_finetune_data(...)` for `Simplify.delete("style_guide")` | Scholar's Reading List rebuilt this from functai internals and an AST hack; it broke on the next update |
| no_guide | ✓ | the style guide is in no message | the student must learn it, as `fixed=` does |
| free | ✓ | no request reached the provider | every row already has its answer |

## photos: shape and colour in 12 pictures

| check | required | passes when | why |
|---|---|---|---|
| files | ✓ | predictions.csv with file, shape, color for all 12 | |
| images | ✓ | ≥ 12 requests carry an image part; none pastes base64 as text | the species pilot spent dozens of tool calls finding how to send an image |
| shape, color | ✓ | ≥ 11 of 12 right | |

## files: questions about a PDF report and a spreadsheet

| check | required | passes when | why |
|---|---|---|---|
| files | ✓ | answers.csv with id and answer for q1..q5 | |
| real | ✓ | the report's text and the spreadsheet's numbers reached the model | answers must come from the documents |
| picture | ✓ | at least one request carries an image part | Figure 1's values are only in its picture; a text-only reading cannot answer q1 and q2 |
| chart | ✓ | q1 (highest region) and q2 (South's value) right | the reason the picture must be sent |
| text | ✓ | q3 (dividend, PDF text), q4 (most over budget), q5 (Engineering's spending) right | |

The company was renamed from "Northwind" to "Larkspur" so that "North" (q1's
answer) is in no text layer. Both environments have the same document
tools (Attachments, pypdf, PyMuPDF, openpyxl, pandas); only the AI library
differs, and only FunctAI's arm gets a skill that names Attachments.

## Reading the results

- Compare **success** first, then **time** and **agent cost** among successes.
- "read skill" says whether the `with` agent opened the skill; a `with` run that
  didn't is effectively a `without` run (the skill's description is what makes
  agents open it).
- "contaminated" means the agent looked inside this evaluation or the skill in
  the repository; discard those runs.
- Three repetitions per cell show the direction, not a precise effect: agent
  runs vary a lot. Add `--reps` before drawing fine conclusions.
