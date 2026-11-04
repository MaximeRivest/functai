# Studio: building a pipeline with a person watching

Use this when someone asks you (a coding agent) for an AI pipeline they will use,
look at or show to others: "sort these messages", "extract this from papers",
"reply to these reviews". You write the code; `functai studio` runs it, records
it and measures it; the person judges the answers in the dashboard. Everything is
plain files in the project: the code is the only source of truth.

A person can do all of it in the dashboard alone (`functai studio`: connect an AI
account, make a pipeline from a sentence, add examples, run, judge, fix). When
they work with you, you do the building and measuring from the command line and
they look and judge there; AI accounts are connected in the dashboard
(Accounts) or with `functai.login()`.

```bash
functai studio --help          # every command; read it once
functai studio doctor          # which model keys are here, the project, the default model
```

## The method (in this order)

1. **Say the plan before building.** The steps, in plain words, named after the
   request ("1. sort each inquiry: new order, reorder or question; 2. …"), what
   each step reads and gives, and which model. **Name every fact the pipeline
   needs that the person has and you don't** (an FAQ, card prices, their notes,
   who handles what) and ask where it comes from: a file, a pasted list, a
   photo. Never let a step guess it: a guessed price settles no argument.
   Offer a sample only to try things out, and say it is one. Ask the person;
   change the plan if they say so. Only then write code.
2. **Write the pipeline**: a `.py` file with one `@ai` function per step and a
   `@module` that joins them (see SKILL.md §1 for writing steps whose words reach
   the model). Choices as `Literal[...]`, numbers as `int`/`float`, so answers can
   be checked exactly. Data the steps need (an FAQ, a style guide) is read from a
   file in the project, never invented. Check it: `functai studio map` shows the
   steps and how they connect, read from the code.
3. **Write examples without their answers**: `examples/<pipeline name>.csv`, one
   column per input. Typical, hard and odd cases (empty, very long, the wrong
   language, a picture where text is expected), and the person's real data when
   there is some. Never write the right answers: a test you were handed the
   answer to proves nothing, and only the person knows what right means.
4. **Run**: `functai studio run --estimate`, then `functai studio run`. It says
   what it will cost first and stops above the budget (exit 3; raise it with
   `--budget DOLLARS` only when the person agreed). Read the failed and odd rows:
   `functai studio show --failed`, `functai studio show #5`.
5. **Hand over**: `functai studio open` prints the dashboard's address. Tell the
   person, in one sentence, what to do there: "Open http://localhost:7781 and
   mark the answers right or wrong; about two minutes". **Do not judge the
   answers yourself.** If the person tells you a judgement in the conversation,
   record it under their name: `functai studio rate #5/kind wrong --answer
   question --by <their name>`; never `--by` your own opinion.
6. **Find the weak step**: `functai studio blame` gives each step judged wrong
   its right answer, reruns only what comes after, and counts which step's fix
   makes the end right. `functai studio why #5` does it for one example.
7. **Improve and measure**: change the weak step (its instruction, its type, its
   model, a split in two), then `functai studio compare --before HEAD` (or
   `--before old.py`). It measures on judged examples kept aside from whoever
   wrote the change (about 4 in 10). Another model for one step, without
   changing code: `functai studio compare --lm kind=<model>`. Say the result in
   words, with its uncertainty: "3 right before, 5 after, on 5 examples: better,
   but too few to be sure".
8. **Share**: `functai studio page` writes one HTML file (what it does, how
   often it's right, where it goes wrong, what it costs, where data goes, three
   examples step by step) that anyone can open.

## Rules

- **Never say a pipeline "works" or "is accurate" without people's judgements
  behind it.** Say how many examples were checked and the range, or say "not
  checked yet".
- **Models**: the project's default is in `.functai/studio/settings.json`
  (`lm`); a step's own `@ai(lm="…")` wins. Check which models exist before
  naming one (`functai studio doctor`, or the provider's model list): your memory
  of model names is out of date.
- **Costs**: `run`, `why`, `blame` and `compare` state their cost first and stop
  above the budget (exit 3); a reply already asked is reused from the project's
  cache, free. Do not raise the budget on your own.
- **Versions**: `functai studio versions` gives each version's score; the person
  goes back to one in the dashboard (Studio keeps each version's code), or you
  use git.
- **The person edits too.** In the dashboard they change models, instructions,
  kinds of answer, wiring, fixed values, files and steps; each change is written
  into the code. Read the file again before changing it, and never undo their
  changes without asking (`.functai/studio/edits/history.jsonl` lists them).
- **Files**: runs and judgements are in `.functai/calls/` (FunctAI's call log,
  git-ignored: it holds what people typed); Studio's own state in
  `.functai/studio/`. Do not edit them by hand.
- Answers are shortened in the terminal (`--full` shows them whole); `--json`
  on any command gives data instead of text.
