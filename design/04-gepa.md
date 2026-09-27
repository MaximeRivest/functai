# 04. GEPA: instructions rewritten from mistakes

Status: Python, R, TypeScript and Julia. No change to the contract: the result is a
function's `state.instructions`, which every language already saves,
loads and versions.

## What GEPA is

GEPA (Agrawal et al., 2025, *Reflective Prompt Evolution Can Outperform
Reinforcement Learning*) improves the instruction of a language-model
program by reading its mistakes:

1. Keep a **pool** of candidate instructions, starting with the written
   one, each scored row by row on a **selection set**.
2. **Pick a parent** from the Pareto frontier: a candidate is on it when
   it is the best on at least one selection row, and it is picked with
   probability proportional to how many rows it is best on. This keeps
   candidates that are good at *different* things, instead of climbing
   one hill.
3. Run the parent on a small **minibatch** of feedback rows (not the
   selection rows), and show a **reflection model** the inputs, the
   answers, the scores and *feedback in words* ("the right answer is
   billing"). It writes a new instruction.
4. Run the child on the same minibatch. If it does better than the
   parent there, score it on the whole selection set and add it to the
   pool.
5. Stop at a budget of calls; return the candidate with the best mean on
   the selection set.

Its real ideas are two: **feedback in words** is far more informative
than a score, and **a Pareto pool** keeps diversity cheaply. Both are
kept here.

## Where it is weak, for functions like ours

1. **Reflection forgets.** Each proposal sees only its parent. A child
   rejected on the minibatch leaves no trace, so the next reflection on
   the same parent often proposes the same idea again.
2. **Reflection memorises.** Shown five messages, a model writes rules
   about those five messages ("if the customer mentions the knife set,
   answer billing"), sometimes quoting them. That scores on the
   minibatch and generalises poorly.
3. **Its merge does nothing for one function.** GEPA's merge takes each
   *module* from one ancestor or another: with a single AI function
   there is nothing to take apart.
4. **Instructions only grow.** Every reflection adds rules; nothing
   rewards a shorter one, and every call pays for its length.
5. **Paying twice for the same row.** A parent picked again on a row it
   has already answered is run again.
6. **The final score flatters.** The winner is the best of many on the
   selection rows, so its score there is optimistic (the winner's curse,
   tutorial 4). GEPA reports it as if it were not.

## What functai does

The same loop (steps 1-5), with these changes:

1. **Memory of what failed.** The reflection sees the last three
   instructions tried from the same parent that did not beat it, and is
   told to try something else.
2. **Rules, not cases.** The reflection is asked for general rules, never
   for notes on the cases shown; and a proposal that copies an input of
   30 characters or more, verbatim, is dropped before any call is spent
   on it (it counts as a failed attempt, so the next reflection is told).
3. **A combine step** for one function: every fourth step, when two
   candidates on the frontier each win rows the other loses, the
   reflection model is shown both and writes one instruction keeping
   what makes each right. It enters the pool on the same terms as any
   child (not worse than the better parent on a minibatch, then scored
   on the selection set).
4. **Ties go to the shorter instruction**, so length is paid for only
   when it buys accuracy.
5. **Answers are remembered** per candidate and row: a row is never run
   twice for the same instruction.
6. **Honest reporting.** The search is returned as a table (every
   candidate, its parent, how it was made, its minibatch and selection
   scores, its length). The documentation says the selection score is
   optimistic, and the tutorial measures the result on rows the search
   never saw. In tidymodels (R) and MLJ (Julia, `AIModel(method =
   :gepa)`), running GEPA inside `fit()` makes resampling measure the
   whole procedure, search included.

Kept from GEPA: the budget counts calls of the function being improved
(the reflection model's calls are counted apart, and logged); the
minibatch has 4 rows by default (GEPA's is 3); a parent that got every
minibatch row right is not reflected on; the feedback rows are visited
in shuffled rounds, so every row is seen before any is seen twice.

## What it takes

- `data`: rows with the right answers. Without a separate selection set,
  half of them (at random, `seed`) select and the other half give
  feedback. Selection rows are never shown to the reflection model.
- `metric`: a score per row, 1 meaning right (exact match by default).
- `feedback`: words per row, `(row, prediction) -> text`. By default:
  "right", or "wrong: the right answer is ...", or the call's error.
- `teacher`: the reflection model (default: the function's own).
- `budget`: calls of the function (default 300).

It improves one AI function's instruction and leaves its worked
examples as they are; add examples after with the few-shot optimizers.
A Python `@module` (several AI functions) is refused for now: its
feedback is about the module's answer, and deciding which function to
blame is the part of GEPA this first version leaves out.

## The two prompts

Both languages send the same words (they are AI functions themselves,
so they are logged like any other, marked as part of the optimization).

*Reflect* (inputs `fields`, `instruction`, `cases`, `tried`; output
`instruction`):

> You improve the instruction of a function that a language model
> runs. You are given what the function takes and returns, its current
> instruction, and cases it was run on: each with its inputs, the answer
> it gave, its score and feedback. Find what the instruction is missing,
> or gets wrong, that explains the mistakes, and write an improved
> instruction. Write general rules a careful person could follow on new
> cases; never copy an input or describe these particular cases. Keep
> what already works. The instruction is everything the model is told
> besides the inputs: keep the task, and say what each output must be.
> Instructions listed as tried did not do better: try something
> different. Reply with the new instruction only.

*Combine* (inputs `fields`, `first`, `second`; output `instruction`):

> Two instructions for the same function each get right some cases the
> other gets wrong. Write one instruction that keeps what makes each of
> them right, without repeating itself. The instruction is everything
> the model is told besides the inputs: keep the task, and say what each
> output must be. Reply with the new instruction only.
