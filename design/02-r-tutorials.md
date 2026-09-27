# 02: Teaching FunctAI for R: what the best R tutorials do, and the series it gave

Written 2026-09-27, before the eight tutorials in `docs/r/`. The list of
resources comes from what the R community recommends (r/rstats threads,
Posit Community's "favorite intro to R", R-bloggers and 2024-2026
round-ups, read through a web search), plus the teaching pages closest to
FunctAI's job: tidymodels' Get Started series, Julia Silge's screencast
posts, and the two LLM packages R users meet first, ellmer and mall. Each
was read at its opening and its structure, the parts that decide whether
a reader stays.

## What each one does

| Resource | Voice | How it opens | How it teaches | What it covers well | What it glosses over |
|---|---|---|---|---|---|
| **R for Data Science, 2e** (Wickham, Çetinkaya-Rundel, Grolemund) | Warm, plural "we", confident, a joke now and then | A question about real data ("Do penguins with longer flippers weigh more?"), then the **ultimate goal**: the finished plot, shown before any code | Builds that plot one layer at a time; asks the reader to predict ("will we modify the aesthetic or the geom? If you guessed…"); reads every warning out loud as a lesson ("missing values should never silently go missing"); exercises at each section's end; a summary | The one mental model (a plot maps variables to aesthetics) before the catalogue | Modelling (deliberately, and says so in "What you won't learn") |
| **Hands-On Programming with R** (Grolemund) | First person, playful, a storyteller | A mission: "be the casino"; build dice, cards, a slot machine | Projects; every concept arrives because the project needs it; asides ("Isn't R a language?") for the curious | Why each piece exists | Real data; statistics |
| **tidymodels, Get Started** (5 articles) | Neutral, precise, patient | A dataset with a domain primer (sea urchins, cell segmentation), then "plot the data first" | One task per article, each building on the last; the **trap before the fix** ("What happened here?": the training set scores 99%, the test set 81%); a teaching analogy (a test whose answers the students saw); the whole game as a final case study; session info at the end | Honest evaluation; the same verbs for every model | Choosing a metric for a real decision; what a mistake costs |
| **Tidy Modeling with R** (Kuhn, Silge) | Textbook-careful, opinionated where it matters | Why the chapter matters, with a figure where two metrics disagree | Motivation, then syntax; one running dataset (Ames) | Metrics have consequences ("choosing the wrong metric can easily result in unintended consequences") | Text; anything a language model changes |
| **Supervised ML for Text Analysis in R** (Hvitfeldt, Silge) | Practical, calm | The dataset and the question in two paragraphs | A first model, then **"compare to the null model"**, then better models; "what evaluation metrics are appropriate?"; "the full game"; "in this chapter, you learned" | Baselines; precision and recall; confusion matrices | Cost of errors per decision; asking a model instead of training one |
| **Julia Silge's screencast posts** | Conversational, "let's" | "Our modeling goal is…", and an honest expectation ("it's not likely we can build a high performing model… but") | Explore, then a "data budget", then models; one post, one idea | Realistic workflow and expectations | Anything beyond the one idea |
| **Advanced R** (Wickham) | Exact, dense, respectful of the reader's time | A **quiz: answer these to see if you can skip the chapter**, then an outline | Definitions, then small experiments at the console, answers at the end | The mental model of a function (formals, body, environment) | Applied work, by design |
| **dplyr's "Introduction"** | Brisk | Three steps of working with data, and how dplyr helps each | A familiar dataset (starwars), one verb at a time | The vocabulary of verbs | Why, when |
| **ellmer: Get started, Structured data, Tool calling** (Posit) | Friendly, knowledgeable | Vocabulary first (prompt, token, conversation), then examples | Tool calling opens with the **failure** (the model doesn't know today's date), draws the **wrong mental model next to the right one**, then fixes it | Types for structured output; how tools really work | Whether the answers are right: no measurement anywhere; cost appears only as a token table |
| **mall** (Posit/mlverse) | Plain, pragmatic | What it does, then install and setup | One function per NLP task, piped into dplyr | Rows of text through an LLM, inside dplyr | "Key considerations" says to "always check the output", but never shows how; no accuracy, no baseline, no intervals |
| **fasteR** (Matloff) | Personal, insistent | "Nonpassive learning is absolutely key!" | "Your Turn" after every lesson; "When in doubt, try it out" | Getting productive fast | Modern tooling |
| **Data Carpentry, R for Ecologists** | Workshop-plain | Who it's for, what it assumes, how long it takes | Episodes with objectives, callouts, challenges, key points | Setting expectations; data from the learner's field | Depth |
| **STAT 545** (Bryan) | Candid, a colleague | States what it covers **and what it doesn't** ("everything… except statistical modelling") | Project habits taught alongside code | The unglamorous work that decides quality | Modelling, deliberately |
| **Teacups, Giraffes and Statistics** | Whimsical | A story world | Illustrated, interactive (learnr) | Making abstract ideas memorable | Scale |
| **Epidemiologist R Handbook** | Task-first reference | Why R, key terms | Find the task, copy the pattern | The learner's own domain | Narrative |

## What to take from them

1. **Open with a question about real data, and show where you'll end up.**
   R4DS's "ultimate goal" and Silge's "our modeling goal" make every later
   step make sense. Each tutorial opens with a situation (a shop, a bird
   survey, a refund desk) and the result the reader will produce.
2. **Build it in small steps, and read every output aloud.** A warning, an
   `NA`, a failed row is a lesson, the way R4DS treats "Removed 2 rows".
3. **Show the trap before the fix.** tidymodels' "What happened here?" and
   ellmer's wrong date are the moments readers remember. The traps here:
   the model not knowing the house rules, a description tuned on the test
   set, two accuracies compared without intervals, a model sure of an
   answer the rules forbid.
4. **Always a baseline.** "Compare to the null model" (SMLTAR). A keyword
   rule or "always the most common answer" keeps a language model honest.
5. **Measure, with intervals, every time.** This is exactly what the LLM
   tutorials (ellmer, mall) leave out, and it is FunctAI's reason to exist.
   Every tutorial ends knowing how right, how sure, and how much it cost.
6. **Say what it costs, in dollars.** mall's "key considerations" name
   cost without counting it. Each tutorial ends with its own bill, from
   the call log.
7. **Respect the reader's time.** A three-question check at the top of the
   later tutorials (Advanced R) with the answers at the bottom; "what you
   need" up front (Data Carpentry); "what this doesn't cover" (STAT 545).
8. **Make the reader do something.** "Your turn" (fasteR, R4DS exercises)
   after each tutorial: two or three small tasks that reuse what is on the
   page, without new money.
9. **One mental model per tutorial.** An AI function is a function whose
   body is a model; a type is a promise the answer keeps; a score is a
   proportion with an interval; a decision is a choice with costs.
10. **End with the whole game.** tidymodels closes with a case study that
    uses everything; the last tutorial here does the same with the life of
    a function in use.

## What they gloss over, which this series does not

- **Whether the model is right** (ellmer, mall): every tutorial measures.
- **What a mistake costs** (tidymodels, SMLTAR stop at accuracy and
  precision/recall): the decision tutorial prices each kind of mistake and
  chooses actions, thresholds and when to ask a person by expected cost.
- **Run-to-run variation**: recent reasoning models take no temperature,
  so the same question can get another answer; tutorial 3 measures it.
- **Choosing a model by accuracy, cost and speed together**, with paired
  tests rather than eyeballed averages (tutorial 5).

## The series

Eight tutorials, each runnable top to bottom in a fresh R session
(`r/tutorials` runs them and writes their outputs), each under $2 of model
calls (each prints its own bill), all on models current in September 2026:
`gpt-6-luna` as the everyday model, `gpt-6-sol`, `claude-sonnet-5`,
`claude-haiku-4-5`, `gemini-3.8-flash` and `gemini-3.1-flash-lite` where a
comparison or a teacher needs them.

| # | Tutorial | The question | The mental model |
|---|---|---|---|
| 1 | Your first AI function | Which team should answer each of 80 messages? | A function whose body is a model |
| 2 | Answers you can compute with | Can 60 field notes become a table you can plot? | A type is a promise the answer keeps |
| 3 | Is it right? | How often, how sure, compared with what? | A score is a proportion with an interval |
| 4 | Making it better without fooling yourself | Rules, examples, reasoning: which helps? | The test set is used once |
| 5 | Choosing a model | Which model is good enough for the least money? | Accuracy, cost and speed together, compared in pairs |
| 6 | Decision models | Approve, deny, or ask a person? | A decision is a choice with costs |
| 7 | AI functions in tidymodels | Does the language model fit the tools I know? | Specify, fit, predict, score, for any model |
| 8 | Living with it | Tools, the log, corrections, saving | A function's life after the notebook |

Datasets: `tickets` and `field_notes` (as in Python), and `refunds`,
written for tutorial 6: 120 refund requests whose right decision follows
from written rules and facts, so a decision model can be scored exactly.
