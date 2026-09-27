# 4. Making it better without fooling yourself

*A refund desk, a function that decides, and three ways to improve it: write the rules down, show it examples, let a stronger model teach it; and a fourth for when nobody wrote the rules down: let a stronger model write them from the mistakes (GEPA). By the end you will know which helped, by how much, and how sure you can be, because you will have kept one set of rows aside for the honest number.*

**Can you skip this one?** If you can answer these, jump to [tutorial 5](05-choosing-a-model.md). The answers are at the bottom.

1. You try five versions of a description and keep the one that scores best on your test rows. Why is its score too optimistic?
2. Two versions are right on 36 and 38 of the same 40 rows. What should you count to know if the difference is real?
3. What does `labeled_few_shot` change in a function, and what does it leave alone?

**You will:** split rows into three piles, try rules, worked examples (`labeled_few_shot`) and a teacher (`bootstrap_few_shot`), compare versions on the same rows by their disagreements (`compare`, McNemar's test), see the winner's curse in a simulation, and let a stronger model rewrite a small model's instruction from its mistakes (`gepa`).

## The refund desk

The homeware shop from tutorial 1 gets refund requests too. For each one, someone decides: **approve** or **deny**. `FunctAI.refunds()` has 120 of them: what the customer wrote, what the order system knows (price, days since delivery, whether it was a final-sale item), and the decision the shop's rules give.

```julia
using FunctAI, DataFrames, Statistics, Random, HypothesisTests

log_folder = mktempdir()
FunctAI.configure!(lm = "gpt-6-luna", log_calls = log_folder)

refunds = DataFrame(FunctAI.refunds())
select(refunds, :message, :price, :days_since_delivery, :final_sale, :decision)
```

```output
120×5 DataFrame
 Row │ message                            price    days_since_delivery  final_sale  decisi ⋯
     │ String                             Float64  Int64                Bool        String ⋯
─────┼──────────────────────────────────────────────────────────────────────────────────────
   1 │ Hiya, just a heads up - the wall…    28.51                   66       false  deny   ⋯
   2 │ Ordered the grey rug, got sent g…    39.22                   28       false  approv
   3 │ Hiya, so I'd only made about thr…     9.38                    9       false  approv
   4 │ Hi, I bought a planter from you …   139.47                   57       false  approv
   5 │ I'm sorry to even ask this after…    75.9                   368       false  deny   ⋯
   6 │ I need a refund for the chair I …   388.71                   17       false  approv
   7 │ Hello, I ordered a casserole dis…   458.27                   26       false  approv
   8 │ It's been four weeks since this …    11.63                   28        true  approv
  ⋮  │                 ⋮                     ⋮              ⋮               ⋮          ⋮   ⋱
 114 │ Hi, I'm so sorry to bring this u…    22.99                   24       false  approv ⋯
 115 │ Hi, when I opened the box about …    20.84                   24       false  approv
 116 │ Hi, I wanted to reach out about …    31.9                    12       false  approv
 117 │ I opened the parcel about three …    24.66                   22       false  approv
 118 │ hiya, i got this a lil over 3 we…    45.43                   23       false  approv ⋯
 119 │ Oh gosh, so this thing turned up…    20.44                   33       false  approv
 120 │ Hiya, so this arrived ages ago (…    24.89                   62       false  deny
                                                               1 column and 105 rows omitted
```

```julia
refunds.message[5]
```

```output
"I'm sorry to even ask this after so long, it's been about a year since these arrived, I've been using them regularly, but the colour just never really settled in with the rest of my bathroom and I keep reaching for a different set instead, so I was hoping a refund might still be possible?\n\nPriya"
```

A function that decides takes the message *and* the facts, each with its type:

```julia
@enum Decision approve deny

@ai function refund(message::String, price::Float64, days_since_delivery::Int, final_sale::Bool)
    "Should the shop refund this request?"
    decision::Decision = ai"approve or deny"
end
```

```output
AI function refund(message::String, price::Float64, days_since_delivery::Int64, final_sale::Bool) -> Decision
  model:        gpt-6-luna
  instruction:  Should the shop refund this request?

                Output guidance:
                - decision: approve or deny
  version:      sha256:f3da6b9661ae…
  see:          FunctAI.prompt(refund, …) for the exact request
```

Naming the answer `decision`, like the column that holds the right answers, lets `evaluate` and the optimizers find it without being told. The inputs are found the same way: each row's `message`, `price`, `days_since_delivery` and `final_sale`.

## Three piles of rows, used for three different things

Every change you try is a guess. To know if a guess helped, you measure it. The trap is measuring every guess on the same rows and keeping the winner: you end up choosing the version that got lucky on those rows, and its score flatters it. (There's a simulation of exactly this below.)

So split the 120 requests three ways, with the same mix of decisions in each:

- **examples** (about 40): rows the function may learn from (worked examples);
- **dev** (about 40): rows you measure your guesses on, as often as you like;
- **test** (about 40): rows you look at **once**, at the very end, for the version you chose.

```julia
rng = Xoshiro(2026)
refunds.pile = fill(:none, nrow(refunds))
for g in groupby(refunds, :decision)               # the same mix of decisions in each pile
    g.pile[shuffle(rng, 1:nrow(g))] = repeat([:examples, :dev, :test], outer = cld(nrow(g), 3))[1:nrow(g)]
end
examples, dev, test = (filter(:pile => ==(p), refunds) for p in (:examples, :dev, :test))

(examples = nrow(examples), dev = nrow(dev), test = nrow(test))
```

```output
(examples = 41, dev = 40, test = 39)
```

## Where we start

```julia
ev_plain = evaluate(refund, dev)
```

```output
Evaluation of refund on 40 rows
  exact_match  0.93  (95% range 0.80 to 0.97)
  every row: DataFrame(e)
```

The model has never seen the shop's rules, yet it's right most of the time: it knows what refund policies usually say. What's it getting wrong? Look at the dev rows (never the test rows):

```julia
missed = filter(:exact_match => ==(0), DataFrame(ev_plain))
leftjoin(select(missed, :id, :pred_decision), select(dev, :id, :state, :days_since_delivery, :final_sale, :price, :decision), on = :id)
```

```output
3×7 DataFrame
 Row │ id     pred_decision  state    days_since_delivery  final_sale  price     decision
     │ Int64  Decision       String?  Int64?               Bool?       Float64?  String?
─────┼────────────────────────────────────────────────────────────────────────────────────
   1 │    62  deny           faulty                    71       false     15.76  approve
   2 │    70  approve        used                      20       false     14.77  deny
   3 │    75  approve        used                      11       false     66.37  deny
```

(`state` is the item's true condition, which the shop's staff recorded. The function doesn't see it; we do, to understand the mistakes.)

Read the misses by their state and their days since delivery. The model assumed the most common policy, a 30-day window for everything and final sale meaning final. This shop is more generous with damage and faults, and nothing told the model so.

## 1. Write the rules down

The rules are in `?FunctAI.refunds`. They're the shop's policy, not something we invented by staring at the mistakes, which matters: rules written to fix particular dev rows would be fitted to those rows.

```julia
@ai function refund_rules(message::String, price::Float64, days_since_delivery::Int, final_sale::Bool)
    """
    Should the shop refund this request? Follow the refund rules exactly:
    - Damaged on arrival, or the wrong item (or part of the order missing): refund within 60 days
      of delivery, final sale or not.
    - Faulty (it failed in normal use): refund within 365 days, final sale or not.
    - Unopened, or opened but not used, and no longer wanted: refund within 30 days, never for a
      final-sale item.
    - Used and no longer wanted: no refund.
    """
    decision::Decision = ai"approve or deny"
end

ev_rules = evaluate(refund_rules, dev)
```

```output
Evaluation of refund_rules on 40 rows
  exact_match  1.00  (95% range 0.91 to 1.00)
  every row: DataFrame(e)
```

## 2. Show it worked examples

A new colleague learns from rules, and also from seeing past cases. `labeled_few_shot` picks rows with known answers and puts them in front of every question as solved examples. It reads each example's inputs and answer from the columns named like them:

```julia
refund_shown = labeled_few_shot(refund_rules, examples; k = 8)

ev_shown = evaluate(refund_shown, dev)
```

```output
Evaluation of refund_rules on 40 rows
  exact_match  1.00  (95% range 0.91 to 1.00)
  every row: DataFrame(e)
```

`labeled_few_shot` returns a *new* function; `refund_rules` is unchanged (no `!`, so nothing is modified: Julia's convention). Each version has its own `version`, a fingerprint of everything it sends besides the inputs (the instruction, the layout, the examples). The call log files every call under it, so later you can tell which version gave which answer:

```julia
(plain = version(refund), rules = version(refund_rules), shown = version(refund_shown))
```

```output
(plain = "sha256:f3da6b9661aecb0ba374edf8d81a1cb16045e48fa554ba1e87766b70cae2e454", rules = "sha256:9115960c6a7ed39f2762171e88d79b33ef557e238962e7571667fc755d8b27e2", shown = "sha256:d063378b602871d028af66981fff83851bd2a0fc06fbdf831b20e557744238a6")
```

## 3. Let a stronger model teach

Labelled rows show the answer, not the thinking. `bootstrap_few_shot` runs a **teacher** on the example rows, keeps the runs whose answer was right, and uses those as the worked examples. Here the teacher is `gpt-6-sol`, OpenAI's larger current model: twenty times the price per token of `gpt-6-luna`, but it only answers a handful of rows, once.

```julia
refund_taught = bootstrap_few_shot(refund_rules, examples; teacher = "gpt-6-sol", max_bootstrapped = 4, max_labeled = 4)

ev_taught = evaluate(refund_taught, dev)
```

```output
Evaluation of refund_rules on 40 rows
  exact_match  1.00  (95% range 0.91 to 1.00)
  every row: DataFrame(e)
```

(A fourth lever, `configure(f; reasoning = true)`, asks the model to write its reasoning before answering. Current models like `gpt-6-luna` already think before they answer, so it changes little here; it helps older and smaller models that don't.)

## Which helped?

All four on the dev rows, with their intervals:

```julia
versions = ["1. no rules" => ev_plain, "2. the rules" => ev_rules,
            "3. rules + 8 examples" => ev_shown, "4. rules + taught by gpt-6-sol" => ev_taught]
DataFrame([(version = v, score = e.score, low = e.low, high = e.high) for (v, e) in versions])
```

```output
4×4 DataFrame
 Row │ version                         score    low       high
     │ String                          Float64  Float64   Float64
─────┼─────────────────────────────────────────────────────────────
   1 │ 1. no rules                       0.925  0.801358  0.974164
   2 │ 2. the rules                      1.0    0.912378  1.0
   3 │ 3. rules + 8 examples             1.0    0.912378  1.0
   4 │ 4. rules + taught by gpt-6-sol    1.0    0.912378  1.0
```

The intervals are wide: forty rows can't separate versions a few points apart. But these versions were run on the *same* forty rows, and that gives a much sharper test. Only the rows where two versions **disagree** carry information about which is better. Count them:

```julia
right(e) = FunctAI.scores(e) .== 1
disagree = [count(right(ev_plain) .& right(ev_rules))   count(right(ev_plain) .& .!right(ev_rules));
            count(.!right(ev_plain) .& right(ev_rules)) count(.!right(ev_plain) .& .!right(ev_rules))]
```

```output
2×2 Matrix{Int64}:
 37  0
  3  0
```

Rows are the plain version (right, wrong), columns the rules (right, wrong). The diagonal (both right, both wrong) says nothing about which is better. The two other cells are the evidence: rows the plain version got wrong and the rules got right, and the other way round. If the rules made no difference, each disagreement would be a coin flip. So the question is how surprising the split of the disagreements is, for a fair coin: a binomial test on them, which is McNemar's exact test:

```julia
fixed, broken = disagree[2, 1], disagree[1, 2]      # wrong before and right after, and the other way round
pvalue(BinomialTest(fixed, fixed + broken))
```

```output
0.25
```

`compare` does the same pairing for every metric two evaluations share: the difference, with a 95% interval, and how many rows got better, worse, or stayed the same:

```julia
select(DataFrame(compare(ev_plain, ev_rules)), :before, :after, :diff, :low, :high, :better, :worse)
```

```output
1×7 DataFrame
 Row │ before   after    diff     low         high      better  worse
     │ Float64  Float64  Float64  Float64     Float64   Int64   Int64
─────┼────────────────────────────────────────────────────────────────
   1 │   0.925      1.0    0.075  -0.0103079  0.160308       3      0
```

Read the p-value with the counts in mind. A lopsided split on few disagreements can still be luck, and the test says so. The honest summary is "the rules fixed most of the mistakes we saw; on forty rows that's suggestive, not proof". More dev rows would settle it. So does knowing *why* it helped, which we do: the rules are the shop's policy.

## The trap, simulated

Why not just pick the best dev score and report it? Suppose you tried five versions that are all, truly, right 90% of the time, and scored each on 40 rows. Simulate it, for free:

```julia
score_on_40(rng) = count(<(0.9), rand(rng, 40)) / 40         # one version, truly right 90% of the time
best_of_five(rng) = maximum(score_on_40(rng) for _ in 1:5)

mean(best_of_five(Xoshiro(i)) for i in 1:10_000)
```

```output
0.9512775000000393
```

Every version is 90%, yet the winner scores about 95% on average, just by being the luckiest of five. The more versions you try on the same rows, the bigger the flattery. That's why the test rows exist.

## Once, at the end

Choose on dev. When versions tie, choose the **simplest**: the rules alone. Worked examples make every call longer and so dearer; when they buy nothing measurable, the cheaper, simpler thing wins. Now, once, the test rows:

```julia
ev_final = evaluate(refund_rules, test)
```

```output
Evaluation of refund_rules on 39 rows
  exact_match  1.00  (95% range 0.91 to 1.00)
  every row: DataFrame(e)
```

That is the number to report. When it's lower than on dev, as it often is, that's the flattery leaving, not a failure.

## When nobody wrote the rules down

We could write the rules because the shop had them. Often nobody has: there are only past decisions, and a model that gets some of them wrong. And the model you can afford to run on every request may be a small one. Here is `refund` on `gpt-5.4-nano`, the small model of six months ago, without the rules:

```julia
refund_nano = configure(refund; lm = "gpt-5.4-nano");
```

`gepa` improves the instruction by reading the mistakes. It runs the function on a few example rows, shows a stronger model (the **teacher**) the answers with a word of feedback on each ("wrong: the right answer is approve"), and asks it for a better instruction. Each new instruction is tried on the same few rows; one that does better is scored on the rows you choose with, and joins a pool of candidates. The teacher then works on the candidates that are best on at least one row, so ideas that fix *different* mistakes both survive, and every few steps it combines two of them. It stops at a budget of calls, and keeps the best on the choosing rows; of equally good ones, the shorter.

The three piles fit it exactly: it learns from `examples`, chooses on `dev`, and never sees `test`. It returns the improved copy and every instruction it tried:

```julia
refund_gepa, trials = gepa(refund_nano, examples; selection = dev, teacher = "gpt-6-sol", budget = 300)

select(DataFrame(trials), :candidate, :kind, :minibatch_parent => :parent, :minibatch, :score, :length, :note)
```

```output
8×7 DataFrame
 Row │ candidate  kind     parent   minibatch  score        length  note                   ⋯
     │ Int64?     Symbol   Int64?   Int64?     Float64?     Int64   String                 ⋯
─────┼──────────────────────────────────────────────────────────────────────────────────────
   1 │         1  written  missing    missing        0.775     100  the written instructio ⋯
   2 │   missing  reflect        2          2  missing         799  not better on the mini
   3 │         2  reflect        2          4        0.85      862  joined the pool
   4 │         3  combine        3          3        0.85      686  joined the pool
   5 │         4  combine        4          4        0.825     611  joined the pool        ⋯
   6 │   missing  reflect        3          3  missing         651  not better on the mini
   7 │   missing  combine        4          3  missing         716  worse on the minibatch
   8 │         5  reflect        1          4        0.8       806  joined the pool
                                                                            1 column omitted
```

Each row is an instruction the teacher wrote (`candidate` 1 is the one you wrote). `parent` (the search's `minibatch_parent`) and `minibatch` are how many of the same few example rows its parent and it got right. Those that did better joined the pool and were scored on dev (`score`); the others were dropped, and the note says why. The chosen one is the best on dev; of equally good ones, the shorter (`length`, in characters), because every call pays for its length:

```julia
println(FunctAI.instructions(refund_gepa))
```

```output
Function: refund

Decide whether the shop should refund the request. Output only `decision`: `approve` or `deny`.

Approve a non-final-sale request made within 30 days of delivery if the message describes either a change of mind about an unused item or damage or a functional defect. Damage or a defect can qualify even if the item was opened or used normally before it failed. Deny final-sale requests, requests made after 30 days, and preference-based returns of used items. Use `days_since_delivery` for the deadline, regardless of approximate timing in the message or when the package was opened. Do not base the decision on price, writing style, or the customer’s preferred remedy.
```

Read it as you'd read a fitted model's coefficients: it is what the search learned from the mistakes, and you can check it against `?FunctAI.refunds`. It found part of the policy from nothing but "the right answer is approve": a change of mind is refunded only within 30 days and only when unused, and damage or a fault qualifies even when the item was used. It also wrote two rules the shop doesn't have: a 30-day limit for everything (the shop allows 60 days for damage and a year for a fault) and no refund on final sale ever (the shop refunds damage and faults on final-sale items). Both are the common policy the small model already assumed; a search learns only from the mistakes its minibatches happen to show, and a model that already denies a late claim makes no mistake there to learn from.

Its dev score was the best of many on dev, so it may flatter (the trap simulated above, and this time the search did the trying). The test rows give the honest number, once, for this question:

```julia
ev_nano = evaluate(refund_nano, test)
ev_nano_gepa = evaluate(refund_gepa, test)
select(DataFrame(compare(ev_nano, ev_nano_gepa)), :before, :after, :diff, :low, :high, :better, :worse)
```

```output
1×7 DataFrame
 Row │ before    after     diff      low         high      better  worse
     │ Float64   Float64   Float64   Float64     Float64   Int64   Int64
─────┼───────────────────────────────────────────────────────────────────
   1 │ 0.666667  0.794872  0.128205  -0.0844808  0.340891      11      6
```

Read the difference with its interval, as always. It fixed eleven rows and broke six, and the interval includes zero: on forty rows, this gain could be luck. And the search is itself random (which rows it learns from, in which order, what the teacher writes): another run of this page gave 64% to 95%, thirteen fixed and one broken, an interval well clear of zero. That spread is the real lesson. One search is one draw; before you ship its instruction, measure it on rows it never saw, and when the number matters, run the search more than once, or with a larger budget, and measure again.

`instruction_search` is the other way to let a stronger model write the instruction: it proposes instructions from a few solved examples and tries them on minibatches, without ever seeing what the function got wrong. `gepa` reads the mistakes, with words about each, which tells the teacher far more; try it first.

A small model with an instruction written by a large one, once. The large model's price is paid for a handful of calls; the small model's for every request, forever. It will still make mistakes the written rules don't, because it learned part of the policy, not all of it. When the rules are yours to write, write them: they are exact, and you know why they work. When they aren't, this is how to find a good share of them, and measure what you found.

## What it cost

```julia
prices = DataFrame(model  = ["gpt-6-luna", "gpt-5.4-nano", "gpt-6-sol"],   # dollars per million tokens, 2026-09-27
                   input  = [0.10, 0.20, 2.00],
                   output = [0.50, 1.25, 10.00])

bill = leftjoin(DataFrame(calls(folder = log_folder)), prices, on = :model)
bill.dollars = (bill.input_tokens .* bill.input .+ (bill.total_tokens .- bill.input_tokens) .* bill.output) ./ 1e6
combine(groupby(bill, :model), nrow => :calls, :dollars => sum => :dollars)
```

```output
3×3 DataFrame
 Row │ model         calls  dollars
     │ String        Int64  Float64
─────┼────────────────────────────────
   1 │ gpt-6-luna      199  0.0150645
   2 │ gpt-6-sol        14  0.044958
   3 │ gpt-5.4-nano    378  0.0237142
```

Worked examples make every question longer (each call now carries its solved cases), so they cost more per call. On a small model that's still cents. Weigh it anyway: it's paid on every call, forever.

`gepa` was most of this bill: the small model's calls and the teacher's, paid once. Its instruction is also longer than the one-line original, so every call after it costs a little more; that is the trade it made for accuracy, and why ties go to the shorter instruction.

## Your turn

1. Try `k = 16` instead of 8 in `labeled_few_shot`. Measure it on dev, and compare it with the rules-only version using `compare` and the disagreement counts.
2. Add one sentence to the rules that you think would fix a dev mistake. Is it policy, or is it fitted to that row? How could you tell?
3. Give `gepa` a `feedback` function that says *why* a decision was wrong, using the `state` column: `feedback = (row, out, err) -> "the item was $(row.state), $(row.days_since_delivery) days after delivery; the right decision is $(row.decision)"`. Does the teacher find more of the rules? Is that fair, when `state` is something a person had to read?
4. `FunctAI.prompt(refund_taught, "x", 1.0, 1, false)` shows the whole request. Find the taught examples in it, and count how much longer it is than `refund_rules`'s.

## What you learned

- Split once, before you start: rows to learn from, rows to choose on, rows to test once.
- Writing the rules down is the most direct improvement, when the rules are yours to write.
- `labeled_few_shot` adds solved examples; `bootstrap_few_shot` adds a teacher's runs that were right. Both return a new function with a new `version`; the one you passed is unchanged.
- `gepa` has a stronger model rewrite the instruction from the mistakes: learn on one pile, choose on another, measure on a third. Its second result shows the search, every instruction tried.
- On the same rows, compare versions by their disagreements (`compare`, McNemar's test), not by eyeballing two intervals.
- When versions tie, keep the simplest and cheapest.
- Picking the best of several on the same rows flatters the winner. The test rows, used once, give the honest number.

**Answers to the check at the top.** (1) It was chosen for being the luckiest on those rows, so part of its score is luck that won't come back: the winner's curse. (2) The rows where they disagree: how many one got right and the other wrong, each way (`compare`, or McNemar's test). (3) It adds worked examples to the request; the instruction, inputs, outputs and model stay as they were, and the function passed in is unchanged.

**Next:** [5. Choosing a model](05-choosing-a-model.md) compares eight models on accuracy, cost and speed.
