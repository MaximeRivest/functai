# 4. Making it better without fooling yourself

*A refund desk, a function that decides, and three ways to improve it: write the rules down, show it examples, let a stronger model teach it; and a fourth for when nobody wrote the rules down: let a stronger model write them from the mistakes. By the end you will know which helped, by how much, and how sure you can be, because you will have kept one set of rows aside for the honest number.*

**Can you skip this one?** If you can answer these, jump to [tutorial 5](05-choosing-a-model.md). The answers are at the bottom.

1. You try five versions of a description and keep the one that scores best on your test rows. Why is its score too optimistic?
2. Two versions are right on 36 and 38 of the same 40 rows. What should you count to know if the difference is real?
3. What does `labeled_few_shot()` change in a function, and what does it leave alone?

## The refund desk

The homeware shop from tutorial 1 gets refund requests too. For each one, someone decides: **approve** or **deny**. `refunds` has 120 of them: what the customer wrote, what the order system knows (price, days since delivery, whether it was a final-sale item), and the decision the shop's rules give.

```r
library(functai)
library(dplyr)
library(rsample)

log_folder <- tempfile("functai-calls-")
ai_config(lm = "gpt-6-luna", log_calls = log_folder)

refunds |> select(message, price, days_since_delivery, final_sale, decision)
```

```output
# A tibble: 120 × 5
   message                         price days_since_delivery final_sale decision
   <chr>                           <dbl>               <int> <lgl>      <fct>   
 1 "Hiya, just a heads up - the …  28.5                   66 FALSE      deny    
 2 "Ordered the grey rug, got se…  39.2                   28 FALSE      approve 
 3 "Hiya, so I'd only made about…   9.38                   9 FALSE      approve 
 4 "Hi, I bought a planter from … 139.                    57 FALSE      approve 
 5 "I'm sorry to even ask this a…  75.9                  368 FALSE      deny    
 6 "I need a refund for the chai… 389.                    17 FALSE      approve 
 7 "Hello, I ordered a casserole… 458.                    26 FALSE      approve 
 8 "It's been four weeks since t…  11.6                   28 TRUE       approve 
 9 "I'm writing about the pair o… 129.                    25 FALSE      approve 
10 "Hi! So this ended up still b…  50.0                   94 FALSE      deny    
# ℹ 110 more rows
```

```r
refunds$message[5]
```

```output
[1] "I'm sorry to even ask this after so long, it's been about a year since these arrived, I've been using them regularly, but the colour just never really settled in with the rest of my bathroom and I keep reaching for a different set instead, so I was hoping a refund might still be possible?\n\nPriya"
```

A function that decides takes the message *and* the facts. Write it the way you'd write a model of the decision, with the table the columns come from:

```r
refund <- ai(decision ~ message + price + days_since_delivery + final_sale,
  "Should the shop refund this request?",
  .data = refunds, .name = "refund")
```

`.data = refunds` works as it does in `lm()`: each field takes the type of its column, so `price` is a number, `days_since_delivery` a whole number, `final_sale` yes or no, and `decision`, a factor, is a choice between `approve` and `deny`. Only the column types are read; no row is sent anywhere. `.name` calls the function `refund` (it would otherwise be named after its output, `decision`).

## Three piles of rows, used for three different things

Every change you try is a guess. To know if a guess helped, you measure it. The trap is measuring every guess on the same rows and keeping the winner: you end up choosing the version that got lucky on those rows, and its score flatters it. (There's a simulation of exactly this below.)

So split the 120 requests three ways, with the same mix of decisions in each (`strata`):

- **examples** (about 40): rows the function may learn from (worked examples);
- **dev** (about 40): rows you measure your guesses on, as often as you like;
- **test** (about 40): rows you look at **once**, at the very end, for the version you chose.

```r
set.seed(2026)
first <- initial_split(refunds, prop = 1/3, strata = decision)
examples <- training(first)
rest <- initial_split(testing(first), prop = 1/2, strata = decision)
dev <- training(rest)
test <- testing(rest)

c(examples = nrow(examples), dev = nrow(dev), test = nrow(test))
```

```output
examples      dev     test 
      39       40       41 
```

## Where we start

```r
ev_plain <- evaluate(refund, dev)
ev_plain
```

```output
<evaluation of refund> 40 rows
  exact_match: 0.88  (95% interval 0.74 to 0.95)
```

The formula said the answer is `decision`, so `evaluate()` compares with the `decision` column. (When the right answers are in a column of another name, say so: `expected = category`.)

The model has never seen the shop's rules, yet it's right most of the time: it knows what refund policies usually say. What's it getting wrong? Look at the dev rows (never the test rows):

```r
augment(ev_plain) |>
  filter(exact_match == 0) |>
  select(decision, .pred_class, days_since_delivery, final_sale, state, message)
```

```output
# A tibble: 5 × 6
  decision .pred_class days_since_delivery final_sale state   message           
  <fct>    <fct>                     <int> <lgl>      <fct>   <chr>             
1 approve  deny                         53 FALSE      damaged "I am writing to …
2 approve  deny                         27 TRUE       faulty  "Hello, I hope yo…
3 approve  deny                         34 FALSE      faulty  "The drawer runne…
4 approve  deny                         64 FALSE      faulty  "I bought this ov…
5 deny     approve                      11 FALSE      used    "I got this about…
```

(`state` is the item's true condition, which the shop's staff recorded. The function doesn't see it; we do, to understand the mistakes.)

Most of the misses deny a refund the shop would give: damage reported after seven or eight weeks, a fault after a month or two, a fault on a final-sale item. The model assumed the most common policy, a 30-day window for everything, and final sale meaning final. This shop is more generous with damage and faults, and nothing told the model so.

## 1. Write the rules down

The rules are in `?refunds`. They're the shop's policy, not something we invented by staring at the mistakes, which matters: rules written to fix particular dev rows would be fitted to those rows.

```r
refund_rules <- ai(decision ~ message + price + days_since_delivery + final_sale,
  "Should the shop refund this request? Follow the refund rules exactly:
  - Damaged on arrival, or the wrong item (or part of the order missing): refund within 60 days
    of delivery, final sale or not.
  - Faulty (it failed in normal use): refund within 365 days, final sale or not.
  - Unopened, or opened but not used, and no longer wanted: refund within 30 days, never for a
    final-sale item.
  - Used and no longer wanted: no refund.",
  .data = refunds, .name = "refund")

ev_rules <- evaluate(refund_rules, dev)
ev_rules
```

```output
<evaluation of refund> 40 rows
  exact_match: 1.00  (95% interval 0.91 to 1.00)
```

## 2. Show it worked examples

A new colleague learns from rules, and also from seeing past cases. `labeled_few_shot()` picks rows with known answers and puts them in front of every question as solved examples. It takes each example's answer from the column the formula names, `decision`:

```r
refund_shown <- refund_rules |> labeled_few_shot(examples, k = 8)

ev_shown <- evaluate(refund_shown, dev)
ev_shown
```

```output
<evaluation of refund> 40 rows
  exact_match: 1.00  (95% interval 0.91 to 1.00)
```

`labeled_few_shot()` returns a *new* function; `refund_rules` is unchanged. Each version has its own `ai_version()`, a fingerprint of everything it sends besides the inputs (the instruction, the layout, the examples). The call log files every call under it, so later you can tell which version gave which answer:

```r
c(plain = ai_version(refund), rules = ai_version(refund_rules), shown = ai_version(refund_shown))
```

```output
                                                                    plain 
"sha256:519d420f6444c4c02c12f66c84f6fd9b69a757874b3756c913f2cd18b5432e4b" 
                                                                    rules 
"sha256:a2c10c409bce3179d6426d0f98a31231675e3436b0b7fef2ff9ecda494b40902" 
                                                                    shown 
"sha256:40021545a87b3a0db366afa4754563b6e095d036c0006a481b04fef06454713c" 
```

## 3. Let a stronger model teach

Labelled rows show the answer, not the thinking. `bootstrap_few_shot()` runs a **teacher** on the example rows, keeps the runs whose answer was right, and uses those as the worked examples. Here the teacher is `gpt-6-sol`, OpenAI's larger current model: twenty times the price per token of `gpt-6-luna`, but it only answers a handful of rows, once.

```r
refund_taught <- refund_rules |>
  bootstrap_few_shot(examples, teacher = "gpt-6-sol", max_bootstrapped = 4, max_labeled = 4)

ev_taught <- evaluate(refund_taught, dev)
ev_taught
```

```output
<evaluation of refund> 40 rows
  exact_match: 1.00  (95% interval 0.91 to 1.00)
```

(A fourth lever, `update(fn, module = "cot")`, asks the model to reason before answering. Current models like `gpt-6-luna` already think before they answer, so it changes little here; it helps older and smaller models that don't.)

## Which helped?

All four on the dev rows, with their intervals:

```r
dev_scores <- bind_rows(
  tidy(ev_plain)  |> mutate(version = "1. no rules"),
  tidy(ev_rules)  |> mutate(version = "2. the rules"),
  tidy(ev_shown)  |> mutate(version = "3. rules + 8 examples"),
  tidy(ev_taught) |> mutate(version = "4. rules + taught by gpt-6-sol")
)
dev_scores |> select(version, estimate, conf.low, conf.high)
```

```output
# A tibble: 4 × 4
  version                        estimate conf.low conf.high
  <chr>                             <dbl>    <dbl>     <dbl>
1 1. no rules                       0.875    0.739     0.945
2 2. the rules                      1        0.912     1    
3 3. rules + 8 examples             1        0.912     1    
4 4. rules + taught by gpt-6-sol    1        0.912     1    
```

The intervals are wide: forty rows can't separate versions a few points apart. But these four versions were run on the *same* forty rows, and that gives a much sharper test. Only the rows where two versions **disagree** carry information about which is better. Count them:

```r
right_or_wrong <- function(ev) factor(augment(ev)$exact_match, levels = c(0, 1), labels = c("wrong", "right"))

pairs <- table(no_rules = right_or_wrong(ev_plain), rules = right_or_wrong(ev_rules))
pairs
```

```output
        rules
no_rules wrong right
   wrong     0     5
   right     0    35
```

The diagonal (both right, both wrong) says nothing about which is better. The two off-diagonal cells are the evidence: rows the plain version got wrong and the rules got right, and the other way round. If the rules made no difference, each disagreement would be a coin flip; McNemar's test asks how surprising the split is:

```r
mcnemar.test(pairs)
```

```output

	McNemar's Chi-squared test with continuity correction

data:  pairs
McNemar's chi-squared = 3.2, df = 1, p-value = 0.07364
```

Read the p-value with the counts in mind. Every disagreement went the rules' way, but there are only a handful of them, and the test says so: a split that lopsided, on so few rows, could still be luck. The honest summary is "the rules fixed every mistake we saw; on forty rows that's suggestive, not proof". More dev rows would settle it. So does knowing *why* it helped, which we do: the rules are the shop's policy.

The last two versions add nothing measurable on top of the rules: there was nothing left to fix on these forty rows. That's a result too.

## The trap, simulated

Why not just pick the best dev score and report it? Suppose you tried five versions that are all, truly, right 90% of the time, and scored each on 40 rows. Simulate it, for free:

```r
set.seed(1)
best_of_five <- replicate(10000, max(rbinom(5, size = 40, prob = 0.9)) / 40)
mean(best_of_five)
```

```output
[1] 0.9517475
```

Every version is 90%, yet the winner scores about 95% on average, just by being the luckiest of five. The more versions you try on the same rows, the bigger the flattery. That's why the test rows exist.

## Once, at the end

Choose on dev. Three versions tie at the top, so choose the **simplest**: the rules alone. Worked examples make every call longer and so dearer, and they bought nothing we can measure. When results tie, the cheaper, simpler thing wins. Now, once, the test rows:

```r
ev_final <- evaluate(refund_rules, test)
ev_final
```

```output
<evaluation of refund> 41 rows
  exact_match: 0.98  (95% interval 0.87 to 1.00)
```

That is the number to report. When it's lower than on dev, as it often is, that's the flattery leaving, not a failure.

## When nobody wrote the rules down

We could write the rules because the shop had them. Often nobody has: there are only past decisions, and a model that gets some of them wrong. And the model you can afford to run on every request may be a small one. Here is `refund` on `gpt-5.4-nano`, the small model of six months ago, without the rules:

```r
refund_nano <- update(refund, lm = "gpt-5.4-nano")
```

`gepa()` improves the instruction by reading the mistakes. It runs the function on a few example rows, shows a stronger model (the **teacher**) the answers with a word of feedback on each ("wrong: the right answer is approve"), and asks it for a better instruction. Each new instruction is tried on the same few rows; one that does better is scored on the rows you choose with, and joins a pool of candidates. The teacher then works on the candidates that are best on at least one row, so ideas that fix *different* mistakes both survive, and every few steps it combines two of them. It stops at a budget of calls, and keeps the best on the choosing rows; of equally good ones, the shorter.

The three piles fit it exactly: it learns from `examples`, chooses on `dev`, and never sees `test`:

```r
refund_gepa <- gepa(refund_nano, examples, selection = dev, teacher = "gpt-6-sol", budget = 300)

ai_trials(refund_gepa) |>
  select(candidate, kind, minibatch_parent, minibatch, score, length, note)
```

```output
# A tibble: 10 × 7
   candidate kind    minibatch_parent minibatch  score length note              
       <int> <chr>              <dbl>     <dbl>  <dbl>  <int> <chr>             
 1         1 written               NA        NA  0.8       54 the written instr…
 2        NA reflect                3         3 NA        706 not better on the…
 3         2 reflect                3         4  0.8      513 joined the pool   
 4         3 reflect                3         4  0.875   1006 joined the pool   
 5         4 combine                3         3  0.775    413 joined the pool   
 6         5 reflect                3         4  0.875   1174 joined the pool   
 7        NA reflect                2         4 NA       1116 better on the min…
 8        NA combine                4         4 NA        762 better on the min…
 9        NA reflect                2         4 NA        779 better on the min…
10        NA reflect                3         4 NA       1165 better on the min…
```

Each row is an instruction the teacher wrote. `minibatch_parent` and `minibatch` are how many of the same four example rows its parent and it got right. Those that did better joined the pool and were scored on dev (`score`); the last few did better too, but the budget ran out before they could be scored, and one did no better, so it was dropped. Two tie at the top, 0.875 on dev; the shorter one (`length`, in characters) is kept, because every call pays for its length:

```r
cat(ai_instructions(refund_gepa))
```

```output
Function: refund

Decide whether the request merits a refund using the message, days_since_delivery, and final_sale. Return `result` as exactly `approve` or `deny`.

First identify the reason for the request. Approve a credible report that the item arrived damaged or developed a defect during ordinary use soon after delivery. A defect discovered after use still counts; do not apply the change-of-mind return window to it or treat a claim as invalid merely because the customer did not notice the problem immediately. Final-sale status does not override a genuine defect.

For a preference-based request, such as disliking the colour or no longer wanting the item, approve only if it is not final sale, is requested within 30 days of delivery, and the message indicates the item remains unused and in its original condition. Otherwise deny. Deny requests based only on ordinary wear, misuse, or an unsupported desire to return an item long after delivery. Do not use price as a reason to approve or deny.
```

Read it as you'd read a fitted model's coefficients: it is what the search learned from the mistakes, and you can check it against `?refunds`. It found most of the policy from nothing but "the right answer is approve": a fault or damage is refunded even after the usual window and even on final sale, and a change of mind only within 30 days, unused, and never on final sale. It missed the exact limits: damage is refunded within 60 days and a fault within a year, where it says only "soon after delivery". The examples did hold four late claims, all denied, but a search learns only from mistakes, and a model that already denies a late claim makes none there; most likely those rows never taught it anything. A search can only learn what its rows show it getting wrong.

Its dev score was the best of many on dev, so it may flatter (the trap simulated above, and this time the search did the trying). The test rows give the honest number, once, for this question:

```r
ev_nano <- evaluate(refund_nano, test)
ev_nano_gepa <- evaluate(refund_gepa, test)
bind_rows(
  tidy(ev_nano)      |> mutate(version = "gpt-5.4-nano, as written"),
  tidy(ev_nano_gepa) |> mutate(version = "gpt-5.4-nano, instruction by gepa()")
) |> select(version, estimate, conf.low, conf.high)

nano_pairs <- table(written = right_or_wrong(ev_nano), gepa = right_or_wrong(ev_nano_gepa))
nano_pairs
mcnemar.test(nano_pairs)$p.value
```

```output
# A tibble: 2 × 4
  version                             estimate conf.low conf.high
  <chr>                                  <dbl>    <dbl>     <dbl>
1 gpt-5.4-nano, as written               0.683    0.530     0.804
2 gpt-5.4-nano, instruction by gepa()    0.927    0.806     0.975
       gepa
written wrong right
  wrong     1    12
  right     2    26
[1] 0.01615693
```

On rows it never saw, the small model went from about two in three right to more than nine in ten. The disagreements say it's real: twelve rows fixed, two broken, and a split that lopsided on fourteen disagreements is unlikely to be luck (p ≈ 0.02). Here the test rows were even kinder than dev; on another draw they may not be, which is why they're kept apart.

A small model with an instruction written by a large one, once, from its own mistakes. The large model's price is paid for about ten calls; the small model's for every request, forever. It still makes mistakes the written rules don't (`refund_rules` made one on these rows), because it learned part of the policy, not all of it. When the rules are yours to write, write them: they are exact, and you know why they work. When they aren't, this is how to find a good share of them, and measure what you found.

## What it cost

```r
prices <- tribble(
  ~model,         ~input, ~output,   # dollars per million tokens, 2026-09-27
  "gpt-6-luna",     0.10,    0.50,
  "gpt-5.4-nano",   0.20,    1.25,
  "gpt-6-sol",      2.00,   10.00
)

calls(folder = log_folder) |>
  left_join(prices, by = "model") |>
  group_by(model) |>
  summarise(calls = n(),
            dollars = sum(input_tokens * input + (total_tokens - input_tokens) * output, na.rm = TRUE) / 1e6)
```

```output
# A tibble: 3 × 3
  model        calls dollars
  <chr>        <int>   <dbl>
1 gpt-5.4-nano   377  0.0245
2 gpt-6-luna     201  0.0147
3 gpt-6-sol       17  0.0582
```

Worked examples make every question longer (each call now carries four to eight solved cases), so they cost more per call. On a small model that's still cents. Weigh it anyway: it's paid on every call, forever.

`gepa()` was most of this bill: the nano calls and most of the teacher's, about eight cents, paid once. Its instruction is also longer than the one-line original, so every call after it costs a little more; that is the trade it made for accuracy, and why ties go to the shorter instruction.

## Your turn

1. Try `k = 16` instead of 8 in `labeled_few_shot()`. Measure it on dev, and compare it with the rules-only version using the disagreement table and `mcnemar.test()`.
2. Add one sentence to the rules that you think would fix a dev mistake. Is it policy, or is it fitted to that row? How could you tell?
3. Give `gepa()` a `feedback` function that says *why* a decision was wrong, using the `state` column (`feedback = function(row, prediction, error) ...`: the item's state and the days since delivery, for example). Does the teacher find more of the rules? Is that fair, when `state` is something a person had to read?
4. `ai_render(refund_taught, message = "x", price = 1, days_since_delivery = 1L, final_sale = FALSE)` shows the whole request. Find the taught examples in it, and count how much longer it is than `refund_rules`'s.

## What you learned

- Split once, before you start: rows to learn from, rows to choose on, rows to test once.
- Writing the rules down is the most direct improvement, when the rules are yours to write.
- `labeled_few_shot()` adds solved examples; `bootstrap_few_shot()` adds a teacher's runs that were right. Both return a new function with a new `ai_version()`.
- `gepa()` has a stronger model rewrite the instruction from the mistakes: learn on one pile, choose on another, measure on a third. `ai_trials()` shows the search. A small model with a large model's instruction can be most of the way to written rules, at the small model's price.
- On the same rows, compare versions by their disagreements (`mcnemar.test()`), not by eyeballing two intervals.
- When versions tie, keep the simplest and cheapest.
- Picking the best of several on the same rows flatters the winner. The test rows, used once, give the honest number.

**Answers to the check at the top.** (1) It was chosen for being the luckiest on those rows, so part of its score is luck that won't come back: the winner's curse. (2) The rows where they disagree: how many one got right and the other wrong, each way (`mcnemar.test()`). (3) It adds worked examples to the request; the instruction, inputs, outputs and model stay as they were, and the function passed in is unchanged.

**Next:** [5. Choosing a model](05-choosing-a-model.md) puts six current models through the same test and weighs accuracy against cost and speed.
