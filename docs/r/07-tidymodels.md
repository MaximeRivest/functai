# 7. AI functions in tidymodels

*If you use tidymodels, you already know how to use a language model:
specify, fit, predict, score. By the end you will have put one in a
workflow, resampled it, tuned how many worked examples it sees, fitted
one that learns its own instruction from its mistakes, turned its votes
into probabilities for `roc_auc()`, and trained a free classical model
on its answers.*

**Can you skip this one?** If you can answer these, jump to
[tutorial 8](08-living-with-it.md). The answers are at the bottom.

1. What does `fit()` do to an `ai_model()`, and what does it cost?
2. Why does an AI model's workflow use `add_formula()` rather than a
   recipe?
3. Where do an AI model's class probabilities come from?

## What you need

tidymodels and textrecipes (`install.packages(c("tidymodels",
"textrecipes", "glmnet"))`), and some tidymodels habits: this tutorial
follows [tidymodels.org/start](https://www.tidymodels.org/start/), with
a language model as the model. About five cents.

```r
library(functai)
library(tidymodels)
library(textrecipes)

log_folder <- tempfile("functai-calls-")
ai_config(lm = "gpt-6-luna", log_calls = log_folder)
```

## The data budget

The tickets from tutorial 1, split in half, with the same mix of teams in
each half:

```r
tickets <- tickets |> mutate(category = factor(category))

set.seed(2026)
split <- initial_split(tickets, prop = 1/2, strata = category)
train <- training(split)
test  <- testing(split)
```

## A classical baseline

What you'd reach for without a language model: turn each message into
word weights (tf-idf) and let a penalised multinomial regression learn
which words point to which team.

```r
tfidf <- recipe(category ~ message, data = train) |>
  step_tokenize(message) |>
  step_tokenfilter(message, max_tokens = 200) |>
  step_tfidf(message)

glmnet_fit <- workflow() |>
  add_recipe(tfidf) |>
  add_model(multinom_reg(penalty = 0.01) |> set_engine("glmnet")) |>
  fit(train)

augment(glmnet_fit, test) |> accuracy(category, .pred_class)
```

```output
# A tibble: 1 × 3
  .metric  .estimator .estimate
  <chr>    <chr>          <dbl>
1 accuracy multiclass     0.525
```

Forty messages is very little to learn language from: most words in the
test messages never appeared in training.

## Specify, fit, predict, score

The same four verbs, with a language model:

```r
ai_spec <- ai_model("classification", "Which team should answer this customer message?") |>
  set_engine("functai")                                                # 1. specify
ai_spec

ai_fit <- fit(ai_spec, category ~ message, data = train)               # 2. fit

ai_test <- augment(ai_fit, test)                                        # 3. predict
ai_test |> accuracy(category, .pred_class)                             # 4. score
```

```output
AI Model Specification (classification)

Main Arguments:
  description = Which team should answer this customer message?

Computational engine: functai 

Warning: probabilities are NA: one answer per row measures no probability
ℹ for them, answer each row several times (`set_engine("functai", samples =
  5)`, 5 calls a row), or use a model that measures them (`lm = "jev-latest"`)
# A tibble: 1 × 3
  .metric  .estimator .estimate
  <chr>    <chr>          <dbl>
1 accuracy multiclass     0.975
```

`fit()` was instant and free. A language model already knows language,
so fitting only reads the formula (the input is `message`) and the
outcome's levels (the only answers it may give). With `examples = 0`,
the default, it used none of the training answers. And `augment()`
warned that the probability columns are `NA`: one answer per row
measures no probability, and functai never makes one up. We'll get real
ones below.

Inside the fit is an ordinary AI function, the kind you've written by
hand since tutorial 1:

```r
team <- extract_fit_engine(ai_fit)
team
```

```output
<ai function> category ~ message
  Which team should answer this customer message?
  message   text
  category  one of account, billing, product, shipping
  model: gpt-6-luna
```

It is the function `ai(category ~ message, "Which team should answer this
customer message?", .data = train)` writes: the formula you give `fit()`
is the one you give `ai()`, read the same way (*category, from the
message*), with the types of `train`'s columns. What a formula means only
to a regression, an interaction (`a * b`) or a transformed column
(`log(x)`), an AI function refuses, and says why: it reads all its inputs
together, as they are.

`evaluate()` scores any model the same way, so the two compare on equal
terms, with intervals:

```r
bind_rows(
  tidy(evaluate(glmnet_fit, test)) |> mutate(model = "glmnet on tf-idf, trained on 40"),
  tidy(evaluate(ai_fit, test))     |> mutate(model = "gpt-6-luna, no training")
) |> select(model, estimate, conf.low, conf.high)
```

```output
# A tibble: 2 × 4
  model                           estimate conf.low conf.high
  <chr>                              <dbl>    <dbl>     <dbl>
1 glmnet on tf-idf, trained on 40    0.525    0.375     0.671
2 gpt-6-luna, no training            0.95     0.835     0.986
```

## Resampled and tuned

A language model can learn from worked examples shown before each
question, and how many to show is a tuning parameter, `examples`. In a
workflow, tuned by cross-validation like any other:

```r
ai_wf <- workflow() |>
  add_formula(category ~ message) |>
  add_model(ai_model("classification", "Which team should answer this customer message?",
                     examples = tune()) |> set_engine("functai"))

set.seed(1)
folds <- vfold_cv(train, v = 5, strata = category)

tuned <- tune_grid(ai_wf, folds,
                   grid = tibble(examples = c(0, 4, 8)),
                   metrics = metric_set(accuracy))

collect_metrics(tuned) |> select(examples, mean, std_err)
```

```output
# A tibble: 3 × 3
  examples  mean std_err
     <dbl> <dbl>   <dbl>
1        0 0.975  0.025 
2        4 0.91   0.0392
3        8 1      0     
```

Two things differ from a classical workflow:

- **A formula, not a recipe.** The language model reads the message
  itself. A recipe that turned it into word weights would take the words
  away.
- **Accuracy only.** tune's default metrics include `roc_auc`, which
  needs probabilities; one answer per row has none.

And one thing to keep in mind: **every prediction is a paid call.** This
grid predicted each training message three times (once per value of
`examples`), 120 calls. A grid of ten values on ten-fold resampling of a
thousand rows would be ten thousand. Price it before you run it.

Finish as tidymodels always does: pick a value, fit on all the training
data, and evaluate once on the test set:

```r
best <- select_best(tuned, metric = "accuracy")
best

final <- finalize_workflow(ai_wf, best) |> last_fit(split, metrics = metric_set(accuracy))
collect_metrics(final)
```

```output
# A tibble: 1 × 2
  examples .config        
     <dbl> <chr>          
1        8 pre0_mod3_post0
# A tibble: 1 × 4
  .metric  .estimator .estimate .config        
  <chr>    <chr>          <dbl> <chr>          
1 accuracy multiclass     0.975 pre0_mod0_post0
```

## A fit that learns

So far `fit()` learned nothing from the training answers, unless you
asked for worked examples. With `method = "gepa"`, it does: a stronger
model (the `teacher`) reads the function's mistakes on the training rows
and rewrites its instruction ([tutorial 4](04-making-it-better.md) shows
how). The instruction is then what was fitted, as coefficients are for a
regression.

That makes fitting cost calls. And it makes resampling mean what it
means for any model that learns: each fold runs the whole search on its
own training rows and is scored on rows the search never saw, so the
resampled accuracy measures the *procedure*, search included, not one
lucky instruction. Here it is on `gpt-5.4-nano`, the small model of six
months ago, next to the same model fitted plainly:

```r
nano <- ai_model("classification", "Which team should answer this customer message?")

wf_plain <- workflow() |>
  add_formula(category ~ message) |>
  add_model(nano |> set_engine("functai", lm = "gpt-5.4-nano"))
wf_gepa <- wf_plain |>
  update_model(nano |> set_engine("functai", lm = "gpt-5.4-nano",
                                  method = "gepa", teacher = "gpt-6-sol", budget = 150))

learned <- control_resamples(extract = function(fit) ai_instructions(extract_fit_engine(fit)))
res_plain <- fit_resamples(wf_plain, folds, metrics = metric_set(accuracy))
res_gepa  <- fit_resamples(wf_gepa, folds, metrics = metric_set(accuracy), control = learned)

bind_rows(collect_metrics(res_plain) |> mutate(fit = "plain"),
          collect_metrics(res_gepa)  |> mutate(fit = "gepa")) |>
  select(fit, mean, std_err)
```

```output
# A tibble: 2 × 3
  fit    mean std_err
  <chr> <dbl>   <dbl>
1 plain 0.897  0.0280
2 gepa  0.967  0.0333
```

Fitting that learns is right about seven points more often, on messages
no fold's search saw. The same five folds as above, so the two compare
fold by fold:

```r
tibble(fold = collect_metrics(res_plain, summarize = FALSE)$id,
       plain = collect_metrics(res_plain, summarize = FALSE)$.estimate,
       gepa  = collect_metrics(res_gepa,  summarize = FALSE)$.estimate)
```

```output
# A tibble: 5 × 3
  fold  plain  gepa
  <chr> <dbl> <dbl>
1 Fold1 0.9   1    
2 Fold2 0.875 1    
3 Fold3 0.875 1    
4 Fold4 1     1    
5 Fold5 0.833 0.833
```

It won three folds and tied the other two; it lost none. What did each
fold learn? One of them:

```r
instructions <- collect_extracts(res_gepa)
cat(instructions$.extracts[[1]])
```

```output
Choose the team that should resolve the customer’s main request. Return exactly one label: account, billing, product, or shipping.

- account: login, profile, credentials, or account settings.
- billing: charges, invoices, payments, discounts, refunds, or returns requesting a refund.
- product: product features, use, compatibility, or problems with how a product works.
- shipping: delivery, tracking, missing packages, or items that arrived damaged.

When a message mentions multiple topics, prioritize the action requested. In particular, route a request for a refund to billing, even if it also mentions returning a product.
```

Put it next to the house rules in `?tickets`. From nothing but "wrong:
the right answer is billing", the teacher found the rule that trips up
everyone who hasn't read them, *every request for money back is billing*,
and the one about damage on arrival, and wrote them down for the small
model. Print `instructions$.extracts` for what the other folds wrote; a
fold whose training rows hold no such mistake has nothing to learn from,
and keeps the written instruction. Its cost, a few hundred calls of the small
model and eight of the large one, is in the bill below: about four
cents.

## Probabilities, from votes

`roc_auc()`, calibration and thresholds need a probability per class.
OpenAI, Anthropic and Gemini don't report how likely each answer is, so
functai asks the same question several times instead. With
`samples = 5`, each message is answered five times; the probability of a
team is its share of the five answers, and the class is the majority:

```r
voter <- ai_model("classification", "Which team should answer this customer message?") |>
  set_engine("functai", samples = 5) |>
  fit(category ~ message, data = train)

votes <- augment(voter, test)
votes |> select(category, .pred_class, .pred_account:.pred_shipping)
votes |> roc_auc(category, .pred_account:.pred_shipping)
```

```output
# A tibble: 40 × 6
   category .pred_class .pred_account .pred_billing .pred_product .pred_shipping
   <fct>    <fct>               <dbl>         <dbl>         <dbl>          <dbl>
 1 shipping shipping                0           0             0                1
 2 billing  billing                 0           1             0                0
 3 product  product                 0           0             1                0
 4 shipping shipping                0           0             0                1
 5 account  account                 1           0             0                0
 6 shipping shipping                0           0             0                1
 7 billing  billing                 0           1             0                0
 8 product  product                 0           0             1                0
 9 account  account                 1           0             0                0
10 billing  billing                 0           0.8           0.2              0
# ℹ 30 more rows
# A tibble: 1 × 3
  .metric .estimator .estimate
  <chr>   <chr>          <dbl>
1 roc_auc hand_till          1
```

Where did the votes split?

```r
votes |>
  filter(pmax(.pred_account, .pred_billing, .pred_product, .pred_shipping) < 1) |>
  select(category, .pred_class, .pred_account:.pred_shipping, message)
```

```output
# A tibble: 5 × 7
  category .pred_class .pred_account .pred_billing .pred_product .pred_shipping
  <fct>    <fct>               <dbl>         <dbl>         <dbl>          <dbl>
1 billing  billing               0             0.8           0.2            0  
2 billing  billing               0             0.6           0.4            0  
3 billing  shipping              0             0.4           0              0.6
4 billing  billing               0             0.8           0.2            0  
5 account  account               0.6           0             0.4            0  
# ℹ 1 more variable: message <chr>
```

Five calls per message, shared between the class and the probabilities.
As tutorial 6 showed, votes measure how *consistent* a model is, which
is related to, but not the same as, how likely it is to be right.

## The other way round: a classical model that learns from the language model

Say you had a hundred thousand unlabelled messages and wanted to route
them offline, for free, forever. The language model can label them once,
and a classical model can learn from those labels. Here we pretend
`train` has no labels:

```r
ai_labelled <- train |>
  select(message) |>
  mutate(category = team(message))

student <- workflow() |>
  add_recipe(recipe(category ~ message, data = ai_labelled) |>
               step_tokenize(message) |>
               step_tokenfilter(message, max_tokens = 200) |>
               step_tfidf(message)) |>
  add_model(multinom_reg(penalty = 0.01) |> set_engine("glmnet")) |>
  fit(ai_labelled)

bind_rows(
  tidy(evaluate(glmnet_fit, test)) |> mutate(labels = "a person's"),
  tidy(evaluate(student, test))    |> mutate(labels = "the language model's")
) |> select(labels, estimate, conf.low, conf.high)
```

```output
# A tibble: 2 × 4
  labels               estimate conf.low conf.high
  <chr>                   <dbl>    <dbl>     <dbl>
1 a person's              0.525    0.375     0.671
2 the language model's    0.525    0.375     0.671
```

The student is limited by how few messages it saw, far more than by who
labelled them. The pattern is what scales: label forty thousand messages
with the language model (a couple of dollars), train the classical model
on them, and it predicts in microseconds with no network. Before you
trust it, measure it against a few hundred rows a person labelled.

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
  summarise(calls = n(), dollars = sum(input_tokens * input + (total_tokens - input_tokens) * output, na.rm = TRUE) / 1e6)
```

```output
# A tibble: 3 × 3
  model        calls dollars
  <chr>        <int>   <dbl>
1 gpt-5.4-nano   428  0.0173
2 gpt-6-luna     480  0.0148
3 gpt-6-sol        8  0.0214
```

## Your turn

1. Tune `examples` over `c(0, 2, 16)` instead. Does sixteen help, or does
   it only make every call longer?
2. Put `set_engine("functai", lm = "gemini:gemini-3.1-flash-lite")` in the
   workflow and compare its resampled accuracy with `gpt-6-luna`'s.
3. Fit `wf_gepa` on all of `train` and score it once on `test`. Is the
   resampled accuracy a fair forecast of it? Print the fitted function
   (`extract_fit_engine()`): which of the house rules in `?tickets` did
   it find?
4. With the voter's probabilities, draw a gain curve
   (`gain_curve(votes, category, .pred_account:.pred_shipping) |> autoplot()`).
   What would a model with no idea look like?

## What you learned

- `ai_model(mode, description)` is a parsnip model with the engine
  `"functai"`: it goes wherever parsnip models go.
- `fit()` reads the formula and the outcome's levels; with `examples = k`
  it picks k training rows as worked examples. No weights, no cost.
- Use `add_formula()`: the model reads the text itself.
- `examples` tunes with `tune()`; every resampled prediction is a paid
  call, so price the grid first.
- `set_engine("functai", method = "gepa", teacher = ...)` makes `fit()`
  learn: a stronger model rewrites the instruction from the mistakes.
  Resampling then measures the whole search, fold by fold, and
  `control_resamples(extract = ...)` shows what each fold learned.
- `samples = 5` gives probabilities from votes, for `roc_auc()` and
  friends. Without it they are `NA`, never invented.
- A language model can label data for a classical model that then runs
  for free.

**Answers to the check at the top.** (1) It reads the formula and the
outcome's levels, and picks `examples` worked examples from the training
rows. It calls nothing, so it costs nothing; unless you ask it to learn
(`method = "gepa"`), and then it pays for the search. (2) The model reads the raw
text; a recipe would replace the words with numbers. (3) From votes:
`set_engine("functai", samples = 5)` asks each question five times and
reports each class's share.

**Next:** [8. Living with it](08-living-with-it.md): tools, the call
log, people's corrections, and saving a function for other languages.
