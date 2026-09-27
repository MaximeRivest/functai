# 7. AI functions in MLJ and formulas

*If you use MLJ, you already know how to use a language model: a model, a machine, `fit!`, `predict`, `evaluate!`. By the end you will have put one in a machine, cross-validated it, tuned how many worked examples it sees, compared it with a classical text classifier, trained that classifier on the language model's labels, and used an AI function as a feature inside a GLM formula.*

**Can you skip this one?** If you can answer these, jump to [tutorial 8](08-living-with-it.md). The answers are at the bottom.

1. What does `fit!` do to a machine holding an `AIModel`, and what does it cost?
2. FunctAI and MLJ both export `predict` and `evaluate`. How do you use both packages in one session?
3. How does an AI function become a column of a `GLM.lm` formula, and how many calls does that make?

**You will:** use `AIModel` as an MLJ model (`machine`, `fit!`, `predict`, `evaluate!` with cross-validation, `TunedModel`), compare it with naive Bayes on word counts, label data with the language model to train a free classical model, and put an AI function inside `@formula`.

## What you need

MLJ and a few model packages (`Pkg.add(["MLJ", "NaiveBayes", "MLJNaiveBayesInterface", "GLM", "StatsModels", "CategoricalArrays"])`), and MLJ's habits: this tutorial follows MLJ's own "Getting started", with a language model as the model. About five cents.

MLJ and FunctAI both export a `predict` and an `evaluate`. In Julia, when two packages loaded with `using` export different functions under one name, using that name is an error: Julia refuses to guess. The usual answer is to bring in one package whole and the other by name. Here MLJ is the frame, so it comes in with `using`, and FunctAI with `import` (its functions as `FunctAI.evaluate`, …), plus the two names we use most:

```julia
using MLJ, DataFrames, CategoricalArrays, Statistics
import FunctAI
using FunctAI: AIModel, @ai

log_folder = mktempdir()
FunctAI.configure!(lm = "gpt-6-luna", log_calls = log_folder);
```

## The data

The tickets from tutorial 1. MLJ asks what kind of data each column is (its *scientific type*): the team is a `Multiclass` target, the message is text.

```julia
tickets = DataFrame(FunctAI.tickets())
y = coerce(tickets.category, Multiclass)
X = select(tickets, :message)
schema(X)
```

```output
┌─────────┬──────────┬────────┐
│ names   │ scitypes │ types  │
├─────────┼──────────┼────────┤
│ message │ Textual  │ String │
└─────────┴──────────┴────────┘
```

## Specify, fit, predict, score

The same verbs as any MLJ model:

```julia
model = AIModel("Which team should answer this customer message?"; name = "team", examples = 0)   # 1. specify
mach = machine(model, X, y)
fit!(mach);                                                                                      # 2. fit
```

```output
[ Info: Training machine(AIModel(description = Which team should answer this customer message?, …), …).
[ Info: AIModel: 0 worked examples; nothing was called
```

`fit!` was instant and free, and the log line says so. A language model already knows language, so fitting only reads the inputs (the columns of `X`) and the target's levels (the only answers it may give). With `examples = 0` it used none of the training answers. `predict` is where the calls happen, a row each, eight at a time:

```julia
ŷ = predict(mach, X)                                                                             # 3. predict
accuracy(ŷ, y)                                                                                   # 4. score
```

```output
0.95
```

The predictions are a `CategoricalVector` with the training levels (`levels(ŷ) == levels(y)`), like any MLJ classifier's. Predictions are *deterministic*: a language model gives an answer, not a probability for each class, and FunctAI never makes one up (tutorial 6 shows where real probabilities come from).

Inside the machine is an ordinary AI function, the kind you've written by hand since tutorial 1:

```julia
fitted_params(mach).fn
```

```output
AI function team(message::String) -> OneOf("account", "billing", "product", "shipping")
  model:        gpt-6-luna
  instruction:  Which team should answer this customer message?
  version:      sha256:153b7ec373a2…
  see:          FunctAI.prompt(team, …) for the exact request
```

## Cross-validated

Scoring on the rows you fitted on flatters any model that learns from them. `evaluate!` resamples, as for every MLJ model: here five folds, each fitted on four fifths and scored on the fifth, with the same mix of teams in each (`StratifiedCV`):

```julia
cv = StratifiedCV(nfolds = 5, shuffle = true, rng = 2026)
ai_cv = evaluate!(machine(AIModel("Which team should answer this customer message?"; name = "team", examples = 4), X, y);
                  resampling = cv, measure = accuracy, verbosity = 0)
```

```output
PerformanceEvaluation object with these fields:
  model, tag, measure, operation,
  measurement (per-fold aggregate), uncertainty_radius_95 (1.96*SE),
  per_fold, per_observation,
  fitted_params_per_fold, report_per_fold,
  train_test_rows, resampling, repeats
Tag: AIModel-920
Extract:
┌────────────┬───────────┬─────────────┬─────────┐
│ measure    │ operation │ measurement │ 1.96*SE │
├────────────┼───────────┼─────────────┼─────────┤
│ Accuracy() │ predict   │ 0.988       │ 0.027   │
└────────────┴───────────┴─────────────┴─────────┘
┌─────────────────────────────┐
│ per_fold                    │
├─────────────────────────────┤
│ [0.938, 1.0, 1.0, 1.0, 1.0] │
└─────────────────────────────┘
Apply `describe` to this result for a named tuple summary.
```

Each message was predicted once, by the fold that didn't see it, with four worked examples drawn from the other folds. `ai_cv.measurement` is the mean; `ai_cv.per_fold` the five scores.

## A classical baseline

What you'd reach for without a language model: count the words in each message, and let naive Bayes learn which words point to which team. First the counts, a column per word that appears in at least three messages:

```julia
words(m) = [w.match for w in eachmatch(r"[a-z']+", lowercase(m))]
seen = Dict{String,Int}()                              # in how many messages each word appears
for m in tickets.message, w in unique(words(m))
    seen[w] = get(seen, w, 0) + 1
end
vocabulary = sort([w for (w, n) in seen if n >= 3])
bag(messages) = DataFrame([Symbol(w) => [count(==(w), words(m)) for m in messages] for w in vocabulary])
Xbag = bag(tickets.message)                            # counts: MLJ reads Int columns as Count
size(Xbag)
```

```output
(80, 62)
```

```julia
NaiveBayes = @load MultinomialNBClassifier pkg = NaiveBayes verbosity = 0
nb_cv = evaluate!(machine(NaiveBayes(), Xbag, y); resampling = cv, measure = accuracy, verbosity = 0);
```

Both models, the same folds, side by side:

```julia
DataFrame(model = ["naive Bayes on word counts", "AIModel (gpt-6-luna)"],
          accuracy = [nb_cv.measurement[1], ai_cv.measurement[1]],
          per_fold = [round.(nb_cv.per_fold[1]; digits = 2), round.(ai_cv.per_fold[1]; digits = 2)])
```

```output
2×3 DataFrame
 Row │ model                       accuracy  per_fold
     │ String                      Float64   Array…
─────┼──────────────────────────────────────────────────────────────────────
   1 │ naive Bayes on word counts    0.775   [0.88, 0.81, 0.75, 0.81, 0.62]
   2 │ AIModel (gpt-6-luna)          0.9875  [0.94, 1.0, 1.0, 1.0, 1.0]
```

Sixty-four messages per fold is very little to learn language from: most words in a new message never appeared in training. The language model starts from knowing the words already.

## Tuned

`examples` is a hyperparameter like any other: how many worked examples the function sees. MLJ tunes it the usual way, a `TunedModel` over a range, each value cross-validated:

```julia
ai = AIModel("Which team should answer this customer message?"; name = "team")
tuned = TunedModel(model = ai, range = range(ai, :examples, values = [0, 2, 8]),
                   tuning = Grid(), resampling = StratifiedCV(nfolds = 3, shuffle = true, rng = 1),
                   measure = accuracy)
tuned_mach = machine(tuned, X, y)
fit!(tuned_mach, verbosity = 0)
DataFrame(examples = [h.model.examples for h in report(tuned_mach).history],
          accuracy = [h.measurement[1] for h in report(tuned_mach).history])
```

```output
3×2 DataFrame
 Row │ examples  accuracy
     │ Int64     Float64
─────┼────────────────────
   1 │        0    0.9625
   2 │        2    0.975
   3 │        8    0.975
```

Each value was scored on eighty predictions, so the differences are a message or two: read them with tutorial 3's intervals in mind. What tuning chose:

```julia
fitted_params(tuned_mach).best_model.examples
```

```output
2
```

## The other way round: a classical model that learns from the language model

Say you had a hundred thousand unlabelled messages and wanted to route them offline, for free, forever. The language model can label them once, and a classical model can learn from those labels. Here we pretend half the tickets have no labels: the language model labels them, naive Bayes learns from its labels, and the other half, labelled by a person, scores it.

```julia
train, test = partition(1:nrow(tickets), 0.5; stratify = y, shuffle = true, rng = 2026)

ai_labels = predict(mach, X[train, :])                     # the language model labels the "unlabelled" half
student = machine(NaiveBayes(), Xbag[train, :], ai_labels)
fit!(student, verbosity = 0)
teacher_taught = machine(NaiveBayes(), Xbag[train, :], y[train])
fit!(teacher_taught, verbosity = 0)

(labels_by_a_person = accuracy(predict_mode(teacher_taught, Xbag[test, :]), y[test]),
 labels_by_the_language_model = accuracy(predict_mode(student, Xbag[test, :]), y[test]))
```

```output
(labels_by_a_person = 0.675, labels_by_the_language_model = 0.7)
```

The student is limited by how few messages it saw, far more than by who labelled them. The pattern is what scales: label forty thousand messages with the language model (a couple of dollars), train the classical model on them, and it predicts in microseconds with no network. Before you trust it, measure it against a few hundred rows a person labelled.

## An AI function inside a formula

MLJ is one frame; formulas are the other. StatsModels' `@formula` (GLM's, MixedModels', …) applies a function to its column, so an AI function can be a feature, estimated alongside the others. The refund desk from tutorial 4: does the item's problem explain the decision, once the days and final sale are accounted for?

```julia
import GLM
using StatsModels: @formula

@ai function problem(message::String)::Bool
    "Did the item arrive damaged, arrive wrong, or fail in normal use (not merely unwanted)?"
end

refunds = DataFrame(FunctAI.refunds())
refunds.approved = Float64.(refunds.decision .== "approve")

ols = GLM.lm(@formula(approved ~ problem(message) + days_since_delivery + final_sale), refunds)
GLM.coeftable(ols)
```

```output
───────────────────────────────────────────────────────────────────────────────────────
                           Coef.  Std. Error      t  Pr(>|t|)    Lower 95%    Upper 95%
───────────────────────────────────────────────────────────────────────────────────────
(Intercept)           0.392738    0.0557119    7.05    <1e-09   0.282393     0.503082
problem(message)      0.544545    0.0612516    8.89    <1e-14   0.423228     0.665861
days_since_delivery  -0.00209041  0.00029397  -7.11    <1e-09  -0.00267266  -0.00150817
final_sale           -0.187334    0.0930929   -2.01    0.0465  -0.371716    -0.00295137
───────────────────────────────────────────────────────────────────────────────────────
```

That was 120 calls, eight at a time: StatsModels calls `problem` on the whole `message` column with a dot, and a dot over an AI function is concurrent. The `Bool` answer is a 0/1 column, so its coefficient reads as the difference in the chance of approval: here, the policy's generosity towards real problems, measured from the decisions, with a standard error.

The same formula syntax fits an `AIModel` directly (StatsAPI's `fit`, with StatsModels loaded):

```julia
team_fit = FunctAI.fit(AIModel("Which team should answer this customer message?"; examples = 4),
                       @formula(category ~ message), tickets[train, :])
team_fit
```

```output
AI model of category from message
AI function category(message::String) -> String
  model:        gpt-6-luna
  instruction:  Which team should answer this customer message?
  examples:     4
  version:      sha256:49bf80f329e0…
  see:          FunctAI.prompt(category, …) for the exact request
```

```julia
FunctAI.evaluate(team_fit, tickets[test, :])
```

```output
Evaluation of category on 40 rows
  exact_match  0.82  (95% range 0.68 to 0.91)
  every row: DataFrame(e)
```

An AI model reads its inputs as they are, together. What a formula means only to a regression, an interaction (`a & b`) or a transformed column (`log(x)`), it refuses, and says why.

## What it cost

```julia
bill = dropmissing(DataFrame(FunctAI.calls(folder = log_folder)), :model)
(calls = nrow(bill),
 dollars = sum(bill.input_tokens .* 0.10 .+ (bill.total_tokens .- bill.input_tokens) .* 0.50) / 1e6)   # gpt-6-luna, 2026-09-27
```

```output
(calls = 600, dollars = 0.0242948)
```

## Your turn

1. Tune `examples` over `[0, 16]` instead. Does sixteen help, or does it only make every call longer?
2. Put `settings = (lm = "gemini:gemini-3.1-flash-lite",)` in the `AIModel` and cross-validate it. Is it as accurate, for less?
3. Add `problem(message) & final_sale` to the GLM formula. What happens, and why is that fine for GLM but refused by an `AIModel`?

## What you learned

- `AIModel` is an MLJ model: `machine`, `fit!` (free: no call), `predict` (the calls), `evaluate!` with any resampling, `TunedModel` over `examples`.
- With two packages that export the same name, bring one in with `using` and the other with `import`, and qualify.
- A classical text model needs far more rows than a language model; the language model can label rows for it.
- In a `@formula`, an AI function is a feature: StatsModels broadcasts it, so its calls are concurrent.

**Answers to the check at the top.** (1) It reads the input columns and the target's levels and picks worked examples from the rows (`examples`); it calls nothing and costs nothing. (2) `using MLJ` and `import FunctAI` (plus `using FunctAI: AIModel, @ai`), then `FunctAI.evaluate` for FunctAI's. (3) Write it as a function of a column, `@formula(y ~ f(x) + …)`: StatsModels calls it with a dot on the whole column, one call per row, eight at a time.

**Next:** [8. Living with it](08-living-with-it.md): tools, the call log, people's corrections, versions, saving.
