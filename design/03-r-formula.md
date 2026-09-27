# 03. Writing an AI function in R: a model formula and a codebook

Status: done (R 0.1.0). Changes R's API only; the contract is untouched.

## The problem

R's first `ai()` was a translation of the Python signature:

```r
team <- ai("team", "Which team should answer this customer message?",
  message = character(),
  .returns = factor(levels = c("shipping", "billing", "product", "account")))
```

The name was typed twice (`team <- ai("team", ...)`). Every text input
needed `character()`, an empty vector used as a type that the tutorials
had to explain. Descriptions went around types (`optional(described(integer(), ...))`).
And the answer was called `result`, so worked examples, evaluations and
ratings needed a column called `result` that no dataset has.

The people this package is for already have a notation for "this, from
those": the model formula. They learned it for `lm()` and `glm()`, and
they use it in parsnip's `fit()`, lavaan and brms.

## The design

```r
team <- ai(team ~ message, "Which team should answer this customer message?",
  team = choice("shipping", "billing", "product", "account"))

triage <- ai(summary + urgent ~ message, "Read the support ticket.",
  message = "the customer's own words",
  summary = "one sentence, no names",
  urgent  = logical(),
  .name = "triage")

refund <- ai(decision ~ message + price + days_since_delivery + final_sale,
  "Should the shop refund this request?", .data = refunds, .name = "refund")
```

1. **The formula.** Outputs on the left, inputs on the right, names
   joined by `+`; `~ .` is every other column of `.data`. One output
   names the function (`team`); several need `.name`.
2. **The codebook.** Each field is one of three things: left out (text,
   or its column's type), a sentence (the same, described), or a type.
   That one rule replaces `character()` everywhere and most of the
   `described()` calls. `record()`, `optional()` and `ai_tool()` follow
   it too.
3. **Types from data**, as `lm(y ~ x, data)` reads them: a factor is a
   choice of its levels, integer and double are numbers, logical is yes
   or no, anything else is text. Only column types are read. A text
   outcome stays text: we don't guess that a character column is a set
   of answers.
4. **`choice()`** replaces `factor(levels = )`, and a named level says
   what it means (`choice(approve = "the rules allow it", ...)`). The
   model reads the meanings after the field's own words.
5. **The formula's column is the answer's column** in `evaluate()` (with
   no `expected`), the worked examples, `bootstrap_few_shot()`'s scoring
   and `rated()`. In the *definition*, one output is still `result`, as
   in Python and TypeScript, so a function has the same version in every
   language. `core$columns` maps the definition's names to the formula's.
   A function loaded by `read_ai()` has its own name as the answer's
   column.
6. **What only a regression means is refused, with the reason**:
   interactions (`a * b`, `a:b`), transformed columns (`log(x)`) and
   intercepts (`- 1`). An AI function reads all its inputs together, as
   they are. The same formula therefore fits `glm()`, `ai()` and
   `ai_model()` whenever it names columns.

## Rejected

- **A lavaan-style model string** (`"team ~ message"` in quotes, types
  after colons). R's parser already reads formulas. A string has no
  syntax checking, and its errors point at a character, not a name.
- **Type words exported as `text()`, `number()`, `one_of()`.**
  `graphics::text()`, `scales::number()` and `dplyr::one_of()` exist, and
  masking them breaks code that has nothing to do with functai. A
  sentence already means text; base R's `integer()`, `double()` and
  `logical()` are the other types.
- **Types that exist only inside `ai()`** (a data mask, as brms does for
  priors). They are invisible to help and autocomplete, and fail
  confusingly outside `ai()`.
- **ggplot2's `+` to add examples or instructions.** The pipe does that
  (`team |> labeled_few_shot(train)`), and ggplot2's author has said he
  would use a pipe today.
- **A path-model syntax for chains of AI functions.** `mutate(state =
  item_state(message), decision = policy(state, ...))` already expresses
  the chain, and runs it.
- **Naming a single output after the formula in the definition.** That
  reads well in the prompt ("team:"), but then the same function written
  in Python (`-> result`) would have another version.
