# 8. Living with it

*A function in a notebook is an experiment. A function the shop relies
on needs more: it has to look things up instead of guessing, leave a
record of every answer, learn from the people who correct it, and travel
to wherever it's needed. By the end you will have done all four.*

**Can you skip this one?** If you can answer these, you've finished the
series. The answers are at the bottom.

1. When a model "calls a tool", who runs the code?
2. Where does a correction a person makes to an answer end up, and how
   do you use it?
3. What does `ai_version()` identify, and what doesn't change it?

## Setting up

```r
library(functai)
library(dplyr)

log_folder <- tempfile("functai-calls-")
ai_config(lm = "gpt-6-luna", log_calls = log_folder)
```

In real use you'd write `ai_config(log_calls = TRUE)` (or set
`FUNCTAI_LOG_CALLS=1` in `~/.Renviron`), and calls would go to one folder
on your machine, the same one functai in Python and TypeScript use. We
keep this tutorial's calls in a folder of their own.

## Looking things up instead of guessing

Customers ask where their order is. The model can't know: the answer is
in the shop's order system, which here is a table:

```r
orders <- tribble(
  ~order,   ~status,      ~note,
  "A-1042", "in transit", "held at the Montreal depot since September 8",
  "C-3319", "delivered",  "left with a neighbour at 14 Elm Street on September 20",
  "A-1299", "delivered",  "the courier's photo shows it at the side door, September 22",
  "B-2417", "processing", "waiting for stock; ships October 2",
  "A-1350", "in transit", "delayed by the carrier; new estimate September 30",
  "C-3480", "processing", "not shipped yet, so the address can still be changed"
)
```

A **tool** is an R function the model may ask for. `ai_tool()` wraps it
with a name, a sentence and its inputs' types, exactly like `ai()`:

```r
looked_up <- character()

lookup_order <- ai_tool(function(order) {
  looked_up <<- c(looked_up, order)                  # so we can see what the model asked for
  row <- filter(orders, order == toupper(!!order))
  if (nrow(row) == 0) return("there is no order with that number")
  paste0(row$status, ": ", row$note)
}, "lookup_order", "Look up an order's delivery status by its number.",
  order = described(character(), "a letter, a dash and four digits, like A-1042"))

lookup_order
```

```output
<ai tool> lookup_order(order): Look up an order's delivery status by its number.
```

Get the mental model right, because it's easy to get wrong. The model
does **not** run your function. It replies, in effect, "please run
`lookup_order` with `A-1042`". functai runs your R function, on your
machine, sends the result back, and the model then writes its answer,
or asks for another lookup. The model's skill is knowing *when* to look
something up and *what* to ask for.

```r
where_is <- ai("where_is",
  "Tell the customer where their order is, in one or two friendly sentences. Use only what the
  order system says; never guess a date. If the message has no order number, ask for it.",
  message = character(),
  .tools = list(lookup_order))

asked <- tickets |>
  filter(order_id %in% orders$order) |>
  select(message) |>
  bind_rows(tibble(message = c("Where is my order Z-9999?", "Hi, where's my parcel? It's been ages.")))

replied <- asked |> mutate(reply = where_is(message))
writeLines(paste0(replied$message, "\n  -> ", replied$reply, "\n"))
looked_up
```

```output
Hi, my order A-1042 still hasn't arrived and it's been three weeks.
  -> Your order is in transit and has been held at the Montreal depot since September 8. I’m sorry it hasn’t arrived yet.

Tracking for C-3319 hasn't moved since Monday.
  -> Your order was delivered and left with a neighbour at 14 Elm Street on September 20.

Order c3319 was delivered to my neighbour's address instead of mine.
  -> Could you share your order number in the format of a letter, a dash, and four digits (for example, A-1042)?

My parcel says delivered but it's not here. Order A-1299.
  -> The courier’s photo shows your parcel at the side door. It was marked delivered on September 22.

When will order B-2417 ship? It says 'processing' for a week.
  -> Your order is processing while it waits for stock and is scheduled to ship October 2.

Order #A-1350 is two weeks late. Any news?
  -> Your order is in transit and delayed by the carrier. The current estimated delivery date is September 30.

Can I change the delivery address for order C-3480? It hasn't shipped yet.
  -> Your order hasn’t shipped yet, so the delivery address can still be changed.

Where is my order Z-9999?
  -> I couldn’t find an order with number Z-9999. Could you check the number and send it again?

Hi, where's my parcel? It's been ages.
  -> I’m sorry it’s taking so long. Could you share your order number so I can check where your parcel is?

[1] "B-2417" "A-1042" "C-3480" "Z-9999" "A-1299" "A-1350" "C-3319"
```

Read the replies against `orders`, and `looked_up` against the messages.
The unknown order is reported, not invented; the message with no number
gets a question, because there was nothing to look up; every date comes
from the table.

Now look closely at the message that writes its number as "c3319". Did
the model look up `C-3319`, or ask the customer for a number they had
already given? Models read a tool's description literally: we said "a
letter, a dash and four digits", and "c3319" has no dash. If it asked
again, the fix is one sentence in the description ("customers often
leave out the dash or write the letter in lower case; write it as
A-1042"), or one line of R in the tool (`toupper()` plus inserting the
dash), then reading the replies again. This is the everyday work of
living with an AI function: read what it did, and fix the words or the
code.

## The log: every answer, on the record

Every call so far is a line in the log folder. `calls()` reads them all,
or one function's:

```r
calls(folder = log_folder) |>
  count(name, model)

calls(where_is, folder = log_folder) |>
  select(started, seconds, input_tokens, total_tokens, error)
```

```output
# A tibble: 1 × 3
  name     model          n
  <chr>    <chr>      <int>
1 where_is gpt-6-luna     9
# A tibble: 9 × 5
  started             seconds input_tokens total_tokens error
  <dttm>                <dbl>        <dbl>        <dbl> <chr>
1 2026-09-27 14:39:12    2.63          344          402 <NA> 
2 2026-09-27 14:39:12    3.41          336          409 <NA> 
3 2026-09-27 14:39:12    2.70          148          287 <NA> 
4 2026-09-27 14:39:13    2.83          345          397 <NA> 
5 2026-09-27 14:39:13    2.80          344          424 <NA> 
6 2026-09-27 14:39:13    3.30          339          418 <NA> 
7 2026-09-27 14:39:13    3.54          347          421 <NA> 
8 2026-09-27 14:39:13    2.85          323          407 <NA> 
9 2026-09-27 14:39:15    1.39          144          195 <NA> 
```

Each call knows its function's name, its **version**, the model, the
time it took, its tokens and, since the log keeps content by default,
its inputs and outputs (`log_content = FALSE` keeps only their sizes,
for private data). That's the raw material for everything below.

## People correct it; corrections become data

Here's the loop that keeps a function honest after it ships. Someone
reviews a sample of real answers, says which were right, and corrects
the wrong ones. Let's play the reviewer, using the right answers we
happen to have:

```r
team <- ai("team", "Which team should answer this customer message?",
  message = character(),
  .returns = factor(levels = c("shipping", "billing", "product", "account")))

set.seed(8)
sample_rows <- tickets |> slice_sample(n = 30)
answered <- augment(team, sample_rows)            # .pred_class, and .call: each answer's id in the log

answered |> select(category, .pred_class, .call) |> head()
```

```output
# A tibble: 6 × 3
  category .pred_class .call                               
  <chr>    <fct>       <chr>                               
1 account  account     01a0e34e-578c-77b8-9f4c-cb62a331a0ed
2 product  product     01a0e34e-578f-7f79-b355-b09ab2acd409
3 account  account     01a0e34e-5791-78c6-b462-3abb6908892d
4 billing  billing     01a0e34e-5794-77a7-a657-a1b433bbbeac
5 shipping shipping    01a0e34e-5796-7642-a137-749e26c68bc7
6 billing  billing     01a0e34e-5799-7118-a71c-61e0897b197d
```

`rate()` records a verdict on a call, by its id. A wrong one carries the
right answer:

```r
right <- answered |> filter(.pred_class == category)
wrong <- answered |> filter(.pred_class != category)

rate(right$.call, "right")
rate(wrong$.call, "wrong", answer = wrong$category)
```

`rated()` turns the reviews back into rows with known answers, typed
like the function's inputs and output:

```r
reviewed <- rated(team)
reviewed |> select(message, result, rating)
```

```output
# A tibble: 30 × 3
   message                                                         result rating
   <chr>                                                           <fct>  <chr> 
 1 Can I change the password without the old one?                  accou… right 
 2 The rice cooker's inner pot has a scratch after the first use.  produ… right 
 3 Please delete my account and all my data.                       accou… right 
 4 The duvet shrank in the wash, I'd like my money back.           billi… right 
 5 Hi, my order A-1042 still hasn't arrived and it's been three w… shipp… right 
 6 Refund please: the towels are much thinner than in the photos.  billi… right 
 7 I was charged twice for order B-2210, please fix this.          billi… right 
 8 I'd like my money back for the toaster, it burns everything.    billi… right 
 9 Please send the password reset to my new address, not the old … accou… right 
10 I get 'invalid token' every time I try to sign in.              accou… right 
# ℹ 20 more rows
```

Those rows are an evaluation set that grows by itself as people review.
Any new version of the function is measured on it:

```r
team_rules <- ai("team",
  "Which team should answer this customer message? House rules: anything wrong with the delivery
  itself (late, lost, wrong item, missing, broken on arrival) is shipping; anything about money,
  including every request for money back, is billing; problems in use and product questions are
  product; sign-in, passwords, profile and personal data are account.",
  message = character(),
  .returns = factor(levels = c("shipping", "billing", "product", "account")))

bind_rows(
  tidy(evaluate(team, reviewed)) |> mutate(version = "as shipped"),
  tidy(evaluate(team_rules, reviewed)) |> mutate(version = "with the house rules")
) |> select(version, estimate, conf.low, conf.high, n)
```

```output
# A tibble: 2 × 5
  version              estimate conf.low conf.high     n
  <chr>                   <dbl>    <dbl>     <dbl> <int>
1 as shipped              0.967    0.833     0.994    30
2 with the house rules    1        0.886     1        30
```

The ratings live in the same folder as the calls, in the same format
across languages: a correction made from Python or TypeScript shows up
in R's `rated()`, and the other way round.

## Versions

Every call in the log carries the version of the function that made it:

```r
c(team = ai_version(team), team_rules = ai_version(team_rules))
calls(team, folder = log_folder) |> count(version)
```

```output
                                                                     team 
"sha256:99ae724adb8e3da04da85dddbeec5dc1feed95478e91e3c027342bba47f6013c" 
                                                               team_rules 
"sha256:7c39a3ba5dd65d056dbaf91a7d8c323f809d6829e1e0acaab54f58bfb7feaa7b" 
# A tibble: 2 × 2
  version                                                                     n
  <chr>                                                                   <int>
1 sha256:7c39a3ba5dd65d056dbaf91a7d8c323f809d6829e1e0acaab54f58bfb7feaa7b    30
2 sha256:99ae724adb8e3da04da85dddbeec5dc1feed95478e91e3c027342bba47f6013c    60
```

A version is a fingerprint of everything the function sends besides its
inputs: the instruction, the layout, the worked examples. Change a word
of the description and the version changes. Change the *model* and it
doesn't: the model is a setting, so you can compare models on one
version. And the same function written in Python or TypeScript has the
same version, so their calls and ratings add up.

## Saving it

A function is worth keeping once you've measured it. `write_ai()` saves
it as a folder:

```r
dir <- file.path(tempdir(), "team_rules")
write_ai(team_rules, dir)
list.files(dir)
readLines(file.path(dir, "functai.json"), n = 12)
```

```output
[1] "functai.json"
 [1] "{"                                            
 [2] "  \"functai_saved\": 1,"                      
 [3] "  \"language\": \"r\","                       
 [4] "  \"entry\": \"__main__:team\","              
 [5] "  \"created\": \"2026-09-27T14:39:36+00:00\","
 [6] "  \"functai\": \"0.1.0\","                    
 [7] "  \"nodes\": {"                               
 [8] "    \"__main__:team\": {"                     
 [9] "      \"kind\": \"ai\","                      
[10] "      \"module\": \"__main__\","              
[11] "      \"name\": \"team\","                    
[12] "      \"ai\": {"                              
```

The folder holds everything the function sends, and fingerprints of the
requests it makes. `read_ai()` loads it back and checks, before any call,
that it would send exactly what it sent when it was saved:

```r
team_loaded <- read_ai(dir)
identical(ai_version(team_loaded), ai_version(team_rules))
team_loaded("The courier left my package in the rain and the box is soaked.")
```

```output
[1] TRUE
[1] shipping
Levels: shipping billing product account
```

`read_ai()` also loads folders saved by functai in Python and
TypeScript, and refuses, with the reason, what only the saving language
can run (code of its own around the model, tools, a trained model). A
function with tools can't be saved from R for the same reason: a tool is
code.

## What it cost

```r
calls(folder = log_folder) |>
  mutate(dollars = (input_tokens * 0.10 + (total_tokens - input_tokens) * 0.50) / 1e6) |>   # gpt-6-luna's prices
  summarise(calls = n(), dollars = sum(dollars, na.rm = TRUE))
```

```output
# A tibble: 1 × 2
  calls dollars
  <int>   <dbl>
1   100 0.00322
```

## Your turn

1. Add a second tool, `cancel_order(order)`, that only works when the
   status is "processing", and a function that handles cancellation
   requests. What does the model do for an order already in transit?
2. Rate five of `where_is`'s replies. Which ones would you mark wrong,
   and why? What would you add to its description?
3. Save `team_rules` with a worked example or two
   (`labeled_few_shot()`), load it back, and check the versions differ
   from the plain `team_rules`.

## What you learned

- `ai_tool()` gives the model an R function to ask for; functai runs it
  and sends back the result. The model decides when, and with what.
- The call log records every call: `calls()` reads it as a tibble.
- `rate()` records people's verdicts and corrections by call id;
  `rated()` turns them into an evaluation set that grows with use.
- `ai_version()` names what the function sends; models don't change it,
  and languages share it.
- `write_ai()` and `read_ai()` save and load a function, checking it
  still sends exactly what it did.

**Answers to the check at the top.** (1) Your program does: the model
asks, functai runs the R function and sends the result back. (2) In the
log folder, next to the calls; `rated(fn)` gives them back as rows with
the right answers, ready for `evaluate()`. (3) Everything the function
sends besides its inputs (instruction, layout, worked examples). The
model, and the language it was written in, don't change it.

## The whole series, in one page

You have now done, in R, the whole life of an AI function:

1. **Write it**: a name, a sentence, typed inputs and a typed answer
   (`ai()`); read what the model reads (`ai_render()`).
2. **Get typed answers**: factors, integers, optional values, records,
   lists; several outputs at once (tutorial 2).
3. **Measure it**: `evaluate()`, intervals, baselines, confusion
   matrices (tutorial 3).
4. **Improve it without fooling yourself**: rules, examples, a teacher;
   three piles of rows; paired tests (tutorial 4).
5. **Choose the model**: accuracy, cost and speed on your rows, and a
   rule written before the chart (tutorial 5).
6. **Decide with it**: costs of mistakes, the model reading and R
   ruling, probabilities from votes, the cheapest action (tutorial 6).
7. **Fit it into tidymodels**: resampling, tuning, probabilities, a
   classical student (tutorial 7).
8. **Live with it**: tools, the log, ratings, versions, saving (here).

Three habits carry through all of it: look at what the model reads,
measure with intervals on rows you didn't tune on, and count the cost in
dollars.
