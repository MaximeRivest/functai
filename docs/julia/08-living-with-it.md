# 8. Living with it

*A function in a notebook is an experiment. A function the shop relies on needs more: it has to look things up instead of guessing, leave a record of every answer, learn from the people who correct it, and travel to wherever it's needed. By the end you will have done all four, and watched a call as it happens.*

**Can you skip this one?** If you can answer these, you've finished the series. The answers are at the bottom.

1. When a model "calls a tool", who runs the code?
2. Where does a correction a person makes to an answer end up, and how do you use it?
3. What does `version` identify, and what doesn't change it?

**You will:** give the model a Julia function to call (a tool), watch a call stream with `eachevent`, read the call log, rate answers with `rate` and turn the ratings into rows with `rated`, compare versions, and save a function with `FunctAI.save` and load it back with `FunctAI.load`.

## Setting up

```julia
using FunctAI, DataFrames, Statistics, Random

log_folder = mktempdir()
FunctAI.configure!(lm = "gpt-6-luna", log_calls = log_folder);
```

In real use you'd write `FunctAI.configure!(log_calls = true)` (or set `FUNCTAI_LOG_CALLS=1` in the environment), and calls would go to one folder on your machine, the same one FunctAI in Python, TypeScript and R uses. We keep this tutorial's calls in a folder of their own.

## Looking things up

Customers ask where their order is. The model can't know: the answer is in the shop's order system, which here is a table:

```julia
orders = DataFrame([
    (order = "A-1042", status = "in transit", note = "held at the Montreal depot since September 8"),
    (order = "C-3319", status = "delivered",  note = "left with a neighbour at 14 Elm Street on September 20"),
    (order = "A-1299", status = "delivered",  note = "the courier's photo shows it at the side door, September 22"),
    (order = "B-2417", status = "processing", note = "waiting for stock; ships October 2"),
    (order = "A-1350", status = "in transit", note = "delayed by the carrier; new estimate September 30"),
    (order = "C-3480", status = "processing", note = "not shipped yet, so the address can still be changed")])
```

```output
6×3 DataFrame
 Row │ order   status      note
     │ String  String      String
─────┼───────────────────────────────────────────────────────
   1 │ A-1042  in transit  held at the Montreal depot since…
   2 │ C-3319  delivered   left with a neighbour at 14 Elm …
   3 │ A-1299  delivered   the courier's photo shows it at …
   4 │ B-2417  processing  waiting for stock; ships October…
   5 │ A-1350  in transit  delayed by the carrier; new esti…
   6 │ C-3480  processing  not shipped yet, so the address …
```

A **tool** is a Julia function the model may ask for. Write it as usual, with a docstring: the docstring tells the model what it does, and the method's argument names and types tell it how to call it:

```julia
looked_up = String[]

"Look up an order's delivery status by its number: a letter, a dash and four digits, like A-1042."
function lookup_order(order::String)
    push!(looked_up, order)                              # so we can see what the model asked for
    i = findfirst(==(uppercase(order)), orders.order)
    i === nothing && return "there is no order with that number"
    "$(orders.status[i]): $(orders.note[i])"
end

tool(lookup_order)
```

```output
tool lookup_order(order::String)
```

Get the mental model right, because it's easy to get wrong. The model does **not** run your function. It replies, in effect, "please run `lookup_order` with `A-1042`". FunctAI runs your Julia function, in your session, sends the result back, and the model then writes its answer, or asks for another lookup. The model's skill is knowing *when* to look something up and *what* to ask for.

```julia
@ai tools = [lookup_order] function where_is(message::String)::String
    """
    Tell the customer where their order is, in one or two friendly sentences. Use only what the
    order system says; never guess a date. If the message has no order number, ask for it.
    """
end

tickets = DataFrame(FunctAI.tickets())
asked = vcat(filter(:order_id => id -> !ismissing(id) && id in orders.order, tickets).message,
             ["Where is my order Z-9999?", "Hi, where's my parcel? It's been ages."])

for (message, reply) in zip(asked, where_is.(asked))
    println(message, "\n  -> ", reply, "\n")
end
looked_up
```

```output
Hi, my order A-1042 still hasn't arrived and it's been three weeks.
  -> Your order is in transit and has been held at the Montreal depot since September 8. I’m sorry it hasn’t arrived yet.

Tracking for C-3319 hasn't moved since Monday.
  -> Your order is marked as delivered and was left with a neighbour at 14 Elm Street on September 20.

Order c3319 was delivered to my neighbour's address instead of mine.
  -> The order system says your package was delivered to a neighbour at 14 Elm Street on September 20.

My parcel says delivered but it's not here. Order A-1299.
  -> The courier’s photo shows your parcel at the side door. It’s marked as delivered on September 22.

When will order B-2417 ship? It says 'processing' for a week.
  -> Your order is processing while it waits for stock, and it’s scheduled to ship October 2.

Order #A-1350 is two weeks late. Any news?
  -> Your order is in transit but delayed by the carrier. The new estimated delivery date is September 30.

Can I change the delivery address for order C-3480? It hasn't shipped yet.
  -> Your order is processing and hasn’t shipped yet, so the delivery address can still be changed.

Where is my order Z-9999?
  -> I can’t find an order with number Z-9999. Could you check the number and send it again?

Hi, where's my parcel? It's been ages.
  -> I’m sorry it’s taking so long. Could you share your order number so I can check where your parcel is?

8-element Vector{String}:
 "B-2417"
 "Z-9999"
 "C-3319"
 "C-3319"
 "A-1042"
 "A-1350"
 "A-1299"
 "C-3480"
```

Read the replies against `orders`, and `looked_up` against the messages. The unknown order is reported, not invented; the message with no number gets a question, because there was nothing to look up; every date comes from the table.

Now look closely at any message that writes its number oddly ("c3319"). Did the model look up `C-3319`, or ask the customer for a number they had already given? Models read a tool's description literally: we said "a letter, a dash and four digits". If it asked again, the fix is one sentence in the docstring ("customers often leave out the dash or write the letter in lower case"), or one line of Julia in the tool, then reading the replies again. This is the everyday work of living with an AI function: read what it did, and fix the words or the code.

## Watching a call

A reply that takes a few seconds can be shown as it's written. `stream` starts the call and returns at once; iterating it gives the answer's text piece by piece, and `fetch` the finished answer, typed, exactly what calling the function returns:

```julia
s = stream(where_is, "Where is my order A-1042?")
pieces = collect(s)
(pieces = length(pieces), answer = fetch(s))
```

```output
(pieces = 17, answer = "Your order is in transit and has been held at the Montreal depot since September 8.")
```

`eachevent` shows everything that happened in the call, in order: the tool it asked for, what the tool returned, the text, the end:

```julia
for e in eachevent(s)
    e.kind === :tool_call   && println("asked for ", e.name, " ", e.input)
    e.kind === :tool_result && println("the tool said: ", e.output)
    e.kind in (:started, :done) && println(e.kind)
end
```

```output
started
asked for lookup_order OrderedCollections.OrderedDict{String, Any}("order" => "A-1042")
the tool said: in transit: held at the Montreal depot since September 8
done
```

In an app you'd print each piece as it arrives (`stream(print, where_is, message)` does that and returns the answer), and `close(s)` stops a call nobody is waiting for any more.

## The log

Every call so far is a line in the log folder. `calls` reads them all, or one function's, as a table:

```julia
log = DataFrame(calls(folder = log_folder))
combine(groupby(log, :name), nrow => :calls, :seconds => median => :seconds, :total_tokens => sum => :tokens)
```

```output
1×4 DataFrame
 Row │ name      calls  seconds  tokens
     │ String    Int64  Float64  Int64
─────┼──────────────────────────────────
   1 │ where_is     10  7.18404    4007
```

```julia
replies = DataFrame(calls(where_is; folder = log_folder))
first(select(replies, :model, :seconds, :inputs => ByRow(i -> i["message"]) => :message,
             :outputs => ByRow(o -> o["result"]) => :reply), 3)
```

```output
3×4 DataFrame
 Row │ model       seconds   message                            reply                      ⋯
     │ String      Float64   String                             String                     ⋯
─────┼──────────────────────────────────────────────────────────────────────────────────────
   1 │ gpt-6-luna  25.0375   Hi, my order A-1042 still hasn't…  Your order is in transit a ⋯
   2 │ gpt-6-luna   7.19032  Tracking for C-3319 hasn't moved…  Your order is marked as de
   3 │ gpt-6-luna   7.18837  Order c3319 was delivered to my …  The order system says your
                                                                            1 column omitted
```

Each call knows its function's name, its **version**, the model, the time it took, its tokens and, since the log keeps content by default, its inputs and outputs (`log_content = false` keeps only their sizes, for private data). That's the raw material for everything below.

## People's corrections

Here's the loop that keeps a function honest after it ships. Someone reviews a sample of real answers, says which were right, and corrects the wrong ones. Let's play the reviewer, using the right answers we happen to have:

```julia
@enum Team shipping billing product account

@ai function team(message::String)::Team
    "Which team should answer this customer message?"
end

sample = tickets[shuffle(Xoshiro(8), 1:nrow(tickets))[1:30], :]
predictions = predict.(team, sample.message)        # each keeps its answer and its call's id
first(DataFrame(category = sample.category, answer = [p.value for p in predictions],
                call = [p.call for p in predictions]), 6)
```

```output
6×3 DataFrame
 Row │ category  answer    call
     │ String    Team      String
─────┼───────────────────────────────────────────────────────
   1 │ product   product   01a0e4e9-a0ad-7f72-b8a6-099eabdb…
   2 │ shipping  shipping  01a0e4e9-a0d6-7e0e-9f86-aca3db0c…
   3 │ shipping  shipping  01a0e4e9-a0d6-7344-851d-c4925d56…
   4 │ shipping  shipping  01a0e4e9-a0d7-7254-82cd-767bf059…
   5 │ product   product   01a0e4e9-a0d7-77ce-ac5a-35016e74…
   6 │ product   product   01a0e4e9-a0d7-722d-964c-9782a412…
```

`rate` records a verdict on a call. A wrong one carries the right answer, in the function's own type:

```julia
by_name = Dict(string(t) => t for t in instances(Team))
for (p, truth) in zip(predictions, sample.category)
    p.value == by_name[truth] ? rate(p, :right) : rate(p; answer = by_name[truth])
end
```

`rated` turns the reviews back into rows with known answers, typed like the function's inputs and answer:

```julia
reviewed = DataFrame(rated(team))
select(reviewed, :message, :result, :rating)
```

```output
30×3 DataFrame
 Row │ message                            result    rating
     │ String                             Team      String
─────┼─────────────────────────────────────────────────────
   1 │ Can the dutch oven go on an indu…  product   right
   2 │ The vase arrived with a big crac…  shipping  right
   3 │ Two of the six plates were broke…  shipping  right
   4 │ The mug arrived in pieces.         shipping  right
   5 │ The mixer's bowl doesn't lock in…  product   right
   6 │ The non-stick coating is peeling…  product   right
   7 │ My parcel says delivered but it'…  shipping  right
   8 │ Package arrived but the bowl ins…  shipping  right
  ⋮  │                 ⋮                     ⋮        ⋮
  24 │ How often should I descale the k…  product   right
  25 │ I was charged for an order I can…  billing   right
  26 │ The thermos doesn't keep coffee …  product   right
  27 │ Refund please: the towels are mu…  billing   right
  28 │ i cant log in it says my account…  account   right
  29 │ When will order B-2417 ship? It …  shipping  right
  30 │ The knife rusted after I left it…  product   right
                                            15 rows omitted
```

Those rows are an evaluation set that grows by itself as people review. Any new version of the function is measured on it:

```julia
@ai function team_rules(message::String)::Team
    """
    Which team should answer this customer message? House rules: anything wrong with the delivery
    itself (late, lost, wrong item, broken on arrival) is shipping; anything about money, and every
    request for money back whatever the reason, is billing; problems in use and product questions
    are product; signing in, passwords and personal data are account.
    """
end

DataFrame([(version = name, score = e.score, low = e.low, high = e.high, n = length(e))
           for (name, e) in ["team" => evaluate(team, reviewed), "team_rules" => evaluate(team_rules, reviewed)]])
```

```output
2×5 DataFrame
 Row │ version     score    low       high     n
     │ String      Float64  Float64   Float64  Int64
─────┼───────────────────────────────────────────────
   1 │ team            1.0  0.886487      1.0     30
   2 │ team_rules      1.0  0.886487      1.0     30
```

The ratings live in the same folder as the calls, in the same format across languages: a correction made from Python, TypeScript or R shows up in Julia's `rated`, and the other way round.

## Versions

Every call in the log carries the version of the function that made it:

```julia
(team = version(team), team_rules = version(team_rules))
```

```output
(team = "sha256:99ae724adb8e3da04da85dddbeec5dc1feed95478e91e3c027342bba47f6013c", team_rules = "sha256:4578cc4c2c60dce4fd1ee35eda8a4be03d1a479cc0dd36c8b6b3eeb07a19adc2")
```

```julia
combine(groupby(DataFrame(calls(team; folder = log_folder)), :version), nrow => :calls)
```

```output
1×2 DataFrame
 Row │ version                            calls
     │ String                             Int64
─────┼──────────────────────────────────────────
   1 │ sha256:99ae724adb8e3da04da85dddb…     60
```

A version is a fingerprint of everything the function sends besides its inputs: the instruction, the layout, the worked examples. Change a word of the description and the version changes. Change the *model* and it doesn't: the model is a setting, so you can compare models on one version.

```julia
version(configure(team; lm = "claude-haiku-4-5")) == version(team)
```

```output
true
```

And the same function written in Python, TypeScript or R has the same version, so their calls and ratings add up.

## Saving

A function is worth keeping once you've measured it. `FunctAI.save` writes it as a folder:

```julia
dir = joinpath(mktempdir(), "team_rules")
FunctAI.save(dir, team_rules)
readdir(dir)
```

```output
1-element Vector{String}:
 "functai.json"
```

```julia
print(join(first(readlines(joinpath(dir, "functai.json")), 12), "\n"))
```

```output
{
 "functai_saved": 1,
 "language": "julia",
 "entry": "__main__:team_rules",
 "created": "2026-09-27T22:08:50+00:00",
 "functai": "0.1.0",
 "nodes": {
  "__main__:team_rules": {
   "kind": "ai",
   "module": "__main__",
   "name": "team_rules",
   "ai": {
```

The folder holds everything the function sends, and fingerprints of the requests it makes. `FunctAI.load` loads it back and checks, before any call, that it would send exactly what it sent when it was saved. A folder holds JSON shapes, not Julia types; `types` gives the answer its type back:

```julia
team_loaded = FunctAI.load(dir; types = (result = Team,))
version(team_loaded) == version(team_rules)
```

```output
true
```

```julia
team_loaded("The courier left my package in the rain and the box is soaked.")
```

```output
shipping::Team = 0
```

`FunctAI.load` also loads folders saved by FunctAI in Python, TypeScript and R, and refuses, with the reason, what only the saving language can run (code of its own around the model, tools, a trained model). A function with tools can't be saved from Julia for the same reason: a tool is code.

## What it cost

```julia
bill = dropmissing(DataFrame(calls(folder = log_folder)), :model)
(calls = nrow(bill),
 dollars = sum(bill.input_tokens .* 0.10 .+ (bill.total_tokens .- bill.input_tokens) .* 0.50) / 1e6)   # gpt-6-luna, 2026-09-27
```

```output
(calls = 101, dollars = 0.0030575)
```

## Your turn

1. Add a second tool, `cancel_order(order::String)`, that only works when the order is still processing. Ask `where_is` (with both tools) to cancel `C-3480` and then `A-1042`. What does it do with each?
2. Rate five of `where_is`'s replies. Which ones would you mark wrong, and what would the right reply have been?
3. Save `team_rules` with a worked example or two (`with_demos`), load it back, and check the versions differ from the one without examples.

## What you learned

- A tool is a Julia function with a docstring (`tools = [f]`): the model asks, FunctAI runs it and sends the result back.
- `stream` shows a call as it's made: the text piece by piece, `eachevent` for everything, `fetch` for the typed answer.
- The call log records every call: `calls` reads it as a table.
- `rate` records people's verdicts and corrections; `rated` turns them into an evaluation set that grows with use.
- `version` names what the function sends; the model doesn't change it, and neither does the language it was written in.
- `FunctAI.save` and `FunctAI.load` save and load a function, checking it sends the same bytes; folders from Python, TypeScript and R load too.

**Answers to the check at the top.** (1) Your program does: the model asks, FunctAI runs the Julia function and sends the result back. (2) In the log folder, next to the calls; `rated(f)` gives them back as rows with the right answers, ready for `evaluate`. (3) Everything the function sends besides its inputs (instruction, layout, worked examples). The model, and the language it was written in, don't change it.

## The whole game

You have now done, in Julia, the whole life of an AI function:

1. **Write it**: `@ai`, a sentence, typed inputs and a typed answer; read what the model reads (`FunctAI.prompt`).
2. **Get typed answers**: `@enum`s, `Union{Int,Missing}`, structs, vectors, several answers at once.
3. **Measure it**: `evaluate`, intervals, baselines, confusion tables, run-to-run variation.
4. **Improve it without fooling yourself**: rules, examples, a teacher, a search; three piles; `compare`.
5. **Choose the model**: accuracy, cost and speed on your rows, and a rule written first.
6. **Decide with it**: costs of mistakes, the model reading and Julia deciding, calibrated probabilities.
7. **Fit it into MLJ and formulas**: machines, resampling, tuning, a feature in a GLM.
8. **Live with it**: tools, streaming, the log, ratings, versions, saving (here).

Three habits carry through all of it: look at what the model reads, measure with intervals on rows you didn't tune on, and count the cost in dollars.
