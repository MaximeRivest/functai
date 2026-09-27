# GEPA: the instruction rewritten from the function's mistakes (Agrawal et
# al., 2025), with FunctAI's changes. The algorithm, what it changes and why
# are in design/04-gepa.md; Python's `GEPA`, R's `gepa()` and TypeScript's
# `gepa()` are the same algorithm, and its two prompts send the same words.
#
# The contract fixes what improving means, not its random choices: a seed
# makes a search repeatable here, not equal to another language's.

const REFLECT_TEXT = """
You improve the instruction of a function that a language model runs. You are given what the
function takes and returns, its current instruction, and cases it was run on: each with its inputs,
the answer it gave, its score and feedback. Find what the instruction is missing, or gets wrong, that
explains the mistakes, and write an improved instruction. Write general rules a careful person could
follow on new cases; never copy an input or describe these particular cases. Keep what already works.
The instruction is everything the model is told besides the inputs: keep the task, and say what each
output must be. Instructions listed as tried did not do better: try something different. Reply with
the new instruction only."""

const COMBINE_TEXT = """
Two instructions for the same function each get right some cases the other gets wrong. Write one
instruction that keeps what makes each of them right, without repeating itself. The instruction is
everything the model is told besides the inputs: keep the task, and say what each output must be.
Reply with the new instruction only."""

"One candidate instruction in the pool: its parents, how it was made, its score on each choosing row, and its children that did not beat it."
mutable struct Candidate
    instruction::String
    parents::Vector{Int}
    kind::Symbol                    # :written, :reflect or :combine
    scores::Vector{Float64}
    tried::Vector{String}
end
Candidate(text, parents, kind) = Candidate(text, parents, kind, Float64[], String[])

"One row's result for one instruction: its score, its outputs (`nothing` when the call failed), its error."
const RowRun = @NamedTuple{score::Float64, outputs::Union{Nothing,NamedTuple}, error::Union{Nothing,String}}

# ------------------------------------------------------------------ parts (the same in every language)

"""
The candidates on the Pareto frontier (best on at least one row, dominated by
none), with how many rows each is best on.
"""
function frontier(scores::AbstractVector{<:AbstractVector{<:Real}})
    n = isempty(scores) ? 0 : length(first(scores))
    wins = OrderedDict{Int,Int}()
    n == 0 && return OrderedDict(1 => 1)
    for r in 1:n
        best = maximum(s[r] for s in scores)
        for (k, s) in enumerate(scores)
            s[r] == best && (wins[k] = get(wins, k, 0) + 1)
        end
    end
    dominated(a) = any(b -> b != a && all(scores[b] .>= scores[a]) && any(scores[b] .> scores[a]), keys(wins))
    OrderedDict(k => w for (k, w) in sort!(collect(wins); by=first) if !dominated(k))
end

"Two frontier candidates that each win rows the other loses: the pair with the most such rows on its weaker side, or `nothing`."
function best_pair(scores, front)
    ks = sort!(collect(keys(front)))
    best, pair = 0, nothing
    for x in eachindex(ks), y in eachindex(ks)
        x < y || continue
        a, b = scores[ks[x]], scores[ks[y]]
        w = min(count(a .> b), count(b .> a))
        w > best && ((best, pair) = (w, (ks[x], ks[y])))
    end
    pair
end

"A frontier candidate, with probability proportional to the rows it is best on."
function pick(front, rng)
    x = rand(rng) * sum(values(front))
    for (k, w) in front
        x -= w
        x < 0 && return k
    end
    last(collect(keys(front)))
end

"An answer as the teacher reads it: text as it is, anything else as JSON."
function answer_text(v)
    j = logvalue(v)
    j isa AbstractString ? String(j) : LMCC.json_text(j)
end

"A JSON Schema shape in words, for the teacher (the same words in every language)."
function shape_words(shape::AbstractDict)
    options = get(shape, "anyOf", get(shape, "oneOf", nothing))
    if options !== nothing
        kept = [o for o in options if get(o, "type", nothing) != "null"]
        words = isempty(kept) ? "nothing" : join(shape_words.(kept), " or ")
        return words * (length(kept) < length(options) ? ", or nothing" : "")
    end
    haskey(shape, "enum") && return "one of " * join(answer_text.(shape["enum"]), ", ")
    t = get(shape, "type", nothing)
    t == "array" && return "a list of " * shape_words(something(get(shape, "items", nothing), JObj()))
    if t == "object"
        props = something(get(shape, "properties", nothing), JObj())
        return isempty(props) ? "an object" : "a record of " * join(("$k ($(shape_words(v)))" for (k, v) in props), ", ")
    end
    get(Dict("string" => "text", "integer" => "a whole number", "number" => "a number", "boolean" => "true or false"), t, "a value")
end

"What a function takes and returns, one field a line, with its type and words."
function fields_text(f::AIFunction)
    line(x) = "- $(x.name): $(shape_words(x.shape))" * (x.desc === nothing ? "" : ". $(x.desc)")
    join(["Inputs:", line.(f.definition.inputs)..., "Outputs:", line.(f.definition.outputs)...], "\n")
end

"Does the instruction quote an input of `at_least` characters or more, verbatim (case and spacing ignored)?"
function copies_an_input(text::AbstractString, rows, inputs; at_least::Int=30)
    said = normalize_text(text)
    any(rows) do row
        any(inputs) do k
            v = get(row, k, nothing)
            v isa AbstractString || return false
            n = normalize_text(v)
            length(n) >= at_least && occursin(n, said)
        end
    end
end

"Words about one answer: \"right\", \"wrong: the right answer is …\", or the call's error."
function default_feedback(mapping::AbstractDict, single::Bool)
    (row, outputs, error) -> begin
        (error !== nothing || outputs === nothing) && return "the call failed: $error"
        wrong = [k for (k, col) in mapping
                 if haskey(row, Symbol(col)) && exact_match(JObj(k => row[Symbol(col)]), JObj(k => outputs[Symbol(k)]))["exact_match"] != 1]
        isempty(wrong) && return "right"
        "wrong: " * join(("the right $(single || k == "result" ? "answer" : k) is $(answer_text(row[Symbol(mapping[k])]))" for k in wrong), "; ")
    end
end

"The cases the teacher reads: each row's inputs, the answer given, its score and the feedback."
function cases_text(rows, batch, results, inputs, outputs, feedback)
    lines = String[]
    for (n, (i, r)) in enumerate(zip(batch, results))
        row = rows[i]
        push!(lines, "Case $n")
        for k in inputs
            haskey(row, Symbol(k)) && push!(lines, "  $k: $(answer_text(row[Symbol(k)]))")
        end
        if r.outputs === nothing
            push!(lines, "  answer given: (none)")
        else
            for k in outputs
                push!(lines, "  answer given$(k == "result" ? "" : " $k"): $(answer_text(get(r.outputs, Symbol(k), nothing)))")
            end
        end
        push!(lines, "  score: $(r.score == round(r.score) ? Int(r.score) : r.score)",
              "  feedback: $(feedback(row, r.outputs, r.error))")
    end
    join(lines, "\n")
end

# ------------------------------------------------------------------ the search

"""
    gepa(f, data; selection, teacher, budget = 300, minibatch = 4, expected,
         metric, feedback, seed = 0, concurrency) -> (f, trials)

Rewrite the instruction from the function's mistakes. A teacher model reads
the function's answers on a few rows with feedback in words ("wrong: the
right answer is billing") and writes a better instruction. The instructions
it writes are kept in a pool, each scored on rows the teacher is never shown
(`selection`, or half of `data` at random). Candidates that are best on at
least one of those rows stay in the pool, so ideas that fix different
mistakes both survive, and every fourth step two of them are combined. The
teacher sees what it tried that failed; a proposal that copies an input is
dropped; ties go to the shorter instruction; no row runs twice for one
instruction. It stops at `budget` calls of `f` (the teacher's calls are
counted apart, and logged, marked as part of the optimization).

Returns an improved copy (`f` itself when nothing beat the written
instruction) and every instruction tried, as rows: `candidate` (its number in
the pool; 1 is the written one; `missing` when it did not join), `kind`
(`:written`, `:reflect`, `:combine`), `parents`, `minibatch_parent` and
`minibatch` (how many of the same minibatch rows its parent and it got
right), `score` (its mean on the choosing rows), `length`, `note`, `calls`
(of `f` so far), `instruction`, `chosen`.

The chosen score flatters: it is the best of many on the choosing rows (the
winner's curse). Measure the result with [`evaluate`](@ref) on rows it never
saw.

- `data`, `selection`: tables (or vectors of rows) with the right answers, in
  columns named like the outputs, or `expected` (as for `evaluate`).
- `metric`: `(row, outputs) -> score`, 1 meaning right (default: exact match).
- `feedback`: `(row, outputs, error) -> words` (`outputs` is `nothing` when
  the call failed): what the teacher reads about each answer. Default:
  "right", "wrong: the right answer is …", or the call's error.
- `teacher`: the model that writes instructions (`"gpt-6-sol"`); default
  the function's own.

```julia
better, trials = gepa(configure(refund; lm = "gpt-5.4-nano"), examples;
                      selection = dev, teacher = "gpt-6-sol", budget = 300)
DataFrame(trials)
evaluate(better, test)              # the honest number: rows the search never saw
```

It improves one AI function's instruction and leaves its worked examples as
they are; add examples after it with [`labeled_few_shot`](@ref).
"""
function gepa(f::AIFunction, data; selection=nothing, teacher=nothing, budget::Integer=300, minibatch::Integer=4,
              expected=nothing, metric=nothing, feedback=nothing, seed::Integer=0, concurrency=nothing)
    minibatch = max(1, minibatch)
    rng = Xoshiro(seed)
    rows = [row_dict(r) for r in rows_of(data)]
    outputs, inputs = output_names(f), input_names(f)
    columns = Set(k for r in rows for k in keys(r))
    mapping = expected === nothing ? OrderedDict{String,String}(o => o for o in outputs if o in columns) :
              expected isa Union{AbstractString,Symbol} ? OrderedDict{String,String}(answer_name(f) => String(expected)) :
              OrderedDict{String,String}(String(k) => String(v) for (k, v) in pairs(expected))
    metric === nothing && isempty(mapping) &&
        throw(ArgumentError("gepa needs the right answers: a column named like an output ($(join(outputs, ", "))), or expected"))
    if selection === nothing
        length(rows) >= 2 || throw(ArgumentError("gepa needs at least 2 rows: some to learn from, some to choose with"))
        shuffled = shuffle(rng, rows)
        half = length(shuffled) ÷ 2
        select_rows, feed_rows = shuffled[1:half], shuffled[half+1:end]
    else
        select_rows, feed_rows = [row_dict(r) for r in rows_of(selection)], rows
    end
    isempty(select_rows) && throw(ArgumentError("gepa needs rows to choose with"))
    isempty(feed_rows) && throw(ArgumentError("gepa needs rows to learn from"))
    nt(row) = NamedTuple(Symbol(k) => v for (k, v) in row)
    feed_nt = nt.(feed_rows)
    judge = something(feedback, default_feedback(mapping, length(outputs) == 1))
    run = new_id()
    written_text = instructions(f)
    fields = fields_text(f)

    # the teacher's two functions: fixed layout, the function's router and log, the teacher's model
    meta = Dict{Symbol,Any}(:module_name => "functai.meta", :include_name => false, :adapter => :xml,
                            :reasoning => false, :temperature => 1.0)
    for k in (:router, :log_calls, :log_content, :lm)
        haskey(f.own, k) && (meta[k] = f.own[k])
    end
    teacher === nothing || (meta[:lm] = String(teacher))
    reflect = AIFunction("_reflect", REFLECT_TEXT; inputs=(fields=String, instruction=String, cases=String, tried=String),
                         output=String, meta...)
    combine = AIFunction("_combine", COMBINE_TEXT; inputs=(fields=String, first=String, second=String), output=String, meta...)
    tagged(g) = with_settings(g; caller=Dict("optimization" => run))

    calls = Ref(0)
    reflections = Ref(0)
    memo = Dict{Tuple{Symbol,Int,String},RowRun}()
    version_of(text) = text == written_text ? f : with_instructions(f, text)

    "Score, outputs and error of each row for an instruction; each (instruction, row) runs once."
    function scores_of(text, set::Symbol, idx)
        source = set === :feed ? feed_rows : select_rows
        todo = [i for i in idx if !haskey(memo, (set, i, text))]
        if !isempty(todo)
            e = tagged(() -> evaluate(version_of(text), source[todo]; expected, metric, concurrency))
            calls[] += length(todo)
            s = scores(e)
            for (j, i) in enumerate(todo)
                r = e.results[j]
                memo[(set, i, text)] = (score=s[j], outputs=r.outputs, error=r.error)
            end
        end
        [memo[(set, i, text)] for i in idx]
    end
    batch_sum(text, batch) = sum(r.score for r in scores_of(text, :feed, batch))
    score_select!(c) = (c.scores = [r.score for r in scores_of(c.instruction, :select, eachindex(select_rows))])
    solved(c) = all(i -> get(memo, (:feed, i, c.instruction), (score=0.0,)).score >= 1, eachindex(feed_rows))

    trials = NamedTuple[]
    function record!(c, k, before, after, note)
        count_of(x) = x === nothing ? missing : isinteger(x) ? Int(x) : x      # right-or-wrong scores: a count of rows
        push!(trials, (candidate=something(k, missing), kind=c.kind, parents=copy(c.parents),
                       minibatch_parent=count_of(before), minibatch=count_of(after),
                       score=k === nothing ? missing : mean_of(c.scores), length=length(c.instruction), note=note,
                       calls=calls[], instruction=c.instruction))
    end
    order = Int[]
    function next_batch()
        batch = Int[]
        while length(batch) < min(minibatch, length(feed_rows))
            isempty(order) && append!(order, shuffle(rng, collect(eachindex(feed_rows))))
            i = pop!(order)
            i in batch || push!(batch, i)
        end
        batch
    end

    pool = [Candidate(written_text, Int[], :written)]
    score_select!(pool[1])
    record!(pool[1], 1, nothing, nothing, "the written instruction")
    function admit!(child, before, after)
        if calls[] + length(select_rows) > budget
            record!(child, nothing, before, after, "better on the minibatch; no budget left to score it")
            return
        end
        score_select!(child)
        push!(pool, child)
        record!(child, length(pool), before, after, "joined the pool")
    end
    is_new(text) = !isempty(text) && !any(c -> c.instruction == text, pool)
    ask(g, args...) = trim_white(string(something(tagged(() -> g(args...)), "")))

    step = 0
    while calls[] + 2minibatch <= budget && step < budget      # steps: a bound when rows are remembered
        step += 1
        front = frontier([c.scores for c in pool])
        if all(k -> solved(pool[k]), keys(front))
            record!(pool[1], nothing, nothing, nothing, "right on every feedback row: no mistake left to learn from")
            break
        end
        pair = step % 4 == 0 ? best_pair([c.scores for c in pool], front) : nothing
        if pair !== nothing
            a, b = pair
            text = ask(combine, fields, pool[a].instruction, pool[b].instruction)
            reflections[] += 1
            child = Candidate(text, [a, b], :combine)
            is_new(text) || (record!(child, nothing, nothing, nothing, "no new instruction"); continue)
            batch = next_batch()
            before = max(batch_sum(pool[a].instruction, batch), batch_sum(pool[b].instruction, batch))
            after = batch_sum(text, batch)
            after >= before ? admit!(child, before, after) :
                record!(child, nothing, before, after, "worse on the minibatch than its better parent")
            continue
        end
        k = pick(front, rng)
        parent = pool[k]
        batch = next_batch()
        results = scores_of(parent.instruction, :feed, batch)
        before = sum(r.score for r in results)
        all(r -> r.score >= 1, results) && continue           # nothing to learn from these rows
        tried = parent.tried[max(1, end - 2):end]
        text = ask(reflect, fields, parent.instruction,
                   cases_text(feed_nt, batch, results, inputs, outputs, judge),
                   isempty(tried) ? "(none)" : join(("Tried $i:\n$x" for (i, x) in enumerate(tried)), "\n\n"))
        reflections[] += 1
        child = Candidate(text, [k], :reflect)
        is_new(text) || (record!(child, nothing, before, nothing, "no new instruction"); continue)
        if copies_an_input(text, feed_rows, inputs)
            push!(parent.tried, "$text\n(dropped: it copied an input instead of stating a rule)")
            record!(child, nothing, before, nothing, "copied an input: dropped")
            continue
        end
        after = batch_sum(text, batch)
        if after > before
            admit!(child, before, after)
        else
            push!(parent.tried, text)
            record!(child, nothing, before, after, "not better on the minibatch")
        end
    end
    best = 1                                                   # ties go to the shorter instruction
    for (i, c) in enumerate(pool)
        s, b = mean_of(c.scores), mean_of(pool[best].scores)
        (s > b || (s == b && length(c.instruction) < length(pool[best].instruction))) && (best = i)
    end
    out = best == 1 ? f : with_instructions(f, pool[best].instruction)
    table = [(; t..., chosen=t.candidate === best) for t in trials]
    (out, table)
end

mean_of(xs) = isempty(xs) ? 0.0 : sum(xs) / length(xs)
