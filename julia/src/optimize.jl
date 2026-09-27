# Improving a function: choosing its worked examples and its instruction.
# Each returns an improved copy; the function you pass is unchanged.
# Improving never edits the fields or the layout, only what the version
# counts as the function's state (the instruction and the demos), so the
# copy has a new version.
#
# The contract fixes what improving means, not its random choices
# (design/01-many-languages.md, decision 3): a seed makes a run repeatable
# here, not equal to Python's run with the same seed.

"A row as a worked example: its inputs and the outputs it has an answer for (none: no example)."
function labeled_demo(f::AIFunction, row::AbstractDict)
    ins = JObj(n => jsonvalue(row[n]) for n in input_names(f) if haskey(row, n) && row[n] !== missing)
    outs = JObj(n => jsonvalue(row[n]) for n in output_names(f) if haskey(row, n) && row[n] !== missing)
    isempty(outs) ? nothing : LMCC.jobj("inputs" => ins, "outputs" => outs)
end

"The rows of `data`, each with the answer under its output's name when `expected` names another column."
function training_rows(f::AIFunction, data, expected)
    rows = [row_dict(r) for r in rows_of(data)]
    expected === nothing && return rows
    mapping = expected isa Union{AbstractString,Symbol} ? Dict(answer_name(f) => String(expected)) :
              Dict(String(k) => String(v) for (k, v) in pairs(expected))
    for r in rows, (o, c) in mapping
        haskey(r, c) && (r[o] = r[c])
    end
    rows
end

"The default judge of a run: exact_match against the row's answers."
function default_metric(f::AIFunction)
    outs = output_names(f)
    (row, outputs) -> begin
        answers = JObj(String(k) => v for (k, v) in pairs(row) if String(k) in outs)
        exact_match(answers, JObj(k => outputs[Symbol(k)] for k in keys(answers)))["exact_match"]
    end
end

"""
    labeled_few_shot(f, data; k = 16, seed = 0, sample = true, expected)

A copy of `f` whose worked examples are up to `k` rows of `data` with known
answers (a seeded random sample; `sample = false`: the first `k`). Nothing is
called.
"""
function labeled_few_shot(f::AIFunction, data; k::Integer=16, seed::Integer=0, sample::Bool=true, expected=nothing)
    rows = training_rows(f, data, expected)
    chosen = sample ? shuffle(Xoshiro(seed), rows)[1:min(k, length(rows))] : rows[1:min(k, length(rows))]
    remake(f; demos=Any[d for d in (labeled_demo(f, r) for r in chosen) if d !== nothing])
end

"""
    bootstrap_few_shot(f, data; metric, threshold, max_bootstrapped = 4, max_labeled = 16,
                       teacher, seed = 0, concurrency = 4, expected)

Run `f` (or a `teacher`: a stronger model's name, or another AI function of
the same signature) on rows with known answers; the runs `metric` accepts
become worked examples, whole turns included (reasoning, tool calls).
Labeled rows fill the rest, up to `max_labeled`. Returns an improved copy.
"""
function bootstrap_few_shot(f::AIFunction, data; metric=nothing, threshold=nothing, max_bootstrapped::Integer=4,
                            max_labeled::Integer=16, teacher=nothing, seed::Integer=0, concurrency::Integer=4, expected=nothing)
    rows = training_rows(f, data, expected)
    runner = teacher === nothing ? f : teacher isa AbstractString ? configure(f; lm=teacher) :
             teacher isa AIFunction ? teacher : throw(ArgumentError("teacher is a model name or an AI function"))
    judge = something(metric, default_metric(f))
    passes(score) = threshold === nothing ? score > 0 : score >= threshold
    boot = Any[]
    used = Set{Int}()
    id = new_id()
    next = Ref(0)
    if max_bootstrapped > 0
        with_settings(; caller=Dict("optimization" => id)) do
            @sync for _ in 1:max(1, concurrency)
                @async while length(boot) < max_bootstrapped
                    i = (next[] += 1)
                    i > length(rows) && break
                    row = rows[i]
                    try
                        p = predict_inputs(runner, row_inputs(f, row))
                        p === missing && continue
                        score = Float64(judge(NamedTuple(Symbol(k) => v for (k, v) in row), p.outputs))
                        if passes(score) && length(boot) < max_bootstrapped
                            push!(boot, LMCC.turn_to_dict(p.turn))
                            push!(used, i)
                        end
                    catch
                        # a failed run is not an example
                    end
                end
            end
        end
    end
    room = max_labeled - length(boot)
    rest = [r for (i, r) in enumerate(rows) if !(i in used)]
    fill = room > 0 ? [d for d in (labeled_demo(f, r) for r in shuffle(Xoshiro(seed), rest)[1:min(room, length(rest))]) if d !== nothing] : Any[]
    remake(f; demos=Any[boot..., fill...])
end

"The mean score of `f` on rows (a failed row counts 0)."
function mean_score(f::AIFunction, rows, metric, concurrency)
    isempty(rows) && return 0.0
    e = evaluate(f, rows; metric=metric === nothing ? nothing : Dict("score" => metric), concurrency)
    something(e.score, 0.0)
end

"""
    random_search(f, data; valset, candidates = 8, metric, max_bootstrapped = 4,
                  max_labeled = 16, teacher, seed = 0, stop_at, expected) -> (f, trials)

Try several sets of worked examples (none, labeled only, bootstrapped,
bootstrapped from shuffled rows of random sizes), score each on `valset`
(default: `data`), and return the best copy with every trial as rows.
"""
function random_search(f::AIFunction, data; valset=nothing, candidates::Integer=8, metric=nothing, max_bootstrapped::Integer=4,
                       max_labeled::Integer=16, teacher=nothing, seed::Integer=0, stop_at=nothing, expected=nothing, concurrency::Integer=4)
    rows = training_rows(f, data, expected)
    val = valset === nothing ? rows : training_rows(f, valset, expected)
    trials = NamedTuple[]
    best = (score=-Inf, fn=f)
    for c in -3:(candidates - 1)
        label, candidate = if c == -3
            "zero-shot", remake(f; demos=Any[])
        elseif c == -2
            "labeled", labeled_few_shot(f, rows; k=max_labeled, seed)
        else
            rng = Xoshiro(seed + max(c, 0))
            shuffled = c >= 0 ? shuffle(rng, rows) : rows
            size = c >= 0 ? rand(rng, 1:max(1, max_bootstrapped)) : max_bootstrapped
            (c == -1 ? "bootstrapped" : "bootstrapped (seed $c, $size examples)"),
            bootstrap_few_shot(f, shuffled; metric, max_bootstrapped=size, max_labeled, teacher, seed=seed + c, concurrency)
        end
        score = mean_score(candidate, val, metric, concurrency)
        push!(trials, (candidate=label, examples=length(candidate.demos), score=score, version=version(candidate)))
        score > best.score && (best = (score=score, fn=candidate))
        stop_at !== nothing && score >= stop_at && break
    end
    (best.fn, trials)
end

const TIPS = ["", "Be concise and direct.", "Be precise: the task is high-stakes and mistakes are costly.",
              "Describe the expected output format exactly.", "Spell out the steps to follow before answering.",
              "Name the common mistakes on this task and how to avoid them.",
              "Give the model a helpful persona suited to the task.", "Consider edge cases and unusual inputs."]

"Proposes instructions for AI functions: itself an AI function, so its calls are logged and rated like any other."
const PROPOSER = Ref{Any}(nothing)
function proposer()
    PROPOSER[] === nothing || return PROPOSER[]
    PROPOSER[] = AIFunction("propose_instruction",
        "Propose a new instruction (a system prompt) for an AI function that will make it score higher on its task. " *
        "Use the signature and the examples to understand the task. Make it different from the previous proposals, " *
        "and follow the tip. Keep the exact input and output names. Reply with the instruction text only.";
        inputs=(function_name=String, signature=String, current_instruction=String, examples=String,
                previous_proposals=Vector{String}, tip=String),
        output=String, module_name="functai.meta", adapter=:xml, include_name=false, reasoning=false)
end

"A few rows as the proposer sees them: the function's inputs and outputs only (never other columns: they may be answers a person had to read)."
function examples_text(f::AIFunction, rows; limit=5)
    ins, outs = input_names(f), output_names(f)
    join((LMCC.json_text(LMCC.jobj("inputs" => JObj(k => logvalue(v) for (k, v) in r if k in ins),
                                   "outputs" => JObj(k => logvalue(v) for (k, v) in r if k in outs)))
          for r in rows[1:min(limit, length(rows))]), "\n")
end

"""
    instruction_search(f, data; candidates = 6, trials = 12, minibatch = 20, valset,
                       prompt_lm, metric, max_bootstrapped = 4, max_labeled = 4,
                       seed = 0, finalists = 3, expected) -> (f, trials)

Search instructions a model writes, with sets of worked examples, and keep
the best: instruction candidates (the current one, plus proposals written by
`prompt_lm` from the signature and a few examples) × example sets
(bootstrapped), tried on minibatches of `valset`; the top `finalists` are
then scored on all of `valset`. MIPRO-style: random search with greedy
refinement, not Bayesian.
"""
function instruction_search(f::AIFunction, data; candidates::Integer=6, trials::Integer=12, minibatch::Integer=20, valset=nothing,
                            prompt_lm=nothing, metric=nothing, max_bootstrapped::Integer=4, max_labeled::Integer=4,
                            seed::Integer=0, finalists::Integer=3, teacher=nothing, expected=nothing, concurrency::Integer=4)
    rng = Xoshiro(seed)
    rows = training_rows(f, data, expected)
    val = valset === nothing ? rows : training_rows(f, valset, expected)
    writer = prompt_lm === nothing ? proposer() : configure(proposer(); lm=prompt_lm)
    writer = configure(writer; temperature=1.0)
    texts = Union{Nothing,String}[f.instructions]
    proposals = String[]
    sample = copy(rows)
    for i in 1:(candidates - 1)
        shuffle!(rng, sample)
        tip = TIPS[mod1(i, length(TIPS))]
        text = try
            writer(f.definition.name, sprint(show, f), instructions(f), examples_text(f, sample), copy(proposals),
                   isempty(tip) ? "(no tip)" : tip)
        catch err
            @warn "an instruction proposal failed" error = sprint(showerror, unwrap(err))
            continue
        end
        text = trim_white(text)
        isempty(text) || text in proposals || (push!(proposals, text); push!(texts, text))
    end
    demo_sets = Vector{Any}[copy(f.demos)]
    if max_bootstrapped > 0 || max_labeled > 0
        for k in 1:(candidates - 1)
            shuffled = shuffle(Xoshiro(seed + k), rows)
            push!(demo_sets, bootstrap_few_shot(f, shuffled; metric, max_bootstrapped, max_labeled, teacher, seed=seed + k, concurrency).demos)
        end
    end
    make(c) = remake(f; instructions=texts[c[1]], demos=demo_sets[c[2]])
    scored = Dict{Tuple{Int,Int},Vector{Float64}}()
    log = NamedTuple[]
    for t in 1:trials
        c = if t == 1
            (1, 1)
        elseif !isempty(scored) && t > trials ÷ 2 && rand(rng) < 0.6
            best = argmax(k -> sum(scored[k]) / length(scored[k]), collect(keys(scored)))
            rand(rng, Bool) ? (rand(rng, 1:length(texts)), best[2]) : (best[1], rand(rng, 1:length(demo_sets)))
        else
            (rand(rng, 1:length(texts)), rand(rng, 1:length(demo_sets)))
        end
        batch = length(val) <= minibatch ? val : shuffle(rng, val)[1:minibatch]
        s = mean_score(make(c), batch, metric, concurrency)
        push!(get!(scored, c, Float64[]), s)
        push!(log, (trial=t, instruction=c[1], examples=c[2], minibatch_score=s))
    end
    ranked = sort(collect(keys(scored)); by=k -> -sum(scored[k]) / length(scored[k]))
    top = ranked[1:min(finalists, length(ranked))]
    winner = if length(val) <= minibatch
        first(top)
    else
        full = Dict(c => mean_score(make(c), val, metric, concurrency) for c in top)
        argmax(c -> full[c], top)
    end
    (make(winner), [(; t..., instruction_text=texts[t.instruction]) for t in log])
end
