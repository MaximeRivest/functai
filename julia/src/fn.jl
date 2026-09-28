# AI functions: a typed signature whose body a model writes.
#
#     @enum Mood happy unhappy mixed
#     @ai function mood(review::String)::Mood
#         "How does the customer feel about what they bought?"
#     end
#     mood("Broke after a day.")          # unhappy::Mood
#     df.mood = mood.(df.review)          # the column, 8 calls at a time

"""
    AIFunction

A function whose body a language model writes. Call it like any function;
broadcast it over a column (`f.(xs)`, `map(f, xs)`, `ByRow(f)`), and the
calls run `concurrency` at a time. Made by [`@ai`](@ref), by the
`AIFunction(name; inputs, output, …)` constructor, or by `FunctAI.load`.

Its parts: [`instructions`](@ref), [`demos`](@ref), [`version`](@ref),
[`signature_id`](@ref), [`predict`](@ref), [`render`](@ref),
`stream`, [`configure`](@ref) (a copy with other settings).
"""
struct AIFunction <: Function
    definition::Definition
    own::Dict{Symbol,Any}
    tools::Vector{AITool}
    module_name::String
    file::Union{Nothing,String}
    line::Union{Nothing,Int}
    saved::Union{Nothing,String}
    instructions::Union{Nothing,String}
    demos::Vector{Any}
    binder::Any                 # (args...; kw...) -> Dict of inputs; nothing: positional in order
    from_row::Any               # Dict of columns -> Dict of inputs (defaults for absent ones); nothing: by name
    body::Any                   # nothing, or (inputs::Dict, outputs::NamedTuple) -> the value
    code::Union{Nothing,String} # the hash of the body's code, when it has code of its own
    returns::Any                # the declared return type, or nothing
    source::Any                 # the definition's expression (code of its own is saved from it)
    declared::Union{Nothing,JObj}   # the interface a saved node declared (its words), else derived from the definition
    cache::Dict{Any,Any}
    lock::ReentrantLock
end

const PARTS = (:definition, :own, :tools, :module_name, :file, :line, :saved, :instructions, :demos,
               :binder, :from_row, :body, :code, :returns, :source, :declared)

"A copy of `f` with some parts replaced (and a fresh cache)."
function remake(f::AIFunction; kw...)
    parts = map(p -> haskey(kw, p) ? kw[p] : getfield(f, p), PARTS)
    AIFunction(parts..., Dict{Any,Any}(), ReentrantLock())
end

Base.nameof(f::AIFunction) = Symbol(f.definition.name)
answer_name(f::AIFunction) = answer_name(f.definition)
input_names(f::AIFunction) = [x.name for x in f.definition.inputs]
output_names(f::AIFunction) = [x.name for x in f.definition.outputs]

cached(g, f::AIFunction, key) = lock(() -> get!(g, f.cache, key), f.lock)

# ------------------------------------------------------------------ constructing

"Named entries: a NamedTuple, a Dict, or a vector of `name => spec` pairs."
entries(x::Union{NamedTuple,AbstractDict}) = pairs(x)
entries(x::Union{AbstractVector,Tuple}) = (all(e -> e isa Pair, x) ? x : throw(ArgumentError("expected name => type pairs, not $(repr(x))")))

"""
A field written as `T`, `T => \"words\"`, `OneOf(...)` or a JSON Schema Dict;
`default`, when given (`Some(value)`), makes it an input a caller may leave out.
"""
function field_def(name, spec, where; default=nothing)
    desc = nothing
    if spec isa Pair && last(spec) isa AbstractString
        spec, desc = first(spec), last(spec)
    end
    spec isa Union{Type,OneOf,AbstractDict} || throw(ArgumentError("$where: a type, OneOf(...), a JSON Schema Dict, or T => \"words\"; not $(repr(spec))"))
    shape = shape_of(spec; where)
    desc = desc === nothing || isempty(desc) ? nothing : String(desc)
    default === nothing && return FieldDef(String(name), spec, shape, desc)
    native = something(default)
    data = json_form(native)
    data === NOJSON && throw(InterfaceError("interface-malformed", String(name),
        "$where: its default is sent to the model when it is left out, so it needs a JSON form; a $(typeof(native)) has none"))
    haskey(shape, "default") && !same_json(shape["default"], data) &&
        throw(ArgumentError("$where: its shape's default $(LMCC.json_text(shape["default"])) is not its default $(LMCC.json_text(data))"))
    FieldDef(String(name), spec, data_shape(shape), desc, true, data, native)
end
shape_of(spec::Union{OneOf,AbstractDict}; where="") = shape_of(spec)

"""
    AIFunction(name, description = ""; inputs, output = String, outputs, settings...)

An AI function without the macro, from data:

```julia
mood = AIFunction("mood", "How does the customer feel about what they bought?";
                  inputs = (review = String => "the customer's own words",),
                  output = OneOf(:happy, :unhappy, :mixed), temperature = 0)
```

`inputs` and `outputs` are `NamedTuple`s (or pairs) of types, `T => "words"`,
`OneOf(...)` or JSON Schema Dicts; `outputs` names several (the last is
the answer). `defaults` gives the inputs a caller may leave out their
values (`defaults = (tone = "kind",)`): each is written in the function's
[`interface`](@ref) and sent when the input is left out. Other keywords:
`tools`, `demos`, `instructions`, `module_name`, and any setting (see
[`configure!`](@ref)).

The definition is checked here: lmcc's signature first, then the
interface (contract/programs.md; `InterfaceError` `interface-malformed`,
say for a default that does not fit its type), then an own `log_content`
map (`LogContentError` for a name that is not a field).
"""
function AIFunction(name::AbstractString, description::AbstractString=""; inputs=(;), output=nothing, outputs=nothing,
                    defaults=(;), tools=(), demos=(), instructions=nothing, module_name::AbstractString="__main__",
                    file=nothing, line=nothing, binder=nothing, from_row=nothing, body=nothing, code=nothing,
                    returns=nothing, source=nothing, declared=nothing, settings...)
    is_name(name) || throw(ArgumentError("an AI function's name is an identifier, not $(repr(name))"))
    output !== nothing && outputs !== nothing && throw(ArgumentError("$name: give output (one answer) or outputs (several), not both"))
    given_defaults = Dict{String,Any}(String(k) => v for (k, v) in entries(defaults))
    ins = FieldDef[field_def(k, v, "$name.$k"; default=haskey(given_defaults, String(k)) ? Some(given_defaults[String(k)]) : nothing)
                   for (k, v) in entries(inputs)]
    unknown = setdiff(keys(given_defaults), [x.name for x in ins])
    isempty(unknown) || throw(ArgumentError("$name: defaults names $(join(sort!(collect(unknown)), ", ")), which is not an input"))
    outs = outputs === nothing ? FieldDef[field_def("result", something(output, String), "$name.result")] :
           FieldDef[field_def(k, v, "$name.$k") for (k, v) in entries(outputs)]
    isempty(outs) && throw(ArgumentError("$name: outputs is empty"))
    names = [x.name for x in vcat(ins, outs)]
    allunique(names) || throw(ArgumentError("$name: a name is both an input and an output, or used twice: $(join(names, ", "))"))
    for n in ("tools", "calls")
        n in names && !isempty(tools) && throw(ArgumentError("$name: with tools, $n is a name FunctAI uses; rename that field"))
    end
    definition = Definition(String(name), String(description), ins, outs, nothing)
    f = AIFunction(definition, settings_dict(settings), AITool[tool(t) for t in tools], String(module_name),
                   file === nothing ? nothing : String(file), line, nothing, nothing, Any[], binder, from_row, body, code,
                   returns, source, declared, Dict{Any,Any}(), ReentrantLock())
    f = instructions === nothing ? f : remake(f; instructions=String(instructions))
    check_definition(f)
    isempty(demos) ? f : with_demos(f, demos)
end

"""
Check a definition as the contract orders it (functions.md, "A definition"):
lmcc's signature (`signature-malformed`), then the interface every program
keeps (programs.md: `InterfaceError` `interface-malformed`), then an own
`log_content` map (calls.md, "Content": `log-content-field`).
"""
function check_definition(f::AIFunction)
    signature_of(f.definition, f.instructions, true, false, !isempty(f.tools))
    check_interface(interface(f); ai=true, what=f.definition.name)
    check_own_content(f.own, program_fields(f; reasoning=get(f.own, :reasoning, false) === true), f.definition.name)
    f
end

"""
The fields of a program's calls (calls.md, "Words"): its interface's inputs
and outputs, and for an AI function the outputs FunctAI adds (`reasoning`
with reasoning on, unless a field has that name; `calls` with tools).
`added` are among `outputs`, in the signature's order.
"""
function program_fields(f::AIFunction; reasoning::Bool)
    ins, outs = input_names(f), output_names(f)
    added = String[]
    reasoning && !("reasoning" in ins || "reasoning" in outs) && push!(added, "reasoning")
    isempty(f.tools) || push!(added, "calls")
    (inputs=ins, outputs=vcat(added, outs), added=added)
end
program_fields(f::AIFunction, s::AbstractDict{Symbol}) = program_fields(f; reasoning=s[:reasoning] === true)

"""
    interface(f)

A program's interface as data (contract/programs.md): its description, its
inputs (an input a caller may leave out is `optional`, its default in its
shape) and its outputs, without the fields FunctAI adds to an AI function.
The same JSON in every language.

```julia
FunctAI.interface(mood)["inputs"]       # [{"name": "review", "shape": {"type": "string"}, "type": "String"}]
```
"""
interface(f::AIFunction) = f.declared === nothing ? interface_of(f.definition) : f.declared

"""
    interface_signature(f)

The call log's `program.interface`: the signature of `f`'s interface (what
its recorded data looks like). Equal to [`signature_id`](@ref) for an AI
function with neither reasoning nor tools.
"""
interface_signature(f::AIFunction) = cached(() -> interface_signature(interface(f)), f, :interface_signature)

# ------------------------------------------------------------------ state

"""
    instructions(f)

The instruction the model gets: the improved one when there is one, else
the one written from the name, the description and the guidance.
"""
instructions(f::AIFunction) = signature_now(f, effective(f.own)).instructions

"""
    with_instructions(f, text)

A copy of `f` whose instruction is `text` (`nothing`: the written one again).
"""
with_instructions(f::AIFunction, text) = remake(f; instructions=text === nothing ? nothing : String(text))

"""
    demos(f)

The worked examples, as `Dict("inputs" => …, "outputs" => …)` (or recorded turns).
"""
demos(f::AIFunction) = LMCC.deepcopy_json.(f.demos)

"""
    with_demos(f, examples)

A copy of `f` with these worked examples. Each is `input => answer`
(`"Broke in a day." => unhappy`), a row (a `NamedTuple` or `Dict` with the
inputs and outputs under their names), `Dict("inputs" => …, "outputs" => …)`,
or a recorded turn. Rows of a table work too: `with_demos(f, Tables.rowtable(df))`.
"""
with_demos(f::AIFunction, examples) = remake(f; demos=Any[demo_of(f, x) for x in rows_of(examples)])

function demo_of(f::AIFunction, item)
    names = input_names(f)
    if item isa Pair
        i, o = first(item), last(item)
        ins = i isa Union{NamedTuple,AbstractDict} ? JObj(String(k) => jsonvalue(v) for (k, v) in pairs(i)) : JObj(names[1] => jsonvalue(i))
        outs = o isa Union{NamedTuple,AbstractDict} && !(length(f.definition.outputs) == 1 && valuetype(f.definition.outputs[1].spec) <: Union{NamedTuple,AbstractDict}) ?
               JObj(String(k) => jsonvalue(v) for (k, v) in pairs(o)) : JObj(answer_name(f) => jsonvalue(o))
        return LMCC.jobj("inputs" => ins, "outputs" => outs)
    end
    d = item isa NamedTuple ? JObj(String(k) => v for (k, v) in pairs(item)) : item isa AbstractDict ? JObj(String(k) => v for (k, v) in item) :
        throw(ArgumentError("a worked example is input => answer, a row, or Dict(\"inputs\" => …, \"outputs\" => …); not $(repr(item))"))
    haskey(d, "signature") && haskey(d, "steps") && return LMCC.deepcopy_json(jsonvalue(d))          # a recorded turn
    if haskey(d, "inputs") && haskey(d, "outputs") && d["inputs"] isa AbstractDict && !("inputs" in names)
        return LMCC.jobj("inputs" => jsonvalue(d["inputs"]), "outputs" => jsonvalue(d["outputs"]))
    end
    ins, outs = JObj(), JObj()
    for (k, v) in d
        v === missing && continue
        (k in names ? ins : outs)[k] = jsonvalue(v)
    end
    LMCC.jobj("inputs" => ins, "outputs" => outs)
end

"Rows of anything tabular: a Tables.jl table, or a vector of rows (NamedTuples, Dicts, pairs)."
function rows_of(data)
    (data isa AbstractVector || data isa Tuple) && return collect(Any, data)
    Tables.istable(data) || throw(ArgumentError("expected a table (a DataFrame, any Tables.jl source) or a vector of rows, not a $(typeof(data))"))
    collect(Any, Tables.rowtable(data))
end

"A row as a Dict of its columns by name."
row_dict(row::AbstractDict) = OrderedDict{String,Any}(String(k) => v for (k, v) in row)
row_dict(row::NamedTuple) = OrderedDict{String,Any}(String(k) => v for (k, v) in pairs(row))
row_dict(row) = OrderedDict{String,Any}(String(n) => Tables.getcolumn(row, n) for n in Tables.columnnames(row))

# ------------------------------------------------------------------ what it sends

layout_key(s) = (let a = setting(s, :adapter)
    a === nothing || a isa Union{Symbol,AbstractString} ? (a === nothing ? nothing : lowercase(String(a))) : (:object, objectid(a))
end, setting(s, :template) === nothing ? nothing : hash(repr(setting(s, :template))))

function signature_now(f::AIFunction, s)
    cached(f, (:signature, s[:reasoning], s[:include_name])) do
        signature_of(f.definition, f.instructions, s[:include_name], s[:reasoning], !isempty(f.tools))
    end
end

"Worked examples as example turns for this plan (contract/functions.md, \"Worked examples\")."
function past_turns(f::AIFunction, plan)
    fields = plan.signature.fields
    plain = Set(x.name for x in fields if x.direction == "input" && x.purpose == "plain")
    kept = Set(x.name for x in fields if x.direction == "output" && x.purpose in ("plain", "reasoning"))
    out = Any[]
    for d in f.demos
        if haskey(d, "signature") && haskey(d, "steps")
            if d["signature"] == LMCC.fingerprint(plan)
                try
                    push!(out, LMCC.load_turn(plan, d))
                    continue
                catch err
                    err isa LMCC.Refusal || rethrow()
                end
            end
        end
        ins = prepare_inputs(plan.signature, JObj(k => v for (k, v) in something(get(d, "inputs", nothing), JObj()) if k in plain))
        outs = JObj(k => v for (k, v) in something(get(d, "outputs", nothing), JObj()) if k in kept)
        isempty(outs) && continue
        try
            push!(out, LMCC.example(plan, ins, outs))
        catch err
            err isa LMCC.Refusal || rethrow()
        end
    end
    out
end

tool_values(f::AIFunction) = Any[tool_data(t) for t in f.tools]

"""
    probe_request(f, inputs = sample)

The request `f` renders for `inputs` under the fixed facts of the `probe`
provider (contract/calls.md, "Versions"), as lm15 canonical JSON.
"""
function probe_request(f::AIFunction, inputs=nothing)
    s = effective(f.own)
    plan = bind_layout(setting(s, :adapter), setting(s, :template), signature_now(f, s), PROBE, "probe")
    values = prepare_inputs(plan.signature, inputs === nothing ? sample_inputs(plan.signature) : JObj(String(k) => v for (k, v) in pairs(inputs)))
    isempty(f.tools) || (values["tools"] = tool_values(f))
    LMCC.request(LMCC.render(plan, LMCC.new_turn(plan, values); turns=past_turns(f, plan)), "probe")
end

"The hash of a probe's request, or `refused:<code>` when it cannot be rendered."
function request_hash(f::AIFunction, inputs=nothing)
    try
        LMCC.sha256_of(probe_request(f, inputs))
    catch err
        err isa LMCC.Refusal || rethrow()
        "refused:$(err.code)"
    end
end

"""
    version(f)

A fingerprint of everything that decides what `f` sends besides its inputs
(contract/calls.md, "Versions"): the same function has the same version in
every FunctAI language, and an improved one a new version.
"""
function version(f::AIFunction)
    s = effective(f.own)
    cached(f, (:version, layout_key(s), s[:reasoning], s[:include_name])) do
        r = request_hash(f)
        LMCC.sha256_of(f.code === nothing ? LMCC.jobj("request" => r) : LMCC.jobj("code" => f.code, "request" => r))
    end
end

"""
    signature_id(f)

The call log's `program.signature`: the fields' names and shapes. Calls with
the same signature id share data even when the instruction changed.
"""
signature_id(f::AIFunction) = signature_id(signature_now(f, effective(f.own)))

"""
    signature(f)

The lmcc signature the model gets: the instruction and the fields.
"""
signature(f::AIFunction) = signature_now(f, effective(f.own))

function resolve_route(s)
    lm = something(setting(s, :lm), default_model(), Some(nothing))
    lm === nothing && throw(ArgumentError("no model configured, and no API key found to pick one: set OPENAI_API_KEY (or " *
        "ANTHROPIC_API_KEY, GEMINI_API_KEY, GROQ_API_KEY, …), sign in with FunctAI.login(), or name a model: " *
        "FunctAI.configure!(lm = \"gpt-4.1-mini\")"))
    model = model_string(lm)
    router = something(setting(s, :router), Some(nothing))
    router === nothing && (router = default_router(model))
    provider, wire = route(router, model)
    (; model, router, provider, wire)
end

function plan_for(f::AIFunction, given)
    r = resolve_route(given)
    s = adjust_settings(given, r.provider, r.wire)
    caps = call_capabilities(r.provider, r.wire, s)
    key = (:plan, layout_key(s), sort!(collect(caps)), r.provider, s[:reasoning], s[:include_name])
    plan = cached(f, key) do
        bind_layout(setting(s, :adapter), setting(s, :template), signature_now(f, s), caps, r.provider)
    end
    (; plan, r..., settings=s)
end

"The call log's `program` of this function."
function program_of(f::AIFunction)
    p = LMCC.jobj("name" => f.definition.name, "kind" => "ai", "module" => f.module_name, "version" => version(f),
                  "signature" => signature_id(f), "interface" => interface_signature(f), "answer" => answer_name(f))
    f.saved === nothing || (p["saved"] = f.saved)
    f.file === nothing || (p["file"] = f.file)
    f.line === nothing || (p["line"] = f.line)
    p
end

# ------------------------------------------------------------------ calling

"""
The inputs of a call: positional and keyword arguments by the function's
binder, else by position; an optional input left out takes its default
(programs.md: a model is sent every input), in the definition's order.
"""
function bind_inputs(f::AIFunction, args, kw)
    given = if f.binder !== nothing
        try
            f.binder(args...; kw...)
        catch err
            err isa MethodError && err.f === f.binder || rethrow()
            throw(ArgumentError("$(f.definition.name) takes ($(join(input_names(f), ", "))); it was given $(length(args)) " *
                                "argument(s)$(isempty(kw) ? "" : " and $(join(keys(kw), ", "))")"))
        end
    else
        names = input_names(f)
        length(args) <= length(names) || throw(ArgumentError("$(f.definition.name) takes $(length(names)) input(s) ($(join(names, ", "))), not $(length(args))"))
        out = OrderedDict{String,Any}(names[i] => a for (i, a) in enumerate(args))
        for (k, v) in kw
            String(k) in names || throw(ArgumentError("$(f.definition.name) has no input $(k); its inputs: $(join(names, ", "))"))
            out[String(k)] = v
        end
        out
    end
    with_defaults(f, given)
end

"The given inputs, with each optional one left out taking its default; a required one left out is an error."
function with_defaults(f::AIFunction, given::AbstractDict)
    out = OrderedDict{String,Any}()
    for x in f.definition.inputs
        if haskey(given, x.name)
            out[x.name] = given[x.name]
        elseif x.optional
            out[x.name] = x.native
        end
    end
    missing_names = [x.name for x in f.definition.inputs if !haskey(out, x.name)]
    isempty(missing_names) || throw(ArgumentError("$(f.definition.name): no value for $(join(missing_names, ", "))"))
    out
end

"The inputs of a table row: its columns named like the inputs (defaults for absent ones)."
function row_inputs(f::AIFunction, row::AbstractDict)
    f.from_row === nothing || return with_defaults(f, f.from_row(row))
    names = input_names(f)
    absent = [x.name for x in f.definition.inputs if !haskey(row, x.name) && !x.optional]
    isempty(absent) || throw(ArgumentError("the row has no column $(join(absent, ", ")) for $(f.definition.name)'s input(s)"))
    with_defaults(f, OrderedDict{String,Any}(n => row[n] for n in names if haskey(row, n)))
end

has_missing(inputs) = any(v -> v === missing, values(inputs))

"The typed outputs of a reading's values: each declared output read as its type (a misfit is `parse-value`)."
function typed_outputs(f::AIFunction, values::AbstractDict)
    names = Symbol[]
    vals = Any[]
    haskey(values, "reasoning") && !any(x -> x.name == "reasoning", f.definition.outputs) &&
        (push!(names, :reasoning); push!(vals, values["reasoning"]))
    for x in f.definition.outputs
        haskey(values, x.name) || throw(LMCC.Refusal("parse-missing-fields", "the reply has no $(x.name)"))
        p = misfit(x.shape, values[x.name], x.name)
        p === nothing || throw(LMCC.Refusal("parse-value", p))
        push!(names, Symbol(x.name))
        push!(vals, x.spec isa AbstractDict ? convert_loaded(julia_type(x.shape), values[x.name]) : fromjson(x.spec, values[x.name], x.name))
    end
    NamedTuple{Tuple(names)}(Tuple(vals))
end

"What calling returns: the code's value; else the answer, or every output when there are several."
function value_of(f::AIFunction, inputs, outputs::NamedTuple)
    if f.body !== nothing
        v = f.body(inputs, outputs)
        return f.returns === nothing ? v : convert(f.returns, v)
    end
    declared = Tuple(Symbol(x.name) for x in f.definition.outputs)
    length(declared) == 1 ? outputs[only(declared)] : NamedTuple{declared}(Tuple(outputs[n] for n in declared))
end

"""
    predict(f, args...; kw...) -> Prediction

Call `f` and keep everything: the value, every output (typed), the call's
id (to rate it), the turn and the replies. `missing` in, `missing` out.
"""
StatsAPI.predict(f::AIFunction, args...; kw...) = predict_inputs(f, bind_inputs(f, args, kw))

function predict_inputs(f::AIFunction, inputs::AbstractDict)
    has_missing(inputs) && return missing
    s = effective(f.own)
    call = start_call(program_of(f), f.definition.name, s, f.own, inputs, program_fields(f, s))
    prediction = Ref{Any}(nothing)
    run_call(call) do call
        r = plan_for(f, s)
        call.provider = r.provider
        job = Job(f.definition.name, r.plan, past_turns(f, r.plan), inputs, r.settings, r.router, r.model,
                  f.tools, call, values -> typed_outputs(f, values))
        outputs, turn, responses, reading = run_job(job)
        call.outputs = outputs
        probs = reading.probabilities
        if !isempty(probs)
            chosen = [get(p, string(jsonvalue(outputs[Symbol(k)])), nothing) for (k, p) in probs if haskey(outputs, Symbol(k))]
            chosen = filter(!isnothing, chosen)
            isempty(chosen) || (call.confidence = minimum(chosen))
        end
        value = value_of(f, inputs, outputs)
        call.returned = f.body === nothing ? outputs[Symbol(answer_name(f))] : value
        call.has_returned = true
        prediction[] = Prediction(value, outputs, answer_name(f), call.id, turn, responses, reading.repairs, probs)
        value
    end
    prediction[]
end

(f::AIFunction)(args...; kw...) = (inputs = bind_inputs(f, args, kw); has_missing(inputs) ? missing : predict_inputs(f, inputs).value)

"""
    render(f, args...; kw...) -> LM15.Request

The exact request the next call would send, without sending it (nothing is
paid): the instruction, the worked examples, the layout, the model.
"""
function render(f::AIFunction, args...; kw...)
    s = effective(f.own)
    r = plan_for(f, s)
    plan = r.plan
    values = prepare_inputs(plan.signature, bind_inputs(f, args, kw))
    isempty(f.tools) || (values["tools"] = tool_values(f))
    LMCC.lm15_request(LMCC.render(plan, LMCC.new_turn(plan, values); turns=past_turns(f, plan)); model=r.model, config=config_of(r.settings))
end

"""
    configure(f; settings...)

A copy of `f` with its own settings changed (a setting given as `nothing`
goes back to what `configure!` and the defaults say):
`configure(mood; lm = "claude-haiku-4-5", temperature = 0)`.
"""
function configure(f::AIFunction; kw...)
    own = copy(f.own)
    for (k, v) in settings_dict(kw)
        v === nothing ? delete!(own, k) : (own[k] = v)
    end
    g = remake(f; own)
    check_own_content(g.own, program_fields(g; reasoning=get(g.own, :reasoning, false) === true), g.definition.name)
    g
end

"""
    settings(f)

The settings `f` sets itself (what `configure!` and `with_settings` add is not shown).
"""
settings(f::AIFunction) = (; sort!(collect(f.own); by=first)...)

# ------------------------------------------------------------------ over a column

"The last column run's failures: `(row, call, error)` for each row whose call failed."
const PROBLEMS = Ref{Vector{Any}}(Any[])

"""
    problems()

The rows whose calls failed in the last run over a column (`f.(xs)`,
`map(f, xs)`): their index and error. Their answers are `missing`.
"""
problems() = copy(PROBLEMS[])

"Run `g` on each item, `n` at a time (tasks on this thread: calls wait on the network, not the CPU)."
function run_concurrently(g, items::AbstractVector, n::Integer)
    results = Vector{Any}(undef, length(items))
    errors = Vector{Any}(nothing, length(items))
    next = Ref(0)
    @sync for _ in 1:min(n, length(items))
        @async while true
            i = (next[] += 1)
            i > length(items) && break
            try
                results[i] = g(items[i])
            catch err
                errors[i] = err
            end
        end
    end
    (results, errors)
end

"""
Call `f` on each tuple of arguments, `concurrency` at a time. A row with a
`missing` input is `missing` without a call. A failed call is `missing`
with one warning (see `problems()`); when every call fails, the first
error is thrown.
"""
function call_each(f::Function, argtuples::AbstractArray; each=f)
    n = effective(f isa AIFunction ? f.own : Dict{Symbol,Any}())[:concurrency]
    items = vec(collect(argtuples))
    results, errors = run_concurrently(t -> each(t...), items, n)
    failed = findall(!isnothing, errors)
    for i in failed
        results[i] = missing
    end
    if !isempty(failed)
        PROBLEMS[] = Any[(row=i, error=unwrap(errors[i])) for i in failed]
        first_error = unwrap(errors[first(failed)])
        length(failed) == length(items) && throw(first_error)
        @warn "$(length(failed)) of $(length(items)) calls of $(nameof(f)) failed; their answers are missing. " *
              "FunctAI.problems() lists them" first_error = sprint(showerror, first_error)
    else
        PROBLEMS[] = Any[]
    end
    reshape(map(identity, results), size(argtuples))   # map(identity) narrows the element type
end

broadcast_args(args) = Base.Broadcast.materialize(Base.Broadcast.broadcasted(tuple, args...))

function Base.Broadcast.broadcasted(f::AIFunction, args...)
    argtuples = broadcast_args(args)
    argtuples isa Tuple ? f(argtuples...) : call_each(f, argtuples)
end
# predict.(f, column): every row's Prediction, concurrently too
function Base.Broadcast.broadcasted(::typeof(StatsAPI.predict), f::AIFunction, args...)
    argtuples = broadcast_args(args)
    argtuples isa Tuple ? predict(f, argtuples...) : call_each(f, argtuples; each=(a...) -> predict(f, a...))
end
Base.map(f::AIFunction, xs::AbstractArray, more::AbstractArray...) =
    isempty(more) ? call_each(f, map(tuple, xs)) : call_each(f, map(tuple, xs, more...))
Base.map(f::AIFunction, xs, more...) = call_each(f, collect(zip(xs, more...)))

# ------------------------------------------------------------------ showing

spec_text(spec) = spec isa Type ? string(spec) : spec isa OneOf ? string(spec) : "(shape)"

function Base.show(io::IO, f::AIFunction)
    d = f.definition
    ins = join(("$(x.name)::$(spec_text(x.spec))" for x in d.inputs), ", ")
    outs = length(d.outputs) == 1 ? spec_text(only(d.outputs).spec) :
           "(" * join(("$(x.name)::$(spec_text(x.spec))" for x in d.outputs), ", ") * ")"
    print(io, d.name, "(", ins, ") -> ", outs)
end

function Base.show(io::IO, ::MIME"text/plain", f::AIFunction)
    print(io, "AI function ")
    show(io, f)
    s = effective(f.own)
    model = something(setting(s, :lm), default_model(), "no model yet")
    println(io, "\n  model:        ", model)
    text = try
        instructions(f)
    catch err
        "(cannot be written: $(sprint(showerror, err)))"
    end
    lines = [l for l in split(text, '\n') if !startswith(l, "Function: ")]
    while !isempty(lines) && isempty(strip(first(lines)))
        popfirst!(lines)
    end
    shown = lines[1:min(4, length(lines))]
    println(io, "  instruction:  ", isempty(shown) ? "" : first(shown))
    for l in shown[2:end]
        println(io, isempty(strip(l)) ? "" : "                " * l)
    end
    length(lines) > 4 && println(io, "                …")
    isempty(f.demos) || println(io, "  examples:     ", length(f.demos))
    isempty(f.tools) || println(io, "  tools:        ", join((t.name for t in f.tools), ", "))
    f.body === nothing || println(io, "  code:         its own, after the model answers")
    v = try
        version(f)
    catch err
        "(none: $(sprint(showerror, err)))"
    end
    println(io, "  version:      ", first(v, 19), "…")
    print(io, "  see:          FunctAI.prompt($(f.definition.name), …) for the exact request")
end

"""
    FunctAI.prompt(f, args...; kw...)

The request the next call would send, shown as a conversation (the system
message, the worked examples, the question, the settings); nothing is sent.
`prompt(f, …).request` is the lm15 request itself ([`render`](@ref)).
"""
prompt(f::AIFunction, args...; kw...) = Prompt(render(f, args...; kw...))

struct Prompt
    request::LM15.Request
end
function Base.show(io::IO, ::MIME"text/plain", p::Prompt)
    r = p.request
    printstyled(io, "model: ", r.model, "\n"; color=:light_black)
    part_text(x) = x isa LM15.TextPart ? x.text : x isa LM15.ToolCallPart ? "(calls $(x.name) with $(LMCC.json_text(x.input)))" :
                   x isa LM15.ToolResultPart ? "(result of $(something(x.name, x.id)))" : "($(x.type))"
    if r.system !== nothing
        printstyled(io, "system\n"; bold=true, color=:cyan)
        println(io, r.system isa AbstractString ? r.system : join(part_text.(r.system)))
    end
    for m in r.messages
        printstyled(io, m.role, "\n"; bold=true, color=m.role == "user" ? :green : :yellow)
        println(io, join(part_text.(m.parts)))
    end
    isempty(r.tools) || printstyled(io, "tools: ", join((t.name for t in r.tools), ", "), "\n"; color=:light_black)
    config = LM15.to_dict(r.config)
    isempty(config) || printstyled(io, "config: ", LMCC.json_text(config); color=:light_black)
end
Base.show(io::IO, p::Prompt) = print(io, "Prompt(", p.request.model, ", ", length(p.request.messages), " messages)")
