# The @ai macro: a Julia function whose body a model writes.
#
#     @ai function triage(ticket::String; team::String = "support")
#         """
#         Read the ticket.
#
#         # Arguments
#         - `ticket`: the customer's own words
#         """
#         summary::String = ai"one sentence, no names"
#         minutes::Int    = ai"minutes to fix"
#     end
#
# The body is the description (a string), then the outputs the model writes
# (`name::T = ai"words"`; none: one output, `result`, of the return type),
# then, optionally, code of your own that runs on them. Nothing is evaluated
# here but the types, the options and the default values.

"""
    ai"words"

In the body of an [`@ai`](@ref) function, an output the model writes, with
words about it: `summary::String = ai"one sentence, no names"`.
It is read by `@ai` from your code before anything runs, so it needs no
import, and FunctAI doesn't export it (PromptingTools.jl exports its own
`ai"…"`): outside `@ai` it is `FunctAI.@ai_str`.
"""
macro ai_str(s)
    AIOutput(s)
end

"An output's words, from `ai\"…\"` outside `@ai` (inside it, `@ai` reads them itself)."
struct AIOutput
    desc::String
end

const OPTION_NAMES = (:tools, :demos, :instructions, :module_name)

is_ai_marker(x) = x === :ai || (x isa Expr && x.head === :macrocall &&
    (x.args[1] === Symbol("@ai_str") || (x.args[1] isa Expr && x.args[1].head === :. && x.args[1].args[end] == QuoteNode(Symbol("@ai_str")))))
marker_desc(x) = x === :ai ? nothing : (s = x.args[end]; s isa AbstractString ? String(s) :
    throw(ArgumentError("ai\"…\" takes plain words, without interpolation")))

"An input: its name, its type expression, its default (or nothing), whether it is a keyword."
struct InputSyntax
    name::Symbol
    type::Any
    default::Any
    has_default::Bool
    keyword::Bool
end

function parse_input(x, keyword::Bool)
    default, has_default = nothing, false
    if x isa Expr && x.head === :kw
        x, default, has_default = x.args[1], x.args[2], true
    end
    x isa Symbol && return InputSyntax(x, :Any, default, has_default, keyword)
    if x isa Expr && x.head === :(::) && length(x.args) == 2 && x.args[1] isa Symbol
        return InputSyntax(x.args[1], x.args[2], default, has_default, keyword)
    end
    x isa Expr && x.head === :... && throw(ArgumentError("@ai: an AI function's inputs are named; no varargs ($(x))"))
    throw(ArgumentError("@ai: cannot read the input $(x); write name::Type, name::Type = default, or name"))
end

"`(name, inputs, return type expression or nothing)` of the function's head."
function parse_head(head)
    ret = nothing
    if head isa Expr && head.head === :(::)
        head, ret = head.args[1], head.args[2]
    end
    head isa Expr && head.head === :where && throw(ArgumentError("@ai: an AI function has concrete types; no `where`"))
    (head isa Expr && head.head === :call && head.args[1] isa Symbol) ||
        throw(ArgumentError("@ai: expected `function name(inputs...)::Type … end`"))
    name = head.args[1]
    # positional inputs, then keywords: the order they are written in (Julia's
    # parser stores the keywords first), which is the order the model reads them
    inputs = InputSyntax[]
    keywords = InputSyntax[]
    for a in head.args[2:end]
        if a isa Expr && a.head === :parameters
            append!(keywords, parse_input(k, true) for k in a.args)
        else
            push!(inputs, parse_input(a, false))
        end
    end
    append!(inputs, keywords)
    (name, inputs, ret)
end

statements(body) = body isa Expr && body.head === :block ? [x for x in body.args if !(x isa LineNumberNode)] : Any[body]

"The description and `# Arguments` guidance of a docstring: `(description, Dict(name => words))`."
function split_arguments(doc::AbstractString)
    lines = split(doc, '\n')
    start = findfirst(l -> occursin(r"^\s*#+\s*Arguments\s*$"i, l), lines)
    start === nothing && return (String(doc), Dict{String,String}())
    stop = something(findnext(l -> occursin(r"^\s*#+\s+\S", l), lines, start + 1), length(lines) + 1)
    guidance = OrderedDict{String,String}()
    current = nothing
    for l in lines[start+1:stop-1]
        m = match(r"^\s*[-*+]\s+`?([A-Za-z_][A-Za-z0-9_]*)(?:::[^`:]*)?`?\s*:\s*(.*)$", l)
        if m !== nothing
            current = String(m.captures[1])
            guidance[current] = String(strip(m.captures[2]))
        elseif current !== nothing && !isempty(strip(l))
            guidance[current] = strip(guidance[current] * " " * strip(l))
        end
    end
    description = join(vcat(lines[1:start-1], lines[stop:end]), '\n')
    (String(strip(description)), guidance)
end

"The parsed code as JSON data: no layout, no comments, stable across Julia versions' printing."
code_data(x::Expr) = LMCC.jobj("h" => String(x.head), "a" => Any[code_data(a) for a in x.args if !(a isa LineNumberNode)])
code_data(x::Symbol) = LMCC.jobj("s" => String(x))
code_data(x::QuoteNode) = LMCC.jobj("q" => code_data(x.value))
code_data(x::Union{AbstractString,Bool,Nothing}) = x isa AbstractString ? String(x) : x
code_data(x::Integer) = typemin(Int64) <= x <= typemax(Int64) ? Int64(x) : LMCC.jobj("v" => string(x))
code_data(x::AbstractFloat) = isfinite(x) ? Float64(x) : LMCC.jobj("v" => string(x))
code_data(x) = LMCC.jobj("v" => string(typeof(x)), "r" => repr(x))

"""
    @ai [settings...] function name(inputs...)::Type
        "What it does."
        [output::Type = ai"words" ...]
        [code of your own ...]
    end

An AI function: typed inputs, typed outputs, and a body a language model
writes. The first string of the body is what the function does (a
docstring's `# Arguments` list describes the inputs); `name::T = ai"words"`
declares outputs (the last is the answer); without them the one output,
`result`, has the return type (`String` when none is written). Types are
Julia's: `String`, numbers, `Bool`, an `@enum`, [`OneOf`](@ref), `Vector`,
`Dict`, `NamedTuple`s and structs, `Union{T,Nothing}`.

Settings go before `function`: `@ai lm = "claude-haiku-4-5" temperature = 0 function …`,
and so do `tools = [f, g]` (Julia functions the model may call), `demos`
(worked examples), `instructions` (an instruction that replaces the written
one) and `module_name` (the call log's module).

```julia
@enum Mood happy unhappy mixed

@ai function mood(review::String)::Mood
    "How does the customer feel about what they bought?"
end

mood("Broke after a day.")        # unhappy
df.mood = mood.(df.review)        # the column, 8 calls at a time
```

An input with a default may be left out: `tone::String = "kind"`. The
default is written in the function's [`interface`](@ref) and sent to the
model whenever the input is left out. A literal (`"kind"`, `3`,
`String[]`, `(a = 1,)`) or a constant whose value can never change
(`const TONE = "kind"`, an `@enum` value) is data: it counts in the
function's [`version`](@ref) by its value, and each call gets its own copy.
A computed default (`day::String = string(today())`) is run at each call
that leaves the input out, as Julia runs a default; the interface holds
the value it gave when the function was defined, and the version counts
its code (`string(today())`), so the version is the same every day and
changes when the code does. A default that uses another input, or a
constant that can change (`const TAGS = ["a"]`: a `Vector` can be pushed
to), is refused when the function is defined.

With several outputs, calling returns them all as a `NamedTuple`
(`(; summary, minutes) = triage(ticket)`). With code after the outputs, that
code runs on them, and its value is what calling returns:

```julia
@ai function price(item::String)::Float64
    "Estimate the price in US dollars."
    usd::Float64 = ai"the price"
    round(usd; digits = 2)
end
```

Each input is bound to its type (contract/programs.md, "Binding a call's
inputs"): `3` for a `String` is `"3"`, `"5"` for an `Int` is `5`, a record
keeps only its fields; one that does not bind is refused
(`InterfaceError`), recorded, before any request. `missing` or `nothing`
for an input whose type takes null is null, sent; for an optional one whose
type does not, the input is left out (its default); for a required one,
`missing` out, the refused call recorded, and no request.
"""
macro ai(args...)
    isempty(args) && throw(ArgumentError("@ai needs a function: @ai function name(x::String)::String … end"))
    fexpr = args[end]
    options = args[1:end-1]
    define_ai(__module__, __source__, options, fexpr, Expr(:macrocall, Symbol("@ai"), nothing, args...))
end

function define_ai(mod::Module, source::LineNumberNode, options, fexpr, whole)
    (fexpr isa Expr && fexpr.head in (:function, :(=)) && length(fexpr.args) == 2) ||
        throw(ArgumentError("@ai goes before a function: @ai function name(x::String)::String … end"))
    name, inputs, ret = parse_head(fexpr.args[1])
    body = statements(fexpr.args[2])

    # the description
    description = ""
    guidance = Dict{String,String}()
    i = 1
    if i <= length(body) && (body[i] isa AbstractString || (body[i] isa Expr && body[i].head === :string))
        body[i] isa Expr && throw(ArgumentError("@ai $name: the description is plain text, without \$ interpolation"))
        description, guidance = split_arguments(body[i])
        i += 1
    end
    input_names = [String(x.name) for x in inputs]
    for g in keys(guidance)
        g in input_names || throw(ArgumentError("@ai $name: `# Arguments` describes $g, which is not an input ($(join(input_names, ", ")))"))
    end

    # the outputs the model writes
    outputs = Tuple{String,Any,Union{Nothing,String}}[]
    while i <= length(body)
        s = body[i]
        (s isa Expr && s.head === :(=) && is_ai_marker(s.args[2])) || break
        target = s.args[1]
        oname, otype = target isa Symbol ? (target, :String) :
                       (target isa Expr && target.head === :(::) && target.args[1] isa Symbol) ? (target.args[1], target.args[2]) :
                       throw(ArgumentError("@ai $name: an output is `name::Type = ai\"words\"`, not $(s)"))
        push!(outputs, (String(oname), otype, marker_desc(s.args[2])))
        i += 1
    end
    rest = body[i:end]
    for s in rest
        s isa Expr && s.head === :(=) && is_ai_marker(s.args[2]) &&
            throw(ArgumentError("@ai $name: declare every output (`… = ai\"…\"`) before code of your own"))
    end
    declared = !isempty(outputs)
    declared || push!(outputs, ("result", something(ret, :String), nothing))
    answer = first(last(outputs))
    output_names = first.(outputs)

    # code of its own, or the model writes the whole body
    returns_answer(x) = x === nothing || x == Symbol(answer) || x == Expr(:return, Symbol(answer)) ||
                        x == Expr(:return, nothing) || x == Expr(:return)
    has_code = !(isempty(rest) || (length(rest) == 1 && returns_answer(only(rest))))
    if length(rest) == 1
        x = only(rest)
        y = x isa Expr && x.head === :return && !isempty(x.args) ? x.args[1] : x
        if y isa Symbol && String(y) in output_names && String(y) != answer
            throw(ArgumentError("@ai $name: the answer is the last output ($answer); to return $y, declare it last"))
        end
    end
    if !has_code && declared && ret !== nothing && length(outputs) > 1
        throw(ArgumentError("@ai $name: with several outputs, calling returns them all as a NamedTuple; " *
                            "drop the return type ::$(ret), or add code that returns what it says"))
    end

    # the inputs, positional then keyword, as Julia binds them. An input with a
    # default may be left out: the binder leaves it out, and the call takes the
    # default from the interface. The default is data (written in the interface
    # and sent to the model; contract/programs.md), so it is a literal or a
    # constant, known when the function is defined; each call gets a copy.
    given = NOTHING_GIVEN
    params = Any[x.has_default ? Expr(:kw, x.name, given) : x.name for x in inputs if !x.keyword]
    kwparams = Any[x.has_default ? Expr(:kw, x.name, given) : x.name for x in inputs if x.keyword]
    dict = :($(given_inputs)($((:($(String(x.name)) => $(x.name)) for x in inputs)...)))
    binder = isempty(kwparams) ? Expr(:->, Expr(:tuple, params...), dict) :
             Expr(:->, Expr(:tuple, Expr(:parameters, kwparams...), params...), dict)
    row = gensym(:row)
    assigns = Any[]
    for x in inputs
        col = String(x.name)
        absent = x.has_default ? given : :(throw(ArgumentError($("the row has no column $col for $name's input"))))
        push!(assigns, :($(x.name) = haskey($row, $col) ? $row[$col] : $absent))
    end
    from_row = Expr(:->, row, Expr(:block, assigns..., dict))
    defaults = Expr(:tuple, (Expr(:(=), x.name, ai_default(mod, x.default, input_names, String(name), String(x.name)))
                             for x in inputs if x.has_default)...)

    body_fn = nothing
    if has_code
        ins, outs = gensym(:inputs), gensym(:outputs)
        binds = Any[:($(x.name) = $ins[$(String(x.name))]) for x in inputs]
        append!(binds, (:($(Symbol(o)) = $outs.$(Symbol(o))) for o in output_names))
        body_fn = Expr(:->, Expr(:tuple, ins, outs), Expr(:let, Expr(:block, binds...), Expr(:block, rest...)))
    end
    code = has_code ? LMCC.sha256_of(code_data(Base.remove_linenums!(deepcopy(fexpr)))) : nothing

    input_specs = Expr(:tuple, (Expr(:(=), x.name, guidance_spec(x.type, get(guidance, String(x.name), nothing))) for x in inputs)...)
    output_specs = Expr(:tuple, (Expr(:(=), Symbol(o), guidance_spec(t, d)) for (o, t, d) in outputs)...)
    settings = Any[]
    for opt in options
        (opt isa Expr && opt.head === :(=) && opt.args[1] isa Symbol) ||
            throw(ArgumentError("@ai: settings are name = value before `function`; not $(opt)"))
        k = opt.args[1]
        (k in SETTING_NAMES || k in OPTION_NAMES) ||
            (check_setting(k, nothing); throw(ArgumentError("@ai: unknown setting $k")))
        push!(settings, Expr(:kw, k, opt.args[2]))
    end
    mname = mod === Main ? "__main__" : join(string.(fullname(mod)), ".")
    any(o -> o.args[1] === :module_name, options) && (mname = nothing)     # the option says it
    fn = gensym(:fn)
    esc(:($name = let $fn = $(define_function)($(String(name)), $description;
            inputs=$input_specs, outputs=$output_specs, defaults=$defaults, binder=$binder, from_row=$from_row, body=$body_fn,
            code=$code, returns=$(has_code ? ret : nothing), declared_return=$(ret === nothing ? nothing : ret),
            $((mname === nothing ? () : (Expr(:kw, :module_name, mname),))...), file=$(source.file === nothing ? nothing : String(source.file)), line=$(source.line),
            source=$(QuoteNode(Base.remove_linenums!(deepcopy(whole)))), $(settings...))
        $(document)($mod, $(QuoteNode(name)), $fn, $(string(something(source.file, :none))), $(source.line))
        $fn
    end))
end

guidance_spec(type, desc) = desc === nothing || isempty(desc) ? type : :($type => $desc)

"The inputs a call gave, by name (an input left out is not there)."
given_inputs(pairs::Pair...) = OrderedDict{String,Any}(k => v for (k, v) in pairs if v !== NOTHING_GIVEN)

# ------------------------------------------------------------------ defaults that are data

"Whether an expression names one of `names` (a quoted symbol, `:x`, names nothing)."
mentions(x, names) = x isa Symbol ? String(x) in names :
                     x isa Expr ? x.head !== :quote && any(a -> mentions(a, names), x.args) : false

"""
A literal default: a number, text, a Bool, `nothing`, a quoted symbol, and
lists, tuples, named tuples, typed vectors (`String[]`, `T[…]` where `T`
names a type in `mod`: `COUNTS[1]` reads a global, and is not one) and
`Dict`s of them.
"""
function is_literal(x, mod::Module)
    lit(a) = is_literal(a, mod)
    x isa Union{Number,AbstractString} && return true
    x === :nothing && return true
    x isa QuoteNode && return x.value isa Symbol
    x isa Expr || return false
    named(a) = a isa Expr && a.head === :(=) && a.args[1] isa Symbol && lit(a.args[2])
    x.head === :vect && return all(lit, x.args)
    x.head === :tuple && return all(a -> lit(a) || named(a), x.args)
    x.head === :parameters && return all(a -> named(a) || (a isa Expr && a.head === :kw && lit(a.args[2])), x.args)
    x.head === :ref && return !isempty(x.args) && names_type(mod, x.args[1]) && all(lit, x.args[2:end])
    pair(a) = a isa Expr && a.head === :call && a.args[1] === :(=>) && length(a.args) == 3 && lit(a.args[2]) && lit(a.args[3])
    x.head === :call && x.args[1] === :Dict && return all(pair, x.args[2:end])
    false
end
"""
Whether `x` names a type in `mod` (a typed vector's element type: `String`
in `String[]`, `Vector{Int}` in `Vector{Int}[]`, `Base.String`): a constant
holding a type, or one applied to such names and numbers.
"""
function names_type(mod::Module, x)
    is_name_path(x) && return (v = constant_value(mod, x); v !== NOTHING_GIVEN && v isa Type)
    x isa Expr && x.head === :curly || return false
    names_type(mod, x.args[1]) && all(a -> a isa Integer || names_type(mod, a), x.args[2:end])
end

"""
Whether a value can never change: numbers, characters, text, symbols,
`nothing`, `missing`, enum values, and immutable values (tuples, named
tuples, immutable structs) made only of them. A constant holding one is the
same value on every call; a constant holding a `Vector` or a `Dict` is not
(`push!(ITEMS, …)` changes it after the function is defined).
"""
frozen(v) = v isa Union{Number,AbstractChar,String,Symbol,Nothing,Missing,Enum} ? true :
            ismutable(v) ? false : all(i -> !isdefined(v, i) || frozen(getfield(v, i)), 1:nfields(v))

"A name, or a dotted path of names (`TONE`, `Settings.TONE`): what may be a constant."
is_name_path(x) = x isa Symbol || (x isa Expr && x.head === :. && length(x.args) == 2 && is_name_path(x.args[1]) &&
                                   x.args[2] isa QuoteNode && x.args[2].value isa Symbol)

"The value of a constant named by a name path in `mod`, or `NOTHING_GIVEN` when it is not a constant."
function constant_value(mod::Module, x)
    owner, name = if x isa Symbol
        (mod, x)
    else
        m = constant_value(mod, x.args[1])
        m isa Module || return NOTHING_GIVEN
        (m, x.args[2].value)
    end
    isdefined(owner, name) && isconst(owner, name) ? getfield(owner, name) : NOTHING_GIVEN
end

"""
A `@program` argument's default as the interface writes it, when it is data
(a literal, or a constant whose value can never change, and no other
argument): the expression that gives it when the program is defined, else
`NOTHING_GIVEN` (it is Julia code, run on each call as Julia runs it, and the
input is optional with no default in its shape: a call that leaves it out
records no value for it). The code evaluates a data default too, on each
call; its value is the interface's, so the record holds what the code got.
"""
function default_data(mod::Module, default, arg_names)
    mentions(default, arg_names) && return NOTHING_GIVEN
    is_literal(default, mod) && return default
    is_name_path(default) && return :($(frozen_constant)($mod, $(QuoteNode(default))))
    NOTHING_GIVEN
end

"The value of a constant that can never change, named by `x` in `mod`, else `NOTHING_GIVEN`."
function frozen_constant(mod::Module, x)
    v = constant_value(mod, x)
    v !== NOTHING_GIVEN && frozen(v) ? v : NOTHING_GIVEN
end

"""
An `@ai` input's default, as the expression that gives it when the function
is defined: it is written in the interface and sent to the model whenever
the input is left out, so it is data, the same for every call: a literal or
a constant whose value can never change. Anything else is refused when the
function is defined, rather than computed once and silently shared (`time()`
would be the definition's time on every call; a constant `Vector` would be
its contents then, whatever was pushed to it since).
"""
function ai_default(mod::Module, default, input_names, fname, input)
    if mentions(default, input_names)
        used = join(sort!([n for n in input_names if mentions(default, [n])]), ", ")
        return :(throw(ArgumentError($("@ai $fname: the default of $input uses $used, another input. A default is data, " *
                                        "sent to the model whenever $input is left out: write a literal or a constant, or give $input at each call"))))
    end
    is_literal(default, mod) && return default
    is_name_path(default) && return :($(ai_constant)($mod, $(QuoteNode(default)), $fname, $input))
    # computed (`today()`): run at each call that leaves the input out, as Julia runs a default; the version counts
    # its code, and the interface holds the value it gives when the function is defined (calls.md, "Defaults")
    code = string(Base.remove_linenums!(deepcopy(default)))
    :($(ComputedDefault)($code, () -> $default))
end

function ai_constant(mod::Module, x, fname, input)
    v = constant_value(mod, x)
    v === NOTHING_GIVEN && throw(ArgumentError("@ai $fname: the default of $input ($x) is not a constant. A default is data, " *
        "sent to the model whenever $input is left out, the same for every call: make it one (const $x = …), or write a literal"))
    frozen(v) || throw(ArgumentError("@ai $fname: the default of $input ($x) is a constant whose value can change (a " *
        "$(typeof(v)): what it holds can be changed after the function is defined). A default is data, written in the " *
        "function's interface when it is defined and sent to the model whenever $input is left out, the same for every " *
        "call: write it as a literal, or give $input at each call"))
    v
end

"The @ai macro's constructor: checks what only evaluated types can say, then makes the function."
function define_function(name, description; inputs, outputs, declared_return, code, body, kw...)
    if body === nothing && declared_return !== nothing && length(outputs) == 1
        spec = only(values(outputs))
        spec = spec isa Pair ? first(spec) : spec
        spec == declared_return || throw(ArgumentError(
            "@ai $name: it returns its answer, declared as $(spec_text(spec)), but its return type says $(spec_text(declared_return))"))
    end
    AIFunction(name, description; inputs, outputs, code, body, kw...)
end

"The docstring `?name` shows: the call, and what the model is told it does."
function document(mod::Module, name::Symbol, f::AIFunction, file::String, line::Int)
    d = f.definition
    text = "```julia\n$(sprint(show, f))\n```\n\n" * (isempty(d.description) ? "" : d.description * "\n\n") *
           "An AI function: a language model writes its answer. " *
           "See `FunctAI.instructions($name)` for what the model is told, `render($name, …)` for the request."
    try
        Base.Docs.doc!(mod, Base.Docs.Binding(mod, name), Base.Docs.docstr(text, Dict{Symbol,Any}(:module => mod, :path => file, :linenumber => line, :binding => Base.Docs.Binding(mod, name))), Union{})
    catch
        # documenting is a courtesy: a binding that cannot hold docs keeps its function
    end
    nothing
end
