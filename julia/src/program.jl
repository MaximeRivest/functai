# Programs: ordinary Julia code that calls AI functions, followed as one call
# with theirs as its children (in the call log and in a stream). The call
# log names them `kind: "module"`, as every FunctAI language does. Each has
# an interface (contract/programs.md), derived from its typed arguments and
# return type, checked when it is defined and on every call.
#
#     @program function support(ticket::String; tone::String = "kind")::String
#         "Answer a support ticket."
#         t = triage(ticket)
#         t.minutes > 60 ? escalate(ticket) : reply(ticket; tone)
#     end

"""
    AIProgram

Julia code that calls AI functions, made by [`@program`](@ref) (or from an
interface as data: `AIProgram(name, code; interface)`). Calling it checks
its inputs against its [`interface`](@ref), runs the code, and checks what
the code returned; the AI functions it calls are its children in the call
log and in a stream. Its [`version`](@ref) changes when its code or its
interface changes, or when an AI function it names is improved.
"""
struct AIProgram <: Function
    name::String
    mod::Union{Nothing,Module}
    module_name::String
    run::Any                        # (inputs::OrderedDict{String,Any}) -> what the code returns
    binder::Any                     # (args...; kw...) -> the inputs given, by name
    code::String
    body::Any
    file::Union{Nothing,String}
    line::Union{Nothing,Int}
    interface::JObj
    own_defaults::Bool              # its code has its own defaults (@program): it gets only the inputs given
    returns::Any                    # the declared return type, or nothing
    output_types::Dict{String,Any}  # the declared type of each of several outputs
end
Base.nameof(p::AIProgram) = Symbol(p.name)
Base.show(io::IO, p::AIProgram) = print(io, "program ", p.name, "(", join((f["name"] for f in p.interface["inputs"]), ", "), ") (", p.module_name, ")")
(p::AIProgram)(args...; kw...) = run_program(p, p.binder(args...; kw...))

"""
The inputs a call gave, by name: positional arguments in the order of
`positional`, keywords by their names (any input may be given by name, as
a call from data gives them). What the interface cannot take is kept to be
refused as the call's own error: arguments beyond the positional ones, and
an input given twice.
"""
function bind_named(positional::Vector{String}, args, kw)
    given = OrderedDict{String,Any}()
    for (i, a) in enumerate(args)
        i <= length(positional) && (given[positional[i]] = a)
    end
    twice = String[]
    for (k, v) in kw
        key = String(k)
        haskey(given, key) && push!(twice, key)
        given[key] = v
    end
    (given=given, extra=max(0, length(args) - length(positional)), twice=sort!(twice), positional=positional)
end

"The refusal of what a call gave that binds to no input (programs.md: `interface-input`), or `nothing`."
function binding_refusal(p::AIProgram, bound)
    isempty(bound.twice) || return InterfaceError("interface-input", bound.twice[1], "$(bound.twice[1]) was given twice (by position and by name)")
    bound.extra > 0 && return InterfaceError("interface-input", nothing,
        "$(p.name) takes $(length(bound.positional)) input(s) by position ($(join(bound.positional, ", "))), not $(length(bound.positional) + bound.extra)")
    nothing
end

interface(p::AIProgram) = p.interface
interface_signature(p::AIProgram) = interface_signature(p.interface)

"Every symbol the code names."
function names_in(x, out=Set{Symbol}())
    x isa Symbol && push!(out, x)
    x isa Expr && foreach(a -> names_in(a, out), x.args)
    x isa QuoteNode && names_in(x.value, out)
    out
end

"The AI functions and programs a program's code names (resolved now, in its module)."
function reaches(p::AIProgram)
    p.mod === nothing && return Any[]
    found = Any[]
    for n in sort!(collect(names_in(p.body)))
        isdefined(p.mod, n) || continue
        v = getfield(p.mod, n)
        (v isa AIFunction || (v isa AIProgram && v !== p)) && push!(found, v)
    end
    found
end

function version_parts!(p::AIProgram, code::JObj, ai::JObj, seen::Set{UInt})
    objectid(p) in seen && return
    push!(seen, objectid(p))
    code["$(p.module_name):$(p.name)"] = p.code
    for x in reaches(p)
        x isa AIFunction ? (ai["$(x.module_name):$(x.definition.name)"] = version(x)) : version_parts!(x, code, ai, seen)
    end
end

"""
    version(p::AIProgram)

`{"code": {key: C}, "ai": {key: version}, "interface": I}` hashed
(contract/calls.md, "Versions"): the program's code, and that of the
programs it names, the version of every AI function they name, and its
interface. Plain Julia functions it calls are not followed: a change inside
one does not change the version.
"""
function version(p::AIProgram)
    code, ai = JObj(), JObj()
    version_parts!(p, code, ai, Set{UInt}())
    LMCC.sha256_of(LMCC.jobj("code" => code, "ai" => ai, "interface" => p.interface))
end

function program_of(p::AIProgram)
    out = LMCC.jobj("name" => p.name, "kind" => "module", "module" => p.module_name, "version" => version(p),
                    "interface" => interface_signature(p), "answer" => last(p.interface["outputs"])["name"])
    p.file === nothing || (out["file"] = p.file)
    p.line === nothing || (out["line"] = p.line)
    out
end

"""
Run a program as one call: its inputs checked (and optional ones left out
given their defaults), its code, what it returned checked; logged and
watched, with the AI calls inside it as its children. A call its interface
refuses is a call too: it starts, fails with the `InterfaceError`, and is
recorded (with only the inputs the interface names); its code does not run.
"""
function run_program(p::AIProgram, bound)
    s = effective(Dict{Symbol,Any}())
    names = [f["name"] for f in p.interface["inputs"]]
    fields = (inputs=names, outputs=[f["name"] for f in p.interface["outputs"]], added=String[])
    refusal = binding_refusal(p, bound)
    checked = nothing
    if refusal === nothing
        try
            checked = check_inputs(p.interface, bound.given)
        catch err
            err isa InterfaceError || rethrow()
            refusal = err
        end
    end
    # the record names the inputs as the interface does: the values given, and defaults taken
    # (a call refused for an input the interface lacks did not receive it)
    logged = checked !== nothing ? checked : OrderedDict{String,Any}(k => v for (k, v) in bound.given if k in names)
    call = start_call(program_of(p), p.name, s, Dict{Symbol,Any}(), logged, fields)
    run_call(call) do call
        refusal === nothing || throw(refusal)
        # @program's code takes its own defaults, made anew on each call as Julia makes them; a program
        # from data gets the interface's (a copy each call)
        inputs = p.own_defaults ? OrderedDict{String,Any}(k => v for (k, v) in checked if haskey(bound.given, k)) : checked
        value = as_declared(p, p.run(inputs))
        call.outputs = check_returned(p.interface, value)
        value
    end
end

"A value as a declared type: itself, converted, or read from its JSON form; else as it is (the interface says why it does not fit)."
function as_output(T, v)
    (T isa Type && !(v isa T)) || return v
    try
        return convert(T, v)
    catch
    end
    data = json_form(v)
    if data !== NOJSON
        try
            return fromjson(T, data, "")
        catch
        end
    end
    v
end

"What the code returned, as its declaration types it (as a Julia function's return type converts): one output, or each of several."
function as_declared(p::AIProgram, value)
    p.returns !== nothing && return as_output(p.returns, value)
    isempty(p.output_types) && return value
    if value isa NamedTuple
        ks = keys(value)
        return NamedTuple{ks}(Tuple(as_output(get(p.output_types, String(k), nothing), value[k]) for k in ks))
    elseif value isa AbstractDict
        out = empty(value, keytype(value), Any)
        for (k, v) in value
            out[k] = as_output(get(p.output_types, string(k), nothing), v)
        end
        return out
    end
    value
end

# ------------------------------------------------------------------ an interface from Julia's declarations

"Whether a field's declared Julia type is opaque (no type, `Any`, or a type with no JSON form) and its shape."
function declared_shape(T)
    (T === nothing || T === Any) && return (true, JObj())
    try
        (false, shape_of(T))
    catch err
        err isa ArgumentError || rethrow()
        (true, JObj())
    end
end

function declared_field(name, T; optional::Bool=false, default=NOTHING_GIVEN)
    opaque, shape = declared_shape(T)
    out = LMCC.jobj("name" => String(name))
    if !opaque && default !== NOTHING_GIVEN
        data = json_form(default)
        data === NOJSON || (shape["default"] = data)
    end
    out["shape"] = shape
    T === nothing || (out["type"] = string(T))
    opaque && (out["opaque"] = true)
    optional && (out["optional"] = true)
    out
end

"""
The interface `@program` declares (contract/programs.md, "How each program
has one"): each argument an input (untyped, `Any`, or a type with no JSON
form: opaque); an argument with a default is optional, and a default that
is data (a literal, or a constant: the rule `@ai` keeps) is in its shape;
any other default is Julia code, and the input stays left out. The code
always makes its defaults anew on each call, as Julia does. The return type
is one output, `result`, or `outputs = (name = T, …)` declares several.
"""
function define_program(name, mod, module_name, code, body, file, line; description, inputs, outputs, returns, positional, run)
    ins = Any[declared_field(n, T; optional=has_default, default=data) for (n, T, has_default, data) in inputs]
    outs = if outputs !== nothing
        Any[declared_field(n, T) for (n, T) in pairs(outputs)]
    else
        Any[declared_field("result", returns)]
    end
    iface = LMCC.jobj("description" => description, "inputs" => ins, "outputs" => outs)
    check_interface(iface; what="@program $name")
    order = String[positional...]
    binder = (args...; kw...) -> bind_named(order, args, kw)
    types = outputs === nothing ? Dict{String,Any}() : Dict{String,Any}(String(n) => T for (n, T) in pairs(outputs))
    AIProgram(name, mod, module_name, run, binder, code, body, file, line, iface, true, outputs === nothing ? returns : nothing, types)
end

"""
    AIProgram(name, code; interface, module_name = "__main__")

A program from its interface as data (contract/programs.md): `code` is
called with the inputs as keyword arguments (an optional input left out
with no default in its shape is not passed: its own default applies), and
returns the output, or a record of the outputs by name. Called with
positional arguments in the interface's order, or keywords.
"""
function AIProgram(name::AbstractString, code; interface::AbstractDict, module_name::AbstractString="__main__")
    iface = LMCC.deepcopy_json(interface)
    check_interface(iface; what="program $name")
    names = String[f["name"] for f in iface["inputs"]]
    binder = (args...; kw...) -> bind_named(names, args, kw)
    run = inputs -> code(; (Symbol(k) => v for (k, v) in inputs)...)
    code_hash = LMCC.sha256_of(LMCC.jobj("program" => String(name), "code" => string(code)))
    AIProgram(String(name), nothing, String(module_name), run, binder, code_hash, nothing, nothing, nothing, iface, false, nothing, Dict{String,Any}())
end

"A value given to a typed argument, as that type: itself, converted, or read from its JSON form."
function as_input(::Type{T}, v, name) where {T}
    v isa T && return v
    try
        return convert(T, v)
    catch
    end
    data = json_form(v)
    if data !== NOJSON
        try
            return fromjson(T, data, name)
        catch
        end
    end
    throw(InterfaceError("interface-input", name, "$name: a $(typeof(v)) is not a $T"))
end


"""
    @program [outputs = (name = T, …)] function name(args...; kw...)::T
        "What it does."
        … code that calls AI functions …
    end

A program: Julia code that calls AI functions, followed as one call. In the
call log it is the parent of every AI call it makes, and a [`stream`](@ref)
of it shows them all. Broadcasting it (`support.(tickets)`) runs the rows
`concurrency` at a time.

Its [`interface`](@ref) is read from the declaration (contract/programs.md):
each argument is an input, typed by its Julia type (an untyped argument,
`Any`, or a type with no JSON form is opaque: never checked); an argument
with a default may be left out (a default that is data, a literal or a
constant, is written in the interface; the code makes its defaults anew on
each call, as Julia does); the return type is its one output, `result`
(none: opaque), or `outputs = (summary = String, minutes = Int)` declares
several, returned as a `NamedTuple` or `Dict` and converted to the declared
types. Arguments are given by position, as declared, or any of them by name.
Every call checks its inputs before the code runs and its outputs when it
returns (`InterfaceError`, naming the field): a refused call is still a call
(its events, its record), and its code does not run. The first string of
the body is its description.

A call made inside it, even on a task it starts, is a step of it: its call
ends once they all have. Start work meant to outlive it with
[`FunctAI.detached`](@ref).
"""
macro program(args...)
    isempty(args) && throw(ArgumentError("@program needs a function: @program function name(x::String) … end"))
    fexpr = args[end]
    outputs = nothing
    for opt in args[1:end-1]
        (opt isa Expr && opt.head === :(=) && opt.args[1] === :outputs) ||
            throw(ArgumentError("@program: the one option before `function` is outputs = (name = Type, …); not $(opt)"))
        outputs = opt.args[2]
    end
    (fexpr isa Expr && fexpr.head in (:function, :(=)) && length(fexpr.args) == 2) ||
        throw(ArgumentError("@program goes before a function: @program function name(x::String) … end"))
    head = fexpr.args[1]
    ret = nothing
    if head isa Expr && head.head === :(::)
        head, ret = head.args[1], head.args[2]
    end
    (head isa Expr && head.head === :call && head.args[1] isa Symbol) || throw(ArgumentError("@program: expected function name(args...) … end"))
    ret !== nothing && outputs !== nothing && throw(ArgumentError("@program: give a return type (one output) or outputs = (…) (several), not both"))
    name = head.args[1]
    parsed = Any[]           # (name, type expression or nothing, default expression or nothing, has default, keyword)
    for a in head.args[2:end]
        keyword = a isa Expr && a.head === :parameters
        for x in (keyword ? a.args : (a,))
            default, has_default = nothing, false
            if x isa Expr && x.head === :kw
                x, default, has_default = x.args[1], x.args[2], true
            end
            x isa Expr && x.head === :... && throw(ArgumentError("@program: an input is named; no varargs ($(x))"))
            T = x isa Expr && x.head === :(::) ? x.args[2] : nothing
            y = x isa Expr && x.head === :(::) ? x.args[1] : x
            y isa Symbol || throw(ArgumentError("@program: cannot read the argument $(x)"))
            push!(parsed, (y, T, default, has_default, keyword))
        end
    end
    # positional first, then keywords: the order they are written in
    parsed = vcat([p for p in parsed if !p[5]], [p for p in parsed if p[5]])
    body = fexpr.args[2]
    stmts = statements(body)
    description = !isempty(stmts) && stmts[1] isa AbstractString ? String(stmts[1]) : ""
    given = NOTHING_GIVEN
    arg_names = [String(p[1]) for p in parsed]
    positional = [String(p[1]) for p in parsed if !p[5]]
    ins = gensym(:inputs)
    binds = Any[]
    for (n, T, default, has_default, _) in parsed
        key = String(n)
        value = T === nothing ? :($ins[$key]) : :($(as_input)($T, $ins[$key], $key))
        push!(binds, has_default ? :($n = haskey($ins, $key) ? $value : $default) : :($n = $value))
    end
    run = Expr(:->, ins, Expr(:let, Expr(:block), Expr(:block, binds..., body)))
    # a default that is data (a literal, or a constant, and no other input) is in the interface
    input_specs = Expr(:vect, (Expr(:tuple, String(n), T === nothing ? nothing : T, has_default,
                                    has_default ? default_data(__module__, default, arg_names) : given)
                               for (n, T, default, has_default, _) in parsed)...)
    code = LMCC.sha256_of(code_data(Base.remove_linenums!(deepcopy(fexpr))))
    mname = __module__ === Main ? "__main__" : join(string.(fullname(__module__)), ".")
    body_q = QuoteNode(Base.remove_linenums!(deepcopy(body)))
    file = __source__.file === nothing ? nothing : String(__source__.file)
    esc(:($name = $(define_program)($(String(name)), $__module__, $mname, $code, $body_q, $file, $(__source__.line);
                                    description=$description, inputs=$input_specs, outputs=$outputs, returns=$ret,
                                    positional=$positional, run=$run)))
end

Base.Broadcast.broadcasted(p::AIProgram, args...) = (t = Base.Broadcast.materialize(Base.Broadcast.broadcasted(tuple, args...));
                                                     t isa Tuple ? p(t...) : call_each(p, t))
Base.map(p::AIProgram, xs::AbstractArray, more::AbstractArray...) = call_each(p, map(tuple, xs, more...))
