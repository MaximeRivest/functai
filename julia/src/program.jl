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
    defaults::Dict{String,Any}      # native values of the defaults written in the interface
    returns::Any                    # the declared return type, or nothing
end
Base.nameof(p::AIProgram) = Symbol(p.name)
Base.show(io::IO, p::AIProgram) = print(io, "program ", p.name, "(", join((f["name"] for f in p.interface["inputs"]), ", "), ") (", p.module_name, ")")
(p::AIProgram)(args...; kw...) = run_program(p, p.binder(args...; kw...))

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
watched, with the AI calls inside it as its children.
"""
function run_program(p::AIProgram, given::AbstractDict)
    s = effective(Dict{Symbol,Any}())
    names = [f["name"] for f in p.interface["inputs"]]
    fields = (inputs=names, outputs=[f["name"] for f in p.interface["outputs"]], added=String[])
    # the record names the inputs as the interface does: the values given, and defaults taken
    # (a call refused for an input the interface lacks did not receive it)
    logged = try
        check_inputs(p.interface, given; defaults=p.defaults)
    catch err
        err isa InterfaceError || rethrow()
        OrderedDict{String,Any}(String(k) => v for (k, v) in given if String(k) in names)
    end
    call = start_call(program_of(p), p.name, s, Dict{Symbol,Any}(), logged, fields)
    run_call(call) do call
        inputs = check_inputs(p.interface, given; defaults=p.defaults)
        value = p.run(inputs)
        if p.returns !== nothing && !(value isa p.returns)
            value = try
                convert(p.returns, value)            # as a Julia function's return type converts
            catch
                value                                # and the interface says why it does not fit
            end
        end
        call.outputs = check_returned(p.interface, value)
        value
    end
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
form: opaque); an argument with a default is optional, and a default
written as a literal is in its shape (any other default is Julia code, run
on each call: the input stays left out); the return type is one output,
`result`, or `outputs = (name = T, …)` declares several.
"""
function define_program(name, mod, module_name, code, body, file, line; description, inputs, outputs, returns, binder, run)
    ins = Any[declared_field(n, T; optional=has_default, default=literal) for (n, T, has_default, literal) in inputs]
    outs = if outputs !== nothing
        Any[declared_field(n, T) for (n, T) in pairs(outputs)]
    else
        Any[declared_field("result", returns)]
    end
    iface = LMCC.jobj("description" => description, "inputs" => ins, "outputs" => outs)
    check_interface(iface; what="@program $name")
    defaults = Dict{String,Any}(n => literal for (n, T, has_default, literal) in inputs
                                if literal !== NOTHING_GIVEN && haskey(iface["inputs"][findfirst(f -> f["name"] == n, iface["inputs"])]["shape"], "default"))
    AIProgram(name, mod, module_name, run, binder, code, body, file, line, iface, defaults, outputs === nothing ? returns : nothing)
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
    names = [f["name"] for f in iface["inputs"]]
    binder = (args...; kw...) -> begin
        length(args) <= length(names) || throw(InterfaceError("interface-input", nothing, "$name takes $(length(names)) input(s), not $(length(args))"))
        out = OrderedDict{String,Any}(names[i] => a for (i, a) in enumerate(args))
        for (k, v) in kw
            out[String(k)] = v
        end
        out
    end
    run = inputs -> code(; (Symbol(k) => v for (k, v) in inputs)...)
    code_hash = LMCC.sha256_of(LMCC.jobj("program" => String(name), "code" => string(code)))
    AIProgram(String(name), nothing, String(module_name), run, binder, code_hash, nothing, nothing, nothing, iface, Dict{String,Any}(), nothing)
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

"A literal default (a number, text, a Bool, nothing, a symbol, and lists or tuples of them): data, the same on every call."
is_literal(x) = x isa Union{Number,AbstractString,Bool} || x === :nothing || x isa QuoteNode ||
                (x isa Expr && x.head in (:vect, :tuple) && all(is_literal, x.args))

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
with a default may be left out; the return type is its one output,
`result` (none: opaque), or `outputs = (summary = String, minutes = Int)`
declares several, returned as a `NamedTuple` or `Dict`. Every call checks
its inputs before the code runs and its outputs when it returns
(`InterfaceError`). The first string of the body is its description.
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
    params = Any[p[4] ? Expr(:kw, p[1], given) : p[1] for p in parsed if !p[5]]
    kwparams = Any[p[4] ? Expr(:kw, p[1], given) : p[1] for p in parsed if p[5]]
    dict = :($(given_inputs)($((:($(String(p[1])) => $(p[1])) for p in parsed)...)))
    binder = isempty(kwparams) ? Expr(:->, Expr(:tuple, params...), dict) :
             Expr(:->, Expr(:tuple, Expr(:parameters, kwparams...), params...), dict)
    ins = gensym(:inputs)
    binds = Any[]
    for (n, T, default, has_default, _) in parsed
        key = String(n)
        value = T === nothing ? :($ins[$key]) : :($(as_input)($T, $ins[$key], $key))
        push!(binds, has_default ? :($n = haskey($ins, $key) ? $value : $default) : :($n = $value))
    end
    run = Expr(:->, ins, Expr(:let, Expr(:block), Expr(:block, binds..., body)))
    input_specs = Expr(:vect, (Expr(:tuple, String(n), T === nothing ? nothing : T, has_default,
                                    has_default && is_literal(default) ? default : given)
                               for (n, T, default, has_default, _) in parsed)...)
    code = LMCC.sha256_of(code_data(Base.remove_linenums!(deepcopy(fexpr))))
    mname = __module__ === Main ? "__main__" : join(string.(fullname(__module__)), ".")
    body_q = QuoteNode(Base.remove_linenums!(deepcopy(body)))
    file = __source__.file === nothing ? nothing : String(__source__.file)
    esc(:($name = $(define_program)($(String(name)), $__module__, $mname, $code, $body_q, $file, $(__source__.line);
                                    description=$description, inputs=$input_specs, outputs=$outputs, returns=$ret,
                                    binder=$binder, run=$run)))
end

Base.Broadcast.broadcasted(p::AIProgram, args...) = (t = Base.Broadcast.materialize(Base.Broadcast.broadcasted(tuple, args...));
                                                     t isa Tuple ? p(t...) : call_each(p, t))
Base.map(p::AIProgram, xs::AbstractArray, more::AbstractArray...) = call_each(p, map(tuple, xs, more...))
