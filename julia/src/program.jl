# Programs: ordinary Julia code that calls AI functions, followed as one call
# with theirs as its children (in the call log and in a stream). The call
# log names them `kind: "module"`, as every FunctAI language does.
#
#     @program function support(ticket::String)
#         t = triage(ticket)
#         t.minutes > 60 ? escalate(ticket) : reply(ticket)
#     end

"""
    AIProgram

Julia code that calls AI functions, made by [`@program`](@ref). Calling it
runs the code; the AI functions it calls are its children in the call log
and in a stream. Its [`version`](@ref) changes when its code changes or when
an AI function it names is improved.
"""
struct AIProgram <: Function
    name::String
    mod::Module
    module_name::String
    run::Any
    code::String
    body::Any
    file::Union{Nothing,String}
    line::Union{Nothing,Int}
end
Base.nameof(p::AIProgram) = Symbol(p.name)
(p::AIProgram)(args...; kw...) = p.run(args...; kw...)
Base.show(io::IO, p::AIProgram) = print(io, "program ", p.name, " (", p.module_name, ")")

"Every symbol the code names."
function names_in(x, out=Set{Symbol}())
    x isa Symbol && push!(out, x)
    x isa Expr && foreach(a -> names_in(a, out), x.args)
    x isa QuoteNode && names_in(x.value, out)
    out
end

"The AI functions and programs a program's code names (resolved now, in its module)."
function reaches(p::AIProgram)
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

`{"code": {key: C}, "ai": {key: version}}` hashed (contract/calls.md,
"Versions"): the program's code, and that of the programs it names, and the
version of every AI function they name. Plain Julia functions it calls are
not followed: a change inside one does not change the version.
"""
function version(p::AIProgram)
    code, ai = JObj(), JObj()
    version_parts!(p, code, ai, Set{UInt}())
    LMCC.sha256_of(LMCC.jobj("code" => code, "ai" => ai))
end

function program_of(p::AIProgram)
    out = LMCC.jobj("name" => p.name, "kind" => "module", "module" => p.module_name, "version" => version(p), "answer" => "result")
    p.file === nothing || (out["file"] = p.file)
    p.line === nothing || (out["line"] = p.line)
    out
end

"Run a program's code as one call: logged, watched, with the AI calls inside it as children."
function run_program(p::AIProgram, inputs::AbstractDict, code::Function; watch=nothing)
    watch = watch === nothing ? WATCHING[] : watch
    watch_check(watch)
    call = start_call(() -> program_of(p), effective(Dict{Symbol,Any}()), inputs)
    with(CURRENT_CALL => call, WATCHING => watch) do
        watch_started(watch, call, inputs)
        try
            value = code()
            finish_call(call; returned=value, has_returned=true)
            watch_ended(watch, call, value)
            value
        catch err
            finish_call(call; error=err)
            watch_ended(watch, call, nothing, err)
            rethrow()
        end
    end
end

"""
    @program function name(args...)
        … code that calls AI functions …
    end

A program: Julia code that calls AI functions, followed as one call. In the
call log it is the parent of every AI call it makes, and a [`stream`](@ref)
of it shows them all. Broadcasting it (`support.(tickets)`) runs the rows
`concurrency` at a time.
"""
macro program(fexpr)
    (fexpr isa Expr && fexpr.head in (:function, :(=)) && length(fexpr.args) == 2) ||
        throw(ArgumentError("@program goes before a function: @program function name(x::String) … end"))
    head = fexpr.args[1]
    call = head isa Expr && head.head === :(::) ? head.args[1] : head
    (call isa Expr && call.head === :call && call.args[1] isa Symbol) || throw(ArgumentError("@program: expected function name(args...) … end"))
    name = call.args[1]
    names = Symbol[]
    for a in call.args[2:end]
        for x in (a isa Expr && a.head === :parameters ? a.args : (a,))
            y = x isa Expr && x.head === :kw ? x.args[1] : x
            y = y isa Expr && y.head === :(::) ? y.args[1] : y
            y isa Symbol || throw(ArgumentError("@program: cannot read the argument $(x)"))
            push!(names, y)
        end
    end
    ref = gensym(:program)
    dict = :($OrderedDict{String,Any}($((:($(String(n)) => $n) for n in names)...)))
    anon_head = Expr(:tuple, call.args[2:end]...)
    body = fexpr.args[2]
    runner = Expr(:->, anon_head, :($(run_program)($ref[], $dict, () -> $body)))
    code = LMCC.sha256_of(code_data(Base.remove_linenums!(deepcopy(fexpr))))
    mname = __module__ === Main ? "__main__" : join(string.(fullname(__module__)), ".")
    body_q = QuoteNode(Base.remove_linenums!(deepcopy(body)))
    file = __source__.file === nothing ? nothing : String(__source__.file)
    esc(:($name = let $ref = Ref{Any}(nothing)
        $ref[] = $(AIProgram)($(String(name)), $__module__, $mname, $runner, $code, $body_q, $file, $(__source__.line))
    end))
end

Base.Broadcast.broadcasted(p::AIProgram, args...) = (t = Base.Broadcast.materialize(Base.Broadcast.broadcasted(tuple, args...));
                                                     t isa Tuple ? p(t...) : call_each(p, t))
Base.map(p::AIProgram, xs::AbstractArray, more::AbstractArray...) = call_each(p, map(tuple, xs, more...))
