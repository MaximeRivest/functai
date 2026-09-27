# Tools: Julia functions the model may call. A tool is described to the model
# by its name, its docstring and the JSON Schema of its arguments, all read
# from the function itself:
#
#     "Current weather in a city."
#     weather(city::String) = …
#     @ai tools = [weather] function assistant(question::String)::String
#         "Answer, using tools when needed."
#     end

"""
    AITool

A tool the model may call: `name`, `description`, `parameters` (JSON Schema)
and the Julia function that runs it. Made by [`tool`](@ref), or from a
function given in `tools = [...]`.
"""
struct AITool
    name::String
    description::Union{Nothing,String}
    parameters::JObj
    run::Any
    argnames::Vector{Symbol}
    argspecs::Vector{Any}
end

Base.show(io::IO, t::AITool) = print(io, "tool ", t.name, "(", join(("$n::$(s isa Type ? s : typeof(s))" for (n, s) in zip(t.argnames, t.argspecs)), ", "), ")")

"The docstring of a function, as text, or `nothing` when it has none."
function doc_text(f)
    mod, sym = parentmodule(f), nameof(f)
    b = Base.Docs.Binding(mod, sym)
    meta = Base.Docs.meta(mod)
    haskey(meta, b) || return nothing
    texts = String[]
    for d in values(meta[b].docs)
        push!(texts, join(string.(collect(d.text))))
    end
    text = trim_white(join(texts, "\n\n"))
    isempty(text) ? nothing : text
end

"""
    tool(f; name, description, parameters)

A tool from a Julia function. Its arguments (names and types) come from the
function's one method, its description from its docstring; give `parameters`
(a JSON Schema) and `description` to say them yourself.

```julia
"Look up an order by its number."
lookup(order::String) = orders[order]
@ai tools = [tool(lookup)] function helper(question::String)::String
    "Help with the order."
end
```
"""
function tool(f; name=nothing, description=nothing, parameters=nothing)
    f isa AITool && return f
    tool_name = String(something(name, string(nameof(f))))
    occursin(r"^[A-Za-z_][A-Za-z0-9_-]*$", tool_name) || throw(ArgumentError("a tool's name is letters, digits, _ and -: $(repr(tool_name))"))
    desc = description === nothing ? doc_text(f) : String(description)
    if parameters !== nothing
        return AITool(tool_name, desc, LMCC.deepcopy_json(JObj(String(k) => v for (k, v) in parameters)), f, Symbol[], Any[])
    end
    ms = [m for m in methods(f) if m.nargs >= 1]
    length(ms) == 1 || throw(ArgumentError("tool $tool_name: $(length(ms)) methods, so its arguments are not clear; " *
                                           "define one method, or give parameters = Dict(\"type\" => \"object\", …)"))
    m = only(ms)
    m.isva && throw(ArgumentError("tool $tool_name: a tool takes named arguments, not varargs"))
    names = Base.method_argnames(m)[2:end]
    specs = Any[m.sig.parameters[2:end]...]
    any(n -> n === Symbol("#unused#") || n === Symbol(""), names) &&
        throw(ArgumentError("tool $tool_name: every argument needs a name"))
    props = JObj(String(n) => shape_of(T; where="$tool_name.$n") for (n, T) in zip(names, specs))
    # no additionalProperties: Gemini refuses the keyword in function declarations
    params = LMCC.jobj("type" => "object", "properties" => props, "required" => Any[String(n) for n in names])
    AITool(tool_name, desc, params, f, collect(names), specs)
end

tool_data(t::AITool) = LMCC.jobj("name" => t.name, "description" => t.description, "parameters" => t.parameters)

"Run a tool on the model's input (a JSON object): each argument read as its type."
function invoke_tool(t::AITool, input::AbstractDict)
    isempty(t.argnames) && return t.run(input)
    args = map(t.argnames, t.argspecs) do n, T
        haskey(input, String(n)) || (T >: Nothing ? (return nothing) : throw(ArgumentError("the model gave no $(n)")))
        v = input[String(n)]
        try
            fromjson(T, v, String(n))
        catch err
            err isa LMCC.Refusal ? throw(ArgumentError(err.hint)) : rethrow()
        end
    end
    t.run(args...)
end
