# Layouts: how a call is written as messages and how the reply is read back
# (contract/functions.md, "The layout"). A layout is an lmcc adapter; the
# three named ones are the contract's artifacts, carried as data.

"The whole reply is the one output's value: a template with no output pattern."
struct ReplyReader <: LMCC.Reader
    spec::JObj
end
function reply_reader(spec)
    any(k -> k != "kind", keys(spec)) && LMCC.refuse("entry-malformed", "reader: functai_reply takes only 'kind'";
                                                    fix=LMCC.jobj("action" => "edit-entry", "path" => "reader"))
    ReplyReader(JObj(String(k) => v for (k, v) in spec))
end
LMCC.reader_spec(r::ReplyReader) = r.spec
function LMCC.reader_split(::ReplyReader, text, names)
    length(names) == 1 || LMCC.refuse("parse-ambiguous",
        "the template has no output pattern, so the reply can hold one output, not $(length(names)): $(join(names, ", "))")
    JObj(String(first(names)) => LMCC.wstrip(text))
end
LMCC.reader_join(::ReplyReader, spelled) = join((last(p) for p in spelled), "\n")

"The registry every plan binds with: lmcc's standard pack and FunctAI's reply reader."
const REGISTRY = Ref{Any}(nothing)
const LAYOUT_LOCK = ReentrantLock()
function registry()
    lock(LAYOUT_LOCK) do
        if REGISTRY[] === nothing
            reg = LMCC.Registry()
            LMCC.Std.install!(reg)
            LMCC.register_reader!(reg, "functai_reply", reply_reader; version="1.0.0", exist_ok=true)
            REGISTRY[] = reg
        end
        REGISTRY[]
    end
end

const NAMED_LAYOUTS = Dict("xml" => "xml", "default" => "xml", "tags" => "xml", "chat" => "chat",
                           "chatadapter" => "chat", "json" => "json", "jsonadapter" => "json")
const LOADED_LAYOUTS = Dict{String,Any}()

"A named layout (`:xml`, `:chat`, `:json`), an lmcc `Adapter`, or an adapter artifact (a Dict)."
function resolve_adapter(adapter)
    adapter === nothing && return resolve_adapter("xml")
    adapter isa LMCC.Adapter && return adapter
    if adapter isa Union{Symbol,AbstractString}
        key = replace(lowercase(String(adapter)), r"[-_ ]" => "")
        name = get(NAMED_LAYOUTS, key, nothing)
        name === nothing && throw(ArgumentError("unknown adapter $(repr(adapter)); use :xml, :chat, :json, an lmcc adapter, or template = [...]"))
        return lock(LAYOUT_LOCK) do
            get!(() -> LMCC.load(LMCC.deepcopy_json(LAYOUT_DATA[name]); registry=registry()), LOADED_LAYOUTS, name)
        end
    end
    adapter isa AbstractDict && return LMCC.load(LMCC.deepcopy_json(adapter); registry=registry())
    throw(ArgumentError("adapter must be :xml, :chat, :json, an lmcc adapter or an artifact, not $(repr(adapter))"))
end

"A named layout's name, for the saved manifest; an adapter as its artifact."
adapter_data(adapter) = adapter isa Union{Symbol,AbstractString} ? String(adapter) :
                        adapter isa LMCC.Adapter ? LMCC.dump(adapter; registry=registry()) : adapter

"""
A template's messages as lmcc's: `:system => "…"`, `:user => "…"`,
`:assistant => "…"`, `:developer => "…"`, `:turns`, or lmcc message Dicts.
"""
function template_messages(messages)
    out = JObj[]
    for (i, m) in enumerate(messages)
        if m isa Pair
            role = String(first(m))
            role in ("system", "developer", "user", "assistant") ||
                throw(ArgumentError("template[$i]: a role is :system, :developer, :user or :assistant, not $(repr(first(m)))"))
            push!(out, LMCC.jobj("role" => role, "text" => String(last(m))))
        elseif m === :turns || m == "turns"
            push!(out, LMCC.jobj("directive" => "turns"))
        elseif m isa AbstractDict
            d = JObj(String(k) => v for (k, v) in m)
            if haskey(d, "content") && !haskey(d, "text")
                d["text"] = pop!(d, "content")
            end
            push!(out, d)
        else
            throw(ArgumentError("template[$i]: expected :system => \"…\", :user => \"…\", :assistant => \"…\", :turns, or an lmcc message; got $(repr(m))"))
        end
    end
    out
end

"A layout from a template: the xml layout's formats and transports; `turns` before the last user message."
function template_adapter(messages, reader="derived")
    msgs = template_messages(messages)
    if !any(m -> haskey(m, "directive"), msgs)
        last_user = findlast(m -> get(m, "role", nothing) == "user", msgs)
        insert!(msgs, last_user === nothing ? length(msgs) + 1 : last_user, LMCC.jobj("directive" => "turns"))
    end
    xml = LAYOUT_DATA["xml"]
    LMCC.load(LMCC.jobj("name" => "functai_template", "versions" => LMCC.deepcopy_json(xml["versions"]),
                        "template" => Any[msgs...], "reader" => LMCC.jobj("kind" => reader),
                        "transports" => LMCC.deepcopy_json(xml["transports"]), "formats" => LMCC.deepcopy_json(xml["formats"]));
              registry=registry())
end

const INPUT_TAGS = "{% for f in inputs %}<{f.name}>\n{f.value}\n</{f.name}>\n{% endfor %}"

"The layout of a judgment-only provider: the input(s) as the user message; each output a typed question."
function judgment_layout(sig::LMCC.Signature)
    data = LMCC.signature_to_dict(sig)
    fields = data["fields"]
    inputs = [f for f in fields if f["direction"] == "input" && get(f, "purpose", "plain") == "plain"]
    outputs = [f for f in fields if f["direction"] == "output"]
    doc = data["instructions"]
    new_fields = Any[]
    for f in fields
        f = copy(f)
        if f["direction"] == "output" && isempty(something(get(f, "desc", nothing), ""))
            isempty(doc) || (f["desc"] = length(outputs) == 1 ? doc : "$doc ($(f["name"]))")
        end
        push!(new_fields, f)
    end
    body = length(inputs) == 1 ? "{$(inputs[1]["name"])}" : INPUT_TAGS
    xml = LAYOUT_DATA["xml"]
    adapter = LMCC.load(LMCC.jobj("name" => "functai_judgment",
                                  "versions" => LMCC.jobj("kernel" => xml["versions"]["kernel"],
                                                          "vocab" => LMCC.jobj("reader/json_object" => "0.2.1", "format/json" => "0.1.0")),
                                  "template" => Any[LMCC.jobj("directive" => "turns"), LMCC.jobj("role" => "user", "text" => body)],
                                  "reader" => LMCC.jobj("kind" => "json_object"), "formats" => LMCC.jobj("*" => LMCC.jobj("use" => "json")));
                        registry=registry())
    (adapter, LMCC.signature_from_dict(LMCC.jobj("instructions" => doc, "fields" => new_fields)))
end

"Bind a layout to a signature for a model's facts: every refusal fires here, before anything is sent."
function bind_layout(adapter, template, sig::LMCC.Signature, caps::AbstractDict, provider::AbstractString)
    capabilities = JObj(String(k) => v for (k, v) in caps)
    if template !== nothing
        try
            return LMCC.bind(template_adapter(template), sig; capabilities, registry=registry())
        catch err
            (err isa LMCC.Refusal && err.code == "not-readable" && err.fix !== nothing && get(err.fix, "path", nothing) == "template") || rethrow()
        end
        visible = [f.name for f in sig.fields if f.direction == "output" && f.purpose == "plain"]
        length(visible) == 1 || throw(LMCC.Refusal("not-readable",
            "the template has no output pattern, so the reply can only be one output, but this function has $(length(visible)): $(join(visible, ", ")). Add the pattern to the template";
            fix=LMCC.jobj("action" => "edit-template", "path" => "template")))
        return LMCC.bind(template_adapter(template, "functai_reply"), sig; capabilities, registry=registry())
    end
    if adapter === nothing && provider in JUDGMENT_ONLY
        a, s = judgment_layout(sig)
        return LMCC.bind(a, s; capabilities, registry=registry())
    end
    LMCC.bind(resolve_adapter(adapter), sig; capabilities, registry=registry())
end
