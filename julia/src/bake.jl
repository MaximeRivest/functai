# Baking (contract/baked.md): what a generative student is trained on and
# called with, so a model trained by any language, or by any trainer from
# examples one language wrote, is called correctly by every other.
#
#     table = FunctAI.bake_examples(summarize, rows)                  # the training conversations
#     FunctAI.export_examples("summaries.jsonl", summarize, rows)    # for TRL, Axolotl, Unsloth, a service
#     student = FunctAI.baked("baked/summarize"; url = "http://localhost:8000/v1")   # vLLM serving the folder
#     configure(summarize; lm = student)("...")                      # the function, on the weights
#
# Training itself is not here: Julia writes the examples every trainer reads
# and runs the weights wherever they are served; Python's `functai.bake`
# trains them (here, on Tinker, on Prime), or any trainer does from the
# exported table. Weights trained elsewhere are adopted when their folder's
# `baked.json` says which function they answer and its signature still has
# the fingerprint it was baked with.

const BAKE_CAPABILITIES = JObj("instruct" => true)
const BAKED_FORMAT = 2
const EXAMPLES_FORMAT = 1

"""
    BakeError

A bake, or a call on a baked model, refused (contract/baked.md): `code` is
`baked-fixed` (a call gives a fixed input another value), `baked-derived` (a
derived input the student never learned for its source), `baked-changed`
(the function changed since it was baked), `baked-format` (a folder of a
format this reader does not know), or `bake-rows` (rows that cannot be
examples).
"""
struct BakeError <: FunctAIError
    code::String
    msg::String
end
Base.showerror(io::IO, e::BakeError) = print(io, "BakeError [", e.code, "]: ", e.msg)

"`\"sha256:\"` of a value's canonical JSON (calls.md, \"Canonical JSON\"): how fixed and derived inputs are kept."
value_hash(v) = LMCC.sha256_of(logvalue(v))

"""
One function a generative student answers: the contract between training and
calling. Training writes every example through it; a call on the baked model
lays out its request through the same entry, read back from `baked.json`.
"""
struct BakeEntry
    name::String
    fingerprint::String                # lmcc's, of the function's full signature
    signature::LMCC.Signature          # what the student reads: the full one, minus fixed and derived inputs
    layout::JObj                       # the lmcc adapter artifact, replies written from values
    outputs::Vector{String}
    reasoning::Bool
    fixed::OrderedDict{String,String}  # input => value hash
    derived::OrderedDict{String,Any}   # input => {"from", "values"}
    capabilities::JObj
    fn::Any
end

left_out(e::BakeEntry) = vcat(collect(keys(e.fixed)), collect(keys(e.derived)))

function reduced_signature(sig::LMCC.Signature, leave_out)
    isempty(leave_out) && return sig
    d = LMCC.signature_to_dict(sig)
    d["fields"] = Any[x for x in d["fields"] if !(x["direction"] == "input" && x["name"] in leave_out)]
    LMCC.signature_from_dict(d)
end

"The layout a student is trained and called with: the given one, else the function's own; with replies written from values."
function student_layout(f::AIFunction, s, layout=nothing)
    adapter = if layout !== nothing
        layout isa AbstractVector ? template_adapter(layout) : resolve_adapter(layout)
    elseif setting(s, :template) !== nothing
        template_adapter(setting(s, :template))
    else
        resolve_adapter(something(setting(s, :adapter), "xml"))
    end
    art = LMCC.dump(adapter; registry=registry())
    art["replay"] = "values"
    art
end

"""
    FunctAI.bake_entry(f; layout, reasoning = false, fixed = Dict(), derived = Dict(), rows = []) -> BakeEntry

What a student of `f` reads (contract/baked.md, "The examples"): `f`'s
signature without the inputs left out (`fixed`: one value in every row, its
hash kept; `derived`: decided by another input, `Dict("guidance" =>
"section")`, a table of hashes kept), without `reasoning` unless the bake
keeps it, laid out in `f`'s layout with replies written from values, and no
worked examples. `rows` are checked against `fixed` and `derived`.
"""
function bake_entry(f::AIFunction; layout=nothing, reasoning::Bool=false, fixed=Dict(), derived=Dict(), rows=())
    isempty(f.tools) || throw(BakeError("bake-rows", "$(f.definition.name) uses tools; training tool-calling students is not supported yet"))
    s = effective(f.own)
    cot = reasoning && s[:reasoning] === true
    full = signature_of(f.definition, f.instructions, s[:include_name], cot, false)
    names = [x.name for x in full.fields if x.direction == "input" && x.purpose == "plain"]
    fixed = OrderedDict{String,Any}(String(k) => v for (k, v) in pairs(fixed))
    derived = OrderedDict{String,String}(String(k) => String(v) for (k, v) in pairs(derived))
    for n in vcat(collect(keys(fixed)), collect(keys(derived)))
        n in names || throw(BakeError("bake-rows", "$(f.definition.name) has no input $(repr(n)) to leave out (its inputs: $(join(names, ", ")))"))
    end
    both = intersect(keys(fixed), keys(derived))
    isempty(both) || throw(BakeError("bake-rows", "$(join(sort!(collect(both)), ", ")) cannot be both fixed and derived"))
    for (n, src) in derived
        (src in names && !haskey(fixed, src) && !haskey(derived, src)) ||
            throw(BakeError("bake-rows", "derived = Dict($(repr(n)) => $(repr(src))): $(repr(src)) must be another input the student still reads"))
    end
    prepared(vals) = prepare_inputs(full, JObj(String(k) => v for (k, v) in vals))
    fixed_hashes = OrderedDict{String,String}(n => value_hash(prepared(Dict(n => v))[n]) for (n, v) in fixed)
    rows = [row_dict(r) for r in rows]
    for (i, row) in enumerate(rows), (n, h) in fixed_hashes
        haskey(row, n) && value_hash(prepared(Dict(n => row[n]))[n]) != h &&
            throw(BakeError("bake-rows", "fixed = Dict($(repr(n)) => …): row $i gives $n another value; a fixed input has one value in every row"))
    end
    table = OrderedDict{String,Any}()
    for (n, src) in derived
        values = OrderedDict{String,String}()
        for (i, row) in enumerate(rows)
            (haskey(row, n) && haskey(row, src)) || throw(BakeError("bake-rows", "derived = Dict($(repr(n)) => $(repr(src))): row $i lacks $n or $src"))
            p = prepared(Dict(n => row[n], src => row[src]))
            k, v = value_hash(p[src]), value_hash(p[n])
            get!(values, k, v) == v || throw(BakeError("bake-rows", "derived = Dict($(repr(n)) => $(repr(src))): two rows with the same $src " *
                                                                    "give $n different values, so $src does not decide it. Keep $n as an input"))
        end
        table[n] = LMCC.jobj("from" => src, "values" => JObj(values))
    end
    sig = reduced_signature(full, vcat(collect(keys(fixed)), collect(keys(derived))))
    BakeEntry(f.definition.name, LMCC.signature_fingerprint(full), sig, student_layout(f, s, layout),
              String[x.name for x in full.fields if x.direction == "output" && x.purpose != "tools.calls"], cot, fixed_hashes, table,
              copy(BAKE_CAPABILITIES), f)
end

student_plan(e::BakeEntry) = LMCC.bind(LMCC.load(LMCC.deepcopy_json(e.layout); registry=registry()), e.signature;
                                       capabilities=e.capabilities, registry=registry())

"""
An lm15 request (canonical JSON) as chat-template messages: the system text
first, a developer message as a system one, each message's text parts joined.
"""
function chat_messages(request::AbstractDict)
    out = Any[]
    system = get(request, "system", nothing)
    if system !== nothing && !isempty(system)
        push!(out, LMCC.jobj("role" => "system", "content" => system isa AbstractString ? String(system) : join(p["text"] for p in system)))
    end
    for m in request["messages"]
        role = m["role"]
        role in ("user", "assistant", "system", "developer") ||
            throw(BakeError("bake-rows", "a generative student reads text chats; the request has a $(repr(role)) message"))
        texts = String[]
        for p in m["parts"]
            get(p, "type", nothing) == "text" || throw(BakeError("bake-rows", "a generative student reads text; the request carries a $(get(p, "type", "?")) part"))
            push!(texts, p["text"])
        end
        push!(out, LMCC.jobj("role" => role == "developer" ? "system" : role, "content" => join(texts)))
    end
    out
end

"""
The chat messages of one row's call as the student sees it, and the reply the
layout writes for its outputs (`nothing` without outputs): a fresh render of
the call, with no earlier turns, and the assistant message of the same call
rendered with that example as its one earlier turn.
"""
function student_messages(e::BakeEntry, inputs::AbstractDict, outputs=nothing)
    plan = student_plan(e)
    values = prepare_inputs(e.signature, JObj(String(k) => v for (k, v) in inputs if !(String(k) in left_out(e))))
    msgs = chat_messages(LMCC.request(LMCC.render(plan, LMCC.new_turn(plan, values)), "student"))
    outputs === nothing && return (msgs, nothing)
    example = LMCC.example(plan, values, JObj(k => logvalue(outputs[k]) for k in e.outputs if haskey(outputs, k)))
    both = chat_messages(LMCC.request(LMCC.render(plan, LMCC.new_turn(plan, values); turns=[example]), "student"))
    (msgs, last(m for m in both if m["role"] == "assistant")["content"])
end

"An entry as `baked.json` keeps it."
entry_meta(e::BakeEntry) = LMCC.jobj("name" => e.name, "fingerprint" => e.fingerprint, "signature" => LMCC.signature_to_dict(e.signature),
                                     "layout" => e.layout, "outputs" => Any[e.outputs...], "reasoning" => e.reasoning,
                                     "fixed" => JObj(e.fixed), "derived" => JObj(e.derived), "capabilities" => e.capabilities)
entry_from_meta(d::AbstractDict, fn=nothing) =
    BakeEntry(String(d["name"]), String(d["fingerprint"]), LMCC.signature_from_dict(d["signature"]), JObj(d["layout"]),
              String[something(get(d, "outputs", nothing), Any[])...], get(d, "reasoning", false) === true,
              OrderedDict{String,String}(something(get(d, "fixed", nothing), JObj())),
              OrderedDict{String,Any}(something(get(d, "derived", nothing), JObj())), JObj(something(get(d, "capabilities", nothing), BAKE_CAPABILITIES)), fn)

"""
    FunctAI.bake_examples(f, rows; fixed, derived, reasoning, layout, validation = 0.1, seed = 0, weight = nothing, tag = nothing)

The training conversations of `f` on rows with known answers (contract/baked.md,
"The examples table"): one row per conversation, `function`, `messages` (the
prompt, then `{"role": "assistant", "content": <reply>}`: the loss belongs on
the reply only), `tag`, `weight`, `row_id`, `split` (`"train"` or
`"validation"`, a seeded share of `validation`) and `source` (`"data"`). A
Tables.jl table (a vector of NamedTuples). Turning messages into tokens is
the student's own chat template's rule, so any trainer can train on them, and
a model trained on them is called by [`baked`](@ref) with the same messages.
`weight` and `tag` name columns of the rows.
"""
function bake_examples(f::AIFunction, rows; fixed=Dict(), derived=Dict(), reasoning::Bool=false, layout=nothing,
                       validation::Real=0.1, seed::Integer=0, weight=nothing, tag=nothing)
    data = [row_dict(r) for r in rows_of(rows)]
    e = bake_entry(f; layout, reasoning, fixed, derived, rows=data)
    n = length(data)
    held = Set(shuffle(Xoshiro(seed), collect(1:n))[1:(n > 1 ? min(n - 1, round(Int, n * validation)) : 0)])
    out = NamedTuple[]
    for (i, row) in enumerate(data)
        inputs = JObj(x.name => row[x.name] for x in f.definition.inputs if haskey(row, x.name) && row[x.name] !== missing)
        for (k, v) in fixed
            haskey(inputs, String(k)) || (inputs[String(k)] = v)
        end
        outs = JObj(k => row[k] for k in e.outputs if haskey(row, k) && row[k] !== missing)
        isempty(outs) && throw(BakeError("bake-rows", "row $i has no answer for $(join(e.outputs, ", ")): a student learns from rows with known answers"))
        msgs, reply = student_messages(e, inputs, outs)
        push!(out, (var"function"=e.name, messages=Any[msgs..., LMCC.jobj("role" => "assistant", "content" => reply)],
                    tag=tag === nothing ? missing : something(get(row, String(tag), missing), missing),
                    weight=weight === nothing ? 1.0 : Float64(row[String(weight)]), row_id=i - 1,
                    split=i in held ? "validation" : "train", source="data"))
    end
    out
end

"""
    FunctAI.export_examples(path, f, rows; options...) -> path

[`bake_examples`](@ref) written as JSON lines (`function`, `messages`, `tag`,
`weight`, `row_id`, `split`, `source`), with `<path>.meta.json` beside it
(`{"functai_examples": 1, "student": null, "template": null, "functions":
[<entry>]}`, entries as in `baked.json`): what made them, so a model trained
on them is adopted by checking its function's fingerprint. Any trainer reads
the file (TRL's `SFTTrainer` with `assistant_only_loss`, Axolotl's chat
template datasets, a service's upload).
"""
function export_examples(path::AbstractString, f::AIFunction, rows; fixed=Dict(), derived=Dict(), reasoning::Bool=false, layout=nothing, kw...)
    table = bake_examples(f, rows; fixed, derived, reasoning, layout, kw...)
    e = bake_entry(f; layout, reasoning, fixed, derived, rows=[row_dict(r) for r in rows_of(rows)])
    mkpath(dirname(abspath(path)))
    open(path, "w") do io
        for r in table
            println(io, LMCC.json_text(LMCC.jobj("function" => r.function, "messages" => r.messages, "tag" => r.tag === missing ? nothing : r.tag,
                                                 "weight" => r.weight, "row_id" => r.row_id, "split" => r.split, "source" => r.source)))
        end
    end
    meta = LMCC.jobj("functai_examples" => EXAMPLES_FORMAT, "student" => nothing, "template" => nothing, "functions" => Any[entry_meta(e)])
    write(path * ".meta.json", LMCC.json_text(meta))
    path
end

# ------------------------------------------------------------------ a baked model, used

"""
A baked generative model, served by an OpenAI-compatible server (vLLM, SGLang,
TGI, llama.cpp) that serves its folder: what [`baked`](@ref) returns, and what
`configure(f; lm = student)` runs `f` on.
"""
struct BakedModel
    folder::String
    meta::JObj
    entries::OrderedDict{String,BakeEntry}
    url::String
    model::String
    api_key::Union{Nothing,String}
    timeout::Float64
end
Base.show(io::IO, b::BakedModel) = print(io, "BakedModel(", repr(get(b.meta, "name", basename(b.folder))), " at ", b.url, ": ",
                                         join(keys(b.entries), ", "), ")")

"""
    FunctAI.baked(folder; url, model = nothing, api_key = nothing, timeout = 120) -> BakedModel

A model baked anywhere (Python's `functai.bake`, or a trainer that wrote
`baked.json` format 2), its weights served by an OpenAI-compatible server
at `url` (`vllm serve <folder>/model` serves `/v1/chat/completions`).
`configure(f; lm = student)` runs `f` on it, laid out exactly as it was
trained: the student's signature and layout, no worked examples, the chat
template the server applies with thinking off. A call is refused when `f`
changed since it was baked (`baked-changed`), or gives a fixed input another
value (`baked-fixed`) or a derived one a pair the student never saw
(`baked-derived`).
"""
function baked(folder::AbstractString; url::AbstractString, model=nothing, api_key=nothing, timeout::Real=120.0)
    path = joinpath(folder, "baked.json")
    isfile(path) || throw(BakeError("baked-format", "$folder has no baked.json"))
    meta = LMCC.parse_json(read(path, String))
    get(meta, "functai_baked", nothing) == BAKED_FORMAT ||
        throw(BakeError("baked-format", "$folder is baked.json format $(repr(get(meta, "functai_baked", nothing))); this reader reads format " *
                                        "$BAKED_FORMAT (a format 1 folder: bake it again)"))
    get(meta, "kind", nothing) == "generative" ||
        throw(BakeError("baked-format", "$folder is a $(get(meta, "kind", "?")) model; Julia runs generative students (a head runs in Python)"))
    entries = OrderedDict{String,BakeEntry}(String(d["name"]) => entry_from_meta(d) for d in meta["functions"])
    name = something(model, get(meta, "name", nothing), basename(abspath(folder)))
    BakedModel(abspath(folder), JObj(meta), entries, rstrip(String(url), '/'), String(name), api_key === nothing ? nothing : String(api_key),
               Float64(timeout))
end

"The entry a function runs through: by name, or (a model of one function) its only one; refused when the function changed."
function entry_for(b::BakedModel, f::AIFunction)
    e = get(b.entries, f.definition.name, length(b.entries) == 1 ? first(values(b.entries)) : nothing)
    e === nothing && throw(BakeError("baked-changed", "this baked model answers $(join(keys(b.entries), ", ")), not $(f.definition.name)"))
    s = effective(f.own)
    full = signature_of(f.definition, f.instructions, s[:include_name], e.reasoning && s[:reasoning] === true, false)
    LMCC.signature_fingerprint(full) == e.fingerprint ||
        throw(BakeError("baked-changed", "$(f.definition.name) has changed since it was baked (its inputs, outputs, types or instruction); bake it again"))
    e
end

"The inputs the student reads, after checking fixed inputs have their baked values and derived ones follow their source."
function student_inputs(e::BakeEntry, sig::LMCC.Signature, inputs::AbstractDict)
    isempty(left_out(e)) && return inputs
    p = prepare_inputs(sig, JObj(String(k) => (v isa LeftOut ? jsonvalue(v) : v) for (k, v) in inputs))
    for (n, h) in e.fixed
        haskey(p, n) && value_hash(p[n]) != h &&
            throw(BakeError("baked-fixed", "$(e.name): this baked model was trained with $n fixed to one value, and this call gives another. " *
                                           "The student never learned to read $n: call it with the baked value, or bake again with this one"))
    end
    for (n, d) in e.derived
        src = d["from"]
        haskey(p, src) || continue
        want = get(d["values"], value_hash(p[src]), nothing)
        want === nothing && throw(BakeError("baked-derived", "$(e.name): this baked model never saw this $src in training, so it does not know the $n that goes with it"))
        haskey(p, n) && value_hash(p[n]) != want &&
            throw(BakeError("baked-derived", "$(e.name): $n is not the one the baked model learned for this $src; bake again with this pair"))
    end
    OrderedDict{String,Any}(k => v for (k, v) in inputs if !(String(k) in left_out(e)))
end

"Calls an OpenAI-compatible chat server for a baked student: the request's messages as chat messages, thinking off."
struct BakedRouter
    model::BakedModel
end
route(::BakedRouter, model::AbstractString) = ("functai-baked-lm", String(model))
function LM15.complete(r::BakedRouter, request::LM15.Request)
    b = r.model
    gen = something(get(b.meta, "generation", nothing), JObj())
    body = LMCC.jobj("model" => b.model, "messages" => chat_messages(LM15.to_dict(request)), "temperature" => 0,
                     "chat_template_kwargs" => LMCC.jobj("enable_thinking" => false))
    max_new = something(request.config === nothing ? nothing : request.config.max_tokens, get(gen, "max_new_tokens", nothing), Some(nothing))
    max_new === nothing || (body["max_tokens"] = max_new)
    headers = Pair{String,String}["content-type" => "application/json"]
    b.api_key === nothing || push!(headers, "authorization" => "Bearer $(b.api_key)")
    resp = HTTP.post(b.url * "/chat/completions", headers, LMCC.json_text(body); status_exception=false, readtimeout=round(Int, b.timeout))
    resp.status >= 400 && throw(ErrorException("the baked model's server at $(b.url) answered $(resp.status): $(first(String(resp.body), 300))"))
    d = LMCC.parse_json(String(resp.body))
    choice = d["choices"][1]
    text = something(get(choice["message"], "content", nothing), "")
    u = something(get(d, "usage", nothing), JObj())
    LM15.Response(; model=String(something(get(d, "model", nothing), b.model)),
                  message=LM15.Message(; role="assistant", parts=(LM15.TextPart(; text=String(text)),)),
                  finish_reason=get(choice, "finish_reason", "stop") == "length" ? "length" : "stop",
                  usage=LM15.Usage(; input_tokens=get(u, "prompt_tokens", nothing), output_tokens=get(u, "completion_tokens", nothing)))
end
