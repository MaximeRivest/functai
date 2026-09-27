# Small labelled datasets to learn FunctAI with: tables of text with the
# right answers filled in, so every example in the documentation runs as is
# and "how often is it right?" has an answer from the first minute. The same
# tables as Python's and R's (data/datasets/*.csv, copied from
# python/functai/datasets). Each is a Tables.jl column table (a NamedTuple of
# vectors): `DataFrame(FunctAI.tickets())` makes it a data frame.

const DATASETS = normpath(joinpath(@__DIR__, "..", "data", "datasets"))

"The fields of one CSV line (RFC 4180: quoted fields may hold commas, quotes and line breaks)."
function csv_records(text::AbstractString)
    records = Vector{Vector{Union{Nothing,String}}}()
    row = Union{Nothing,String}[]
    field = IOBuffer()
    quoted = false      # inside quotes
    was_quoted = false  # this field was quoted: "" is text, not missing
    i = firstindex(text)
    while i <= lastindex(text)
        c = text[i]
        if quoted
            if c == '"'
                j = nextind(text, i)
                if j <= lastindex(text) && text[j] == '"'
                    write(field, '"')
                    i = j
                else
                    quoted = false
                end
            else
                write(field, c)
            end
        elseif c == '"'
            quoted, was_quoted = true, true
        elseif c == ','
            s = String(take!(field))
            push!(row, isempty(s) && !was_quoted ? nothing : s)
            was_quoted = false
        elseif c == '\n' || c == '\r'
            if c == '\r' && nextind(text, i) <= lastindex(text) && text[nextind(text, i)] == '\n'
                i = nextind(text, i)
            end
            s = String(take!(field))
            push!(row, isempty(s) && !was_quoted ? nothing : s)
            (length(row) > 1 || row[1] !== nothing) && push!(records, row)
            row = Union{Nothing,String}[]
            was_quoted = false
        else
            write(field, c)
        end
        i = nextind(text, i)
    end
    s = String(take!(field))
    if !isempty(s) || was_quoted || !isempty(row)
        push!(row, isempty(s) && !was_quoted ? nothing : s)
        push!(records, row)
    end
    records
end

"A column read as its values say: integers, numbers, true/false, or text; an empty field is `missing`."
function typed_column(values::Vector{Union{Nothing,String}})
    present = [v for v in values if v !== nothing]
    T, parse_one = if !isempty(present) && all(v -> occursin(r"^-?\d+$", v), present)
        Int, v -> parse(Int, v)
    elseif !isempty(present) && all(v -> tryparse(Float64, v) !== nothing, present)
        Float64, v -> parse(Float64, v)
    elseif !isempty(present) && all(v -> v in ("TRUE", "FALSE", "true", "false"), present)
        Bool, v -> lowercase(v) == "true"
    else
        String, identity
    end
    out = [v === nothing ? missing : parse_one(v) for v in values]
    any(ismissing, out) ? Vector{Union{Missing,T}}(out) : Vector{T}(out)
end

function read_dataset(name::AbstractString)
    records = csv_records(read(joinpath(DATASETS, "$name.csv"), String))
    header = Symbol.(first(records))
    rows = records[2:end]
    NamedTuple{Tuple(header)}(Tuple(typed_column(Union{Nothing,String}[r[j] for r in rows]) for j in eachindex(header)))
end

"""
    FunctAI.tickets()

Customer support messages to a small homeware shop, with the team each
belongs to: 80 rows, as a Tables.jl column table (`DataFrame(FunctAI.tickets())`).

| column | what it holds |
|:--|:--|
| `id` | the message's number |
| `message` | what the customer wrote |
| `channel` | `"email"` or `"chat"` |
| `category` | the team: `"shipping"`, `"billing"`, `"product"` or `"account"` |
| `order_id` | the order number, like `"A-1042"` (a letter, a dash, four digits); `missing` when there is none |

The house rules, which the labels follow:

- Anything wrong with the delivery itself (late, lost, sent to the wrong
  place, the wrong item, something missing, or **broken when it arrived**)
  is **shipping**: the carrier pays.
- Anything about money (charges, invoices, coupons, cards, and **every
  request for money back**, whatever the reason) is **billing**.
- Problems that appear while using a product, and questions about
  products, are **product**.
- Signing in, passwords, profile details, personal data and emails from the
  shop are **account**.

# Examples
```jldoctest
julia> t = FunctAI.tickets();

julia> length(t.message), t.category[1]
(80, "shipping")
```

See also [`FunctAI.field_notes`](@ref), [`FunctAI.refunds`](@ref).
"""
tickets() = read_dataset("tickets")

"""
    FunctAI.field_notes()

Bird survey notes written by volunteers, with the species, count and
behaviour of each: 60 rows, one species per note, from four sites between
April and May 2026, as a Tables.jl column table.

| column | what it holds |
|:--|:--|
| `id` | the note's number |
| `site` | `"Marsh boardwalk"`, `"North field"`, `"Creek trail"` or `"Old orchard"` |
| `date` | `YYYY-MM-DD` |
| `note` | what the volunteer wrote |
| `species` | the checklist's common name (American robin, black-capped chickadee, blue jay, northern cardinal, mallard, Canada goose, great blue heron, red-tailed hawk, downy woodpecker, song sparrow, American crow, barn swallow), or `"other"` |
| `count` | how many birds; `missing` when the note gives no number |
| `behaviour` | `"feeding"`, `"nesting"`, `"flying"`, `"resting"` or `"calling"` |

The protocol, which the labels follow:

- **Species**: the checklist's name. Nicknames count ("robin", "heron",
  "red-tail", "downy"); a species not on the checklist is `"other"`.
- **Count**: every bird seen or heard, young included. One bird named
  without a number ("a blue jay") is 1; "a pair" or "a couple" is 2; an
  approximate number ("about 40", "~25", "maybe 6") is that number; no
  number ("a few", "several", "a flock", "lots") is missing: **never guess**.
- **Behaviour**: singing, calling and drumming are *calling*; building,
  sitting on a nest or bringing food to young are *nesting*; perched,
  swimming, roosting or standing still are *resting*.

# Examples
```jldoctest
julia> n = FunctAI.field_notes();

julia> length(n.note), count(ismissing, n.count) > 0
(60, true)
```
"""
field_notes() = read_dataset("field_notes")

"""
    FunctAI.refunds()

Refund requests to the homeware shop of [`FunctAI.tickets`](@ref), with the
decision its rules give: 120 rows, as a Tables.jl column table.

| column | what it holds |
|:--|:--|
| `id` | the request's number |
| `message` | what the customer wrote |
| `item` | what they bought |
| `price` | what they paid, in dollars |
| `days_since_delivery` | from the order system, not the message |
| `final_sale` | bought on final sale (clearance) |
| `state` | `"unopened"`, `"opened_unused"`, `"used"` (works, no longer wanted), `"damaged"` (on arrival), `"wrong_item"` or `"faulty"` (failed in normal use) |
| `decision` | `"approve"` or `"deny"`: what the rules below give |

The refund rules, which `decision` follows exactly:

- Damaged on arrival, or the wrong item (or part of the order missing): a
  refund within **60 days** of delivery, final sale or not.
- Faulty (it failed in normal use): a refund within **365 days**, final
  sale or not.
- Unopened, or opened but not used, and no longer wanted: a refund within
  **30 days**, and **never for a final-sale item**.
- Used and no longer wanted: **no refund**.

The messages were written by a language model from each row's facts, in
varied tones and lengths; some say what happened only indirectly. The
facts, and so the decisions, were drawn first.

# Examples
```jldoctest
julia> r = FunctAI.refunds();

julia> length(r.message), eltype(r.price), eltype(r.final_sale)
(120, Float64, Bool)
```
"""
refunds() = read_dataset("refunds")
