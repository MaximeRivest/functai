# Text rules the contract spells out (contract/scores.md): which characters
# are white space, and Unicode full case folding. Julia's own `isspace`,
# `strip` and `lowercase` differ from both (isspace adds U+0085's neighbours
# differently and misses U+001C to U+001F), so they are not used here.

const WHITE = Set{Char}(Char.([
    0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x1c, 0x1d, 0x1e, 0x1f, 0x20, 0x85, 0xa0, 0x1680,
    0x2000, 0x2001, 0x2002, 0x2003, 0x2004, 0x2005, 0x2006, 0x2007, 0x2008, 0x2009, 0x200a,
    0x2028, 0x2029, 0x202f, 0x205f, 0x3000]))

iswhite(c::Char) = c in WHITE

"The pieces of `text` between runs of white space."
split_white(text::AbstractString) = split(text, iswhite; keepempty=false)

"`text` without white space at either end (what Python's `str.strip()` removes)."
trim_white(text::AbstractString) = String(strip(iswhite, text))

"Line breaks: what Python's `str.splitlines()` splits on."
const BREAKS = Set{Char}(Char.([0x0a, 0x0b, 0x0c, 0x0d, 0x1c, 0x1d, 0x1e, 0x85, 0x2028, 0x2029]))

"""
A choice's meaning as the contract writes it (functions.md, "A choice's
answers, described one by one"): trimmed, and each run of white space that
holds a line break made one space.
"""
function one_line(text::AbstractString)
    io = IOBuffer()
    run = IOBuffer()
    broke = false
    for c in text
        if iswhite(c)
            print(run, c)
            broke |= c in BREAKS
            continue
        end
        r = String(take!(run))
        print(io, broke ? " " : r, c)
        broke = false
    end
    trim_white(String(take!(io)))
end

"""
A field's words with what each described answer of its choice means: the
words (or `one of:`), then `  - <answer>: <meaning>` per answer, joined by
`\n` (functions.md). No described answer: the words as they are.
"""
function with_choices(words, meanings)
    isempty(meanings) && return words
    join([words === nothing || isempty(words) ? "one of:" : words; ["  - $a: $m" for (a, m) in meanings]], "\n")
end

"""
    casefold(text)

Unicode full case folding, code point by code point (`CaseFolding.txt`,
statuses C and F; Python's `str.casefold`), from the contract's table.

# Examples
```jldoctest
julia> casefold("Straße")
"strasse"
```

See also [`normalize_text`](@ref).
"""
function casefold(text::AbstractString)
    io = IOBuffer()
    for c in text
        folded = get(CASEFOLD, c, nothing)
        folded === nothing ? print(io, c) : print(io, folded)
    end
    String(take!(io))
end

"""
    normalize_text(text)

Text as `exact_match` compares it: white space collapsed to one space and
trimmed, then case folded: what [`exact_match`](@ref) compares.

# Examples
```jldoctest
julia> normalize_text("  New\u00a0  York ")
"new york"
```
"""
normalize_text(text::AbstractString) = casefold(join(split_white(text), ' '))
