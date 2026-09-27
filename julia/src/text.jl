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

"""
    casefold(text)

Unicode full case folding, code point by code point (`CaseFolding.txt`,
statuses C and F; Python's `str.casefold`), from the contract's table:
`casefold("Straße") == "strasse"`.
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
trimmed, then case folded. `normalize_text("  New   York ") == "new york"`.
"""
normalize_text(text::AbstractString) = casefold(join(split_white(text), ' '))
