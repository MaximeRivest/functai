# The contract's data this package carries (data/contract, copied from the
# repository's contract/ by julia/check): the three layouts, the model table
# and Unicode case folding. test/runtests.jl checks the copies are equal to
# the contract's. Read once, when the package is compiled.

const CONTRACT_DATA = normpath(joinpath(@__DIR__, "..", "data", "contract"))

function contract_json(path::AbstractString)
    file = joinpath(CONTRACT_DATA, path)
    include_dependency(file)
    LMCC.parse_json(read(file, String))
end

const LAYOUT_DATA = Dict(name => contract_json("layouts/$name.json") for name in ("xml", "chat", "json"))
const MODELS = contract_json("models.json")
const CASEFOLD = Dict{Char,String}(Char(parse(UInt32, k; base=16)) => String(v)
                                   for (k, v) in contract_json("unicode/casefold.json")["folds"])
