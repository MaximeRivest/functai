using FunctAI
using Test
import LMCC, LM15
using Random: Xoshiro, shuffle

include("fake.jl")

# no test writes to the real call log folder
ENV["FUNCTAI_LOG_CALLS"] = "0"
delete!(ENV, "FUNCTAI_CALLER")

@testset "FunctAI" begin
    include("contract.jl")
    include("events.jl")
    include("functions.jl")
    include("calls.jl")
    include("foundations.jl")
    include("guarantees.jl")
    include("gepa.jl")
    include("models.jl")
end

# the examples in the docstrings (jldoctest), as the manual shows them
using Documenter
DocMeta.setdocmeta!(FunctAI, :DocTestSetup, :(using FunctAI); recursive=true)
doctest(FunctAI; manual=false)
