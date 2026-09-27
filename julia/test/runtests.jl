using FunctAI
using Test
import LMCC, LM15

include("fake.jl")

# no test writes to the real call log folder
ENV["FUNCTAI_LOG_CALLS"] = "0"
delete!(ENV, "FUNCTAI_CALLER")

@testset "FunctAI" begin
    include("contract.jl")
    include("functions.jl")
    include("calls.jl")
    include("models.jl")
end
