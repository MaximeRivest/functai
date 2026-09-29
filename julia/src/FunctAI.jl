"""
    FunctAI

Typed Julia functions whose body a language model writes. Write the
signature, a model writes the body; run it over a column, measure how often
it is right, have people rate its calls, improve it, and save it for any
FunctAI language to load.

```julia
using FunctAI

@enum Mood happy unhappy mixed

@ai function mood(review::String)::Mood
    "How does the customer feel about what they bought?"
end

mood("Broke after a day.")          # unhappy
df.mood = mood.(df.review)          # the column, 8 calls at a time
evaluate(mood, labelled)            # how often it is right, with a 95% range
```

It follows the FunctAI contract, so a function has the same version, the
same call log and the same saved form as the same function in Python,
TypeScript and R. It stands on lmcc (how values are written into a prompt
and read back) and lm15 (every provider, one wire).
"""
module FunctAI

import Dates
using Dates: @dateformat_str
using OrderedCollections: OrderedDict
using PrecompileTools: @setup_workload, @compile_workload
using Printf: @sprintf
using Random: Xoshiro, shuffle, shuffle!
using ScopedValues: ScopedValue, with
import LMCC
import LM15
import LM15: stream, text, configure
import LMCC: render, tool
import MLJModelInterface as MMI
import StatsAPI
import StatsAPI: predict, fit
import Tables

const FUNCTAI_VERSION = pkgversion(@__MODULE__) === nothing ? v"0.1.0" : pkgversion(@__MODULE__)

include("contract.jl")
include("text.jl")
include("values.jl")
include("settings.jl")
include("interface.jl")
include("schema.jl")
include("content.jl")
include("events.jl")
include("journal.jl")
include("models.jl")
include("layouts.jl")
include("definition.jl")
include("calllog.jl")
include("saw.jl")
include("tools.jl")
include("engine.jl")
include("run.jl")
include("fn.jl")
include("macro.jl")
include("program.jl")
include("stream.jl")
include("evaluate.jl")
include("optimize.jl")
include("gepa.jl")
include("model.jl")
include("saved.jl")
include("api.jl")
include("accounts.jl")
include("datasets.jl")
include("precompile.jl")

export @ai, @program, AIFunction, AIProgram, OneOf, Prediction
export predict, render, stream, eachevent, version, signature_id, instructions, demos, with_demos, with_instructions
export configure, configure!, with_settings, settings, problems
export evaluate, Evaluation, exact_match, score_interval, compare
export labeled_few_shot, bootstrap_few_shot, random_search, instruction_search, gepa
export AIModel, AIModelFit, fit
export rate, calls, rated
export tool, AITool, Event, AIStream, StepLimit, Cancelled, LoadRefused
export InterfaceError, LogContentError, JournalError
export model_capabilities, casefold, normalize_text

# observers and journal writers work off the calls' tasks: give them a moment when Julia exits
__init__() = atexit(() -> (drain(2.0); EXITING[] = true; nothing))

end
