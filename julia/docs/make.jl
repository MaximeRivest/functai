# The FunctAI.jl manual, built with Documenter:
#
#     julia --project=julia/docs julia/docs/make.jl
#
# Home, the eight tutorials (docs/julia/*.md at the repository's root: the
# same pages as the website's Julia section, with the outputs of their last
# real run, written by julia/tutorials), how-to guides whose examples run on
# every build (they need no model and cost nothing), and the reference,
# generated from the docstrings. Doctests run here and in Pkg.test().
# Published at https://maximerivest.github.io/functai/julia/manual/ by
# .github/workflows/docs.yml, beside the website.
# Nothing here calls a model.

using Documenter, FunctAI

const HERE = @__DIR__
const TUTORIALS = normpath(joinpath(HERE, "..", "..", "docs", "julia"))
const WORK = joinpath(HERE, ".work")

"Dollars in prose (\$40) escaped for Documenter, which reads a bare one as interpolation; code is left as it is."
function escape_dollars(text)
    out = String[]
    fenced = false
    for line in split(text, '\n')
        if startswith(line, "```")
            fenced = !fenced
        elseif !fenced
            # outside inline code spans (odd-numbered pieces between backticks)
            pieces = split(line, '`')
            line = join((isodd(i) ? replace(p, "\$" => "\\\$") : p for (i, p) in enumerate(pieces)), '`')
        end
        push!(out, line)
    end
    join(out, '\n')
end

# The tutorials, copied in: their shown-only cells become plain Julia, their
# outputs plain text, and their links point at each other's pages.
rm(WORK; recursive=true, force=true)
cp(joinpath(HERE, "src"), WORK)
mkpath(joinpath(WORK, "tutorials"))
cp(joinpath(TUTORIALS, "figures"), joinpath(WORK, "tutorials", "figures"))
tutorials = sort([f for f in readdir(TUTORIALS) if endswith(f, ".md") && f != "news.md"])   # news.md: the website's Julia news
for f in tutorials
    text = read(joinpath(TUTORIALS, f), String)
    text = replace(text, "```{.julia .no-run}" => "```julia", "```output" => "```text")
    text = escape_dollars(text)
    text = replace(text, "](../tutorials/index.md)" => "](https://maximerivest.github.io/functai/tutorials/index.html)",
                   "](../r/index.md)" => "](https://maximerivest.github.io/functai/r/index.html)")
    write(joinpath(WORK, "tutorials", f), text)
end

DocMeta.setdocmeta!(FunctAI, :DocTestSetup, :(using FunctAI); recursive=true)

makedocs(;
    root=HERE, source=".work", build="build",
    sitename="FunctAI.jl",
    modules=[FunctAI],
    authors="Maxime Rivest",
    format=Documenter.HTML(; prettyurls=get(ENV, "CI", "false") == "true", edit_link=nothing,
                           canonical="https://maximerivest.github.io/functai/julia/manual/",
                           repolink="https://github.com/MaximeRivest/functai", size_threshold=400_000),
    remotes=nothing,
    checkdocs=:exports,
    warnonly=[:cross_references],
    pages=[
        "Home" => "index.md",
        "Tutorials" => ["tutorials/index.md", ("tutorials/" .* filter(!=("index.md"), tutorials))...],
        "Guides" => [
            "guides/functions.md",
            "guides/tables.md",
            "guides/settings.md",
            "guides/measuring.md",
            "guides/calls.md",
            "guides/log.md",
            "guides/saving.md",
            "guides/conversations.md",
            "guides/approvals.md",
            "guides/plugins.md",
            "guides/serving.md",
            "guides/long-runs.md",
            "guides/baking.md",
        ],
        "Where Julia differs" => "differences.md",
        "Reference" => ["reference.md", "reference-logs.md"],
    ],
)
