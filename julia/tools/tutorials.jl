# Run Julia tutorials (Markdown with ```julia cells) top to bottom, each in a
# fresh Julia process, and write every cell's output under it, as the R
# pages (r/tools/tutorials.R) and the Python pages carry theirs:
#
#   ```julia                a cell: run, in Main, as the REPL runs it
#   ```output               what it printed, then its value as the REPL shows
#                           it (not when the cell ends with `;`): rewritten on
#                           every run
#   ![](figures/...)        a figure it returned (a Makie figure): rewritten
#   ```{.julia .no-run}     shown, never run (an install line, a sign-in)
#
#   julia/tutorials docs/julia/01-first-function.md ...   (the wrapper finds Julia and the keys)
#
# A cell that fails stops the run: the page is written up to that cell, with
# the error under it, and the exit status is 1. Tutorials call real models, so
# a run costs money; each tutorial counts its own cost in its last section.

using Logging

const WIDTH = 92          # the width outputs are shown at (the site's code blocks)
const ROWS = 24           # the height: a DataFrame shows this many lines, as in a terminal

"The page as blocks: prose lines, cells, outputs and figures (dropped: they are written again)."
function parse_page(lines)
    blocks = Tuple{Symbol,Vector{String}}[]
    i = 1
    while i <= length(lines)
        line = lines[i]
        if startswith(line, "```")
            info = strip(lstrip(line, '`'))
            j = i + 1
            while j <= length(lines) && !occursin(r"^```\s*$", lines[j])
                j += 1
            end
            body = lines[i+1:min(j - 1, end)]
            kind = info == "julia" ? :cell : info == "output" ? :output : :prose
            push!(blocks, (kind, kind === :prose ? lines[i:min(j, end)] : body))
            i = j + 1
            continue
        end
        if startswith(line, "![](figures/")
            i += 1
            continue
        end
        push!(blocks, (:prose, [line]))
        i += 1
    end
    blocks
end

"Whether a value is a figure (Makie's, or any value that draws itself), not a table that also shows as an image."
is_figure(v) = v !== nothing && showable(MIME"image/png"(), v) && !showable(MIME"text/html"(), v) ||
               (v !== nothing && occursin(r"Figure|FigureAxisPlot|Scene", string(typeof(v))))

"Evaluate a cell's code in Main the way the REPL does: every expression, the last one's value shown."
function run_cell(code::String, io::IO)
    ex = Meta.parseall(code)
    value = nothing
    for x in ex.args
        x isa LineNumberNode && continue
        value = Core.eval(Main, x)
    end
    hide = endswith(rstrip(replace(code, r"#[^\n]*$" => "")), ";")
    (value, hide)
end

function run_page(path::String)
    lines = readlines(path)
    blocks = parse_page(lines)
    stem = replace(basename(path), r"\.md$" => "")
    figdir = joinpath(dirname(path), "figures")
    mkpath(figdir)
    foreach(f -> startswith(f, stem * "-") && endswith(f, ".png") && rm(joinpath(figdir, f)), readdir(figdir))
    cd(dirname(path))
    written = String[]
    failed = false
    ncell = 0
    figures = 0
    for (kind, body) in blocks
        kind === :output && continue
        if kind === :prose
            append!(written, body)
            continue
        end
        push!(written, "```julia", body..., "```")
        failed && continue
        ncell += 1
        started = time()
        out = tempname()
        value, hide, err = nothing, true, nothing
        open(out, "w") do f
            ctx = IOContext(f, :limit => true, :displaysize => (ROWS, WIDTH), :color => false)
            logger = ConsoleLogger(ctx, Logging.Info; meta_formatter=(level, _module, group, id, file, line) ->
                (level == Logging.Warn ? :yellow : :cyan, string(level == Logging.Warn ? "Warning" : level, ":"), ""))
            redirect_stdout(f) do
                redirect_stderr(f) do
                    with_logger(logger) do
                        try
                            value, hide = run_cell(join(body, "\n"), ctx)
                        catch e
                            err = (e, catch_backtrace())
                        end
                    end
                end
            end
            flush(f)
            if err === nothing && !hide && value !== nothing && !Base.invokelatest(is_figure, value)
                Base.invokelatest(show, ctx, MIME"text/plain"(), value)
                println(f)
            elseif err !== nothing
                e = err[1] isa LoadError ? err[1].error : err[1]
                print(f, "ERROR: ")
                Base.invokelatest(showerror, ctx, e)
                println(f)
            end
        end
        text = rstrip(read(out, String))
        rm(out)
        isempty(text) || push!(written, "", "```output", split(text, '\n')..., "```")
        if err === nothing && !hide && Base.invokelatest(is_figure, value)
            figures += 1
            file = "$stem-$(lpad(figures, 2, '0')).png"
            open(joinpath(figdir, file), "w") do f
                Base.invokelatest(show, f, MIME"image/png"(), value)
            end
            push!(written, "", "![](figures/$file)")
        end
        failed = err !== nothing
        println(stdout, "  cell $(lpad(ncell, 2))  $(lpad(round(time() - started; digits=1), 5))s", failed ? "  FAILED" : "")
        failed && println(stdout, text)
    end
    keep = [i == 1 || !(written[i] == "" && written[i-1] == "") for i in eachindex(written)]
    write(path, join(written[keep], "\n") * "\n")
    !failed
end

if length(ARGS) == 2 && ARGS[1] == "--page"
    exit(run_page(abspath(ARGS[2])) ? 0 : 1)
end

isempty(ARGS) && (println(stderr, "usage: julia tutorials.jl PAGE.md [PAGE.md ...]"); exit(2))
ok = true
for page in ARGS
    println(page)
    t0 = time()
    # each page in its own process: a fresh session, as a reader starts one
    cmd = `$(Base.julia_cmd()) --project=$(Base.active_project()) $(@__FILE__) --page $(abspath(page))`
    status = success(pipeline(cmd; stdout=stdout, stderr=stderr))
    println("  ", status ? "ok" : "FAILED", " in ", round(Int, time() - t0), "s")
    global ok = ok && status
end
exit(ok ? 0 : 1)
