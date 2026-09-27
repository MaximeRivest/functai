# Run R tutorials (Markdown with ```r cells) top to bottom, each in a fresh R
# session, and write every cell's output under it, the way the Python pages in
# docs/ carry theirs:
#
#   ```r                 a cell: run
#   ```output            what it printed (rewritten on every run)
#   ![](figures/...)     a plot it drew (rewritten on every run)
#   ```{.r .no-run}      shown, never run (an install line, a sign-in)
#   #| fig-height: 2.5    in a cell: the height of its plots, in inches (width 7)
#
#   r/tutorials docs/r/01-first-function.md ...   (the wrapper finds R and the keys)
#
# A cell that fails stops the run: the page is written up to that cell, with
# the error under it, and the exit status is 1. Tutorials call real models, so
# a run costs money; each tutorial counts its own cost in its last section.

args <- commandArgs(trailingOnly = TRUE)
if (!length(args)) stop("usage: Rscript tutorials.R PAGE.md [PAGE.md ...]", call. = FALSE)

fence_open <- function(line) sub("^```+\\s*", "", line)
is_fence <- function(line) grepl("^```", line)

# The page as blocks: prose lines, cells (code), outputs and figures (dropped:
# they are written again).
parse_page <- function(lines) {
  out <- list(); i <- 1L; n <- length(lines)
  while (i <= n) {
    line <- lines[[i]]
    if (is_fence(line)) {
      info <- fence_open(line)
      j <- i + 1L
      while (j <= n && !grepl("^```\\s*$", lines[[j]])) j <- j + 1L
      body <- if (j > i + 1L) lines[(i + 1L):(j - 1L)] else character(0)
      kind <- if (identical(info, "r")) "cell" else if (identical(info, "output")) "output" else "prose"
      out[[length(out) + 1L]] <- list(kind = kind, lines = if (kind == "prose") lines[i:j] else body)
      i <- j + 1L
      next
    }
    if (grepl("^!\\[\\]\\(figures/", line)) { i <- i + 1L; next }   # a plot this runner wrote
    out[[length(out) + 1L]] <- list(kind = "prose", lines = line)
    i <- i + 1L
  }
  out
}

format_condition <- function(c) {
  msg <- sub("\n+$", "", conditionMessage(c))
  if (inherits(c, "error")) return(paste0("Error: ", msg))
  if (inherits(c, "warning")) return(paste0("Warning: ", msg))
  msg
}

run_page <- function(path) {
  lines <- readLines(path, warn = FALSE, encoding = "UTF-8")
  blocks <- parse_page(lines)
  stem <- sub("\\.md$", "", basename(path))
  figdir <- file.path(dirname(path), "figures")
  dir.create(figdir, showWarnings = FALSE)
  unlink(Sys.glob(file.path(figdir, paste0(stem, "-*.png"))))
  env <- new.env(parent = globalenv())
  old <- setwd(dirname(path)); on.exit(setwd(old), add = TRUE)
  options(width = 80, pillar.width = 80, cli.unicode = TRUE, cli.num_colors = 1, crayon.enabled = FALSE,
          warn = 1, pillar.bold = FALSE, cli.dynamic = FALSE, cli.progress_show_after = Inf)
  written <- character(0); figures <- 0L; failed <- FALSE; n_cells <- 0L
  for (b in blocks) {
    if (b$kind == "output") next
    if (b$kind == "prose") { written <- c(written, b$lines); next }
    written <- c(written, "```r", b$lines, "```")
    if (failed) next
    n_cells <- n_cells + 1L
    started <- Sys.time()
    height <- as.numeric(sub(".*fig-height:\\s*", "", grep("^#\\| fig-height:", b$lines, value = TRUE)[1L]))
    if (is.na(height)) height <- 4
    # plots are drawn on a device of the figure's own size, so layouts that
    # measure text (rpart.plot, legends) fit the file they are written to
    grDevices::pdf(NULL, width = 7, height = height)
    grDevices::dev.control("enable")
    res <- evaluate::evaluate(paste(b$lines, collapse = "\n"), envir = env, stop_on_error = 1L,
                              new_device = FALSE, keep_warning = TRUE, keep_message = TRUE,
                              output_handler = evaluate::new_output_handler(value = function(x, visible) if (visible) print(x)))
    grDevices::dev.off()
    text <- character(0); plots <- list()
    for (r in res) {
      if (inherits(r, "source")) next
      if (is.character(r)) text <- c(text, sub("\n$", "", r))
      else if (inherits(r, "recordedplot")) plots[[length(plots) + 1L]] <- r
      else if (inherits(r, "packageStartupMessage")) next         # the reader's console shows them; the page need not
      else if (inherits(r, "condition")) {
        text <- c(text, format_condition(r))
        if (inherits(r, "error")) failed <- TRUE
      }
    }
    text <- unlist(strsplit(paste(text, collapse = "\n"), "\n", fixed = TRUE))
    if (length(text) && any(nzchar(text))) written <- c(written, "", "```output", text, "```")
    for (p in plots) {
      figures <- figures + 1L
      file <- sprintf("%s-%02d.png", stem, figures)
      grDevices::png(file.path(figdir, file), width = 7, height = height, units = "in", res = 110)
      grDevices::replayPlot(p)
      grDevices::dev.off()
      written <- c(written, "", sprintf("![](figures/%s)", file))
    }
    secs <- as.numeric(Sys.time() - started, units = "secs")
    cat(sprintf("  cell %2d  %5.1fs%s\n", n_cells, secs, if (failed) "  FAILED" else ""))
    if (failed) cat(paste(tail(text, 20), collapse = "\n"), "\n")
  }
  # squeeze runs of blank lines the rewrite may leave
  keep <- !(written == "" & c(FALSE, head(written, -1L) == ""))
  writeLines(written[keep], path, useBytes = TRUE)
  !failed
}

`%||%` <- function(a, b) if (is.null(a)) b else a

self <- sub("^--file=", "", grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)[[1L]])
if (identical(args[[1L]], "--page")) quit(status = if (run_page(args[[2L]])) 0 else 1)

ok <- TRUE
for (page in args) {
  cat(sprintf("%s\n", page))
  t0 <- Sys.time()
  # each page in its own R process: a fresh session, as a reader starts one
  status <- system2(file.path(R.home("bin"), "Rscript"), c(shQuote(self), "--page", shQuote(normalizePath(page))))
  cat(sprintf("  %s in %.0fs\n", if (status == 0) "ok" else "FAILED", as.numeric(Sys.time() - t0, units = "secs")))
  ok <- ok && status == 0
}
quit(status = if (ok) 0 else 1)
