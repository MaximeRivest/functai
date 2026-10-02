# Built-in plugins, made with the public hooks only (contract/plugins.md,
# "Built-in plugins"): anyone can replace them with their own.

# The AI function compaction summarizes with by default: the same instruction
# and fields as Python's and Julia's, so the same version.
default_summarizer <- function() {
  if (is.null(the$summarizer)) the$summarizer <- ai(result ~ earlier_summary + new_turns, paste(
    "Write the summary of a conversation that whoever continues it needs:",
    "what was asked, what was answered and decided, names, numbers and",
    "facts that came up, and what is still open. Start from the earlier",
    "summary (empty when there is none) and fold the new turns into it. Be",
    "complete about facts and short about everything else.", sep = "\n"),
    new_turns = json_shape(list(type = "array", items = list(type = "object"))),
    .name = "summarize_conversation", .defined_in = "functai.builtins")
  the$summarizer
}

#' Keep a long conversation short: older turns folded into a summary
#'
#' When a turn ends and more than `keep + every` turns of its branch are not
#' yet summarized, every turn but the last `keep` is folded into the summary
#' (the earlier summary and those turns, given to `summarize(earlier_summary,
#' new_turns)`, `new_turns` a list of `list(input..., output...)` rows;
#' default: an AI function of functai's, on `lm` when given). The summary is
#' kept in the conversation (an entry at that turn, so each branch has its
#' own), and the next turns are shown it as a section of the instruction,
#' with only the turns after it. What a turn was shown is in its record, so a
#' rated turn is asked again with the same summary. `every` keeps the
#' summary's text the same for `every` turns: a provider's prompt cache keeps
#' its prefix meanwhile.
#' @param keep How many recent turns are always shown whole.
#' @param every How many more turns may pile up before the next summary.
#' @param summarize A function of the earlier summary and the new turns,
#'   giving the summary's text.
#' @param lm The model the default summarizer uses.
#' @param name The plugin's name (two compactions on one conversation need two).
#' @return A plugin.
#' @examples
#' \dontrun{
#' chat <- ai_conversation(tutor, "alex", store = "tutoring/", plugins = list(compaction(keep = 20)))
#' }
#' @export
compaction <- function(keep = 20L, every = 10L, summarize = NULL, lm = NULL, name = "compaction") {
  if (!is_num(keep) || keep < 0 || !is_num(every) || every < 1) cli::cli_abort("{.fn compaction}: {.arg keep} a whole number of at least 0, {.arg every} at least 1")
  latest <- function(es) if (length(es)) es[[length(es)]]$data else NULL
  fold <- function(event) {
    if (!identical(event$state, "done")) return(NULL)
    ts <- branch_turns(event)
    summary <- latest(entries(event, "summary"))
    ids <- vapply(ts, function(t) t$id, "")
    through <- if (is.null(summary)) 0L else match(summary$through, ids, nomatch = 0L)
    open <- ts[seq_along(ts) > through]
    if (length(open) < keep + every) return(NULL)
    folded <- open[seq_len(length(open) - keep)]
    rows <- lapply(folded, function(t) c(t$inputs, t$outputs))
    fn <- summarize %||% { f <- default_summarizer(); if (!is.null(lm)) update(f, lm = lm) else f }
    text <- with_ai_config(fn(if (is.null(summary)) "" else summary$text, list(rows)), caller = list(compaction = event$turn))
    if (!is_str(text) || !nzchar(trim_white(text))) stop("a summary is a text")
    keep_entry(event, "summary", list(through = folded[[length(folded)]]$id, text = text,
                                      turns = (if (is.null(summary)) 0L else as.integer(summary$turns)) + length(folded)))
    NULL
  }
  show <- function(event) {
    summary <- latest(entries(event, "summary"))
    if (is.null(summary)) return(NULL)
    log <- read_conv(event$conv)
    ids <- vapply(branch_of(log, event$parent), turn_id, "")
    i <- match(summary$through, ids, nomatch = 0L)
    if (!i) return(NULL)
    covered <- ids[seq_len(i)]
    ai_change(keep = Filter(function(id) !id %in% covered, vapply(event$turns, function(t) t$id, "")),
              sections = sprintf("Earlier in this conversation (%d turns, summarized):\n%s", as.integer(summary$turns), summary$text))
  }
  ai_plugin(name, version = "1.0.0", description = sprintf("compaction: summarize all but the last %d turns", as.integer(keep)), turn_end = fold, context = show)
}

#' Another program as a tool
#'
#' An assistant hands part of its work to another AI function or program.
#' Asked inside a conversation's turn, the program answers in a conversation
#' of its own (`<conversation>.<name>`, in the same store), which follows the
#' branch of the turn that asked: asked again later on that branch, it
#' remembers what it was asked before; on another branch, it does not. Its
#' calls are in the asking turn's call tree, under the tool call. Outside a
#' conversation, or with `remember = FALSE`, it is called plainly.
#' @param program An AI function or a program.
#' @param description What it does, for the model.
#' @param name The tool's name (default: the program's).
#' @param remember Whether it keeps its own conversation (see above).
#' @param effects `"reads"` or `"changes"`; default: `"reads"` for an AI
#'   function whose tools all read, else unknown (counts as changes).
#' @return A tool, for `.tools`.
#' @export
delegate <- function(program, description = NULL, name = NULL, remember = TRUE, effects = NULL) {
  if (!inherits(program, c("functai_fn", "functai_program"))) cli::cli_abort("{.fn delegate} hands work to an AI function or a program")
  core <- program_core_of(program)
  tool_name <- name %||% core$definition$name
  run <- function(...) {
    given <- list(...)
    call <- the$current
    turn_run <- if (is.null(call)) NULL else call$turn_run
    if (!remember || is.null(turn_run)) return(do.call(program, given))
    outer <- turn_run$conv
    sub_id <- paste0(outer$id, ".", tool_name)
    if (nchar(sub_id) > 200L) sub_id <- paste0(substr(outer$id, 1L, 150L), ".", substr(lmcc::sha256_hex(sub_id), 1L, 16L))
    sub <- ai_conversation(program, sub_id, store = outer$store)
    d <- conv_of(sub); d$delegated <- TRUE
    previous <- conv_entries(outer, "delegate", tool_name, turn_run$turn)
    if (!length(previous)) { d$follow <- FALSE; d$head <- NULL; d$exact <- TRUE }
    else { d$follow <- FALSE; d$head <- previous[[length(previous)]]$data$turn; d$exact <- TRUE }
    value <- send_turn(d, given)
    keep_entry(outer, tool_name, list(turn = d$head), plugin = "delegate", turn = turn_run$turn)
    value
  }
  iface <- program_iface(program)
  props <- lapply(iface$inputs, function(f) if (is.null(f$desc)) no_defaults(f$shape) else c(no_defaults(f$shape), list(description = f$desc)))
  names(props) <- vapply(iface$inputs, function(f) f$name, "")
  required <- Filter(function(f) !isTRUE(f$optional), iface$inputs)
  eff <- effects %||% if (!is_program(program) && all(vapply(core$tools, function(t) identical(t$effects, "reads"), NA))) "reads" else NULL
  structure(list(name = tool_name, description = description %||% core$definition$description,
                 parameters = list(type = "object", properties = if (length(props)) props else lmcc::jobj(), required = lapply(required, function(f) f$name)),
                 fn = run, effects = eff), class = "functai_tool")
}
