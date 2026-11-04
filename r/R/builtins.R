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

# ---------------------------------------------------------------- long tool loops (the step hook)

LIMIT_REACHED <- "This task has reached its limit (%s). Do not ask for any more tools: give your answer now, from what you have found so far."
LAST_STEP <- "This is the last step this task may take. Do not ask for any more tools: give your answer now, from what you have found so far."
PICTURE_GONE <- "(A picture was here. It is no longer shown, to save space.)"
COST_UNKNOWN <- paste0("a request's cost is not known from its token counts (a price without cache_read_per_million and ",
                       "cache_write_per_million is exact only for requests that report their cache counts; give both, equal to ",
                       "input_per_million when the provider does not charge them apart)")

# A limit as the model is told it: a whole number without a fraction.
limit_number <- function(x) if (x == round(x)) format(round(x), scientific = FALSE) else format(x, digits = 6, scientific = FALSE)

# Which limits a tree's spending reached (plugins.md, limits): each
# list(label, why), why NULL unless its count is not known (then it counts as reached).
limits_reached <- function(spent, tokens = NULL, cost = NULL, seconds = NULL, prices = NULL) {
  out <- list()
  if (!is.null(tokens)) {
    got <- spent$tokens
    if (is.null(got)) out[[length(out) + 1L]] <- list(label = paste(limit_number(tokens), "tokens"), why = "a request did not say how many it used")
    else if (got >= tokens) out[[length(out) + 1L]] <- list(label = paste(limit_number(tokens), "tokens"), why = NULL)
  }
  if (!is.null(cost)) {
    got <- ai_cost(spent, prices)
    if (is.null(got)) {
      unknown <- sort_code_points(unique(vapply(Filter(function(x) is.null(pricing(prices, x$model)), spent$exchanges), function(x) x$model, "")))
      out[[length(out) + 1L]] <- list(label = paste("cost", limit_number(cost)),
                                      why = if (length(unknown)) paste("no price is known for", paste(unknown, collapse = ", ")) else COST_UNKNOWN)
    } else if (got >= cost) out[[length(out) + 1L]] <- list(label = paste("cost", limit_number(cost)), why = NULL)
  }
  if (!is.null(seconds) && spent$seconds >= seconds) out[[length(out) + 1L]] <- list(label = paste(limit_number(seconds), "seconds"), why = NULL)
  out
}

stopped_by <- function(hit) paste0("limit reached: ", paste(vapply(hit, function(h) if (is.null(h$why)) h$label else sprintf("%s (%s)", h$label, h$why), ""), collapse = "; "))

# A price in limits' config: its fields by lm15's names, without those at lm15's defaults (an unset rate, USD, no tiers).
price_config <- function(p) {
  d <- plain_lm15(p)
  d <- d[!vapply(d, is.null, NA)]
  if (identical(d$currency, "USD")) d$currency <- NULL
  if (!length(d$tiers)) d$tiers <- NULL
  d
}

#' Limits on what a call tree spends
#'
#' Stop a call tree (an agent, a program) when it has used enough: tokens,
#' money or time. Nothing is limited by default; this plugin is opt-in.
#'
#' Counted over the whole tree a call starts (every AI function inside a
#' program, every tool's own calls), from the requests a provider answered
#' in this session (replies from the reply cache, and those a resumed turn
#' reuses, cost nothing now and are not counted), plus what code said it
#' spent ([ai_spend()]). Checked before each AI function's call and before
#' each step of a tool loop, so one request may go past a limit; the next
#' does not start. FunctAI ships no price list: prices change, and a wrong
#' price would mislead.
#' @param tokens Every token read and written (`total_tokens`), summed.
#' @param cost The list price of the requests, from `prices` (needed with
#'   it). Exact or nothing: a request to a model with no price, or whose
#'   counts do not settle its price, counts as reaching the limit.
#' @param seconds Since the tree's outermost call started.
#' @param prices A model's name (as you ask for it) to an
#'   `lm15::inference_pricing()`, or a list of its fields
#'   (`list(input_per_million = 3, output_per_million = 15,
#'   cache_read_per_million = 0.3, cache_write_per_million = 3.75)`).
#' @param then `"answer"` (the default): a tool loop that reaches a limit is
#'   offered no more tools and told to answer with what it has; so is a loop
#'   at its last step (`max_steps`), instead of failing. `"stop"`: the call
#'   stops. Either way a call that would start past a limit stops
#'   (`functai_plugin_stopped`, code `plugin-stopped`).
#' @param name The plugin's name.
#' @return A plugin, for the `plugins` setting.
#' @examples
#' prices <- list("claude-sonnet-5" = list(input_per_million = 3, output_per_million = 15,
#'                                         cache_read_per_million = 0.3, cache_write_per_million = 3.75))
#' limits(cost = 3, seconds = 600, prices = prices)
#' @export
limits <- function(tokens = NULL, cost = NULL, seconds = NULL, prices = NULL, then = c("answer", "stop"), name = "limits") {
  then <- match.arg(then)
  if (!is.null(cost) && !length(prices))
    cli::cli_abort("a cost limit needs prices: {.code list(model = list(input_per_million = ..., output_per_million = ...))}; FunctAI ships none")
  for (k in c("tokens", "cost", "seconds")) {
    v <- get(k)
    if (!is.null(v) && !(is_num(v) && is.finite(v) && v > 0)) cli::cli_abort("{.arg {k}} is a number above 0")
  }
  if (is.null(tokens) && is.null(cost) && is.null(seconds)) cli::cli_abort("{.fn limits} needs at least one of {.arg tokens}, {.arg cost}, {.arg seconds}")
  priced <- if (length(prices)) stats::setNames(lapply(names(prices), function(m) pricing(prices, m)), names(prices)) else NULL
  config <- list()
  if (!is.null(tokens)) config$tokens <- tokens
  if (!is.null(cost)) config$cost <- cost
  if (!is.null(seconds)) config$seconds <- seconds
  if (!is.null(priced)) config$prices <- lapply(priced[sort_code_points(names(priced))], price_config)
  config$then <- then
  reached <- function(spent) limits_reached(spent, tokens, cost, seconds, priced)
  ai_plugin(name, version = "1.0.0", description = "stop a call tree when it used enough tokens, money or time", config = config,
    before_call = function(call) {
      hit <- reached(call$spent)
      if (!length(hit) && !is.null(cost) && is_str(call$lm) && is.null(pricing(priced, call$lm)))
        hit <- list(list(label = paste("cost", limit_number(cost)), why = paste("no price is known for", call$lm)))
      if (length(hit)) ai_change(stop = stopped_by(hit))
    },
    step = function(s) {
      hit <- reached(s$spent)
      if (length(hit) && then == "stop") return(ai_change(stop = stopped_by(hit)))
      if (then == "answer" && length(s$tools) && (length(hit) || isTRUE(s$last_step)))
        return(ai_change(tools = character(0), reminder = if (length(hit)) sprintf(LIMIT_REACHED, paste(vapply(hit, function(h) h$label, ""), collapse = ", ")) else LAST_STEP))
      NULL
    })
}

# Which earlier tool results keep_pictures shows otherwise, and as what
# (plugins.md): the pictures before the newest `last` become a short text,
# once at least `at_once` are due. A named list by invocation.
pictures_to_drop <- function(results, last, at_once = 1L) {
  seen <- 0L; drop <- list()
  for (r in rev(results)) {
    if (is_str(r$output)) next
    for (i in rev(seq_along(r$output))) {
      if (!identical(r$output[[i]]$type, "image")) next
      seen <- seen + 1L
      if (seen > last) drop[[as.character(r$invocation)]] <- c(drop[[as.character(r$invocation)]], i)
    }
  }
  if (sum(lengths(drop)) < at_once) return(list())
  out <- list()
  for (r in results) {
    gone <- drop[[as.character(r$invocation)]]
    if (is.null(gone)) next
    out[[as.character(r$invocation)]] <- lapply(seq_along(r$output), function(i) if (i %in% gone) list(type = "text", text = PICTURE_GONE) else r$output[[i]])
  }
  out
}

#' Show a long tool loop only its newest pictures
#'
#' Before each step, the pictures tool results gave earlier than the newest
#' `last` are replaced by a short text (the results' own text stays). A
#' browsing agent that takes a screenshot at every click otherwise sends
#' every screenshot again with every request. `at_once`: drop old pictures
#' only once this many are due, all together: a provider's prompt cache
#' keeps the start of a request only while it stays the same, and every
#' drop changes it. The turn and the call log keep every picture as the tool
#' gave it.
#' @param last How many of the newest pictures stay shown (0 or more).
#' @param at_once How many are dropped together (1 or more).
#' @param name The plugin's name.
#' @return A plugin, for the `plugins` setting.
#' @examples
#' keep_pictures(3, at_once = 5)
#' @export
keep_pictures <- function(last = 3L, at_once = 1L, name = "keep-pictures") {
  if (!(is_num(last) && last >= 0 && last == round(last))) cli::cli_abort("{.arg last} is how many of the newest pictures stay shown (0 or more)")
  if (!(is_num(at_once) && at_once >= 1 && at_once == round(at_once))) cli::cli_abort("{.arg at_once} is 1 or more")
  ai_plugin(name, version = "1.0.0", description = "show a tool loop only its newest pictures",
            config = list(last = as.integer(last), at_once = as.integer(at_once)),
            step = function(s) {
              drop <- pictures_to_drop(s$results, last, at_once)
              if (length(drop)) ai_change(replace = drop)
            })
}
