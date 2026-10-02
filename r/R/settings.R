# Settings: a function's own beat a with_ai_config() block, which beats
# ai_config(), which beats the defaults (the order every language uses).

DEFAULTS <- list(retries = 1L, api_retries = 3L, max_steps = 8L, tool_errors = "report",
                 include_fn_name = TRUE, concurrency = 8L)

SETTINGS <- c("lm", "router", "temperature", "max_tokens", "top_p", "stop", "seed", "config", "adapter", "template",
              "module", "include_fn_name", "capabilities", "retries", "api_retries", "max_steps", "tool_errors",
              "log_calls", "log_content", "caller", "concurrency", "on_error", "observers", "journal", "program_observers",
              "cache_replies", "replicate", "progress", "plugins", "program_plugins", "approve", "escalate_to", "escalate_below")

check_settings <- function(s, call = rlang::caller_env()) {
  twice <- unique(names(s)[duplicated(names(s))])
  if (length(twice)) cli::cli_abort("setting{?s} {.val {twice}} given twice")
  bad <- setdiff(names(s), SETTINGS)
  if (length(bad)) cli::cli_abort(c("unknown setting{?s}: {.val {bad}}", i = "settings are {.val {SETTINGS}}"))
  if ("log_content" %in% names(s)) s["log_content"] <- list(normalize_log_content(s$log_content, call = call))
  if ("journal" %in% names(s)) s["journal"] <- list(journal_setting(s$journal))
  if ("observers" %in% names(s)) {
    o <- s$observers
    if (is.function(o)) o <- list(o)
    if (!is.null(o) && (!is.list(o) || !all(vapply(o, is.function, NA))))
      cli::cli_abort("{.arg observers} is a list of functions, each given every event as it is made", call = call)
    s["observers"] <- list(o)
  }
  if ("program_observers" %in% names(s) && !is.null(s$program_observers) && !rlang::is_bool(s$program_observers))
    cli::cli_abort("{.arg program_observers} is TRUE or FALSE", call = call)
  if ("cache_replies" %in% names(s) && !is.null(s$cache_replies) && !isTRUE(s$cache_replies) && !isFALSE(s$cache_replies)) reply_store(s$cache_replies)
  if ("replicate" %in% names(s) && !is.null(s$replicate) && !(is_num(s$replicate) && s$replicate >= 0 && s$replicate == round(s$replicate)))
    cli::cli_abort("{.arg replicate} is a whole number from 0: the n-th independent answer to the same request", call = call)
  if ("plugins" %in% names(s)) s["plugins"] <- list(check_plugins(s$plugins, call = call))
  if ("approve" %in% names(s)) s["approve"] <- list(check_approve(s$approve, call = call))
  s
}

#' Settings for every AI function
#'
#' `ai_config()` sets them for the session (a function's own settings still
#' win); `with_ai_config()` and `local_ai_config()` for a block of code, as
#' withr does.
#'
#' **What the call log keeps** (`log_content`) is not overridden by a
#' closer setting: every layer (a function's own, each block, `ai_config()`,
#' and the environment variable `FUNCTAI_LOG_CONTENT`) can only remove. A
#' value is written only when no layer drops it, so a host holds every
#' function it runs to "never write the transcript":
#' `with_ai_config(code, log_content = c(transcript = FALSE))`, or, surest
#' against a misspelt name, the list of what may be kept,
#' `log_content = c("question")` (the same as `list("*" = FALSE, question =
#' TRUE)`). Mind the difference: `c(question = TRUE)` names one field and
#' says nothing of the others, so it keeps everything; `c("question")` keeps
#' only the question. A name is always the field of that name (a field
#' FunctAI adds included); a one-output function's answer may also be called
#' by its formula's name when no field has that name.
#' `FUNCTAI_LOG_CONTENT=0` drops every value, whatever the
#' settings say. A field FunctAI adds (the reasoning, the tool calls) is
#' written only when no field of the call is dropped. A record that does not
#' keep every value keeps no request or reply either, and no error message.
#'
#' @param ... Settings: `lm` (the model: `"gpt-4.1-mini"`, `"claude-haiku-4-5"`,
#'   `"gemini:gemini-2.5-flash"`, ...), `temperature`, `max_tokens`, `adapter`
#'   (`"xml"`, `"chat"`, `"json"`), `module` (`"cot"`: reasoning first),
#'   `retries`, `api_retries`, `concurrency` (calls in flight at once, default
#'   8), `on_error` (`"warn"`, the default: a failed row is `NA`; or `"stop"`),
#'   `log_calls` (a folder, or `TRUE`), `log_content` (`TRUE`, `FALSE`,
#'   `c(transcript = FALSE)`, or the names of the fields to keep), `caller`, `router` (an
#'   `lm15::new_router()`); `cache_replies` (`TRUE`: in memory; `"disk"`, a
#'   folder or a `.sqlite` file: kept across runs, shared by every language)
#'   and `replicate` (the n-th independent answer to the same request);
#'   `progress` (a progress line over a column); `observers` (functions given
#'   every event), `journal` ([ai_journal()]) and `program_observers`;
#'   `plugins` ([ai_plugin()]) and `program_plugins`; `approve` (a function,
#'   `"changes"`, `"all"`, or tool names: see [ai_conversation()]);
#'   `escalate_to` (a model or an AI function that answers when the first
#'   model is less sure than `escalate_below`, default 0.9).
#' @return The settings in force, invisibly (`ai_config()`), or the value of
#'   `code` (`with_ai_config()`).
#' @examples
#' ai_config(lm = "gpt-4.1-mini", temperature = 0)
#' @export
ai_config <- function(...) {
  s <- check_settings(list(...))
  for (k in names(s)) the$config[k] <- list(s[[k]])
  invisible(the$config)
}

#' @rdname ai_config
#' @param code Code to run with these settings.
#' @export
with_ai_config <- function(code, ...) {
  old <- enter_block(check_settings(list(...)))
  on.exit(leave_block(old))
  force(code)
}

#' @rdname ai_config
#' @param .local_envir The environment whose end ends these settings.
#' @export
local_ai_config <- function(..., .local_envir = parent.frame()) {
  old <- enter_block(check_settings(list(...)))
  withr::defer(leave_block(old), envir = .local_envir)
  invisible(the$scoped)
}

# A block's settings: each replaces the one outside it, except log_content,
# which is one more layer that can only remove (content.R).
enter_block <- function(s) {
  old <- list(scoped = the$scoped, blocks = the$content_blocks, layers = the$blocks)
  the$scoped <- set_all(old$scoped, s[names(s) != "log_content"])
  if ("log_content" %in% names(s)) the$content_blocks <- c(old$blocks, list(s$log_content))
  the$blocks <- c(old$layers, list(s))          # each block's own settings, for what adds up or holds across layers
  old
}

leave_block <- function(old) {
  the$scoped <- old$scoped
  the$content_blocks <- old$blocks
  the$blocks <- old$layers
}

# `base` with each of `new`'s settings replacing its own (not merged into it).
set_all <- function(base, new) {
  base <- base %||% list()
  for (k in names(new)) base[k] <- list(new[[k]])
  base
}

effective <- function(own = list()) {
  out <- DEFAULTS
  for (layer in list(the$config, the$scoped, own)) for (k in names(layer)) if (!is.null(layer[[k]])) out[k] <- list(layer[[k]])
  out$caller <- c(the$config$caller, the$scoped$caller, own$caller)
  out$caller <- out$caller[!duplicated(names(out$caller), fromLast = TRUE)]
  out$log_content <- NULL          # not overridden by the closest layer: content_layers() reads every one
  out
}

# The lm15 config of the settings.
config_of <- function(s, overrides = list()) {
  args <- s$config %||% list()
  for (k in c("temperature", "max_tokens", "top_p", "seed")) if (!is.null(s[[k]])) args[[k]] <- s[[k]]
  if (!is.null(s$stop) && length(s$stop)) args$stop <- as.list(s$stop)
  args[names(overrides)] <- overrides
  if (!length(args)) return(NULL)
  if (!is.null(args$max_tokens)) args$max_tokens <- as.integer(args$max_tokens)
  do.call(lm15::config, args)
}
