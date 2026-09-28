# The tests read no settings from the environment they run in (an agent's
# caller, a log folder, a model key).
withr::local_envvar(FUNCTAI_CALLER = NA, FUNCTAI_LOG_CALLS = NA, FUNCTAI_LOG_CONTENT = NA, .local_envir = testthat::teardown_env())

# A fake model: answers each request from a script or a function of the
# request, and records every request. No network.

fake_router <- function(replies = list(), responder = NULL, provider = "openai") {
  env <- new.env()
  env$requests <- list()
  env$replies <- as.list(replies)
  complete <- function(request) {
    env$requests[[length(env$requests) + 1L]] <- request
    i <- length(env$requests)
    reply <- if (!is.null(responder)) responder(request, i) else { r <- env$replies[[1L]]; env$replies <- env$replies[-1L]; r }
    if (inherits(reply, "condition")) stop(reply)
    if (is.character(reply)) reply <- list(text = reply)
    parts <- list()
    if (!is.null(reply$text)) parts[[length(parts) + 1L]] <- lm15::text_part(reply$text)
    for (c in reply$calls) parts[[length(parts) + 1L]] <- lm15::tool_call_part(c$id, c$name, input = do.call(lm15::json_object, c$input))
    lm15::response(request$model, lm15::message_assistant(parts), reply$finish %||% if (length(reply$calls)) "tool_call" else "stop",
                   usage = lm15::usage(input_tokens = 10L, output_tokens = 5L, total_tokens = 15L))
  }
  resolve <- function(model) {
    head <- sub(":.*$", "", model)
    if (grepl(":", model, fixed = TRUE)) list(provider = head, model = substring(model, nchar(head) + 2L)) else list(provider = provider, model = model)
  }
  list(resolve = resolve, complete = complete, env = env)
}

# The text of a request's message (an lm15 request).
message_text <- function(request, i) {
  m <- lmcc::lm15_plain(lm15::as_dict(request))$messages[[i]]
  paste(vapply(m$parts, function(p) p$text %||% "", ""), collapse = "")
}

last_text <- function(request) {
  msgs <- lmcc::lm15_plain(lm15::as_dict(request))$messages
  message_text(request, length(msgs))
}

contract_root <- function() {
  root <- Sys.getenv("FUNCTAI_CONTRACT", file.path("..", "..", "..", "contract"))
  if (!dir.exists(root)) testthat::skip("the contract folder (../contract) is not here")
  normalizePath(root)
}

read_json_file <- function(path) lmcc::parse_json(paste(readLines(path, warn = FALSE, encoding = "UTF-8"), collapse = "\n"))

# The records of a log folder, in the order they were written.
log_lines <- function(folder) {
  lines <- unlist(lapply(sort(list.files(folder, recursive = TRUE, full.names = TRUE, pattern = "\\.jsonl$")), readLines, encoding = "UTF-8"))
  lapply(lines, lmcc::parse_json)
}

# The inputs an AI function's call binds, given these (its R function's own
# defaults filling what is left out), as the JSON row it sends.
bound_row <- function(fn, args) {
  f <- unclass(fn)
  body(f) <- quote(input_rows(.core, mget(.inputs, envir = environment()))[[1L]])
  do.call(f, args)
}
