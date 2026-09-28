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

# A value as canonical JSON, to compare as the contract compares.
plain <- function(x) lmcc::canonical_json(unclass(x))

read_json_file <- function(path) lmcc::parse_json(paste(readLines(path, warn = FALSE, encoding = "UTF-8"), collapse = "\n"))

# The records of a log folder, in the order they were written.
log_lines <- function(folder) {
  lines <- unlist(lapply(sort(list.files(folder, recursive = TRUE, full.names = TRUE, pattern = "\\.jsonl$")), readLines, encoding = "UTF-8"))
  lapply(lines, lmcc::parse_json)
}

# The inputs an AI function's call binds, given these, as the JSON row it
# sends: the binding alone (no call), for unit tests of it. The contract's
# `binds` and `sends` are checked by calling the function (probe_call()).
bound_row <- function(fn, args) {
  f <- unclass(fn)
  body(f) <- quote(input_rows(.core, given_inputs(environment(), .inputs))[[1L]])
  do.call(f, args)
}

# Calls `fn` for real with `args` (by name), under the probe facts
# (contract/models.json): its own code binds the inputs, renders and sends one
# request to a fake provider "probe", which fails at once (so the call ends
# after that one request, whatever the function's outputs), and writes its
# record. Returns the record the library wrote and the requests the provider
# got.
probe_call <- function(fn, args) {
  router <- fake_router(responder = function(req, i) simpleError("the probe provider answers nothing"), provider = "probe")
  folder <- tempfile("probe-log-")
  on.exit(unlink(folder, recursive = TRUE))
  g <- update(fn, router = router, lm = "probe-model", capabilities = probe_capabilities(), log_calls = folder,
              log_content = TRUE, api_retries = 0L, on_error = "stop")
  tryCatch(do.call(g, args), error = function(e) NULL)
  recs <- log_lines(folder)
  list(record = if (length(recs)) recs[[1L]], requests = router$env$requests)
}

# The hash of the request the probe renders for these inputs (`model`
# "probe", as a version's), and the hash lmcc records for the same render as
# a model step's request (no model): what a call's exchange keeps.
probe_hashes <- function(fn, inputs) {
  p <- probe_request(core_of(fn), inputs)
  step <- p; step$model <- NULL
  list(probe = lmcc::sha256_of(p), step = lmcc::sha256_of(step))
}

# The contract's `sends`/`binds` expectation, checked on a real call: the
# record holds the inputs the call bound; the request it sent (the hash of
# its render, which the record keeps) is the probe's render of those inputs;
# and that render's hash is `request_hash` (when given).
expect_sends <- function(fn, inputs, bound = NULL, request_hash = NULL) {
  got <- probe_call(fn, inputs)
  testthat::expect_false(is.null(got$record), info = "the call wrote no record")
  if (is.null(got$record)) return(invisible())
  rec <- got$record
  testthat::expect_length(got$requests, 1L)
  if (!is.null(bound)) testthat::expect_identical(lmcc::canonical_json(rec$inputs), lmcc::canonical_json(bound))
  h <- probe_hashes(fn, rec$inputs)
  testthat::expect_identical(rec$exchanges[[1L]]$request_hash, h$step)
  if (!is.null(request_hash)) testthat::expect_identical(h$probe, request_hash)
  invisible(rec)
}

# A folder holding these records as one log file (one line each), as a
# writer leaves it.
log_folder_of <- function(records) {
  folder <- tempfile("log-")
  day <- file.path(folder, "2026-09-28")
  dir.create(day, recursive = TRUE)
  writeLines(vapply(records, lmcc::json_text, ""), file.path(day, "case.jsonl"), useBytes = TRUE)
  folder
}
