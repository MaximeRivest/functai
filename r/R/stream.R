# Streaming (contract/streaming.md): the same call, watched while it is made.
# A stream asks the model for the same thing, retries the same way, runs the
# same tools, writes the same line to the call log and ends with the same
# value as calling the program; it adds a view. R's streams are callbacks, as
# lm15's and curl's are: each event is given to `.each` as it is made.

#' Watch a call while it is made
#'
#' Calls `fn` with `...` as calling it does, and gives each event of the call
#' (and of every call made inside it: a program's steps, a tool that calls an
#' AI function) to `.each` as it happens: `started`, `request`, `text` (a
#' piece of an output's text, `e$field`, `e$text`; `e$answer` is `TRUE` for
#' the answer), `thinking`, `tool_call`, `tool_result`, `retry` (the text so
#' far is void: the model is asked again), and `done` or `failed`. With
#' `.show`, the answer's text is written to the console as it comes.
#'
#' A call watched this way is streamed from the provider when it is the only
#' one in flight; the rows of a column go through the connection pool as
#' always, and each reply is shown whole. Calling `stop_stream()` inside
#' `.each` stops the call: the request in progress at its next piece, no new
#' request or call starts, and the call ends with the error `Cancelled` (in
#' the call log too).
#' @param fn An AI function or a program.
#' @param ... Its inputs.
#' @param .each A function given each event, as it is made.
#' @param .show Whether the answer's text is written to the console as it
#'   comes (default: in an interactive session).
#' @return A stream (invisibly): `$value` (what the call returned), `$error`
#'   (the error it ended with, or `NULL`), `$events` (every event, in order);
#'   `ai_text(stream)` is the answer's text.
#' @examples
#' \dontrun{
#' s <- ai_stream(reply, "Where is my parcel?")
#' s$value
#' ai_stream(reply, "Hi", .each = function(e) if (e$kind == "text") message(e$text))
#' }
#' @export
ai_stream <- function(fn, ..., .each = NULL, .show = interactive()) {
  if (!is.function(fn)) cli::cli_abort("{.arg fn} is an AI function or a program")
  if (!is.null(.each) && !is.function(.each)) cli::cli_abort("{.arg .each} is a function given each event, or NULL")
  s <- new.env(parent = emptyenv())
  s$events <- list(); s$calls <- character(0); s$closed <- FALSE; s$open <- FALSE
  shown <- if (isTRUE(.show)) function(e) {
    if (e$kind == "text" && isTRUE(e$answer) && e$call %in% s$calls) { cat(e$text); s$open <- TRUE; utils::flush.console() }
    else if (e$kind == "retry" && e$call %in% s$calls && s$open) { cat("\n\u21bb ", e$reason %||% "asked again", "\n", sep = ""); s$open <- FALSE }
  }
  s$each <- if (is.null(shown)) .each else if (is.null(.each)) shown else function(e) { shown(e); .each(e) }
  old <- the$stream_opening
  the$stream_opening <- s
  on.exit(the$stream_opening <- old)
  out <- tryCatch(list(value = fn(...)), error = identity)
  the$stream_opening <- old
  if (s$open) cat("\n")
  s$value <- if (inherits(out, "condition")) NULL else out$value
  s$error <- if (inherits(out, "condition")) out else NULL
  invisible(structure(s, class = "functai_stream"))
}

#' @rdname ai_stream
#' @export
stop_stream <- function() invisible(signalCondition(structure(class = c("functai_stop_stream", "condition"), list(message = "the stream was closed", call = NULL))))

#' @rdname ai_stream
#' @param stream A stream from `ai_stream()`.
#' @export
ai_text <- function(stream) {
  st <- replay_log(stream$events)
  vapply(stream$calls, function(id) {
    started <- Filter(function(e) e$kind == "started" && e$call == id, stream$events)
    answer <- if (length(started)) started[[1L]]$program$answer %||% "result" else "result"
    st$calls[[id]]$fields[[answer]] %||% ""
  }, "", USE.NAMES = FALSE)
}

#' @export
print.functai_stream <- function(x, ...) {
  kinds <- table(vapply(x$events, function(e) e$kind, ""))
  cat(sprintf("<stream> %d events (%s)\n", length(x$events), paste(sprintf("%s %d", names(kinds), kinds), collapse = ", ")))
  if (!is.null(x$error)) cat("error: ", conditionMessage(x$error), "\n", sep = "")
  else { cat("value: "); utils::str(x$value, give.head = FALSE) }
  invisible(x)
}

# A call opened while a stream is opening is watched by it: every event of the
# call and of the calls inside it.
stream_attach <- function(call) {
  s <- the$stream_opening
  if (is.null(s)) return(invisible())
  t <- call$tree
  # a call inside a tree already watched by this stream needs nothing more
  if (any(vapply(t$streams, function(x) identical(x$sink, s), NA))) return(invisible())
  s$calls <- c(s$calls, call$id)
  t$streams[[length(t$streams) + 1L]] <- list(call = call$id, sink = s, last = NULL)
  invisible()
}

# Whether `call` is `watched` or a call inside it.
inside <- function(t, call, watched) {
  id <- call
  while (!is.null(id)) { if (identical(id, watched)) return(TRUE); id <- t$parents[[id]] }
  FALSE
}

# Give a stream an event of a call it watches, `after` set to the event before
# it in the stream; a stream closed from its `.each` cancels the call it watches.
stream_offer <- function(t, i, e) {
  st <- t$streams[[i]]
  if (!inside(t, e$call, st$call)) return(invisible())
  linked <- relinked(e, st$last)
  t$streams[[i]]$last <- event_position(linked)
  s <- st$sink
  s$events[[length(s$events) + 1L]] <- linked
  if (!is.null(s$each) && !s$closed) {
    withCallingHandlers(
      tryCatch(with_no_stream(s$each(linked)), error = function(err) {
        s$closed <- TRUE
        warn_once("stream-each", sprintf("a stream's .each failed (%s); the stream is closed, and its call cancelled", conditionMessage(err)))
        t$cancelled <- c(t$cancelled, st$call)
      }),
      functai_stop_stream = function(c) { s$closed <- TRUE; t$cancelled <- c(t$cancelled, st$call) })
  }
  invisible()
}

# Code a stream's `.each` runs is not a step of the call it watches, nor watched.
with_no_stream <- function(code) {
  old <- list(current = the$current, opening = the$stream_opening)
  the$current <- NULL; the$stream_opening <- NULL
  on.exit({ the$current <- old$current; the$stream_opening <- old$opening })
  force(code)
}
