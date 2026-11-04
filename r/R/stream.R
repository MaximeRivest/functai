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

# ---------------------------------------------------------------- the answer so far, read by its shape

PARTIAL_SPACE <- c(" ", "\t", "\n", "\r")
PARTIAL_HEX <- strsplit("0123456789abcdefABCDEF", "")[[1L]]
PARTIAL_ESCAPES <- c('"' = '"', "\\" = "\\", "/" = "/", b = "\b", f = "\f", n = "\n", r = "\r", t = "\t")
PARTIAL_NUMBER <- "^-?(0|[1-9][0-9]*)(\\.[0-9]+)?([eE][+-]?[0-9]+)?$"
not_json <- function() stop(structure(class = c("functai_not_json", "error", "condition"), list(message = "not JSON", call = NULL)))
NOT_YET <- structure(list(), class = "functai_not_yet")
is_not_yet <- function(x) inherits(x, "functai_not_yet")

# A shape as the reader follows it: $ref followed; an anyOf of one shape and
# null read as that shape; any other anyOf, or what it cannot follow, {}.
partial_resolve <- function(shape, root) {
  seen <- character(0)
  while (is.list(shape) && is_str(shape[["$ref"]])) {
    ref <- shape[["$ref"]]
    if (ref %in% seen || !startsWith(ref, "#/$defs/") || !is.list(root)) return(list())
    seen <- c(seen, ref)
    shape <- root[["$defs"]][[substring(ref, 9L)]] %||% list()
  }
  if (!is.list(shape)) return(list())
  options <- shape$anyOf
  if (is.list(options)) {
    is_null <- function(o) is.list(o) && length(o) == 1L && identical(o$type, "null")
    rest <- Filter(Negate(is_null), options)
    return(if (length(rest) == 1L && length(options) == 2L) partial_resolve(rest[[1L]], root) else list())
  }
  shape
}

partial_choice <- function(shape) has_key(shape, "enum") || has_key(shape, "const")

partial_member <- function(shape, key, root) {
  props <- shape$properties
  if (is.list(props) && has_key(props, key)) return(partial_resolve(props[[key]], root))
  extra <- shape$additionalProperties
  if (is.list(extra)) partial_resolve(extra, root) else list()
}

partial_item <- function(shape, i, root) {                    # i: the item's 0-based place
  prefix <- shape$prefixItems
  if (is.list(prefix) && i < length(prefix)) return(partial_resolve(prefix[[i + 1L]], root))
  items <- shape$items
  if (is.list(items)) partial_resolve(items, root) else list()
}

# (text so far, index after, closed) for the string whose quote is cs[i].
partial_string <- function(cs, i) {
  out <- character(0); n <- length(cs); i <- i + 1L
  while (i <= n) {
    c <- cs[[i]]
    if (c == '"') return(list(text = paste(out, collapse = ""), i = i + 1L, closed = TRUE))
    if (c != "\\") { out <- c(out, c); i <- i + 1L; next }
    if (i + 1L > n) break                                       # an escape still being written
    e <- cs[[i + 1L]]
    if (e %in% names(PARTIAL_ESCAPES)) { out <- c(out, PARTIAL_ESCAPES[[e]]); i <- i + 2L; next }
    if (e != "u") not_json()
    digits <- slice_of(cs, i + 2L, i + 5L)
    if (!all(digits %in% PARTIAL_HEX)) not_json()
    if (length(digits) < 4L) break
    code <- strtoi(paste(digits, collapse = ""), 16L)
    if (code >= 0xD800 && code < 0xDC00) {                       # a pair: wait for its second half
      rest <- slice_of(cs, i + 6L, i + 11L)
      if (length(rest) < 6L) {
        head <- paste(rest[seq_len(min(2L, length(rest)))], collapse = "")
        if ((length(rest) && !startsWith("\\u", head)) || !all(rest[-seq_len(min(2L, length(rest)))] %in% PARTIAL_HEX)) not_json()
        break
      }
      if (paste(rest[1:2], collapse = "") != "\\u" || !all(rest[3:6] %in% PARTIAL_HEX)) not_json()
      low <- strtoi(paste(rest[3:6], collapse = ""), 16L)
      if (low < 0xDC00 || low >= 0xE000) not_json()
      out <- c(out, intToUtf8(0x10000 + (code - 0xD800) * 1024 + (low - 0xDC00)))
      i <- i + 12L; next
    }
    out <- c(out, intToUtf8(code))
    i <- i + 6L
  }
  list(text = paste(out, collapse = ""), i = n + 1L, closed = FALSE)
}

slice_of <- function(cs, a, b) { b <- min(b, length(cs)); if (a > b) character(0) else cs[a:b] }

partial_skip <- function(cs, i) { while (i <= length(cs) && cs[[i]] %in% PARTIAL_SPACE) i <- i + 1L; i }

# list(value or NOT_YET, i after, done)
partial_value <- function(cs, i, shape, root) {
  n <- length(cs)
  i <- partial_skip(cs, i)
  if (i > n) return(list(value = NOT_YET, i = i, done = FALSE))
  c <- cs[[i]]
  if (c == '"') {
    s <- partial_string(cs, i)
    if (!s$closed && partial_choice(shape)) return(list(value = NOT_YET, i = s$i, done = FALSE))   # "med" is no value of a choice
    return(list(value = s$text, i = s$i, done = s$closed))
  }
  if (c == "{") {
    obj <- lmcc::jobj(); i <- i + 1L
    repeat {
      i <- partial_skip(cs, i)
      if (i > n) return(list(value = obj, i = i, done = FALSE))
      if (cs[[i]] == "}") return(list(value = obj, i = i + 1L, done = TRUE))
      if (cs[[i]] == ",") { i <- i + 1L; next }
      if (cs[[i]] != '"') not_json()
      k <- partial_string(cs, i); i <- k$i
      if (!k$closed) return(list(value = obj, i = i, done = FALSE))
      i <- partial_skip(cs, i)
      if (i > n) return(list(value = obj, i = i, done = FALSE))
      if (cs[[i]] != ":") not_json()
      v <- partial_value(cs, i + 1L, partial_member(shape, k$text, root), root); i <- v$i
      if (!is_not_yet(v$value)) obj[k$text] <- list(v$value)
      if (!v$done) return(list(value = obj, i = i, done = FALSE))
    }
  }
  if (c == "[") {
    items <- list(); i <- i + 1L
    repeat {
      i <- partial_skip(cs, i)
      if (i > n) return(list(value = items, i = i, done = FALSE))
      if (cs[[i]] == "]") return(list(value = items, i = i + 1L, done = TRUE))
      if (cs[[i]] == ",") { i <- i + 1L; next }
      v <- partial_value(cs, i, partial_item(shape, length(items), root), root); i <- v$i
      if (!is_not_yet(v$value)) items[length(items) + 1L] <- list(v$value)
      if (!v$done) return(list(value = items, i = i, done = FALSE))
    }
  }
  j <- i
  while (j <= n && !cs[[j]] %in% c(",", "]", "}", PARTIAL_SPACE)) j <- j + 1L
  if (j > n) return(list(value = NOT_YET, i = j, done = FALSE))   # a number or a word that may still grow: 1 may become 12
  word <- paste(cs[i:(j - 1L)], collapse = "")
  if (word == "true") return(list(value = TRUE, i = j, done = TRUE))
  if (word == "false") return(list(value = FALSE, i = j, done = TRUE))
  if (word == "null") return(list(value = NULL, i = j, done = TRUE))
  if (!grepl(PARTIAL_NUMBER, word)) not_json()
  list(value = if (grepl("[.eE]", word)) as.numeric(word) else { x <- as.numeric(word); if (abs(x) <= .Machine$integer.max) as.integer(x) else x },
       i = j, done = TRUE)
}

partial_kind <- function(shape) {
  s <- partial_resolve(shape, shape)
  t <- s$type
  types <- setdiff(if (is.null(t)) "<none>" else unlist(t), "null")
  if (identical(types, "string") && !partial_choice(s)) return("text")
  if (length(types) && all(types %in% c("object", "array"))) return("json")
  "none"
}

partial_json <- function(text, shape = NULL) {
  s <- gsub("^[ \t\n\r]+|[ \t\n\r]+$", "", text)
  if (startsWith(s, "```")) {                                   # a fenced block: read after its first line break
    k <- regexpr("\n", s, fixed = TRUE)
    s <- if (k < 0) "" else substring(s, k + 1L)
    r <- sub("[ \t\n\r]+$", "", s)
    if (endsWith(r, "```")) s <- substr(r, 1L, nchar(r) - 3L)
    s <- gsub("^[ \t\n\r]+|[ \t\n\r]+$", "", s)
  }
  top <- if (is.null(shape)) list() else partial_resolve(shape, shape)
  t <- unlist(top$type)
  opens <- paste(c(if ("object" %in% t) "{", if ("array" %in% t) "["), collapse = "")
  if (!nzchar(opens)) opens <- "[{"
  if (!nzchar(s) || !grepl(substr(s, 1L, 1L), opens, fixed = TRUE)) return(NULL)
  v <- tryCatch(partial_value(strsplit(s, "")[[1L]], 1L, top, shape)$value, functai_not_json = function(e) NOT_YET)
  if (is_not_yet(v)) NULL else v
}

#' The answer so far
#'
#' The answer a call is writing, read from its text by its shape
#' (contract/streaming.md, "The answer so far"): the text for a text answer;
#' the JSON read so far for a record or a list (complete values; a string
#' still being written as its text so far, unless it is a choice, which waits
#' for its closing quote: never `"med"` on its way to `"medium"`; a number,
#' `TRUE`, `FALSE` or `NULL` once a character after it ends it); `NULL` for
#' any other answer, and when the text is not the start of what the shape
#' gives. Provisional: the call's value is the typed, checked one. In a
#' stream's `.each`, collect the answer's `text` events and read them so; a
#' page that knows the program reads a call's `text` events the same way.
#' @param text The answer's text so far.
#' @param shape The answer's shape: an AI function (its answer's), or a JSON
#'   Schema as a list.
#' @return A value, or `NULL`.
#' @examples
#' shape <- list(type = "object", properties = list(summary = list(type = "string"),
#'                                                  severity = list(enum = list("low", "medium", "high"))))
#' answer_so_far('{"summary": "Checkout hi', shape)
#' answer_so_far('{"summary": "x", "severity": "med', shape)
#' @export
answer_so_far <- function(text, shape) {
  if (inherits(shape, "functai_fn")) { iface <- interface_of(core_of(shape)); shape <- iface$outputs[[length(iface$outputs)]]$shape }
  kind <- partial_kind(shape)
  if (kind == "text") return(text)
  if (kind == "none") return(NULL)
  partial_json(text, shape)
}

# ---------------------------------------------------------------- notes (contract/streaming.md, "Notes")

#' Say what happened, from code inside a call
#'
#' From a program's code or a tool: `ai_note("screen captured", list(screen =
#' 4))`. Whatever watches the call sees a `note` event (a stream's `.each`,
#' an observer, a journal), in its place among the call's events. `data` is
#' any JSON value; the kept form of a call whose content is not whole keeps
#' the note's `name` and leaves `data` out. A note is not a value of the
#' call: the call log does not keep it. Outside any call it does nothing. A
#' resumed turn does not show again the notes it made before it stopped.
#'
#' A program may declare its notes (`ai_program(..., .notes = )`,
#' `ai_tool(..., .notes = )`: [note_type()]). Then, inside its calls, a note
#' it does not declare, or whose data does not fit what a program around it
#' declares, refuses `interface-note` here, whether or not anything watches.
#' @param name What happened: a non-empty text.
#' @param data Any JSON value, or `NULL`.
#' @return Nothing, invisibly.
#' @examples
#' ai_note("nothing watches")       # outside any call: nothing happens
#' @export
ai_note <- function(name, data = NULL) {
  if (!is_str(name) || !nzchar(trimws(name))) cli::cli_abort("a note's name is a non-empty text: what happened ({.val screen captured})")
  call <- the$current
  if (is.null(call)) return(invisible())
  chain <- list(); x <- call
  while (!is.null(x)) { chain <- c(list(declared_notes(x$core)), chain); x <- x$up }
  if (any(!vapply(chain, is.null, NA))) {                     # declared somewhere: checked (contract/programs.md, Notes)
    why <- if (!has_json_form(data)) "its data has no JSON form" else note_refusal(chain, name, plain_json(data))
    if (!is.null(why)) refuse("interface-note", c("{call$name}: the note {.val {name}} is refused: {why}"), field = name)
    data <- plain_json(data)
  }
  d <- list(name = name); d["data"] <- list(value_json(data))
  call_emit(call, "note", d)
  invisible()
}
