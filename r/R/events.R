# A call tree's events as data (contract/streaming.md, format 2): one log per
# tree, numbered by its writer; the kept form of a log (what may be kept
# outside the process, by each call's log_content); replaying a form,
# resuming it, following it live across writers; and the rules a store of
# logs keeps (memory_store()). Streams, observers and journals (journal.R)
# are made of these.
#
# An event is its JSON form, a named list in the contract's order, with the
# class `functai_event`: `e$kind`, `e$text`, `e$value`, ... read its keys.

EVENT_FORMAT <- 2L
EVENT_KINDS <- c("started", "request", "text", "thinking", "tool_call", "tool_result", "retry", "done", "failed",
                 "approval", "approved")
ENVELOPE <- c("functai_event", "kind", "tree", "writer", "seq", "after", "at", "call", "function")
KIND_KEYS <- list(
  started = c("parent", "root", "program", "inputs", "content", "omitted", "saw", "invocation"),
  request = c("request", "model"), text = c("field", "answer", "text"), thinking = "text",
  tool_call = c("id", "name", "input", "content", "invocation"), tool_result = c("id", "name", "output", "content", "invocation"),
  retry = c("reason", "wait", "content"), done = c("value", "content"), failed = c("error", "content"),
  approval = c("id", "invocation", "name", "input", "effects", "path", "to", "plugin", "question", "content"),
  approved = c("id", "invocation", "verdict", "by", "reason", "plugin", "content"))

# ---------------------------------------------------------------- positions

# A position names an event: its writer and its seq (NULL: before the first).
position <- function(writer, seq) list(writer = as.integer(writer), seq = as.integer(seq))
is_count <- function(x) is.numeric(x) && length(x) == 1L && !is.na(x) && is.finite(x) && x == round(x) && x >= 1 && x <= .Machine$integer.max
position_of <- function(x) {
  if (is.null(x)) return(NULL)
  if (!is.list(x) || !setequal(names(x), c("writer", "seq")) || length(x) != 2L || !is_count(x$writer) || !is_count(x$seq))
    stop(sprintf("a position is {\"writer\", \"seq\"} (whole numbers from 1) or null, not %s", short_json(x)), call. = FALSE)
  position(x$writer, x$seq)
}
same_position <- function(a, b) if (is.null(a) || is.null(b)) is.null(a) && is.null(b) else a$writer == b$writer && a$seq == b$seq
event_position <- function(e) position(e$writer, e$seq)
position_text <- function(p) if (is.null(p)) "the start" else sprintf("%d/%d", p$writer, p$seq)

# ---------------------------------------------------------------- an event

new_event <- function(kind, tree, writer, seq, after, at, call, fn, data = list()) {
  e <- list(functai_event = EVENT_FORMAT, kind = kind, tree = tree, writer = as.integer(writer), seq = as.integer(seq))
  e["after"] <- list(after)
  e$at <- at; e$call <- call; e[["function"]] <- fn
  for (k in names(data)) e[k] <- list(data[[k]])
  structure(e, class = "functai_event")
}

#' @export
print.functai_event <- function(x, ...) {
  data <- unclass(x)[setdiff(names(x), ENVELOPE)]
  shown <- vapply(names(data), function(k) paste0(k, " = ", short_json(data[[k]])), "")
  cat(sprintf("<event %s %d/%d> %s%s\n", x$kind, x$writer, x$seq, x[["function"]], if (length(shown)) paste0(": ", paste(shown, collapse = ", ")) else ""))
  invisible(x)
}

event_data <- function(e) unclass(e)[setdiff(names(e), ENVELOPE)]
with_data <- function(e, data) new_event(e$kind, e$tree, e$writer, e$seq, e$after, e$at, e$call, e[["function"]], data)
relinked <- function(e, after) { e["after"] <- list(after); e }
event_json <- function(e) unclass(e)

# Why a JSON object is not an event of format 2 (the contract's schema), or NULL.
event_fault <- function(d) {
  if (!is.list(d) || is.null(names(d))) return("not an object")
  if (!is_num(d$functai_event) || num(d$functai_event) != EVENT_FORMAT) return(sprintf("not format %d", EVENT_FORMAT))
  schema_fault(unclass(d), "event.schema.json")
}

unknown_format <- function(fmt) rlang::error_cnd(c("functai_unknown_format"), format = fmt,
  message = sprintf("an event of format %s; this reader knows format %d", short_json(fmt), EVENT_FORMAT))

# An event read from its JSON form: an error of class functai_unknown_format
# for a format this reader does not know, else an error for an object that is
# not an event of format 2.
as_event <- function(d) {
  if (inherits(d, "functai_event")) return(d)
  if (!is.list(d) || !is_num(d$functai_event) || num(d$functai_event) != EVENT_FORMAT) stop(unknown_format(d$functai_event))
  fault <- event_fault(d)
  if (!is.null(fault)) stop(sprintf("not an event of format %d: %s", EVENT_FORMAT, fault), call. = FALSE)
  data <- d[setdiff(names(d), ENVELOPE)]
  new_event(d$kind, d$tree, as.integer(d$writer), as.integer(d$seq), position_of(d$after), d$at, d$call, d[["function"]], data)
}

# The events of a form, `after` set to the event before each in the form.
relink <- function(events, last = NULL) {
  out <- vector("list", length(events))
  for (i in seq_along(events)) { out[[i]] <- relinked(events[[i]], last); last <- event_position(events[[i]]) }
  out
}

# ---------------------------------------------------------------- replaying

# What a watcher of a form was shown (contract/streaming.md, "Replaying"):
# for each call started so far, whether it ended and each field's text so far.
new_log_state <- function() list(tree = NULL, calls = list(), finished = FALSE)

replay_event <- function(s, e) {
  if (is.null(s$tree)) s$tree <- e$tree
  if (!e$kind %in% EVENT_KINDS) return(s)
  if (e$kind == "started") { s$calls[[e$call]] <- list(ended = NULL, fields = list()); return(s) }
  c <- s$calls[[e$call]]
  if (is.null(c)) stop(sprintf("event %s: its call's started is not in this form", position_text(event_position(e))), call. = FALSE)
  if (e$kind %in% c("request", "retry")) c$fields <- list()
  else if (e$kind == "text") c$fields[[e$field]] <- paste0(c$fields[[e$field]] %||% "", e$text)
  else if (e$kind %in% c("done", "failed")) { c["ended"] <- list(e$kind); if (identical(e$call, e$tree)) s$finished <- TRUE }
  s$calls[[e$call]] <- c
  s
}

replay_log <- function(events) Reduce(replay_event, lapply(events, as_event), new_log_state())

state_json <- function(s) list(calls = if (length(s$calls)) lapply(s$calls, function(c) list(ended = c$ended, fields = if (length(c$fields)) c$fields else lmcc::jobj())) else lmcc::jobj(),
                                finished = s$finished)

# A store refusing an append, a claim or a read: `code` the contract's,
# `event` the position of the event refused.
store_refusal <- function(code, message = "", event = NULL) {
  rlang::error_cnd(c(paste0("functai_", gsub("-", "_", code)), "functai_store_refusal", "functai_refusal"),
                   code = code, event = event, message = sprintf("[%s]%s%s", code, if (is.null(event)) "" else paste0(" at ", position_text(event)),
                                                                 if (nzchar(message)) paste0(": ", message) else ""))
}

# What a source holding these events gives a reader that has events up to
# `after`: the events after it (event-unknown when it has no such event).
resume_events <- function(events, after) {
  evs <- lapply(events, as_event)
  if (is.null(after)) return(evs)
  p <- position_of(after)
  i <- Position(function(e) same_position(event_position(e), p), evs)
  if (is.na(i)) stop(store_refusal("event-unknown", sprintf("this source has no event %s", position_text(p))))
  evs[seq_len(length(evs) - i) + i]
}

# ---------------------------------------------------------------- following

#' Follow call trees' logs live
#'
#' A reader that follows logs as they come (contract/streaming.md,
#' "Following a log"): a page, another process. Give it each event with
#' `follow_event()`; it says what the event was: `"kept"` (the next one),
#' `"duplicate"` or `"stale"` (dropped), `"rewind"` (a later writer went on
#' from an event it holds), `"loss"` (events were lost: read again with
#' `follow_recover()`), `"malformed"`, or `"unknown-format"` (it stops).
#' `follow_state()` is what it shows of a tree.
#' @param form `"kept"` (the kept form, or a view made from it: the same from
#'   every source) or `"live"` (a form that may show values the kept form
#'   lacks: only the process that gave them can give them again).
#' @return A follower (`follower()`), a word (`follow_event()`), the replay
#'   (`follow_state()`: each call's `ended` and `fields`, and `finished`), or
#'   the reads made (`follow_recover()`).
#' @export
follower <- function(form = c("kept", "live")) {
  f <- new.env(parent = emptyenv())
  f$form <- match.arg(form); f$trees <- list(); f$stopped <- FALSE
  structure(f, class = "functai_follower")
}

#' @rdname follower
#' @param f A follower.
#' @param event An event (or its JSON form).
#' @param from The process that gave it (a writer number), or `"store"`.
#' @export
follow_event <- function(f, event, from = NULL) {
  if (f$stopped) cli::cli_abort("this follower stopped at an event of a format it does not know")
  if (!inherits(event, "functai_event") && !(is.list(event) && is_num(event$functai_event) && num(event$functai_event) == EVENT_FORMAT)) {
    f$stopped <- TRUE
    return("unknown-format")
  }
  if (!inherits(event, "functai_event") && !is.null(event_fault(event))) return("malformed")
  e <- tryCatch(as_event(event), error = function(err) NULL)
  if (is.null(e)) return("malformed")
  t <- f$trees[[e$tree]] %||% list(held = list(), from = list())
  last <- if (length(t$held)) event_position(t$held[[length(t$held)]]) else NULL
  writer <- if (is.null(last)) 0L else last$writer
  if (e$writer < writer) return("stale")
  if (!is.null(last) && e$writer == writer && e$seq <= last$seq) return("duplicate")
  held_at <- function(p) Position(function(h) same_position(event_position(h), p), t$held)
  result <- if (same_position(e$after, last)) "kept"
    else if (e$writer > writer && (is.null(e$after) || !is.na(held_at(e$after)))) {
      i <- if (is.null(e$after)) 0L else held_at(e$after)
      t$held <- t$held[seq_len(i)]; t$from <- t$from[seq_len(i)]
      "rewind"
    } else return("loss")
  t$held[[length(t$held) + 1L]] <- e
  t$from[[length(t$from) + 1L]] <- from %||% e$writer
  f$trees[[e$tree]] <- t
  result
}

#' @rdname follower
#' @param tree A tree's id (its outermost call's).
#' @export
follow_state <- function(f, tree) state_json(replay_log(f$trees[[tree]]$held %||% list()))

#' @rdname follower
#' @param source What gives the events again: a store, or a list of events.
#' @export
follow_recover <- function(f, tree, source, from = "store") {
  t <- f$trees[[tree]] %||% list(held = list(), from = list())
  reads <- list(); got <- NULL
  in_place <- identical(f$form, "kept") || (!identical(from, "store") && all(vapply(t$from, function(x) identical(as.character(x), as.character(from)), NA)))
  after_of <- function(src, after) if (is_event_store(src)) store_read(src, tree, after) else resume_events(src, after)
  if (in_place) {
    after <- if (length(t$held)) event_position(t$held[[length(t$held)]]) else NULL
    got <- tryCatch(after_of(source, after), functai_event_unknown = function(e) e)
    reads[[length(reads) + 1L]] <- list(after = after, answer = got)
  }
  if (is.null(got) || inherits(got, "condition")) {
    got <- after_of(source, NULL)
    reads[[length(reads) + 1L]] <- list(after = NULL, answer = got)
    t$held <- list(); t$from <- list()
  }
  t$held <- c(t$held, got); t$from <- c(t$from, rep(list(from), length(got)))
  f$trees[[tree]] <- t
  invisible(reads)
}

# ---------------------------------------------------------------- the kept form

ERROR_KEYS <- c("type", "message", "code")
PROGRAM_KEYS <- c("name", "kind", "module", "version", "signature", "interface", "answer", "saved", "file", "line")
SAW_KEYS <- c("call", "steps", "without", "slot", "saw_of")

# What a form maker keeps of an event of a kind it knows: the keys it knows,
# and inside the objects it knows, their known members.
known_event <- function(e) {
  data <- event_data(e)
  data <- data[names(data) %in% KIND_KEYS[[e$kind]]]
  if (is.list(data$error) && !is.null(names(data$error))) data$error <- data$error[names(data$error) %in% ERROR_KEYS]
  if (is.list(data$program) && !is.null(names(data$program))) data$program <- data$program[names(data$program) %in% PROGRAM_KEYS]
  if (is.list(data$saw) && is.null(names(data$saw)))
    data$saw <- lapply(data$saw, function(x) if (is.list(x) && length(x) && all(names(x) %in% SAW_KEYS)) x else lmcc::jobj())
  with_data(e, data)
}

# Which fields of a call are kept: `inputs` and `outputs` by name, TRUE/FALSE;
# `holds` the outputs a done event's value holds, when the writer knows.
new_keep <- function(inputs, outputs, holds = NULL) list(inputs = as.list(inputs), outputs = as.list(outputs), holds = holds)
keep_of <- function(fields, kept) new_keep(as.list(kept[fields$inputs]), as.list(kept[fields$outputs]))
keep_whole <- function(keep) all(unlist(c(keep$inputs, keep$outputs)) %in% TRUE)

value_holds <- function(keep, program) {
  if (!is.null(keep$holds)) return(keep$holds)
  if (identical(program$kind %||% "ai", "ai")) program$answer else names(keep$outputs)
}

error_without_content <- function(err) if (is.list(err)) err[names(err) %in% c("type", "code")] else err

# The kept form of one event (contract/streaming.md, "The kept form"), given
# which fields of its call are kept and its call's program; NULL: not kept.
kept_event <- function(e, keep, program) {
  if (!e$kind %in% EVENT_KINDS) return(NULL)
  out <- known_event(e)
  if (keep_whole(keep)) return(out)
  data <- event_data(out)
  kind <- e$kind
  if (kind == "started") {
    inputs <- data$inputs %||% list()
    inputs <- inputs[vapply(names(inputs), function(k) isTRUE(keep$inputs[[k]]), NA)]
    order <- names(event_data(e)); order <- append(order, "omitted", after = match("content", order, nomatch = length(order)))
    rest <- list()
    for (k in names(data)) {
      if (k == "inputs") next
      if (k == "content") {
        rest$content <- FALSE
        rest$omitted <- list(inputs = as.list(names(keep$inputs)[!unlist(keep$inputs)]), outputs = as.list(names(keep$outputs)[!unlist(keep$outputs)]))
      } else rest[k] <- list(data[[k]])
    }
    if (length(inputs)) rest$inputs <- inputs
    rest <- rest[order(match(names(rest), order, nomatch = length(order) + 1L))]
    return(with_data(out, rest))
  }
  if (kind == "request") return(out)
  if (kind == "text") return(if (isTRUE(keep$outputs[[data$field]])) out else NULL)
  if (kind == "thinking") return(NULL)
  if (kind %in% c("tool_call", "approval")) { data$input <- NULL; data$question <- NULL }
  else if (kind == "approved") data$reason <- NULL
  else if (kind == "tool_result") data$output <- NULL
  else if (kind == "retry") data$reason <- NULL
  else if (kind == "done") {
    if (all(vapply(value_holds(keep, program), function(k) isTRUE(keep$outputs[[k]]), NA))) return(out)
    data$value <- NULL
    data["value"] <- NULL
  } else if (kind == "failed") data$error <- error_without_content(data$error)
  data$content <- FALSE
  with_data(out, data)
}

# The kept form of a whole log: each event's kept form, by its call's keep
# (`keeps[[call]]`), with `after` set to the kept event before it.
kept_log <- function(events, keeps) {
  evs <- lapply(events, as_event)
  programs <- list()
  for (e in evs) if (e$kind == "started") programs[[e$call]] <- e$program
  relink(Filter(Negate(is.null), lapply(evs, function(e) kept_event(e, keeps[[e$call]], programs[[e$call]] %||% list()))))
}

# ---------------------------------------------------------------- a store of logs

#' A store of call trees' logs
#'
#' What keeps logs for others to read, by the rules every store keeps
#' (contract/streaming.md, "The rules a store keeps"). `memory_store()`
#' keeps them in this R session. A store of your own is a list (or an
#' environment) of three functions:
#'
#' * `append(events)`: append one event, or a list of events as one step
#'   (kept whole or not at all). Returns `"kept"` or `"duplicate"`, or
#'   raises the contract's refusal with [store_refuse()]. Any other error is
#'   "no answer": the append may have been kept, and the writer sends it
#'   again.
#' * `claim(tree)`: a later writer claims an unfinished log: `list(writer,
#'   after)`.
#' * `read(tree, after = NULL)`: the kept events after a position (all of
#'   them after `NULL`); refuses `event-unknown` when it has no such event.
#'
#' Every event is checked against the contract's event schema first.
#' @param name A name, for messages.
#' @return A store.
#' @examples
#' store <- memory_store()
#' # with_ai_config(team(message), journal = ai_journal(store, required = TRUE))
#' @export
memory_store <- function(name = "memory") {
  s <- new.env(parent = emptyenv())
  s$name <- name; s$logs <- list(); s$writers <- list()
  s$append <- function(events) memory_append(s, events)
  s$claim <- function(tree) memory_claim(s, tree)
  s$read <- function(tree, after = NULL) resume_events(s$logs[[tree]] %||% list(), after)
  s$trees <- function() names(s$logs)
  s$writer_of <- function(tree) s$writers[[tree]] %||% 1L
  s$finished <- function(tree) is_log_end(s$logs[[tree]] %||% list(), tree)
  structure(s, class = c("functai_memory_store", "functai_event_store"))
}

#' @export
print.functai_memory_store <- function(x, ...) { cat(sprintf("<memory store %s> %d logs\n", x$name, length(x$logs))); invisible(x) }

#' @rdname memory_store
#' @param code The refusal's code: `"event-malformed"`, `"event-conflict"`,
#'   `"event-gap"`, `"event-after-end"`, `"event-start"` or `"event-unknown"`.
#' @param message Why, in words.
#' @param event The position of the event refused, `list(writer, seq)`.
#' @export
store_refuse <- function(code, message = "", event = NULL) stop(store_refusal(code, message, event))

is_event_store <- function(x) (is.environment(x) || is.list(x)) && is.function(x$append) && is.function(x$read)

store_read <- function(store, tree, after = NULL) lapply(store$read(tree, after), as_event)
store_trees <- function(store) if (is.function(store$trees)) store$trees() else character(0)
store_finished <- function(store, tree) if (is.function(store$finished)) store$finished(tree) else is_log_end(store_read(store, tree), tree)

is_log_end <- function(log, tree) length(log) > 0L && identical(log[[length(log)]]$call, tree) && log[[length(log)]]$kind %in% c("done", "failed")

memory_claim <- function(s, tree) {
  log <- s$logs[[tree]] %||% list()
  if (!length(log)) store_refuse("event-unknown", sprintf("no log %s", tree))
  if (is_log_end(log, tree)) store_refuse("event-after-end", sprintf("the log %s is finished", tree))
  s$writers[[tree]] <- (s$writers[[tree]] %||% 1L) + 1L
  list(writer = s$writers[[tree]], after = event_position(log[[length(log)]]))
}

stated_position <- function(x) if (is.list(x) && is_count(x$writer) && is_count(x$seq)) position(x$writer, x$seq) else NULL

checked_event <- function(x) {
  json <- if (inherits(x, "functai_event")) unclass(x) else x
  fault <- event_fault(json)
  if (!is.null(fault)) store_refuse("event-malformed", sprintf("not an event of format %d: %s", EVENT_FORMAT, fault), stated_position(json))
  tryCatch(as_event(json), error = function(e) store_refuse("event-malformed", conditionMessage(e), stated_position(json)))
}

# One append, checked in the contract's order; returns the answer and the log.
append_one <- function(log, writer, x, tree = NULL) {
  e <- checked_event(x)
  p <- event_position(e)
  if (!is.null(tree) && !identical(e$tree, tree)) store_refuse("event-malformed", "an event of another log", p)
  a <- e$after
  if (!is.null(a) && (e$seq <= a$seq || a$writer > e$writer)) store_refuse("event-malformed", "its after is not an event before it", p)
  if (length(log) && e$writer != writer)
    store_refuse("event-conflict", sprintf("writer %d is not the log's (writer %d): fenced, or a number never given", e$writer, writer), p)
  i <- Position(function(k) k$seq == e$seq, log)
  if (!is.na(i)) {
    if (identical(lmcc::canonical_json(unclass(log[[i]])), lmcc::canonical_json(unclass(e)))) return(list(answer = "duplicate", log = log))
    store_refuse("event-conflict", sprintf("another event holds seq %d: two writers are writing one log", e$seq), p)
  }
  if (is_log_end(log, e$tree)) store_refuse("event-after-end", "the log is finished", p)
  last <- if (length(log)) event_position(log[[length(log)]]) else NULL
  if (!same_position(a, last)) {
    if ((if (is.null(a)) 0L else a$seq) > (if (is.null(last)) 0L else last$seq)) store_refuse("event-gap", "events are missing before it", p)
    store_refuse("event-conflict", "the log went on another way", p)
  }
  if (!length(log) && (e$kind != "started" || !identical(e$call, e$tree) || e$writer != 1L))
    store_refuse("event-start", "a log starts with its outermost call's started, from writer 1", p)
  list(answer = "kept", log = c(log, list(e)))
}

memory_append <- function(s, x) {
  batch <- is.list(x) && !inherits(x, "functai_event") && is.null(names(x))
  xs <- if (batch) x else list(x)
  if (!length(xs)) return("duplicate")
  first <- xs[[1L]]
  tree <- if (is.list(first)) first$tree else NULL
  if (!is_str(tree)) store_refuse("event-malformed", "not an event")
  log <- s$logs[[tree]] %||% list()
  writer <- s$writers[[tree]] %||% 1L
  answers <- character(0)
  for (one in xs) {
    r <- tryCatch(append_one(log, writer, one, if (batch) tree else NULL), functai_store_refusal = function(e) {
      if (batch && is.null(e$event)) e$event <- stated_position(if (inherits(one, "functai_event")) unclass(one) else one)
      stop(e)
    })
    log <- r$log; answers <- c(answers, r$answer)
  }
  if (length(log)) s$logs[[tree]] <- log
  if (all(answers == "duplicate")) "duplicate" else "kept"
}

#' What a journal kept of an end it could not confirm
#'
#' A call whose required journal did not answer about its end raises a
#' `functai_journal_error` with code `journal-end` and `journal = "unknown"`;
#' `settle()` reads the journal: `"kept"` (the log holds that end),
#' `"another-end"` (another writer ended the log: the outcome was not kept),
#' or `"not-kept"` (the log is unfinished; final only once the caller has
#' claimed the log).
#' @param err The `functai_journal_error`.
#' @return A string.
#' @export
settle <- function(err) {
  if (!identical(err$code, "journal-end") || is.null(err$store)) cli::cli_abort("only a journal-end error names an end to settle")
  settle_log(err$store, err$tree, err$event)
}

settle_log <- function(store, tree, event) {
  log <- store_read(store, tree)
  if (any(vapply(log, function(e) same_position(event_position(e), event), NA))) return("kept")
  if (is_log_end(log, tree)) "another-end" else "not-kept"
}
