# Keeping a log while it is written (contract/streaming.md): observers and
# journals, set where log_calls is (a function's own settings, with_ai_config()
# blocks, ai_config()); how the layers combine (observers add up; one journal
# per tree, and the host's holds: `journal-policy`); the writer that appends a
# tree's kept events to its journal in order, sends again what is not
# confirmed, and, for a required journal, waits at the three barriers; and
# the per-tree log every stream, observer and journal is fed from.
#
# R runs one thing at a time: an observer and a journal's store run in the
# call's own process, as the event is made, in order. A slow observer or store
# slows the call (the contract allows handing events to another thread; R has
# none to hand them to), and a store that hangs holds the call.

#' Keep each call tree's log while it is written
#'
#' A journal keeps the kept form of every call tree whose outermost call
#' starts where it is set (`ai_config(journal = ...)`, `with_ai_config(...,
#' journal = ...)`, or an AI function's own `.journal`), in a store
#' ([memory_store()], or one of your own). **Best effort** (the default)
#' never stops a call: a store that fails is warned about once, and events it
#' did not answer are sent again with the next one. **Required** makes the
#' call wait until its start, each tool call that changes things, and its
#' end are confirmed kept: when its start or a tool call is not, the code or
#' the tool does not run (`journal-barrier`); when its end is not, the call
#' raises `journal-end` holding what it did (see [settle()]).
#'
#' A host's journal holds: a function's own setting cannot replace or remove
#' it, and nothing closer can weaken or remove a required one
#' (`journal-policy`, refused before the call runs). Observers add up across
#' layers (`observers = list(f)`); a host's `program_observers = FALSE`
#' gives a function's own observers nothing.
#' @param store A store: [memory_store()], or a list of `append`, `claim`
#'   and `read` functions.
#' @param required Whether calls wait for it (see above).
#' @param retries How many times an event is sent again in a row before the
#'   writer gives up on it for now.
#' @return A journal.
#' @export
ai_journal <- function(store, required = FALSE, retries = 2L) {
  if (!is_event_store(store)) cli::cli_abort("a journal keeps logs in a store: {.fn memory_store}, or a list of {.code append}, {.code claim} and {.code read} functions")
  if (!rlang::is_bool(required)) cli::cli_abort("{.arg required} is TRUE or FALSE")
  if (!is_count(retries + 1)) cli::cli_abort("{.arg retries} is a whole number of at least 0")
  structure(list(store = store, required = required, retries = as.integer(retries)), class = "functai_journal")
}

#' @export
print.functai_journal <- function(x, ...) { cat(sprintf("<journal%s> %s\n", if (x$required) ", required" else "", class(x$store)[[1L]])); invisible(x) }

journal_mode <- function(j) if (j$required) "required" else "best-effort"
same_journal <- function(a, b) {
  if (isFALSE(a) || isFALSE(b)) return(isFALSE(a) && isFALSE(b))
  identical(a$store, b$store) && identical(a$required, b$required)
}

# A journal setting: a journal, a store (best effort), FALSE (no journal) or
# NULL (none set here).
journal_setting <- function(v) {
  if (is.null(v) || isFALSE(v) || inherits(v, "functai_journal")) return(v)
  if (isTRUE(v)) cli::cli_abort("{.arg journal} is a store, {.code ai_journal(store, required = TRUE)}, or FALSE; not TRUE")
  if (is_event_store(v)) return(ai_journal(v))
  cli::cli_abort("{.arg journal} is a store, {.code ai_journal(store)}, or FALSE")
}

journal_error <- function(code, message, journal = NULL, tree = NULL, event = NULL, outcome = NULL, cause = NULL, store = NULL) {
  rlang::error_cnd(c("functai_journal_error", paste0("functai_", gsub("-", "_", code)), "functai_refusal"),
                   code = code, journal = journal, tree = tree, event = event, outcome = outcome, cause = cause, store = store,
                   message = message)
}

# ---------------------------------------------------------------- receivers across layers

# The layers around a program's call, closest first: its own settings, each
# with_ai_config() block from the closest out, then ai_config()'s.
receiver_layers <- function(own = list()) {
  layer <- function(where, s) list(where = where, observers = s$observers %||% list(),
                                   journal = if ("journal" %in% names(s)) s$journal else NULL,
                                   set = "journal" %in% names(s), program_observers = s$program_observers)
  c(list(layer("own", own)), lapply(rev(the$blocks %||% list()), layer, where = "block"), list(layer("configure", the$config %||% list())))
}

# The observers and the journal a tree gets from its layers (closest first):
# observers add up, outermost first, less those whose own code is making the
# call; the closest journal decides, except that a program's own setting
# cannot replace or remove a host's, and nothing closer can replace, weaken or
# remove a required one (`refused`: the journal is then the one the layers
# farther out than every refused setting give).
receivers <- function(layers) {
  vetoed <- any(vapply(layers, function(l) l$where != "own" && isFALSE(l$program_observers), NA))
  observers <- list()
  for (l in rev(layers)) {
    if (vetoed && l$where == "own") next
    for (o in l$observers) if (!any(vapply(the$observing %||% list(), identical, NA, o))) observers[[length(observers) + 1L]] <- o
  }
  refused <- refused_layers(layers)
  rest <- if (length(refused)) layers[-seq_len(max(refused))] else layers
  list(observers = observers, journal = chosen_journal(rest), refused = length(refused) > 0L)
}

chosen_journal <- function(layers) {
  for (l in layers) if (isTRUE(l$set)) return(if (isFALSE(l$journal) || is.null(l$journal)) NULL else l$journal)
  NULL
}

refused_layers <- function(layers) {
  setting <- Filter(function(i) isTRUE(layers[[i]]$set), seq_along(layers))
  value <- function(i) { j <- layers[[i]]$journal; if (is.null(j)) FALSE else j }
  out <- integer(0)
  for (n in seq_along(setting)) {
    i <- setting[[n]]; far <- value(i)
    for (k in setting[seq_len(n - 1L)]) {
      near <- value(k)
      if (same_journal(near, far)) next
      if (!isFALSE(far) && far$required) out <- c(out, k)
      else if (!isFALSE(far) && layers[[k]]$where == "own" && layers[[i]]$where != "own" &&
               !(!isFALSE(near) && identical(near$store, far$store) && near$required)) out <- c(out, k)
    }
  }
  sort(unique(out))
}

# ---------------------------------------------------------------- one tree's log, in this process

# The whole log of one call tree, numbered here: given to the streams watching
# calls in it (the whole form), to each call's observers (the kept form), to
# the tree's journal (the kept form) and to its sinks (a conversation store's
# copy, the kept form). `later`: a later writer's claim (writer, after, at),
# for a resumed turn: it numbers on from the last kept event, and holds back
# what it does again until it does something new.
new_tree_log <- function(tree, journal = NULL, later = NULL, clock = NULL, tap = NULL) {
  t <- new.env(parent = emptyenv())
  t$tree <- tree
  t$writer <- if (is.null(later)) 1L else as.integer(later$writer)
  t$seq <- if (is.null(later)) 0L else as.integer(later$after$seq)
  t$last <- if (is.null(later)) NULL else later$after
  t$at <- if (is.null(later) || !nzchar(later$at %||% "")) 0 else unix_of(later$at)
  t$keeps <- list(); t$programs <- list(); t$parents <- list(); t$observers <- list()
  t$observer_last <- list()
  t$journal <- journal
  t$pending <- list(); t$given <- 0L; t$confirmed <- 0L; t$refused <- FALSE; t$stopped <- FALSE; t$cause <- NULL
  t$barrier_n <- list(); t$journal_last <- NULL; t$warned <- FALSE
  t$ended <- FALSE; t$streams <- list(); t$sinks <- list(); t$cancelled <- character(0)
  t$replaying <- !is.null(later); t$held <- list()
  t$clock <- clock %||% function(seq) as.numeric(Sys.time())
  t$tap <- tap
  t
}

unix_of <- function(text) {
  m <- regmatches(text, regexec("^(\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2})(\\.(\\d+))?Z$", text))[[1L]]
  if (!length(m)) return(0)
  as.numeric(as.POSIXct(m[[2L]], format = "%Y-%m-%dT%H:%M:%S", tz = "UTC")) + if (nzchar(m[[4L]])) as.numeric(paste0("0.", m[[4L]])) else 0
}

tree_required <- function(t) !is.null(t$journal) && isTRUE(t$journal$required)
tree_watched <- function(t) length(t$streams) > 0L || !is.null(t$journal) || length(t$sinks) > 0L || length(unlist(t$observers, recursive = FALSE)) > 0L
# whether a live reader wants the text piece by piece (a stream, an observer, a journal); a conversation
# store's own copy of the log does not ask for it (streaming.md, "Streaming is asked for by a live reader")
wants_pieces <- function(call) {
  t <- call$tree
  !is.null(t) && (length(t$streams) > 0L || !is.null(t$journal) || length(unlist(t$observers, recursive = FALSE)) > 0L)
}

BARRIER_KINDS <- c("started", "tool_call", "done", "failed")

# Number an event and give it to its readers. `withhold`: a required journal's
# last event, given to streams and observers only once confirmed (deliver()).
emit_event <- function(t, kind, call, fn, data, withhold = FALSE, last = FALSE) {
  if (t$replaying) {
    # a resumed turn doing again what it did before: nothing is shown until it does something new
    if (kind == "started") { t$held[[call]] <- list(fn = fn, data = data); t$programs[[call]] <- data$program; return(NULL) }
    if (kind %in% c("done", "failed") && identical(call, t$tree)) tree_frontier(t)
    else if (kind %in% c("done", "failed")) { t$held[[call]] <- NULL; return(NULL) }
    else return(NULL)
  }
  if (t$ended) stop(sprintf("FunctAI: a %s event of call %s after its tree %s ended (a fault of functai, not of your code)", kind, call, t$tree), call. = FALSE)
  t$seq <- t$seq + 1L
  t$at <- max(t$at, t$clock(t$seq))
  e <- new_event(kind, t$tree, t$writer, t$seq, t$last, iso(t$at), call, fn, data)
  t$last <- event_position(e)
  if (last) t$ended <- TRUE
  if (!is.null(t$tap)) t$tap$events[[length(t$tap$events) + 1L]] <- e
  if (kind == "started") t$programs[[call]] <- data$program
  if (!is.null(t$journal) && !t$stopped) {
    k <- kept_event(e, t$keeps[[call]], t$programs[[call]])
    if (!is.null(k)) {
      k <- relinked(k, t$journal_last)
      t$journal_last <- event_position(k)
      n <- journal_send(t, k)
      if (kind %in% BARRIER_KINDS) t$barrier_n[[as.character(e$seq)]] <- n
    }
  }
  if (!withhold) deliver(t, e)
  e
}

# A resumed turn does something it had not done: the calls it started while
# replaying and that are still open are shown from here.
tree_frontier <- function(t) {
  if (!t$replaying) return(invisible())
  t$replaying <- FALSE
  held <- t$held; t$held <- list()
  for (id in names(held)) if (!identical(id, t$tree)) emit_event(t, "started", id, held[[id]]$fn, held[[id]]$data)
  invisible()
}

# Give an event to the streams watching its call, and the kept form to its
# call's observers and the tree's sinks.
deliver <- function(t, e) {
  for (i in seq_along(t$streams)) stream_offer(t, i, e)
  obs <- t$observers[[e$call]] %||% list()
  if (!length(obs) && !length(t$sinks)) return(invisible())
  k <- kept_event(e, t$keeps[[e$call]], t$programs[[e$call]])
  if (is.null(k)) return(invisible())
  for (o in obs) {
    key <- observer_key(o)
    linked <- relinked(k, t$observer_last[[key]])
    t$observer_last[[key]] <- event_position(linked)
    give_observer(o, linked)
  }
  for (i in seq_along(t$sinks)) {
    key <- paste0("sink", i)
    linked <- relinked(k, t$observer_last[[key]])
    t$observer_last[[key]] <- event_position(linked)
    t$sinks[[i]](linked)
  }
  invisible()
}

# An observer's identity, for its own `after` chain and for being broken.
observer_key <- function(o) {
  for (i in seq_along(the$observer_ids)) if (identical(the$observer_ids[[i]], o)) return(paste0("o", i))
  the$observer_ids[[length(the$observer_ids) + 1L]] <- o
  paste0("o", length(the$observer_ids))
}

# Give an observer an event, as it happens. Its own code's calls start trees
# of their own and are not given to it (it would feed itself); one that fails
# is warned about once and given nothing more.
give_observer <- function(o, e) {
  key <- observer_key(o)
  if (isTRUE(the$broken_observers[[key]])) return(invisible())
  old <- list(current = the$current, observing = the$observing)
  the$current <- NULL
  the$observing <- c(the$observing, list(o))
  on.exit({ the$current <- old$current; the$observing <- old$observing })
  tryCatch(o(e), error = function(err) {
    the$broken_observers[[key]] <- TRUE
    warn_once(paste0("observer:", key), sprintf("an observer failed (%s); it is given no more events", conditionMessage(err)))
  })
  invisible()
}

# ---------------------------------------------------------------- the journal writer

# Send one kept event after those before it: it joins what is not confirmed,
# and the writer sends it all again, in order, each at most 1 + retries times
# in a row. Returns its number among the events given.
journal_send <- function(t, k) {
  t$given <- t$given + 1L
  t$pending[[length(t$pending) + 1L]] <- k
  journal_round(t)
  t$given
}

journal_round <- function(t) {
  if (t$stopped) return(invisible())
  store <- t$journal$store
  while (length(t$pending)) {
    answered <- FALSE
    for (try in seq_len(1L + t$journal$retries)) {
      answer <- with_no_call(tryCatch(store$append(t$pending[[1L]]), error = identity))
      if (inherits(answer, "functai_store_refusal") || inherits(answer, "interrupt")) {
        t$refused <- inherits(answer, "functai_store_refusal"); t$stopped <- TRUE; t$cause <- answer
        journal_warn(t, answer)
        return(invisible())
      }
      if (identical(answer, "kept") || identical(answer, "duplicate")) { answered <- TRUE; break }
      t$cause <- if (inherits(answer, "condition")) answer else sprintf("the store answered %s", short_json(answer))
    }
    if (!answered) { journal_warn(t, t$cause); return(invisible()) }
    t$pending <- t$pending[-1L]
    t$confirmed <- t$confirmed + 1L
  }
  invisible()
}

# Code a journal or an observer runs is not a step of the call it serves.
with_no_call <- function(code) {
  old <- the$current; the$current <- NULL
  on.exit(the$current <- old)
  force(code)
}

journal_warn <- function(t, why) {
  if (t$journal$required || t$warned) return(invisible())
  t$warned <- TRUE
  warn_once(paste0("journal:", observer_key(t$journal$store)),
            sprintf("a best-effort journal did not keep an event (%s); calls go on", if (inherits(why, "condition")) conditionMessage(why) else as.character(why)))
}

# Whether the journal confirmed an event (by its seq in the whole log):
# "confirmed", "unanswered" or "refused".
confirmation <- function(t, e) {
  if (is.null(t$journal) || is.null(e)) return("confirmed")
  n <- t$barrier_n[[as.character(e$seq)]] %||% 0L
  if (n == 0L) return("refused")
  if (t$confirmed >= n) "confirmed" else if (t$refused) "refused" else "unanswered"
}

barrier_error <- function(t, e) journal_error("journal-barrier", sprintf("the journal did not keep event %d", e$seq), cause = t$cause, store = t$journal$store)

end_error <- function(t, status, terminal, outcome) {
  word <- if (status == "refused") "refused" else "unknown"
  journal_error("journal-end", sprintf("the journal did not confirm the call's end (%s): %s", position_text(event_position(terminal)),
                                       if (word == "refused") "it refused it" else "no answer came; it may be kept (settle() reads the journal)"),
                journal = word, tree = terminal$tree, event = event_position(terminal), outcome = outcome, cause = t$cause, store = t$journal$store)
}
