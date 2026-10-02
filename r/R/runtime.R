# One call, from its start to its end, the same for an AI function's row and a
# program's call: it joins the tree of the call it runs inside (or starts a
# tree, with the observers and journal its layers give), its `started` event,
# the journal's barrier at a tree's start, its outcome (a value or an error)
# decided before the journal is asked, its `done` or `failed`, its record, and
# a required journal's answer about its end (contract/streaming.md).

# Open a call: `program` is a function giving the call log's program object,
# `own` the program's own settings (its receiver layer), `keep` content_kept()'s
# named logical. Sets the call's tree, observers, site and, when its layers
# break the journal policy or set a required journal inside a tree, the
# refusal it ends with before its code runs.
open_call <- function(call, own = list(), name = NULL) {
  parent <- the$current
  if (!is.null(parent) && isTRUE(parent$ended)) parent <- NULL
  call$name <- name %||% call$program_json$name
  layers <- receiver_layers(own)
  got <- receivers(layers)
  call$observers <- got$observers
  call$refusal <- NULL
  call$requests <- 0L; call$invocations <- 0L; call$names <- list()
  call$invocation <- if (is.null(parent)) NULL else the$invocation
  call$saw <- call$saw %||% list()
  scripted <- if (is.null(parent)) the$scripted else NULL
  if (!is.null(scripted)) { call$id <- scripted$id; call$root <- scripted$id }
  if (is.null(parent)) {
    if (got$refused) call$refusal <- journal_error("journal-policy",
      "a setting replaces or removes the journal a host layer set, or weakens a required one: the tree does not run")
    later <- call$later
    call$tree <- new_tree_log(call$id, got$journal, later = later, clock = scripted$clock, tap = scripted$tap)
    call$site <- paste0(call$name, "#1")
  } else {
    call$tree <- parent$tree
    mine <- chosen_journal(layers)
    if (!is.null(mine) && (is.null(call$tree$journal) || !same_journal(mine, call$tree$journal))) {
      if (mine$required) call$refusal <- journal_error("journal-scope",
        "a required journal set only inside a call tree keeps nothing of it: set it around the tree's outermost call")
      else warn_once(paste0("journal-scope:", observer_key(mine$store)), "a journal set only inside a call tree keeps nothing of it: set it around the tree's outermost call")
    }
    parent$names[[call$name]] <- (parent$names[[call$name]] %||% 0L) + 1L
    call$site <- sprintf("%s/%s#%d", parent$site, call$name, parent$names[[call$name]])
    call$up <- parent
  }
  t <- call$tree
  t$keeps[[call$id]] <- call$keep_events
  t$parents[[call$id]] <- call$parent
  t$observers[[call$id]] <- call$observers
  call
}

started_data <- function(call) {
  data <- list(); data["parent"] <- list(call$parent); data$root <- call$root; data$program <- call$program_json
  data$inputs <- if (length(call$inputs)) call$inputs else lmcc::jobj()
  data$content <- TRUE
  data$saw <- call$saw %||% list()
  if (!is.null(call$invocation)) data$invocation <- call$invocation
  data
}

call_emit <- function(call, kind, data = list(), ...) emit_event(call$tree, kind, call$id, call$name, data, ...)

# The call begins: its `started`, then (a tree's outermost call, with a
# required journal) the start barrier. Returns the error the call ends with
# before its code runs (a policy refusal, the barrier), or NULL.
call_begin <- function(call) {
  start <- call_emit(call, "started", started_data(call))
  if (!is.null(call$refusal)) return(call$refusal)
  if (is.null(call$up) && tree_required(call$tree) && confirmation(call$tree, start) != "confirmed") return(barrier_error(call$tree, start))
  NULL
}

# What a done event's value holds and is: an AI function's answer (several
# outputs: all of them, as R returns them), a program's outputs.
done_value <- function(call) call$done_value

# End a call with its outcome: its `done` or `failed`, its record, and at a
# tree's end the journal's answer. With a required journal the end is given to
# readers only once confirmed; when it is not, the error returned is the
# journal-end error holding the outcome, and the record says `journal`.
# Returns the error the caller gets (NULL when it gets the value).
call_end <- function(call, err = NULL) {
  t <- call$tree
  outermost <- is.null(call$up)
  if (is.null(call$ended_at)) call$ended_at <- as.numeric(Sys.time())
  if (is.null(err) && call_cancelled(call)) err <- cancelled_error()
  withhold <- outermost && tree_required(t)
  data <- if (is.null(err)) list(value = json_value(done_value(call))) else list(error = error_json(err))
  terminal <- call_emit(call, if (is.null(err)) "done" else "failed", data, withhold = withhold, last = outermost)
  call$ended <- TRUE
  status <- if (withhold) confirmation(t, terminal) else "confirmed"
  unconfirmed <- withhold && status != "confirmed"
  if (unconfirmed) call$journal_word <- if (status == "refused") "refused" else "unknown"
  if (!is.null(call$on_end)) call$on_end(call, err)
  finish_call(call, err)
  if (unconfirmed) return(end_error(t, status, terminal, if (is.null(err)) list(done = done_value(call)) else list(failed = err)))
  if (withhold && !is.null(terminal)) deliver(t, terminal)
  err
}

# A value as JSON for an event: its JSON form, or the description of a value
# with none (calls.md, *Values*).
json_value <- function(v) tryCatch(plain_json(v), error = function(e) describe_value(v))
describe_value <- function(v) list(`$type` = class(v)[[1L]], `$repr` = substr(paste(utils::capture.output(print(v)), collapse = "\n"), 1L, 2000L))

# Run a program's code as a call: its value is what the call returns.
run_call <- function(call, body) {
  old <- the$current
  the$current <- call
  on.exit(the$current <- old)
  err <- call_begin(call)
  if (is.null(err)) {
    out <- tryCatch(list(value = body(call)), functai_waiting = function(w) w, error = identity,
                    interrupt = function(e) cancelled_error())
    if (inherits(out, "functai_waiting")) {
      # a turn stopped to wait for a person: nothing ended; its log stays unfinished, for the process that resumes it
      call$ended <- TRUE
      stop(out)
    }
    if (inherits(out, "condition")) err <- out else {
      call$value <- out$value
      if (!isTRUE(call$has_done_value)) call$done_value <- out$value
    }
  }
  ending <- call_end(call, err)
  if (!is.null(ending)) stop(ending)
  call$value
}

# The model asked for a tool: its `tool_call` event, then, with a required
# journal, the barrier before a tool that changes things (it does not run when
# its events are not confirmed); a closed stream stops it too.
tool_called <- function(call, data, changes = TRUE) {
  asked <- call_emit(call, "tool_call", data)
  if (changes && tree_required(call$tree) && confirmation(call$tree, asked) != "confirmed") stop(barrier_error(call$tree, asked))
  check_cancelled(call)
  asked
}

# ---------------------------------------------------------------- cancelling

cancelled_error <- function() rlang::error_cnd(c("functai_cancelled"), message = "the call was cancelled: its stream was closed, or its turn stopped")

# A call is cancelled when a stream watching it (or a call it runs inside) was
# closed, or its conversation turn was asked to stop.
call_cancelled <- function(call) {
  t <- call$tree
  if (is.null(t) || !length(t$cancelled)) return(FALSE)
  id <- call$id
  while (!is.null(id)) { if (id %in% t$cancelled) return(TRUE); id <- t$parents[[id]] }
  FALSE
}

check_cancelled <- function(call) {
  if (!is.null(call$turn_run)) turn_check_stop(call$turn_run)
  if (call_cancelled(call)) stop(cancelled_error())
}
