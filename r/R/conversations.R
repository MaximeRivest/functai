# Conversations (contract/conversations.md): a program's calls that remember
# each other. A conversation is records in a store (stores.R), appended and
# never changed: a `turn` record before the call (the turn's id is its call's
# id, known before the model is asked), `ended` after it, `lease` records
# while it runs, and what resuming it needs (`reply`, `tool`, `waiting`,
# `approval`). Memory belongs to the conversation, never to the function.

LEASE_SECONDS <- 30
RENEW_SECONDS <- 10
STOP_POLL <- 1

conversation_now <- function() (the$conversation_clock %||% function() as.numeric(Sys.time()))()

# ---------------------------------------------------------------- what the model sees

#' Which earlier turns a turn is shown
#'
#' `all_turns()` (the default): every earlier turn of the branch. Running out
#' of the model's context, and being told, is better than a model that
#' silently misses what was said. `last_turns(n)`: only the last `n`; the
#' others are still kept, and each turn's record says which turns it was
#' shown. `without`: inputs or outputs left out of every *earlier* turn (a
#' long document the model already answered about).
#' @param n How many earlier turns.
#' @param without Field names left out of earlier turns.
#' @return A context rule, for [ai_conversation()].
#' @export
all_turns <- function(without = character(0)) structure(list(last = NULL, without = as.character(without)), class = "functai_context_rule")

#' @rdname all_turns
#' @export
last_turns <- function(n, without = character(0)) {
  if (!is_num(n) || n < 0 || n != round(n)) cli::cli_abort("{.fn last_turns} takes a whole number of turns")
  structure(list(last = as.integer(n), without = as.character(without)), class = "functai_context_rule")
}

pick_turns <- function(rule, ts) if (is.null(rule$last)) ts else if (rule$last > 0L) utils::tail(ts, rule$last) else list()

#' What an AI function inside a program's conversation remembers
#'
#' In a program's conversation, the program's turns remember each other, but
#' the AI functions it calls (its helpers) start fresh at each call, unless
#' the conversation says otherwise: `ai_conversation(support, remembers =
#' list(answer = remember("conversation", steps = TRUE)))`. `"conversation"`:
#' its own calls in the earlier turns of this branch, then in this turn;
#' `"turn"`: in this turn only. `steps`: with their tool steps.
#' @param mode `"conversation"` or `"turn"`.
#' @param steps Whether its earlier calls are shown with their tool steps.
#' @return A memory, for `remembers`.
#' @export
remember <- function(mode = c("conversation", "turn"), steps = FALSE) structure(list(mode = match.arg(mode), steps = isTRUE(steps)), class = "functai_memory")

# ---------------------------------------------------------------- the records, read

new_turn_state <- function(r) list(record = r, ended = NULL, lease = NULL, waiting = NULL, stops = 0L, answers = list(), tools = list(),
                                   replies = list(), calls = list(), children = character(0), attempt = 1L)
seq_of <- function(r) if (is.null(r)) 0L else as.integer(r$seq %||% 0L)
ended_outputs <- function(st) if (is.null(st$ended)) list() else st$ended$outputs %||% list()

# A turn's state from its records (conversations.md, "A turn's state").
turn_state_of <- function(st, now = conversation_now()) {
  if (!is.null(st$ended)) return(st$ended$state)
  if (!is.null(st$waiting) && seq_of(st$waiting) > seq_of(st$lease)) return("waiting")
  until <- if (is.null(st$lease)) 0 else unix_of(st$lease$until %||% "")
  if (until >= now) "running" else "interrupted"
}

approval_key <- function(a) paste(a$site %||% "", as.integer(a$invocation %||% 0L), a$plugin %||% "approval", sep = "\u0001")

unanswered <- function(st) {
  if (is.null(st$waiting)) return(list())
  done <- vapply(Filter(function(a) seq_of(a) > seq_of(st$waiting), st$answers), approval_key, "")
  Filter(function(a) !approval_key(a) %in% done, st$waiting$approvals %||% list())
}

unfinished <- function(st) {
  started <- list()
  for (t in st$tools) {
    k <- paste(t$site %||% "", t$invocation %||% "", sep = "\u0001")
    if (identical(t$state, "started")) started[[k]] <- t
    else if (t$state %in% c("done", "given", "rerun")) started[[k]] <- NULL
  }
  unname(started)
}

new_conv_log <- function() { l <- new.env(parent = emptyenv()); l$n <- 0L; l$turns <- list(); l$order <- character(0); l$head <- NULL
                             l$request_ids <- list(); l$programs <- list(); l$entries <- list(); l }

apply_records <- function(log, records) {
  for (r in records) {
    log$n <- max(log$n, as.integer(r$seq %||% (log$n + 1L)))
    if (!identical(as.integer(r$functai_conversation %||% 0L), CONVERSATION_FORMAT)) next
    kind <- r$kind %||% ""
    if (kind == "program") { if (is.null(log$programs[[r$version]])) log$programs[[r$version]] <- r; next }
    if (kind == "entry") { log$entries[[length(log$entries) + 1L]] <- r; next }
    tid <- r$turn
    if (!is_str(tid)) next
    if (kind == "turn") {
      if (!is.null(log$turns[[tid]])) next
      log$turns[[tid]] <- new_turn_state(r)
      log$order <- c(log$order, tid)
      p <- r$parent
      if (is_str(p) && !is.null(log$turns[[p]])) log$turns[[p]]$children <- c(log$turns[[p]]$children, tid)
      if (is_str(r$request_id) && is.null(log$request_ids[[r$request_id]])) log$request_ids[[r$request_id]] <- tid
      log$head <- tid
      next
    }
    st <- log$turns[[tid]]
    if (kind == "head") { if (!is.null(st)) log$head <- tid; next }
    if (is.null(st)) next
    switch(kind,
      ended = if (is.null(st$ended)) st$ended <- r,
      lease = { st$lease <- r; st$attempt <- max(st$attempt, as.integer(r$attempt %||% 1L)) },
      waiting = st$waiting <- r,
      stop = st$stops <- st$stops + 1L,
      approval = st$answers[[length(st$answers) + 1L]] <- r,
      tool = st$tools[[length(st$tools) + 1L]] <- r,
      reply = st$replies[[length(st$replies) + 1L]] <- r,
      call = st$calls[[length(st$calls) + 1L]] <- r,
      NULL)
    log$turns[[tid]] <- st
  }
  log
}

branch_of <- function(log, turn) {
  out <- list(); seen <- character(0)
  while (is_str(turn) && !is.null(log$turns[[turn]]) && !turn %in% seen) {
    seen <- c(seen, turn); out <- c(list(log$turns[[turn]]), out); turn <- log$turns[[turn]]$record$parent
  }
  out
}

done_on <- function(log, turn) {
  while (is_str(turn) && !is.null(log$turns[[turn]])) {
    if (identical(turn_state_of(log$turns[[turn]]), "done")) return(turn)
    turn <- log$turns[[turn]]$record$parent
  }
  NULL
}

turn_id <- function(st) st$record$turn

# ---------------------------------------------------------------- programs, described

is_program <- function(x) inherits(x, "functai_program")
program_core_of <- function(p) if (is_program(p)) program_core(p) else core_of(p)
program_name <- function(p) program_core_of(p)$definition$name
program_info <- function(p) { core <- program_core_of(p); if (is_program(p)) program_json(core) else program_of(core)() }
program_iface <- function(p) if (is_program(p)) program_interface(program_core(p)) else interface_of(core_of(p))
answer_of <- function(p) { iface <- program_iface(p); iface$outputs[[length(iface$outputs)]]$name }

# A program's fields as data: what decides whether an earlier turn can be shown whole.
program_fields_data <- function(p) {
  if (is_program(p)) {
    iface <- program_interface(program_core(p))
    return(c(lapply(iface$inputs, function(x) list(name = x$name, direction = "input", purpose = "plain", shape = data_shape(x$shape))),
             lapply(iface$outputs, function(x) list(name = x$name, direction = "output", purpose = "plain", shape = data_shape(x$shape)))))
  }
  core <- core_of(p)
  fields <- lmcc::signature_to_list(signature_of(core, effective(core$own)))$fields
  fields <- Filter(function(f) !(f$direction == "input" && !identical(f$purpose %||% "plain", "plain")), fields)
  lapply(fields, function(f) list(name = f$name, direction = f$direction, purpose = f$purpose %||% "plain", shape = data_shape(f$shape)))
}

program_record <- function(p) {
  info <- program_info(p)
  out <- list(functai_conversation = CONVERSATION_FORMAT, kind = "program", at = iso(Sys.time()), version = info$version, name = info$name,
              program_kind = info$kind, module = info$module, interface = program_iface(p), fields = program_fields_data(p), answer = info$answer)
  if (!is.null(info$signature)) out$signature <- info$signature
  out
}

# Refuse a program that cannot be shown its earlier turns (conversations.md,
# "The program changed").
check_conversation_signature <- function(p, log, ts, earlier_without) {
  now <- program_fields_data(p); names(now) <- vapply(now, function(x) x$name, "")
  seen <- character(0)
  for (st in ts) {
    v <- st$record$program
    if (!is_str(v) || v %in% seen || is.null(log$programs[[v]])) next
    seen <- c(seen, v)
    was <- log$programs[[v]]$fields %||% list(); names(was) <- vapply(was, function(x) x$name, "")
    if (length(was) == length(now) && all(names(was) %in% names(now)) && all(vapply(names(was), function(k) same_json(was[[k]], now[[k]]), NA))) next
    both <- intersect(names(was), names(now))
    changed <- sort(both[!vapply(both, function(k) same_json(was[[k]], now[[k]]), NA)])
    gone <- sort(setdiff(names(was), names(now)))
    new <- setdiff(names(now), names(was))
    added <- vapply(new, function(n) now[[n]]$purpose %in% c("reasoning", "tools.calls"), NA)
    unexplained <- new[!(added & new %in% earlier_without)]
    if (!length(changed) && !length(gone) && !length(unexplained)) next
    parts <- character(0)
    if (length(unexplained)) {
      hidden <- unexplained[vapply(unexplained, function(n) now[[n]]$purpose %in% c("reasoning", "tools.calls"), NA)]
      parts <- c(parts, paste0("it now writes ", paste(unexplained, collapse = ", "), ", which earlier turns lack",
        if (length(hidden)) sprintf(" (to go on: earlier_without = c(%s); earlier turns are shown without them, nothing is rewritten)", paste(encodeString(hidden, quote = '"'), collapse = ", ")) else ""))
    }
    if (length(changed)) parts <- c(parts, paste(paste(changed, collapse = ", "), "changed type"))
    if (length(gone)) parts <- c(parts, paste(paste(gone, collapse = ", "), "is no longer one of its fields"))
    stop(conversation_error("conversation-signature", sprintf("%s: its earlier turns in this conversation were made with other inputs or outputs: %s",
                                                              program_name(p), paste(parts, collapse = "; "))))
  }
}

# ---------------------------------------------------------------- a conversation

#' A conversation with an AI function or a program
#'
#' A conversation is called like its program, and each call is a **turn**
#' that sees the earlier ones. Memory belongs to the conversation; the program
#' is unchanged, and still callable on its own. Turns are kept in a store,
#' and nothing in it is ever deleted: continuing from an earlier turn makes a
#' branch ([continue_from()]).
#'
#' ```r
#' chat <- ai_conversation(tutor, "alex", store = "tutoring/")   # the same line tomorrow opens it again
#' chat("Hi, I'm Alex.")
#' chat("What is 1/2 + 1/3?")                                     # sees the first turn
#' ai_turns(chat)                                                 # a tibble: each turn, its inputs and answer
#' ```
#'
#' Each turn's id is its call's id in the call log ([rate()] it), recorded
#' before the model is asked. A turn whose tool needs a person's answer
#' (`approve = "changes"`) waits, saved: the call raises a `functai_waiting`
#' condition holding `$turn`, and anyone who opens the conversation answers
#' it with [approve()] or [deny()], and it goes on, paying for nothing twice.
#' @param program An AI function or a program.
#' @param id The conversation's id (letters, digits, `.`, `_`, `-`): the same
#'   id in the same store opens the same conversation. Default: a new one.
#' @param store `NULL` (this R session's memory), a folder, `TRUE` (the
#'   default folder), or a store ([folder_store()], [memory_conversations()],
#'   or your own).
#' @param context [all_turns()] (default) or [last_turns()].
#' @param earlier_without Outputs the program now writes that its earlier
#'   turns lack (reasoning turned on, a first tool): earlier turns are shown
#'   without them.
#' @param remembers A program's helpers' memory: `list(answer =
#'   "conversation")`, `"turn"`, or [remember()]; a conversation used inside
#'   its turns, `"own"`. Helpers remember nothing otherwise.
#' @param sends Two sends at once (from two processes): `"queue"` (the
#'   default: the second waits, then continues from the first), `"refuse"`
#'   (`conversation-busy`), or `"branch"`.
#' @param ... Settings for every turn (`approve`, `lm`, `plugins`, ...).
#' @return A conversation (class `functai_conversation`): a function.
#' @export
ai_conversation <- function(program, id = NULL, store = NULL, context = all_turns(), earlier_without = character(0), remembers = NULL,
                            sends = c("queue", "refuse", "branch"), ...) {
  if (!inherits(program, c("functai_fn", "functai_program"))) cli::cli_abort("a conversation is with an AI function or a program")
  sends <- match.arg(sends)
  if (!inherits(context, "functai_context_rule")) cli::cli_abort("{.arg context} is {.fn all_turns} or {.fn last_turns}")
  settings <- check_settings(list(...))
  mems <- list()
  for (k in names(remembers)) {
    if (!is_program(program)) cli::cli_abort("{.arg remembers} is for a program's helpers; an AI function's conversation is its own memory")
    v <- remembers[[k]]
    mems[[k]] <- if (identical(v, "own")) "own" else if (inherits(v, "functai_memory")) v else remember(v)
  }
  c <- new.env(parent = emptyenv())
  c$program <- program; c$id <- if (is.null(id)) new_id() else check_conversation_id(id); c$store <- conversation_store(store)
  c$context <- context; c$earlier_without <- as.character(earlier_without); c$sends <- sends; c$settings <- settings
  c$remembers <- mems; c$head <- NULL; c$follow <- TRUE; c$exact <- FALSE; c$delegated <- FALSE
  c$log <- new_conv_log()
  check_conversation(c)
  make_conversation(c)
}

check_conversation <- function(c) {
  check_remembers(c)
  check_content(c)
  iface <- program_iface(c$program)
  opaque <- vapply(c(iface$inputs, iface$outputs), function(x) if (isTRUE(x$opaque)) x$name else NA_character_, "")
  opaque <- opaque[!is.na(opaque)]
  if (length(opaque)) stop(conversation_error("conversation-opaque", sprintf(
    "%s: %s may hold values with no JSON form, and a conversation keeps its turns as data: give %s a type", program_name(c$program), paste(opaque, collapse = ", "), if (length(opaque) > 1L) "them" else "it")))
  log <- read_conv(c)
  check_conversation_signature(c$program, log, branch_of(log, view_head(c, log)), c$earlier_without)
  invisible(c)
}

make_conversation <- function(c) {
  core <- program_core_of(c$program)
  env <- new.env(parent = asNamespace("functai"))
  env$.conv <- c; env$.inputs <- names(core$definition$inputs)
  args <- lapply(core$definition$inputs, function(f) if (isTRUE(f$optional) && has_key(f$shape, "default")) default_value(f) else rlang::missing_arg())
  names(args) <- env$.inputs
  args <- c(args, alist(.request_id = NULL, ... = ))
  f <- rlang::new_function(args, quote(send_turn(.conv, given_inputs(environment(), .inputs), list(...), .request_id)), env)
  structure(f, class = c("functai_conversation", "function"))
}

conv_of <- function(x) {
  if (inherits(x, "functai_conversation")) return(environment(x)$.conv)
  if (inherits(x, "functai_turn")) return(x$conv)
  if (is.environment(x) && !is.null(x$program) && !is.null(x$store)) return(x)
  cli::cli_abort("expected a conversation from {.fn ai_conversation}")
}

#' @export
print.functai_conversation <- function(x, ...) {
  c <- conv_of(x)
  t <- ai_turns(x)
  cat(sprintf("<conversation %s with %s> %d turn%s\n", c$id, program_name(c$program), nrow(t), if (nrow(t) == 1L) "" else "s"))
  for (i in seq_len(nrow(t))) {
    ins <- paste(vapply(t$inputs[[i]], function(v) short_text(if (is_str(v)) v else lmcc::canonical_json(v)), ""), collapse = " ")
    v <- t$outputs[[i]][[answer_of(c$program)]]
    out <- if (identical(t$state[[i]], "done")) short_text(if (is_str(v)) v else lmcc::canonical_json(v)) else sprintf("[%s]", t$state[[i]])
    cat(sprintf("%3d. %s \u2192 %s\n", i, substr(ins, 1L, 60L), substr(out, 1L, 70L)))
  }
  invisible(x)
}

check_remembers <- function(c) {
  if (!length(c$remembers)) return(invisible())
  reach <- names(program_parts(program_core(c$program)))
  reach <- sub("^.*:", "", reach)
  for (k in names(c$remembers)) if (!identical(c$remembers[[k]], "own") && !k %in% reach)
    cli::cli_abort("{.arg remembers} names {.field {k}}, which {program_name(c$program)} does not call (its AI functions: {if (length(reach)) paste(reach, collapse = ', ') else 'none'})")
}

# A conversation whose store keeps records refuses a program a log_content
# layer drops a field of (refuse, never forget).
check_content <- function(c) {
  if (!store_persistent(c$store)) return(invisible())
  p <- c$program; core <- program_core_of(p)
  dropped <- if (is_program(p)) {
    fields <- list(inputs = names(core$definition$inputs), outputs = names(core$definition$outputs), added = character(0))
    keep <- content_kept(fields, content_layers_program(core)); names(keep)[!keep]
  } else dropped_fields(core)
  if (length(dropped)) stop(conversation_error("conversation-content", sprintf(
    "%s: a log_content setting keeps %s out of every record, and this store keeps a conversation's records. A conversation that must remember what it may not keep refuses rather than forgets: keep it in memory (store = NULL), or let the log keep those fields",
    program_name(p), paste(dropped, collapse = ", "))))
}

durable_turns <- function(c) is_program(c$program) || length(core_of(c$program)$tools) > 0L

read_conv <- function(c) apply_records(c$log, c$store$read(c$id, c$log$n))
add_records <- function(c, records, expect = NULL) c$store$append(c$id, records, expect)

# Where this view continues: by id, the head; once it made a turn, or was
# made by continue_from(), its own branch.
view_head <- function(c, log) {
  if (isTRUE(c$follow)) return(log$head)
  pinned <- c$head
  if (isTRUE(c$exact)) return(pinned)
  for (tid in rev(log$order)) {
    t <- tid
    while (is_str(t)) { if (identical(t, pinned)) return(tid); t <- log$turns[[t]]$record$parent }
  }
  pinned
}

# What a record saw, saw_of expanded, from a list of records; an error when it cannot be known.
expanded_saw <- function(records, id) {
  by_id <- stats::setNames(records, vapply(records, function(r) r$id, ""))
  got <- expand_saw(by_id, id)
  if (inherits(got, "functai_saw_unknown")) stop(sprintf("what %s saw is not known (%s)", got$call, got$code), call. = FALSE)
  got
}

saw_records <- function(log) lapply(log$order, function(tid) {
  st <- log$turns[[tid]]
  list(functai_call = 2L, id = tid, saw = (if (!is.null(st$ended)) st$ended$saw) %||% st$record$saw %||% list())
})

# ---------------------------------------------------------------- turns, as their records say

turn_object <- function(c, st) structure(list(conv = c, id = turn_id(st), st = st), class = "functai_turn")

#' @export
`$.functai_turn` <- function(x, name) {
  st <- .subset2(x, "st"); c <- .subset2(x, "conv")
  ended <- st$ended %||% list()
  switch(name,
    conv = c, st = st, id = , call = .subset2(x, "id"), conversation = c$id, parent = st$record$parent,
    request_id = st$record$request_id, inputs = st$record$inputs %||% list(), outputs = ended_outputs(st),
    value = if ("value" %in% names(ended)) ended$value else ended_outputs(st)[[answer_of(c$program)]],
    state = turn_state_of(st), model = ended$model %||% st$record$settings$lm, error = ended$error, usage = ended$usage %||% list(),
    reads = unlist(st$record$reads), made_by = st$record$made_by, saw = turn_saw(c, st),
    waiting = lapply(unanswered(st), approval_from),
    unfinished = lapply(unfinished(st), function(t) list(invocation = t$invocation, id = t$id, name = t$name, input = t$input, site = t$site)),
    ended_outputs(st)[[name]])
}

#' @export
print.functai_turn <- function(x, ...) {
  ins <- paste(sprintf("%s = %s", names(x$inputs), vapply(x$inputs, function(v) short_text(lmcc::canonical_json(v)), "")), collapse = ", ")
  cat(sprintf("<turn %s> %s%s\n", substr(x$id, 1L, 8L), ins, if (identical(x$state, "done")) paste0(" \u2192 ", short_text(lmcc::canonical_json(x$value))) else sprintf(" [%s]", x$state)))
  invisible(x)
}

turn_saw <- function(c, st) {
  log <- read_conv(c)
  entries <- tryCatch(expanded_saw(saw_records(log), turn_id(st)), error = function(e) Filter(function(x) !is.null(x$call), (st$ended$saw %||% list())))
  ids <- unlist(lapply(entries, function(e) e$call))
  unname(lapply(ids[ids %in% names(log$turns)], function(id) turn_object(c, log$turns[[id]])))
}

#' The turns of a conversation
#'
#' `ai_turns()` gives the turns from the first to the head, in order (`all =
#' TRUE`: every turn of every branch, in the order they were made), as a
#' tibble: `turn` (its id, the call's id in the call log), `parent`,
#' `state` (`"running"`, `"waiting"`, `"interrupted"`, `"done"`, `"failed"`,
#' `"stopped"`, `"abandoned"`), `inputs` (a list column), `answer` (of the
#' answer's type; `NA` for a turn not done), `outputs` (a list column),
#' `model`. `ai_turn()` gives one turn, as an object for
#' [approve()], [resume_turn()], [stop_turn()].
#' @param chat A conversation.
#' @param all Every turn of every branch.
#' @param which A turn: its index in `ai_turns(chat)`, its id (`-1`, the
#'   default: the head).
#' @return A tibble (`ai_turns()`), or a turn.
#' @export
ai_turns <- function(chat, all = FALSE) {
  c <- conv_of(chat)
  log <- read_conv(c)
  sts <- if (all) lapply(log$order, function(t) log$turns[[t]]) else branch_of(log, view_head(c, log))
  ts <- lapply(sts, turn_object, c = c)
  core <- program_core_of(c$program)
  field <- core$definition$outputs[[length(core$definition$outputs)]]
  answers <- lapply(ts, function(t) if (identical(t$state, "done")) t$outputs[[answer_of(c$program)]] else NULL)
  answer <- if (isTRUE(field$opaque)) answers else tryCatch(assemble(field, answers), error = function(e) answers)
  tibble::tibble(turn = vapply(ts, function(t) t$id, ""), parent = vapply(ts, function(t) t$parent %||% NA_character_, ""),
                 state = vapply(ts, function(t) t$state, ""), inputs = lapply(ts, function(t) t$inputs), answer = answer,
                 outputs = lapply(ts, function(t) t$outputs), model = vapply(ts, function(t) as.character(t$model %||% NA_character_), ""))
}

#' @rdname ai_turns
#' @export
ai_turn <- function(chat, which = -1L) {
  c <- conv_of(chat)
  log <- read_conv(c)
  turn_object(c, log$turns[[turn_id_of(c, which)]])
}

turn_id_of <- function(c, t) {
  if (inherits(t, "functai_turn")) return(t$id)
  log <- read_conv(c)
  if (is_num(t)) {
    ids <- vapply(branch_of(log, view_head(c, log)), turn_id, "")
    i <- if (t < 0) length(ids) + 1L + t else t
    if (i >= 1L && i <= length(ids)) return(ids[[i]])
  }
  if (is_str(t) && !is.null(log$turns[[t]])) return(t)
  stop(conversation_error("turn-unknown", sprintf("conversation %s has no turn %s", c$id, short_json(t))))
}

#' Continue a conversation from an earlier turn
#'
#' The same conversation, continuing after `turn`: its next turn is a new
#' branch. Nothing is deleted.
#' @param chat A conversation.
#' @param turn A turn, its id, or its index in [ai_turns()].
#' @return A conversation.
#' @export
continue_from <- function(chat, turn) {
  c <- conv_of(chat)
  tid <- turn_id_of(c, turn)
  d <- new.env(parent = emptyenv())
  for (k in ls(c, all.names = TRUE)) assign(k, get(k, envir = c), envir = d)
  d$head <- tid; d$follow <- FALSE; d$exact <- TRUE; d$log <- new_conv_log()
  make_conversation(d)
}

#' Make a turn the conversation's head, for everyone who opens it
#' @param chat A conversation.
#' @param turn A turn, its id, or its index.
#' @return The conversation, invisibly.
#' @export
move_head <- function(chat, turn) {
  c <- conv_of(chat); tid <- turn_id_of(c, turn)
  add_records(c, list(list(functai_conversation = CONVERSATION_FORMAT, kind = "head", at = iso(Sys.time()), turn = tid)))
  invisible(chat)
}

# ---------------------------------------------------------------- sending a turn

check_nested <- function(c) {
  call <- the$current
  outer <- if (!is.null(call)) call$turn_run else the$turn_starting
  if (is.null(outer) || isTRUE(c$delegated) || (identical(outer$conv$id, c$id) && identical(outer$conv$store, c$store))) return(invisible())
  name <- program_name(c$program)
  if (identical(outer$conv$remembers[[name]], "own")) return(invisible())
  stop(conversation_error("conversation-nested", sprintf(
    "conversation %s (%s) is used inside a turn of conversation %s, which does not say so: a remembering program inside another is refused unless declared (remembers = list(%s = \"own\"))",
    c$id, name, outer$conv$id, name)))
}

# The turn's inputs as they are bound: a refused input is refused before anything is recorded.
bound_turn_inputs <- function(c, given) {
  p <- c$program
  if (is_program(p)) {
    core <- program_core(p)
    row <- program_row(core, given, 1L, character(0))
    if (inherits(row, "functai_misfit")) stop(rlang::error_cnd(c("functai_interface_input", "functai_refusal"), code = "interface-input", field = row$field,
                                                              message = sprintf("%s: input %s", core$definition$name, row$message)))
    return(list(given = given, values = row))
  }
  core <- core_of(p)
  row <- input_rows(core, given, n = 1L)[[1L]]
  if (inherits(row, "functai_misfit")) stop(misfit_error(core, row))
  list(given = given, values = row)
}

turn_start_hooks <- function(c, given, settings) {
  core <- program_core_of(c$program)
  plugins <- plugins_around(core$own, list(settings))
  if (!has_hook(plugins, "turn_start")) return(list(given = given, changes = list()))
  log <- read_conv(c)
  ev <- hook_event("turn_start", inputs = given, conversation = c$id, parent = view_head(c, log), program = c$program, turn_run = NULL, conv = c)
  applied <- new_applied()
  names_ <- names(core$definition$inputs)
  run_hook("turn_start", plugins, ev, applied, function(ch) {
    if (!is.list(ch$inputs) || is.null(names(ch$inputs))) stop("inputs is a named list of the turn's inputs")
    for (k in names(ch$inputs)) { if (!k %in% names_) stop(sprintf("%s has no input %s", program_name(c$program), k)); ev$inputs[k] <- list(ch$inputs[[k]]) }
  })
  list(given = ev$inputs, changes = applied$items)
}

parent_for <- function(c, log) {
  h <- view_head(c, log)
  if (!is_str(h) || is.null(log$turns[[h]])) return(NULL)
  st <- log$turns[[h]]
  state <- turn_state_of(st)
  if (state == "waiting" && c$sends != "branch")
    stop(conversation_error("conversation-busy", sprintf("conversation %s: turn %s waits for a person's answer (answer it, or continue from another turn)", c$id, h), turn = h))
  if (state %in% c("running", "waiting")) {
    if (c$sends == "queue") return(structure(list(), class = "functai_busy"))
    if (c$sends == "refuse") stop(conversation_error("conversation-busy", sprintf("conversation %s: turn %s is %s", c$id, h, state), turn = h))
    return(done_on(log, st$record$parent))
  }
  done_on(log, h)
}

turn_holder <- function() {
  if (is.null(the$turn_holder)) the$turn_holder <- sprintf("%s:%d:%s", Sys.info()[["nodename"]], Sys.getpid(), paste(format(openssl::rand_bytes(3)), collapse = ""))
  the$turn_holder
}
lease_record <- function(tid, attempt) list(functai_conversation = CONVERSATION_FORMAT, kind = "lease", at = iso(Sys.time()), turn = tid,
                                            holder = turn_holder(), until = iso(as.numeric(Sys.time()) + LEASE_SECONDS), attempt = as.integer(attempt))

send_turn <- function(c, given, extra = list(), request_id = NULL) {
  core <- program_core_of(c$program)
  if (length(extra) && (is.null(names(extra)) || any(!nzchar(names(extra))))) cli::cli_abort("a turn's extra arguments are settings, by name")
  own <- check_settings(extra)
  check_nested(c)
  check_content(c)
  turn_settings <- set_all(c$settings, own)
  hooked <- turn_start_hooks(c, given, turn_settings)
  bound <- bound_turn_inputs(c, hooked$given)
  repeat {
    log <- read_conv(c)
    if (!is.null(request_id) && !is.null(log$request_ids[[as.character(request_id)]]))
      return(turn_outcome(turn_object(c, wait_turn_state(c, log$request_ids[[as.character(request_id)]]))))
    parent <- parent_for(c, log)
    if (inherits(parent, "functai_busy")) { if (is.function(c$store$wait)) c$store$wait(c$id, log$n, 0.25) else Sys.sleep(0.25); next }
    check_conversation_signature(c$program, log, branch_of(log, parent), c$earlier_without)
    tid <- new_id()
    context <- turn_context(c, log, parent, turn_settings)
    context$changes <- c(hooked$changes, context$changes)
    desc <- program_record(c$program)
    recs <- if (is.null(log$programs[[desc$version]])) list(desc) else list()
    rec <- list(functai_conversation = CONVERSATION_FORMAT, kind = "turn", at = iso(Sys.time()), turn = tid)
    rec["parent"] <- list(parent); rec$program <- desc$version; rec$inputs <- if (length(bound$values)) bound$values else lmcc::jobj()
    if (!is.null(request_id)) rec$request_id <- as.character(request_id)
    if (is_str(turn_settings$lm)) rec$settings <- list(lm = turn_settings$lm)
    if (!is.null(context$recorded)) rec$context <- context$recorded
    if (length(context$changes)) rec$changes <- context$changes
    recs <- c(recs, list(rec, lease_record(tid, 1L)))
    start <- tryCatch(add_records(c, recs, expect = log$n), functai_store_conflict = function(e) NULL)
    if (is.null(start)) next
    c$head <- tid; c$follow <- FALSE; c$exact <- FALSE
    break
  }
  run <- new_turn_run(c, tid, 1L, context)
  run$settings <- turn_settings
  run$start_seq <- start
  start_turn(c, run, bound$given, turn_settings)
}

# What the turn after `parent` is shown (conversations.md, "What a turn is
# shown"): the earlier turns the rule picks, then the context hooks' changes;
# the saw entries; the rows earlier() gives; the sections; the changes made
# and the context to record. `fixed`: the context a turn recorded when made.
turn_context <- function(c, log, parent, settings = c$settings, fixed = NULL) {
  shown <- shown_turns_of(c, log, parent, settings, fixed)
  is_ai <- !is_program(c$program)
  sig <- if (is_ai) ai_signature_id(c$program) else NULL
  ts <- list(); ids <- character(0); rows <- list(); dropped <- list()
  for (st in shown$picked) {
    outs <- ended_outputs(st)
    row <- c(st$record$inputs %||% list(), outs)
    gone_names <- shown$without[[turn_id(st)]] %||% character(0)
    rows[[length(rows) + 1L]] <- if (length(row)) row[!names(row) %in% gone_names] else lmcc::jobj()
    ids <- c(ids, turn_id(st))
    if (!is_ai) next
    desc <- log$programs[[st$record$program %||% ""]] %||% list()
    t <- if (!is.null(st$ended$lmcc) && identical(desc$signature, sig)) fitted_turn(c$program, st$ended$lmcc)
      else list(inputs = st$record$inputs %||% lmcc::jobj(), outputs = if (length(outs)) outs else lmcc::jobj())
    w <- without_fields(t, gone_names)
    if (length(w$gone)) dropped[[turn_id(st)]] <- w$gone
    ts[[length(ts) + 1L]] <- w$turn
  }
  records <- saw_records(log)
  finish <- function(entries) {
    out <- lapply(entries, function(e) {
      gone <- dropped[[e$call %||% ""]]
      if (!is.null(gone)) { e$without <- as.list(sort(unique(c(unlist(e$without), gone)))); e$steps <- NULL }
      e
    })
    compress_saw(out, parent, records)
  }
  base <- list(rows = rows, parent = parent, sections = shown$sections, changes = shown$changes, recorded = shown$recorded)
  if (!is_ai) {
    entries <- finish(lapply(ids, function(i) list(call = i)))
    return(c(base, list(turns = list(), ids = character(0), finish = function(e) entries, module_saw = entries)))
  }
  c(base, list(turns = ts, ids = ids, finish = finish))
}

without_fields <- function(t, names_) {
  had <- c(names(t$inputs), names(t$outputs))
  gone <- sort(intersect(had, names_))
  if (!length(gone)) return(list(turn = t, gone = character(0)))
  keep <- function(x) { x <- x %||% list(); x <- x[!names(x) %in% gone]; if (length(x)) x else lmcc::jobj() }
  list(turn = list(inputs = keep(t$inputs), outputs = keep(t$outputs)), gone = gone)
}

# `[{"saw_of": parent}, <parent's entry>]` when the entries are exactly what
# the parent saw, then the parent (calls.md, "Saw").
compress_saw <- function(entries, parent, records) {
  if (length(entries) < 2L || is.null(parent) || !identical(entries[[length(entries)]]$call, parent)) return(entries)
  if (!any(vapply(records, function(r) identical(r$id, parent), NA))) return(entries)
  before_ <- tryCatch(expanded_saw(records, parent), error = function(e) NULL)
  if (is.null(before_)) return(entries)
  if (same_json(before_, entries[-length(entries)])) list(list(saw_of = parent), entries[[length(entries)]]) else entries
}

shown_turns_of <- function(c, log, parent, settings, fixed = NULL) {
  done <- Filter(function(st) identical(turn_state_of(st), "done"), branch_of(log, parent))
  if (!is.null(fixed)) {
    by_id <- stats::setNames(done, vapply(done, turn_id, ""))
    picked <- Filter(Negate(is.null), lapply(unlist(fixed$turns), function(t) by_id[[t]]))
    return(list(picked = picked, without = lapply(fixed$without %||% list(), unlist), sections = as.character(unlist(fixed$sections)), changes = list(), recorded = NULL))
  }
  picked <- pick_turns(c$context, done)
  without <- list()
  for (st in picked) {
    fields <- c(names(st$record$inputs), names(ended_outputs(st)))
    gone <- sort(intersect(fields, c$context$without))
    if (length(gone)) without[[turn_id(st)]] <- gone
  }
  core <- program_core_of(c$program)
  plugins <- plugins_around(core$own, list(settings))
  if (!has_hook(plugins, "context")) return(list(picked = picked, without = without, sections = character(0), changes = list(), recorded = NULL))
  shown <- function(st) list(id = turn_id(st), inputs = st$record$inputs %||% list(), outputs = ended_outputs(st), without = without[[turn_id(st)]] %||% character(0))
  ev <- hook_event("context", turns = lapply(picked, shown), sections = character(0), conversation = c$id, parent = parent, program = c$program, turn_run = NULL, conv = c)
  applied <- new_applied()
  branch_done <- stats::setNames(done, vapply(done, turn_id, ""))
  current <- picked
  run_hook("context", plugins, ev, applied, function(ch) {
    if (!is.null(ch$keep)) {
      ids <- as.character(unlist(ch$keep))
      unknown <- setdiff(ids, names(branch_done))
      if (length(unknown)) stop(sprintf("keep names %s, which is not a done turn of this branch", unknown[[1L]]))
      current <<- Filter(function(st) turn_id(st) %in% ids, done)
      ev$turns <- lapply(current, shown)
    }
    if (!is.null(ch$without)) {
      targets <- if (is.list(ch$without) && !is.null(names(ch$without))) ch$without else stats::setNames(rep(list(as.character(unlist(ch$without))), length(current)), vapply(current, turn_id, ""))
      for (tid in names(targets)) without[[tid]] <<- sort(unique(c(without[[tid]], as.character(unlist(targets[[tid]])))))
    }
    if (!is.null(ch$sections)) ev$sections <- c(ev$sections, texts_of(ch$sections))
  })
  picked <- current
  ids <- vapply(picked, turn_id, "")
  without <- without[names(without) %in% ids]
  recorded <- list(turns = as.list(ids), without = if (length(without)) lapply(without, as.list) else lmcc::jobj(), sections = as.list(ev$sections))
  list(picked = picked, without = without, sections = ev$sections, changes = applied$items, recorded = recorded)
}

# A stored turn made for this function's signature, with the fingerprint lmcc
# checks (this plan's); by its values when no plan can be made.
fitted_turn <- function(p, t) {
  if (is.null(t$signature)) return(t)
  core <- core_of(p)
  out <- tryCatch({
    s <- effective(core$own)
    plan <- bind_layout(s$adapter, s$template, signature_of(core, s), probe_capabilities(), "probe")
    t$signature <- lmcc::signature_fingerprint(plan$signature); t
  }, error = function(e) list(inputs = t$inputs %||% lmcc::jobj(), outputs = t$outputs %||% lmcc::jobj()))
  out
}

# ---------------------------------------------------------------- plugin entries

#' A plugin's entries in a conversation
#'
#' `keep_entry()` keeps what a plugin needs later in a conversation, at a
#' turn (it then belongs to the branches through that turn), or with no turn
#' (every branch); never shown to a model by itself. Inside a hook, give the
#' event: the entry is the hook's plugin's, at its turn. `entries()` reads a
#' plugin's entries of a kind on a branch, oldest first.
#' @param x A conversation, or a hook's event.
#' @param kind The entry's kind (a name).
#' @param data What it keeps (JSON).
#' @param plugin The plugin's name (an event knows its own).
#' @param turn The turn it belongs to (`NULL`: every branch).
#' @param branch The branch through this turn (default: the head).
#' @return `entries()`: a list of `list(turn, data, at)`.
#' @export
keep_entry <- function(x, kind, data, plugin = NULL, turn = NULL) {
  if (inherits(x, "functai_hook_event")) {
    plugin <- plugin %||% x$plugin$name
    run <- x$turn_run
    c <- x$conv %||% (if (!is.null(run)) run$conv)
    if (is.null(c)) cli::cli_abort("{.fn keep_entry} keeps an entry in a conversation; this call runs in none")
    turn <- turn %||% x$turn_id %||% (if (!is.null(run)) run$turn)
  } else c <- conv_of(x)
  if (!is_str(kind) || !nzchar(kind)) cli::cli_abort("an entry's kind is a name")
  rec <- list(functai_conversation = CONVERSATION_FORMAT, kind = "entry", at = iso(Sys.time()), plugin = plugin, entry = kind, data = value_json(data))
  if (!is.null(turn)) rec$turn <- turn
  add_records(c, list(rec))
  invisible()
}

#' @rdname keep_entry
#' @export
entries <- function(x, kind, plugin = NULL, branch = NULL) {
  if (inherits(x, "functai_hook_event")) {
    plugin <- plugin %||% x$plugin$name
    run <- x$turn_run
    c <- x$conv %||% (if (!is.null(run)) run$conv)
    if (is.null(c)) return(list())
    branch <- branch %||% x$parent %||% (if (!is.null(run)) run$turn)
    out <- conv_entries(c, plugin, kind, branch)
    if (!is.null(x$turn_id)) out <- c(out, Filter(function(r) identical(r$turn, x$turn_id), conv_entries(c, plugin, kind, NULL, only_turnless = FALSE, all = TRUE)))
    return(out)
  }
  c <- conv_of(x)
  conv_entries(c, plugin, kind, branch %||% view_head(c, read_conv(c)))
}

conv_entries <- function(c, plugin, kind, through, only_turnless = TRUE, all = FALSE) {
  log <- read_conv(c)
  on <- vapply(branch_of(log, through), turn_id, "")
  hits <- Filter(function(r) identical(r$plugin, plugin) && identical(r$entry, kind) && (all || is.null(r$turn) || r$turn %in% on), log$entries)
  lapply(hits, function(r) list(turn = r$turn, data = r$data, at = r$at))
}

# ---------------------------------------------------------------- the turn running in this process

new_turn_run <- function(c, tid, attempt, context, replay = NULL, later = NULL) {
  run <- new.env(parent = emptyenv())
  run$conv <- c; run$store <- c$store; run$program <- c$program; run$turn <- tid; run$attempt <- as.integer(attempt)
  run$context <- context; run$root_taken <- FALSE; run$helper_calls <- list(); run$usage <- list(); run$later <- later
  run$start_seq <- 0L; run$settings <- list(); run$durable <- durable_turns(c); run$root <- NULL
  run$replies <- list(); run$tools <- list(); run$unfinished <- list(); run$answers <- list()
  run$seen <- 0L; run$renewed <- as.numeric(Sys.time()); run$polled <- 0
  for (rec in replay$replies) run$replies[[rec$key]] <- c(run$replies[[rec$key]], list(rec$response))
  for (t in replay$tools) {
    k <- paste(t$site %||% "", as.integer(t$invocation %||% 0L), sep = "\u0001")
    if (t$state %in% c("done", "given")) { run$tools[[k]] <- t; run$unfinished[[k]] <- NULL }
    else if (identical(t$state, "rerun")) { run$tools[[k]] <- NULL; run$unfinished[[k]] <- NULL }
    else if (identical(t$state, "started") && is.null(run$tools[[k]])) run$unfinished[[k]] <- t
  }
  last_wait <- replay$waiting_seq %||% 0L
  for (a in replay$answers) run$answers[[approval_key(a)]] <- list(allowed = identical(a$verdict, "yes"), reason = a$reason, by = a$by, fresh = seq_of(a) > last_wait)
  run
}

record_base <- function(run, kind) list(functai_conversation = CONVERSATION_FORMAT, kind = kind, at = iso(Sys.time()), turn = run$turn, attempt = run$attempt)

# The turn's own call (its id the turn's): its record says which
# conversation and turn it is, its tree keeps its kept log in the store.
attach_turn <- function(call, core) {
  run <- call$turn_run
  run$root <- call
  call$conversation <- list(id = run$conv$id, turn = run$turn); call$conversation["parent"] <- list(run$context$parent)
  call$changes <- c(call$changes, run$context$changes)
  if (identical(core$kind, "module")) call$saw <- run$context$module_saw %||% list()
  events <- run$store$events
  if (!is.null(events) && is.null(call$up)) {
    t <- call$tree
    t$sinks[[length(t$sinks) + 1L]] <- function(e) tryCatch(events$append(e), error = function(err)
      warn_once(paste0("conversation-events:", conditionMessage(err)), sprintf("a turn's events could not be kept in its store (%s); the turn goes on, and its records are kept", conditionMessage(err))))
    t$observer_last[[paste0("sink", length(t$sinks))]] <- if (is.null(run$later)) NULL else run$later$after
  }
  if (!is.null(run$later)) call$requests <- as.integer(run$later$requests %||% 0L)
  invisible()
}

# A call of the turn ended: its tokens count in the turn's usage; a helper the
# conversation remembers keeps its call.
turn_call_ended <- function(call, err) {
  run <- call$turn_run
  if (is.null(run)) return(invisible())
  for (e in call$exchanges) if (!is.null(e$response)) for (k in names(u <- usage_of(e$response))) run$usage[[k]] <- (run$usage[[k]] %||% 0L) + u[[k]]
  if (identical(call$id, run$turn) || !is.null(err) || is.null(call$lmcc) || is.null(call$core) || identical(call$core$kind, "module")) return(invisible())
  memory <- run$conv$remembers[[call$core$definition$name]]
  if (!inherits(memory, "functai_memory")) return(invisible())
  rec <- record_base(run, "call")
  rec$call <- call$id; rec$site <- call$site
  rec$program <- list(name = call$core$definition$name, module = call$core$module, signature = ai_signature_id(make_fn(call$core)))
  rec$lmcc <- call$lmcc; rec$saw <- call$saw %||% list()
  run$helper_calls[[length(run$helper_calls) + 1L]] <- rec
  add_records(run$conv, list(rec))
}

# A turn's heartbeat, from the call's checks: every second, a stop asked from
# another process; every 10 seconds, its lease renewed.
turn_check_stop <- function(run) {
  now <- as.numeric(Sys.time())
  if (now - run$polled < STOP_POLL) return(invisible())
  run$polled <- now
  recs <- tryCatch(run$store$read(run$conv$id, run$seen), error = function(e) list())
  run$seen <- run$seen + length(recs)
  mine <- Filter(function(r) identical(r$turn, run$turn), recs)
  stop_now <- any(vapply(mine, function(r) identical(r$kind, "stop"), NA))
  taken <- any(vapply(mine, function(r) identical(r$kind, "lease") && as.integer(r$attempt %||% 1L) > run$attempt, NA))
  if (taken) warn_once(paste0("lease-lost:", run$turn), sprintf("turn %s was taken over by another process (its lease ran out here): this one stops", run$turn))
  if ((stop_now || taken) && !is.null(run$root)) run$root$tree$cancelled <- c(run$root$tree$cancelled, run$turn)
  if (now - run$renewed >= RENEW_SECONDS) { run$renewed <- now; tryCatch(add_records(run$conv, list(lease_record(run$turn, run$attempt))), error = function(e) NULL) }
  invisible()
}

# ---- resuming: what was recorded

recorded_reply <- function(job, request) {
  run <- job$call$turn_run
  if (is.null(run) || !length(run$replies)) return(NULL)
  key <- reply_key(request, job$settings$replicate %||% 0L)
  q <- run$replies[[key]]
  if (!length(q)) return(NULL)
  run$replies[[key]] <- q[-1L]
  lm15::from_dict(q[[1L]], "response")
}

reply_note <- function(job, response) {
  run <- job$call$turn_run
  if (is.null(run) || !run$durable) return(invisible())
  tryCatch({
    rec <- record_base(run, "reply")
    rec$key <- reply_key(job$sent %||% job$request, job$settings$replicate %||% 0L)
    rec$response <- plain_lm15(response)
    add_records(run$conv, list(rec))
  }, error = function(e) warn_once(paste0("note-reply:", conditionMessage(e)), sprintf("a reply could not be kept in the conversation (%s)", conditionMessage(e))))
  invisible()
}

tool_recorded <- function(call, n) {
  run <- call$turn_run
  if (is.null(run)) return(NULL)
  k <- paste(call$site, n, sep = "\u0001")
  if (!is.null(run$tools[[k]])) return(as.character(run$tools[[k]]$output %||% ""))
  if (!is.null(run$unfinished[[k]])) stop(conversation_error("turn-unfinished", sprintf(
    "%s started before the turn stopped, and whether it ran is not known: resume_turn(turn, results = list(\"%d\" = what it returned)) or resume_turn(turn, rerun = %d)",
    run$unfinished[[k]]$name, n, n), turn = run$turn))
  NULL
}

tool_started <- function(call, tool, c, n, input) {
  run <- call$turn_run
  if (is.null(run) || !tool_changes(tool)) return(invisible())
  rec <- record_base(run, "tool")
  rec$site <- call$site; rec$invocation <- n; rec$id <- c$id; rec$name <- c$name; rec$input <- value_json(input)
  rec["effects"] <- list(tool$effects); rec$state <- "started"
  add_records(run$conv, list(rec))
}

tool_done <- function(call, tool, c, n, out) {
  run <- call$turn_run
  if (is.null(run)) return(invisible())
  rec <- record_base(run, "tool")
  rec$site <- call$site; rec$invocation <- n; rec$id <- c$id; rec$name <- c$name; rec$state <- "done"; rec$output <- as.character(out)
  add_records(run$conv, list(rec))
}

recorded_approval <- function(run, call, a) run$answers[[paste(call$site, as.integer(a$invocation), a$plugin, sep = "\u0001")]]

note_approval <- function(run, call, a, allowed, reason, by) {
  rec <- list(functai_conversation = CONVERSATION_FORMAT, kind = "approval", at = iso(Sys.time()), turn = run$turn, site = call$site,
              invocation = a$invocation, path = a$path, plugin = a$plugin, verdict = if (allowed) "yes" else "no")
  rec["by"] <- list(by); rec["reason"] <- list(reason)
  add_records(run$conv, list(rec))
}

turn_waiting <- function(a) structure(class = c("functai_turn_waiting", "error", "condition"),
  list(message = sprintf("waiting for a person's answer: %s", a$path), call = NULL, approval = a))

# ---- what a call is shown

# The earlier turns an AI function's call is shown (a conversation's turn,
# a remembered helper, a rated row asked again), as the plan shows them,
# and the `saw` entries that say so; NULL when none.
context_turns <- function(core, call, plan, s) {
  found <- context_for(core, call)
  if (is.null(found)) return(NULL)
  got <- shown_as(plan, core, found$turns, found$ids)
  entries <- if (!is.null(found$finish)) found$finish(got$entries) else got$entries
  list(turns = got$shown, saw = entries)
}

context_for <- function(core, call) {
  replay <- the$replaying
  if (!is.null(replay)) { got <- replay_context(replay, core, call); if (!is.null(got)) return(got) }
  run <- call$turn_run
  if (is.null(run)) return(NULL)
  if (identical(call$id, run$turn)) return(list(turns = run$context$turns, ids = run$context$ids, finish = run$context$finish))
  memory <- run$conv$remembers[[core$definition$name]]
  if (!inherits(memory, "functai_memory")) return(NULL)
  helper_context(run, core, memory)
}

helper_context <- function(run, core, memory) {
  found <- list()
  if (memory$mode == "conversation") {
    log <- read_conv(run$conv)
    for (st in branch_of(log, run$context$parent)) {
      if (!identical(turn_state_of(st), "done")) next
      final <- as.integer(st$ended$attempt %||% st$attempt)
      found <- c(found, Filter(function(x) as.integer(x$attempt %||% 1L) == final, st$calls))
    }
  }
  found <- c(found, run$helper_calls)
  sig <- ai_signature_id(make_fn(core))
  ts <- list(); ids <- character(0)
  for (x in found) {
    if (!identical(x$program$name, core$definition$name) || !identical(x$program$module, core$module)) next
    t <- x$lmcc
    if (!memory$steps) t$steps <- list()
    if (!identical(x$program$signature, sig)) t <- list(inputs = t$inputs %||% lmcc::jobj(), outputs = t$outputs %||% lmcc::jobj())
    ts[[length(ts) + 1L]] <- fitted_turn(make_fn(core), t); ids <- c(ids, x$call)
  }
  list(turns = ts, ids = ids, finish = NULL)
}

# An earlier turn as this plan shows it: whole when made for it (with its
# steps), else an example of its values; NULL when no output is left.
fit_turn <- function(plan, d) {
  if (!is.null(d$signature) && !is.null(d$steps) && identical(d$signature, lmcc::signature_fingerprint(plan$signature))) {
    t <- tryCatch(lmcc::load_turn(plan, d), lmcc_refusal = function(e) NULL)
    if (!is.null(t)) return(list(turn = t, whole = TRUE))
  }
  fields <- lmcc::signature_to_list(plan$signature)$fields
  plain <- vapply(Filter(function(f) f$direction == "input" && (f$purpose %||% "plain") == "plain", fields), function(f) f$name, "")
  kept <- vapply(Filter(function(f) f$direction == "output" && (f$purpose %||% "plain") %in% c("plain", "reasoning"), fields), function(f) f$name, "")
  ins <- prepare_inputs(plan$signature, (d$inputs %||% list())[names(d$inputs %||% list()) %in% plain])
  outs <- (d$outputs %||% list())[names(d$outputs %||% list()) %in% kept]
  if (!length(outs)) return(NULL)
  t <- tryCatch(lmcc::example_turn(plan, ins, outs), lmcc_refusal = function(e) NULL)
  if (is.null(t)) NULL else list(turn = t, whole = FALSE, had = c(names(d$inputs), names(d$outputs)), now = c(names(ins), names(outs)))
}

shown_as <- function(plan, core, ts, ids) {
  entries <- list(); shown <- list()
  for (i in seq_along(ts)) {
    t <- ts[[i]]; cid <- ids[[i]]
    got <- fit_turn(plan, t)
    if (is.null(got)) next
    entries[[length(entries) + 1L]] <- if (!nzchar(cid)) list(unrecorded = TRUE)
      else if (got$whole) { if (length(t$steps)) list(call = cid, steps = TRUE) else list(call = cid) }
      else { left <- sort(setdiff(got$had, got$now)); if (length(left)) list(call = cid, without = as.list(left)) else list(call = cid) }
    shown[[length(shown) + 1L]] <- got$turn
  }
  list(entries = entries, shown = shown)
}

# The instruction sections a call is given before its own before_call hooks:
# its turn's (the conversation's context hooks), or a row asked again.
context_sections <- function(call) {
  if (!is.null(the$replaying)) return(replay_sections(the$replaying, call))
  run <- call$turn_run
  if (is.null(run) || !identical(call$id, run$turn)) return(character(0))
  as.character(run$context$sections %||% character(0))
}

# The setting layers a turn adds around its calls (the conversation's and the turn's).
turn_settings_layers <- function(call) {
  run <- call$turn_run
  if (is.null(run)) list() else list(run$settings %||% run$conv$settings)
}

#' The conversation so far, inside a program's turn
#'
#' Inside a program's turn, one element per earlier turn it is shown: its
#' inputs and outputs by name, without the fields the conversation leaves
#' out. Outside a conversation, an empty list. Give it to a helper that
#' declares an input for it (a handoff summary, say).
#' @return A list of named lists.
#' @export
earlier <- function() {
  if (!is.null(the$replaying)) return(replay_rows(the$replaying))
  call <- the$current
  while (!is.null(call) && !is.null(call$up)) call <- call$up
  run <- if (is.null(call)) the$turn_starting else call$turn_run
  if (is.null(run)) return(list())
  run$context$rows %||% list()
}

# ---------------------------------------------------------------- running a turn

start_turn <- function(c, run, given, settings) {
  started <- as.numeric(Sys.time())
  old <- the$turn_starting
  the$turn_starting <- run
  on.exit(the$turn_starting <- old)
  p <- c$program
  out <- tryCatch(list(value = rlang::inject(with_ai_config(call_turn(p, given), !!!settings))),
                  functai_turn_waiting = function(w) w, error = identity, interrupt = function(e) cancelled_error())
  the$turn_starting <- old
  err <- if (inherits(out, "condition")) out else NULL
  value <- if (is.null(err)) out$value else NULL
  err <- conclude_turn(c, run, value, err, started)
  if (!is.null(err)) stop(err)
  value
}

call_turn <- function(p, given) {
  if (is_program(p)) {
    core <- program_core(p)
    r <- run_program_row(core, given, 1L)
    if (!is.null(r$error)) stop(r$error)
    return(program_answers(core, list(r)))
  }
  core <- core_of(p)
  rows <- input_rows(core, given, n = 1L)
  results <- run_rows(core, rows)
  if (!is.null(results[[1L]]$error)) stop(results[[1L]]$error)
  answers(core, results)
}

# The turn's last record (`ended`, or `waiting`); returns the error the caller gets.
conclude_turn <- function(c, run, value, err, started) {
  root <- run$root
  saw <- if (!is.null(root)) root$saw %||% list() else run$context$module_saw %||% list()
  if (inherits(err, "functai_turn_waiting")) {
    a <- err$approval
    rec <- record_base(run, "waiting")
    rec$approvals <- list(approval_json(a)); rec$saw <- saw
    add_records(c, list(rec))
    t <- turn_object(c, read_conv(c)$turns[[run$turn]])
    return(rlang::error_cnd(c("functai_waiting", "functai_turn_waiting_user", "functai_refusal"), code = "turn-waiting", turn = t, approvals = t$waiting,
      functai_type = "Waiting", message = sprintf("turn %s waits for a person's answer: %s(%s); approve() or deny() it", run$turn, a$path, short_text(lmcc::canonical_json(value_json(a$input))))))
  }
  rec <- record_base(run, "ended")
  rec$saw <- saw
  rec$seconds <- round(as.numeric(Sys.time()) - started, 6)
  rec$usage <- if (length(run$usage)) run$usage else lmcc::jobj()
  if (!is.null(err)) {
    rec$state <- if (inherits(err, "functai_cancelled")) "stopped" else "failed"
    rec$error <- error_json(err)
  } else {
    rec$state <- "done"
    rec$outputs <- if (!is.null(root) && !is.null(root$outputs)) root$outputs else stats::setNames(list(value_json(value)), answer_of(c$program))
    rec["value"] <- list(if (!is.null(root)) json_value(root$done_value) else value_json(value))
    if (!is.null(root$lmcc)) rec$lmcc <- root$lmcc
    last_model <- if (!is.null(root)) Filter(function(e) !is.null(e$response), root$exchanges) else list()
    rec["model"] <- list(if (length(last_model)) last_model[[length(last_model)]]$model else NULL)
  }
  turn_end_hooks(c, run, rec)
  tryCatch(add_records(c, list(rec)), error = function(e) warn_once(paste0("ended:", conditionMessage(e)), sprintf("a turn's end could not be kept in its store (%s)", conditionMessage(e))))
  err
}

turn_end_hooks <- function(c, run, rec) {
  core <- program_core_of(c$program)
  plugins <- plugins_around(core$own, list(set_all(c$settings, run$settings)))
  if (!has_hook(plugins, "turn_end")) return(invisible())
  st <- read_conv(c)$turns[[run$turn]]
  ev <- hook_event("turn_end", turn = run$turn, turn_id = run$turn, state = rec$state, inputs = st$record$inputs %||% list(), outputs = rec$outputs %||% list(),
                   conversation = c$id, parent = st$record$parent, conv = c, turn_run = NULL)
  for (p in plugins) for (f in p$handlers$turn_end) {
    ev$plugin <- p
    tryCatch(f(ev), error = function(err) warn_once(paste0("turn_end:", p$name, conditionMessage(err)), sprintf("plugin %s failed in turn_end (%s)", p$name, conditionMessage(err))))
  }
  invisible()
}

#' The done turns of an ended turn's branch, for a turn_end hook
#'
#' Inside a `turn_end` hook: the done turns of the ended turn's branch, the
#' ended turn last when it is done, each `list(id, inputs, outputs)`.
#' @param event The `turn_end` event.
#' @return A list.
#' @export
branch_turns <- function(event) {
  c <- event$conv
  log <- read_conv(c)
  out <- lapply(Filter(function(st) identical(turn_state_of(st), "done"), branch_of(log, event$parent)),
                function(st) list(id = turn_id(st), inputs = st$record$inputs %||% list(), outputs = ended_outputs(st)))
  if (identical(event$state, "done")) out[[length(out) + 1L]] <- list(id = event$turn, inputs = event$inputs, outputs = event$outputs)
  out
}

turn_outcome <- function(t) {
  state <- t$state
  if (state == "done") return(t$value)
  if (state == "waiting") stop(rlang::error_cnd(c("functai_waiting", "functai_refusal"), code = "turn-waiting", turn = t, approvals = t$waiting,
                                                functai_type = "Waiting", message = sprintf("turn %s waits for a person's answer", t$id)))
  err <- t$error
  stop(conversation_error("turn-state", sprintf("turn %s ended %s%s", t$id, state, if (length(err)) paste0(": ", err$type, ": ", err$message %||% "") else ""), turn = t$id))
}

wait_turn_state <- function(c, tid, timeout = NULL) {
  deadline <- if (is.null(timeout)) Inf else as.numeric(Sys.time()) + timeout
  repeat {
    st <- read_conv(c)$turns[[tid]]
    if (!identical(turn_state_of(st), "running")) return(st)
    if (as.numeric(Sys.time()) >= deadline) cli::cli_abort("turn {tid} is still running")
    if (is.function(c$store$wait)) c$store$wait(c$id, c$log$n, 0.5) else Sys.sleep(0.25)
  }
}

#' What can be done with a turn
#'
#' `wait_turn()` waits until a turn (run by another process) is no longer
#' running. `stop_turn()` stops a running turn wherever it runs: it ends
#' `stopped` within about a second (a waiting or interrupted turn, which
#' nothing runs, ends `abandoned`). `resume_turn()` goes on with a turn that
#' waits (every approval answered) or was interrupted, here: its program runs
#' again with the same inputs and earlier turns, each model reply it had and
#' each tool result it kept are reused (nothing is paid for or run twice). A
#' tool that started and has no result may have run: `results` says what it
#' returned, `rerun` runs it again. `abandon_turn()` ends a waiting or
#' interrupted turn without going on.
#' @param turn A turn ([ai_turn()], or a waiting condition's `$turn`).
#' @param timeout Seconds to wait at most (`NULL`: no limit).
#' @param results What tools that may have run returned, by invocation:
#'   `list("2" = "refunded")`.
#' @param rerun The invocations of tools to run again.
#' @return `resume_turn()`: the turn's answer; `wait_turn()`: the turn.
#' @export
wait_turn <- function(turn, timeout = NULL) { c <- conv_of(turn); turn_object(c, wait_turn_state(c, turn$id, timeout)) }

#' @rdname wait_turn
#' @export
stop_turn <- function(turn) {
  c <- conv_of(turn)
  st <- read_conv(c)$turns[[turn$id]]
  state <- turn_state_of(st)
  if (state %in% c("waiting", "interrupted")) return(abandon_turn(turn))
  if (state != "running") return(invisible())
  add_records(c, list(list(functai_conversation = CONVERSATION_FORMAT, kind = "stop", at = iso(Sys.time()), turn = turn$id)))
  invisible()
}

#' @rdname wait_turn
#' @export
abandon_turn <- function(turn) {
  c <- conv_of(turn)
  st <- read_conv(c)$turns[[turn$id]]
  if (!turn_state_of(st) %in% c("waiting", "interrupted"))
    stop(conversation_error("turn-state", sprintf("only a waiting or interrupted turn can be abandoned; turn %s is %s", turn$id, turn_state_of(st)), turn = turn$id))
  add_records(c, list(list(functai_conversation = CONVERSATION_FORMAT, kind = "ended", at = iso(Sys.time()), turn = turn$id, state = "abandoned", attempt = st$attempt)))
  invisible()
}

answer_turn <- function(t, approval, allowed, reason, by, resume) {
  c <- conv_of(t)
  st <- read_conv(c)$turns[[t$id]]
  waiting <- unanswered(st)
  if (!length(waiting)) stop(conversation_error("turn-state", sprintf("turn %s waits for no approval (it is %s)", t$id, turn_state_of(st)), turn = t$id))
  target <- if (is.null(approval)) {
    if (length(waiting) > 1L) cli::cli_abort("turn {t$id} waits for {length(waiting)} approvals: name one")
    waiting[[1L]]
  } else {
    inv <- if (is.list(approval)) as.integer(approval$invocation) else as.integer(approval)
    hit <- Filter(function(a) as.integer(a$invocation) == inv && (!is.list(approval) || is.null(approval$plugin) || identical(a$plugin %||% "approval", approval$plugin)), waiting)
    if (!length(hit)) stop(conversation_error("turn-state", sprintf("turn %s waits for no approval %s", t$id, short_json(approval)), turn = t$id))
    hit[[1L]]
  }
  rec <- list(functai_conversation = CONVERSATION_FORMAT, kind = "approval", at = iso(Sys.time()), turn = t$id, site = target$site, invocation = target$invocation,
              path = target$path, plugin = target$plugin %||% "approval", verdict = if (allowed) "yes" else "no")
  rec["by"] <- list(by); rec["reason"] <- list(reason)
  add_records(c, list(rec))
  st <- read_conv(c)$turns[[t$id]]
  if (resume && !length(unanswered(st))) return(resume_turn(turn_object(c, st)))
  invisible()
}

#' @rdname wait_turn
#' @export
resume_turn <- function(turn, results = list(), rerun = integer(0)) {
  c <- conv_of(turn); tid <- turn$id
  repeat {
    log <- read_conv(c)
    st <- log$turns[[tid]]
    if (is.null(st)) stop(conversation_error("turn-unknown", sprintf("conversation %s has no turn %s", c$id, tid)))
    state <- turn_state_of(st)
    if (!state %in% c("waiting", "interrupted")) stop(conversation_error("turn-state", sprintf("turn %s is %s: only a waiting or interrupted turn goes on", tid, state), turn = tid))
    if (length(unanswered(st))) stop(conversation_error("turn-state", sprintf("turn %s still waits for %d approval(s)", tid, length(unanswered(st))), turn = tid))
    given <- list()
    for (x in unfinished(st)) {
      inv <- as.integer(x$invocation)
      base <- list(functai_conversation = CONVERSATION_FORMAT, kind = "tool", at = iso(Sys.time()), turn = tid, site = x$site, invocation = inv, id = x$id, name = x$name)
      hit <- results[[as.character(inv)]] %||% results[[x$id %||% ""]]
      if (!is.null(hit)) given[[length(given) + 1L]] <- c(base, list(state = "given", output = as.character(hit)))
      else if (inv %in% rerun) given[[length(given) + 1L]] <- c(base, list(state = "rerun"))
      else stop(conversation_error("turn-unfinished", sprintf("turn %s: %s (invocation %d) started and may have run: resume_turn(turn, results = list(\"%d\" = what it returned)) or resume_turn(turn, rerun = %d)",
                                                              tid, x$name, inv, inv, inv), turn = tid))
    }
    attempt <- st$attempt + 1L
    start <- tryCatch(add_records(c, c(given, list(lease_record(tid, attempt))), expect = log$n), functai_store_conflict = function(e) NULL)
    if (!is.null(start)) break
  }
  log <- read_conv(c); st <- log$turns[[tid]]
  later <- NULL
  events <- c$store$events
  if (!is.null(events)) later <- tryCatch({
    claim <- events$claim(tid)
    kept <- events$read(tid)
    reqs <- vapply(Filter(function(e) e$kind == "request" && identical(e$call, tid), kept), function(e) as.integer(e$request), 0L)
    list(writer = claim$writer, after = claim$after, at = if (length(kept)) kept[[length(kept)]]$at else "", requests = if (length(reqs)) max(reqs) else 0L)
  }, error = function(e) NULL)
  replay <- list(replies = st$replies, tools = st$tools, answers = st$answers, waiting_seq = seq_of(st$waiting))
  context <- turn_context(c, log, st$record$parent, c$settings, st$record$context)
  context$changes <- list()
  run <- new_turn_run(c, tid, attempt, context, replay, later)
  run$start_seq <- start
  settings <- c$settings
  if (is_str(st$record$settings$lm)) settings$lm <- st$record$settings$lm
  run$settings <- settings
  start_turn(c, run, st$record$inputs %||% list(), settings)
}

#' A turn's events, from its store
#'
#' The kept form of a turn's call tree log, as its store keeps it (another
#' process, or a page after a reload, reads it so), after the event `after`
#' names (`list(writer, seq)`): those kept so far, then each as it is kept,
#' until its last. `view = "outside"`: what a caller who sees only the
#' program's boundary may see.
#' @param turn A turn.
#' @param after A position, or `NULL` for every event.
#' @param view `"kept"` or `"outside"`.
#' @param timeout Seconds without a new event before it gives up (`NULL`: until the turn ends).
#' @return A list of events.
#' @export
turn_events <- function(turn, after = NULL, view = c("kept", "outside"), timeout = NULL) {
  view <- match.arg(view)
  c <- conv_of(turn)
  events <- c$store$events
  if (is.null(events)) return(list())
  out <- list(); last <- NULL; quiet <- as.numeric(Sys.time())
  v <- if (view == "outside") new_view("outside", program_core_of(c$program)$answer_from) else NULL
  waiting_for <- after
  repeat {
    got <- tryCatch(events$read(turn$id, last), functai_event_unknown = function(e) list())
    for (e in got) {
      last <- event_position(e); quiet <- as.numeric(Sys.time())
      shown <- if (is.null(v)) e else view_event(v, e)
      if (!is.null(shown)) {
        if (!is.null(v) && !is.null(waiting_for)) { if (same_position(event_position(shown), waiting_for)) waiting_for <- NULL }
        else if (is.null(v) && !is.null(after) && !is.null(waiting_for)) { if (same_position(event_position(shown), waiting_for)) waiting_for <- NULL }
        else out[[length(out) + 1L]] <- shown
      }
      if (identical(e$call, turn$id) && e$kind %in% c("done", "failed")) return(out)
    }
    if (turn_state_of(read_conv(c)$turns[[turn$id]]) %in% c("waiting", "interrupted", "abandoned")) return(out)
    if (!is.null(timeout) && as.numeric(Sys.time()) - quiet > timeout) return(out)
    Sys.sleep(0.05)
  }
}

# ---------------------------------------------------------------- the rest

# The exact request a conversation's next turn would send (nothing is sent or recorded).
render_turn <- function(fn, ...) {
  c <- conv_of(fn)
  if (is_program(c$program)) cli::cli_abort("{program_name(c$program)} is a program: render the request one of its helpers would get with {.fn ai_render}")
  log <- read_conv(c)
  parent <- view_head(c, log)
  parent <- if (is.null(parent)) NULL else done_on(log, parent)
  ctx <- turn_context(c, log, parent, c$settings)
  old <- the$rendering
  the$rendering <- list(core = core_of(c$program), turns = ctx$turns, ids = ctx$ids, sections = ctx$sections)
  on.exit(the$rendering <- old)
  ai_render(c$program, ...)
}

# ---------------------------------------------------------------- rows asked again with their context (stage 5)

# A rated row asked again (evaluate, the optimizers) with what its call was
# shown: the program's own call gets the row's `earlier` turns, each helper
# call the earlier turns its original call was shown, and its record says
# `saw_of` the original. No conversation is read or written.
new_row_replay <- function(row, core) {
  r <- new.env(parent = emptyenv())
  r$core <- core; r$earlier <- row$earlier %||% list(); r$helpers <- row$helpers %||% list()
  r$sections <- as.character(unlist(row$sections)); r$call <- row$call; r$helper_sections <- list()
  r
}

replay_turn <- function(core, t) {
  sig <- ai_signature_id(make_fn(core))
  if (length(t$steps) && identical(t$signature, sig))
    return(fitted_turn(make_fn(core), list(signature = sig, inputs = t$inputs %||% lmcc::jobj(), steps = t$steps, outputs = t$outputs %||% lmcc::jobj())))
  list(inputs = t$inputs %||% lmcc::jobj(), outputs = t$outputs %||% lmcc::jobj())
}

replay_context <- function(r, core, call) {
  if (is.null(call$up) && identical(core$definition$name, r$core$definition$name)) {
    if (!length(r$earlier)) return(NULL)
    ts <- lapply(r$earlier, replay_turn, core = core)
    original <- r$call
    return(list(turns = ts, ids = rep("", length(ts)), finish = if (is.null(original)) NULL else function(e) list(list(saw_of = original))))
  }
  i <- Position(function(h) identical(h$program, core$definition$name), r$helpers)
  if (is.na(i)) return(NULL)
  h <- r$helpers[[i]]; r$helpers <- r$helpers[-i]
  r$helper_sections[[call$id]] <- as.character(unlist(h$sections))
  ts <- lapply(h$earlier %||% list(), replay_turn, core = core)
  if (!length(ts)) return(NULL)
  original <- h$call
  list(turns = ts, ids = rep("", length(ts)), finish = if (is.null(original)) NULL else function(e) list(list(saw_of = original)))
}

replay_sections <- function(r, call) {
  if (is.null(call$up) && identical(call$core$definition$name, r$core$definition$name)) return(r$sections)
  r$helper_sections[[call$id]] %||% character(0)
}

replay_rows <- function(r) lapply(r$earlier, function(t) c(t$inputs %||% list(), t$outputs %||% list()))

# Ask a rated row again with the context its call had; a row with none changes nothing.
replaying <- function(row, core, code) {
  has <- length(row$earlier) || length(row$helpers) || length(row$sections)
  if (!has) return(force(code))
  old <- the$replaying
  the$replaying <- new_row_replay(row, core)
  on.exit(the$replaying <- old)
  force(code)
}
