# What a call was shown (contract/calls.md, "Saw"): the earlier calls it was
# given as context, read from the call log. Knowing which calls it saw is one
# question; whether the log keeps what showing them again needs is another.
# R's own calls see none yet (their records say `saw: []`); these read any
# language's records, for the day R shows earlier turns (rated() with
# earlier turns, conversations).

SAW_ENTRY_KEYS <- c("call", "steps", "without", "slot")

saw_unknown <- function(code, call) structure(list(code = code, call = call), class = "functai_saw_unknown")

# The entries a call saw, every saw_of replaced by the entries of the call it
# names; or why they cannot be known (not-recorded, missing-call,
# unknown-key, saw-cycle), naming the call whose record says so.
expand_saw <- function(by_id, call, following = character(0)) {
  rec <- by_id[[call]]
  if (is.null(rec) || !has_key(rec, "saw"))
    return(saw_unknown(if (length(following)) "missing-call" else "not-recorded", call))
  if (!is_arr(rec$saw)) return(saw_unknown("unknown-key", call))       # not a list of entries: not one a reader knows
  out <- list()
  for (i in seq_along(rec$saw)) {
    entry <- rec$saw[[i]]
    if (!is_obj(entry)) return(saw_unknown("unknown-key", call))
    if (has_key(entry, "saw_of")) {
      if (i != 1L || !identical(names(entry), "saw_of")) return(saw_unknown("unknown-key", call))
      target <- entry$saw_of
      if (!is_str(target)) return(saw_unknown("unknown-key", call))
      if (target %in% following || identical(target, call)) return(saw_unknown("saw-cycle", target))
      inner <- expand_saw(by_id, target, c(following, call))
      if (inherits(inner, "functai_saw_unknown")) return(inner)
      out <- c(out, inner)
    } else {
      if (!known_entry(entry)) return(saw_unknown("unknown-key", call))
      out[[length(out) + 1L]] <- entry
    }
  }
  out
}

# An entry this reader knows: a call's id, and only the keys calls.md names,
# each with a value of its kind (`steps` true; `without` names, at least one,
# each once; `slot` a name). A known key with a value of another kind says
# something this reader does not know: it fails closed, as for a key it does
# not know.
known_entry <- function(entry) {
  if (!has_key(entry, "call") || !is_str(entry$call) || length(setdiff(names(entry), SAW_ENTRY_KEYS))) return(FALSE)
  if (has_key(entry, "steps") && !isTRUE(entry$steps)) return(FALSE)
  if (has_key(entry, "without")) {
    w <- entry$without
    if (!is_arr(w) || !length(w) || !all(vapply(w, function(x) is_str(x) && nzchar(x), NA)) || anyDuplicated(unlist(w))) return(FALSE)
  }
  if (has_key(entry, "slot") && !is_name(entry$slot)) return(FALSE)
  TRUE
}

# Whether a call's record keeps what an entry says it was shown: the values
# of the fields it was shown with, as data; with steps, a whole record whose
# every exchange keeps its request hash and, when a reply came, the reply.
keeps_values <- function(rec, entry) {
  if (isTRUE(rec$truncated)) return(FALSE)
  left_out <- unlist(entry$without)
  shown <- setdiff(c(names(rec$sizes$inputs), names(rec$sizes$outputs)), left_out)
  described <- c(unlist(rec$described$inputs), unlist(rec$described$outputs))
  if (length(intersect(shown, described))) return(FALSE)
  if (isTRUE(entry$steps))
    return(isTRUE(rec$content) && all(vapply(rec$exchanges, function(ex)
      has_key(ex, "request_hash") && (has_key(ex, "response") || is.null(ex$finish)), NA)))
  if (isTRUE(rec$content)) return(TRUE)
  if (!has_key(rec, "omitted")) return(FALSE)                  # format 1, or no value kept
  !length(setdiff(c(unlist(rec$omitted$inputs), unlist(rec$omitted$outputs)), left_out))
}

# What a call saw, and whether the log keeps what showing it again needs
# (calls.md, "Reading it", "Knowing is not replaying"). `records`: lines of
# a call log, of any kind: only call records of a format this reader knows
# (1 and 2) are read; any other line (a rating, a later format) is skipped
# whole, as a reader skips what it does not know. Two different records
# with one id cannot both be the call: that id is read as a call whose record
# this reader does not have (the same record written twice is one). Returns
# list(saw = entries, or unknown = list(code, call)) and keeps = list(ok =
# TRUE) or list(refuses = code, call = id).
read_saw <- function(records, call) {
  by_id <- list(); twice <- character(0)
  for (r in records) {
    if (!is_obj(r) || !is_format(r$functai_call, CALL_FORMATS) || !is_str(r$id)) next
    old <- by_id[[r$id]]
    if (!is.null(old) && !same_json(old, r)) twice <- c(twice, r$id)
    by_id[[r$id]] <- r
  }
  by_id[unique(twice)] <- NULL
  entries <- expand_saw(by_id, call)
  if (inherits(entries, "functai_saw_unknown")) {
    why <- unclass(entries)
    return(list(unknown = why, keeps = list(refuses = why$code, call = why$call)))
  }
  for (entry in entries) {
    if (isTRUE(entry$steps) && "calls" %in% unlist(entry$without))
      return(list(saw = entries, keeps = list(refuses = "turn-invalid", call = entry$call)))
    rec <- by_id[[entry$call]]
    if (is.null(rec)) return(list(saw = entries, keeps = list(refuses = "missing-call", call = entry$call)))
    if (!keeps_values(rec, entry)) return(list(saw = entries, keeps = list(refuses = "not-kept", call = entry$call)))
  }
  list(saw = entries, keeps = list(ok = TRUE))
}

# The turn a saw entry stands for, shown again (calls.md, "The turn an entry
# stands for"): every field in `without` taken out of its inputs, outputs and
# each model step's outputs; a model step whose outputs held one loses its
# recorded message (it would show the field); without `steps`, its inputs and
# outputs only. With steps, leaving out the field that holds a step's tool
# calls is refused (turn-invalid): tool steps would answer no call.
shown_turn <- function(turn, entry) {
  left <- unlist(entry$without)
  drop <- function(x) { x <- x %||% list(); x <- x[!names(x) %in% left]; if (length(x)) x else lmcc::jobj() }
  out <- list(signature = turn$signature, inputs = drop(turn$inputs))
  if (isTRUE(entry$steps)) {
    calls_fields <- unique(unlist(lapply(turn$steps, function(st) st$calls_field)))
    if (any(calls_fields %in% left)) return(list(refuses = "turn-invalid"))
    out$steps <- lapply(turn$steps, function(st) {
      if (!identical(st$kind, "model")) return(st)
      held <- any(names(st$outputs) %in% left)
      st$outputs <- drop(st$outputs)
      if (held) st$message <- NULL
      st
    })
  } else out$steps <- list()
  out["outputs"] <- list(if (is.null(turn$outputs)) NULL else drop(turn$outputs))
  list(slot = entry$slot %||% "turns", turn = out)
}
