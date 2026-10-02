# Learning from conversations (stage 5, contract/calls.md, "Rows that keep
# their context"): a rated call's earlier turns as data its row carries, so
# evaluating and improving ask it again as it was asked.

context_unknown <- function(code, call) rlang::error_cnd(c("functai_saw_unknown", paste0("functai_", gsub("-", "_", code))), code = code, call_id = call,
                                                         message = sprintf("what call %s was shown cannot be shown again (%s)", call, code))

# The turn a saw entry stands for, as data: the record's inputs and outputs
# without the entry's `without`; with `steps`, the record's steps and its
# program.signature.
turn_of_entry <- function(rec, entry) {
  left <- unlist(entry$without)
  keep <- function(x) { x <- x %||% list(); x <- x[!names(x) %in% left]; if (length(x)) x else lmcc::jobj() }
  out <- list(inputs = keep(rec$inputs), outputs = keep(rec$outputs))
  if (isTRUE(entry$steps)) {
    if (!is.list(rec$steps)) stop(context_unknown("not-kept", rec$id))
    out$steps <- rec$steps
    out["signature"] <- list(rec$program$signature)
  }
  out
}

runs_under <- function(rec, ancestor, by_id) {
  seen <- character(0); parent <- rec$parent
  while (is_str(parent) && !parent %in% seen) {
    if (identical(parent, ancestor)) return(TRUE)
    seen <- c(seen, parent); parent <- by_id[[parent]]$parent
  }
  FALSE
}

records_by_id <- function(records) {
  recs <- Filter(function(r) is_obj(r) && is_format(r$functai_call, CALL_FORMATS) && is_str(r$id), records)
  stats::setNames(recs, vapply(recs, function(r) r$id, ""))
}

# The turns a call was shown, as data, when the log keeps what showing them
# again needs (else an error naming why: never a part).
shown_turns <- function(records, id, by_id) {
  got <- read_saw(records, id)
  if (!is.null(got$keeps$refuses)) stop(context_unknown(got$keeps$refuses, got$keeps$call))
  lapply(got$saw, function(e) turn_of_entry(by_id[[e$call]], e))
}

# What a rated call was shown before its own inputs, as data a row carries:
# `earlier`, `conversation`, `sections` (when any) and, for a program's call,
# `helpers` (each call inside it that was shown earlier turns).
earlier_of <- function(call, records) {
  by_id <- records_by_id(records)
  rec <- by_id[[call]]
  if (is.null(rec)) stop(context_unknown("missing-call", call))
  earlier <- if (length(rec$saw)) shown_turns(records, call, by_id) else list()
  out <- list(earlier = earlier)
  out["conversation"] <- list(if (is.list(rec$conversation)) rec$conversation$id else NULL)
  if (length(rec$sections)) out$sections <- rec$sections
  if (identical(rec$program$kind, "module")) {
    inside <- Filter(function(c) identical(c$root, rec$root) && !identical(c$id, call) && runs_under(c, call, by_id) && (length(c$saw) || length(c$sections)), by_id)
    inside <- inside[order(vapply(inside, function(c) paste0(c$started %||% "", "\u0001", c$id), ""), method = "radix")]
    out$helpers <- unname(lapply(inside, function(c) {
      h <- list(program = c$program$name, call = c$id, earlier = shown_turns(records, c$id, by_id))
      if (length(c$sections)) h$sections <- c$sections
      h
    }))
  }
  out
}

needs_context <- function(rec, by_id) {
  if (is.null(rec)) return(FALSE)
  if (length(rec$saw) || is.list(rec$conversation) || length(rec$sections)) return(TRUE)
  if (!identical(rec$program$kind, "module")) return(FALSE)
  any(vapply(by_id, function(c) identical(c$root, rec$root) && (length(c$saw) || length(c$sections)) && runs_under(c, rec$id, by_id), NA))
}

# Rows of calls that were shown earlier turns, each with its context; and how
# many rows were left out because the log cannot show those turns again.
# A row's added key (`call`, or `_call` when the data has a field of that name).
row_meta <- function(row, key) {
  ks <- names(row)
  hit <- rev(which(sub("^_+", "", ks) == key))
  if (length(hit)) row[[hit[[1L]]]] else NULL
}

with_context <- function(rows, calls) {
  by_id <- records_by_id(calls)
  if (!any(vapply(rows, function(r) needs_context(by_id[[row_meta(r, "call") %||% ""]], by_id), NA))) return(list(rows = rows, dropped = 0L))
  meta <- c("call", "version", "rating", "rated_by", "origin", "sample", "disputed")
  out <- list(); dropped <- 0L
  for (r in rows) {
    ctx <- tryCatch(earlier_of(row_meta(r, "call"), calls), functai_saw_unknown = function(e) NULL)
    if (is.null(ctx)) { dropped <- dropped + 1L; next }
    ks <- names(r)
    is_meta <- sub("^_+", "", ks) %in% meta & seq_along(ks) > length(ks) - length(meta)
    rebuilt <- add_meta(r[!is_meta], ctx)
    out[[length(out) + 1L]] <- c(rebuilt, r[is_meta])
  }
  list(rows = out, dropped = dropped)
}
