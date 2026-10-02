# Where conversations are kept (contract/conversations.md, "Stores"). A store
# keeps each conversation as an ordered list of records (JSON objects) and
# never changes one. A store is a list (or an environment) of functions:
#
# - `append(conversation, records, expect = NULL)`: adds the records at the
#   end, all or none, giving each its `seq`; with `expect`, only when the
#   conversation holds exactly `expect` records (else `store-conflict`).
#   Returns how many it holds after.
# - `read(conversation, after = 0)`: the records after position `after`.
#
# and may have `events` (a store of call tree logs: each turn's kept log, so
# another process follows it), `wait(conversation, after, timeout)`,
# `durability` and `persistent`. Two are here: memory_conversations() (this
# R session's memory) and folder_store() (files in a folder, locked across
# processes: the same files Python's and Julia's FolderStore write).

CONVERSATION_FORMAT <- 1L
CONVERSATION_ID <- "^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$"

conversation_error <- function(code, message, turn = NULL) {
  rlang::error_cnd(c("functai_conversation_error", paste0("functai_", gsub("-", "_", code)), "functai_refusal"), code = code, turn = turn,
                   message = message, functai_type = "ConversationError")
}

check_conversation_id <- function(id) {
  if (!is_str(id) || !grepl(CONVERSATION_ID, id))
    stop(conversation_error("conversation-id", sprintf("a conversation's id is 1 to 200 letters, digits, '.', '_' or '-', starting with a letter or digit; not %s", short_json(id))))
  id
}

#' Where conversations are kept
#'
#' `memory_conversations()` keeps them in this R session's memory (lost when
#' it ends): the store a conversation uses when none is named.
#' `folder_store()` keeps them in a folder, shared by every process that
#' opens it (Python's, Julia's and R's on the same folder):
#' `<folder>/conversations/<id>.jsonl` (the records, one a line, each append
#' locked across processes) and `<folder>/trees/<turn>.jsonl` (each turn's
#' events, so another process can watch it). Files are readable by their
#' owner only. A store of your own is a list of `append(conversation,
#' records, expect = NULL)` and `read(conversation, after = 0)` functions
#' (contract/conversations.md, *Stores*).
#'
#' R cannot ask the system to flush a file to the disk: the folder store
#' writes and closes each append (the operating system keeps it; a power cut
#' can lose the last records), where Python's flushes each to the disk.
#' @param folder The folder.
#' @return A store.
#' @export
folder_store <- function(folder) {
  rlang::check_installed("filelock", reason = "to keep conversations in a folder, locked across processes")
  root <- normalizePath(path.expand(folder), mustWork = FALSE)
  dir.create(file.path(root, "conversations"), recursive = TRUE, showWarnings = FALSE, mode = "0700")
  if (!is.null(the$folder_stores[[root]])) return(the$folder_stores[[root]])
  s <- new.env(parent = emptyenv())
  s$folder <- root; s$durability <- "written"; s$persistent <- TRUE
  s$events <- folder_events(file.path(root, "trees"))
  s$files <- new.env(parent = emptyenv())
  file_of <- function(c, ext) file.path(root, "conversations", paste0(check_conversation_id(c), ".", ext))
  lines_of <- function(c) { key <- c; if (is.null(s$files[[key]])) s$files[[key]] <- json_lines(file_of(c, "jsonl")); s$files[[key]] }
  s$append <- function(conversation, records, expect = NULL) with_file_lock(file_of(conversation, "lock"), {
    n <- length(lines_load(lines_of(conversation)))
    if (!is.null(expect) && expect != n) stop(conversation_error("store-conflict", sprintf("conversation %s holds %d records, not %d", conversation, n, expect)))
    text <- character(0)
    for (r in records) { n <- n + 1L; r$seq <- n; text <- c(text, lmcc::json_text(r)) }
    if (length(text)) append_lines(file_of(conversation, "jsonl"), text)
    n
  })
  s$read <- function(conversation, after = 0L) { items <- lines_load(lines_of(conversation)); if (after >= length(items)) list() else items[(after + 1L):length(items)] }
  s$wait <- function(conversation, after, timeout) {
    deadline <- as.numeric(Sys.time()) + timeout; pause <- 0.02
    while (as.numeric(Sys.time()) < deadline) {
      if (length(lines_load(lines_of(conversation))) > after) return(invisible())
      Sys.sleep(min(pause, max(0, deadline - as.numeric(Sys.time())))); pause <- min(2 * pause, 0.2)
    }
  }
  s$conversations <- function() sub("\\.jsonl$", "", sort(list.files(file.path(root, "conversations"), pattern = "\\.jsonl$")))
  the$folder_stores[[root]] <- structure(s, class = c("functai_folder_store", "functai_conversation_store"))
  the$folder_stores[[root]]
}

#' @rdname folder_store
#' @export
memory_conversations <- function() {
  s <- new.env(parent = emptyenv())
  s$records <- list(); s$durability <- "memory"; s$persistent <- FALSE
  s$events <- memory_store("conversations")
  s$append <- function(conversation, records, expect = NULL) {
    check_conversation_id(conversation)
    log <- s$records[[conversation]] %||% list()
    if (!is.null(expect) && expect != length(log)) stop(conversation_error("store-conflict", sprintf("conversation %s holds %d records, not %d", conversation, length(log), expect)))
    for (r in records) { r$seq <- length(log) + 1L; log[[length(log) + 1L]] <- r }
    s$records[[conversation]] <- log
    length(log)
  }
  s$read <- function(conversation, after = 0L) { log <- s$records[[conversation]] %||% list(); if (after >= length(log)) list() else log[(after + 1L):length(log)] }
  s$wait <- function(conversation, after, timeout) Sys.sleep(min(timeout, 0.1))
  s$conversations <- function() sort(names(Filter(length, s$records)))
  structure(s, class = c("functai_memory_conversations", "functai_conversation_store"))
}

#' @export
print.functai_conversation_store <- function(x, ...) {
  cat(sprintf("<conversation store%s> %d conversations\n", if (!is.null(x$folder)) paste0(" ", x$folder) else " in memory", length(x$conversations())))
  invisible(x)
}

# The store a `store` value names: NULL (this session's memory), TRUE (the
# default folder), a folder, or a store.
conversation_store <- function(store) {
  if (is.null(store) || isFALSE(store)) { if (is.null(the$memory_conversations)) the$memory_conversations <- memory_conversations(); return(the$memory_conversations) }
  if (isTRUE(store)) store <- file.path(dirname(default_log_folder()), "conversations")
  if (is_str(store)) return(folder_store(store))
  if ((is.environment(store) || is.list(store)) && is.function(store$append) && is.function(store$read)) return(store)
  cli::cli_abort("{.arg store} is NULL (memory), TRUE (the default folder), a folder, or a list of {.code append} and {.code read} functions")
}

store_persistent <- function(store) !isFALSE(store$persistent)

# ---------------------------------------------------------------- files

# Run `code` holding an exclusive lock on a file, across processes (flock, as
# Python's FolderStore takes on the same file).
with_file_lock <- function(path, code) {
  lock <- filelock::lock(path, exclusive = TRUE, timeout = 60000)
  if (is.null(lock)) stop(sprintf("could not lock %s in 60 seconds", path), call. = FALSE)
  on.exit(filelock::unlock(lock))
  if (file.exists(path)) Sys.chmod(path, "0600")
  force(code)
}

append_lines <- function(path, lines) {
  fresh <- !file.exists(path)
  con <- file(path, open = "ab")
  on.exit(close(con))
  writeBin(charToRaw(enc2utf8(paste0(paste(lines, collapse = "\n"), "\n"))), con)
  if (fresh) Sys.chmod(path, "0600")
}

# A JSON lines file read incrementally: what was parsed is kept, and only
# what was appended since is read again (a line still being written is read
# next time).
json_lines <- function(path) { l <- new.env(parent = emptyenv()); l$path <- path; l$size <- 0; l$items <- list(); l }
lines_load <- function(l) {
  size <- if (file.exists(l$path)) file.size(l$path) else 0
  if (size < l$size) { l$size <- 0; l$items <- list() }
  if (size > l$size) {
    con <- file(l$path, open = "rb"); on.exit(close(con))
    seek(con, l$size)
    data <- readBin(con, "raw", size - l$size)
    nl <- which(data == as.raw(10L))
    if (length(nl)) {
      stop_at <- nl[[length(nl)]]
      text <- rawToChar(data[seq_len(stop_at)]); Encoding(text) <- "UTF-8"
      for (raw in strsplit(text, "\n", fixed = TRUE)[[1L]]) {
        if (!nzchar(trimws(raw))) next
        l$items[[length(l$items) + 1L]] <- tryCatch(lmcc::parse_json(raw), error = function(e) list(unreadable = TRUE))
      }
      l$size <- l$size + stop_at
    }
  }
  l$items
}

# Call tree logs kept in files, by the rules every store keeps
# (contract/streaming.md): `<folder>/<tree>.jsonl` the kept events,
# `<tree>.writer` the last writer number a claim gave, `<tree>.lock` held for
# each claim and append.
folder_events <- function(folder) {
  dir.create(folder, recursive = TRUE, showWarnings = FALSE, mode = "0700")
  s <- new.env(parent = emptyenv())
  s$folder <- normalizePath(folder, mustWork = FALSE); s$files <- new.env(parent = emptyenv())
  file_of <- function(tree, ext) {
    if (!is_str(tree) || !grepl(CONVERSATION_ID, tree)) store_refuse("event-malformed", sprintf("%s is not a log's id", short_json(tree)))
    file.path(s$folder, paste0(tree, ".", ext))
  }
  events_of <- function(tree) {
    if (is.null(s$files[[tree]])) s$files[[tree]] <- json_lines(file_of(tree, "jsonl"))
    lapply(Filter(function(x) is.list(x) && is.null(x$unreadable), lines_load(s$files[[tree]])), as_event)
  }
  writer_of <- function(tree) { p <- file_of(tree, "writer"); if (!file.exists(p)) 1L else as.integer(trimws(readLines(p, warn = FALSE)[[1L]])) }
  s$append <- function(events) {
    batch <- is.list(events) && !inherits(events, "functai_event") && is.null(names(events))
    xs <- if (batch) events else list(events)
    if (!length(xs)) return("duplicate")
    tree <- xs[[1L]]$tree
    if (!is_str(tree)) store_refuse("event-malformed", "not an event")
    with_file_lock(file_of(tree, "lock"), {
      log <- events_of(tree); n <- length(log); writer <- writer_of(tree); answers <- character(0)
      for (one in xs) {
        r <- tryCatch(append_one(log, writer, one, tree), functai_store_refusal = function(e) {
          if (is.null(e$event)) e$event <- stated_position(if (inherits(one, "functai_event")) unclass(one) else one)
          stop(e)
        })
        log <- r$log; answers <- c(answers, r$answer)
      }
      if (length(log) > n) append_lines(file_of(tree, "jsonl"), vapply(log[(n + 1L):length(log)], function(e) lmcc::json_text(unclass(e)), ""))
      if (all(answers == "duplicate")) "duplicate" else "kept"
    })
  }
  s$claim <- function(tree) with_file_lock(file_of(tree, "lock"), {
    log <- events_of(tree)
    if (!length(log)) store_refuse("event-unknown", sprintf("no log %s", tree))
    if (is_log_end(log, tree)) store_refuse("event-after-end", sprintf("the log %s is finished", tree))
    w <- writer_of(tree) + 1L
    p <- file_of(tree, "writer"); tmp <- paste0(p, ".tmp")
    writeLines(as.character(w), tmp); file.rename(tmp, p)
    list(writer = w, after = event_position(log[[length(log)]]))
  })
  s$read <- function(tree, after = NULL) resume_events(events_of(tree), after)
  s$trees <- function() sub("\\.jsonl$", "", list.files(s$folder, pattern = "\\.jsonl$"))
  s$writer_of <- writer_of
  s$finished <- function(tree) is_log_end(events_of(tree), tree)
  structure(s, class = "functai_event_store")
}
