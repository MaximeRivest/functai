# The call log (contract/calls.md): every call of an AI function as one line
# of JSON in a folder, ratings of those calls, and the rows with known
# answers they make. The folder is the interface: Python, TypeScript and R
# read and write the same one.

CALL_FORMAT <- 2L            # the call log R writes (calls.md, "Formats")
CALL_FORMATS <- c(1L, 2L)    # the ones it reads
RATING_FORMAT <- 1L
MAX_LINE <- 8 * 1024^2
OFF <- c("", "0", "false", "no", "off")
ON <- c("1", "true", "yes", "on")

# ---------------------------------------------------------------- ids, times

new_id <- function() {
  ms <- floor(as.numeric(Sys.time()) * 1000)
  bytes <- openssl::rand_bytes(16)
  for (i in 6:1) { bytes[[i]] <- as.raw(ms %% 256); ms <- ms %/% 256 }
  bytes[[7]] <- as.raw(bitwOr(bitwAnd(as.integer(bytes[[7]]), 0x0f), 0x70))
  bytes[[9]] <- as.raw(bitwOr(bitwAnd(as.integer(bytes[[9]]), 0x3f), 0x80))
  h <- paste(format(bytes), collapse = "")
  paste(substr(h, 1, 8), substr(h, 9, 12), substr(h, 13, 16), substr(h, 17, 20), substr(h, 21, 32), sep = "-")
}

# A time in the log's format (RFC 3339 UTC, exactly six fraction digits),
# rounded to the microsecond (never truncated: 0.01 is .010000).
iso <- function(t) {
  us <- round(as.numeric(t) * 1e6)
  secs <- us %/% 1e6
  paste0(format(as.POSIXct(secs, origin = "1970-01-01", tz = "UTC"), "%Y-%m-%dT%H:%M:%S", tz = "UTC"), sprintf(".%06dZ", as.integer(us - secs * 1e6)))
}

# Byte order, never the locale's (the contract compares code points).
before <- function(a, b) !identical(a, b) && order(c(a, b), method = "radix")[[1L]] == 1L

# ---------------------------------------------------------------- where

#' Where calls are logged when no folder is named
#'
#' `$XDG_DATA_HOME/functai/calls` (`~/.local/share/functai/calls` on Linux),
#' `~/Library/Application Support/functai/calls` on macOS,
#' `%LOCALAPPDATA%\functai\calls` on Windows: the folder FunctAI in every
#' language shares. Nothing is written there unless you turn the log on.
#' @return A path.
#' @export
default_log_folder <- function() {
  sys <- Sys.info()[["sysname"]]
  base <- if (sys == "Darwin") file.path(path.expand("~"), "Library", "Application Support")
    else if (.Platform$OS.type == "windows") Sys.getenv("LOCALAPPDATA", file.path(path.expand("~"), "AppData", "Local"))
    else { x <- Sys.getenv("XDG_DATA_HOME"); if (nzchar(x)) x else file.path(path.expand("~"), ".local", "share") }
  file.path(base, "functai", "calls")
}

folder_of <- function(setting) {
  if (isFALSE(setting)) return(NULL)
  raw <- trimws(Sys.getenv("FUNCTAI_LOG_CALLS"))
  on <- !tolower(raw) %in% OFF
  env_folder <- if (on && !tolower(raw) %in% ON) raw else NULL
  if (is.null(setting) && !on) return(NULL)
  folder <- if (is.null(setting) || isTRUE(setting)) env_folder %||% default_log_folder() else setting
  normalizePath(path.expand(folder), mustWork = FALSE)
}

caller_of <- function(settings) {
  raw <- Sys.getenv("FUNCTAI_CALLER")
  base <- list()
  if (nzchar(raw)) {
    parsed <- tryCatch(lmcc::parse_json(raw), error = function(e) NULL)
    if (is.list(parsed) && !is.null(names(parsed))) base <- parsed
    else warn_once(paste0("caller:", raw), "$FUNCTAI_CALLER is not a JSON object; ignored")
  }
  own <- settings$caller %||% list()
  for (k in names(own)) base[k] <- list(own[[k]])
  if (!length(base)) lmcc::jobj() else base
}

warn_once <- function(key, message) {
  if (isTRUE(the$warned[[key]])) return(invisible())
  the$warned[[key]] <- TRUE
  cli::cli_warn(message, .envir = parent.frame())
}

# ---------------------------------------------------------------- a call

size_of <- function(v) nchar(lmcc::canonical_json(v), type = "chars")

# A call, from its start: `fields` are its fields (content_kept()'s names),
# `keep` whether each one's value is written, `own` the program's own settings
# (its layer of observers and journal). It joins the tree of the call it runs
# inside, or starts one (runtime.R, open_call()).
start_call <- function(program, settings, inputs, fields, keep, own = list(), later = NULL, id = NULL) {
  call <- new.env(parent = emptyenv())
  parent <- the$current
  if (!is.null(parent) && isTRUE(parent$ended)) parent <- NULL
  call$id <- id %||% new_id()
  call$parent <- if (is.null(parent)) the$remote_parent else parent$id
  call$root <- if (is.null(parent)) call$id else parent$root
  call$program <- program
  call$program_json <- program()
  call$started <- as.numeric(Sys.time())
  call$exchanges <- list()
  call$provider <- NULL
  call$outputs <- NULL
  call$probabilities <- NULL
  call$folder <- tryCatch(folder_of(settings$log_calls), error = function(e) NULL)
  call$fields <- fields
  call$keep <- keep
  call$keep_events <- keep_of(fields, keep)
  call$caller <- caller_of(settings)
  call$inputs <- if (length(inputs)) inputs else lmcc::jobj()
  call$later <- later
  open_call(call, own)
}

# One request and its reply (or error). `request_hash`: lmcc's hash of the
# rendered request it was sent from (kernel section 3a, a model step's
# `request`); a request asked again after an unreadable reply extends that
# render with FunctAI's own words, and carries its hash.
exchange <- function(call, model, request, response, started, seconds, error = NULL, request_hash = NULL,
                     cached = FALSE, streamed = FALSE, first_delta = NULL) {
  call$exchanges[[length(call$exchanges) + 1L]] <- list(model = model, provider = call$provider, started = started,
    seconds = seconds, request = request, request_hash = request_hash, response = response, error = error,
    cached = cached, streamed = streamed, first_delta = first_delta)
}

error_json <- function(err) {
  cls <- class(err)
  type <- if (inherits(err, "lmcc_refusal")) "Refusal"
    else if (inherits(err, c("functai_interface_input", "functai_interface_output"))) "InterfaceError"
    else if (!is.null(err$functai_type)) err$functai_type
    else if (inherits(err, "functai_journal_error")) "JournalError"
    else if (inherits(err, "functai_cancelled")) "Cancelled"
    else if (inherits(err, "functai_store_refusal")) "EventRefused"
    else if (inherits(err, "LM15Error")) (setdiff(cls, c("LM15Error", "error", "condition"))[1L] %|na|% "LM15Error") else cls[[1L]]
  out <- list(type = type)
  if (inherits(err, "lmcc_refusal") || inherits(err, "functai_refusal")) out$code <- err$code
  out$message <- conditionMessage(err)
  out
}
`%|na|%` <- function(x, y) if (is.na(x)) y else x

plain_lm15 <- function(x) lmcc::lm15_plain(lm15::as_dict(x))

usage_of <- function(response) {
  u <- plain_lm15(response)$usage %||% list()
  u <- Filter(function(v) is.numeric(v) && length(v) == 1L && v == round(v), u)
  if (!length(u)) lmcc::jobj() else lapply(u, as.integer)
}

exchange_json <- function(ex) {
  out <- list(model = ex$model, provider = ex$provider, started = iso(ex$started), seconds = round(ex$seconds, 6), cached = isTRUE(ex$cached))
  if (!is.null(ex$response)) { out$finish <- ex$response$finish_reason; out$usage <- usage_of(ex$response) }
  if (!is.null(ex$error)) out$error <- error_json(ex$error)
  out$request <- plain_lm15(ex$request)
  if (!is.null(ex$request_hash)) out$request_hash <- ex$request_hash
  if (!is.null(ex$response)) out$response <- plain_lm15(ex$response)
  if (isTRUE(ex$streamed)) { out$streamed <- TRUE; out["first_delta"] <- list(if (is.null(ex$first_delta)) NULL else round(ex$first_delta, 6)) }
  out
}

process_json <- function() {
  if (is.null(the$process)) {
    info <- Sys.info()
    the$process <- list(host = info[["nodename"]], pid = Sys.getpid(), user = info[["user"]], language = "r",
                        runtime = paste(R.version$major, R.version$minor, sep = "."),
                        functai = as.character(utils::packageVersion("functai")))
    # the libraries that build the request and send it (calls.md, process)
    for (lib in c("lmcc", "lm15")) {
      v <- tryCatch(as.character(utils::packageVersion(lib)), error = function(e) NULL)
      if (!is.null(v)) the$process[[lib]] <- v
    }
  }
  the$process
}

# The call's record (format 2): the whole record, then what log_content lets
# it keep (content.R, kept_record()).
call_record <- function(call, error = NULL) {
  program <- call$program_json %||% call$program()
  answered <- Filter(function(e) !is.null(e$response), call$exchanges)
  usage <- list()
  for (e in answered) for (k in names(u <- usage_of(e$response))) usage[[k]] <- (usage[[k]] %||% 0L) + u[[k]]
  if (!is.null(call$outputs)) {                 # in the fields' order (the ones FunctAI adds first, as lmcc's signature has them)
    ord <- c(intersect(call$fields$outputs, names(call$outputs)), setdiff(names(call$outputs), call$fields$outputs))
    call$outputs <- call$outputs[ord]
  }
  in_sizes <- lmcc::jobj(); out_sizes <- lmcc::jobj()
  for (k in names(call$inputs)) in_sizes[[k]] <- size_of(call$inputs[[k]])
  for (k in names(call$outputs)) out_sizes[[k]] <- size_of(call$outputs[[k]])
  rec <- list(functai_call = CALL_FORMAT, id = call$id, parent = call$parent, root = call$root, program = program,
              started = iso(call$started), seconds = round((call$ended_at %||% as.numeric(Sys.time())) - call$started, 6), content = TRUE)
  rec["inputs"] <- list(call$inputs)
  rec["outputs"] <- list(if (is.null(call$outputs)) NULL else if (length(call$outputs)) call$outputs else lmcc::jobj())
  probabilities <- Filter(length, call$probabilities %||% list())
  if (length(probabilities) && !is.null(call$outputs)) rec$probabilities <- probabilities
  rec$sizes <- list(inputs = in_sizes, outputs = out_sizes)
  rec["error"] <- list(if (is.null(error)) NULL else error_json(error))
  rec["model"] <- list(if (length(answered)) answered[[length(answered)]]$model else NULL)
  rec$usage <- if (length(usage)) usage else lmcc::jobj()
  rec["confidence"] <- list(call$confidence)
  rec$exchanges <- lapply(call$exchanges, exchange_json)
  rec$saw <- call$saw %||% list()
  if (!is.null(call$returned)) rec["returned"] <- list(call$returned)
  if (isTRUE(call$escalated)) rec$escalated <- TRUE
  rec$caller <- call$caller
  rec$process <- process_json()
  if (!is.null(call$steps)) rec$steps <- call$steps
  if (!is.null(call$invocation)) rec$invocation <- call$invocation
  if (!is.null(call$conversation)) rec$conversation <- call$conversation
  if (length(call$changes)) rec$changes <- call$changes
  if (length(call$sections)) rec$sections <- as.list(call$sections)
  if (isFALSE(call$replayable)) rec$replayable <- FALSE
  if (!is.null(call$tree) && call$tree$writer > 1L) rec$writer <- call$tree$writer
  if (!is.null(call$journal_word)) rec$journal <- call$journal_word
  kept_record(rec, call$fields, call$keep)
}

record_line <- function(rec) {
  line <- lmcc::json_text(rec)
  if (nchar(line, type = "bytes") <= MAX_LINE) return(line)
  rec$truncated <- TRUE
  rec$exchanges <- lapply(rec$exchanges, function(e) { e$request <- NULL; e$response <- NULL; e })
  line <- lmcc::json_text(rec)
  if (nchar(line, type = "bytes") <= MAX_LINE) return(line)
  rec$inputs <- NULL; rec$outputs <- NULL
  lmcc::json_text(rec)
}

log_file <- function() {
  if (is.null(the$log_name)) {
    rand <- paste(format(openssl::rand_bytes(3)), collapse = "")
    the$log_name <- sprintf("%s-%d-%s.jsonl", Sys.info()[["nodename"]], Sys.getpid(), rand)
  }
  the$log_name
}

append_record <- function(folder, rec) {
  day <- file.path(folder, format(Sys.time(), "%Y-%m-%d", tz = "UTC"))
  if (!dir.exists(day)) { dir.create(day, recursive = TRUE, mode = "0700"); Sys.chmod(c(folder, day), "0700") }
  path <- file.path(day, log_file())
  new <- !file.exists(path)
  con <- file(path, open = "ab")
  on.exit(close(con))
  writeBin(charToRaw(enc2utf8(paste0(record_line(rec), "\n"))), con)
  if (new) Sys.chmod(path, "0600")
}

finish_call <- function(call, error = NULL) {
  if (is.null(call$folder)) return(invisible())
  tryCatch(append_record(call$folder, call_record(call, error)), error = function(e)
    warn_once(paste0(call$folder, conditionMessage(e)), sprintf("could not log a call to %s (%s); calls go on, unlogged", call$folder, conditionMessage(e))))
  invisible()
}

# ---------------------------------------------------------------- reading

read_log <- function(folder = NULL, since = NULL) {
  root <- normalizePath(path.expand(folder %||% folder_of(TRUE)), mustWork = FALSE)
  cutoff <- if (is.null(since)) "" else iso(time_since(since))
  calls <- list(); ratings <- list()
  if (!dir.exists(root)) return(list(calls = calls, ratings = ratings))
  # the files at the top (what prune_calls() kept), then each day's
  files <- sort(list.files(root, pattern = "\\.jsonl$", full.names = TRUE), method = "radix")
  for (day in sort(list.files(root), method = "radix")) {
    if (!grepl("^[0-9]{4}-[0-9]{2}-[0-9]{2}$", day) || (nzchar(cutoff) && before(day, substr(cutoff, 1, 10)))) next
    files <- c(files, sort(list.files(file.path(root, day), pattern = "\\.jsonl$", full.names = TRUE), method = "radix"))
  }
  {
    for (f in files) {
      lines <- tryCatch(readLines(f, warn = FALSE, encoding = "UTF-8"), error = function(e) character(0))
      for (line in lines) {
        if (!nzchar(trimws(line))) next
        rec <- tryCatch(lmcc::parse_json(line), error = function(e) NULL)
        if (!is.list(rec) || is.null(names(rec))) next
        if (is_format(rec$functai_call, CALL_FORMATS) && !before(rec$started %||% "", cutoff)) calls[[length(calls) + 1L]] <- rec
        else if (is_format(rec$functai_rating, RATING_FORMAT) && !before(rec$at %||% "", cutoff)) ratings[[length(ratings) + 1L]] <- rec
      }
    }
  }
  list(calls = highest_writer(calls), ratings = ratings)
}

# For one id, the record of the highest writer (a resumed turn's call is
# written again by its later writer: calls.md, `writer`).
highest_writer <- function(calls) {
  if (!length(calls)) return(calls)
  ids <- vapply(calls, function(c) as.character(c$id %||% ""), "")
  if (!anyDuplicated(ids)) return(calls)
  w <- vapply(calls, function(c) as.numeric(c$writer %||% 1), 0)
  keep <- vapply(seq_along(calls), function(i) { same <- which(ids == ids[[i]]); i == same[which.max(w[same])] }, NA)
  calls[keep]
}

# A time from `since`: a date or date-time, or text like "90d", "12h", "2w".
time_since <- function(x) {
  if (inherits(x, c("Date", "POSIXt"))) return(as.numeric(as.POSIXct(x, tz = "UTC")))
  if (is.character(x) && length(x) == 1L) {
    m <- regmatches(x, regexec("^\\s*([0-9]+)\\s*([hdw])\\s*$", x))[[1L]]
    if (length(m)) return(as.numeric(Sys.time()) - as.numeric(m[[2L]]) * c(h = 3600, d = 86400, w = 604800)[[m[[3L]]]])
    t <- suppressWarnings(as.POSIXct(x, tz = "UTC", tryFormats = c("%Y-%m-%dT%H:%M:%OS", "%Y-%m-%d %H:%M:%OS", "%Y-%m-%d")))
    if (!is.na(t)) return(as.numeric(t))
  }
  if (is.numeric(x)) return(as.numeric(x))
  cli::cli_abort("a time is a date, a date-time, or text like {.val 2026-09-20}, {.val 90d}, {.val 12h}, {.val 2w}")
}

#' Make the call log smaller, keeping what ratings need
#'
#' Deletes the log's day folders older than `older_than` (the log is kept
#' small by deleting whole days: contract/calls.md, *The folder*). Before a
#' day goes, every rated call in it is kept: the call, every call of its tree
#' (a program's steps), every call its `saw` names (the earlier turns a row
#' asks again) and their ratings are copied into one file at the folder's
#' top level, which every language's reader reads. A row of [rated()] made
#' before pruning is the same after.
#' @param older_than Day folders before this go: `"90d"`, `"12w"`, a date.
#' @param folder The log folder (default: the one calls are logged to here).
#' @param keep_rated `FALSE` deletes rated calls too.
#' @return Counts, invisibly: `days` deleted, `calls` deleted, `kept`.
#' @export
prune_calls <- function(older_than = "90d", folder = NULL, keep_rated = TRUE) {
  root <- normalizePath(path.expand(log_folder(folder) %||% cli::cli_abort("no log folder: pass {.arg folder}")), mustWork = FALSE)
  first <- substr(iso(time_since(older_than)), 1L, 10L)
  none <- list(days = 0L, calls = 0L, kept = 0L)
  if (!dir.exists(root)) return(invisible(none))
  days <- Filter(function(d) grepl("^[0-9]{4}-[0-9]{2}-[0-9]{2}$", d) && before(d, first) && dir.exists(file.path(root, d)), sort(list.files(root), method = "radix"))
  if (!length(days)) return(invisible(none))
  log <- read_log(root)
  by_id <- stats::setNames(log$calls, vapply(log$calls, function(c) as.character(c$id %||% ""), ""))
  old <- list()
  for (d in days) for (f in list.files(file.path(root, d), pattern = "\\.jsonl$", full.names = TRUE))
    for (line in readLines(f, warn = FALSE, encoding = "UTF-8")) {
      rec <- tryCatch(lmcc::parse_json(line), error = function(e) NULL)
      if (is.list(rec) && !is.null(names(rec))) old[[length(old) + 1L]] <- rec
    }
  keep <- character(0)
  if (keep_rated) {
    rated <- unique(vapply(log$ratings, function(r) as.character(r$call %||% ""), ""))
    trees <- unique(unlist(lapply(rated, function(id) by_id[[id]]$root)))
    keep <- names(by_id)[names(by_id) %in% rated | vapply(by_id, function(c) isTRUE(c$root %in% trees), NA)]
    todo <- keep
    while (length(todo)) {
      c <- by_id[[todo[[1L]]]]; todo <- todo[-1L]
      for (entry in c$saw %||% list()) for (k in c("call", "saw_of")) {
        id <- entry[[k]]
        if (is_str(id) && !is.null(by_id[[id]]) && !id %in% keep) { keep <- c(keep, id); todo <- c(todo, id) }
      }
    }
  }
  kept <- Filter(function(r) (!is.null(r$functai_call) && isTRUE(r$id %in% keep)) || (!is.null(r$functai_rating) && isTRUE(r$call %in% keep)), old)
  if (length(kept)) {
    host <- gsub("[^A-Za-z0-9_.-]", "_", Sys.info()[["nodename"]])
    path <- file.path(root, sprintf("kept-%s-%d-%s.jsonl", host, Sys.getpid(), paste(format(openssl::rand_bytes(3)), collapse = "")))
    con <- file(path, open = "wb")
    writeBin(charToRaw(enc2utf8(paste0(vapply(kept, lmcc::json_text, ""), "\n", collapse = ""))), con)
    close(con)
    Sys.chmod(path, "0600")
  }
  for (d in days) unlink(file.path(root, d), recursive = TRUE)
  n_old <- sum(vapply(old, function(r) !is.null(r$functai_call), NA))
  n_kept <- sum(vapply(kept, function(r) !is.null(r$functai_call), NA))
  invisible(list(days = length(days), calls = n_old - n_kept, kept = n_kept))
}

# A record of a format this reader knows (a record of another format is
# skipped whole: a later format may mean something else by the same keys).
is_format <- function(v, known) is_num(v) && num(v) %in% known

later <- function(a, b, key) {
  ta <- a[[key]] %||% ""; tb <- b[[key]] %||% ""
  if (!identical(ta, tb)) return(before(tb, ta))
  before(b$id %||% "", a$id %||% "")
}

# For each call, the ratings that count: each person's latest, none for a
# withdrawn one. A rating with no `by` (made under an account, which may be
# shared) counts on its own: it replaces none, none replaces it, and its null
# verdict withdraws nothing (calls.md, rule 2).
current_ratings <- function(ratings, by = NULL) {
  latest <- list()
  for (r in ratings) {
    if (!is.null(by) && !identical(r$by, by)) next
    key <- if (is_str(r$by) && nzchar(r$by)) paste0(r$call, "\u0001by\u0001", r$by) else paste0(r$call, "\u0001rating\u0001", r$id)
    if (is.null(latest[[key]]) || later(r, latest[[key]], "at")) latest[[key]] <- r
  }
  out <- list()
  for (r in latest) if (identical(r$verdict, "right") || identical(r$verdict, "wrong")) out[[r$call]] <- c(out[[r$call]], list(r))
  lapply(out, function(rs) rs[order(vapply(rs, function(r) paste0(r$at %||% "", "\u0001", r$id %||% ""), ""), method = "radix")])
}

# The values a counting rating gives (calls.md, rule 4): "right" gives the
# call's own answer, when it was recorded as data; "wrong" its correction.
rating_says <- function(rating, call) {
  answer <- call$program$answer %||% "result"
  if (identical(rating$verdict, "right")) {
    outputs <- call$outputs
    described <- answer %in% unlist(call$described$outputs)
    return(if (is.list(outputs) && answer %in% names(outputs) && !described) stats::setNames(list(outputs[[answer]]), answer) else NULL)
  }
  values <- list()
  if ("answer" %in% names(rating)) values[answer] <- list(rating$answer)
  for (k in names(rating$outputs)) if (!k %in% names(values)) values[k] <- list(rating$outputs[[k]])
  if (length(values)) values else NULL
}

# Whether a call's record holds every input as data (calls.md, rule 3):
# format 1's content false kept none; format 2's names what it left out;
# a value written as a description is not data.
all_inputs_kept <- function(call) {
  if (isFALSE(call$content) && (!has_key(call, "omitted") || length(call$omitted$inputs))) return(FALSE)
  if (length(call$described$inputs)) return(FALSE)
  "inputs" %in% names(call)
}

# Whether a call is of the program as it is now (calls.md, rule 3): its
# program.interface is the given interface (format 1, with none: its
# program.signature, which equals it for an AI function with neither
# reasoning nor tools), or its program.signature the given signature.
same_program <- function(call, signature = NULL, interface = NULL) {
  if (is.null(signature) && is.null(interface)) return(TRUE)
  p <- call$program
  by_interface <- !is.null(interface) && identical(if (has_key(p, "interface")) p$interface else p$signature, interface)
  by_signature <- !is.null(signature) && identical(p$signature, signature)
  by_interface || by_signature
}

add_meta <- function(row, meta) {
  for (k in names(meta)) {
    key <- k
    while (key %in% names(row)) key <- paste0("_", key)
    row[key] <- list(meta[[k]])
  }
  row
}

# Rows with known answers (contract/calls.md, "Rows with known answers").
rated_rows <- function(calls, ratings, name, module = NULL, signature = NULL, by = NULL, interface = NULL, file = NULL) {
  calls <- Filter(function(c) is_format(c$functai_call, CALL_FORMATS), calls)
  ratings <- Filter(function(r) is_format(r$functai_rating, RATING_FORMAT), ratings)
  counting <- current_ratings(ratings, by)
  left <- list(other_signature = 0L, no_content = 0L, no_answer = 0L)
  # a program defined at the top level (a notebook, a script) is known by its file too: with a file, a call with none does not match
  mine <- Filter(function(c) identical(c$program$name, name) && (is.null(module) || identical(c$program$module, module)) &&
                   (is.null(file) || identical(c$program$file, file)), calls)
  mine <- mine[order(vapply(mine, function(c) paste0(c$started %||% "", "\u0001", c$id %||% ""), ""), method = "radix")]
  rows <- list()
  for (call in mine) {
    rs <- counting[[call$id]]
    if (!length(rs)) next
    if (!same_program(call, signature, interface)) { left$other_signature <- left$other_signature + 1L; next }
    if (!all_inputs_kept(call)) { left$no_content <- left$no_content + 1L; next }
    said <- lapply(rs, rating_says, call = call)
    usable <- which(!vapply(said, is.null, NA))
    if (!length(usable)) { left$no_answer <- left$no_answer + 1L; next }
    rating <- rs[[usable[[length(usable)]]]]; values <- said[[usable[[length(usable)]]]]
    verdicts <- unique(vapply(rs, function(r) r$verdict, ""))
    spelled <- unique(vapply(said[usable], lmcc::canonical_json, ""))
    answer <- call$program$answer %||% "result"
    row <- call$inputs %||% list()
    if (answer %in% names(values)) row[answer] <- list(values[[answer]])
    for (k in setdiff(names(values), answer)) row[k] <- list(values[[k]])
    row <- add_meta(row, list(call = call$id, version = call$program$version, rating = rating$verdict, rated_by = rating$by %||% NULL,
                              origin = rating$origin %||% "review", sample = rating$sample, disputed = length(verdicts) > 1L || length(spelled) > 1L))
    rows[[length(rows) + 1L]] <- row
  }
  list(rows = rows, left_out = left)
}

# ---------------------------------------------------------------- rating

rating_record <- function(call_id, verdict, answer, outputs, note, reasons, by, origin, sample, settings) {
  # a person when one was named; else the account, which names no one (it may be shared: calls.md, "A rating record")
  person <- by %||% caller_of(settings)$user
  rec <- list(functai_rating = RATING_FORMAT, id = new_id(), call = call_id, at = iso(as.numeric(Sys.time())))
  if (is_str(person) && nzchar(person)) rec$by <- person else rec$account <- process_json()$user %||% "unknown"
  rec["verdict"] <- list(verdict)
  if (!is.null(answer)) rec["answer"] <- list(answer)
  if (length(outputs)) rec$outputs <- outputs
  if (length(reasons)) rec$reasons <- as.list(reasons)
  if (!is.null(note)) rec$note <- note
  rec$origin <- origin %||% "review"
  if (!is.null(sample)) rec$sample <- sample
  rec
}
