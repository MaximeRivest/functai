# The reply cache (contract/replies.md): a model's reply to a request, reused
# for the same request. Off unless asked: `cache_replies = TRUE` keeps replies
# in this R session's memory; `"disk"` (or a folder, or a `.sqlite` file)
# keeps them in one SQLite file every process on the machine shares, Python's
# and Julia's included, so a long run stopped and started again sends only
# what has no kept reply. A reply is kept only once it was read; one flight
# per key, in this session and across processes (a claim with a lease).

REPLY_FORMAT <- 1L
REPLY_LEASE <- 120

#' The reply cache's key of a request
#'
#' `"sha256:"` and the SHA-256 of the canonical JSON of `{"functai_reply":
#' 1, "request": <lm15's canonical JSON of the request>, "replicate": n}`,
#' the same in every language (contract/replies.md).
#' @param request An lm15 request, or its canonical JSON as a list.
#' @param replicate The n-th independent answer to the same request.
#' @return A string.
#' @export
reply_key <- function(request, replicate = 0L) {
  req <- if (inherits(request, "lm15_Request") || inherits(request, "lm15_value")) plain_lm15(request) else request
  lmcc::sha256_of(list(functai_reply = REPLY_FORMAT, request = req, replicate = as.integer(replicate %||% 0L)))
}

reply_owner <- function() {
  if (is.null(the$reply_owner)) the$reply_owner <- sprintf("%s:%d:%s", Sys.info()[["nodename"]], Sys.getpid(), paste(format(openssl::rand_bytes(3)), collapse = ""))
  the$reply_owner
}

response_text <- function(response) lmcc::canonical_json(plain_lm15(response))
response_from <- function(text) tryCatch(lm15::from_dict(lmcc::parse_json(text), "response"), error = function(e) NULL)

# ---------------------------------------------------------------- stores

memory_replies <- function(capacity = 20000L) {
  s <- new.env(parent = emptyenv())
  s$data <- new.env(parent = emptyenv()); s$order <- character(0); s$durable <- FALSE
  s$get <- function(key) s$data[[key]]
  s$put <- function(key, response) {
    s$data[[key]] <- response
    s$order <- c(setdiff(s$order, key), key)
    if (length(s$order) > capacity) { gone <- s$order[seq_len(length(s$order) - capacity)]; rm(list = gone, envir = s$data); s$order <- setdiff(s$order, gone) }
  }
  s$discard <- function(key) if (exists(key, envir = s$data, inherits = FALSE)) { rm(list = key, envir = s$data); s$order <- setdiff(s$order, key) }
  s$clear <- function() { rm(list = ls(s$data, all.names = TRUE), envir = s$data); s$order <- character(0) }
  s$size <- function() length(s$order)
  structure(s, class = "functai_replies")
}

#' Where `cache_replies = "disk"` keeps replies
#'
#' The user's cache folder: `$XDG_CACHE_HOME/functai/replies.sqlite`
#' (`~/.cache/functai/replies.sqlite` on Linux),
#' `~/Library/Caches/functai/replies.sqlite` on macOS,
#' `%LOCALAPPDATA%\functai\replies.sqlite` on Windows: the file FunctAI in
#' every language shares.
#' @return A path.
#' @export
default_reply_cache <- function() {
  sys <- Sys.info()[["sysname"]]
  base <- if (sys == "Darwin") file.path(path.expand("~"), "Library", "Caches")
    else if (.Platform$OS.type == "windows") Sys.getenv("LOCALAPPDATA", file.path(path.expand("~"), "AppData", "Local"))
    else { x <- Sys.getenv("XDG_CACHE_HOME"); if (nzchar(x)) x else file.path(path.expand("~"), ".cache") }
  file.path(base, "functai", "replies.sqlite")
}

disk_replies <- function(path = NULL) {
  rlang::check_installed(c("DBI", "RSQLite"), reason = "to keep replies on disk (cache_replies = \"disk\")")
  path <- path.expand(path %||% default_reply_cache())
  if (!grepl("\\.(sqlite3?|db)$", path)) path <- file.path(path, "replies.sqlite")
  path <- normalizePath(path, mustWork = FALSE)
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE, mode = "0700")
  fresh <- !file.exists(path)
  db <- DBI::dbConnect(RSQLite::SQLite(), path)
  DBI::dbExecute(db, "PRAGMA busy_timeout = 30000")
  DBI::dbGetQuery(db, "PRAGMA journal_mode = WAL")
  DBI::dbExecute(db, "PRAGMA synchronous = NORMAL")
  DBI::dbExecute(db, "CREATE TABLE IF NOT EXISTS replies (key TEXT PRIMARY KEY, format INTEGER NOT NULL, created TEXT NOT NULL, model TEXT, response TEXT NOT NULL)")
  DBI::dbExecute(db, "CREATE TABLE IF NOT EXISTS claims (key TEXT PRIMARY KEY, owner TEXT NOT NULL, until REAL NOT NULL)")
  if (fresh) Sys.chmod(path, "0600")
  s <- new.env(parent = emptyenv())
  s$path <- path; s$db <- db; s$durable <- TRUE
  transaction <- function(code) {
    DBI::dbExecute(db, "BEGIN IMMEDIATE")
    ok <- FALSE
    on.exit(if (!ok) DBI::dbExecute(db, "ROLLBACK"))
    out <- force(code)
    DBI::dbExecute(db, "COMMIT"); ok <- TRUE
    out
  }
  s$get <- function(key) {
    row <- DBI::dbGetQuery(db, "SELECT response, format FROM replies WHERE key = ?", params = list(key))
    if (!nrow(row) || row$format[[1L]] != REPLY_FORMAT) return(NULL)
    response_from(row$response[[1L]])
  }
  s$put <- function(key, response) transaction({
    DBI::dbExecute(db, "INSERT OR REPLACE INTO replies (key, format, created, model, response) VALUES (?, ?, ?, ?, ?)",
                   params = list(key, REPLY_FORMAT, iso(as.numeric(Sys.time())), as.character(plain_lm15(response)$model %||% NA_character_), response_text(response)))
    DBI::dbExecute(db, "DELETE FROM claims WHERE key = ? AND owner = ?", params = list(key, reply_owner()))
  })
  s$discard <- function(key) DBI::dbExecute(db, "DELETE FROM replies WHERE key = ?", params = list(key))
  s$clear <- function() { DBI::dbExecute(db, "DELETE FROM replies"); DBI::dbExecute(db, "DELETE FROM claims") }
  s$size <- function() DBI::dbGetQuery(db, "SELECT COUNT(*) AS n FROM replies")$n[[1L]]
  # a claim for one flight across processes: TRUE (this process asks the model) or FALSE (another holds it)
  s$claim <- function(key) transaction({
    now <- as.numeric(Sys.time())
    row <- DBI::dbGetQuery(db, "SELECT owner, until FROM claims WHERE key = ?", params = list(key))
    mine <- !nrow(row) || identical(row$owner[[1L]], reply_owner()) || row$until[[1L]] < now
    if (mine) DBI::dbExecute(db, "INSERT OR REPLACE INTO claims (key, owner, until) VALUES (?, ?, ?)", params = list(key, reply_owner(), now + REPLY_LEASE))
    mine
  })
  s$unclaim <- function(key) DBI::dbExecute(db, "DELETE FROM claims WHERE key = ? AND owner = ?", params = list(key, reply_owner()))
  structure(s, class = "functai_replies")
}

#' @export
print.functai_replies <- function(x, ...) {
  cat(sprintf("<reply cache%s> %d replies\n", if (isTRUE(x$durable)) paste0(" ", x$path) else " in memory", x$size()))
  invisible(x)
}

# The store a cache_replies setting names, or NULL when off.
reply_store <- function(setting) {
  if (is.null(setting) || isFALSE(setting)) return(NULL)
  if (is.null(the$memory_replies)) the$memory_replies <- memory_replies()
  if (isTRUE(setting) || identical(setting, "memory")) return(the$memory_replies)
  if (is_str(setting)) {
    path <- if (identical(setting, "disk")) default_reply_cache() else setting
    where <- normalizePath(path.expand(path), mustWork = FALSE)
    if (is.null(the$disk_replies[[where]])) the$disk_replies[[where]] <- disk_replies(path)
    return(the$disk_replies[[where]])
  }
  if ((is.environment(setting) || is.list(setting)) && is.function(setting$get) && is.function(setting$put)) return(setting)
  cli::cli_abort("{.arg cache_replies} is FALSE, TRUE (memory), \"disk\", a folder or .sqlite path, or a list with {.code get(key)} and {.code put(key, reply)}")
}

#' Forget kept replies
#'
#' @param which The cache: `"memory"` (this session's, the default), `"disk"`,
#'   a path, or a store.
#' @return Nothing, invisibly.
#' @export
clear_replies <- function(which = "memory") {
  s <- reply_store(if (identical(which, "memory")) TRUE else which)
  if (is.function(s$clear)) s$clear()
  invisible()
}

# ---------------------------------------------------------------- a request's turn at the cache

# Before a request is sent: a reply its resumed turn recorded (replies.R
# defers to conversations.R's recorded_reply()), else the cache's: a kept
# reply (`hit`), the key taken for this job (one flight), or `wait` (another
# job or process holds it: the job tries again shortly).
reply_lookup <- function(job, request) {
  job$replayed <- FALSE
  job$hit <- NULL
  recorded <- recorded_reply(job, request)
  if (!is.null(recorded)) { job$replayed <- TRUE; return(recorded) }
  job$flight <- NULL
  store <- tryCatch(reply_store(job$settings$cache_replies), error = function(e) {
    warn_once(paste0("replies:", conditionMessage(e)), sprintf("the reply cache cannot be opened (%s); replies are not cached", conditionMessage(e)))
    NULL
  })
  if (is.null(store)) return(NULL)
  # a disk cache keeps whole requests and replies: never for a call whose log_content drops a field
  if (!isFALSE(store$durable) && !keep_whole(job$call$keep_events)) store <- reply_store(TRUE)
  key <- reply_key(request, job$settings$replicate %||% 0L)
  hit <- tryCatch(store$get(key), error = function(e) NULL)
  if (!is.null(hit)) { job$hit <- list(store = store, key = key); return(hit) }
  holder <- the$flights[[key]]
  if (!is.null(holder) && !identical(holder, job$call$id)) return(structure(list(), class = "functai_wait"))
  if (is.function(store$claim) && !isTRUE(tryCatch(store$claim(key), error = function(e) TRUE))) return(structure(list(), class = "functai_wait"))
  the$flights[[key]] <- job$call$id
  job$flight <- list(store = store, key = key)
  NULL
}

# The reply was read: keep it (a cache that cannot write never fails a call).
reply_keep <- function(job, response) {
  f <- job$flight
  if (is.null(f)) return(invisible())
  tryCatch(f$store$put(f$key, response), error = function(e)
    warn_once(paste0("replies-put:", conditionMessage(e)), sprintf("a reply could not be kept in the cache (%s)", conditionMessage(e))))
  reply_end(job)
}

# An unreadable reply: a kept one is forgotten; this flight ends.
reply_forget <- function(job) {
  h <- job$hit
  if (!is.null(h) && is.function(h$store$discard)) tryCatch(h$store$discard(h$key), error = function(e) NULL)
  job$hit <- NULL
  f <- job$flight
  if (is.null(f)) return(invisible())
  reply_end(job)
}

reply_drop <- function(job) reply_end(job)

reply_end <- function(job) {
  f <- job$flight
  if (is.null(f)) return(invisible())
  job$flight <- NULL
  the$flights[[f$key]] <- NULL
  if (is.function(f$store$unclaim)) tryCatch(f$store$unclaim(f$key), error = function(e) NULL)
  invisible()
}
