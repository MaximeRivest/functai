# The call log, as tibbles: every call, people's ratings, and the rows with
# known answers they make (contract/calls.md).

log_folder <- function(folder) folder %||% folder_of(effective()$log_calls %||% TRUE)

#' Every logged call
#'
#' @param fn An AI function or a function's name; `NULL` for every call.
#' @param folder The log folder (default: where calls are logged here).
#' @param since Only calls from this time on (a date or date-time).
#' @return A tibble, one row per call, oldest first: `id`, `started`,
#'   `name`, `module`, `version`, `model`, `seconds`, `inputs` and `outputs`
#'   (list columns of JSON values), `error`, `input_tokens`, `output_tokens`,
#'   `reasoning_tokens`, `total_tokens`, `parent`, `caller`, `language`,
#'   `content` (whether every value was kept), `omitted` (when it was not:
#'   the names of the fields whose values were not kept, as `inputs` and
#'   `outputs`; `NULL` otherwise), `saw` (the earlier calls it was shown,
#'   `NULL` when its record does not say).
#'   Providers count differently: OpenAI's and Anthropic's `output_tokens`
#'   include the model's hidden reasoning, Gemini's leave it out (it is in
#'   `reasoning_tokens`). Every one bills `total_tokens - input_tokens` as
#'   output.
#' @export
calls <- function(fn = NULL, folder = NULL, since = NULL) {
  recs <- read_log(log_folder(folder), since)$calls
  if (!is.null(fn)) {
    name <- if (is.character(fn)) fn else core_of(fn)$definition$name
    module <- if (is.character(fn)) NULL else core_of(fn)$module
    recs <- Filter(function(c) identical(c$program$name, name) && (is.null(module) || identical(c$program$module, module)), recs)
  }
  recs <- recs[order(vapply(recs, function(c) paste0(c$started, "\u0001", c$id), ""), method = "radix")]
  chr <- function(f) vapply(recs, function(c) { v <- f(c); if (is.null(v)) NA_character_ else as.character(v) }, "")
  num <- function(f) vapply(recs, function(c) { v <- f(c); if (is.null(v)) NA_real_ else as.numeric(v) }, 0)
  tibble::tibble(
    id = chr(function(c) c$id),
    started = as.POSIXct(chr(function(c) c$started), format = "%Y-%m-%dT%H:%M:%OSZ", tz = "UTC"),
    name = chr(function(c) c$program$name), module = chr(function(c) c$program$module),
    version = chr(function(c) c$program$version), model = chr(function(c) c$model),
    seconds = num(function(c) c$seconds),
    inputs = lapply(recs, function(c) c$inputs), outputs = lapply(recs, function(c) c$outputs),
    error = chr(function(c) if (is.null(c$error)) NULL else paste0(c$error$type, if (!is.null(c$error$message)) paste0(": ", c$error$message))),
    input_tokens = num(function(c) c$usage$input_tokens), output_tokens = num(function(c) c$usage$output_tokens),
    reasoning_tokens = num(function(c) c$usage$reasoning_tokens), total_tokens = num(function(c) c$usage$total_tokens),
    parent = chr(function(c) c$parent), caller = lapply(recs, function(c) c$caller),
    language = chr(function(c) c$process$language),
    content = vapply(recs, function(c) isTRUE(c$content), NA),
    omitted = lapply(recs, function(c) c$omitted),
    saw = lapply(recs, function(c) c$saw))
}

#' Say whether calls were right
#'
#' "Right" means correct for this input, not "nice". A correction becomes a
#' row of data: [rated()] gives it back with the inputs. Vectorised over
#' `call`, `verdict` and `answer`, so a column of call ids from [predict()]
#' or [augment()] can be rated at once.
#' @param call Call ids (the `.call` column of `predict()`/`augment()`).
#' @param verdict `"right"`, `"wrong"`, `TRUE`, `FALSE`, or `NA` to withdraw
#'   your earlier rating. Default: `"wrong"` when `answer` is given.
#' @param answer The right answers, for `"wrong"`.
#' @param note,reasons Why, in a sentence, and short tags.
#' @param by Who is judging: a person (default: the caller's `user`). With
#'   no person named, the rating is made under this computer's account,
#'   which may be shared: it is kept on its own, and never replaces nor is
#'   replaced by another rating.
#' @param origin `"review"` (someone judged it) or `"edit"` (someone changed the output while using it).
#' @param sample The id of a random draw of calls these ratings belong to.
#' @param folder The log folder (default: where calls are logged here).
#' @return The ratings as written, invisibly (a tibble).
#' @export
rate <- function(call, verdict = NULL, answer = NULL, note = NULL, reasons = NULL, by = NULL,
                 origin = NULL, sample = NULL, folder = NULL) {
  n <- length(call)
  if (is.null(verdict)) {
    if (is.null(answer)) cli::cli_abort("say whether the answers are right: {.code rate(ids, \"right\")}, or give the right {.arg answer}")
    verdict <- "wrong"
  }
  verdict <- rep_len(verdict, n)
  answers <- if (is.null(answer)) NULL else rep_len(if (is.list(answer) && !is.data.frame(answer)) answer else lapply(seq_len(vctrs::vec_size(answer)), function(i) element(answer, i)), n)
  root <- log_folder(folder)
  if (is.null(root)) cli::cli_abort("no log folder: pass {.arg folder}, or turn the call log on ({.code ai_config(log_calls = TRUE)})")
  s <- effective()
  recs <- lapply(seq_len(n), function(i) {
    v <- verdict[[i]]
    v <- if (is.na(v)) NULL else if (isTRUE(v) || identical(v, "right")) "right" else if (isFALSE(v) || identical(v, "wrong")) "wrong"
      else cli::cli_abort("a verdict is \"right\", \"wrong\", TRUE, FALSE or NA, not {.val {v}}")
    a <- if (is.null(answers)) NULL else plain_json(answers[[i]])
    if (!is.null(a) && !identical(v, "wrong")) cli::cli_abort("a right answer needs no correction: give {.arg answer} only with \"wrong\"")
    rec <- rating_record(call[[i]], v, a, NULL, note, reasons, by, origin, sample, s)
    append_record(root, rec)
    rec
  })
  invisible(tibble::tibble(id = vapply(recs, function(r) r$id, ""), call = as.character(call),
                           verdict = vapply(recs, function(r) r$verdict %||% NA_character_, ""),
                           by = vapply(recs, function(r) r$by %||% NA_character_, ""),
                           account = vapply(recs, function(r) r$account %||% NA_character_, "")))
}

#' Rows with known answers, from people's ratings
#'
#' The inputs of every rated call of `fn` and the right answer (the model's,
#' when it was rated right; the correction, when wrong), typed like the
#' function's inputs and outputs: ready for [evaluate()] and the optimizers.
#' Calls whose data has another shape (the inputs or outputs changed since)
#' are left out and counted (`other_signature`); calls that record the same
#' data pool, even when the instruction, a default or reasoning changed. So
#' are calls whose inputs the log did not keep (`no_content`, `log_content`),
#' and rated calls that give no answer (`no_answer`: a right verdict on an
#' answer that was not kept). Reads every language's log, formats 1 and 2.
#' @param fn An AI function.
#' @param by Only this person's ratings.
#' @param folder The log folder.
#' @param since Only calls and ratings from this time on.
#' @param any_file A function defined in a notebook or a script is known by
#'   its file too, so two notebooks' `summarize` are two programs. `TRUE`
#'   takes its calls from any file (a notebook that moved).
#' @return A tibble: the inputs, the outputs, then `call`, `version`,
#'   `rating`, `rated_by`, `origin`, `sample`, `disputed`. Attribute
#'   `left_out`: the counts.
#' @export
rated <- function(fn, by = NULL, folder = NULL, since = NULL, any_file = FALSE) {
  core <- core_of(fn)
  log <- read_log(log_folder(folder), since)
  # a function defined at the top level is known by its file too: two notebooks' summarize are two programs
  file <- if (isTRUE(any_file) || !identical(core$module, "__main__")) NULL else core$file
  out <- rated_rows(log$calls, log$ratings, name = core$definition$name, module = core$module,
                    signature = signature_id(signature_of(core, effective(core$own))),
                    interface = interface_signature(interface_of(core)), by = by, file = file)
  rows <- out$rows
  fields <- c(core$definition$inputs, core$definition$outputs)
  cols <- list()
  keys <- unique(unlist(lapply(rows, names)))
  for (k in keys) {
    values <- lapply(rows, function(r) r[[k]])
    cols[[k]] <- if (k %in% names(fields)) assemble(fields[[k]], values)
      else if (k == "disputed") vapply(values, isTRUE, NA)
      else vapply(values, function(v) if (is.null(v)) NA_character_ else as.character(v), "")
  }
  if (core$single && "result" %in% names(cols)) names(cols)[names(cols) == "result"] <- columns_of(core)[["result"]]
  structure(tibble::new_tibble(cols, nrow = length(rows)), left_out = out$left_out)
}
