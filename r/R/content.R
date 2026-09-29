# What the call log keeps (contract/calls.md, "Content"): `log_content`, per
# field, over every layer: the function's own setting, each enclosing
# with_ai_config() block, ai_config(), and FUNCTAI_LOG_CONTENT. A value is
# written only when no layer drops it: a layer can ask for less, never for
# more.

LOG_CONTENT_OFF <- c("0", "false", "no", "off")

# A log_content setting in its one form: TRUE, FALSE, or a named list of
# TRUE/FALSE by field name ("*" for the fields it does not name). R spells a
# map as a named list or a named logical vector (`c(transcript = FALSE)`);
# a character vector of names is the list of what may be kept
# (`c("question")` is `list("*" = FALSE, question = TRUE)`). A key that is
# not a field's name (an ASCII identifier) nor "*" refuses wherever it is set
# (kept for kinds of data).
normalize_log_content <- function(x, call = rlang::caller_env()) {
  if (is.null(x)) return(NULL)
  if (is.null(names(x)) && (isTRUE(x) || isFALSE(x))) return(x)
  if (is.character(x) && is.null(names(x))) {
    if (anyNA(x)) cli::cli_abort("{.arg log_content}'s names of the fields to keep cannot be NA", call = call)
    x <- c(list(`*` = FALSE), stats::setNames(as.list(rep(TRUE, length(x))), x))
  }
  if (is.logical(x) && !is.null(names(x))) x <- as.list(x)
  if (!is.list(x) || (length(x) && is.null(names(x))) || !all(vapply(x, function(v) isTRUE(v) || isFALSE(v), NA)))
    cli::cli_abort(c("{.arg log_content} is {.code TRUE}, {.code FALSE}, the names of the fields to keep, or TRUE/FALSE by field",
      i = "{.code log_content = c(transcript = FALSE)}; {.code log_content = c(\"question\")} keeps only the question"), call = call)
  keys <- names(x)
  if (anyDuplicated(keys)) cli::cli_abort("{.arg log_content} names {.field {keys[duplicated(keys)][[1L]]}} twice", call = call)
  for (k in keys) if (!identical(k, "*") && !is_name(k))
    refuse("log-content-field", c("{.arg log_content} names {.val {k}}, which is not a field's name",
      i = "a key is a field's name (letters, digits and _), or {.val *} for every field it does not name"), field = k, call = call)
  if (!length(x)) return(TRUE)
  x
}

# Whether one layer drops a field: FALSE, a map that says FALSE for it, or a
# map whose "*" is FALSE and that does not name it.
says_drop <- function(layer, name) {
  if (is.null(layer)) return(FALSE)
  if (!is.list(layer)) return(isFALSE(layer))
  if (name %in% names(layer)) return(isFALSE(layer[[name]]))
  "*" %in% names(layer) && isFALSE(layer[["*"]])
}

environment_drops_all <- function() {
  raw <- tolower(trimws(Sys.getenv("FUNCTAI_LOG_CONTENT"), whitespace = "[ \t\r\n\v\f]"))
  raw %in% LOG_CONTENT_OFF
}

# A call's fields (calls.md, "Words"): its interface's inputs; its outputs,
# with those FunctAI adds (`reasoning`, `calls`), in the signature's order.
call_fields <- function(core, settings) {
  sig <- fields_of(core$definition, identical(settings$module, "cot"), length(core$tools) > 0L)
  outs <- Filter(function(f) f$direction == "output", sig)
  list(inputs = names(core$definition$inputs),
       outputs = vapply(outs, function(f) f$name, ""),
       added = vapply(Filter(function(f) !identical(f$purpose, "plain"), outs), function(f) f$name, ""))
}

# R calls a one-output function's answer by its formula's name (`reply ~
# message`); its field is `result`, as in every language. A map may say
# either: this one says the field's name (a drop wins over a keep). A name
# that is a field of the call (an input, an output, one FunctAI adds) is that
# field, never an alias: a loaded function's answer column is named after the
# function, and a function may be named like one of its inputs (`secret`),
# whose rule must not move to the answer.
field_names_in <- function(layer, core) {
  if (!is.list(layer)) return(layer)
  cols <- columns_of(core)
  fields <- call_fields(core, effective(core$own))
  canonical <- c(fields$inputs, fields$outputs)
  for (field in names(cols)) {
    col <- cols[[field]]
    if (identical(col, field) || col %in% canonical || !col %in% names(layer)) next
    v <- layer[[col]]
    layer[[col]] <- NULL
    layer[[field]] <- if (field %in% names(layer)) isTRUE(layer[[field]]) && isTRUE(v) else v
  }
  layer
}

# The layers in force for a call of `core`, closest first: the function's own
# setting, the settings given to this call (predict(..., log_content = )),
# each enclosing block from the closest out, then ai_config().
content_layers <- function(core, extra = NULL) {
  blocks <- rev(the$content_blocks %||% list())
  lapply(c(list(core$own$log_content), list(extra), blocks, list(the$config$log_content)), field_names_in, core = core)
}

# For each field, whether its value is written: only when no layer (nor the
# environment) drops it; an added field only when no field of the call is
# dropped (it can quote any input, anticipate any output, quote another).
content_kept <- function(fields, layers, environment_off = environment_drops_all()) {
  names_ <- c(fields$inputs, fields$outputs)
  keep <- vapply(names_, function(n) !environment_off && !any(vapply(layers, says_drop, NA, name = n)), NA)
  names(keep) <- names_
  if (!all(keep)) keep[fields$added] <- FALSE
  keep
}

# A function's own map may name only its fields (`tools` is not one): a
# misspelt name would write the very value it was meant to keep out.
check_own_content <- function(core, call = NULL) {
  own <- core$own$log_content
  if (!is.list(own)) return(invisible())
  fields <- call_fields(core, effective(core$own))
  known <- c(fields$inputs, fields$outputs)
  cols <- columns_of(core)
  spelled <- c(fields$inputs, ifelse(fields$outputs %in% names(cols), cols[fields$outputs], fields$outputs))
  for (k in setdiff(names(own), "*")) if (!k %in% known)
    refuse("log-content-field", c("{.fn {core$definition$name}}'s {.arg log_content} names {.field {k}}, which is not one of its fields",
      i = "its fields are {.field {spelled}}",
      i = if (identical(k, "tools")) "its tools are the function itself, not a value of a call: nothing records them"),
      field = k, call = call)
  invisible()
}

# CALL RECORD KEYS kept whatever is dropped; everything else of a record not
# whole is rebuilt from what may be kept (an error keeps its type and code,
# never a member this contract does not name).
ALWAYS_KEPT <- c("functai_call", "id", "parent", "root", "program", "started", "seconds", "sizes", "model", "usage",
                 "confidence", "caller", "process", "saw", "escalated", "truncated", "journal")
ERROR_KEPT <- c("type", "code")

# The record as written: the whole record when every field is kept, else
# format 2's record of some values (calls.md, "What the record keeps").
kept_record <- function(rec, fields, keep) {
  if (all(keep)) return(rec)
  out <- list()
  for (k in names(rec)) {
    if (k %in% ALWAYS_KEPT) out[k] <- list(rec[[k]])
    if (k == "content") {
      out$content <- FALSE
      out$omitted <- list(inputs = as.list(fields$inputs[!keep[fields$inputs]]), outputs = as.list(fields$outputs[!keep[fields$outputs]]))
    }
  }
  kept <- function(values) values[names(values) %in% names(keep)[keep]]
  inputs <- kept(rec$inputs %||% list())
  if (length(inputs)) out$inputs <- inputs
  if (is.null(rec$outputs)) out["outputs"] <- list(NULL)
  else if (length(outputs <- kept(rec$outputs))) out$outputs <- outputs
  if (!is.null(rec$described)) {
    described <- lapply(rec$described, function(v) as.list(unlist(v)[unlist(v) %in% names(keep)[keep]]))
    if (any(lengths(described) > 0L)) out$described <- described
  }
  if (has_key(rec, "returned") && isTRUE(keep[[rec$program$answer]])) out["returned"] <- list(rec$returned)
  if (!is.null(rec$probabilities) && length(p <- kept(rec$probabilities))) out$probabilities <- p
  out["error"] <- list(if (is.null(rec$error)) NULL else rec$error[names(rec$error) %in% ERROR_KEPT])
  out$exchanges <- lapply(rec$exchanges, function(ex) {
    ex <- ex[!names(ex) %in% c("request", "response", "request_hash")]
    if (!is.null(ex$error)) ex$error <- ex$error[names(ex$error) %in% ERROR_KEPT]
    ex
  })
  order <- append(names(rec), "omitted", after = match("content", names(rec)))
  out[order(match(names(out), order))]
}
