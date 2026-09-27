# Saved functions (contract/saved.md): run an AI function saved in any
# language, from its folder's functai.json, after checking it sends exactly
# what was saved; refuse, with the reason, what R cannot run.

load_refused <- function(code, message) {
  stop(structure(class = c("functai_load_refused", "error", "condition"),
                 list(message = sprintf("[%s] %s", code, message), call = NULL, code = code)))
}

SNAKE <- c(retries = "retries", api_retries = "api_retries", max_steps = "max_steps", tool_errors = "tool_errors",
           capabilities = "capabilities", log_content = "log_content")

# A field from a saved signature field: its kind from its shape.
field_from_shape <- function(shape, desc = NULL) {
  nullable <- FALSE
  inner <- shape
  if (!is.null(shape$anyOf)) {
    opts <- Filter(function(s) !identical(s$type, "null"), shape$anyOf)
    if (length(opts) == 1L && length(shape$anyOf) == 2L) { inner <- opts[[1L]]; nullable <- TRUE }
  }
  f <- if (!is.null(inner$enum) && all(vapply(inner$enum, is.character, NA))) new_field(inner, "enum", levels = unlist(inner$enum))
    else if (identical(inner$type, "object") && is.list(inner$properties) && length(inner$properties) && !is.null(inner$required)) {
      fields <- lapply(inner$properties, field_from_shape)
      new_field(inner, "record", fields = fields)
    } else if (identical(inner$type, "array") && is.list(inner$items)) new_field(inner, "list", item = field_from_shape(inner$items))
    else if (is.character(inner$type) && length(inner$type) == 1L && inner$type %in% c("string", "integer", "number", "boolean")) new_field(inner, inner$type)
    else new_field(inner, "json")
  f$shape <- shape
  f$nullable <- nullable
  f$desc <- desc
  f
}

#' Run a function saved in any language
#'
#' Reads a saved folder's `functai.json` (written by `functai.save` in
#' Python, `save()` in TypeScript, or [write_ai()]) and returns the AI
#' function in it. It checks the function sends exactly what it sent where it
#' was saved, and has its version, before any call. It refuses, with the
#' reason, what only the saving language can run: code of its own around the
#' model (`saved-code`), a module (`saved-not-ai`), tools (`saved-tools`), a
#' baked model (`saved-model`).
#' @param path The folder, or its `functai.json`.
#' @param node Which function, by key (`"module:name"`); default the entry.
#' @return An AI function.
#' @export
read_ai <- function(path, node = NULL) {
  file <- if (dir.exists(path)) file.path(path, "functai.json") else path
  text <- paste(readLines(file, warn = FALSE, encoding = "UTF-8"), collapse = "\n")
  manifest <- tryCatch(lmcc::parse_json(text), error = function(e) load_refused("saved-malformed", sprintf("%s: %s", file, conditionMessage(e))))
  bytes <- readBin(file, "raw", file.info(file)$size)
  from_manifest(manifest, node, saved = paste0("sha256:", lmcc::sha256_hex(rawToChar(bytes))))
}

from_manifest <- function(m, node = NULL, saved = NULL) {
  if (!is.list(m) || is.null(names(m))) load_refused("saved-malformed", "functai.json is not a JSON object")
  if (!identical(as.integer(m$functai_saved %||% -1L), 1L)) load_refused("saved-format", sprintf("functai.json is format %s; this loader reads format 1", lmcc::json_text(m$functai_saved)))
  if (!is.character(m$entry) || !is.list(m$nodes)) load_refused("saved-malformed", "functai.json needs entry and nodes")
  language <- m$language %||% "python"
  key <- node %||% m$entry
  n <- m$nodes[[key]]
  if (is.null(n)) load_refused("saved-malformed", sprintf("functai.json has no node %s", key))
  if (!identical(n$kind, "ai")) load_refused("saved-not-ai", sprintf("%s is a %s: code in %s, which this loader cannot run. Its AI functions load by key.", key, n$kind, language))
  d <- n$ai
  if (!"body" %in% names(d) || !is.null(d$body)) load_refused("saved-code", sprintf("%s runs code of its own beside the model (written in %s); only %s can run it", key, language, language))
  if (length(d$tools)) load_refused("saved-tools", sprintf("%s has tools: a tool is code", key))
  settings_in <- d$settings %||% list()
  for (k in names(settings_in)) if (is.list(settings_in[[k]]) && any(c("baked", "node") %in% names(settings_in[[k]])))
    load_refused("saved-model", sprintf("%s: setting %s is %s, not something this loader can reach", key, k, lmcc::json_text(settings_in[[k]])))
  sig <- d$signature
  inputs <- list(); outputs <- list(); cot <- FALSE
  for (f in sig$fields) {
    purpose <- f$purpose %||% "plain"
    if (f$direction == "input" && purpose == "plain") inputs[[f$name]] <- field_from_shape(f$shape, f$desc)
    else if (f$direction == "output" && purpose == "plain") outputs[[f$name]] <- field_from_shape(f$shape)
    else if (purpose == "reasoning") cot <- TRUE
    else load_refused("saved-tools", sprintf("%s: field %s (%s) needs tools", key, f$name, purpose))
  }
  own <- list()
  if (is.character(settings_in$lm)) own$lm <- settings_in$lm
  if (identical(settings_in$module, "cot") || cot) own$module <- "cot"
  if (isFALSE(settings_in$include_fn_name_in_instructions)) own$include_fn_name <- FALSE
  if (!is.null(settings_in$adapter)) own$adapter <- settings_in$adapter
  for (k in names(SNAKE)) if (!is.null(settings_in[[k]])) own[[SNAKE[[k]]]] <- settings_in[[k]]
  if (is.list(d$template)) own$template <- d$template
  cfg <- d$config %||% list()
  for (k in c("temperature", "max_tokens", "top_p", "seed")) if (!is.null(cfg[[k]])) own[[k]] <- lmcc::lm15_plain(cfg[[k]])
  if (!is.null(cfg$stop)) own$stop <- unlist(cfg$stop)
  rest <- cfg[setdiff(names(cfg), c("temperature", "max_tokens", "top_p", "seed", "stop"))]
  if (length(rest)) own$config <- rest
  state <- d$state %||% list()
  core <- list(definition = list(name = n$name, description = "", inputs = inputs, outputs = outputs, written = sig$instructions),
               own = own, tools = list(), single = length(outputs) == 1L && identical(names(outputs), "result"),
               columns = if (identical(names(outputs), "result")) c(result = n$name) else stats::setNames(names(outputs), names(outputs)),
               module = n$module, state = list(instructions = state$instructions, demos = list()), saved = saved)
  core$state$demos <- as_demos(core, state$demos)
  probes <- d$probes %||% list()
  want <- unlist(d$fingerprints$requests)
  for (i in seq_along(probes)) {
    got <- request_hash(core, probes[[i]])
    if (i <= length(want) && !identical(got, want[[i]]))
      load_refused("saved-differs", sprintf("%s: for probe %d it would send %s, but %s was saved", key, i - 1L, got, want[[i]]))
  }
  if (is.character(d$version) && !identical(d$version, version_of(core)))
    load_refused("saved-differs", sprintf("%s: its version here is %s, but %s was saved", key, version_of(core), d$version))
  make_fn(core)
}

#' Save an AI function for any language
#'
#' Writes `functai.json` in `path`: the function as data, which Python
#' (`functai.load`), TypeScript (`load()`) and [read_ai()] run, sending the
#' same requests.
#' @param fn An AI function.
#' @param path A folder.
#' @return The file written, invisibly.
#' @export
write_ai <- function(fn, path) {
  dir.create(path, recursive = TRUE, showWarnings = FALSE)
  file <- file.path(path, "functai.json")
  writeLines(json_indented(to_manifest(fn)), file, useBytes = TRUE)
  invisible(file)
}

to_manifest <- function(fn) {
  core <- core_of(fn)
  if (length(core$tools)) cli::cli_abort("{.fn {core$definition$name}} has tools: a tool is code, and a saved folder carries none from R")
  s <- core$own
  key <- paste0(core$module, ":", core$definition$name)
  settings <- list(module = s$module %||% "predict", include_fn_name_in_instructions = !isFALSE(s$include_fn_name))
  if (is.character(s$lm)) settings$lm <- s$lm
  if (!is.null(s$adapter)) settings$adapter <- if (inherits(s$adapter, "lmcc_adapter")) lmcc::dump_adapter(s$adapter, registry()) else s$adapter
  for (k in names(SNAKE)) if (!is.null(s[[SNAKE[[k]]]])) settings[[k]] <- s[[SNAKE[[k]]]]
  cfg <- s$config %||% list()
  for (k in c("temperature", "max_tokens", "top_p", "seed")) if (!is.null(s[[k]])) cfg[[k]] <- s[[k]]
  if (!is.null(s$stop)) cfg$stop <- as.list(s$stop)
  sig <- signature_of(core, effective(core$own))
  probes <- list(sample_inputs(sig))
  for (d in utils::head(core$state$demos, 3L)) {
    ins <- if (!is.null(d$signature)) d$inputs else d$inputs
    if (!any(vapply(probes, function(p) identical(lmcc::canonical_json(p), lmcc::canonical_json(ins)), NA))) probes[[length(probes) + 1L]] <- ins
  }
  ai <- list(settings = settings, config = if (length(cfg)) cfg else lmcc::jobj(), template = s$template, tools = list(),
             teacher = NULL, state = list(instructions = core$state$instructions, demos = core$state$demos), requires = list(),
             signature = lmcc::signature_to_list(sig), probes = probes,
             fingerprints = list(signature = lmcc::signature_fingerprint(sig), requests = lapply(probes, function(p) request_hash(core, p))),
             body = NULL, version = version_of(core))
  node <- list(kind = "ai", module = core$module, name = core$definition$name, ai = ai)
  nodes <- list(); nodes[[key]] <- node
  list(functai_saved = 1L, language = "r", entry = key, created = format(Sys.time(), "%Y-%m-%dT%H:%M:%S+00:00", tz = "UTC"),
       functai = as.character(utils::packageVersion("functai")), nodes = nodes)
}
