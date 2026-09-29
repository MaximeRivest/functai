# Saved functions (contract/saved.md): run an AI function saved in any
# language, from its folder's functai.json, after checking it sends exactly
# what was saved; refuse, with the reason, what R cannot run.

# A loader's or a describer's refusal: the contract's code and, where a field
# is at fault, its name (`field`), as every refusal carries them; the message
# is taken as written (it quotes JSON, whose braces are not cli's).
load_refused <- function(code, message, field = NULL) {
  rlang::abort(sprintf("[%s] %s", code, message),
               class = c("functai_load_refused", paste0("functai_", gsub("-", "_", code)), "functai_refusal"),
               code = code, field = field, call = NULL)
}

SNAKE <- c(retries = "retries", api_retries = "api_retries", max_steps = "max_steps", tool_errors = "tool_errors",
           capabilities = "capabilities", log_content = "log_content")

# A field of a saved function, from its shape: the R type its values are
# read as. What is sent never depends on it (to_json() reads the shape
# alone); what a caller gets back does. `type`: the R type R saved the field
# as (r_type_of(), read back by read_r_type()), used when it fits the shape,
# so a function R saved is read back with the types it had. Otherwise the
# type is one that holds every value the shape admits exactly: a column of
# text, numbers, yes/no, a factor for a choice of texts, a list_of() for a
# list, a tibble for a record only when it is closed (a tibble has no place
# for a member its columns do not name) and none of its members may be
# both left out and null (a tibble has one NA for both); JSON (a list
# column) for anything else. `root` holds the shape's `$defs`.
field_from_shape <- function(shape, desc = NULL, type = NULL, root = shape) {
  nullable <- FALSE
  inner <- shape
  if (is_arr(shape$anyOf)) {
    opts <- Filter(function(s) !(is_obj(s) && identical(s$type, "null")), shape$anyOf)
    if (length(opts) == 1L && length(shape$anyOf) == 2L && length(setdiff(names(shape), c("anyOf", "default", "description", "title"))) == 0L) {
      inner <- opts[[1L]]; nullable <- TRUE
    }
  }
  f <- (if (!is.null(type)) declared_field(inner, type, root)) %||% exact_field(inner, nullable, root)
  f$shape <- shape
  f$nullable <- nullable
  f$desc <- desc
  f
}

is_text_choice <- function(s) is_arr(s$enum) && length(s$enum) > 0L && all(vapply(s$enum, is_str, NA)) &&
  (is.null(s$type) || identical(s$type, "string"))

exact_field <- function(inner, nullable, root) {
  t <- inner$type
  if (is_text_choice(inner)) new_field(inner, "enum", levels = unlist(inner$enum))
  else if (!nullable && is_closed_record(inner, root))
    new_field(inner, "record", fields = lapply(inner$properties, field_from_shape, root = root))
  else if (identical(t, "array") && is_obj(inner$items) && is.null(inner$prefixItems))
    new_field(inner, "list", item = field_from_shape(inner$items, root = root))
  else if (is_str(t) && t %in% c("string", "integer", "number", "boolean") && is.null(inner$enum) && is.null(inner[["const"]]))
    new_field(inner, t)
  else new_field(inner, "json")
}

# A record a tibble holds exactly: an object that allows no other member,
# names every required member, and whose members are each required (NA is
# null) or take no null (NA is left out). A null record would be a row of
# NAs, which a record of nulls also is: a nullable one stays JSON.
is_closed_record <- function(shape, root) {
  props <- shape$properties
  required <- unlist(shape$required)
  identical(shape$type, "object") && is_obj(props) && length(props) > 0L && isFALSE(shape$additionalProperties) &&
    all(required %in% names(props)) &&
    all(vapply(names(props), function(n) n %in% required || !fits_shape(NULL, props[[n]], root), NA))
}

# A field as the R type R saved it as, when that type fits the shape; NULL
# otherwise (the shape decides then).
declared_field <- function(inner, type, root) {
  if (is.symbol(type)) {
    t <- inner$type
    return(switch(as.character(type),
      character = if (identical(t, "string")) new_field(inner, "string"),
      integer = if (identical(t, "integer")) new_field(inner, "integer"),
      double = if (identical(t, "number")) new_field(inner, "number"),
      logical = if (identical(t, "boolean")) new_field(inner, "boolean"),
      factor = if (is_text_choice(inner)) new_field(inner, "enum", levels = unlist(inner$enum)),
      list = new_field(inner, "json"),
      NULL))
  }
  if (!is.call(type) || !is.symbol(type[[1L]])) return(NULL)
  args <- as.list(type)[-1L]
  switch(as.character(type[[1L]]),
    tibble = {
      props <- inner$properties
      if (!identical(inner$type, "object") || !is_obj(props) || !identical(names(args), names(props))) return(NULL)
      fields <- Map(function(s, a) field_from_shape(s, type = a, root = root), props, args)
      if (!all(vapply(Map(function(s, a) declared_field_of(s, a, root), props, args), isTRUE, NA))) return(NULL)
      new_field(inner, "record", fields = fields)
    },
    list_of = {
      if (length(args) != 1L || !identical(inner$type, "array") || !is_obj(inner$items)) return(NULL)
      if (!isTRUE(declared_field_of(inner$items, args[[1L]], root))) return(NULL)
      new_field(inner, "list", item = field_from_shape(inner$items, type = args[[1L]], root = root))
    },
    NULL)
}

# Whether a saved R type fits a shape all the way down.
declared_field_of <- function(shape, type, root) {
  inner <- shape
  if (is_arr(shape$anyOf)) {
    opts <- Filter(function(s) !(is_obj(s) && identical(s$type, "null")), shape$anyOf)
    if (length(opts) == 1L && length(shape$anyOf) == 2L) inner <- opts[[1L]]
  }
  !is.null(declared_field(inner, type, root))
}

# The R type of a field, as R saves it in the interface's `type` (for
# people, and for R reading its own folder back): `character`, `integer`,
# `double`, `logical`, `factor`, `list` (JSON), `list_of(<type>)`,
# `tibble(<name> = <type>, ...)`.
r_type_of <- function(f) {
  switch(f$kind,
    string = "character", integer = "integer", number = "double", boolean = "logical", enum = "factor", json = "list",
    list = sprintf("list_of(%s)", r_type_of(f$item)),
    record = sprintf("tibble(%s)", paste(sprintf("%s = %s", vapply(names(f$fields), function(n) if (make.names(n) == n) n else sprintf("`%s`", gsub("`", "\\\\`", n)), ""),
                                                 vapply(f$fields, r_type_of, "")), collapse = ", ")),
    "list")
}

# An interface whose fields' `type` is R's for them: what an R folder says
# (a function loaded from another language's folder keeps its fields' R
# types, not the names that language gave them).
r_typed <- function(iface, core) {
  d <- core$definition
  for (i in seq_along(iface$inputs)) iface$inputs[[i]]$type <- r_type_of(d$inputs[[i]])
  for (i in seq_along(iface$outputs)) iface$outputs[[i]]$type <- r_type_of(d$outputs[[i]])
  iface
}

# A saved `type` as R code to read (never run): NULL when it is not one.
read_r_type <- function(type) {
  if (!is_str(type)) return(NULL)
  tryCatch(str2lang(type), error = function(e) NULL)
}

#' Run a function saved in any language
#'
#' Reads a saved folder's `functai.json` (written by `functai.save` in
#' Python, `save()` in TypeScript, or [write_ai()]) and returns the AI
#' function in it. It checks the function sends exactly what it sent where it
#' was saved, and has its version, before any call. It refuses, with the
#' reason, what only the saving language can run: code of its own around the
#' model (`saved-code`), a module (`saved-not-ai`), tools (`saved-tools`), a
#' baked model (`saved-model`). Its inputs a caller may leave out, and what
#' each is sent with then, come from the folder's interface of it
#' ([ai_interface()] describes any program in a folder without loading it).
#'
#' **Types.** A function R saved comes back with the R types it had (a
#' [record()] is a tibble column, a [json_shape()] a list column): R writes
#' them in the interface's `type`. Another language's function comes back
#' with, for each field, the R type that holds every value its shape admits
#' exactly: text, numbers, yes or no, a factor for a choice, a `list_of`
#' for a list, a tibble for a record only when the record is closed
#' (`additionalProperties: false`, so no member can come back that a column
#' does not hold); any other object, a Python dataclass's among them, is a
#' list column of named lists (`tidyr::unnest_wider()` spreads it). What a
#' call sends never depends on these types: a value is sent as it is given.
#' @param path The folder, or its `functai.json`.
#' @param node Which function, by key (`"module:name"`); default the entry.
#' @return An AI function.
#' @export
read_ai <- function(path, node = NULL) {
  m <- read_manifest(path)
  from_manifest(m$manifest, node, saved = m$saved)
}

# A saved folder's manifest, and the hash that names it (`program.saved`).
read_manifest <- function(path) {
  if (!is_str(path)) cli::cli_abort("{.arg path} is a saved folder, or its {.file functai.json}")
  file <- if (dir.exists(path)) file.path(path, "functai.json") else path
  if (!file.exists(file)) cli::cli_abort("no saved program at {.path {path}} (no {.file functai.json})")
  text <- paste(readLines(file, warn = FALSE, encoding = "UTF-8"), collapse = "\n")
  manifest <- tryCatch(lmcc::parse_json(text), error = function(e) load_refused("saved-malformed", sprintf("%s: %s", file, conditionMessage(e))))
  bytes <- readBin(file, "raw", file.info(file)$size)
  list(manifest = manifest, saved = paste0("sha256:", lmcc::sha256_hex(rawToChar(bytes))))
}

# saved.md, step 1: a format this loader knows, then the manifest's schema
# (with every node's interface): what either refuses is not read further.
check_manifest <- function(m) {
  if (!is_obj(m)) load_refused("saved-malformed", "functai.json is not a JSON object")
  if (!is_format(m$functai_saved %||% -1L, 1L)) load_refused("saved-format", sprintf("functai.json is format %s; this loader reads format 1", lmcc::json_text(m$functai_saved)))
  fault <- schema_fault(m, "saved.schema.json")
  if (!is.null(fault)) load_refused("saved-malformed", sprintf("functai.json does not pass the contract's schema: %s", fault))
  invisible(m)
}

manifest_node <- function(m, node) {
  key <- node %||% m$entry
  n <- m$nodes[[key]]
  if (is.null(n)) load_refused("saved-malformed", sprintf("functai.json has no node %s", key))
  list(key = key, node = n)
}

# The plain fields of an AI node's signature: what its interface describes.
plain_fields <- function(n) Filter(function(f) identical(f$purpose %||% "plain", "plain"), n$ai$signature$fields)

# An AI node's interface must describe the data its signature takes and gives
# (saved.md, step 6); a node written before 2026-09-28 has none, and is
# described by its signature (its plain fields, its instruction).
node_interface <- function(key, n) {
  if (has_key(n, "interface")) {
    iface <- n$interface
    refuse_interface(key, iface, identical(n$kind, "ai"))
    if (identical(n$kind, "ai")) {
      fields <- plain_fields(n)
      from_signature <- interface_signature(list(inputs = Filter(function(f) f$direction == "input", fields),
                                                 outputs = Filter(function(f) f$direction == "output", fields)))
      if (!identical(interface_signature(iface), from_signature))
        load_refused("saved-differs", sprintf("%s: its interface promises other data than its signature takes and gives", key))
    }
    return(iface)
  }
  if (!identical(n$kind, "ai")) load_refused("saved-no-interface", sprintf("%s is a %s saved without its interface: what it takes and gives is not known", key, n$kind))
  field <- function(f) {
    out <- list(name = f$name, shape = f$shape)
    if (is_str(f$desc) && nzchar(f$desc)) out$desc <- f$desc
    if (is_str(f$type)) out$type <- f$type
    out
  }
  fields <- plain_fields(n)
  iface <- list(description = n$ai$signature$instructions,
                inputs = lapply(Filter(function(f) f$direction == "input", fields), field),
                outputs = lapply(Filter(function(f) f$direction == "output", fields), field))
  refuse_interface(key, iface, TRUE)        # read from a saved folder, so checked by the same rules (programs.md)
  iface
}

# programs.md's rules for an interface read from a saved folder: refused
# `interface-malformed`, naming the first field at fault.
refuse_interface <- function(key, iface, ai) {
  problem <- interface_problem(iface, ai = ai)
  if (is.null(problem)) return(invisible())
  f <- problem$field
  load_refused("interface-malformed", sprintf("%s: its interface is refused: %s", key, interface_fault(iface, f)), field = f)
}

# Describing a node without loading it (saved.md, "Describing without
# loading").
describe_manifest <- function(m, node = NULL) {
  check_manifest(m)
  x <- manifest_node(m, node)
  if (!x$node$kind %in% c("ai", "module")) load_refused("saved-not-ai", sprintf("%s is a %s: plain code, not a program", x$key, x$node$kind))
  node_interface(x$key, x$node)
}

from_manifest <- function(m, node = NULL, saved = NULL) {
  check_manifest(m)
  language <- m$language %||% "python"
  x <- manifest_node(m, node)
  key <- x$key; n <- x$node
  if (!identical(n$kind, "ai")) load_refused("saved-not-ai", sprintf("%s is a %s: code in %s, which this loader cannot run. Its AI functions load by key.", key, n$kind, language))
  d <- n$ai
  if (!"body" %in% names(d) || !is.null(d$body)) load_refused("saved-code", sprintf("%s runs code of its own beside the model (written in %s); only %s can run it", key, language, language))
  if (length(d$tools)) load_refused("saved-tools", sprintf("%s has tools: a tool is code", key))
  settings_in <- d$settings %||% list()
  for (k in names(settings_in)) if (is.list(settings_in[[k]]) && any(c("baked", "node") %in% names(settings_in[[k]])))
    load_refused("saved-model", sprintf("%s: setting %s is %s, not something this loader can reach", key, k, lmcc::json_text(settings_in[[k]])))
  iface <- node_interface(key, n)
  # the R types R saved its fields as (the interface's `type`); another
  # language's names for them are its own
  r_types <- function(fields) if (identical(language, "r")) stats::setNames(lapply(fields, function(f) read_r_type(f$type)), vapply(fields, function(f) f$name, "")) else list()
  in_types <- r_types(iface$inputs); out_types <- r_types(iface$outputs)
  optional <- Filter(function(f) isTRUE(f$optional), iface$inputs)
  optional <- stats::setNames(optional, vapply(optional, function(f) f$name, ""))
  sig <- d$signature
  inputs <- list(); outputs <- list(); cot <- FALSE
  for (f in sig$fields) {
    purpose <- f$purpose %||% "plain"
    if (f$direction == "input" && purpose == "plain") {
      field <- field_from_shape(f$shape, f$desc, type = in_types[[f$name]])
      if (!is.null(optional[[f$name]])) {                        # its default, from the node's interface
        field$shape["default"] <- list(optional[[f$name]]$shape[["default"]])
        field$optional <- TRUE
      }
      inputs[[f$name]] <- field
    }
    else if (f$direction == "output" && purpose == "plain") outputs[[f$name]] <- field_from_shape(f$shape, type = out_types[[f$name]])
    else if (purpose == "reasoning") cot <- TRUE
    else load_refused("saved-tools", sprintf("%s: field %s (%s) needs tools", key, f$name, purpose))
  }
  own <- list()
  if (is.character(settings_in$lm)) own$lm <- settings_in$lm
  if (identical(settings_in$module, "cot") || cot) own$module <- "cot"
  if (isFALSE(settings_in$include_fn_name_in_instructions)) own$include_fn_name <- FALSE
  if (!is.null(settings_in$adapter)) own$adapter <- settings_in$adapter
  for (k in names(SNAKE)) if (!is.null(settings_in[[k]])) own[[SNAKE[[k]]]] <- settings_in[[k]]
  if (!is.null(own$log_content)) own$log_content <- normalize_log_content(own$log_content)
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
               module = n$module, state = list(instructions = state$instructions, demos = list()), saved = saved,
               interface = iface)
  core$state$demos <- as_demos(core, state$demos)
  # each probe against its own fingerprint: a probe with none cannot be
  # checked, and a fingerprint with no probe checks nothing, so either refuses
  probes <- d$probes %||% list()
  want <- unlist(d$fingerprints$requests) %||% character(0)
  if (length(want) < length(probes))
    load_refused("saved-differs", sprintf("%s: probe %d has no request fingerprint: what it sent where it was saved is not known", key, length(want)))
  if (length(want) > length(probes))
    load_refused("saved-differs", sprintf("%s: request fingerprint %d has no probe: the fingerprints are not the probes'", key, length(probes)))
  for (i in seq_along(probes)) {
    got <- request_hash(core, probes[[i]])
    if (!identical(got, want[[i]]))
      load_refused("saved-differs", sprintf("%s: for probe %d it would send %s, but %s was saved", key, i - 1L, got, want[[i]]))
  }
  if (is.character(d$version) && !identical(d$version, version_of(core)))
    load_refused("saved-differs", sprintf("%s: its version here is %s, but %s was saved", key, version_of(core), d$version))
  tryCatch(check_own_content(core), functai_refusal = function(e) load_refused(e$code, conditionMessage(e), field = e$field))
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
  node <- list(kind = "ai", module = core$module, name = core$definition$name, interface = r_typed(interface_of(core), core), ai = ai)
  nodes <- list(); nodes[[key]] <- node
  list(functai_saved = 1L, language = "r", entry = key, created = format(Sys.time(), "%Y-%m-%dT%H:%M:%S+00:00", tz = "UTC"),
       functai = as.character(utils::packageVersion("functai")), nodes = nodes)
}
