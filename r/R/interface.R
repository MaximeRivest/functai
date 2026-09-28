# A program's interface (contract/programs.md): the inputs it takes and the
# outputs it gives, as data every language reads the same way. An AI
# function's is its definition's inputs and outputs; this file checks one
# (the vocabulary "fits" reads, which interfaces are refused), gives its
# signature, and lets an R user see it (`ai_interface()`) and give an input a
# default (`defaults_to()`).

# ---------------------------------------------------------------- JSON values, as lmcc's parser gives them

is_obj <- function(x) is.list(x) && !is.null(names(x))
is_arr <- function(x) is.list(x) && is.null(names(x))
is_str <- function(x) is.character(x) && length(x) == 1L && !is.na(x)
is_flag <- function(x) is.logical(x) && length(x) == 1L && !is.na(x)
is_num <- function(x) { x <- num(x); is.numeric(x) && length(x) == 1L && !is.na(x) && !is.logical(x) }

# JSON's kind of a value: an integer is a number with no fraction (2.0 is one).
json_type <- function(v) {
  if (is.null(v)) return("null")
  if (is_flag(v)) return("boolean")
  if (is_num(v)) { v <- num(v); return(if (is.finite(v) && v == floor(v)) "integer" else "number") }
  if (is_str(v)) return("string")
  if (is_arr(v)) return("array")
  if (is_obj(v)) return("object")
  "unknown"
}

# A value as JSON data, whatever R built it from (a named list is an object,
# an unnamed one an array): what every check below reads.
json_normal <- function(x) lmcc::parse_json(lmcc::canonical_json(x))

has_key <- function(x, k) is.list(x) && k %in% names(x)

# Whole-text matches (a regular expression's `$` may match before a final
# newline: programs.md asks for the whole text).
NAME_RE <- "\\A[A-Za-z_][A-Za-z0-9_]*\\z"
REF_RE <- "\\A#/\\$defs/([A-Za-z0-9_.-]+)\\z"
is_name <- function(x) is_str(x) && grepl(NAME_RE, x, perl = TRUE)
ref_name <- function(ref) if (is_str(ref) && grepl(REF_RE, ref, perl = TRUE)) sub(REF_RE, "\\1", ref, perl = TRUE) else NULL

# ---------------------------------------------------------------- the vocabulary (programs.md, "Checking values")

TYPE_NAMES <- c("null", "boolean", "integer", "number", "string", "array", "object")
WORDS <- c(title = "string", description = "string", format = "string", `$comment` = "string", deprecated = "boolean",
           readOnly = "boolean", writeOnly = "boolean", examples = "array", default = "any")
ASSERTIONS <- c("type", "enum", "const", "anyOf", "items", "prefixItems", "minItems", "maxItems", "uniqueItems",
                "properties", "required", "additionalProperties", "minLength", "maxLength", "minimum", "maximum",
                "exclusiveMinimum", "exclusiveMaximum", "$ref", "$defs")

# Whether a shape uses the listed keywords, each with a value of its kind.
# `carry`: an AI function's shape, whose other keywords are lmcc's (carried,
# never read here); a module's may have no other keyword.
well_formed <- function(shape, root, carry = FALSE) {
  if (!is_obj(shape)) return(FALSE)
  for (k in names(shape)) {
    v <- shape[[k]]
    if (k %in% names(WORDS)) {
      kind <- WORDS[[k]]
      ok <- switch(kind, string = is_str(v), boolean = is_flag(v), array = is_arr(v), any = TRUE)
      if (!ok) return(FALSE)
      next
    }
    if (!k %in% ASSERTIONS) { if (carry) next else return(FALSE) }
    ok <- switch(k,
      type = {
        names_ <- if (is_arr(v)) v else list(v)
        length(names_) > 0L && all(vapply(names_, function(n) is_str(n) && n %in% TYPE_NAMES, NA)) &&
          !anyDuplicated(vapply(names_, as.character, ""))
      },
      enum = is_arr(v) && length(v) > 0L,
      const = TRUE,
      anyOf = , prefixItems = is_arr(v) && length(v) > 0L && all(vapply(v, well_formed, NA, root = root, carry = carry)),
      items = well_formed(v, root, carry),
      properties = , `$defs` = is_obj(v) && all(vapply(v, well_formed, NA, root = root, carry = carry)),
      additionalProperties = is_flag(v) || well_formed(v, root, carry),
      required = is_arr(v) && all(vapply(v, is_str, NA)) && !anyDuplicated(vapply(v, function(x) if (is_str(x)) x else "", "")),
      minItems = , maxItems = , minLength = , maxLength = identical(json_type(v), "integer") && num(v) >= 0,
      minimum = , maximum = , exclusiveMinimum = , exclusiveMaximum = is_num(v),
      uniqueItems = is_flag(v),
      `$ref` = { n <- ref_name(v); !is.null(n) && has_key(root[["$defs"]], n) })
    if (!isTRUE(ok)) return(FALSE)
  }
  TRUE
}

# The $defs a shape checks the same value against: its $ref, and those of its
# anyOf's shapes, without passing into an item or a member.
same_value_refs <- function(shape) {
  out <- character(0)
  if (has_key(shape, "$ref")) out <- c(out, ref_name(shape[["$ref"]]))
  for (x in shape[["anyOf"]]) out <- c(out, same_value_refs(x))
  unique(out)
}

# Whether a $defs entry comes back to itself through $ref and anyOf alone:
# checking a value against it would never end.
loops <- function(root) {
  defs <- root[["$defs"]]
  if (!is_obj(defs) || !length(defs)) return(FALSE)
  graph <- lapply(defs, same_value_refs)
  reaches <- function(start, seen) {
    for (n in graph[[start]] %||% character(0)) {
      if (identical(n, seen[[1L]]) || (!n %in% seen && reaches(n, c(seen, n)))) return(TRUE)
    }
    FALSE
  }
  any(vapply(names(graph), function(n) reaches(n, n), NA))
}

same_json <- function(a, b) identical(lmcc::canonical_json(a), lmcc::canonical_json(b))

# Whether a JSON value fits a shape, read by the listed keywords alone
# (programs.md): lengths in code points, equality as canonical JSON. Every
# check of a value reads this one rule: a default when a function is defined
# or loaded, an input when a call binds it, an output when a reply is read.
fits_shape <- function(v, shape, root) is.null(shape_fault(v, shape, root))

# A value as a short text for a message (a value can be long).
short_json <- function(v) {
  s <- tryCatch(lmcc::json_text(v), error = function(e) "a value with no JSON form")
  if (nchar(s) > 80L) paste0(substr(s, 1L, 77L), "...") else s
}

# Why a JSON value does not fit a shape (the first place it does not, as
# `where`: `n`, `options.b`, `items[2]`), or NULL when it fits. Only the
# keywords programs.md lists are read; an AI function's other keywords
# (`pattern`, `oneOf`, ...) are lmcc's, and never checked here.
shape_fault <- function(v, shape, root, where = "value") {
  t <- json_type(v)
  if (has_key(shape, "$ref")) {
    p <- shape_fault(v, root[["$defs"]][[ref_name(shape[["$ref"]])]], root, where)
    if (!is.null(p)) return(p)
  }
  if (has_key(shape, "type")) {
    names_ <- unlist(if (is_arr(shape$type)) shape$type else list(shape$type))
    if (!(t %in% names_ || (t == "integer" && "number" %in% names_)))
      return(sprintf("%s: %s is not %s", where, short_json(v), paste(names_, collapse = " or ")))
  }
  if (has_key(shape, "enum") && !any(vapply(shape$enum, same_json, NA, b = v)))
    return(sprintf("%s: %s is not one of %s", where, short_json(v), short_json(shape$enum)))
  if (has_key(shape, "const") && !same_json(shape[["const"]], v))
    return(sprintf("%s: %s is not %s", where, short_json(v), short_json(shape[["const"]])))
  if (has_key(shape, "anyOf") && !any(vapply(shape$anyOf, function(s) fits_shape(v, s, root), NA)))
    return(sprintf("%s: %s fits none of its options", where, short_json(v)))
  if (t == "string") {
    n <- nchar(v, type = "chars")
    if (has_key(shape, "minLength") && n < num(shape$minLength)) return(sprintf("%s: shorter than %s characters", where, num(shape$minLength)))
    if (has_key(shape, "maxLength") && n > num(shape$maxLength)) return(sprintf("%s: longer than %s characters", where, num(shape$maxLength)))
  }
  if (t %in% c("integer", "number")) {
    x <- num(v)
    bound <- function(k, bad, words) if (has_key(shape, k) && bad(x, num(shape[[k]]))) sprintf("%s: %s is %s %s", where, short_json(v), words, short_json(shape[[k]]))
    p <- bound("minimum", `<`, "less than") %||% bound("maximum", `>`, "more than") %||%
      bound("exclusiveMinimum", `<=`, "not more than") %||% bound("exclusiveMaximum", `>=`, "not less than")
    if (!is.null(p)) return(p)
  }
  if (t == "array") {
    prefix <- shape[["prefixItems"]] %||% list()
    at <- function(i) sprintf("%s[%d]", where, i - 1L)
    for (i in seq_len(min(length(prefix), length(v)))) { p <- shape_fault(v[[i]], prefix[[i]], root, at(i)); if (!is.null(p)) return(p) }
    if (has_key(shape, "items") && length(v) > length(prefix))
      for (i in (length(prefix) + 1L):length(v)) { p <- shape_fault(v[[i]], shape$items, root, at(i)); if (!is.null(p)) return(p) }
    if (has_key(shape, "minItems") && length(v) < num(shape$minItems)) return(sprintf("%s: fewer than %s items", where, num(shape$minItems)))
    if (has_key(shape, "maxItems") && length(v) > num(shape$maxItems)) return(sprintf("%s: more than %s items", where, num(shape$maxItems)))
    if (isTRUE(shape$uniqueItems) && anyDuplicated(vapply(v, lmcc::canonical_json, ""))) return(sprintf("%s: an item is there twice", where))
  }
  if (t == "object") {
    props <- shape[["properties"]] %||% list()
    for (k in unlist(shape[["required"]])) if (!k %in% names(v)) return(sprintf("%s: no %s", where, k))
    for (k in names(v)) {
      inner <- paste0(where, ".", k)
      if (k %in% names(props)) {
        p <- shape_fault(v[[k]], props[[k]], root, inner)
        if (!is.null(p)) return(p)
      } else if (has_key(shape, "additionalProperties")) {
        extra <- shape$additionalProperties
        if (isFALSE(extra)) return(sprintf("%s: no member %s is allowed", where, k))
        if (is_obj(extra)) { p <- shape_fault(v[[k]], extra, root, inner); if (!is.null(p)) return(p) }
      }
    }
  }
  NULL
}

# A field's shape without its own `default`: what its data looks like.
data_shape <- function(shape) shape[names(shape) != "default"]

# ---------------------------------------------------------------- interfaces

INPUT_KEYS <- c("name", "shape", "desc", "type", "opaque", "optional")
OUTPUT_KEYS <- c("name", "shape", "desc", "type", "opaque")

# Why an interface is refused (programs.md, "Interfaces that are refused"),
# or NULL: list(field = the first field at fault, inputs then outputs, or
# NULL when the fault is not a field's). `ai`: an AI function's, whose shapes
# may carry lmcc's other keywords.
interface_problem <- function(iface, ai = FALSE) {
  top <- list(field = NULL)
  if (!is_obj(iface) || length(setdiff(names(iface), c("description", "inputs", "outputs"))) ||
      !is_str(iface$description) || !is_arr(iface$inputs) || !is_arr(iface$outputs) || !length(iface$outputs))
    return(top)
  seen <- character(0)
  for (direction in c("inputs", "outputs")) {
    allowed <- if (direction == "inputs") INPUT_KEYS else OUTPUT_KEYS
    for (f in iface[[direction]]) {
      name <- if (is_obj(f) && is_str(f$name)) f$name else NULL
      at_fault <- list(field = name)
      if (!is_obj(f) || length(setdiff(names(f), allowed))) return(at_fault)
      if (!is_name(name) || name %in% seen) return(at_fault)
      seen <- c(seen, name)
      for (k in c("desc", "type")) if (has_key(f, k) && !is_str(f[[k]])) return(at_fault)
      for (k in c("opaque", "optional")) if (has_key(f, k) && !isTRUE(f[[k]])) return(at_fault)
      shape <- f$shape
      if (!has_key(f, "shape") || !is_obj(shape)) return(at_fault)
      if (!well_formed(shape, shape, carry = ai) || loops(shape)) return(at_fault)
      if (ai && isTRUE(f$optional) && !has_key(shape, "default")) return(at_fault)
      if (isTRUE(f$opaque) && length(shape)) return(at_fault)
      if (has_key(shape, "default")) {
        ds <- data_shape(shape)
        if (!fits_shape(shape[["default"]], ds, ds)) return(at_fault)
      }
    }
  }
  NULL
}

# The interface's signature (programs.md): lmcc's fingerprint of its fields,
# each plain and untyped, each shape without its own default. The call log's
# program.interface.
interface_signature <- function(iface) {
  field <- function(f, direction) list(direction = direction, name = f$name, purpose = "plain", shape = data_shape(f$shape), type = "")
  lmcc::sha256_of(c(lapply(iface$inputs, field, "input"), lapply(iface$outputs, field, "output")))
}

# An AI function's interface: its definition's description, inputs (with
# `optional`, the default in the shape) and outputs, as JSON data; a loaded
# function's is its saved node's.
interface_of <- function(core) {
  if (!is.null(core$interface)) return(core$interface)
  d <- core$definition
  field <- function(n, f, input) {
    out <- list(name = n, shape = f$shape)
    desc <- field_desc(f)
    if (!is.null(desc) && nzchar(desc)) out$desc <- desc
    if (isTRUE(f$optional)) out$optional <- TRUE     # on an output: refused, in its turn (programs.md)
    out
  }
  json_normal(list(description = d$description,
                   inputs = unname(Map(field, names(d$inputs), d$inputs, TRUE)),
                   outputs = unname(Map(field, names(d$outputs), d$outputs, FALSE))))
}

# ---------------------------------------------------------------- refusals

# A refusal the contract names (contract/README.md, "Refusal codes"): an
# error with its `code` and, where there is one, the `field` at fault.
refuse <- function(code, message, field = NULL, class = NULL, call = NULL, .envir = parent.frame()) {
  cli::cli_abort(message, class = c(class, paste0("functai_", gsub("-", "_", code)), "functai_refusal"),
                 code = code, field = field, call = call, .envir = .envir)
}

# An AI function is checked when it is defined (functions.md, "A
# definition"): lmcc's own check of its signature, then its interface by
# programs.md's rules, then its own log_content map against its fields.
check_definition <- function(core, call = NULL) {
  signature_of(core, effective(core$own))                     # lmcc refuses signature-malformed first
  iface <- interface_of(core)
  problem <- interface_problem(iface, ai = TRUE)
  if (!is.null(problem)) {
    f <- problem$field
    why <- interface_fault(iface, f)
    column <- if (!is.null(f)) columns_of(core)[f] else NA
    shown <- if (!is.na(column %||% NA) && !identical(unname(column), f)) sprintf(" (%s, as the formula names it)", column) else ""
    refuse("interface-malformed", c("{.fn {core$definition$name}} cannot be defined: {why}{shown}",
      i = "an interface every language reads: see {.fn ai_interface}"), field = f, call = call)
  }
  check_own_content(core, call = call)
  invisible(core)
}

# A function's own settings as it keeps them (and saves them): its
# log_content by field name.
own_settings <- function(core) {
  if (is.list(core$own$log_content)) core$own$log_content <- field_names_in(core$own$log_content, core)
  core
}

# The fault programs.md found in a field, in words.
interface_fault <- function(iface, f) {
  if (is.null(f)) return("its interface is not one programs.md allows")
  field <- NULL
  for (x in c(iface$inputs, iface$outputs)) if (identical(x$name, f)) { field <- x; break }
  if (!is_name(f)) return(sprintf("the field name %s is not an ASCII identifier (letters, digits and _, not starting with a digit); write it as %s",
                                  encodeString(f, quote = "\""), gsub("[^A-Za-z0-9_]+", "_", f)))
  shape <- field$shape
  if (isTRUE(field$optional) && !has_key(shape, "default")) return(sprintf("the optional input %s has no default", f))
  if (has_key(shape, "default")) {
    ds <- data_shape(shape)
    if (well_formed(shape, shape, carry = TRUE) && !loops(shape) && !fits_shape(shape[["default"]], ds, ds))
      return(sprintf("%s's default does not fit its type (%s)", f, shape_fault(shape[["default"]], ds, ds, f)))
  }
  if (!is.null(field) && isTRUE(field$optional) && any(vapply(iface$outputs, function(x) identical(x$name, f), NA)))
    return(sprintf("%s is an output: only an input has a default (defaults_to() is for an input the caller may leave out)", f))
  sprintf("the field %s: its shape %s uses a keyword with a value of the wrong kind, or a reference that never ends",
          f, lmcc::canonical_json(shape %||% lmcc::jobj()))
}

# ---------------------------------------------------------------- an input's default

#' An input the caller may leave out
#'
#' In [ai()]'s codebook, `tone = defaults_to("kind")` makes `tone` an input
#' a caller may leave out: `reply(message)` sends `"kind"`, as an R
#' function's default argument does, and the call log records the value
#' sent. The model is sent every input, so an input left out is sent with
#' its default. The default is part of the function's interface
#' ([ai_interface()]), not of what its calls record as data: changing it
#' changes neither the function's signature nor its version, only what a
#' call that leaves the input out sends.
#'
#' The default must be of the type, as vctrs casts (`defaults_to(2.5,
#' integer())` is refused: it would lose the half; so is `defaults_to(TRUE,
#' character())`), and fit it (a [choice()]'s default is one of its
#' answers); a function whose default does not fit is refused when it is
#' defined (`interface-malformed`). Only inputs have defaults.
#'
#' A call that leaves the input out sends the default exactly as the
#' interface holds it (its JSON), never a copy made through an R type: a
#' record's default that leaves out a member sends it without that member.
#' @param value The default: one value (a string, a number, `TRUE`, a date,
#'   a one-row tibble for a record, `NA` or `NULL` for an [optional()]
#'   type).
#' @param type Its type, as in [ai()]; or a sentence, words about the type
#'   `value` has. Default: the type of `value` (text for a string or a date,
#'   a whole number for an integer, ...).
#' @return A field.
#' @examples
#' reply <- ai(reply ~ message + tone, "Answer the customer.",
#'   message = "the customer's own words",
#'   tone = defaults_to("kind", choice("kind", "brief", "formal")))
#' reply
#' ai_interface(reply)
#' defaults_to(3L, "how many suggestions to give")     # a whole number, described
#' @export
defaults_to <- function(value, type = NULL) {
  sentence <- rlang::is_string(type)
  f <- if (is.null(type) || sentence) type_of_value(value) else as_field(type)
  if (sentence) f <- described(f, type)
  f$shape <- f$shape[names(f$shape) != "default"]
  f$shape["default"] <- list(default_json(f, value))
  f$optional <- TRUE
  f
}

# The type a default's value has: text for a string (or a date, as a column
# of dates is text), a whole number for an integer, a record for a one-row
# tibble, a choice of a factor's levels.
type_of_value <- function(value) {
  if (is.null(value) || (is.atomic(value) && length(value) == 1L && is.na(value)))
    cli::cli_abort(c("a default of {.code {deparse(value)}} needs its type", i = "{.code defaults_to(NA, optional(integer()))}"), call = NULL)
  if (is.data.frame(value)) return(as_field(vctrs::vec_ptype(value)))
  if (is.factor(value)) return(as_field(factor(levels = levels(value))))
  if (is.atomic(value) && length(value) == 1L) return(as_field(prototype_of(value)))
  cli::cli_abort(c("say the type of a default that is not one value", i = "{.code defaults_to(list(), vctrs::list_of(.ptype = character()))}"), call = NULL)
}

# A default as the JSON its field holds: cast to the field's R type first, as
# vctrs casts (refusing what would lose information: 2.5 as a whole number,
# TRUE as text), then written as a value of that type.
default_json <- function(f, value) {
  if (is.null(value)) return(NULL)
  if (is.data.frame(value)) {
    if (f$kind != "record") cli::cli_abort("a one-row tibble is the default of a record, not of {type_label(f)}", call = NULL)
    if (nrow(value) != 1L) cli::cli_abort("a record's default is a one-row tibble, not {nrow(value)} rows", call = NULL)
    return(to_json(f, element(value, 1L)))
  }
  if (is.atomic(value) && length(value) != 1L && !f$kind %in% c("list", "json"))
    cli::cli_abort("a default is one value, not {length(value)}", call = NULL)
  proto <- switch(f$kind, string = character(), enum = character(), integer = integer(), number = double(), boolean = logical(), NULL)
  if (!is.null(proto) && is.atomic(value)) {
    if (inherits(value, c("Date", "POSIXt")) && is.character(proto)) value <- format(value)
    value <- tryCatch(vctrs::vec_cast(unname(value), proto, x_arg = "value"), error = function(e)
      cli::cli_abort(c("the default {.code {deparse(value)}} is not {type_label(f)}",
                       i = "give a default of the input's type, or say its type: {.code defaults_to(value, type)}"), parent = e, call = NULL))
  }
  if (f$kind == "list" && !is.list(value)) value <- as.list(value)
  to_json(f, value)
}

# An input's default as one row of its column: what the R function's
# argument shows as its default. A call that leaves the input out sends the
# shape's default itself (input_rows()), never this R copy of it.
default_value <- function(f) {
  json <- f$shape[["default"]]
  if (f$kind == "json" && is.atomic(json) && length(json) == 1L) return(json)     # one value reads as itself
  v <- assemble(f, list(json))
  if (is.factor(v)) as.character(v) else v
}

# ---------------------------------------------------------------- seeing an interface

#' What a program takes and gives
#'
#' A program's interface (contract/programs.md): its description, its
#' inputs and its outputs, each named and typed as JSON Schema, as every
#' language describes it. For an AI function it is its formula and codebook:
#' an input with a default ([defaults_to()]) is `optional`, its default in
#' its shape; one output is `result`. Given a saved folder, it describes a
#' program in it without loading or running anything, whatever language
#' saved it (a module too, which R cannot run): a server, a check or a
#' notebook can say what a program takes before deciding to load it.
#' @param x An AI function, or a saved folder (or its `functai.json`).
#' @param node For a folder: which program, by key (`"module:name"`);
#'   default the entry.
#' @param ... Unused.
#' @return The interface (a list, as JSON), of class `functai_interface`.
#'   Refused, it is an error of class `functai_refusal` with its `code`
#'   (`saved-format`, `saved-malformed`, `saved-no-interface`,
#'   `interface-malformed`, `saved-differs`).
#' @examples
#' mood <- ai(mood ~ review, "How does the customer feel about what they bought?",
#'   mood = choice("happy", "unhappy", "mixed"))
#' ai_interface(mood)
#' @export
ai_interface <- function(x, ...) UseMethod("ai_interface")

#' @rdname ai_interface
#' @export
ai_interface.functai_fn <- function(x, ...) structure(interface_of(core_of(x)), class = "functai_interface")

#' @rdname ai_interface
#' @export
ai_interface.character <- function(x, node = NULL, ...) {
  m <- read_manifest(x)
  structure(describe_manifest(m$manifest, node), class = "functai_interface")
}

#' @export
ai_interface.default <- function(x, ...) cli::cli_abort("{.fn ai_interface} describes an AI function or a saved folder, not {.cls {class(x)}}")

#' @export
print.functai_interface <- function(x, ...) {
  words <- trim_white(x$description)
  cat("<interface>", if (nzchar(words)) paste0(" ", gsub("\\s*\n\\s*", " ", words)), "\n", sep = "")
  fields <- c(x$inputs, x$outputs)
  w <- max(nchar(vapply(fields, function(f) f$name, "")))
  line <- function(f) {
    shape <- lmcc::canonical_json(data_shape(f$shape))
    # optional with no default of its own (a module's): its code's default applies, which may be no JSON value
    extra <- c(if (isTRUE(f$optional)) (if (has_key(f$shape, "default")) paste0("optional, default ", lmcc::canonical_json(f$shape[["default"]])) else "optional"),
               if (isTRUE(f$opaque)) "opaque")
    cat(sprintf("    %s  %s%s%s\n", formatC(f$name, width = -w), shape, if (length(extra)) paste0("  (", paste(extra, collapse = ", "), ")") else "",
                if (is.null(f$desc)) "" else paste0("  # ", gsub("\\s*\n\\s*", " ", f$desc))))
  }
  cat("  inputs:\n"); for (f in x$inputs) line(f)
  cat("  outputs:\n"); for (f in x$outputs) line(f)
  invisible(x)
}
