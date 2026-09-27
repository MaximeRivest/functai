# Shapes: what each input and output is. You write R prototypes, the way
# vctrs does (character(), integer(), a factor with its levels, a zero-row
# tibble for a record, list_of() for a list); functai writes the JSON Schema
# every language agrees on (contract/functions.md, "A definition") and
# builds answers back into those types.

#' Describe a field
#'
#' Words about an input or an output, which the model reads as guidance.
#' @param x A prototype (`character()`, `factor(levels = ...)`, ...).
#' @param desc What the field means.
#' @return A field.
#' @examples
#' described(character(), "the customer's own words")
#' @export
described <- function(x, desc) {
  f <- as_field(x)
  f$desc <- desc
  f
}

#' An input or output that may be missing
#'
#' The model may answer `null` for it; it comes back as `NA`.
#' @param x A prototype.
#' @return A field.
#' @export
optional <- function(x) {
  f <- as_field(x)
  f$shape <- list(anyOf = list(f$shape, list(type = "null")))
  f$nullable <- TRUE
  f
}

#' A record: several named fields
#'
#' Like a zero-row [tibble::tibble()] used as a type, but its fields may be
#' [described()] or [optional()]. Records come back as tibble columns
#' (`tidyr::unpack()` spreads them).
#' @param ... Fields, as `name = prototype`.
#' @return A field.
#' @examples
#' record(order_id = optional(character()), days_waiting = described(integer(), "whole days"))
#' @export
record <- function(...) {
  fields <- lapply(list(...), as_field)
  if (!length(fields) || is.null(names(fields)) || any(!nzchar(names(fields)))) cli::cli_abort("a record's fields are named: {.code record(name = character())}")
  props <- lapply(fields, function(f) if (is.null(f$desc)) f$shape else c(f$shape, list(description = f$desc)))
  new_field(list(type = "object", properties = props, required = as.list(names(fields))), "record", fields = fields)
}

#' A field written as JSON Schema
#'
#' For shapes no prototype says. Values come back as a list column.
#' @param schema A JSON Schema, as a named list.
#' @return A field.
#' @examples
#' json_shape(list(type = "object"))
#' @export
json_shape <- function(schema) {
  desc <- schema$description
  schema$description <- NULL
  new_field(schema, kind = "json", desc = desc)
}

new_field <- function(shape, kind, desc = NULL, levels = NULL, fields = NULL, item = NULL, nullable = FALSE) {
  structure(list(shape = shape, kind = kind, desc = desc, levels = levels, fields = fields, item = item,
                 nullable = nullable), class = "functai_field")
}

#' @export
print.functai_field <- function(x, ...) {
  cat("<functai field> ", lmcc::json_text(x$shape), if (!is.null(x$desc)) paste0("  # ", x$desc), "\n", sep = "")
  invisible(x)
}

#' Turn a prototype into a field
#'
#' `character()` is text, `integer()` a whole number, `double()` a number,
#' `logical()` yes or no, `factor(levels = c(...))` one of those levels, a
#' zero-row data frame or tibble a record of its columns (or [record()]), and
#' `vctrs::list_of(.ptype = x)` a list of `x`.
#' @param x A prototype or a field.
#' @return A field.
#' @keywords internal
#' @export
as_field <- function(x) {
  if (inherits(x, "functai_field")) return(x)
  if (is.factor(x)) {
    lv <- levels(x)
    if (!length(lv)) cli::cli_abort("a factor prototype needs its levels: {.code factor(levels = c(\"a\", \"b\"))}")
    return(new_field(list(enum = as.list(lv), type = "string"), "enum", levels = lv))
  }
  if (inherits(x, "vctrs_list_of")) {
    item <- as_field(attr(x, "ptype"))
    return(new_field(list(type = "array", items = item$shape), "list", item = item))
  }
  if (is.data.frame(x)) {
    fields <- lapply(x, as_field)
    if (!length(fields)) cli::cli_abort("a record prototype needs columns: {.code tibble::tibble(name = character())}")
    props <- lapply(fields, function(f) if (is.null(f$desc)) f$shape else c(f$shape, list(description = f$desc)))
    return(new_field(list(type = "object", properties = props, required = as.list(names(fields))), "record",
                     fields = fields))
  }
  if (is.character(x)) return(new_field(list(type = "string"), "string"))
  if (is.integer(x)) return(new_field(list(type = "integer"), "integer"))
  if (is.double(x)) return(new_field(list(type = "number"), "number"))
  if (is.logical(x)) return(new_field(list(type = "boolean"), "boolean"))
  cli::cli_abort(c("cannot use {.cls {class(x)}} as a type",
    i = "use character(), integer(), double(), logical(), factor(levels = ...), a zero-row tibble, vctrs::list_of(), or json_shape()"))
}

# ---------------------------------------------------------------- R values -> JSON

# One row's value of a column, for a model.
element <- function(x, i) {
  if (is.data.frame(x)) return(lapply(x, element, i = i))
  if (is.factor(x)) return(as.character(x[[i]]))
  if (inherits(x, "Date") || inherits(x, "POSIXt")) return(format(x[[i]]))
  x[[i]]
}

is_missing <- function(v) is.null(v) || (is.atomic(v) && length(v) == 1L && is.na(v))

# A value as the JSON its field's shape describes (lmcc's R conventions:
# named list = object, unnamed list = array, NULL = null).
to_json <- function(f, v) {
  if (is_missing(v)) return(NULL)
  switch(f$kind,
    string = , enum = as.character(v),
    integer = if (is.numeric(v) && abs(v) <= .Machine$integer.max) as.integer(v) else v,
    number = as.double(v),
    boolean = as.logical(v),
    record = {
      v <- as.list(v)
      out <- list()
      for (n in names(f$fields)) out[n] <- list(to_json(f$fields[[n]], v[[n]]))
      out
    },
    list = unname(lapply(as.list(v), function(x) to_json(f$item, x))),
    json = plain_json(v))
}

# Any R value as plain JSON (for the call log and for text inputs).
plain_json <- function(v) {
  if (is.null(v)) return(NULL)
  if (is.data.frame(v)) return(unname(lapply(seq_len(nrow(v)), function(i) plain_json(element(v, i)))))
  if (is.factor(v)) v <- as.character(v)
  if (inherits(v, "Date") || inherits(v, "POSIXt")) v <- format(v)
  if (is.list(v)) {
    out <- lapply(v, plain_json)
    if (is.null(names(v))) return(unname(out))
    if (!length(out)) return(lmcc::jobj())
    return(out)
  }
  if (is.atomic(v)) {
    if (length(v) == 1L) return(if (is.na(v)) NULL else unname(v))
    return(lapply(unname(v), function(x) if (is.na(x)) NULL else x))
  }
  cli::cli_abort("{.cls {class(v)}} has no JSON form")
}

# ---------------------------------------------------------------- JSON -> R vectors

num <- function(v) if (inherits(v, "lmcc_int")) as.double(unclass(v)) else v

# A column of `n` answers from JSON values (NULL where there is none).
assemble <- function(f, values) {
  n <- length(values)
  pick <- function(g) vapply(values, function(v) if (is_missing(v)) g(NA) else g(num(v)), g(NA))
  switch(f$kind,
    string = pick(function(v) as.character(v)),
    enum = factor(pick(function(v) as.character(v)), levels = f$levels),
    integer = {
      d <- pick(function(v) as.double(v))
      if (all(is.na(d) | abs(d) <= .Machine$integer.max)) as.integer(d) else d
    },
    number = pick(function(v) as.double(v)),
    boolean = pick(function(v) as.logical(v)),
    record = tibble::new_tibble(lapply(stats::setNames(nm = names(f$fields)), function(k)
      assemble(f$fields[[k]], lapply(values, function(v) if (is.list(v)) v[[k]] else NULL))), nrow = n),
    list = vctrs::new_list_of(lapply(values, function(v) if (is.null(v)) NULL else assemble(f$item, v)),
                              ptype = vctrs::vec_ptype(assemble(f$item, list()))),
    json = values)
}

# The first place a value does not fit a shape, or NULL (the JSON Schema
# subset shapes use).
misfit <- function(shape, v, where) {
  opts <- shape$anyOf %||% shape$oneOf
  if (!is.null(opts)) {
    for (o in opts) if (is.null(misfit(o, v, where))) return(NULL)
    return(sprintf("%s: %s fits none of its options", where, lmcc::json_text(v)))
  }
  if (!is.null(shape$enum) && !any(vapply(shape$enum, function(e) identical(lmcc::canonical_json(e), lmcc::canonical_json(v)), NA)))
    return(sprintf("%s: %s is not one of %s", where, lmcc::json_text(v), lmcc::json_text(shape$enum)))
  t <- shape$type
  if (is.character(t) && length(t) == 1L) {
    v2 <- num(v)
    ok <- switch(t,
      string = is.character(v) && length(v) == 1L,
      integer = is.numeric(v2) && length(v2) == 1L && !is.na(v2) && v2 == round(v2),
      number = is.numeric(v2) && length(v2) == 1L,
      boolean = is.logical(v) && length(v) == 1L,
      null = is.null(v),
      array = is.list(v) && is.null(names(v)),
      object = is.list(v) && (!is.null(names(v)) || !length(v)),
      TRUE)
    if (!ok) return(sprintf("%s: expected %s, got %s", where, t, lmcc::json_text(v)))
  }
  if (is.list(v) && is.null(names(v)) && is.list(shape$items))
    for (i in seq_along(v)) { p <- misfit(shape$items, v[[i]], sprintf("%s[%d]", where, i - 1L)); if (!is.null(p)) return(p) }
  if (is.list(v) && !is.null(names(v))) {
    for (k in unlist(shape$required)) if (!k %in% names(v)) return(sprintf("%s: missing %s", where, k))
    for (k in names(v)) {
      if (!is.null(shape$properties[[k]])) { p <- misfit(shape$properties[[k]], v[[k]], paste0(where, ".", k)); if (!is.null(p)) return(p) }
      else if (is.list(shape$additionalProperties)) { p <- misfit(shape$additionalProperties, v[[k]], paste0(where, ".", k)); if (!is.null(p)) return(p) }
    }
  }
  NULL
}
