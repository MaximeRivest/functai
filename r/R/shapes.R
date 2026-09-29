# Shapes: what each input and output is. You write a sentence (text, with
# words about it), a type the way vctrs does (character(), integer(), a
# zero-row tibble for a record, list_of() for a list), or choice(); a column
# of `.data` gives its own. functai writes the JSON Schema
# every language agrees on (contract/functions.md, "A definition") and
# builds answers back into those types.

#' One of a set of answers
#'
#' The model may only answer one of these; the answer comes back as a
#' factor with them as its levels, in this order. Name a level to say what
#' it means: the model reads it.
#' @param ... The answers: strings, character vectors, or a factor (its
#'   levels). Named, `level = "what it means"`.
#' @return A field.
#' @examples
#' choice("shipping", "billing", "product", "account")
#' choice(levels(refunds$state))
#' choice(
#'   approve = "the rules allow a refund",
#'   deny    = "they do not",
#'   review  = "a person should decide")
#' @export
choice <- function(...) {
  parts <- rlang::list2(...)
  outer <- names(parts) %||% rep("", length(parts))
  levels <- character(0); meanings <- character(0)
  for (i in seq_along(parts)) {
    v <- parts[[i]]
    if (is.factor(v)) v <- levels(v)
    if (!is.character(v) || anyNA(v)) cli::cli_abort("{.fn choice}'s answers are strings, not {.cls {class(v)}}")
    if (nzchar(outer[[i]])) {
      if (length(v) != 1L) cli::cli_abort("{.code {outer[[i]]} = } says what the answer {.val {outer[[i]]}} means, in one string")
      levels <- c(levels, outer[[i]]); meanings <- c(meanings, stats::setNames(v, outer[[i]]))
    } else if (!is.null(names(v)) && all(nzchar(names(v)))) {
      levels <- c(levels, names(v)); meanings <- c(meanings, v)
    } else levels <- c(levels, unname(v))
  }
  if (!length(levels)) cli::cli_abort("{.fn choice} needs its answers: {.code choice(\"yes\", \"no\")}")
  if (anyDuplicated(levels)) cli::cli_abort("{.fn choice} has {.val {levels[duplicated(levels)]}} twice")
  f <- new_field(list(enum = as.list(levels), type = "string"), "enum", levels = levels)
  if (length(meanings)) f$meanings <- vapply(meanings, trim_white, "")
  f
}

#' Describe a field
#'
#' Words about an input or an output, which the model reads as guidance.
#' In [ai()], a sentence alone describes a text field (or a field whose
#' type comes from `.data`); `described()` gives words to any other type.
#' @param x A type (`integer()`, [choice()], ...).
#' @param desc What the field means.
#' @return A field.
#' @examples
#' described(integer(), "whole days since the parcel arrived")
#' @export
described <- function(x, desc) {
  f <- as_field(x)
  if (!rlang::is_string(desc)) cli::cli_abort("{.arg desc} is one string")
  f$desc <- trim_white(desc)
  f
}

# The words the model reads about a field: yours, then what each answer of a
# choice means.
field_desc <- function(f) {
  m <- if (length(f$meanings)) paste(sprintf("%s: %s", names(f$meanings), f$meanings), collapse = "; ")
  if (is.null(f$desc)) m else if (is.null(m)) f$desc else paste0(sub("[.;:,]$", "", f$desc), ". ", m)
}

# A field's JSON Schema, with its words as the description.
described_shape <- function(f) {
  d <- field_desc(f)
  if (is.null(d)) f$shape else c(f$shape, list(description = d))
}

# The type of a column, as a prototype: a factor is a choice of its levels,
# dates and anything else are text.
prototype_of <- function(x) {
  if (is.factor(x)) return(factor(levels = levels(x)))
  if (is.integer(x)) return(integer())
  if (is.double(x) && !inherits(x, c("Date", "POSIXt", "difftime"))) return(double())
  if (is.logical(x)) return(logical())
  character()
}

#' An input or output that may be missing
#'
#' The model may answer `null` for it; it comes back as `NA`.
#' @param x A type, or a sentence (text, described by it).
#' @return A field.
#' @examples
#' optional(integer())
#' optional("the order number, when the message gives one")
#' @export
optional <- function(x) {
  f <- as_field(x)
  default <- if ("default" %in% names(f$shape)) f$shape["default"]      # an input's own default stays its own
  f$shape <- c(list(anyOf = list(data_shape(f$shape), list(type = "null"))), default)
  f$nullable <- TRUE
  f
}

#' A record: several named fields
#'
#' Several named fields that belong together, each a type or a sentence
#' (text, described), as in [ai()]. Records come back as tibble columns
#' (`tidyr::unpack()` spreads them).
#' @param ... Fields, as `name = type`.
#' @return A field.
#' @examples
#' record(order_id = optional("a letter, a dash and four digits"), days_waiting = described(integer(), "whole days"))
#' @export
record <- function(...) {
  fields <- lapply(list(...), as_field)
  if (!length(fields) || is.null(names(fields)) || any(!nzchar(names(fields)))) cli::cli_abort("a record's fields are named: {.code record(name = character())}")
  if (any(vapply(fields, function(f) isTRUE(f$optional), NA)))
    cli::cli_abort("a default belongs to an input of the function ({.fn defaults_to}), not to a field of a record")
  new_field(list(type = "object", properties = lapply(fields, described_shape), required = as.list(names(fields))), "record", fields = fields)
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
                 nullable = nullable, meanings = NULL, optional = FALSE), class = "functai_field")
}

#' @export
print.functai_field <- function(x, ...) {
  cat("<functai field> ", type_label(x), if (!is.null(field_desc(x))) paste0("  # ", field_desc(x)), "\n", sep = "")
  invisible(x)
}

#' Turn a type into a field
#'
#' A sentence is text, described by it; `character()` is text, `integer()`
#' a whole number, `double()` a number, `logical()` yes or no, [choice()] or
#' `factor(levels = c(...))` one of a set of answers, [record()] or a
#' zero-row tibble a record of its columns, and `vctrs::list_of(.ptype = x)`
#' a list of `x`.
#' @param x A type, a sentence, or a field.
#' @return A field.
#' @keywords internal
#' @export
as_field <- function(x) {
  if (inherits(x, "functai_field")) return(x)
  if (is.character(x) && length(x)) {
    if (length(x) == 1L && !is.na(x)) return(described(character(), x))
    cli::cli_abort(c("a field is a type or one sentence about it, not {length(x)} strings",
      i = "one of several answers is a choice: {.code choice({paste(encodeString(utils::head(x, 2L), quote = '\"'), collapse = ', ')}, ...)}"))
  }
  if (inherits(x, c("Date", "POSIXt"))) return(new_field(list(type = "string"), "string"))   # dates are text, as a column of them is
  if (is.factor(x)) {
    lv <- levels(x)
    if (!length(lv)) cli::cli_abort("a factor prototype needs its levels: {.code factor(levels = c(\"a\", \"b\"))}")
    return(new_field(list(enum = as.list(lv), type = "string"), "enum", levels = lv))
  }
  if (inherits(x, "vctrs_list_of")) {
    item <- as_field(attr(x, "ptype"))
    if (isTRUE(item$optional)) cli::cli_abort("a default belongs to an input of the function ({.fn defaults_to}), not to a list's items")
    return(new_field(list(type = "array", items = item$shape), "list", item = item))
  }
  if (is.data.frame(x)) {
    fields <- lapply(x, as_field)
    if (!length(fields)) cli::cli_abort("a record prototype needs columns: {.code tibble::tibble(name = character())}")
    return(new_field(list(type = "object", properties = lapply(fields, described_shape), required = as.list(names(fields))),
                     "record", fields = fields))
  }
  if (is.character(x)) return(new_field(list(type = "string"), "string"))
  if (is.integer(x)) return(new_field(list(type = "integer"), "integer"))
  if (is.double(x)) return(new_field(list(type = "number"), "number"))
  if (is.logical(x)) return(new_field(list(type = "boolean"), "boolean"))
  cli::cli_abort(c("cannot use {.cls {class(x)}} as a type",
    i = "use a sentence (text), character(), integer(), double(), logical(), choice(...), record(...), vctrs::list_of(), or json_shape()"))
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

# A value given for a field, as JSON (lmcc's R conventions: named list =
# object, unnamed list = array, NULL = null). The JSON depends on the value
# and the field's shape alone, never on the R type the field is read as, so
# the same value is sent the same by a function and by the one loaded from
# its folder, whatever R type each reads the field as.
to_json <- function(f, v) json_writer(f)(v)

# to_json() for one field, its shape read once (a column writes every row
# by it).
json_writer <- function(f) {
  shape <- data_shape(f$shape)
  switch(guide_type(value_guide(shape, shape)) %||% "",
    string = , boolean = function(v) if (is_missing(v)) NULL else plain_json(v),
    function(v) json_of(v, shape, shape))
}

# A value as JSON, guided by a shape (`root` holds its `$defs`). Nothing is
# lost, made up or changed to fit: a value that is not of the shape's type
# stays what it is (2.5 given to a whole number stays 2.5), an object keeps
# every member it has (those its shape does not name too) and has none it
# does not have, so the check of the value refuses, as given, what does not
# fit (calls.md, *Values*: the log writes a value by what it is). The shape
# only says what R leaves open: a vector for an array is a list of its
# items (even of one), a one-row tibble for an object is that row, and a
# whole double for an integer is written as one. One thing R cannot say is
# a member left out of a row of a tibble: there, and in any record, `NA`
# for a member the shape does not require and whose type takes no null is
# left out, as an answer that leaves it out reads back.
json_of <- function(v, shape, root) {
  if (is_missing(v)) return(NULL)
  s <- value_guide(shape, root)
  switch(guide_type(s) %||% "",
    integer = if (is.numeric(v) && length(v) == 1L && is.finite(v) && v == round(v) && abs(v) <= .Machine$integer.max)
      as.integer(v) else plain_json(v),
    number = if (is.numeric(v) && length(v) == 1L && !inherits(v, "lmcc_int")) as.double(v) else plain_json(v),  # lmcc's big integers keep their digits
    array = array_json(v, s, root),
    object = object_json(v, s, root),
    plain_json(v))
}

array_json <- function(v, s, root) {
  prefix <- if (is_arr(s$prefixItems)) s$prefixItems else list()
  item <- function(i) if (i <= length(prefix)) prefix[[i]] else if (is_obj(s$items)) s$items
  if (is.data.frame(v)) return(lapply(seq_len(nrow(v)), function(i) json_of(element(v, i), item(i), root)))
  if (is.list(v) && !is.null(names(v)) && length(v)) return(plain_json(v))       # a named list is an object
  if (is.list(v) || is.atomic(v)) return(lapply(seq_along(v), function(i) json_of(element(unname(v), i), item(i), root)))
  plain_json(v)
}

object_json <- function(v, s, root) {
  if (is.data.frame(v)) { if (nrow(v) != 1L) return(plain_json(v)); v <- element(v, 1L) }
  if (is.atomic(v) && !is.null(names(v))) v <- as.list(v)
  if (!is.list(v)) return(plain_json(v))
  if (!length(v)) return(lmcc::jobj())
  nm <- names(v)
  if (is.null(nm) || any(!nzchar(nm))) return(plain_json(v))
  props <- if (is_obj(s$properties)) s$properties else list()
  required <- unlist(s$required)
  extra <- if (is_obj(s$additionalProperties)) s$additionalProperties
  out <- list()
  for (i in seq_along(v)) {
    n <- nm[[i]]; m <- v[[i]]
    member <- if (n %in% names(props)) props[[n]] else extra
    if (n %in% names(props) && !n %in% required && is.atomic(m) && length(m) == 1L && is.na(m) &&
        !fits_shape(NULL, member, root)) next                                    # a record's NA: left out
    out[n] <- list(if (is.null(member)) plain_json(m) else json_of(m, member, root))
  }
  if (!length(out)) lmcc::jobj() else out
}

# The one shape a value's JSON form follows: through `$ref`, and an `anyOf`
# whose options but one are null; NULL when there is none (several options).
value_guide <- function(shape, root, seen = character(0)) {
  if (!is_obj(shape)) return(NULL)
  if (has_key(shape, "$ref")) {
    n <- ref_name(shape[["$ref"]])
    if (is.null(n) || n %in% seen) return(NULL)
    return(value_guide(root[["$defs"]][[n]], root, c(seen, n)))
  }
  if (has_key(shape, "anyOf")) {
    opts <- Filter(function(o) !(is_obj(o) && identical(o$type, "null")), shape$anyOf)
    return(if (length(opts) == 1L) value_guide(opts[[1L]], root, seen))
  }
  shape
}

guide_type <- function(s) {
  if (is.null(s)) return(NULL)
  t <- s$type
  if (is_arr(t)) t <- setdiff(unlist(t), "null")
  if (is.character(t) && length(t) == 1L) return(t)
  if (is.null(t) && is_obj(s$properties)) return("object")
  if (is.null(t) && (is_obj(s$items) || is_arr(s$prefixItems))) return("array")
  NULL
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

