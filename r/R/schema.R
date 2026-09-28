# The contract's JSON Schemas (inst/contract/schema, copied from
# contract/schema by r/check), read by the keywords they use. A saved folder's
# manifest must pass saved.schema.json before anything else is read
# (saved.md: `saved-malformed`); the tests check every record R writes
# against call.schema.json and rating.schema.json. Patterns are read as
# ECMA-262 reads them, as JSON Schema says: `$` is the end of the text,
# never before a final newline.

SCHEMA_KEYWORDS <- c("type", "required", "properties", "additionalProperties", "items", "$ref", "const", "enum",
                     "if", "then", "else", "oneOf", "anyOf", "allOf", "not", "minLength", "maxLength", "minimum",
                     "maximum", "exclusiveMinimum", "exclusiveMaximum", "uniqueItems", "minItems", "maxItems",
                     "pattern", "propertyNames", "maxProperties", "minProperties")
SCHEMA_WORDS <- c("$schema", "$id", "$defs", "title", "description", "$comment", "default", "examples")

contract_schema <- function(file) {
  key <- paste0("schema:", file)
  if (is.null(the[[key]])) the[[key]] <- read_contract(file.path("schema", file))
  the[[key]]
}

# A pattern as PCRE reads it with ECMA-262's `$` (only the last anchor).
ecma_pattern <- function(p) sub("\\$(?=\\)*$)", "\\\\z", p, perl = TRUE)

# The first place `v` does not pass `schema`, as text, or NULL. `file` is the
# schema file `schema` is in (for its "#/$defs/..." references).
schema_problem <- function(v, schema, file, where = "") {
  if (isTRUE(schema)) return(NULL)
  if (isFALSE(schema)) return(sprintf("%s: nothing is allowed here", where))
  at <- function(...) sprintf("%s: %s", if (nzchar(where)) where else "(top)", sprintf(...))
  t <- json_type(v)
  if (has_key(schema, "$ref")) {
    ref <- schema[["$ref"]]
    target_file <- if (startsWith(ref, "#")) file else sub("#.*$", "", ref)
    target <- contract_schema(target_file)
    frag <- if (grepl("#", ref, fixed = TRUE)) sub("^[^#]*#", "", ref) else ""
    if (nzchar(frag)) for (part in strsplit(sub("^/", "", frag), "/", fixed = TRUE)[[1L]]) target <- target[[part]]
    if (is.null(target)) return(at("reference %s names nothing", ref))
    p <- schema_problem(v, target, target_file, where)
    if (!is.null(p)) return(p)
  }
  if (has_key(schema, "type")) {
    types <- unlist(if (is_arr(schema$type)) schema$type else list(schema$type))
    if (!(t %in% types || (t == "integer" && "number" %in% types))) return(at("%s is not %s", t, paste(types, collapse = " or ")))
  }
  if (has_key(schema, "const") && !same_json(v, schema[["const"]])) return(at("not %s", lmcc::canonical_json(schema[["const"]])))
  if (has_key(schema, "enum") && !any(vapply(schema$enum, same_json, NA, b = v))) return(at("not one of %s", lmcc::canonical_json(schema$enum)))
  for (s in schema[["allOf"]]) { p <- schema_problem(v, s, file, where); if (!is.null(p)) return(p) }
  if (has_key(schema, "anyOf") && all(vapply(schema$anyOf, function(s) !is.null(schema_problem(v, s, file, where)), NA)))
    return(at("fits none of anyOf"))
  if (has_key(schema, "oneOf") && sum(vapply(schema$oneOf, function(s) is.null(schema_problem(v, s, file, where)), NA)) != 1L)
    return(at("fits not exactly one of oneOf"))
  if (has_key(schema, "not") && is.null(schema_problem(v, schema[["not"]], file, where))) return(at("fits what is not allowed"))
  if (has_key(schema, "if")) {
    branch <- if (is.null(schema_problem(v, schema[["if"]], file, where))) schema[["then"]] else schema[["else"]]
    if (!is.null(branch)) { p <- schema_problem(v, branch, file, where); if (!is.null(p)) return(p) }
  }
  if (t == "string") {
    n <- nchar(v, type = "chars")
    if (has_key(schema, "minLength") && n < num(schema$minLength)) return(at("shorter than %s", schema$minLength))
    if (has_key(schema, "maxLength") && n > num(schema$maxLength)) return(at("longer than %s", schema$maxLength))
    if (has_key(schema, "pattern") && !grepl(ecma_pattern(schema$pattern), v, perl = TRUE)) return(at("does not match %s", schema$pattern))
  }
  if (t %in% c("integer", "number")) {
    x <- num(v)
    if (has_key(schema, "minimum") && x < num(schema$minimum)) return(at("below %s", schema$minimum))
    if (has_key(schema, "maximum") && x > num(schema$maximum)) return(at("above %s", schema$maximum))
    if (has_key(schema, "exclusiveMinimum") && x <= num(schema$exclusiveMinimum)) return(at("not above %s", schema$exclusiveMinimum))
    if (has_key(schema, "exclusiveMaximum") && x >= num(schema$exclusiveMaximum)) return(at("not below %s", schema$exclusiveMaximum))
  }
  if (t == "array") {
    if (has_key(schema, "minItems") && length(v) < num(schema$minItems)) return(at("fewer than %s items", schema$minItems))
    if (has_key(schema, "maxItems") && length(v) > num(schema$maxItems)) return(at("more than %s items", schema$maxItems))
    if (isTRUE(schema$uniqueItems) && anyDuplicated(vapply(v, lmcc::canonical_json, ""))) return(at("items repeat"))
    if (has_key(schema, "items")) for (i in seq_along(v)) {
      p <- schema_problem(v[[i]], schema$items, file, sprintf("%s[%d]", where, i - 1L)); if (!is.null(p)) return(p)
    }
  }
  if (t == "object") {
    for (k in unlist(schema[["required"]])) if (!k %in% names(v)) return(at("no %s", k))
    if (has_key(schema, "maxProperties") && length(v) > num(schema$maxProperties)) return(at("more than %s keys", schema$maxProperties))
    if (has_key(schema, "minProperties") && length(v) < num(schema$minProperties)) return(at("fewer than %s keys", schema$minProperties))
    props <- schema[["properties"]] %||% list()
    for (k in names(v)) {
      if (has_key(schema, "propertyNames")) { p <- schema_problem(k, schema$propertyNames, file, paste0(where, "/", k)); if (!is.null(p)) return(p) }
      sub <- if (k %in% names(props)) props[[k]] else if (has_key(schema, "additionalProperties")) schema$additionalProperties else NULL
      if (!is.null(sub)) { p <- schema_problem(v[[k]], sub, file, paste0(where, "/", k)); if (!is.null(p)) return(p) }
    }
  }
  NULL
}

# The first fault the contract's schema finds in a value, or NULL.
schema_fault <- function(v, file) schema_problem(v, contract_schema(file), file)
