# A definition becomes an lmcc signature (contract/functions.md, "The
# signature"), and the sample input a version is rendered with
# (contract/calls.md, "Versions").

TOOL_LIST <- list(type = "array", items = list(type = "object", properties = list(
  name = list(type = "string"), description = list(anyOf = list(list(type = "string"), list(type = "null"))),
  parameters = list(anyOf = list(list(type = "object"), list(type = "null")))),
  required = list("name", "description", "parameters")))
CALL_LIST <- list(type = "array", items = list(type = "object", properties = list(
  id = list(type = "string"), name = list(type = "string"), input = list(type = "object")),
  required = list("id", "name", "input")))

instructions_of <- function(d, include_name, improved = NULL) {
  if (!is.null(improved)) return(trim_white(improved))
  if (!is.null(d$written)) return(d$written)
  head <- character(0)
  if (isTRUE(include_name) && nzchar(d$name)) head <- c(head, paste0("Function: ", d$name))
  desc <- trim_white(d$description)
  if (nzchar(desc)) head <- c(head, desc)
  top <- trim_white(paste(head, collapse = "\n\n"))
  lines <- character(0)
  guide <- function(fields, title) {
    words <- Filter(Negate(is.null), lapply(fields, field_desc))
    if (length(words)) c(title, vapply(names(words), function(n) sprintf("- %s: %s", n, words[[n]]), ""), "")
  }
  lines <- c(guide(d$inputs, "Parameter guidance:"), guide(d$outputs, "Output guidance:"))
  guidance <- trim_white(paste(lines, collapse = "\n"))
  if (!nzchar(guidance)) return(top)
  paste0(top, if (nzchar(top)) "\n\n", guidance)
}

fields_of <- function(d, cot, tools) {
  out <- list()
  for (n in names(d$inputs)) {
    f <- list(name = n, direction = "input", shape = d$inputs[[n]]$shape, purpose = "plain")
    if (!is.null(field_desc(d$inputs[[n]]))) f$desc <- field_desc(d$inputs[[n]])
    out[[length(out) + 1L]] <- f
  }
  if (tools) out[[length(out) + 1L]] <- list(name = "tools", direction = "input", shape = TOOL_LIST, purpose = "tools", type = "list[Tool]")
  if (cot && !"reasoning" %in% c(names(d$inputs), names(d$outputs)))
    out[[length(out) + 1L]] <- list(name = "reasoning", direction = "output", shape = list(type = "string"), purpose = "reasoning")
  if (tools) out[[length(out) + 1L]] <- list(name = "calls", direction = "output", shape = CALL_LIST, purpose = "tools.calls", type = "list[ToolCall]")
  for (n in names(d$outputs)) out[[length(out) + 1L]] <- list(name = n, direction = "output", shape = d$outputs[[n]]$shape, purpose = "plain")
  out
}

sample_value <- function(shape) {
  if (!is.null(shape$enum)) return(shape$enum[[1L]])
  if (!is.null(shape$anyOf)) {
    opts <- Filter(function(s) !identical(s$type, "null"), shape$anyOf)
    return(if (length(opts)) sample_value(opts[[1L]]) else NULL)
  }
  switch(shape$type %||% "string", string = "example text", integer = 3L, number = 2.5, boolean = TRUE,
         array = list(), object = lmcc::jobj(), null = NULL, "example text")
}

sample_inputs <- function(sig) {
  out <- list()
  for (f in lmcc::signature_to_list(sig)$fields)
    if (f$direction == "input" && (f$purpose %||% "plain") == "plain") out[f$name] <- list(sample_value(f$shape))
  out
}

# A call's program.signature: lmcc's fingerprint with every type name empty.
signature_id <- function(sig) {
  lmcc::sha256_of(lapply(lmcc::signature_to_list(sig)$fields, function(f)
    list(direction = f$direction, name = f$name, purpose = f$purpose %||% "plain", shape = f$shape, type = "")))
}

# Values as their fields expect them: a non-text value given to a text input
# is written as text (objects and lists as JSON indented by two spaces).
prepare_inputs <- function(sig, values) {
  out <- list()
  for (f in lmcc::signature_to_list(sig)$fields) {
    if (f$direction != "input" || !f$name %in% names(values)) next
    v <- values[[f$name]]
    if (identical(f$shape$type, "string") && is.null(f$shape$enum) && !is.null(v) && !(is.character(v) && length(v) == 1L)) {
      v <- if (is.list(v)) json_indented(v) else format(v)
    }
    out[f$name] <- list(v)
  }
  out
}

# JSON indented by two spaces, as Python's json.dumps(indent=2) writes it.
json_indented <- function(v, depth = 0L) {
  pad <- strrep("  ", depth + 1L); end <- strrep("  ", depth)
  if (is.list(v) && !is.null(names(v))) {
    if (!length(v)) return("{}")
    parts <- vapply(names(v), function(k) paste0(pad, lmcc::json_text(k), ": ", json_indented(v[[k]], depth + 1L)), "")
    return(paste0("{\n", paste(parts, collapse = ",\n"), "\n", end, "}"))
  }
  if (is.list(v)) {
    if (!length(v)) return("[]")
    parts <- vapply(v, function(x) paste0(pad, json_indented(x, depth + 1L)), "")
    return(paste0("[\n", paste(parts, collapse = ",\n"), "\n", end, "]"))
  }
  lmcc::json_text(v)
}
