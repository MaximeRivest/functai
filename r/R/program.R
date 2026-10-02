# Programs (contract/programs.md): your own R code around AI functions,
# called and logged as one call, with an interface it declares the way an AI
# function does (a formula and a codebook), checked on every call: inputs
# bound and checked before the code runs, outputs checked when it returns.

#' Your own code around AI functions, as one program
#'
#' A program is an R function that calls AI functions (and anything else):
#' declared like an AI function, with a formula (what comes out `~` what goes
#' in), a sentence, and a codebook of its fields, then the code. Calling it
#' is one call: its inputs are bound to their types and checked before the
#' code runs (`interface-input`), its answer checked when the code returns
#' (`interface-output`), and the AI functions it calls are steps of it, in
#' the call log (one tree), in a stream, and in a conversation.
#'
#' Like an AI function, a program is vectorised: given columns, it runs once
#' per row (one call each, in order: your code is R's to run, one row at a
#' time; the AI functions it calls with a column still run that column at
#' once). Its code is given one row's values, by name, as R values of their
#' types (a choice is a factor, a record a one-row tibble). An input the
#' caller leaves out with no default given in the codebook is left out: your
#' code's own default applies. An `opaque()` input takes any R value, passed
#' whole to every row.
#'
#' With several outputs, the code returns them by name (a named list, or a
#' one-row tibble), and the program returns a tibble, as an AI function does.
#' @param .formula `outputs ~ inputs`, as in [ai()].
#' @param .description What the program does.
#' @param .body The code: a function of the inputs, by name.
#' @param ... The fields, as in [ai()] ([opaque()] for any R value); and
#'   settings, as dotted names (`.log_content`, `.observers`, `.plugins`).
#' @param .data,.name,.defined_in As in [ai()].
#' @param .answer_from The AI function whose answer is the program's, shown
#'   as the program's answer as it is written (a served program's callers
#'   see it stream).
#' @return A program (class `functai_program`), a function.
#' @examples
#' team <- ai(team ~ message, "Which team should answer?", team = choice("shipping", "billing"))
#' answer <- ai(reply ~ message + team, "Answer the customer, as that team.")
#' support <- ai_program(reply ~ message, "Answer a customer's message.", function(message) {
#'   answer(message, team(message))
#' })
#' support
#' @export
ai_program <- function(.formula, .description = "", .body, ..., .data = NULL, .name = NULL, .defined_in = NULL, .answer_from = NULL) {
  here <- rlang::current_env()
  if (missing(.body) || !is.function(.body)) cli::cli_abort(c("{.fn ai_program} takes the program's code: a function of its inputs",
    i = "{.code ai_program(reply ~ message, \"Answer.\", function(message) answer(message))}"))
  sides <- formula_sides(.formula, .data, call = here)
  dots <- list(...)
  nms <- names(dots) %||% rep("", length(dots))
  if (any(!nzchar(nms))) cli::cli_abort("every field is named: {.code message = \"the customer's own words\"}")
  settings <- dots[startsWith(nms, ".")]
  names(settings) <- substring(names(settings), 2L)
  settings <- check_settings(settings)
  specs <- dots[!startsWith(nms, ".")]
  stray <- setdiff(names(specs), c(sides$inputs, sides$outputs))
  if (length(stray)) cli::cli_abort("{.field {stray}} {?is/are} not in the formula {.code {format_formula(sides)}}")
  field <- function(n, input) field_of(n, specs[[n]], .data, input, call = here)
  inputs <- stats::setNames(lapply(sides$inputs, field, input = TRUE), sides$inputs)
  outputs <- stats::setNames(lapply(sides$outputs, field, input = FALSE), sides$outputs)
  single <- length(outputs) == 1L
  name <- .name %||% if (single) sides$outputs else
    cli::cli_abort(c("a program with several outputs needs a name", i = "{.code ai_program({format_formula(sides)}, ..., .name = \"triage\")}"))
  columns <- stats::setNames(sides$outputs, if (single) "result" else sides$outputs)
  if (single) names(outputs) <- "result"
  args <- setdiff(names(formals(.body)), "...")
  missing_args <- setdiff(sides$inputs, args)
  if (length(missing_args) && !"..." %in% names(formals(.body)))
    cli::cli_abort("the program's code takes its inputs by name: it has no argument {.field {missing_args}}")
  core <- list(kind = "module", definition = list(name = name, description = .description, inputs = inputs, outputs = outputs),
               own = settings, single = single, columns = columns, body = .body, module = .defined_in %||% "__main__",
               file = if (is.null(.defined_in)) top_level_file() else NULL, answer_from = .answer_from)
  check_program(core, call = here)
  make_program(core)
}

#' Any R value
#'
#' A program's input or output that takes any R value, a data frame, a model
#' or a connection, unchecked. Its values are logged as JSON when they have a
#' JSON form, and described otherwise (`{"$type", "$repr"}`). A program with
#' one cannot be served or conversed with: only data crosses those.
#' @return A field.
#' @export
opaque <- function() { f <- new_field(lmcc::jobj(), "opaque"); f$opaque <- TRUE; f }

#' @rdname ai_program
#' @param interface A program's interface, as data ([ai_interface()]'s form).
#' @param name The program's name.
#' @export
program_from_interface <- function(interface, .body, name = "program", .defined_in = NULL, ai = FALSE) {
  core <- list(kind = "module", interface = json_normal(interface), own = list(), body = .body, module = .defined_in %||% "__main__")
  problem <- interface_problem(core$interface, ai = ai)
  if (!is.null(problem)) refuse("interface-malformed", c("{name} cannot be defined: {interface_fault(core$interface, problem$field)}"), field = problem$field)
  fields <- function(side) stats::setNames(lapply(core$interface[[side]], function(f) {
    out <- if (isTRUE(f$opaque)) opaque() else field_from_shape(f$shape, f$desc)
    out$shape <- f$shape; out$optional <- isTRUE(f$optional); out$opaque <- isTRUE(f$opaque)
    out
  }), vapply(core$interface[[side]], function(f) f$name, ""))
  core$definition <- list(name = name, description = core$interface$description, inputs = fields("inputs"), outputs = fields("outputs"))
  core$single <- length(core$definition$outputs) == 1L
  core$columns <- stats::setNames(names(core$definition$outputs), names(core$definition$outputs))
  make_program(core)
}

check_program <- function(core, call = NULL) {
  iface <- program_interface(core)
  problem <- interface_problem(iface, ai = FALSE)
  if (!is.null(problem)) refuse("interface-malformed", c("{.fn {core$definition$name}} cannot be defined: {interface_fault(iface, problem$field)}",
    i = "a program's fields use only the types every language checks: see {.fn ai_interface}"), field = problem$field, call = call)
  invisible(core)
}

make_program <- function(core) {
  env <- new.env(parent = asNamespace("functai"))
  env$.core <- core
  env$.inputs <- names(core$definition$inputs)
  args <- lapply(core$definition$inputs, function(f) if (isTRUE(f$optional) && has_key(f$shape, "default")) default_value(f)
                 else if (isTRUE(f$optional)) NULL else rlang::missing_arg())
  names(args) <- env$.inputs
  args <- c(args, alist(... = ))
  f <- rlang::new_function(args, quote(call_program(.core, given_inputs(environment(), .inputs), list(...))), env)
  structure(f, class = c("functai_program", "function"))
}

program_core <- function(fn) {
  if (!inherits(fn, "functai_program")) cli::cli_abort("{.arg fn} is a program from {.fn ai_program}")
  environment(fn)$.core
}

# The program's interface as data: a declared one, else its fields'.
program_interface <- function(core) {
  if (!is.null(core$interface)) return(core$interface)
  d <- core$definition
  field <- function(n, f) {
    out <- list(name = n, shape = if (isTRUE(f$opaque)) lmcc::jobj() else f$shape)
    desc <- field_desc(f)
    if (!is.null(desc) && nzchar(desc)) out$desc <- desc
    if (!isTRUE(f$opaque)) out$type <- r_type_of(f)
    if (isTRUE(f$opaque)) out$opaque <- TRUE
    if (isTRUE(f$optional)) out$optional <- TRUE
    out
  }
  json_normal(list(description = d$description, inputs = unname(Map(field, names(d$inputs), d$inputs)),
                   outputs = unname(Map(field, names(d$outputs), d$outputs))))
}

# The AI functions and programs a program's code names, by key: what its
# version follows (a plain R function it calls is followed by its code only
# when it is the program's own body).
program_parts <- function(core) {
  if (is.null(core$body)) return(list())
  env <- environment(core$body) %||% globalenv()
  out <- list()
  for (n in unique(all.names(body(core$body)))) {
    v <- tryCatch(get(n, envir = env), error = function(e) NULL)
    if (inherits(v, "functai_fn")) out[[paste0(core_of(v)$module, ":", n)]] <- ai_version(v)
    else if (inherits(v, "functai_program") && !identical(program_core(v)$body, core$body)) out[[paste0(program_core(v)$module, ":", n)]] <- program_version(program_core(v))
  }
  if (!length(out)) return(out)
  out[order(names(out), method = "radix")]
}

code_hash <- function(fn) lmcc::sha256_of(paste(deparse(utils::removeSource(fn), width.cutoff = 500L), collapse = "\n"))

# A program's version (calls.md, *Versions*): its code, the AI functions and
# programs it names, its interface without defaults, and its defaults.
program_version <- function(core) {
  iface <- program_interface(core)
  plain <- iface
  for (side in c("inputs", "outputs")) plain[[side]] <- lapply(plain[[side]], function(f) { f$shape <- no_defaults(f$shape); f })
  code <- if (is.null(core$body)) lmcc::jobj() else stats::setNames(list(code_hash(core$body)), paste0(core$module, ":", core$definition$name))
  parts <- program_parts(core)
  doc <- list(code = code, ai = if (length(parts)) parts else lmcc::jobj(), interface = plain)
  defaults <- defaults_document(core)
  if (!is.null(defaults)) doc$defaults <- defaults
  lmcc::sha256_of(doc)
}

program_json <- function(core) {
  iface <- program_interface(core)
  if (!is.null(core$remote))
    return(list(name = core$definition$name, kind = "remote", module = core$module, version = core$remote$version,
                interface = interface_signature(iface), answer = iface$outputs[[length(iface$outputs)]]$name, remote = core$remote$url))
  p <- list(name = core$definition$name, kind = "module", module = core$module, version = program_version(core),
            interface = interface_signature(iface), answer = iface$outputs[[length(iface$outputs)]]$name)
  if (!is.null(core$file)) p$file <- core$file
  p
}

# ---------------------------------------------------------------- values

# Whether an R value has a JSON form (calls.md, *Values*); a data frame, an
# environment, a function, a model have none, and are described.
has_json_form <- function(v) {
  if (is.null(v)) return(TRUE)
  if (is.data.frame(v) || is.environment(v) || is.function(v) || isS4(v) || typeof(v) %in% c("externalptr", "symbol", "language")) return(FALSE)
  if (is.factor(v)) return(TRUE)
  if (is.object(v) && !inherits(v, c("lmcc_int", "json_object", "json_array"))) return(FALSE)
  if (is.list(v)) return(all(vapply(v, has_json_form, NA)))
  is.atomic(v)
}

value_json <- function(v) if (has_json_form(v)) plain_json(v) else describe_value(v)

# The text a value with no JSON form defines for itself (a date, a table), or
# NULL when its only text is R's default for any object.
own_text <- function(v) {
  if (is.data.frame(v)) return(paste(utils::capture.output(print(v)), collapse = "\n"))
  if (inherits(v, c("Date", "POSIXt"))) return(format(v))
  cls <- class(v)
  if (is.object(v) && any(vapply(cls, function(k) !is.null(utils::getS3method("format", k, optional = TRUE)), NA)))
    return(paste(format(v), collapse = "\n"))
  NULL
}

# One row's inputs, bound and checked (programs.md): a JSON row, or a misfit.
program_row <- function(core, given, i, dropped) {
  fields <- core$definition$inputs
  row <- list(); described <- character(0); code <- list()
  if (!is.null(given$.unknown)) return(misfit_row(row, given$.unknown, sprintf("%s: not one of its inputs (%s)", given$.unknown, paste(names(fields), collapse = ", "))))
  for (k in names(fields)) {
    f <- fields[[k]]
    if (!k %in% names(given)) {
      if (!isTRUE(f$optional)) return(misfit_row(row, k, sprintf("%s: no value, and it is required", k)))
      if (has_key(f$shape, "default")) { row[k] <- list(f$shape[["default"]]); code[k] <- list(default_value(f)) }
      next
    }
    v <- given[[k]]
    if (isTRUE(f$opaque)) {
      code[k] <- list(v)
      if (has_json_form(v)) row[k] <- list(plain_json(v)) else { row[k] <- list(describe_value(v)); described <- c(described, k) }
      next
    }
    v <- element_of(v, i)
    if (is_missing(v) && !takes_null(f)) {
      if (isTRUE(f$optional)) {
        if (has_key(f$shape, "default")) { row[k] <- list(f$shape[["default"]]); code[k] <- list(default_value(f)) }
        next
      }
      return(misfit_row(row, k, sprintf("%s: a missing value (NA), and its type takes no null", k)))
    }
    shape <- data_shape(f$shape)
    json <- if (has_json_form(v)) to_json(f, v) else {
      text <- if (is_text_shape(shape)) own_text(v) else NULL
      if (is.null(text)) {
        d <- describe_value(v)
        why <- if (k %in% dropped || "*" %in% dropped) sprintf("%s: its value does not fit %s (the value is not shown: the log drops it)", k, type_label(f))
          else sprintf("%s: %s has no JSON form, and %s wants %s", k, short_text(d[["$repr"]]), k, type_label(f))
        return(misfit_row(row, k, why))
      }
      text
    }
    json <- bind_json(json, shape)
    why <- checker(f)(json, k)
    if (!is.null(why)) {
      if (k %in% dropped || "*" %in% dropped) why <- sprintf("%s: its value does not fit %s (the value is not shown: the log drops it)", k, type_label(f))
      return(misfit_row(row, k, why))
    }
    row[k] <- list(json)
    code[k] <- list(r_value(f, json))
  }
  structure(if (length(row)) row else lmcc::jobj(), code = code, described = described)
}

misfit_row <- function(row, field, message) structure(list(inputs = if (length(row)) row else lmcc::jobj(), field = field, message = message), class = "functai_misfit")

short_text <- function(x) { x <- as.character(x); if (nchar(x) > 80L) paste0(substr(x, 1L, 80L), "\u2026") else x }

# One value of a column (an opaque input is never split).
element_of <- function(v, i) {
  if (is.null(v) || !has_json_form(v) && !is.factor(v) && !inherits(v, c("Date", "POSIXt"))) return(v)
  if (vctrs::vec_size(v) == 1L) element(v, 1L) else element(v, i)
}
value_size <- function(v) if (is.null(v) || (!has_json_form(v) && !inherits(v, c("Date", "POSIXt")))) 1L else vctrs::vec_size(v)

# A bound JSON value as the R value of its field's type: one element.
r_value <- function(f, json) {
  if (f$kind %in% c("string", "enum", "integer", "number", "boolean", "record")) return(assemble(f, list(json)))
  if (identical(f$kind, "list")) return(assemble(f, list(json))[[1L]])
  json
}

# What the code returned, as the program's outputs (by name), checked
# (programs.md, *Outputs*): NULL when they fit, else the refusal's field and why.
program_outputs <- function(core, value) {
  outs <- core$definition$outputs
  names_ <- names(outs)
  if (length(outs) == 1L) {
    f <- outs[[1L]]
    json <- if (isTRUE(f$opaque)) value_json(value) else if (has_json_form(value)) to_json(f, value) else describe_value(value)
    return(list(outputs = stats::setNames(list(json), names_), fault = output_fault(f, names_[[1L]], json, value)))
  }
  if (is.data.frame(value) && nrow(value) == 1L) value <- lapply(value, function(col) element(col, 1L))
  if (!is.list(value) || is.null(names(value)) || is.data.frame(value))
    return(list(fault = list(field = names_[[1L]], why = sprintf("%s returns its outputs by name (a named list): %s", core$definition$name, paste(names_, collapse = ", ")))))
  extra <- setdiff(names(value), names_)
  if (length(extra)) return(list(fault = list(field = sort_code_points(extra)[[1L]], why = sprintf("%s is not one of its outputs", sort_code_points(extra)[[1L]]))))
  out <- list()
  for (k in names_) {
    if (!k %in% names(value)) return(list(fault = list(field = k, why = sprintf("no %s", k))))
    f <- outs[[k]]
    v <- value[[k]]
    json <- if (isTRUE(f$opaque)) value_json(v) else if (has_json_form(v)) to_json(f, v) else describe_value(v)
    fault <- output_fault(f, k, json, v)
    if (!is.null(fault)) return(list(fault = fault))
    out[k] <- list(json)
  }
  list(outputs = out, fault = NULL)
}

sort_code_points <- function(x) x[order(x, method = "radix")]

output_fault <- function(f, name, json, value) {
  if (isTRUE(f$opaque)) return(NULL)
  if (!has_json_form(value)) return(list(field = name, why = sprintf("%s: a value with no JSON form", name)))
  shape <- data_shape(f$shape)
  if (!well_formed(shape, shape, carry = TRUE) || loops(shape)) return(NULL)
  why <- if (has_infinity(json)) sprintf("%s: an infinite number has no JSON form", name) else shape_fault(json, shape, shape, name)
  if (is.null(why)) NULL else list(field = name, why = why)
}

# ---------------------------------------------------------------- calling

call_program <- function(core, inputs, extra = list()) {
  fields <- core$definition$inputs
  if (length(extra) && (is.null(names(extra)) || any(!nzchar(names(extra)))))
    cli::cli_abort("{core$definition$name} takes {length(fields)} input{?s} ({.field {names(fields)}}), by name or in order")
  given <- inputs[intersect(names(fields), names(inputs))]
  if (length(extra)) given$.unknown <- sort_code_points(names(extra))[[1L]]
  sizes <- vapply(setdiff(names(given), ".unknown"), function(k) if (isTRUE(fields[[k]]$opaque)) 1L else value_size(given[[k]]), 1L)
  n <- if (length(sizes)) max(sizes) else 1L
  if (any(sizes != 1L & sizes != n)) cli::cli_abort("inputs of different lengths: {.field {names(sizes)}} have {sizes} values")
  results <- lapply(seq_len(n), function(i) run_program_row(core, given, i))
  report_errors(core, results, effective(core$own)$on_error %||% "warn")
  program_answers(core, results)
}

# One row as a call: bound, checked, run, checked again.
run_program_row <- function(core, given, i, own_extra = list()) {
  s <- effective(set_all(core$own, own_extra))
  iface <- program_interface(core)
  fields <- list(inputs = names(core$definition$inputs), outputs = names(core$definition$outputs), added = character(0))
  keep <- content_kept(fields, content_layers_program(core))
  dropped <- names(keep)[!keep]
  row <- program_row(core, given, i, dropped)
  misfit <- inherits(row, "functai_misfit")
  program <- function() program_json(core)
  call <- start_call(program, s, if (misfit) row$inputs else row, fields, keep, own = core$own, core = core)
  call$described <- if (misfit) NULL else attr(row, "described")
  if (!core$single) call$keep_events$holds <- names(core$definition$outputs)
  call$tree$keeps[[call$id]] <- call$keep_events
  call$core <- core
  out <- tryCatch(list(value = run_call(call, function(call) {
    if (misfit) stop(rlang::error_cnd(c("functai_interface_input", "functai_refusal"), code = "interface-input", field = row$field,
                                      message = sprintf("%s: input %s", core$definition$name, row$message)))
    value <- do.call(core$body, attr(row, "code"))
    checked <- program_outputs(core, value)
    if (!is.null(checked$fault)) stop(rlang::error_cnd(c("functai_interface_output", "functai_refusal"), code = "interface-output",
      field = checked$fault$field, message = sprintf("%s: output %s", core$definition$name, checked$fault$why)))
    call$outputs <- checked$outputs
    call$has_done_value <- TRUE
    call$done_value <- if (core$single) checked$outputs[[1L]] else checked$outputs
    value
  })), error = identity)
  if (inherits(out, "functai_turn_waiting")) stop(out)
  if (inherits(out, "condition")) return(list(error = out, call = call$id))
  list(outputs = call$outputs, value = out$value, call = call$id)
}

content_layers_program <- function(core) {
  blocks <- rev(the$content_blocks %||% list())
  c(list(core$own$log_content), blocks, list(the$config$log_content))
}

# The answers of the rows: one output, a vector of its type; several, a tibble.
program_answers <- function(core, results) {
  outs <- core$definition$outputs
  col <- function(k) {
    f <- outs[[k]]
    if (isTRUE(f$opaque)) return(lapply(results, function(r) if (is.null(r$error)) (if (core$single) r$value else r$value[[k]]) else NULL))
    assemble(f, lapply(results, function(r) r$outputs[[k]]))
  }
  if (core$single) { v <- col(names(outs)); return(if (isTRUE(outs[[1L]]$opaque) && length(v) == 1L) v[[1L]] else v) }
  cols <- lapply(stats::setNames(nm = names(outs)), col)
  tibble::new_tibble(cols, nrow = length(results))
}

#' @export
print.functai_program <- function(x, ...) {
  core <- program_core(x)
  d <- core$definition
  cols <- core$columns
  formula <- paste(paste(cols, collapse = " + "), "~", paste(names(d$inputs), collapse = " + "))
  named_after <- core$single && identical(d$name, unname(cols[[1L]]))
  cat("<ai program> ", if (!named_after) paste0(d$name, ": "), formula, "\n", sep = "")
  if (nzchar(trim_white(d$description))) cat("  ", trim_white(d$description), "\n", sep = "")
  parts <- names(program_parts(core))
  if (length(parts)) cat("  calls: ", paste(sub("^.*:", "", parts), collapse = ", "), "\n", sep = "")
  invisible(x)
}

#' @export
ai_interface.functai_program <- function(x, ...) structure(program_interface(program_core(x)), class = "functai_interface")
