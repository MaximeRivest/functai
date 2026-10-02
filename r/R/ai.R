# AI functions: a vectorised R function whose body a model writes.

#' Write a function a language model does the work of
#'
#' Write it the way you write a model: a formula says what comes out and
#' what goes in (`team ~ message`), a sentence says what the function does,
#' and a model writes the body. The result is an ordinary, vectorised R
#' function, `team(message)`: call it on one value or on whole columns
#' inside [dplyr::mutate()]. Each row is one model call; up to `concurrency`
#' (default 8) run at once.
#'
#' **The formula.** Outputs on the left, inputs on the right, each a name,
#' joined by `+`: `team ~ message`, `summary + urgent ~ message`, `decision
#' ~ message + price + final_sale`. With `.data`, `.` on the right is every
#' other column, as in [stats::lm()]. One output: the function returns a
#' vector and is named after it (`team`). Several: it returns a tibble, one
#' column each, which [dplyr::mutate()] splices in; name it with `.name`.
#'
#' **The fields** (`...`, named like the formula's), are a codebook:
#' * nothing: text, or the type of the column of the same name in `.data`;
#' * a sentence: the same, with words the model reads about the field
#'   (`message = "the customer's own words"`);
#' * a type: [choice()] (one of a set of answers), `integer()`, `double()`,
#'   `logical()`, `character()`, [record()], [vctrs::list_of()],
#'   [optional()], [json_shape()]; [described()] adds words to a type;
#' * [defaults_to()]: an input the caller may leave out, and the value it
#'   is sent with then (`tone = defaults_to("kind")`), as an R function's
#'   default argument.
#'
#' **What it takes and gives** is its interface ([ai_interface()]), checked
#' when the function is defined, as every language checks one: field names
#' are ASCII identifiers (`order_id`, not `order.id`), and a default fits
#' its type. A function that breaks these rules is refused then
#' (`interface-malformed`), not later by whatever reads it.
#'
#' **Values are bound on every call** (contract/programs.md, "Binding a
#' call's inputs"), as every language binds them: a value is converted to
#' its field's type when the meaning is clear, and refused otherwise. Text
#' takes a number (`42` is `"42"`), `TRUE`/`FALSE` (`"true"`, `"false"`), a
#' date (its text), a list or a one-row tibble (JSON indented by two
#' spaces); a whole number takes `5` or `"5"`, not `2.5`; a number takes
#' `"2.5"`; `TRUE`/`FALSE` take nothing else. A record keeps only the members
#' it names: a tibble with more columns than the [record()] names (an id, an
#' e-mail address) sends the record's columns, and the others are dropped. A
#' value that does not bind fails that row before any request
#' (`interface-input`, a `functai_interface_input` condition naming the
#' `field`, its message quoting the value unless the log drops that field),
#' and its call record says so; a reply whose value does not fit is
#' unreadable, and the model is asked again. The call record holds the
#' bound values. An input left out is sent with its default exactly as the
#' interface holds it; a default written as a formula (`defaults_to(~
#' Sys.Date())`) is computed at each call that leaves it out.
#'
#' **Missing values.** `NA` (or `NULL`) is JSON's null. An input whose type
#' takes null (an [optional()] type, a [json_shape()] that allows null) is
#' sent null. An optional input whose type takes no null is left out: it is
#' sent with its default, as a table's gap takes the default. A required one
#' fails that row before any request (`interface-input`): its answer is `NA`,
#' and `predict()`'s `.error` says why. Inside a record (a row of a tibble
#' column, or a named list), `NA` is null too, except for a member the record
#' does not require and whose type takes no null: a tibble cannot leave a
#' member out, so `NA` there leaves it out, as an answer that leaves it out
#' comes back `NA`.
#'
#' **What the call log keeps** is `.log_content` (see [ai_config()]):
#' `TRUE`, `FALSE`, `c(transcript = FALSE)`, or the names of the fields it
#' may keep, `c("question")`. It only removes: a block around the call, or
#' `ai_config()`, may drop more; nothing brings back what one of them drops.
#' A name that is not one of the function's fields is refused
#' (`log-content-field`). A one-output function's answer is `result` in the
#' log, as in every language; `log_content` may call it by the formula's
#' name too (`c(team = FALSE)`).
#'
#' @param .formula `outputs ~ inputs`, as names joined by `+`.
#' @param .description What the function does, in a sentence or a
#'   paragraph: the model's instructions.
#' @param ... The fields, as above; and settings, as dotted names
#'   (`.lm = "claude-haiku-4-5"`, `.temperature = 0`, `.adapter = "json"`,
#'   `.module = "cot"`, ...: see [ai_config()]).
#' @param .data A data frame whose columns give the fields their types (a
#'   factor becomes a choice of its levels). Only its column types are read,
#'   never its rows.
#' @param .name The function's name. The model reads it (`"Function:
#'   team"`), and the call log files calls under it. Default: the output's
#'   name, when there is one.
#' @param .tools Functions the model may call, from [ai_tool()].
#' @param .demos Worked examples: a data frame with input and output columns.
#' @param .instructions An improved instruction that replaces the written one.
#' @param .defined_in The code module the call log files calls under
#'   (default `"__main__"`, as a Python notebook's).
#' @return An AI function (class `functai_fn`).
#' @examples
#' mood <- ai(mood ~ review, "How does the customer feel about what they bought?",
#'   mood = choice("happy", "unhappy", "mixed"))
#' mood
#'
#' triage <- ai(summary + urgent ~ message, "Read the support ticket.",
#'   message = "the customer's own words",
#'   summary = "one sentence, no names",
#'   urgent = logical(),
#'   .name = "triage")
#'
#' # types from a table's columns: `decision` is a factor, `price` a number
#' refund <- ai(decision ~ message + price + final_sale, "Should the shop refund this request?",
#'   .data = refunds)
#' \dontrun{
#' mood("It broke after one day.")
#' tickets |> dplyr::mutate(triage(message))
#' }
#'
#' # an input the caller may leave out
#' reply <- ai(reply ~ message + tone, "Answer the customer.",
#'   tone = defaults_to("kind", choice("kind", "brief")))
#' args(reply)
#' @export
ai <- function(.formula, .description = "", ..., .data = NULL, .name = NULL, .tools = NULL, .demos = NULL,
               .instructions = NULL, .defined_in = NULL) {
  sides <- formula_sides(.formula, .data, call = rlang::current_env())
  if (!is.character(.description) || length(.description) != 1L)
    cli::cli_abort("{.arg .description} is one string: what the function does")
  dots <- list(...)
  nms <- names(dots) %||% rep("", length(dots))
  if (any(!nzchar(nms))) cli::cli_abort(c("every field is named: {.code message = \"the customer's own words\"}",
    i = "the formula comes first, then the description: {.code ai(team ~ message, \"Which team should answer?\")}"))
  settings <- dots[startsWith(nms, ".")]
  names(settings) <- substring(names(settings), 2L)
  settings <- check_settings(settings)
  specs <- dots[!startsWith(nms, ".")]
  stray <- setdiff(names(specs), c(sides$inputs, sides$outputs))
  if (length(stray)) cli::cli_abort(c("{.field {stray}} {?is/are} not in the formula {.code {format_formula(sides)}}",
    i = "fields are named like the formula's outputs and inputs; settings start with a dot ({.code .lm})"))
  if (!is.null(.data) && !is.data.frame(.data)) cli::cli_abort("{.arg .data} is a data frame, whose columns give the fields their types")
  here <- rlang::current_env()
  field <- function(n, input) field_of(n, specs[[n]], .data, input, call = here)
  inputs <- stats::setNames(lapply(sides$inputs, field, input = TRUE), sides$inputs)
  outputs <- stats::setNames(lapply(sides$outputs, field, input = FALSE), sides$outputs)
  single <- length(outputs) == 1L
  name <- .name %||% if (single) sides$outputs else
    cli::cli_abort(c("a function with several outputs needs a name", i = "{.code ai({format_formula(sides)}, ..., .name = \"triage\")}"))
  if (!rlang::is_string(name) || !nzchar(name)) cli::cli_abort("{.arg .name} is the function's name, one string")
  # One output is `result` in the definition, as Python's and TypeScript's
  # functions name it, so the same function has the same version in every
  # language; the formula's name is the column it is read from and written to.
  columns <- stats::setNames(sides$outputs, if (single) "result" else sides$outputs)
  if (single) names(outputs) <- "result"
  core <- list(
    definition = list(name = name, description = .description, inputs = inputs, outputs = outputs),
    own = settings, tools = .tools %||% list(), single = single, columns = columns,
    module = .defined_in %||% "__main__", state = list(instructions = .instructions, demos = list()), saved = NULL,
    file = if (is.null(.defined_in)) top_level_file() else NULL)
  core <- own_settings(core)
  check_definition(core, call = here)          # an output given a default is refused there, in its turn
  fn <- make_fn(core)
  if (!is.null(.demos)) fn <- with_demos(fn, .demos)
  fn
}

# The names on each side of `outputs ~ inputs` (`.` on the right: every other
# column of `data`).
formula_sides <- function(f, data, call) {
  if (!rlang::is_formula(f, lhs = TRUE))
    cli::cli_abort(c("{.fn ai} starts with a formula, outputs on the left and inputs on the right",
      i = "{.code ai(team ~ message, \"Which team should answer this message?\")}"), call = call)
  outputs <- formula_names(rlang::f_lhs(f), "left", call)
  rhs <- rlang::f_rhs(f)
  inputs <- if (identical(rhs, quote(.))) {
    if (is.null(data)) cli::cli_abort("{.code ~ .} means every other column of {.arg .data}: give {.arg .data}", call = call)
    setdiff(names(data), outputs)
  } else formula_names(rhs, "right", call)
  both <- intersect(inputs, outputs)
  if (length(both)) cli::cli_abort("{.field {both}} is on both sides of the formula", call = call)
  if (!length(inputs)) cli::cli_abort("the right side of the formula names the inputs", call = call)
  list(outputs = outputs, inputs = inputs)
}

formula_names <- function(x, side, call) {
  if (is.call(x) && identical(x[[1L]], quote(`+`)) && length(x) == 3L)
    return(c(formula_names(x[[2L]], side, call), formula_names(x[[3L]], side, call)))
  if (is.symbol(x) && !identical(x, quote(.))) return(as.character(x))
  term <- rlang::expr_deparse(x)
  names_in <- paste(all.vars(x), collapse = " + ")
  why <- if (is.call(x) && as.character(x[[1L]])[[1L]] %in% c("*", ":", "^"))
    c(x = "{.code {term}} asks a regression for an interaction; an AI function reads all its inputs together",
      i = "write {.code {names_in}}")
  else if ((is.call(x) && identical(x[[1L]], quote(`-`))) || is.numeric(x))
    c(x = "{.code {term}} is about a regression's intercept; an AI function has no equation, so no intercept")
  else if (is.call(x) && nzchar(names_in))
    c(x = "{.code {term}} transforms a column for a regression; an AI function reads the column itself",
      i = "write {.code {names_in}}, or make the column you want with {.fn dplyr::mutate} first")
  else c(x = "{.code {term}} is not a name", i = "{.code summary + urgent ~ message + channel}")
  cli::cli_abort(c("the {side} side of an AI function's formula names columns, joined by {.code +}", why), call = call)
}

format_formula <- function(sides) paste(paste(sides$outputs, collapse = " + "), "~", paste(sides$inputs, collapse = " + "))

# One field from its codebook entry: nothing (the column's type in `data`,
# else text), a sentence (the same, described), or a type.
field_of <- function(name, spec, data, input, call) {
  if (!is.null(spec) && !rlang::is_string(spec)) return(as_field(spec))
  from_data <- !is.null(data) && name %in% names(data)
  if (input && !is.null(data) && !from_data)
    cli::cli_abort(c("{.field {name}} is not a column of {.arg .data}", i = "give its type ({.code {name} = character()}), or check its name"), call = call)
  f <- as_field(if (from_data) prototype_of(data[[name]]) else character())
  if (is.null(spec)) f else described(f, spec)
}

# The column each output is read from and written to, named by output. One
# output is `result` in the definition; its column is the formula's name.
columns_of <- function(core) {
  outs <- names(core$definition$outputs)
  core$columns %||% stats::setNames(outs, outs)
}

# The R function: one argument per input, in order; an input with a default
# (defaults_to()) shows it as its default argument. A call that leaves such
# an input out sends the interface's default itself, as JSON (input_rows()):
# only the arguments the caller gave are read.
make_fn <- function(core) {
  env <- new.env(parent = asNamespace("functai"))
  env$.core <- core
  env$.inputs <- names(core$definition$inputs)
  args <- lapply(core$definition$inputs, function(f) if (isTRUE(f$optional)) default_value(f) else rlang::missing_arg())
  names(args) <- env$.inputs
  f <- rlang::new_function(args, quote(call_ai(.core, given_inputs(environment(), .inputs))), env)
  structure(f, class = c("functai_fn", "function"))
}

# The arguments a call of an AI function was given, by name: an argument
# left out (its default not given) is not among them.
given_inputs <- function(env, names) {
  out <- list()
  for (n in names) if (!eval(call("missing", as.name(n)), env)) out[n] <- list(get(n, envir = env))
  out
}

core_of <- function(fn) {
  if (!inherits(fn, "functai_fn")) cli::cli_abort("{.arg fn} is an AI function from {.fn ai}")
  environment(fn)$.core
}

answer_name <- function(core) names(core$definition$outputs)[[length(core$definition$outputs)]]

# ---------------------------------------------------------------- signature, plan, version

signature_of <- function(core, settings) {
  d <- core$definition
  sig <- list(instructions = instructions_of(d, settings$include_fn_name %||% TRUE, core$state$instructions),
              fields = fields_of(d, identical(settings$module, "cot"), length(core$tools) > 0L))
  lmcc::signature_from_list(sig)
}

# Worked examples as example turns for this plan (contract/functions.md).
past_turns <- function(core, plan) {
  fields <- lmcc::signature_to_list(plan$signature)$fields
  plain <- vapply(Filter(function(f) f$direction == "input" && (f$purpose %||% "plain") == "plain", fields), function(f) f$name, "")
  kept <- vapply(Filter(function(f) f$direction == "output" && (f$purpose %||% "plain") %in% c("plain", "reasoning"), fields), function(f) f$name, "")
  out <- list()
  for (d in core$state$demos) {
    if (!is.null(d$signature) && !is.null(d$steps)) {
      if (identical(d$signature, lmcc::signature_fingerprint(plan$signature))) {
        t <- tryCatch(lmcc::load_turn(plan, d), lmcc_refusal = function(e) NULL)
        if (!is.null(t)) { out[[length(out) + 1L]] <- t; next }
      }
    }
    ins <- prepare_inputs(plan$signature, d$inputs[names(d$inputs) %in% plain])
    outs <- d$outputs[names(d$outputs) %in% kept]
    if (!length(outs)) next
    t <- tryCatch(lmcc::example_turn(plan, ins, outs), lmcc_refusal = function(e) NULL)
    if (!is.null(t)) out[[length(out) + 1L]] <- t
  }
  out
}

tool_specs <- function(core) lapply(core$tools, function(t) list(name = t$name, description = t$description, parameters = t$parameters))

# The request a function renders for these inputs (default: the sample
# inputs) under the probe facts (contract/models.json), as lmcc renders it.
probe_render <- function(core, inputs = NULL) {
  s <- effective(core$own)
  sig <- signature_of(core, s)
  plan <- bind_layout(s$adapter, s$template, sig, probe_capabilities(), "probe")
  values <- prepare_inputs(plan$signature, inputs %||% sample_inputs(plan$signature))
  if (length(core$tools)) values$tools <- tool_specs(core)
  turns <- past_turns(core, plan)
  lmcc::render(plan, values, if (length(turns)) turns else NULL)
}

probe_request <- function(core, inputs = NULL) lmcc::request_of(probe_render(core, inputs), "probe")

request_hash <- function(core, inputs = NULL) {
  tryCatch(lmcc::sha256_of(probe_request(core, inputs)), lmcc_refusal = function(e) paste0("refused:", e$code))
}

version_of <- function(core) {
  defaults <- defaults_document(core)
  lmcc::sha256_of(if (is.null(defaults)) list(request = request_hash(core)) else list(request = request_hash(core), defaults = defaults))
}

program_of <- function(core, version = NULL) {
  function() {
    s <- effective(core$own)
    p <- list(name = core$definition$name, kind = "ai", module = core$module, version = version %||% version_of(core),
              signature = signature_id(signature_of(core, s)), interface = interface_signature(interface_of(core)),
              answer = answer_name(core))
    if (!is.null(core$saved)) p$saved <- core$saved
    if (!is.null(core$file)) p$file <- core$file
    p
  }
}

# The notebook or script a function defined at the top level is in
# (calls.md, program.file): the document being knitted (R Markdown, Quarto),
# the file being source()d, or the script Rscript runs; NULL when none is
# known (a console), so its calls are not told apart by file.
top_level_file <- function() {
  input <- tryCatch(if (isTRUE(getOption("knitr.in.progress")) && requireNamespace("knitr", quietly = TRUE))
    knitr::current_input(dir = TRUE) else NULL, error = function(e) NULL)
  if (is_str(input)) return(normalizePath(input, mustWork = FALSE))
  for (frame in rev(sys.frames())) {
    f <- tryCatch(get("ofile", envir = frame, inherits = FALSE), error = function(e) NULL)
    if (is_str(f)) return(normalizePath(f, mustWork = FALSE))
  }
  arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(arg)) return(normalizePath(sub("^--file=", "", arg[[1L]]), mustWork = FALSE))
  NULL
}

route <- function(s) {
  lm <- s$lm %||% default_model()
  if (is.null(lm)) cli::cli_abort(c("no model configured, and no API key found to pick one",
    i = "set OPENAI_API_KEY (or ANTHROPIC_API_KEY, GEMINI_API_KEY, ...), or name one: {.code ai_config(lm = \"gpt-4.1-mini\")}"))
  model <- model_string(lm)
  router <- s$router %||% default_router()
  res <- resolve_model(router, model)
  list(model = model, router = router, provider = res$provider, wire = res$model)
}

# ---------------------------------------------------------------- running rows

# Run one call per row. `rows`: a list of JSON input lists (NULL: skip the
# row; a `functai_misfit`: refused before any request). Each row is a call of
# its own (a tree of its own, or a step of the call it runs inside): its
# `started`, its requests, its `done` or `failed` and its record.
run_rows <- function(core, rows, extra = list()) {
  s <- effective(set_all(core$own, extra))
  r <- route(s)
  s <- adjust_settings(s, r$provider, r$wire)
  caps <- call_capabilities(r$provider, r$wire, s)
  sig <- signature_of(core, s)
  plan <- bind_layout(s$adapter, s$template, sig, caps, r$provider)
  past <- past_turns(core, plan)
  version <- version_of(core)
  program <- program_of(core, version)
  fields <- call_fields(core, s)
  keep <- content_kept(fields, content_layers(core, extra$log_content))
  plain_outputs <- names(core$definition$outputs)
  jobs <- list(); calls <- vector("list", length(rows)); refused <- vector("list", length(rows))
  for (i in seq_along(rows)) {
    if (is.null(rows[[i]])) next
    misfit <- inherits(rows[[i]], "functai_misfit")
    call <- start_call(program, s, if (misfit) rows[[i]]$inputs else rows[[i]], fields, keep, own = core$own)
    if (!core$single) call$keep_events$holds <- plain_outputs
    call$tree$keeps[[call$id]] <- call$keep_events
    call$provider <- r$provider
    call$core <- core
    calls[[i]] <- call
    err <- call_begin(call)
    if (is.null(err) && misfit) err <- misfit_error(core, rows[[i]])
    if (is.null(err)) {
      hooked <- tryCatch(before_call_hooks(core, call, s, rows[[i]], plan, past), error = identity)
      if (inherits(hooked, "error")) err <- hooked
    }
    if (!is.null(err)) { refused[[i]] <- err; next }
    job <- tryCatch(new_job(hooked$plan %||% plan, hooked$past %||% past, rows[[i]], hooked$settings %||% s, hooked$model %||% r$model,
                            hooked$tools %||% core$tools, call, core), error = identity)
    if (inherits(job, "error")) { e <- job; job <- new.env(); job$state <- "failed"; job$error <- e; job$call <- call }
    job$escalation <- s$escalate_to
    jobs[[as.character(i)]] <- job
  }
  old <- the$current
  run_jobs(unname(jobs), r$router, s$concurrency)
  the$current <- old
  waiting <- NULL
  out <- lapply(seq_along(rows), function(i) {
    call <- calls[[i]]
    if (is.null(call)) return(list(skipped = TRUE))
    job <- jobs[[as.character(i)]]
    err <- refused[[i]]
    if (is.null(err) && identical(job$state, "waiting")) { waiting <<- waiting %||% job$waiting; call$ended <- TRUE; return(list(waiting = TRUE, call = call$id)) }
    if (is.null(err) && identical(job$state, "done")) {
      job <- escalate(core, job, s)
      call$outputs <- job$outputs
      call$probabilities <- job$probabilities
      call$confidence <- confidence_of(job$outputs, job$probabilities)
      call$done_value <- if (core$single) job$outputs[["result"]] else job$outputs[intersect(plain_outputs, names(job$outputs))]
      call$steps <- steps_of(call, job$turn)
    } else if (is.null(err)) err <- job$error
    ending <- call_end(call, err)
    if (is.null(ending)) list(outputs = job$outputs, call = call$id, model = job$model, turn = job$turn, probabilities = job$probabilities)
    else list(error = ending, call = call$id, model = if (!is.null(job)) job$model else NULL)
  })
  if (!is.null(waiting)) stop(waiting)
  out
}

# How sure the model was (contract/calls.md, `confidence`): the probability it
# gave its own answer, the lowest over the outputs it measured; NULL when it
# measured none.
answer_key <- function(v) if (is.logical(v) && length(v) == 1L) (if (isTRUE(v)) "true" else "false") else if (is.character(v) && length(v) == 1L) v else lmcc::canonical_json(v)

confidence_of <- function(outputs, probabilities) {
  ps <- numeric(0)
  for (k in names(probabilities)) {
    dist <- probabilities[[k]]
    if (!length(dist)) next
    p <- dist[[answer_key(outputs[[k]])]] %||% max(unlist(dist))
    ps <- c(ps, as.numeric(p))
  }
  if (length(ps)) min(ps) else NULL
}

# The inputs given, recycled to `n` rows (default: their common size, or one
# row when every input is left out), as one JSON list per row:
# - an input left out takes its shape's default, as the interface holds it
#   (exact JSON: absent members stay absent, and nothing is made up);
# - a missing value (`NA`, `NULL`) is JSON's null: where the input's type
#   does not take null, the row has no value to send and makes no call (NULL:
#   its answer is `NA`, as a required or an optional input's; the rows'
#   attribute `missing` names the input, by row);
# - a given value is written as JSON by its shape alone (to_json()), then
#   checked as it is: one that does not fit its type (programs.md, *Checking
#   values*) makes a row that fails with `interface-input` before any request
#   (a `functai_misfit`).
input_rows <- function(core, inputs, n = NULL) {
  fields <- core$definition$inputs
  given <- inputs[intersect(names(fields), names(inputs))]
  unset <- setdiff(required_inputs(core), names(given))
  if (length(unset)) cli::cli_abort("no value for input{?s} {.field {unset}}", call = NULL)
  given <- lapply(given, function(v) if (is.null(v)) NA else v)
  if (length(given)) given <- vctrs::vec_recycle_common(!!!given, .size = n)
  n <- n %||% if (length(given)) vctrs::vec_size_common(!!!given) else 1L
  # what each field's check needs, worked out once for all the rows
  nullable <- vapply(fields[names(given)], takes_null, NA)
  check <- lapply(fields[names(given)], checker)
  write <- lapply(fields[names(given)], json_writer)
  shapes <- lapply(fields[names(given)], function(f) data_shape(f$shape))
  dropped <- dropped_fields(core)                     # their values are never quoted in a message (programs.md)
  missing <- rep(NA_character_, n)
  left_out <- function(f) if (is.function(f$compute)) default_json(f, f$compute()) else f$shape[["default"]]
  rows <- lapply(seq_len(n), function(i) {
    row <- list()
    for (k in names(fields)) {
      f <- fields[[k]]
      if (!k %in% names(given)) { row[k] <- list(left_out(f)); next }
      v <- element(given[[k]], i)
      if (is_missing(v) && !nullable[[k]]) {
        # a missing value for an optional input is that input left out: its default (programs.md, Binding);
        # for a required one, the row is refused before any request
        if (isTRUE(f$optional)) { row[k] <- list(left_out(f)); next }
        missing[[i]] <<- k
        return(structure(list(inputs = row, field = k, message = sprintf("%s: a missing value (NA), and its type takes no null", k)),
                         class = "functai_misfit"))
      }
      row[k] <- list(bind_json(write[[k]](v), shapes[[k]]))
    }
    for (k in names(given)) {
      why <- check[[k]](row[[k]], k)
      if (!is.null(why)) {
        if (k %in% dropped || "*" %in% dropped)
          why <- sprintf("%s: its value does not fit %s (the value is not shown: the log drops it)", k, type_label(fields[[k]]))
        return(structure(list(inputs = row, field = k, message = why), class = "functai_misfit"))
      }
    }
    if (!length(row)) lmcc::jobj() else row
  })
  attr(rows, "missing") <- missing
  rows
}

# The fields a log_content layer in effect drops for a call of `core`: an
# error message never quotes their values (programs.md, "The message").
dropped_fields <- function(core) {
  tryCatch({
    keep <- content_kept(call_fields(core, effective(core$own)), content_layers(core))
    names(keep)[!unlist(keep)]
  }, error = function(e) "*")
}

# A prediction's `.error`: the call's error, or why a row made no call (an
# input missing, from input_rows()'s `missing`), else NA.
error_or_missing <- function(errors, rows) {
  m <- attr(rows, "missing") %||% rep(NA_character_, length(rows))
  gap <- is.na(errors) & !is.na(m)
  errors[gap] <- sprintf("no call: input %s is missing (NA), and its type takes no null", m[gap])
  errors
}

# Whether a field's type takes JSON's null (an optional() type; a JSON shape
# that allows it).
takes_null <- function(f) {
  shape <- data_shape(f$shape)
  isTRUE(f$nullable) || (well_formed(shape, shape, carry = TRUE) && !loops(shape) && fits_shape(NULL, shape, shape))
}

# A given input's check: a function of the value's JSON giving why it does
# not fit its field, or NULL. A value given to a text input that is not text
# is sent as text (functions.md), so it is checked as the text sent.
checker <- function(f) {
  shape <- data_shape(f$shape)
  if (!well_formed(shape, shape, carry = TRUE) || loops(shape)) return(function(v, name = "value") NULL)   # refused at definition; never here
  text <- is_text_shape(shape)
  fits <- quick_fit(shape)
  function(v, name = "value") {
    if (has_infinity(v)) return(sprintf("%s: an infinite number has no JSON form", name))      # fits no field of an AI function
    if (isTRUE(fits(v))) return(NULL)
    if (text && !is.null(v) && !is_str(v)) v <- text_form(v)
    shape_fault(v, shape, shape, name)
  }
}

# For a shape that checks nothing but a scalar type (and, for text, a set of
# choices), whether a value fits, worked out without reading the shape again
# (a column checks the same shape on every row); NA when the shape asks more,
# and shape_fault() decides. The same answer as shape_fault() gives.
quick_fit <- function(shape) {
  asks <- intersect(names(shape), ASSERTIONS)
  type <- shape$type
  if (!is_str(type)) return(function(v) NA)
  if (identical(asks, "type") && type %in% c("string", "boolean", "integer", "number")) {
    want <- if (type == "number") c("integer", "number") else type
    return(function(v) if (json_type(v) %in% want) TRUE else NA)
  }
  if (setequal(asks, c("type", "enum")) && type == "string" && all(vapply(shape$enum, is_str, NA))) {
    choices <- unlist(shape$enum)
    return(function(v) if (is_str(v) && v %in% choices) TRUE else NA)
  }
  function(v) NA
}

has_infinity <- function(v) if (is.list(v)) any(vapply(v, has_infinity, NA)) else is.numeric(v) && any(is.infinite(v))

# A row that did not bind, as the error its call ends with.
misfit_error <- function(core, row) {
  rlang::error_cnd(class = c("functai_interface_input", "functai_refusal"),
                   message = sprintf("%s: input %s", core$definition$name, row$message), code = "interface-input", field = row$field)
}

# Inputs given by name or by position, named.
named_inputs <- function(core, args) {
  inputs <- names(core$definition$inputs)
  nm <- names(args) %||% rep("", length(args))
  unnamed <- which(!nzchar(nm))
  free <- setdiff(inputs, nm[nzchar(nm)])
  if (length(unnamed) > length(free)) cli::cli_abort("{core$definition$name} takes {length(inputs)} input{?s} ({.field {inputs}})")
  nm[unnamed] <- free[seq_along(unnamed)]
  names(args) <- nm
  missing <- setdiff(required_inputs(core), nm)
  if (length(missing)) cli::cli_abort("no value for input{?s} {.field {missing}}")
  args[intersect(inputs, nm)]
}

# The inputs a caller must give (those without a default).
required_inputs <- function(core) names(Filter(function(f) !isTRUE(f$optional), core$definition$inputs))

# Columns from the rows' outcomes: a vector (one output) or a tibble.
answers <- function(core, results, names_as = identity) {
  outs <- core$definition$outputs
  cols <- lapply(stats::setNames(nm = names(outs)), function(k)
    assemble(outs[[k]], lapply(results, function(r) r$outputs[[k]])))
  if (core$single) return(cols[[1L]])
  names(cols) <- names_as(names(cols))
  tibble::new_tibble(cols, nrow = length(results))
}

report_errors <- function(core, results, on_error) {
  failed <- which(vapply(results, function(r) !is.null(r$error), NA))
  the$problems <- tibble::tibble(row = failed, call = vapply(results[failed], function(r) r$call %||% NA_character_, ""),
                                 error = vapply(results[failed], function(r) conditionMessage(r$error), ""))
  if (!length(failed)) return(invisible())
  first <- results[[failed[[1L]]]]$error
  if (length(results) == 1L || identical(on_error, "stop")) {
    if (inherits(first, "condition")) stop(first) else cli::cli_abort(conditionMessage(first))
  }
  cli::cli_warn(c("{length(failed)} of {length(results)} call{?s} of {.fn {core$definition$name}} failed; {?its/their} answer{?s} {?is/are} NA",
                  x = "{conditionMessage(first)}", i = "{.fn ai_problems} lists them"))
}

call_ai <- function(core, inputs) {
  rows <- input_rows(core, inputs)
  if (!length(rows)) return(answers(core, list()))
  results <- run_rows(core, rows)
  report_errors(core, results, effective(core$own)$on_error %||% "warn")
  answers(core, results)
}

#' The calls that failed in the last call of an AI function
#'
#' @return A tibble: `row`, `call` (its id in the call log), `error`.
#' @export
ai_problems <- function() the$problems %||% tibble::tibble(row = integer(), call = character(), error = character())

# ---------------------------------------------------------------- the object

#' @export
print.functai_fn <- function(x, ...) {
  core <- core_of(x)
  s <- effective(core$own)
  d <- core$definition
  cols <- columns_of(core)
  ins <- names(d$inputs)
  formula <- paste(paste(cols, collapse = " + "), "~", paste(ins, collapse = " + "))
  named_after_answer <- core$single && identical(d$name, unname(cols[[1L]]))
  cat("<ai function> ", if (!named_after_answer) paste0(d$name, ": "), formula, "\n", sep = "")
  said <- core$state$instructions %||% d$written %||% d$description
  if (nzchar(trim_white(said))) cat(gsub("(?m)^\\s*", "  ", trim_white(said), perl = TRUE), "\n", sep = "")
  fields <- c(d$inputs, stats::setNames(d$outputs, cols))
  w <- max(nchar(names(fields)))
  for (k in names(fields)) {
    f <- fields[[k]]
    default <- if (isTRUE(f$optional)) paste0(" = ", lmcc::json_text(f$shape[["default"]])) else ""
    cat(sprintf("  %s  %s%s%s\n", formatC(k, width = -w), type_label(f), default, if (is.null(f$desc)) "" else paste0("  # ", gsub("\\s*\n\\s*", " ", f$desc))))
    if (length(f$meanings)) {
      lw <- max(nchar(names(f$meanings)))
      cat(sprintf("  %s    %s  %s\n", strrep(" ", w), formatC(names(f$meanings), width = -lw), f$meanings), sep = "")
    }
  }
  n <- length(core$state$demos)
  cat(sprintf("  model: %s%s%s\n", s$lm %||% "(the default)", if (n) sprintf(" \u00b7 %d worked example%s", n, if (n > 1) "s" else "") else "",
              if (length(core$tools)) sprintf(" \u00b7 tools: %s", paste(vapply(core$tools, function(t) t$name, ""), collapse = ", ")) else ""))
  invisible(x)
}

# A field's type, in words.
type_label <- function(f) {
  base <- switch(f$kind,
    enum = if (length(f$meanings)) "one of:" else sprintf("one of %s", paste(f$levels, collapse = ", ")),
    record = sprintf("record of %s", paste(names(f$fields), collapse = ", ")),
    json = if (is_record_field(f)) {
      shape <- data_shape(f$shape)
      sprintf("record of %s (list column)", paste(names(value_guide(shape, shape)$properties), collapse = ", "))
    } else "JSON",
    list = sprintf("list of %s", type_label(f$item)),
    string = "text", integer = "whole number", number = "number", boolean = "yes or no", f$kind)
  if (isTRUE(f$nullable)) paste("optional", base) else base
}

#' Change an AI function's settings
#'
#' A copy with other settings: the model, the temperature, the layout...
#' @param object An AI function.
#' @param ... Settings, as in [ai_config()] (without dots: `lm = "claude-haiku-4-5"`).
#' @return An AI function.
#' @export
update.functai_fn <- function(object, ...) {
  core <- core_of(object)
  core$own <- set_all(core$own, check_settings(list(...)))
  core <- own_settings(core)
  check_definition(core, call = rlang::current_env())
  make_fn(core)
}

#' What an AI function sends, and its version
#'
#' `ai_version()` names everything the function sends besides its inputs (its
#' instruction, layout, worked examples, tools): the same function has the
#' same version in Python, TypeScript and R, and choosing another model does
#' not change it. `ai_render()` is the exact request a call would send.
#' `ai_instructions()` the instruction the model gets; `ai_signature()` its
#' fields as data.
#' @param fn An AI function.
#' @param ... Inputs, as for calling it (one row).
#' @return A string (`ai_version()`, `ai_instructions()`) or a list.
#' @export
ai_version <- function(fn) version_of(core_of(fn))

#' @rdname ai_version
#' @export
ai_instructions <- function(fn) { core <- core_of(fn); signature_of(core, effective(core$own))$instructions }

#' @rdname ai_version
#' @export
ai_signature <- function(fn) { core <- core_of(fn); lmcc::signature_to_list(signature_of(core, effective(core$own))) }

#' @rdname ai_version
#' @export
ai_signature_id <- function(fn) { core <- core_of(fn); signature_id(signature_of(core, effective(core$own))) }

#' @rdname ai_version
#' @export
ai_render <- function(fn, ...) {
  core <- core_of(fn)
  s <- effective(core$own)
  r <- route(s)
  s <- adjust_settings(s, r$provider, r$wire)
  plan <- bind_layout(s$adapter, s$template, signature_of(core, s), call_capabilities(r$provider, r$wire, s), r$provider)
  row <- input_rows(core, named_inputs(core, list(...)))[[1L]]
  if (inherits(row, "functai_misfit")) stop(misfit_error(core, row))
  if (is.null(row)) cli::cli_abort("an input is missing (NA), and its type takes no null: this call would send nothing")
  values <- prepare_inputs(plan$signature, row)
  if (length(core$tools)) values$tools <- tool_specs(core)
  turns <- past_turns(core, plan)
  plain_lm15(lmcc::lm15_request(lmcc::render(plan, values, if (length(turns)) turns else NULL), r$model, config_of(s)))
}

#' Worked examples and instructions
#'
#' Copies of an AI function with other worked examples or another
#' instruction (improving a function changes nothing else). `ai_demos()`
#' gives its worked examples as a list of `inputs`/`outputs`.
#' @param fn An AI function.
#' @param demos A data frame whose columns are named like the formula's
#'   inputs and outputs, or a list of `list(inputs = ..., outputs = ...)`.
#' @param instructions The instruction the model gets instead of the one
#'   written from the name and description (`NULL`: the written one).
#' @return An AI function (`with_*()`) or a list.
#' @export
with_demos <- function(fn, demos) {
  core <- core_of(fn)
  core$state$demos <- as_demos(core, demos)
  make_fn(core)
}

#' @rdname with_demos
#' @export
with_instructions <- function(fn, instructions) {
  core <- core_of(fn)
  core$state$instructions <- instructions
  make_fn(core)
}

#' @rdname with_demos
#' @export
ai_demos <- function(fn) core_of(fn)$state$demos

as_demos <- function(core, demos) {
  if (is.null(demos)) return(list())
  if (is.data.frame(demos)) {
    ins <- core$definition$inputs; outs <- core$definition$outputs; cols <- columns_of(core)
    return(lapply(seq_len(nrow(demos)), function(i) {
      d <- list(inputs = lmcc::jobj(), outputs = lmcc::jobj())
      for (k in intersect(names(ins), names(demos))) d$inputs[k] <- list(to_json(ins[[k]], element(demos[[k]], i)))
      for (k in names(outs)[cols %in% names(demos)]) d$outputs[k] <- list(to_json(outs[[k]], element(demos[[cols[[k]]]], i)))
      d
    }))
  }
  lapply(demos, function(d) if (!is.null(d$signature)) d else list(inputs = d$inputs %||% lmcc::jobj(), outputs = d$outputs %||% lmcc::jobj()))
}

# ---------------------------------------------------------------- tidymodels-style

# The answer's field when the function has one output, else NULL.
single_field <- function(core) if (core$single) core$definition$outputs[[1L]] else NULL

is_choice <- function(core) { f <- single_field(core); !is.null(f) && identical(f$kind, "enum") }

pred_names <- function(core) function(n) if (core$single) (if (is_choice(core)) ".pred_class" else ".pred") else paste0(".pred_", n)

# `samples` answers per row at a sampling temperature, remembered per row in
# `memo` (an environment) so class and probability predictions of the same
# rows share their calls.
sampled <- function(core, rows, samples, temperature, memo = NULL, extra = list()) {
  # what decides the answers: the function's version, the model, the sampling
  s <- effective(set_all(core$own, extra))
  prefix <- paste(version_of(core), s$lm %||% "", samples, temperature, sep = "|")
  keys <- vapply(rows, function(r) if (is.null(r)) NA_character_ else paste0(prefix, "|", lmcc::canonical_json(unclass(r))), "")
  todo <- which(!is.na(keys) & !vapply(keys, function(k) !is.na(k) && !is.null(memo) && !is.null(memo[[k]]), NA))
  todo <- todo[!duplicated(keys[todo])]
  if (length(todo)) {
    # a row refused before any request is refused once, not once a sample
    times <- ifelse(vapply(rows[todo], inherits, NA, what = "functai_misfit"), 1L, samples)
    many <- rep(rows[todo], times = times)
    results <- run_rows(core, many, set_all(extra, list(temperature = temperature)))
    ends <- cumsum(times)
    for (j in seq_along(todo)) {
      got <- results[(ends[[j]] - times[[j]]) + seq_len(times[[j]])]
      answers <- lapply(Filter(function(r) is.null(r$error), got), function(r) r$outputs[[answer_name(core)]])
      value <- list(answers = answers, calls = vapply(got, function(r) r$call %||% NA_character_, ""),
                    error = if (!length(answers)) conditionMessage(got[[1L]]$error) else NA_character_)
      if (is.null(memo)) memo <- new.env()
      memo[[keys[[todo[[j]]]]]] <- value
    }
  }
  lapply(keys, function(k) if (is.na(k)) list(answers = list(), calls = character(0), error = NA_character_) else memo[[k]])
}

vote_table <- function(core, votes) {
  lv <- single_field(core)$levels
  probs <- t(vapply(votes, function(v) {
    a <- unlist(lapply(v$answers, as.character))
    if (!length(a)) return(rep(NA_real_, length(lv)))
    as.numeric(table(factor(a, levels = lv))) / length(a)
  }, numeric(length(lv))))
  if (length(votes) == 1L) probs <- matrix(probs, nrow = 1L)
  colnames(probs) <- lv
  probs
}

#' Predictions for a data frame
#'
#' One call per row of `new_data` (its columns named like the function's
#' inputs). Columns follow tidymodels: `.pred_class` when the answer is a
#' choice (a factor), `.pred` for other answers, `.pred_<output>` for several
#' outputs; then `.call` (the call's id, for [rate()]) and `.error` (why a
#' row has no answer: its call's error, or an input missing where its type
#' takes no null, which makes no call).
#' `augment()` adds them to `new_data`, ready for yardstick.
#'
#' **Probabilities.** Most providers (OpenAI, Anthropic, Gemini) do not
#' measure how likely each answer is, and FunctAI never makes a number up.
#' With `samples = k`, each row is answered `k` times at `temperature` (1
#' by default) and `type = "prob"` gives the share of answers per level
#' (`.pred_<level>`), while `type = "class"` gives the most frequent one (a
#' majority vote, which is often more accurate than one answer). It costs
#' `k` calls per row; class and probability predictions of the same rows
#' share them.
#' @param object,x An AI function.
#' @param new_data A data frame.
#' @param type `"class"` or `"numeric"` (the answer, whatever its type) or
#'   `"prob"` (a choice's probabilities; needs `samples`).
#' @param samples Answers per row (1: one call, no probabilities).
#' @param temperature The sampling temperature when `samples > 1`.
#' @param ... Settings for these calls (`lm = ...`).
#' @return A tibble.
#' @export
predict.functai_fn <- function(object, new_data, type = NULL, samples = 1L, temperature = 1, ...) {
  core <- core_of(object)
  settings <- check_settings(list(...))
  missing <- setdiff(required_inputs(core), names(new_data))
  if (length(missing)) cli::cli_abort("{.arg new_data} has no column for input{?s} {.field {missing}}")
  type <- type %||% "class"
  if (!type %in% c("class", "numeric", "prob", "raw")) cli::cli_abort("{.arg type} is \"class\", \"numeric\" or \"prob\", not {.val {type}}")
  rows <- input_rows(core, as.list(new_data)[intersect(names(core$definition$inputs), names(new_data))], n = nrow(new_data))
  samples <- as.integer(samples)
  if (type == "prob" || samples > 1L) {
    if (!is_choice(core)) cli::cli_abort(c("probabilities and votes need an answer that is a choice: {.code {columns_of(core)[[1L]]} = choice(...)}"))
  }
  if (type == "prob" && samples < 2L) {
    if (!measures_probabilities(core, settings))
      cli::cli_abort(c("this model gave no probabilities: OpenAI, Anthropic and Gemini do not measure how likely each answer is, and FunctAI does not make that number up",
        i = "use a model that measures them ({.code lm = \"jev-latest\"}, TypeSafe's Jev), or ask for {.code samples = 5}: each row is answered that many times and the probability is each answer's share (costs 5 calls a row)"))
    probs <- measured_rows(core, rows, settings)
    out <- tibble::as_tibble(as.data.frame(probs, check.names = FALSE))
    names(out) <- paste0(".pred_", names(out))
    return(out)
  }
  if (samples > 1L) {
    votes <- sampled(core, rows, samples, temperature, core$memo, settings)
    probs <- vote_table(core, votes)
    if (type == "prob") {
      out <- tibble::as_tibble(as.data.frame(probs, check.names = FALSE))
      names(out) <- paste0(".pred_", names(out))
      return(out)
    }
    lv <- single_field(core)$levels
    best <- apply(probs, 1L, function(p) if (all(is.na(p))) NA_character_ else lv[[which.max(p)]])
    return(tibble::tibble(.pred_class = factor(best, levels = lv),
                          .call = vapply(votes, function(v) paste(stats::na.omit(v$calls), collapse = " "), ""),
                          .error = error_or_missing(vapply(votes, function(v) v$error, ""), rows)))
  }
  results <- memo_rows(core, rows, settings)
  out <- answers(core, results, pred_names(core))
  if (core$single) { col <- out; out <- tibble::tibble(x = col); names(out) <- pred_names(core)("result") }
  out$.call <- vapply(results, function(r) r$call %||% NA_character_, "")
  out$.error <- error_or_missing(vapply(results, function(r) if (is.null(r$error)) NA_character_ else conditionMessage(r$error), ""), rows)
  attr(out, "turns") <- lapply(results, function(r) r$turn)
  if (is_choice(core)) attr(out, "probabilities") <- measured_table(core, results)
  out
}

# Probabilities a provider measured itself (TypeSafe's Jev answers a choice
# with a distribution over its levels): one row per result, a column per
# level, NA where a row's answer came without one. Never made up.
measured_table <- function(core, results) {
  lv <- single_field(core)$levels
  field <- answer_name(core)
  probs <- matrix(NA_real_, nrow = length(results), ncol = length(lv), dimnames = list(NULL, lv))
  for (i in seq_along(results)) {
    dist <- results[[i]]$probabilities[[field]]
    if (!length(dist)) next
    probs[i, ] <- vapply(lv, function(l) as.numeric(dist[[l]] %||% 0), 0)
  }
  probs
}

# Does the model this function calls measure its own probabilities? The
# judgment-only providers of the contract's table do (TypeSafe's Jev).
measures_probabilities <- function(core, settings = list()) {
  r <- tryCatch(route(effective(set_all(core$own, settings))), error = function(e) NULL)
  !is.null(r) && r$provider %in% provider_sets()$judgment
}

# A model that measures its probabilities gives them with its answer, so a
# class prediction and a probability prediction of the same rows (parsnip's
# augment() asks for both, in either order) share one call a row: a fitted
# model keeps each row's result in `core$memo`, as it keeps votes. Only a
# fitted model has a memo, and only a measuring model uses it this way.
memo_rows <- function(core, rows, settings) {
  if (is.null(core$memo) || !measures_probabilities(core, settings)) return(if (length(rows)) run_rows(core, rows, settings) else list())
  s <- effective(set_all(core$own, settings))
  prefix <- paste("measured", version_of(core), s$lm %||% "", sep = "|")
  keys <- vapply(rows, function(r) if (is.null(r)) NA_character_ else paste0(prefix, "|", lmcc::canonical_json(unclass(r))), "")
  todo <- which(!is.na(keys) & !vapply(keys, function(k) !is.na(k) && !is.null(core$memo[[k]]), NA))
  todo <- todo[!duplicated(keys[todo])]
  if (length(todo)) {
    results <- run_rows(core, rows[todo], settings)
    for (j in seq_along(todo)) core$memo[[keys[[todo[[j]]]]]] <- results[[j]]
  }
  lapply(keys, function(k) if (is.na(k)) list(skipped = TRUE) else core$memo[[k]])
}

measured_rows <- function(core, rows, settings) {
  results <- memo_rows(core, rows, settings)
  report_errors(core, results, "warn")
  measured_table(core, results)
}

#' @importFrom generics augment
#' @export
generics::augment

#' @rdname predict.functai_fn
#' @method augment functai_fn
#' @export
augment.functai_fn <- function(x, new_data, ...) {
  args <- list(...)
  if (is.null(args$type) && !is.null(args$samples) && args$samples > 1L) {
    core <- core_of(x)
    if (is.null(core$memo)) { core$memo <- new.env(); x <- make_fn(core) }     # the probabilities reuse the class's calls
    p <- predict.functai_fn(x, new_data, ...)
    probs <- predict.functai_fn(x, new_data, type = "prob", ...)
    p <- vctrs::vec_cbind(p[".pred_class"], probs, p[c(".call", ".error")])
  } else {
    p <- predict.functai_fn(x, new_data, ...)
    probs <- attr(p, "probabilities")
    if (!is.null(probs) && any(stats::complete.cases(probs))) {       # the model measured them: they come with the answers
      probs <- tibble::as_tibble(as.data.frame(probs, check.names = FALSE))
      names(probs) <- paste0(".pred_", names(probs))
      p <- vctrs::vec_cbind(p[".pred_class"], probs, p[c(".call", ".error")])
    }
  }
  attr(p, "turns") <- NULL
  attr(p, "probabilities") <- NULL
  tibble::as_tibble(vctrs::vec_cbind(tibble::as_tibble(new_data), p))
}
