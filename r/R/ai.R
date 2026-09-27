# AI functions: a vectorised R function whose body a model writes.

#' Write a function a language model does the work of
#'
#' Name the function, say in a sentence what it does, give its inputs and
#' the type of its answer. The result is an ordinary, vectorised R function:
#' call it on one value or on whole columns inside [dplyr::mutate()]. Each
#' row is one model call; up to `concurrency` (default 8) run at once.
#'
#' Types are prototypes, the way vctrs writes them: `character()`,
#' `integer()`, `double()`, `logical()`, `factor(levels = c(...))` (one of
#' those levels), a zero-row [tibble::tibble()] (a record of its columns),
#' [vctrs::list_of()] (a list), [optional()], [json_shape()]. [described()]
#' adds words about a field.
#'
#' @param .name The function's name. The model reads it (`"Function: mood"`),
#'   and the call log files calls under it.
#' @param .description What it does, in a sentence or a paragraph.
#' @param ... Inputs, as `name = prototype`; and settings, as dotted names
#'   (`.lm = "claude-haiku-4-5"`, `.temperature = 0`, `.adapter = "json"`,
#'   `.module = "cot"`, ...: see [ai_config()]).
#' @param .returns The answer's type (default `character()`), named `result`.
#' @param .outputs Several outputs instead: a named list of prototypes. The
#'   last is the answer. The function then returns a tibble, one column each.
#' @param .tools Functions the model may call, from [ai_tool()].
#' @param .demos Worked examples: a data frame with input and output columns.
#' @param .instructions An improved instruction that replaces the written one.
#' @param .defined_in The code module the call log files calls under
#'   (default `"__main__"`, as a Python notebook's).
#' @return An AI function (class `functai_fn`).
#' @examples
#' mood <- ai("mood", "How does the customer feel about what they bought?",
#'   review = character(),
#'   .returns = factor(levels = c("happy", "unhappy", "mixed")))
#' mood
#' \dontrun{
#' mood("It broke after one day.")
#' tickets |> dplyr::mutate(mood = mood(message))
#' }
#' @export
ai <- function(.name, .description = "", ..., .returns = character(), .outputs = NULL, .tools = NULL,
               .demos = NULL, .instructions = NULL, .defined_in = NULL) {
  if (!rlang::is_string(.name) || !nzchar(.name)) cli::cli_abort("{.arg .name} is the function's name, one string")
  dots <- list(...)
  nms <- names(dots) %||% rep("", length(dots))
  if (any(!nzchar(nms))) cli::cli_abort("every input is named: {.code review = character()}")
  settings <- dots[startsWith(nms, ".")]
  names(settings) <- substring(names(settings), 2L)
  check_settings(settings)
  inputs <- lapply(dots[!startsWith(nms, ".")], as_field)
  single <- is.null(.outputs)
  outputs <- if (single) list(result = as_field(.returns)) else lapply(.outputs, as_field)
  if (!length(outputs) || is.null(names(outputs)) || any(!nzchar(names(outputs)))) cli::cli_abort("{.arg .outputs} is a named list of types")
  core <- list(
    definition = list(name = .name, description = .description, inputs = inputs, outputs = outputs),
    own = settings, tools = .tools %||% list(), single = single,
    module = .defined_in %||% "__main__", state = list(instructions = .instructions, demos = list()), saved = NULL)
  fn <- make_fn(core)
  if (!is.null(.demos)) fn <- with_demos(fn, .demos)
  fn
}

make_fn <- function(core) {
  env <- new.env(parent = asNamespace("functai"))
  env$.core <- core
  env$.inputs <- names(core$definition$inputs)
  args <- rep(list(rlang::missing_arg()), length(env$.inputs))
  names(args) <- env$.inputs
  f <- rlang::new_function(args, quote(call_ai(.core, mget(.inputs, envir = environment()))), env)
  structure(f, class = c("functai_fn", "function"))
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

probe_request <- function(core, inputs = NULL) {
  s <- effective(core$own)
  sig <- signature_of(core, s)
  plan <- bind_layout(s$adapter, s$template, sig, probe_capabilities(), "probe")
  values <- prepare_inputs(plan$signature, inputs %||% sample_inputs(plan$signature))
  if (length(core$tools)) values$tools <- tool_specs(core)
  turns <- past_turns(core, plan)
  lmcc::request_of(lmcc::render(plan, values, if (length(turns)) turns else NULL), "probe")
}

request_hash <- function(core, inputs = NULL) {
  tryCatch(lmcc::sha256_of(probe_request(core, inputs)), lmcc_refusal = function(e) paste0("refused:", e$code))
}

version_of <- function(core) lmcc::sha256_of(list(request = request_hash(core)))

program_of <- function(core, version = NULL) {
  function() {
    s <- effective(core$own)
    p <- list(name = core$definition$name, kind = "ai", module = core$module, version = version %||% version_of(core),
              signature = signature_id(signature_of(core, s)), answer = answer_name(core))
    if (!is.null(core$saved)) p$saved <- core$saved
    p
  }
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

# Run one call per row. `rows`: a list of JSON input lists (NULL: skip the row).
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
  jobs <- list(); calls <- vector("list", length(rows))
  for (i in seq_along(rows)) {
    if (is.null(rows[[i]])) next
    call <- start_call(program, s, rows[[i]])
    call$provider <- r$provider
    calls[[i]] <- call
    job <- tryCatch(new_job(plan, past, rows[[i]], s, r$model, core$tools, call), error = identity)
    if (inherits(job, "error")) { e <- job; job <- new.env(); job$state <- "failed"; job$error <- e }
    job$call <- call
    jobs[[as.character(i)]] <- job
  }
  run_jobs(unname(jobs), r$router, s$concurrency)
  lapply(seq_along(rows), function(i) {
    job <- jobs[[as.character(i)]]
    if (is.null(job)) return(list(skipped = TRUE))
    if (identical(job$state, "done")) {
      job$call$outputs <- job$outputs
      job$call$confidence <- confidence_of(job$outputs, job$probabilities)
      finish_call(job$call)
      list(outputs = job$outputs, call = job$call$id, model = r$model, turn = job$turn, probabilities = job$probabilities)
    } else {
      finish_call(job$call, job$error)
      list(error = job$error, call = job$call$id, model = r$model)
    }
  })
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

# The inputs, recycled, as one JSON list per row (NULL for a row with a missing input).
input_rows <- function(core, inputs) {
  inputs <- vctrs::vec_recycle_common(!!!inputs)
  n <- vctrs::vec_size_common(!!!inputs)
  fields <- core$definition$inputs
  lapply(seq_len(n), function(i) {
    row <- list()
    for (k in names(fields)) {
      v <- element(inputs[[k]], i)
      if (is_missing(v) && !isTRUE(fields[[k]]$nullable)) return(NULL)
      row[k] <- list(to_json(fields[[k]], v))
    }
    if (!length(row)) lmcc::jobj() else row
  })
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
  missing <- setdiff(inputs, nm)
  if (length(missing)) cli::cli_abort("no value for input{?s} {.field {missing}}")
  args[inputs]
}

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
  shape <- function(f) switch(f$kind, enum = sprintf("factor [%s]", paste(f$levels, collapse = ", ")), record = sprintf("record (%s)", paste(names(f$fields), collapse = ", ")),
                              list = sprintf("list of %s", f$item$kind), f$kind)
  outs <- core$definition$outputs
  cat(sprintf("<ai function> %s(%s) -> %s\n", core$definition$name, paste(names(core$definition$inputs), collapse = ", "),
              paste(sprintf("%s: %s", names(outs), vapply(outs, shape, "")), collapse = ", ")))
  cat(gsub("(?m)^", "  ", signature_of(core, s)$instructions, perl = TRUE), "\n", sep = "")
  n <- length(core$state$demos)
  cat(sprintf("model: %s%s%s\n", s$lm %||% "(the default)", if (n) sprintf(" \u00b7 %d worked example%s", n, if (n > 1) "s" else "") else "",
              if (length(core$tools)) sprintf(" \u00b7 tools: %s", paste(vapply(core$tools, function(t) t$name, ""), collapse = ", ")) else ""))
  invisible(x)
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
#' @param demos A data frame whose columns are named like the inputs and
#'   outputs, or a list of `list(inputs = ..., outputs = ...)`.
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
    ins <- core$definition$inputs; outs <- core$definition$outputs
    return(lapply(seq_len(nrow(demos)), function(i) {
      d <- list(inputs = lmcc::jobj(), outputs = lmcc::jobj())
      for (k in intersect(names(ins), names(demos))) d$inputs[k] <- list(to_json(ins[[k]], element(demos[[k]], i)))
      for (k in intersect(names(outs), names(demos))) d$outputs[k] <- list(to_json(outs[[k]], element(demos[[k]], i)))
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
  keys <- vapply(rows, function(r) if (is.null(r)) NA_character_ else paste0(prefix, "|", lmcc::canonical_json(r)), "")
  todo <- which(!is.na(keys) & !vapply(keys, function(k) !is.na(k) && !is.null(memo) && !is.null(memo[[k]]), NA))
  todo <- todo[!duplicated(keys[todo])]
  if (length(todo)) {
    many <- rep(rows[todo], each = samples)
    results <- run_rows(core, many, set_all(extra, list(temperature = temperature)))
    for (j in seq_along(todo)) {
      got <- results[(j - 1L) * samples + seq_len(samples)]
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
#' outputs; then `.call` (the call's id, for [rate()]) and `.error`.
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
  missing <- setdiff(names(core$definition$inputs), names(new_data))
  if (length(missing)) cli::cli_abort("{.arg new_data} has no column for input{?s} {.field {missing}}")
  type <- type %||% "class"
  if (!type %in% c("class", "numeric", "prob", "raw")) cli::cli_abort("{.arg type} is \"class\", \"numeric\" or \"prob\", not {.val {type}}")
  rows <- input_rows(core, as.list(new_data)[names(core$definition$inputs)])
  samples <- as.integer(samples)
  if (type == "prob" || samples > 1L) {
    if (!is_choice(core)) cli::cli_abort(c("probabilities and votes need an answer that is a choice: {.code .returns = factor(levels = ...)}"))
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
                          .error = vapply(votes, function(v) v$error, "")))
  }
  results <- memo_rows(core, rows, settings)
  out <- answers(core, results, pred_names(core))
  if (core$single) { col <- out; out <- tibble::tibble(x = col); names(out) <- pred_names(core)("result") }
  out$.call <- vapply(results, function(r) r$call %||% NA_character_, "")
  out$.error <- vapply(results, function(r) if (is.null(r$error)) NA_character_ else conditionMessage(r$error), "")
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
  keys <- vapply(rows, function(r) if (is.null(r)) NA_character_ else paste0(prefix, "|", lmcc::canonical_json(r)), "")
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
