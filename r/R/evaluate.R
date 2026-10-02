# How often a function is right (contract/scores.md).

Z <- 1.959964
T975 <- c(12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145,
          2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042)

t975 <- function(df) if (df <= 30) T975[[df]] else Z + (Z^3 + Z) / (4 * df) + (5 * Z^5 + 16 * Z^3 + 3 * Z) / (96 * df^2)

#' The mean of scores and its 95% interval
#'
#' Wilson's interval when every score is 0 or 1 (right or wrong), Student's
#' t otherwise, exactly as FunctAI computes it in every language.
#' @param x Numeric scores.
#' @return A list: `mean`, `low`, `high` (`NA` with fewer than two scores).
#' @examples
#' score_interval(c(1, 1, 0, 1))
#' @export
score_interval <- function(x) {
  n <- length(x)
  if (!n) return(list(mean = NA_real_, low = NA_real_, high = NA_real_))
  m <- 0
  for (v in x) m <- m + v
  m <- m / n
  if (n < 2) return(list(mean = m, low = NA_real_, high = NA_real_))
  if (all(x %in% c(0, 1))) {
    z2 <- Z * Z
    center <- (m + z2 / (2 * n)) / (1 + z2 / n)
    half <- Z * sqrt(m * (1 - m) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return(list(mean = m, low = if (m == 0) 0 else max(0, center - half), high = if (m == 1) 1 else min(1, center + half)))
  }
  half <- t975(n - 1) * sqrt(sum((x - m)^2) / (n - 1)) / sqrt(n)
  list(mean = m, low = m - half, high = m + half)
}

norm_value <- function(v) if (is.character(v) && length(v) == 1L && !is.na(v)) normalize_text(v) else v

same_value <- function(a, b) {
  a <- norm_value(a); b <- norm_value(b)
  if (is.character(a) || is.character(b)) return(identical(a, b))
  if (is.numeric(a) && is.numeric(b) && length(a) == 1L && length(b) == 1L) return(isTRUE(a == b))
  identical(lmcc::canonical_json(plain_json(a)), lmcc::canonical_json(plain_json(b)))
}

#' Is an answer right?
#'
#' 1 when every output the answers have a value for equals the prediction's
#' (text compared ignoring case and repeated white space), else 0; with
#' several outputs, each also gets `<name>_match`.
#' @param answers A named list of right answers.
#' @param prediction A named list of predicted outputs.
#' @return A named numeric vector.
#' @examples
#' exact_match(list(result = "New York"), list(result = "  new   york "))
#' @export
exact_match <- function(answers, prediction) {
  keys <- intersect(names(prediction), names(answers))
  if (!length(keys)) cli::cli_abort("exact_match: the data has no column for any output ({names(prediction)})")
  each <- vapply(keys, function(k) as.numeric(same_value(answers[[k]], prediction[[k]])), 0)
  out <- c(exact_match = as.numeric(all(each == 1)))
  if (length(keys) > 1L) out <- c(out, stats::setNames(each, paste0(keys, "_match")))
  out
}

#' How often is a model right?
#'
#' Runs a model on every row of `data` and scores each answer against the
#' right one, with a 95% interval. The model is an AI function, a fitted
#' parsnip model, a fitted workflow, or anything with a `predict()` method
#' returning `.pred_class` or `.pred`: the same score, the same interval, so
#' an AI function and a classical model compare on equal terms. A row whose
#' call failed scores 0. An AI function's calls are logged with
#' `caller.evaluation` set, so they are not mistaken for use.
#' @param fn A model: an AI function, a parsnip fit, a workflow, ...
#' @param data A data frame: the model's inputs, and the right answers.
#' @param expected Where the right answers are: a column (bare or quoted) for
#'   the answer, or a named character vector `c(output = "column")`. Default:
#'   the columns named like the formula's outputs (an AI function) or the
#'   outcome the model was fitted on (a parsnip fit or a workflow).
#' @param metric A function `(row, prediction)` returning a score (both are
#'   named lists), or a named list of such functions. Default: [exact_match()].
#' @param ... Settings for an AI function's calls (`lm = ...`), or arguments
#'   to another model's `predict()`.
#' @return An evaluation: `print()` it, or use [generics::tidy()] (the score
#'   and its 95% interval), [generics::glance()] and [generics::augment()]
#'   (every row, its answer and score).
#' @export
evaluate <- function(fn, data, expected = NULL, metric = NULL, ...) {
  expected_q <- rlang::enexpr(expected)
  expected_v <- if (is.null(expected_q)) NULL else if (is.symbol(expected_q)) as.character(expected_q) else eval(expected_q, parent.frame())
  metrics <- if (is.null(metric)) NULL else if (is.function(metric)) list(metric = metric) else metric
  run <- new_id()
  if (inherits(fn, c("functai_fn", "functai_program"))) {
    is_ai <- inherits(fn, "functai_fn")
    core <- if (is_ai) core_of(fn) else program_core(fn)
    outs <- columns_of(core)                                       # named like the formula
    mapping <- if (is.null(expected_v)) { m <- unname(outs[outs %in% names(data)]); stats::setNames(m, m) }
      else if (is.null(names(expected_v))) stats::setNames(expected_v, outs[[length(outs)]]) else expected_v
    unknown <- setdiff(names(mapping), outs)
    if (length(unknown)) cli::cli_abort("{.fn {core$definition$name}} has no output {.field {unknown}} (its outputs: {.field {unname(outs)}})")
    label <- core$definition$name
    more <- list(...)
    settings <- set_all(more, list(caller = c(list(evaluation = run), more$caller)))    # an optimizer's caller too
    p <- do.call(if (is_ai) predict.functai_fn else predict.functai_program, c(list(fn, data), settings))
    cols <- if (core$single) stats::setNames(if (is_ai) pred_names(core)("result") else ".pred", outs) else stats::setNames(paste0(".pred_", names(outs)), outs)
  } else {
    outcome <- expected_v %||% model_outcome(fn)
    if (is.null(outcome)) cli::cli_abort("which column holds the right answers? pass {.arg expected}")
    p <- tibble::as_tibble(stats::predict(fn, data, ...))
    col <- intersect(c(".pred_class", ".pred"), names(p))
    if (!length(col)) cli::cli_abort("{.fn predict} on {.cls {class(fn)[[1L]]}} gave no {.field .pred_class} or {.field .pred} column")
    mapping <- stats::setNames(unname(outcome[[1L]]), unname(outcome[[1L]]))
    cols <- stats::setNames(col[[1L]], unname(outcome[[1L]]))
    label <- if (inherits(fn, "model_fit")) class(fn$spec)[[1L]]
      else if (inherits(fn, "workflow") && requireNamespace("workflows", quietly = TRUE)) paste0("workflow (", class(workflows::extract_spec_parsnip(fn))[[1L]], ")")
      else class(fn)[[1L]]
    if (!".error" %in% names(p)) p$.error <- rep(NA_character_, nrow(p))
  }
  missing_cols <- setdiff(unname(mapping), names(data))
  if (length(missing_cols)) cli::cli_abort("{.arg data} has no column {.field {missing_cols}}")
  if (is.null(metrics) && !length(mapping)) cli::cli_abort("the data has no column for any output: pass {.arg expected} or {.arg metric}")
  preds <- lapply(seq_len(nrow(data)), function(i) lapply(cols, function(k) element(p[[k]], i)))
  scores <- lapply(seq_len(nrow(data)), function(i) {
    if (!is.na(p$.error[[i]])) return(NULL)
    row <- lapply(as.list(data), element, i = i)
    if (is.null(metrics)) exact_match(lapply(mapping, function(col) row[[col]]), preds[[i]][names(mapping)])
    else vapply(metrics, function(m) as.numeric(m(row, preds[[i]])), 0)
  })
  names_ <- if (!is.null(metrics)) names(metrics) else c("exact_match", if (length(mapping) > 1L) paste0(names(mapping), "_match"))
  table <- tibble::as_tibble(vctrs::vec_cbind(tibble::as_tibble(data), p[setdiff(names(p), names(data))]))
  for (m in names_) table[[m]] <- vapply(scores, function(s) if (is.null(s)) 0 else s[[m]], 0)
  structure(list(run = run, metrics = names_, rows = table, fn = label), class = "functai_evaluation")
}

# The outcome column a fitted parsnip model or workflow was fitted on.
model_outcome <- function(model) {
  if (inherits(model, "workflow") && requireNamespace("workflows", quietly = TRUE)) {
    y <- tryCatch(names(workflows::extract_mold(model)$outcomes), error = function(e) NULL)
    if (length(y) == 1L && !startsWith(y, "..")) return(y)
  }
  if (inherits(model, "model_fit")) {
    y <- model$preproc$y_var
    if (length(y) == 1L && !startsWith(y, "..")) return(y)
  }
  NULL
}

#' @importFrom generics tidy
#' @export
generics::tidy

#' @importFrom generics glance
#' @export
generics::glance

#' Tidy an evaluation
#'
#' `tidy()`: one row per metric, broom's columns (`estimate`, `conf.low`,
#' `conf.high`) and `n`, `failed`. `glance()`: the first metric in one row.
#' `augment()`: every row with its answer (`.pred_class` or `.pred`...), its score, its call
#' id and its error.
#' @param x An evaluation from [evaluate()].
#' @param ... Unused.
#' @return A tibble.
#' @name tidy.functai_evaluation
#' @method tidy functai_evaluation
#' @export
tidy.functai_evaluation <- function(x, ...) {
  failed <- sum(!is.na(x$rows$.error))
  rows <- lapply(x$metrics, function(m) {
    ci <- score_interval(x$rows[[m]])
    tibble::tibble(metric = m, estimate = ci$mean, conf.low = ci$low, conf.high = ci$high, n = nrow(x$rows), failed = failed)
  })
  vctrs::vec_rbind(!!!rows)
}

#' @rdname tidy.functai_evaluation
#' @method glance functai_evaluation
#' @export
glance.functai_evaluation <- function(x, ...) tidy.functai_evaluation(x)[1L, ]

#' @rdname tidy.functai_evaluation
#' @method augment functai_evaluation
#' @export
augment.functai_evaluation <- function(x, ...) x$rows

#' @export
print.functai_evaluation <- function(x, ...) {
  t <- tidy.functai_evaluation(x)
  f <- function(v) if (is.na(v)) "-" else sprintf("%.2f", v)
  cat(sprintf("<evaluation of %s> %d rows%s\n", x$fn, t$n[[1L]], if (t$failed[[1L]]) sprintf(", %d failed (scored 0)", t$failed[[1L]]) else ""))
  for (i in seq_len(nrow(t))) cat(sprintf("  %s: %s  (95%% interval %s to %s)\n", t$metric[[i]], f(t$estimate[[i]]), f(t$conf.low[[i]]), f(t$conf.high[[i]])))
  invisible(x)
}

#' @export
`$.functai_evaluation` <- function(x, name) {
  if (identical(name, "score")) return(score_interval(.subset2(x, "rows")[[.subset2(x, "metrics")[[1L]]]])$mean)
  .subset2(x, name)
}
