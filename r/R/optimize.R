# Improving a function: choosing its worked examples. Each returns an
# improved copy with a new version; the function you pass is unchanged. The
# contract fixes what improving means, not its random choices: a seed makes a
# run repeatable here, not equal to Python's run with the same seed.

#' Worked examples from rows with known answers
#'
#' `labeled_few_shot()` uses up to `k` rows as they are. `bootstrap_few_shot()`
#' runs the function (or a stronger `teacher` model) on the rows and keeps the
#' runs that were right, whole (reasoning and tool calls included), then fills
#' up with labeled rows.
#' @param fn An AI function.
#' @param data A data frame with the formula's input and output columns.
#' @param k,max_labeled How many labeled rows at most.
#' @param sample Pick rows at random (`seed`) rather than the first ones.
#' @param seed The random seed for picking rows.
#' @param max_bootstrapped How many right runs at most.
#' @param teacher A model to write the examples (`"gpt-4.1"`); default the function's own.
#' @param metric `(row, prediction)` returning a score; a run counts when it is
#'   above 0 (or at least `threshold`). Default: [exact_match()].
#' @param threshold See `metric`.
#' @return An AI function.
#' @examples
#' \dontrun{
#' taught <- mood |> labeled_few_shot(train, k = 8)
#' }
#' @export
labeled_few_shot <- function(fn, data, k = 16L, sample = TRUE, seed = 0L) {
  chosen <- if (sample) withr::with_seed(seed, sample.int(nrow(data), min(k, nrow(data)))) else seq_len(min(k, nrow(data)))
  with_demos(fn, keep_labeled(fn, data[chosen, , drop = FALSE]))
}

keep_labeled <- function(fn, rows) {
  cols <- columns_of(core_of(fn))
  if (!any(cols %in% names(rows))) cli::cli_abort("the rows have no column for any output ({.field {unname(cols)}})")
  rows
}

#' @rdname labeled_few_shot
#' @export
bootstrap_few_shot <- function(fn, data, max_bootstrapped = 4L, max_labeled = 16L, teacher = NULL, metric = NULL,
                               threshold = NULL, seed = 0L) {
  core <- core_of(fn)
  outs <- names(core$definition$outputs)
  cols <- unname(columns_of(core))
  runner <- if (is.null(teacher)) fn else stats::update(fn, lm = teacher)
  passes <- function(s) if (is.null(threshold)) s > 0 else s >= threshold
  score <- function(row, pred) {
    if (!is.null(metric)) return(as.numeric(metric(row, pred)))
    exact_match(row[intersect(cols, names(row))], pred[intersect(cols, names(row))])[["exact_match"]]
  }
  boot <- list(); used <- integer(0)
  step <- max(1L, as.integer(effective(core$own)$concurrency))
  id <- new_id()
  for (start in seq(1L, nrow(data), by = step)) {
    if (length(boot) >= max_bootstrapped) break
    idx <- start:min(nrow(data), start + step - 1L)
    p <- predict.functai_fn(runner, data[idx, , drop = FALSE], caller = list(optimization = id))
    turns <- attr(p, "turns")
    for (j in seq_along(idx)) {
      if (length(boot) >= max_bootstrapped || !is.na(p$.error[[j]])) next
      row <- lapply(as.list(data[idx[[j]], , drop = FALSE]), element, i = 1L)
      pred <- if (core$single) stats::setNames(list(element(p[[pred_names(core)("result")]], j)), cols) else
        stats::setNames(lapply(paste0(".pred_", outs), function(k) element(p[[k]], j)), cols)
      if (passes(score(row, pred))) { boot[[length(boot) + 1L]] <- lmcc::turn_to_list(turns[[j]]); used <- c(used, idx[[j]]) }
    }
  }
  rest <- setdiff(seq_len(nrow(data)), used)
  room <- max(0L, max_labeled - length(boot))
  fill <- if (room && length(rest)) rest[withr::with_seed(seed, sample.int(length(rest), min(room, length(rest))))] else integer(0)
  labeled <- as_demos(core, data[fill, , drop = FALSE])
  core$state$demos <- c(boot, labeled)
  make_fn(core)
}
