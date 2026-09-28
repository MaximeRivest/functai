# GEPA: the instruction rewritten from the function's mistakes (Agrawal et
# al., 2025), with functai's changes. The algorithm, what it changes and why
# are in design/04-gepa.md; Python's functai.GEPA is the same algorithm, and
# its two prompts send the same words.

REFLECT_TEXT <- "You improve the instruction of a function that a language model runs. You are given what the
function takes and returns, its current instruction, and cases it was run on: each with its inputs,
the answer it gave, its score and feedback. Find what the instruction is missing, or gets wrong, that
explains the mistakes, and write an improved instruction. Write general rules a careful person could
follow on new cases; never copy an input or describe these particular cases. Keep what already works.
The instruction is everything the model is told besides the inputs: keep the task, and say what each
output must be. Instructions listed as tried did not do better: try something different. Reply with
the new instruction only."

COMBINE_TEXT <- "Two instructions for the same function each get right some cases the other gets wrong. Write one
instruction that keeps what makes each of them right, without repeating itself. The instruction is
everything the model is told besides the inputs: keep the task, and say what each output must be.
Reply with the new instruction only."

# The two AI functions GEPA calls. Their layout is fixed, so a session's
# ai_config() (another layout, reasoning first) does not change how they work;
# they reach models, and log, the way the function being improved does. Their
# inputs quote the function's cases, so when the function's own log_content
# drops any of its fields, theirs drops every value (the blocks and
# ai_config() around them apply to them as to any call).
meta_fn <- function(formula, text, name, teacher, like, run_id) {
  s <- effective(like)
  own <- like$log_content
  drops <- isFALSE(own) || (is.list(own) && any(vapply(own, isFALSE, NA)))
  ai(formula, text, .name = name, .defined_in = "functai.meta", .lm = teacher %||% s$lm, .router = s$router,
     .log_calls = s$log_calls, .log_content = if (drops) FALSE, .caller = list(optimization = run_id),
     .temperature = 1, .include_fn_name = FALSE, .adapter = "xml", .module = "predict", .on_error = "stop")
}

#' Rewrite the instruction from the function's mistakes (GEPA)
#'
#' A model reads the function's answers on rows with known answers, with
#' feedback in words ("wrong: the right answer is billing"), and writes a
#' better instruction; the best of the instructions it writes is kept. This
#' is GEPA (Agrawal et al., 2025) with changes that suit a single function:
#' the model writing instructions sees what it tried that failed; a proposal
#' that copies an input instead of stating a rule is dropped; two
#' instructions that are right on different rows are combined; of equally
#' good instructions the shorter wins; and no row is run twice for the same
#' instruction. `design/04-gepa.md` in the repository gives the reasons.
#'
#' **How it works.** Half the rows (or `selection`) are for choosing and are
#' never shown to the model writing instructions; the other half are for
#' feedback. Every candidate instruction is scored on each choosing row. A
#' parent is picked among the candidates that are best on at least one row,
#' run on `minibatch` feedback rows, and `teacher` writes a new instruction
#' from its answers and their feedback. A new instruction that does better
#' on those rows is scored on the choosing rows and joins the candidates.
#'
#' **What it costs.** `budget` calls of the function, plus one call of
#' `teacher` for each instruction it writes. **Its score flatters:** the
#' instruction kept is the best of many on the choosing rows, so measure it
#' on rows it never saw, with [evaluate()] (or resample it in tidymodels,
#' with `set_engine("functai", method = "gepa")`).
#'
#' @param fn An AI function.
#' @param data Rows with the inputs and the right answers (in the columns
#'   named like the formula's outputs, or see `expected`).
#' @param expected Where the right answers are, as in [evaluate()].
#' @param metric A function `(row, prediction)` returning a score, 1 meaning
#'   right, as in [evaluate()]. Default: exact match.
#' @param feedback A function `(row, prediction, error)` returning words about
#'   one answer (`prediction` is named like the outputs; `error` is `NULL`
#'   unless the call failed). Default: `"right"`, `"wrong: the right answer
#'   is ..."`, or the call's error.
#' @param budget How many calls of `fn` at most.
#' @param minibatch Feedback rows per new instruction.
#' @param teacher The model that writes instructions (`"gpt-6-sol"`).
#'   Default: `fn`'s own.
#' @param selection Rows to choose with, instead of half of `data`.
#' @param seed The random seed for splitting rows and picking parents.
#' @return A copy of `fn` with the best instruction (`fn` itself, with its
#'   search, when none beat the written one). [ai_trials()] gives the search.
#' @examples
#' \dontrun{
#' team <- ai(team ~ message, "Which team should answer this customer message?",
#'   team = choice("shipping", "billing", "product", "account"))
#' better <- gepa(team, train, expected = category, teacher = "gpt-6-sol")
#' ai_instructions(better)
#' ai_trials(better)
#' evaluate(better, test, expected = category)
#' }
#' @export
gepa <- function(fn, data, expected = NULL, metric = NULL, feedback = NULL, budget = 300L, minibatch = 4L,
                 teacher = NULL, selection = NULL, seed = 0L) {
  core <- core_of(fn)
  expected_q <- rlang::enexpr(expected)
  expected_v <- if (is.null(expected_q)) NULL else if (is.symbol(expected_q)) as.character(expected_q) else eval(expected_q, parent.frame())
  outs <- columns_of(core)                                   # definition name -> column name
  mapping <- if (is.null(expected_v)) { m <- unname(outs[outs %in% names(data)]); stats::setNames(m, m) }
    else if (is.null(names(expected_v))) stats::setNames(expected_v, outs[[length(outs)]]) else expected_v
  if (is.null(metric) && !length(mapping))
    cli::cli_abort("{.fn gepa} needs the right answers: a column named like the output ({.field {unname(outs)}}), or {.arg expected}")
  budget <- as.integer(budget); minibatch <- max(1L, as.integer(minibatch))
  if (is.null(selection)) {
    if (nrow(data) < 2L) cli::cli_abort("{.fn gepa} needs at least 2 rows: some to learn from, some to choose with")
    order <- withr::with_seed(seed, sample.int(nrow(data)))
    half <- nrow(data) %/% 2L
    select_rows <- data[order[seq_len(half)], , drop = FALSE]
    feed_rows <- data[order[-seq_len(half)], , drop = FALSE]
  } else {
    select_rows <- selection
    feed_rows <- data
  }
  run_id <- new_id()
  inputs <- names(core$definition$inputs)
  pred_col <- function(col) if (core$single) pred_names(core)("result") else paste0(".pred_", names(outs)[outs == col])
  fb <- feedback %||% default_feedback(mapping, core$single, pred_col)
  written <- ai_instructions(fn)
  fields <- fields_text(lmcc::signature_to_list(signature_of(core, effective(core$own)))$fields)
  reflect <- meta_fn(new_instruction ~ fields + instruction + cases + tried, REFLECT_TEXT, "_reflect", teacher, core$own, run_id)
  combine <- meta_fn(new_instruction ~ fields + first + second, COMBINE_TEXT, "_combine", teacher, core$own, run_id)
  s <- new.env()
  s$calls <- 0L; s$reflections <- 0L; s$trials <- list(); s$memo <- new.env(); s$order <- integer(0)

  # score, prediction and error of each row for an instruction; each (instruction, row) runs once
  scores_of <- function(text, rows, set, idx) {
    key <- function(i) paste(set, i, text, sep = "\r")
    todo <- idx[!vapply(idx, function(i) exists(key(i), envir = s$memo, inherits = FALSE), NA)]
    if (length(todo)) {
      candidate <- if (identical(text, written)) fn else with_instructions(fn, text)
      ev <- do.call(evaluate, list(candidate, rows[todo, , drop = FALSE], expected = expected_v, metric = metric,
                                   caller = list(optimization = run_id)))
      s$calls <- s$calls + length(todo)
      table <- ev$rows
      for (j in seq_along(todo)) {
        pred <- lapply(stats::setNames(nm = unname(outs)), function(col) element(table[[pred_col(col)]], j))
        err <- table$.error[[j]]
        assign(key(todo[[j]]), list(score = table[[ev$metrics[[1L]]]][[j]], pred = pred,
                                    error = if (is.na(err)) NULL else err), envir = s$memo)
      }
    }
    lapply(idx, function(i) get(key(i), envir = s$memo, inherits = FALSE))
  }
  batch_sum <- function(text, batch) sum(vapply(scores_of(text, feed_rows, "feed", batch), function(r) r$score, 0))
  score_select <- function(cand) {
    cand$scores <- vapply(scores_of(cand$instruction, select_rows, "select", seq_len(nrow(select_rows))), function(r) r$score, 0)
    cand
  }
  record <- function(cand, k, before, after, note) {
    s$trials[[length(s$trials) + 1L]] <- list(candidate = k %||% NA_integer_, kind = cand$kind, parents = list(cand$parents),
      minibatch_parent = before %||% NA_real_, minibatch = after %||% NA_real_,
      score = if (is.null(k)) NA_real_ else mean(cand$scores), length = nchar(cand$instruction), note = note,
      calls = s$calls, instruction = cand$instruction)
  }
  next_batch <- function() {
    batch <- integer(0)
    while (length(batch) < min(minibatch, nrow(feed_rows))) {
      if (!length(s$order)) s$order <- withr::with_seed(seed + length(s$trials), sample.int(nrow(feed_rows)))
      i <- s$order[[1L]]; s$order <- s$order[-1L]
      if (!i %in% batch) batch <- c(batch, i)
    }
    batch
  }
  candidate <- function(text, parents, kind) list(instruction = text, parents = parents, kind = kind, scores = numeric(0), tried = character(0))
  pool <- list(score_select(candidate(written, integer(0), "written")))
  record(pool[[1L]], 1L, NULL, NULL, "the written instruction")
  admit <- function(child, before, after) {
    if (s$calls + nrow(select_rows) > budget) { record(child, NULL, before, after, "better on the minibatch; no budget left to score it"); return(pool) }
    pool[[length(pool) + 1L]] <- score_select(child)
    record(pool[[length(pool)]], length(pool), before, after, "joined the pool")
    pool
  }
  new_text <- function(text) is.character(text) && length(text) == 1L && !is.na(text) && nzchar(trim_white(text)) &&
    !trim_white(text) %in% vapply(pool, function(c) c$instruction, "")

  # right on every feedback row, already run: no mistake left to learn from
  solved <- function(cand) all(vapply(seq_len(nrow(feed_rows)), function(i) {
    key <- paste("feed", i, cand$instruction, sep = "\r")
    exists(key, envir = s$memo, inherits = FALSE) && get(key, envir = s$memo)$score >= 1
  }, NA))
  step <- 0L
  while (s$calls + 2L * minibatch <= budget && step < budget) {         # steps: a bound when rows are cached
    step <- step + 1L
    front <- frontier(lapply(pool, function(c) c$scores))
    if (all(vapply(pool[as.integer(names(front))], solved, NA))) {
      record(pool[[1L]], NULL, NULL, NULL, "right on every feedback row: no mistake left to learn from")
      break
    }
    pair <- if (step %% 4L == 0L) best_pair(lapply(pool, function(c) c$scores), front) else NULL
    if (!is.null(pair)) {
      text <- trim_white(combine(fields, pool[[pair[[1L]]]]$instruction, pool[[pair[[2L]]]]$instruction))
      s$reflections <- s$reflections + 1L
      child <- candidate(text, pair, "combine")
      if (!new_text(text)) { record(child, NULL, NULL, NULL, "no new instruction"); next }
      batch <- next_batch()
      before <- max(batch_sum(pool[[pair[[1L]]]]$instruction, batch), batch_sum(pool[[pair[[2L]]]]$instruction, batch))
      after <- batch_sum(text, batch)
      if (after >= before) pool <- admit(child, before, after)
      else record(child, NULL, before, after, "worse on the minibatch than its better parent")
      next
    }
    ks <- as.integer(names(front))
    k <- if (length(ks) == 1L) ks else withr::with_seed(seed + step, sample(ks, 1L, prob = front))
    parent <- pool[[k]]
    batch <- next_batch()
    results <- scores_of(parent$instruction, feed_rows, "feed", batch)
    before <- sum(vapply(results, function(r) r$score, 0))
    if (all(vapply(results, function(r) r$score >= 1, NA))) next           # nothing to learn from these rows
    cases <- cases_text(feed_rows, batch, results, inputs, unname(outs), core$single, fb)
    tried <- utils::tail(parent$tried, 3L)
    text <- trim_white(reflect(fields, parent$instruction, cases,
      if (length(tried)) paste(sprintf("Tried %d:\n%s", seq_along(tried), tried), collapse = "\n\n") else "(none)"))
    s$reflections <- s$reflections + 1L
    child <- candidate(text, k, "reflect")
    if (!new_text(text)) { record(child, NULL, before, NULL, "no new instruction"); next }
    if (copies_an_input(text, feed_rows, inputs)) {
      pool[[k]]$tried <- c(parent$tried, paste0(text, "\n(dropped: it copied an input instead of stating a rule)"))
      record(child, NULL, before, NULL, "copied an input: dropped")
      next
    }
    after <- batch_sum(text, batch)
    if (after > before) pool <- admit(child, before, after)
    else {
      pool[[k]]$tried <- c(parent$tried, text)
      record(child, NULL, before, after, "not better on the minibatch")
    }
  }
  means <- vapply(pool, function(c) mean(c$scores), 0)
  lengths <- vapply(pool, function(c) nchar(c$instruction), 0)
  best <- order(-means, lengths, seq_along(pool))[[1L]]                  # ties go to the shorter instruction
  trials <- vctrs::vec_rbind(!!!lapply(s$trials, tibble::as_tibble))
  trials$chosen <- !is.na(trials$candidate) & trials$candidate == best
  attr(trials, "calls") <- s$calls
  attr(trials, "reflections") <- s$reflections
  out <- if (best == 1L) fn else with_instructions(fn, pool[[best]]$instruction)
  core <- core_of(out)
  core$trials <- trials
  make_fn(core)
}

#' The search that improved a function
#'
#' Every instruction [gepa()] tried: `candidate` (its number in the pool,
#' `NA` when it did not join), `kind` (`"written"`, `"reflect"`,
#' `"combine"`), `parents`, its sum on the minibatch next to its parent's,
#' `score` (its mean on the choosing rows: optimistic for the one chosen),
#' `length` (characters), `note`, `calls` so far, `instruction`, and
#' `chosen`. Attributes `calls` and `reflections` count the function's calls
#' and the instructions written.
#' @param fn An AI function returned by [gepa()].
#' @return A tibble, or `NULL` for a function no search made.
#' @export
ai_trials <- function(fn) core_of(fn)$trials

# ---------------------------------------------------------------- parts (the same in Python)

# The candidates on the Pareto frontier (best on at least one row, dominated
# by none), named by position, with how many rows each is best on.
frontier <- function(scores) {
  m <- do.call(rbind, scores)                            # candidates x rows
  wins <- integer(nrow(m))
  for (r in seq_len(ncol(m))) wins <- wins + as.integer(m[, r] == max(m[, r]))
  on <- which(wins > 0L)
  dominated <- vapply(on, function(a) any(vapply(setdiff(on, a), function(b)
    all(m[b, ] >= m[a, ]) && any(m[b, ] > m[a, ]), NA)), NA)
  keep <- on[!dominated]
  stats::setNames(wins[keep], keep)
}

# Two frontier candidates that each win rows the other loses: the pair with
# the most such rows on its weaker side.
best_pair <- function(scores, front) {
  ks <- as.integer(names(front)); best <- 0L; pair <- NULL
  for (x in seq_along(ks)) for (y in seq_along(ks)) if (x < y) {
    a <- scores[[ks[[x]]]]; b <- scores[[ks[[y]]]]
    w <- min(sum(a > b), sum(b > a))
    if (w > best) { best <- w; pair <- c(ks[[x]], ks[[y]]) }
  }
  pair
}

answer_text <- function(v) {
  if (is.factor(v)) v <- as.character(v)
  if (is.character(v) && length(v) == 1L) return(v)
  lmcc::json_text(plain_json(v))
}

default_feedback <- function(mapping, single, pred_col) {
  function(row, prediction, error) {
    if (!is.null(error)) return(paste("the call failed:", error))
    wrong <- names(mapping)[!vapply(names(mapping), function(k) same_value(row[[mapping[[k]]]], prediction[[k]]), NA)]
    if (!length(wrong)) return("right")
    paste0("wrong: ", paste(vapply(wrong, function(k)
      sprintf("the right %s is %s", if (single) "answer" else k, answer_text(row[[mapping[[k]]]])), ""), collapse = "; "))
  }
}

cases_text <- function(rows, batch, results, inputs, outputs, single, feedback) {
  lines <- character(0)
  for (n in seq_along(batch)) {
    row <- lapply(as.list(rows[batch[[n]], , drop = FALSE]), element, i = 1L)
    r <- results[[n]]
    lines <- c(lines, sprintf("Case %d", n),
               vapply(intersect(inputs, names(row)), function(k) sprintf("  %s: %s", k, answer_text(row[[k]])), ""))
    lines <- c(lines, if (!is.null(r$error)) "  answer given: (none)" else
      vapply(outputs, function(k) sprintf("  answer given%s: %s", if (single) "" else paste0(" ", k), answer_text(r$pred[[k]])), ""))
    lines <- c(lines, sprintf("  score: %s", format(r$score)),
               sprintf("  feedback: %s", feedback(row, r$pred, r$error)))
  }
  paste(lines, collapse = "\n")
}

# A JSON Schema shape in words, for the model writing instructions.
shape_words <- function(shape) {
  opts <- shape$anyOf %||% shape$oneOf
  if (!is.null(opts)) {
    kept <- Filter(function(o) !identical(o$type, "null"), opts)
    words <- if (length(kept)) paste(vapply(kept, shape_words, ""), collapse = " or ") else "nothing"
    return(paste0(words, if (length(kept) < length(opts)) ", or nothing" else ""))
  }
  if (!is.null(shape$enum)) return(paste("one of", paste(vapply(shape$enum, answer_text, ""), collapse = ", ")))
  switch(shape$type %||% "",
    array = paste("a list of", shape_words(shape$items %||% list())),
    object = if (length(shape$properties)) paste("a record of", paste(sprintf("%s (%s)", names(shape$properties),
               vapply(shape$properties, shape_words, "")), collapse = ", ")) else "an object",
    string = "text", integer = "a whole number", number = "a number", boolean = "true or false", "a value")
}

fields_text <- function(fields) {
  line <- function(f) paste0("- ", f$name, ": ", shape_words(f$shape), if (!is.null(f$desc)) paste0(". ", f$desc) else "")
  plain <- Filter(function(f) (f$purpose %||% "plain") == "plain", fields)
  ins <- vapply(Filter(function(f) f$direction == "input", plain), line, "")
  outs <- vapply(Filter(function(f) f$direction == "output", plain), line, "")
  paste(c("Inputs:", ins, "Outputs:", outs), collapse = "\n")
}

# Does the instruction quote an input of 30 characters or more, verbatim
# (case and spacing ignored)? Such a proposal memorises cases.
copies_an_input <- function(text, rows, inputs, at_least = 30L) {
  norm <- function(x) casefold_full(gsub("\\s+", " ", trim_white(x)))
  text <- norm(text)
  for (k in intersect(inputs, names(rows))) for (v in rows[[k]]) {
    if (is.character(v) && !is.na(v)) {
      v <- norm(v)
      if (nchar(v) >= at_least && grepl(v, text, fixed = TRUE)) return(TRUE)
    }
  }
  FALSE
}
