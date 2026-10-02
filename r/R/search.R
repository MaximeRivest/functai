# Searching for a better function: sets of worked examples (random_search),
# instructions a model writes with sets of worked examples
# (instruction_search), and two versions compared row by row (compare). The
# contract fixes what improving means, not its random choices.

# The mean score of a function on rows with known answers.
mean_score <- function(fn, rows, metric = NULL, ...) {
  ev <- suppressWarnings(evaluate(fn, rows, metric = metric, ...))
  t <- tidy(ev)
  if (!nrow(t)) return(-Inf)
  s <- t$estimate[[1L]]
  if (is.na(s)) -Inf else s
}

#' Try several sets of worked examples, keep the best
#'
#' `random_search()` tries no examples, labeled rows only, bootstrapped runs
#' (see [bootstrap_few_shot()]), then `candidates` bootstrapped sets made from
#' shuffled rows, each scored on `valset` (default: the same rows: the score
#' then flatters). `instruction_search()` has a model write instructions (each
#' after a tip: be concise, spell out the steps, name the common mistakes,
#' ...), pairs them with sets of worked examples, tries `trials` pairs on
#' minibatches of `valset`, and keeps the best of the `finalists` on all of
#' it. [ai_trials()] gives the search. Measure the result on rows it never
#' saw, with [evaluate()].
#' @param fn An AI function.
#' @param data Rows with known answers.
#' @param valset Rows to score on (default: `data`).
#' @param candidates How many sets (or instructions) to make.
#' @param metric `(row, prediction)` returning a score; default [exact_match()].
#' @param max_bootstrapped,max_labeled As in [bootstrap_few_shot()].
#' @param teacher The model that writes bootstrapped examples.
#' @param seed The random seed.
#' @param stop_at Stop once a candidate scores this much.
#' @return An AI function, with its search in [ai_trials()].
#' @export
random_search <- function(fn, data, valset = NULL, candidates = 8L, metric = NULL, max_bootstrapped = 4L, max_labeled = 16L,
                          teacher = NULL, seed = 0L, stop_at = NULL) {
  val <- valset %||% data
  trials <- list(); best <- list(score = -Inf, fn = fn)
  for (c in (-3L):(candidates - 1L)) {
    got <- if (c == -3L) list("zero-shot", with_demos(fn, NULL))
      else if (c == -2L) list("labeled", labeled_few_shot(fn, data, k = max_labeled, seed = seed))
      else {
        size <- if (c >= 0L) withr::with_seed(seed + c, sample.int(max(1L, max_bootstrapped), 1L)) else max_bootstrapped
        rows <- if (c >= 0L) data[withr::with_seed(seed + c, sample.int(nrow(data))), , drop = FALSE] else data
        list(if (c == -1L) "bootstrapped" else sprintf("bootstrapped (seed %d, %d examples)", c, size),
             bootstrap_few_shot(fn, rows, max_bootstrapped = size, max_labeled = max_labeled, teacher = teacher, metric = metric, seed = seed + c))
      }
    score <- mean_score(got[[2L]], val, metric)
    trials[[length(trials) + 1L]] <- tibble::tibble(candidate = got[[1L]], examples = length(ai_demos(got[[2L]])), score = score, version = ai_version(got[[2L]]))
    if (score > best$score) best <- list(score = score, fn = got[[2L]])
    if (!is.null(stop_at) && score >= stop_at) break
  }
  core <- core_of(best$fn); core$trials <- do.call(rbind, trials)
  make_fn(core)
}

INSTRUCTION_TIPS <- c("", "Be concise and direct.", "Be precise: the task is high-stakes and mistakes are costly.",
                      "Describe the expected output format exactly.", "Spell out the steps to follow before answering.",
                      "Name the common mistakes on this task and how to avoid them.",
                      "Give the model a helpful persona suited to the task.", "Consider edge cases and unusual inputs.")

proposer <- function(core, prompt_lm) {
  s <- effective(core$own)
  ai(result ~ function_name + signature + current_instruction + examples + previous_proposals + tip,
     paste("Propose a new instruction (a system prompt) for an AI function that will make it score higher on its task.",
           "Use the signature and the examples to understand the task. Make it different from the previous proposals,",
           "and follow the tip. Keep the exact input and output names. Reply with the instruction text only."),
     previous_proposals = vctrs::list_of(.ptype = character()), .name = "propose_instruction", .defined_in = "functai.meta",
     .adapter = "xml", .include_fn_name = FALSE, .module = "predict", .temperature = 1, .lm = prompt_lm %||% s$lm, .router = s$router,
     .on_error = "stop")
}

# A few rows as the proposer sees them: the function's inputs and outputs
# only (never other columns: they may be answers a person had to read).
examples_text <- function(core, rows, limit = 5L) {
  ins <- names(core$definition$inputs); cols <- columns_of(core)
  rows <- rows[seq_len(min(limit, nrow(rows))), , drop = FALSE]
  paste(vapply(seq_len(nrow(rows)), function(i) {
    inputs <- lapply(rows[intersect(ins, names(rows))], function(col) value_json(element(col, i)))
    outputs <- lapply(stats::setNames(cols[cols %in% names(rows)], names(cols)[cols %in% names(rows)]), function(k) value_json(element(rows[[k]], i)))
    lmcc::json_text(list(inputs = if (length(inputs)) inputs else lmcc::jobj(), outputs = if (length(outputs)) outputs else lmcc::jobj()))
  }, ""), collapse = "\n")
}

#' @rdname random_search
#' @param trials How many (instruction, examples) pairs to try.
#' @param minibatch How many rows each trial is scored on.
#' @param prompt_lm The model that writes instructions (default: the function's).
#' @param finalists How many of the best pairs are scored on all of `valset`.
#' @export
instruction_search <- function(fn, data, candidates = 6L, trials = 12L, minibatch = 20L, valset = NULL, prompt_lm = NULL, metric = NULL,
                               max_bootstrapped = 4L, max_labeled = 4L, seed = 0L, finalists = 3L, teacher = NULL) {
  core <- core_of(fn)
  val <- valset %||% data
  writer <- proposer(core, prompt_lm)
  texts <- list(core$state$instructions); proposals <- character(0)
  for (i in seq_len(candidates - 1L)) {
    sample <- data[withr::with_seed(seed + i, sample.int(nrow(data))), , drop = FALSE]
    tip <- INSTRUCTION_TIPS[[(i - 1L) %% length(INSTRUCTION_TIPS) + 1L]]
    text <- tryCatch(writer(core$definition$name, paste(utils::capture.output(print(fn)), collapse = "\n"), ai_instructions(fn),
                            examples_text(core, sample), list(proposals), if (nzchar(tip)) tip else "(no tip)"),
                     error = function(e) { cli::cli_warn("an instruction proposal failed: {conditionMessage(e)}"); NULL })
    if (is.null(text)) next
    text <- trim_white(text)
    if (nzchar(text) && !text %in% proposals) { proposals <- c(proposals, text); texts[[length(texts) + 1L]] <- text }
  }
  demo_sets <- list(core$state$demos)
  if (max_bootstrapped > 0L || max_labeled > 0L) for (k in seq_len(candidates - 1L)) {
    shuffled <- data[withr::with_seed(seed + k, sample.int(nrow(data))), , drop = FALSE]
    demo_sets[[length(demo_sets) + 1L]] <- ai_demos(bootstrap_few_shot(fn, shuffled, max_bootstrapped = max_bootstrapped, max_labeled = max_labeled,
                                                                       teacher = teacher, metric = metric, seed = seed + k))
  }
  make <- function(c) { k <- core; k$state$instructions <- texts[[c[[1L]]]]; k$state$demos <- demo_sets[[c[[2L]]]]; make_fn(k) }
  scored <- list(); log <- list()
  key <- function(c) paste(c, collapse = ",")
  rng_pick <- function(t, n) withr::with_seed(seed * 1000L + t, sample.int(n, 1L))
  for (t in seq_len(trials)) {
    c <- if (t == 1L) c(1L, 1L) else if (length(scored) && t > trials %/% 2L && withr::with_seed(seed + t, stats::runif(1)) < 0.6) {
      best <- as.integer(strsplit(names(scored)[[which.max(vapply(scored, mean, 0))]], ",")[[1L]])
      if (withr::with_seed(seed + 2L * t, stats::runif(1)) < 0.5) c(rng_pick(t, length(texts)), best[[2L]]) else c(best[[1L]], rng_pick(t, length(demo_sets)))
    } else c(rng_pick(t, length(texts)), rng_pick(t + 7919L, length(demo_sets)))
    batch <- if (nrow(val) <= minibatch) val else val[withr::with_seed(seed + t, sample.int(nrow(val), minibatch)), , drop = FALSE]
    s <- mean_score(make(c), batch, metric)
    scored[[key(c)]] <- c(scored[[key(c)]], s)
    log[[length(log) + 1L]] <- tibble::tibble(trial = t, instruction = c[[1L]], examples = c[[2L]], minibatch_score = s,
                                              instruction_text = texts[[c[[1L]]]] %||% NA_character_)
  }
  ranked <- names(scored)[order(-vapply(scored, mean, 0))]
  top <- utils::head(ranked, finalists)
  winner <- if (nrow(val) <= minibatch) top[[1L]] else top[[which.max(vapply(top, function(k) mean_score(make(as.integer(strsplit(k, ",")[[1L]])), val, metric), 0))]]
  out <- core_of(make(as.integer(strsplit(winner, ",")[[1L]])))
  out$trials <- do.call(rbind, log)
  make_fn(out)
}

#' Two versions compared row by row
#'
#' The difference between two evaluations of the same rows, metric by
#' metric: the mean score before and after, their difference with its 95%
#' interval (paired: each row's difference), and how many rows got better,
#' worse, or stayed the same. A difference whose interval holds 0 may be
#' noise.
#' @param before,after Evaluations ([evaluate()]) of the same rows, in the same order.
#' @return A tibble: `metric`, `before`, `after`, `diff`, `conf.low`,
#'   `conf.high`, `better`, `worse`, `same`, `n`.
#' @export
compare <- function(before, after) {
  if (!inherits(before, "functai_evaluation") || !inherits(after, "functai_evaluation")) cli::cli_abort("{.fn compare} takes two evaluations from {.fn evaluate}")
  a_rows <- augment(before); b_rows <- augment(after)
  if (nrow(a_rows) != nrow(b_rows)) cli::cli_abort("compare needs the same rows: {nrow(a_rows)} vs {nrow(b_rows)}")
  shared <- intersect(tidy(before)$metric, tidy(after)$metric)
  if (!length(shared)) cli::cli_abort("no metric in common")
  rows <- lapply(shared, function(m) {
    a <- as.numeric(a_rows[[m]]); b <- as.numeric(b_rows[[m]])
    ok <- !is.na(a) & !is.na(b)
    d <- b[ok] - a[ok]; n <- length(d)
    mean_d <- if (n) mean(d) else NA_real_
    half <- if (n >= 2L) t975(n - 1L) * stats::sd(d) / sqrt(n) else NA_real_
    tibble::tibble(metric = m, before = mean(a[ok]), after = mean(b[ok]), diff = mean_d, conf.low = mean_d - half, conf.high = mean_d + half,
                   better = sum(d > 0), worse = sum(d < 0), same = sum(d == 0), n = n)
  })
  do.call(rbind, rows)
}

#' The last requests of this session
#'
#' What the last `n` calls sent and got: `inspect_history()` as data (each
#' call's function, model, request and reply), `phistory()` printed the way a
#' person reads a conversation. Kept in memory for this session only, at most
#' the last 200 calls; nothing is written unless the call log is on.
#' @param n How many calls, newest last.
#' @return A list (`inspect_history()`); `phistory()` prints and returns it invisibly.
#' @export
inspect_history <- function(n = 1L) {
  h <- utils::tail(the$history %||% list(), n)
  lapply(h, function(x) { x$request <- plain_lm15(x$request); if (!is.null(x$response)) x$response <- plain_lm15(x$response); x })
}

#' @rdname inspect_history
#' @export
phistory <- function(n = 1L) {
  h <- inspect_history(n)
  for (x in h) {
    cat(cli::rule(sprintf("%s \u00b7 %s", x$name, x$model %||% "?")), "\n")
    req <- x$request
    if (!is.null(req$system)) cat(cli::col_grey("system: "), if (is_str(req$system)) req$system else paste(vapply(req$system, function(p) p$text %||% "", ""), collapse = ""), "\n\n", sep = "")
    for (m in req$messages %||% list()) cat(cli::style_bold(paste0(m$role, ": ")), paste(vapply(m$parts, function(p) p$text %||% sprintf("[%s]", p$type), ""), collapse = ""), "\n\n", sep = "")
    if (!is.null(x$response)) cat(cli::style_bold("reply: "), paste(vapply(x$response$message$parts %||% list(), function(p) p$text %||% sprintf("[%s]", p$type), ""), collapse = ""), "\n", sep = "")
  }
  invisible(h)
}

# Keep a call's last exchange for inspect_history() (the request as sent, the reply as read).
remember_history <- function(call) {
  ex <- Filter(function(e) !is.null(e$response) || !is.null(e$error), call$exchanges)
  if (!length(ex)) return(invisible())
  last <- ex[[length(ex)]]
  # kept as they are, and turned into JSON only when someone reads them
  the$history <- utils::tail(c(the$history, list(list(name = call$name, call = call$id, model = last$model,
                                                       request = last$request, response = last$response))), 200L)
  invisible()
}

#' Sign in to a provider
#'
#' Sign in once; every later session (in any functai language) uses it: lm15
#' keeps the sign-in in its credentials file, shared by every language.
#' Subscriptions (`"claude"`, `"chatgpt"`, `"copilot"`) open a browser, or
#' print a link or code over SSH; for an API provider (`"openai"`,
#' `"anthropic"`, `"groq"`, ...) the key is asked for, or given as `key`, and
#' saved. `ai_logins()` lists the providers this machine is signed in to (no
#' secrets shown); `ai_logout()` forgets one.
#' @param provider The provider.
#' @param key An API key to save (instead of asking).
#' @return The connection, invisibly (`ai_login()`); a tibble (`ai_logins()`).
#' @export
ai_login <- function(provider, key = NULL) {
  out <- if (is.null(key)) lm15::login(provider) else lm15::set_api_key(provider, key)
  the$router <- NULL
  invisible(out)
}

#' @rdname ai_login
#' @export
ai_logins <- function() {
  cs <- lm15::connections()
  tibble::tibble(provider = vapply(cs, function(c) as.character(c$provider %||% NA), ""), method = vapply(cs, function(c) as.character(c$method_id %||% NA), ""),
                 label = vapply(cs, function(c) as.character(c$label %||% NA), ""))
}

#' @rdname ai_login
#' @export
ai_logout <- function(provider) { out <- lm15::logout(provider); the$router <- NULL; invisible(out) }
