# GEPA, offline: a fake model answers by rules, and a fake teacher writes instructions.

GOOD <- "Use the labels exactly: booking, cancelation, information."
intents <- tibble::tibble(
  query = c("I need to reserve a room.", "How do I get there?", "Cancel my reservation.", "Book me a suite.",
            "Please book a table for two.", "What time is breakfast?", "I want to cancel tonight.", "Is there parking?"),
  intent = c("booking", "information", "cancelation", "booking", "booking", "information", "cancelation", "information"))

label_of <- function(q) if (grepl("reserve|book", q, ignore.case = TRUE)) "booking" else if (grepl("cancel", q, ignore.case = TRUE)) "cancelation" else "information"
system_of <- function(req) lmcc::lm15_plain(lm15::as_dict(req))$system %||% ""
query_of <- function(req) sub(".*<query>\n(.*?)\n</query>.*", "\\1", last_text(req))
reflecting <- function(req) grepl("You improve the instruction", system_of(req), fixed = TRUE)

classifier <- function(router, ...) ai(intent ~ query, "Classify the user's intent.",
  intent = choice("booking", "cancelation", "information"), .lm = "gpt-4.1-mini", .router = router, .log_calls = FALSE, ...)

test_that("gepa rewrites the instruction from its mistakes, never showing the choosing rows", {
  r <- fake_router(responder = function(req, i) {
    if (reflecting(req)) return(sprintf("<result>\n%s\n</result>", GOOD))
    good <- grepl("Use the labels exactly", system_of(req), fixed = TRUE)
    sprintf("<result>\n%s\n</result>", if (good) label_of(query_of(req)) else "information")
  })
  folder <- withr::local_tempdir()
  f <- update(classifier(r), log_calls = folder)
  better <- gepa(f, intents, budget = 60, seed = 1)
  expect_identical(ai_instructions(better), GOOD)
  expect_false(identical(ai_version(better), ai_version(f)))
  shown <- paste(vapply(Filter(reflecting, r$env$requests), last_text, ""), collapse = "\n")
  expect_match(shown, "wrong: the right answer is", fixed = TRUE)
  expect_match(shown, "- result: one of booking, cancelation, information", fixed = TRUE)
  order <- withr::with_seed(1, sample.int(nrow(intents)))
  choosing <- intents$query[order[1:4]]
  expect_false(any(vapply(choosing, function(q) grepl(q, shown, fixed = TRUE), NA)))
  t <- ai_trials(better)
  expect_identical(t$kind[t$chosen], "reflect")
  expect_identical(t$score[t$chosen], 1)
  expect_lte(attr(t, "calls"), 60L)
  log <- calls(folder = folder)
  expect_true(all(vapply(log$caller, function(c) !is.null(c$optimization), NA)))   # marked as part of it
  expect_true("_reflect" %in% log$name)
})

test_that("gepa runs a row once per instruction, and keeps the written one when nothing beats it", {
  seen <- character(0)
  r <- fake_router(responder = function(req, i) {
    if (reflecting(req)) return("<result>\nStill vague.\n</result>")
    seen <<- c(seen, paste(system_of(req), query_of(req)))
    "<result>\ninformation\n</result>"
  })
  f <- classifier(r)
  kept <- gepa(f, intents, budget = 40)
  expect_identical(anyDuplicated(seen), 0L)
  expect_identical(length(seen), attr(ai_trials(kept), "calls"))
  expect_identical(ai_version(kept), ai_version(f))
})

test_that("a proposal that copies an input is dropped, and the next reflection is told", {
  rows <- tibble::tibble(
    query = c(sprintf("Hello there, I would like to know about option number %d please.", 1:6),
              sprintf("Please book the room number %d for the whole of next week.", 1:6)),
    intent = rep(c("information", "booking"), each = 6))
  said <- character(0)
  r <- fake_router(responder = function(req, i) {
    if (reflecting(req)) {
      said <<- c(said, last_text(req))
      q <- sub(".*\n  query: ([^\n]*).*", "\\1", last_text(req))
      return(sprintf("<result>\nIf the message says '%s', answer booking.\n</result>", q))
    }
    "<result>\ninformation\n</result>"
  })
  kept <- gepa(classifier(r), rows, budget = 40)
  expect_true("copied an input: dropped" %in% ai_trials(kept)$note)
  expect_true(any(grepl("dropped: it copied an input", said[-1L], fixed = TRUE)))
  expect_identical(ai_instructions(kept), ai_instructions(classifier(r)))
})

test_that("the frontier keeps candidates best somewhere and drops the dominated; pairs win different rows", {
  scores <- list(c(1, 0), c(0, 1), c(1, 1))
  expect_identical(frontier(scores), c(`3` = 2L))
  expect_identical(best_pair(scores[1:2], frontier(scores[1:2])), c(1L, 2L))
  expect_null(best_pair(scores, frontier(scores)))
})

test_that("fields are described for the teacher in words, with their types", {
  f <- ai(team ~ message + n + tags, "Which team?", team = choice("a", "b"), n = optional(integer()),
          tags = vctrs::list_of(.ptype = character()), message = "the customer's words")
  core <- core_of(f)
  expect_identical(fields_text(lmcc::signature_to_list(signature_of(core, effective(core$own)))$fields),
    "Inputs:\n- message: text. the customer's words\n- n: a whole number, or nothing\n- tags: a list of text\nOutputs:\n- result: one of a, b")
})

test_that("method = \"gepa\" learns the instruction when a tidymodels model is fitted", {
  skip_if_not_installed("parsnip")
  r <- fake_router(responder = function(req, i) {
    if (reflecting(req)) return(sprintf("<result>\n%s\n</result>", GOOD))
    good <- grepl("Use the labels exactly", system_of(req), fixed = TRUE)
    sprintf("<result>\n%s\n</result>", if (good) label_of(query_of(req)) else "information")
  })
  train <- dplyr::mutate(intents, intent = factor(intent))
  spec <- parsnip::set_engine(ai_model("classification", "Classify the user's intent."), "functai",
                              method = "gepa", budget = 60, lm = "gpt-4.1-mini", router = r, log_calls = FALSE)
  fitted <- parsnip::fit(spec, intent ~ query, data = train)
  fn <- parsnip::extract_fit_engine(fitted)
  expect_identical(ai_instructions(fn), GOOD)
  expect_identical(as.character(predict(fitted, train)$.pred_class), as.character(train$intent))
})
