# Calls, offline: a fake model stands in for every provider.

mood_of <- function(router, ...) {
  settings <- list(.lm = "gpt-4.1-mini", .router = router, .log_calls = FALSE)
  more <- list(...)
  settings[names(more)] <- more
  do.call(ai, c(list("mood", "How does the customer feel about what they bought?",
    review = character(), .returns = factor(levels = c("happy", "unhappy", "mixed"))), settings))
}

guess <- function(request, i) {
  t <- last_text(request)
  sprintf("<result>\n%s\n</result>", if (grepl("Love|Great", t)) "happy" else if (grepl("late", t)) "mixed" else "unhappy")
}

test_that("a call sends the contract's layout and returns the declared type", {
  r <- fake_router(list("<result>\nunhappy\n</result>"))
  mood <- mood_of(r)
  out <- mood("Broke after a day.")
  expect_identical(out, factor("unhappy", levels = c("happy", "unhappy", "mixed")))
  req <- lmcc::lm15_plain(lm15::as_dict(r$env$requests[[1L]]))
  expect_match(req$system, "^Function: mood\n\nHow does the customer feel")
  expect_identical(message_text(r$env$requests[[1L]], 1L), "<review>\nBroke after a day.\n</review>\n")
  expect_null(req$config$stop)                                 # OpenAI's Responses API takes no stop sequences
  expect_s3_class(mood, "functai_fn")
  expect_identical(names(formals(mood)), "review")
})

test_that("functions are vectorised: one call per row, recycled, NA in gives NA out", {
  r <- fake_router(responder = guess)
  mood <- mood_of(r)
  out <- mood(c("Love it", "Broke", NA, "Arrived late"))
  expect_identical(as.character(out), c("happy", "unhappy", NA, "mixed"))
  expect_length(r$env$requests, 3L)                           # no call for the missing review
  expect_identical(mood(character(0)), factor(character(0), levels = c("happy", "unhappy", "mixed")))
})

test_that("they are columns in dplyr: mutate, and several outputs splice in as columns", {
  skip_if_not_installed("dplyr")
  r <- fake_router(responder = guess)
  mood <- mood_of(r)
  reviews <- tibble::tibble(id = 1:3, review = c("Love it", "Broke", "Arrived late"))
  got <- dplyr::mutate(reviews, mood = mood(review))
  expect_identical(as.character(got$mood), c("happy", "unhappy", "mixed"))
  expect_s3_class(got$mood, "factor")

  triage <- ai("triage", "Read the ticket.", ticket = character(),
    .outputs = list(summary = character(), minutes = described(integer(), "minutes to fix")),
    .lm = "gpt-4.1-mini", .log_calls = FALSE,
    .router = fake_router(responder = function(req, i) "<summary>\nCharged twice\n</summary>\n<minutes>\n30\n</minutes>"))
  spliced <- dplyr::mutate(tibble::tibble(ticket = c("a", "b")), triage(ticket))
  expect_identical(names(spliced), c("ticket", "summary", "minutes"))
  expect_identical(spliced$minutes, c(30L, 30L))
  expect_match(ai_instructions(triage), "Output guidance:\n- minutes: minutes to fix$")
})

test_that("records come back as tibble columns; lists as list_of", {
  person <- ai("person", "Who is described?", text = character(),
    .returns = tibble::tibble(name = character(), age = integer()), .adapter = "json",
    .lm = "gpt-4.1-mini", .log_calls = FALSE,
    .router = fake_router(responder = function(req, i) if (i == 1L) '{"result": {"name": "Ana", "age": 31}}' else '{"result": {"name": "Bo", "age": 50}}'))
  got <- person(c("Ana, 31.", "Bo, 50."))
  expect_s3_class(got, "tbl_df")
  expect_identical(got$name, c("Ana", "Bo"))
  expect_identical(got$age, c(31L, 50L))
  tags <- ai("tags", "Tags.", text = character(), .returns = vctrs::list_of(.ptype = character()),
    .lm = "gpt-4.1-mini", .log_calls = FALSE, .router = fake_router(list('<result>\n["a", "b"]\n</result>')))
  out <- tags("x")
  expect_s3_class(out, "vctrs_list_of")
  expect_identical(out[[1L]], c("a", "b"))
})

test_that("an unreadable reply is asked again once, in the contract's words", {
  r <- fake_router(list("They seem sad.", "<result>\nunhappy\n</result>"))
  expect_identical(as.character(mood_of(r)("Broke.")), "unhappy")
  expect_length(r$env$requests, 2L)
  expect_match(last_text(r$env$requests[[2L]]), "^Your reply could not be read: .*\\. Reply again, in exactly the form the instructions give\\.$")
})

test_that("a value outside its type is unreadable too; one failing row is an error, many are NA and a warning", {
  r <- fake_router(list("<result>\nfurious\n</result>", "<result>\nunhappy\n</result>"))
  expect_identical(as.character(mood_of(r)("x")), "unhappy")
  expect_error(mood_of(fake_router(list("nope", "still nope")))("x"), class = "lmcc_refusal")
  r2 <- fake_router(responder = function(req, i) if (grepl("bad", message_text(req, 1L))) "nope" else "<result>\nhappy\n</result>")
  expect_warning(out <- mood_of(r2)(c("good", "bad", "good")), "1 of 3 calls of")
  expect_identical(as.character(out), c("happy", NA, "happy"))
  expect_identical(ai_problems()$row, 2L)
  expect_error(mood_of(r2, .on_error = "stop")(c("good", "bad")), class = "lmcc_refusal")
})

test_that("tools run until the model answers", {
  orders <- c("A-1" = "stuck at the carrier")
  lookup <- ai_tool(function(order) orders[[order]], "lookup_order", "Look up an order.", order = character())
  r <- fake_router(list(list(calls = list(list(id = "c1", name = "lookup_order", input = list(order = "A-1")))),
                        "<result>\nIt is stuck at the carrier.\n</result>"))
  helper <- ai("helper", "Help.", question = character(), .tools = list(lookup), .lm = "gpt-4.1-mini", .router = r, .log_calls = FALSE)
  expect_identical(helper("Where is A-1?"), "It is stuck at the carrier.")
  second <- lmcc::lm15_plain(lm15::as_dict(r$env$requests[[2L]]))
  expect_match(lmcc::json_text(second$messages), "stuck at the carrier")
  expect_identical(second$tools[[1L]]$name, "lookup_order")
})

test_that("the version follows what is sent, not where it runs", {
  mood <- mood_of(NULL)
  v <- ai_version(mood)
  expect_match(v, "^sha256:[0-9a-f]{64}$")
  expect_identical(ai_version(update(mood, lm = "claude-haiku-4-5", temperature = 0.3)), v)
  expect_false(identical(ai_version(update(mood, adapter = "json")), v))
  expect_false(identical(ai_version(update(mood, module = "cot")), v))
  taught <- with_demos(mood, tibble::tibble(review = "Great", result = "happy"))
  expect_false(identical(ai_version(taught), v))
  expect_identical(ai_version(with_demos(taught, NULL)), v)
  expect_false(identical(ai_version(with_instructions(mood, "Say how they feel.")), v))
})

# ---------------------------------------------------------------- predict, augment, evaluate

reviews <- tibble::tibble(review = c("Broke in a day", "Love it", "Good but late", "Terrible"),
                          result = c("unhappy", "happy", "mixed", "unhappy"))

test_that("predict and augment: tidymodels' columns, with call ids", {
  mood <- mood_of(fake_router(responder = guess))
  p <- predict(mood, reviews)
  expect_identical(names(p), c(".pred_class", ".call", ".error"))     # a choice is a class, as in tidymodels
  expect_identical(as.character(p$.pred_class), c("unhappy", "happy", "mixed", "unhappy"))
  a <- augment(mood, reviews)
  expect_identical(names(a), c("review", "result", ".pred_class", ".call", ".error"))
  n <- ai("n", "Count.", text = character(), .returns = integer(), .lm = "gpt-4.1-mini", .log_calls = FALSE,
          .router = fake_router(list("<result>\n3\n</result>")))
  expect_identical(names(predict(n, tibble::tibble(text = "x"))), c(".pred", ".call", ".error"))
  expect_error(predict(mood, tibble::tibble(x = 1)), "no column for input")
})

test_that("evaluate: the score, its range, broom's tidy, glance and augment", {
  mood <- mood_of(fake_router(responder = function(req, i) sprintf("<result>\n%s\n</result>", if (grepl("Love", last_text(req))) "happy" else "unhappy")))
  ev <- evaluate(mood, reviews)
  expect_identical(ev$score, 0.75)
  t <- tidy(ev)
  expect_identical(names(t), c("metric", "estimate", "conf.low", "conf.high", "n", "failed"))
  expect_true(t$conf.low < 0.75 && t$conf.high > 0.75)
  expect_identical(nrow(glance(ev)), 1L)
  expect_identical(augment(ev)$exact_match, c(1, 1, 0, 1))
  expect_output(print(ev), "exact_match: 0.75")
  truth <- dplyr::rename(reviews, feeling = result)
  expect_identical(evaluate(mood, truth, expected = feeling)$score, 0.75)
  failing <- mood_of(fake_router(responder = function(req, i) "?"), .retries = 0L)
  bad <- suppressWarnings(evaluate(failing, reviews[1:2, ]))
  expect_identical(bad$score, 0)
  expect_identical(tidy(bad)$failed, 2L)
})

test_that("improving adds worked examples: a new version, the examples in the request", {
  r <- fake_router(responder = guess)
  mood <- mood_of(r)
  taught <- labeled_few_shot(mood, reviews, k = 2L)
  expect_length(ai_demos(taught), 2L)
  expect_length(ai_demos(mood), 0L)
  expect_length(ai_render(taught, "new")$messages, 5L)
  boot <- bootstrap_few_shot(mood, reviews, max_bootstrapped = 2L, max_labeled = 3L)
  demos <- ai_demos(boot)
  expect_identical(sum(vapply(demos, function(d) !is.null(d$steps), NA)), 2L)
  expect_length(demos, 3L)
  expect_length(ai_render(boot, "new")$messages, 7L)
})

# ---------------------------------------------------------------- the call log

test_that("every call is a line in the log; ratings make rows with known answers", {
  folder <- withr::local_tempdir()
  mood <- update(mood_of(fake_router(responder = guess)), log_calls = folder)
  p <- with_ai_config(predict(mood, reviews[1:2, ]), caller = list(kind = "test", user = "ana"))
  log <- calls(mood, folder = folder)
  expect_identical(nrow(log), 2L)
  expect_setequal(log$id, p$.call)
  expect_identical(unique(log$version), ai_version(mood))
  expect_identical(unique(log$language), "r")
  line <- log_lines(folder)[[1L]]
  expect_identical(line$program$signature, ai_signature_id(mood))
  expect_identical(line$program$module, "__main__")
  expect_identical(line$caller$user, "ana")
  expect_identical(line$sizes$outputs$result, 9L)
  rate(p$.call, c("right", "wrong"), answer = list(NULL, "mixed"), by = "ben", folder = folder)
  rows <- rated(mood, folder = folder)
  expect_identical(nrow(rows), 2L)
  expect_s3_class(rows$result, "factor")
  expect_setequal(as.character(rows$result), c("unhappy", "mixed"))
  expect_identical(attr(rows, "left_out"), list(other_signature = 0L, no_content = 0L, no_answer = 0L))
})

test_that("log_content = FALSE keeps sizes and tokens, never values; a failed call records its error", {
  folder <- withr::local_tempdir()
  mood <- mood_of(fake_router(list("nope", "still nope")), .log_calls = folder, .log_content = FALSE)
  expect_error(mood("secret"))
  line <- log_lines(folder)[[1L]]
  expect_false(line$content)
  expect_null(line$inputs)
  expect_identical(line$sizes$inputs$review, 8L)
  expect_identical(line$error$type, "Refusal")
  expect_null(line$error$message)
  expect_length(line$exchanges, 2L)
})

# ---------------------------------------------------------------- saved

test_that("a function written here loads back with the same version and requests", {
  dir <- withr::local_tempdir()
  mood <- with_demos(mood_of(NULL, .temperature = 0, .defined_in = "shop"), tibble::tibble(review = "Broke", result = "unhappy"))
  write_ai(mood, dir)
  m <- read_json_file(file.path(dir, "functai.json"))
  expect_identical(m$language, "r")
  expect_identical(m$entry, "shop:mood")
  again <- read_ai(dir)
  expect_identical(ai_version(again), ai_version(mood))
  expect_identical(ai_signature_id(again), ai_signature_id(mood))
  expect_identical(lmcc::canonical_json(ai_render(update(again, router = fake_router()), "x")),
                   lmcc::canonical_json(ai_render(update(mood, router = fake_router()), "x")))
})

test_that("record() takes described and optional fields; a null comes back NA", {
  ref <- ai("order_ref", "The order.", message = character(),
    .returns = record(order = optional(character()), days = described(integer(), "whole days")),
    .lm = "gpt-4.1-mini", .log_calls = FALSE,
    .router = fake_router(list("<result>\n{\"order\": null, \"days\": 3}\n</result>")))
  expect_match(ai_instructions(ref), "Function: order_ref")
  shape <- ai_signature(ref)$fields[[2L]]$shape
  expect_identical(lmcc::canonical_json(shape$properties$order), '{"anyOf":[{"type":"string"},{"type":"null"}]}')
  expect_identical(shape$properties$days$description, "whole days")
  out <- ref("x")
  expect_identical(out$order, NA_character_)
  expect_identical(out$days, 3L)
})
