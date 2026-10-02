# Stage 1.1 (design/09): what the shared cases do not reach from R.

test_that("a rating with no person named is made under the account; two on one call are both kept, disputed", {
  folder <- withr::local_tempdir()
  r <- fake_router(responder = function(req, i) "<result>\nbilling\n</result>")
  f <- ai(team ~ message, "Which team?", .router = r, .lm = "gpt-4.1-mini", .log_calls = folder)
  p <- predict(f, tibble::tibble(message = "charged twice"))
  first <- rate(p$.call, "right", folder = folder)
  rate(p$.call, "wrong", answer = "shipping", folder = folder)
  expect_true(is.na(first$by) && !is.na(first$account))
  expect_identical(rate(p$.call, "right", by = "ana", folder = folder)$by, "ana")
  rows <- rated(f, folder = folder)
  expect_identical(nrow(rows), 1L)
  expect_true(rows$disputed)
})

test_that("a record names the lmcc and lm15 that made it; a re-ask has the hash of what it sent", {
  folder <- withr::local_tempdir()
  r <- fake_router(list("no tags here", "<result>\nok\n</result>"))
  f <- ai(answer ~ message, "x", .router = r, .lm = "gpt-4.1-mini", .log_calls = folder)
  expect_identical(f("Hi"), "ok")
  rec <- log_lines(folder)[[1L]]
  expect_identical(rec$process$lmcc, as.character(utils::packageVersion("lmcc")))
  expect_true(is_str(rec$process$lm15))
  expect_length(rec$exchanges, 2L)
  expect_false(identical(rec$exchanges[[1L]]$request_hash, rec$exchanges[[2L]]$request_hash))
})

test_that("a default written as a formula is kept by its code in a saved folder, and the loaded function has its version", {
  day <- function() "2026-09-30"
  f <- ai(answer ~ notes + when, "Plan the day.", when = defaults_to(~ day()), .lm = "gpt-4.1-mini")
  dir <- withr::local_tempdir()
  write_ai(f, dir)
  manifest <- lmcc::parse_json(paste(readLines(file.path(dir, "functai.json")), collapse = "\n"))
  node <- manifest$nodes[[manifest$entry]]
  expect_identical(node$defaults$when$code, "day()")
  expect_identical(ai_version(read_ai(dir)), ai_version(f))
})
