# The contract's cases of stages 1.2 to 5 and plugins (contract/README.md,
# "Which cases each language passes"): views, the reply cache's key,
# conversations, tools, context, plugins and baked examples.

stage_cases <- function(folder) {
  files <- sort(list.files(file.path(contract_root(), "cases", folder), pattern = "\\.json$", full.names = TRUE), method = "radix")
  stats::setNames(lapply(files, read_json_file), sub("\\.json$", "", basename(files)))
}

for (name in names(stage_cases("views"))) {
  test_that(paste("views case", name), {
    c <- stage_cases("views")[[name]]
    got <- outside_view(c$events, answer_from = c$answer_from)
    expect_identical(plain(lapply(got, unclass)), plain(c$expect$events))
  })
}

for (name in names(stage_cases("replies"))) {
  test_that(paste("replies case", name), {
    c <- stage_cases("replies")[[name]]
    expect_identical(reply_key(c$request, c$replicate), c$expect$key)
  })
}

# ---------------------------------------------------------------- the reply cache (stage 1.2)

test_that("a reply is kept once read, and the same request is answered from it", {
  f <- ai(reply ~ message, "Answer.")
  r <- fake_router(responder = function(request, i) sprintf("<result>\nanswer %d\n</result>", i))
  folder <- withr::local_tempdir()
  db <- file.path(withr::local_tempdir(), "replies.sqlite")
  with_ai_config(lm = "gpt-4.1-mini", router = r, cache_replies = db, log_calls = folder, {
    expect_identical(f(c("a", "b", "a")), c("answer 1", "answer 2", "answer 1"))   # one flight per key: "a" asked once
    expect_identical(f("a"), "answer 1")
    expect_identical(with_ai_config(f("a"), replicate = 1L), "answer 3")          # the second independent answer
  })
  expect_length(r$env$requests, 3L)
  recs <- log_lines(folder)
  cached <- vapply(recs, function(x) isTRUE(x$exchanges[[1L]]$cached), NA)
  expect_identical(sum(cached), 2L)
  expect_equal(Filter(function(x) isTRUE(x$exchanges[[1L]]$cached), recs)[[1L]]$exchanges[[1L]]$seconds, 0)
  # another session reads the same file
  clear_replies()
  r2 <- fake_router(list("<result>\nnew\n</result>"))
  expect_identical(with_ai_config(f("b"), lm = "gpt-4.1-mini", router = r2, cache_replies = db), "answer 2")
  expect_length(r2$env$requests, 0L)
})

test_that("an unreadable reply is never kept; a call whose log drops a field is never kept on disk", {
  f <- ai(n ~ text, "Count.", n = integer())
  r <- fake_router(list("<result>\nmany\n</result>", "<result>\n3\n</result>", "<result>\n4\n</result>"))
  db <- file.path(withr::local_tempdir(), "replies.sqlite")
  with_ai_config(lm = "gpt-4.1-mini", router = r, cache_replies = db, expect_identical(f("abc"), 3L))
  store <- reply_store(db)
  expect_identical(store$size(), 1L)
  g <- ai(n ~ text, "Count.", n = integer(), .log_content = FALSE)
  clear_replies()
  with_ai_config(lm = "gpt-4.1-mini", router = r, cache_replies = db, expect_identical(g("xyz"), 4L))
  expect_identical(store$size(), 1L)
})
