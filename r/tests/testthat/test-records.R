# What a call's record keeps (contract/calls.md, "Content"): a field's rule
# is its own, whatever R calls the answer; the fields FunctAI adds are in the
# record like any other.

test_that("a field's name is that field: a function named like its input keeps the input's rule through save, load and update", {
  folder <- withr::local_tempdir()
  f <- ai(answer ~ secret, "Answer.", .name = "secret", .lm = "gpt-4.1-mini", .log_content = c(secret = FALSE))
  dir <- withr::local_tempdir()
  write_ai(f, dir)
  g <- read_ai(dir)
  expect_identical(plain(core_of(g)$own$log_content), '{"secret":false}')
  g <- update(g, router = fake_router(responder = function(req, i) "<result>\nok\n</result>"), log_calls = folder)
  expect_identical(plain(core_of(g)$own$log_content), '{"secret":false}')    # a routing update renames nothing
  g("PRIVATE_MARKER")
  rec <- log_lines(folder)[[1L]]
  expect_null(rec$inputs$secret)
  expect_identical(plain(rec$omitted), '{"inputs":["secret"],"outputs":[]}')
  expect_identical(rec$outputs$result, "ok")
  expect_false(any(grepl("PRIVATE_MARKER", readLines(list.files(folder, recursive = TRUE, full.names = TRUE)), fixed = TRUE)))
})

test_that("a host's rule names the field, never an answer column that happens to share its name", {
  folder <- withr::local_tempdir()
  f <- ai(answer ~ secret, "Answer.", .name = "secret", .lm = "gpt-4.1-mini")
  dir <- withr::local_tempdir()
  write_ai(f, dir)
  g <- update(read_ai(dir), router = fake_router(responder = function(req, i) "<result>\nok\n</result>"), log_calls = folder)
  with_ai_config(g("PRIVATE_MARKER"), log_content = c(secret = FALSE))
  rec <- log_lines(folder)[[1L]]
  expect_null(rec$inputs$secret)
  expect_identical(plain(rec$omitted), '{"inputs":["secret"],"outputs":[]}')
  # an alias that is no field's name still names the answer
  with_ai_config(g("again"), log_content = list(result = FALSE))
  expect_identical(plain(log_lines(folder)[[2L]]$omitted), '{"inputs":[],"outputs":["result"]}')
  # the name of a field FunctAI adds is that field, not the answer column named like it
  h <- ai(reasoning ~ question, "x", .module = "cot", .lm = "gpt-4.1-mini", .log_content = c(reasoning = FALSE))
  expect_identical(plain(core_of(h)$own$log_content), '{"reasoning":false}')
  k <- ai(team ~ message, "x", .lm = "gpt-4.1-mini", .log_content = c(team = FALSE))
  expect_identical(plain(core_of(k)$own$log_content), '{"result":false}')
})

test_that("a tool-using call's record holds `calls` and its size, whole or not", {
  lookup <- ai_tool(function(order) "shipped", "Look up an order.", .name = "lookup")
  replies <- function() fake_router(list(list(calls = list(list(id = "c1", name = "lookup", input = list(order = "A1")))), "<result>\nok\n</result>"))
  folder <- withr::local_tempdir()
  f <- ai(answer ~ question, "x", .tools = list(lookup), .lm = "gpt-4.1-mini", .router = replies(), .log_calls = folder)
  expect_identical(f("Where is A1?"), "ok")
  rec <- log_lines(folder)[[1L]]
  expect_true(rec$content)
  expect_identical(names(rec$outputs), c("calls", "result"))
  expect_identical(plain(rec$outputs$calls), "[]")                 # lmcc's finished turn: its last model step's
  expect_identical(names(rec$sizes$outputs), c("calls", "result"))
  expect_identical(rec$sizes$outputs$calls, 2L)
  expect_length(rec$exchanges, 2L)
  expect_null(schema_fault(rec, "call.schema.json"))
  f <- update(f, router = replies(), log_content = c(question = FALSE))
  f("Where is A1?")
  rec <- log_lines(folder)[[2L]]
  expect_false(rec$content)
  expect_identical(plain(rec$omitted), '{"inputs":["question"],"outputs":["calls"]}')
  expect_identical(names(rec$outputs), "result")
  expect_identical(names(rec$sizes$outputs), c("calls", "result"))
  expect_null(schema_fault(rec, "call.schema.json"))
})

test_that("a record not whole names what it left out in calls(), and keeps no description of a value it dropped", {
  folder <- withr::local_tempdir()
  f <- ai(answer ~ question + notes, "x", .lm = "gpt-4.1-mini", .router = fake_router(responder = function(req, i) "<result>\nok\n</result>"),
          .log_calls = folder, .log_content = c(notes = FALSE))
  f("q", "n")
  got <- calls(folder = folder)
  expect_identical(plain(got$omitted[[1L]]), '{"inputs":["notes"],"outputs":[]}')
  rec <- list(functai_call = 2L, content = TRUE, inputs = list(question = "q", notes = list(`$type` = "x", `$repr` = "y")),
              outputs = list(result = "ok"), described = list(inputs = list("notes"), outputs = list()), exchanges = list(), program = list(answer = "result"))
  kept <- kept_record(rec, list(inputs = c("question", "notes"), outputs = "result", added = character(0)), c(question = TRUE, notes = FALSE, result = TRUE))
  expect_null(kept$described)
  expect_null(kept$inputs$notes)
  rec$described$inputs <- list("question", "notes")
  kept <- kept_record(rec, list(inputs = c("question", "notes"), outputs = "result", added = character(0)), c(question = TRUE, notes = FALSE, result = TRUE))
  expect_identical(plain(kept$described), '{"inputs":["question"],"outputs":[]}')
})
