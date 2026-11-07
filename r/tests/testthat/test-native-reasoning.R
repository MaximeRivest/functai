# A reasoning output on a model that thinks natively (contract/functions.md,
# "Reasoning"): the request asks for the thinking and its summary, and the
# function's own reasoning setting chooses the effort.

thinker <- function() fake_router(responder = function(req, i) {
  parts <- list(lm15::thinking_part("Twelve pens leave 14 for two notebooks."), lm15::text_part("<result>\n12\n</result>"))
  lm15::response(req$model, lm15::message_assistant(parts), "stop", usage = lm15::usage(input_tokens = 10L, output_tokens = 5L, total_tokens = 15L))
}, provider = "anthropic")
sent_reasoning <- function(r) lmcc::lm15_plain(lm15::as_dict(r$env$requests[[length(r$env$requests)]]))$config$reasoning

test_that("Claude 5 thinks natively", {
  for (model in c("claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5", "claude-opus-5", "claude-fable-5-1", "claude-mythos-5", "claude-opus-4-8")) {
    caps <- model_capabilities("anthropic", model)
    expect_true(caps$native_reasoning && !caps$assistant_prefill, label = model)
  }
})

test_that("the thinking and its summary are asked for; the function's own setting chooses the effort", {
  r <- thinker()
  solve <- ai(result ~ puzzle, "Solve the puzzle.", .lm = "anthropic:claude-opus-5-5", .router = r, .module = "cot")
  expect_equal(solve("x"), "12")
  expect_identical(sent_reasoning(r), list(effort = "medium", summary = "auto"))
  cases <- list(list(lm15::reasoning("high"), list(effort = "high", summary = "auto")),
                list(lm15::reasoning("low", summary = "detailed"), list(effort = "low", summary = "detailed")),
                list(lm15::reasoning("off"), list(effort = "off")))
  for (case in cases) {
    f <- update(solve, config = list(reasoning = case[[1L]]))
    f("x")
    expect_identical(sent_reasoning(r), case[[2L]])
  }
})
