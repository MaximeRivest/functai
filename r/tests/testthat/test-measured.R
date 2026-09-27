# Probabilities a provider measures itself (TypeSafe's Jev answers a choice
# with a distribution): read, never made up, logged as `confidence`.

jev_router <- function(dist = c(billing = 0.7, shipping = 0.2, product = 0.1, account = 0)) {
  env <- new.env(); env$requests <- list()
  complete <- function(request) {
    env$requests[[length(env$requests) + 1L]] <- request
    best <- names(dist)[[which.max(dist)]]
    part <- lm15::data_part(lm15::json_object(result = best),
                            probabilities = lm15::json_object(result = do.call(lm15::json_object, as.list(dist))),
                            method = "provider_classification")
    lm15::response(request$model, lm15::message_assistant(list(part)), "stop",
                   usage = lm15::usage(input_tokens = 10L, output_tokens = 0L, total_tokens = 10L))
  }
  list(resolve = function(model) list(provider = "typesafe", model = model), complete = complete, env = env)
}

team_fn <- function(router) ai("team", "Which team should answer this message?", message = character(),
  .returns = factor(levels = c("shipping", "billing", "product", "account")), .lm = "jev-latest", .router = router)

test_that("a model that measures its probabilities gives them with one call a row", {
  r <- jev_router()
  team <- team_fn(r)
  data <- tibble::tibble(message = c("charged twice", "refund please"))
  p <- predict(team, data, type = "prob")
  expect_named(p, c(".pred_shipping", ".pred_billing", ".pred_product", ".pred_account"))
  expect_equal(p$.pred_billing, c(0.7, 0.7))
  expect_length(r$env$requests, 2L)
  a <- augment(team, data)
  expect_named(a, c("message", ".pred_class", ".pred_shipping", ".pred_billing", ".pred_product", ".pred_account", ".call", ".error"))
  expect_equal(as.character(a$.pred_class), c("billing", "billing"))
  expect_length(r$env$requests, 4L)                      # augment: one call a row for both
})

test_that("a model that measures nothing is refused before any call", {
  r <- fake_router(responder = function(req, i) "<result>\nbilling\n</result>")
  team <- update(team_fn(r), lm = "gpt-4.1-mini")
  expect_error(predict(team, tibble::tibble(message = "x"), type = "prob"), "jev-latest")
  expect_length(r$env$requests, 0L)
  a <- augment(team, tibble::tibble(message = "x"))
  expect_false(any(startsWith(names(a), ".pred_b")))
})

test_that("the call log records the probability the model gave its answer", {
  folder <- withr::local_tempdir()
  team <- update(team_fn(jev_router()), log_calls = folder)
  team("charged twice")
  rec <- Filter(function(l) !is.null(l$functai_call), log_lines(folder))[[1L]]
  expect_equal(rec$confidence, 0.7)
})

test_that("a fitted AI model reads its measured probabilities from the class prediction's calls", {
  skip_if_not_installed("parsnip")
  r <- jev_router()
  train <- tibble::tibble(message = c("a", "b"), team = factor(c("billing", "shipping"), levels = c("shipping", "billing", "product", "account")))
  fitted <- ai_model("classification", "Which team should answer this message?") |>
    parsnip::set_engine("functai", lm = "jev-latest", router = r) |>
    parsnip::fit(team ~ message, data = train)
  out <- parsnip::augment(fitted, train)
  expect_equal(out$.pred_billing, c(0.7, 0.7))
  expect_length(r$env$requests, 2L)
})
