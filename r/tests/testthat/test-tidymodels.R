# AI functions as tidymodels models: parsnip, workflows, rsample, tune, yardstick.

skip_if_not_installed("parsnip")
skip_if_not_installed("workflows")
skip_if_not_installed("yardstick")

lv <- c("shipping", "billing", "product", "account")
keyword_router <- function() fake_router(responder = function(req, i) {
  t <- message_text(req, length(lmcc::lm15_plain(lm15::as_dict(req))$messages))
  ans <- if (grepl("charg|refund|invoice|pay", t, ignore.case = TRUE)) "billing"
    else if (grepl("arriv|deliver|parcel|track|ship", t, ignore.case = TRUE)) "shipping"
    else if (grepl("password|account|login|email", t, ignore.case = TRUE)) "account" else "product"
  sprintf("<result>\n%s\n</result>", ans)
})
data <- dplyr::mutate(tickets, category = factor(category, levels = lv))

spec_of <- function(router, ...) ai_model("classification", "Which team should answer this customer message?", ...) |>
  parsnip::set_engine("functai", lm = "gpt-4.1-mini", router = router, log_calls = FALSE)

test_that("a zero-shot fit predicts .pred_class with the outcome's levels; the engine is an AI function", {
  r <- keyword_router()
  fitted <- parsnip::fit(spec_of(r), category ~ message, data = data)
  expect_length(r$env$requests, 0L)                             # fitting calls nothing
  p <- predict(fitted, data[1:5, ])
  expect_identical(names(p), ".pred_class")
  expect_identical(levels(p$.pred_class), lv)
  engine <- parsnip::extract_fit_engine(fitted)
  expect_s3_class(engine, "functai_fn")
  expect_match(ai_instructions(engine), "^Function: category\n\nWhich team should answer")
  expect_identical(as.character(engine("I was charged twice")), "billing")
  expect_warning(a <- parsnip::augment(fitted, data), "probabilities are NA")
  expect_true(all(is.na(a$.pred_billing)))
  acc <- yardstick::accuracy(a, truth = category, estimate = .pred_class)
  expect_gt(acc$.estimate, 0.5)
})

test_that("examples picks worked examples from the training rows (free); a new version", {
  f0 <- parsnip::fit(spec_of(keyword_router()), category ~ message, data = data)
  f4 <- parsnip::fit(spec_of(keyword_router(), examples = 4), category ~ message, data = data)
  e0 <- parsnip::extract_fit_engine(f0); e4 <- parsnip::extract_fit_engine(f4)
  expect_length(ai_demos(e4), 4L)
  expect_false(identical(ai_version(e0), ai_version(e4)))
  expect_length(ai_render(e4, "x")$messages, 9L)
})

test_that("probabilities come from repeated answers, asked for; without samples prob refuses", {
  r <- fake_router(responder = function(req, i) sprintf("<result>\n%s\n</result>", c("billing", "billing", "account")[[(i - 1L) %% 3L + 1L]]))
  fitted <- parsnip::fit(spec_of(r) |> parsnip::set_engine("functai", lm = "gpt-4.1-mini", router = r, log_calls = FALSE, samples = 3L),
                         category ~ message, data = data)
  a <- parsnip::augment(fitted, data[1:2, ])
  expect_identical(names(a)[names(a) %in% c(".pred_class", paste0(".pred_", lv))], c(".pred_class", paste0(".pred_", lv)))
  expect_equal(a$.pred_billing, c(2 / 3, 2 / 3))
  expect_identical(as.character(a$.pred_class), c("billing", "billing"))
  expect_length(r$env$requests, 6L)                             # class and prob share the 3 calls a row
  expect_warning(none <- predict(parsnip::fit(spec_of(keyword_router()), category ~ message, data = data), data[1, ], type = "prob"), "samples")
  expect_true(all(is.na(unlist(none))))                         # through parsnip: NA, never a made-up number
  plain <- parsnip::extract_fit_engine(parsnip::fit(spec_of(keyword_router()), category ~ message, data = data))
  expect_error(predict(plain, data[1, ], type = "prob"), "samples")   # asked directly: refused
  mood <- ai("mood", "Mood?", review = character(), .returns = factor(levels = c("happy", "unhappy")),
             .lm = "gpt-4.1-mini", .log_calls = FALSE, .router = fake_router(responder = function(req, i) "<result>\nhappy\n</result>"))
  expect_identical(names(augment(mood, tibble::tibble(review = "x"), samples = 2L)),
                   c("review", ".pred_class", ".pred_happy", ".pred_unhappy", ".call", ".error"))
})

test_that("it is a model in a workflow, next to any other", {
  wf <- workflows::workflow() |> workflows::add_formula(category ~ message) |> workflows::add_model(spec_of(keyword_router()))
  fitted <- parsnip::fit(wf, data = data)
  p <- predict(fitted, data[1:3, ])
  expect_identical(names(p), ".pred_class")
  engine <- workflows::extract_fit_engine(fitted)
  expect_false(grepl("Function:", ai_instructions(engine)))     # workflows hide the outcome's name (..y)
  ev <- evaluate(fitted, data)
  expect_identical(tidy(ev)$n, 80L)
})

test_that("resampling and tuning the number of worked examples", {
  skip_if_not_installed("rsample")
  skip_if_not_installed("tune")
  skip_if_not_installed("dials")
  set.seed(1)
  folds <- rsample::vfold_cv(data, v = 2)
  wf <- workflows::workflow() |> workflows::add_formula(category ~ message) |>
    workflows::add_model(spec_of(keyword_router(), examples = tune::tune()))
  res <- tune::tune_grid(wf, folds, grid = tibble::tibble(examples = c(0L, 2L)), metrics = yardstick::metric_set(yardstick::accuracy))
  m <- tune::collect_metrics(res)
  expect_identical(sort(m$examples), c(0L, 2L))
  best <- tune::select_best(res, metric = "accuracy")
  final <- tune::finalize_workflow(wf, best)
  expect_identical(rlang::eval_tidy(workflows::extract_spec_parsnip(final)$args$examples), best$examples)
  expect_identical(tune::tunable(spec_of(NULL, examples = tune::tune()))$name, "examples")
  expect_s3_class(worked_examples(), "quant_param")
})

test_that("evaluate scores a classical parsnip fit like an AI function", {
  skip_if_not_installed("nnet")
  fit <- parsnip::fit(parsnip::multinom_reg() |> parsnip::set_engine("nnet"), category ~ channel, data = data)
  ev <- evaluate(fit, data)
  expect_identical(ev$fn, "multinom_reg")
  expect_true(tidy(ev)$conf.low < ev$score)
})

test_that("a spec with no engine arguments fits (the model and settings come from ai_config)", {
  withr::defer(ai_config(lm = NULL, router = NULL, log_calls = NULL))
  ai_config(lm = "gpt-4.1-mini", router = keyword_router(), log_calls = FALSE)
  fitted <- parsnip::fit(ai_model("classification", "Which team?"), category ~ message, data = data)
  expect_identical(as.character(predict(fitted, data[1, ])$.pred_class), "shipping")
})
