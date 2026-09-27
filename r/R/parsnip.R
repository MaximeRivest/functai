# An AI function as a tidymodels model: a parsnip model type, "ai_model",
# with the engine "functai". Everything tidymodels does with a model (fit,
# predict, workflows, resampling, tuning, yardstick) then works with a model
# that is a language model given a description of the task.

#' An AI model, for tidymodels
#'
#' A parsnip model whose predictions a language model makes, from a
#' description of the task. It fits and predicts like any parsnip model, so
#' it goes in workflows, is resampled with rsample and tuned with tune, and
#' its predictions are scored with yardstick.
#'
#' **What fitting does.** No weights are estimated. Fitting reads the
#' outcome (its levels, for classification: they become the answers the
#' model may give) and the predictors (their names and types become the
#' function's inputs), and picks `examples` rows of the training data as
#' worked examples the model sees with every question. So fitting is free,
#' and a fit with `examples = 0` has used no answer from the training data.
#'
#' **Fitting that learns the instruction.** With `method = "gepa"`, fitting
#' runs [gepa()] on the training rows: a model reads the function's mistakes
#' and rewrites its instruction, within `budget` calls. Fitting then costs
#' calls, and resampling repeats it on every fold, so the resampled score
#' measures the whole procedure, the search included. The fitted function's
#' instruction is what it learned: `extract_fit_engine(fit)` prints it.
#'
#' **Engine arguments** (`set_engine("functai", ...)`): `lm` (the model,
#' `"gpt-4.1-mini"` by default from [ai_config()]), `temperature`, any
#' other setting of [ai_config()]; `method` (`"labeled"`: training rows as
#' they are, the default; `"bootstrap"`: rows the model got right, whole,
#' reasoning included; `"gepa"`: the instruction rewritten from mistakes,
#' see below); `budget` (calls `"gepa"` may make, default 300); `teacher` (the
#' model that writes instructions or examples); `samples` (answers per row: with `samples = 5`,
#' `predict(type = "prob")` gives each level's share of the answers and the
#' class is the majority; costs 5 calls a row); `seed`.
#'
#' Every prediction is a paid model call. Resampling and tuning multiply
#' them: a grid of 3 values on 5 folds predicts every training row 3 times.
#'
#' @param mode `"classification"` (a factor outcome) or `"regression"`.
#' @param description What the model is asked, in words: "Which team should
#'   answer this customer message?".
#' @param examples How many training rows the model sees as worked examples
#'   (default 0). Tunable: [worked_examples()].
#' @param name The function's name, which the model reads ("Function:
#'   team") and the call log files calls under. Default: the outcome's name.
#' @return A model specification.
#' @examples
#' ai_model("classification", "Which team should answer this customer message?")
#' \dontrun{
#' fitted <- ai_model("classification", "Which team should answer this customer message?") |>
#'   parsnip::set_engine("functai", lm = "gpt-4.1-mini") |>
#'   parsnip::fit(category ~ message, data = train)
#' predict(fitted, test)
#' }
#' @export
ai_model <- function(mode = "classification", description = NULL, examples = NULL, name = NULL) {
  rlang::check_installed("parsnip", "to use an AI function as a tidymodels model")
  register_parsnip()
  if (!mode %in% c("classification", "regression", "unknown")) cli::cli_abort("{.arg mode} is \"classification\" or \"regression\"")
  args <- list(description = rlang::enquo(description), examples = rlang::enquo(examples), name = rlang::enquo(name))
  parsnip::new_model_spec("ai_model", args = args, eng_args = NULL, mode = mode, user_specified_mode = !missing(mode),
                          method = NULL, engine = "functai", user_specified_engine = FALSE)
}

#' @export
print.ai_model <- function(x, ...) {
  cat("AI Model Specification (", x$mode, ")\n\n", sep = "")
  parsnip::model_printer(x, ...)
  invisible(x)
}

#' @param object A model specification.
#' @param parameters A one-row tibble of new values (from tune).
#' @param fresh Replace every argument, rather than only the ones given.
#' @param ... Engine arguments to change.
#' @rdname ai_model
#' @export
update.ai_model <- function(object, parameters = NULL, description = NULL, examples = NULL, name = NULL, fresh = FALSE, ...) {
  args <- list(description = rlang::enquo(description), examples = rlang::enquo(examples), name = rlang::enquo(name))
  parsnip::update_spec(object = object, parameters = parameters, args_enquo_list = args, fresh = fresh, cls = "ai_model", ...)
}

#' @importFrom generics tunable
#' @export
generics::tunable

#' @rdname ai_model
#' @param x A model specification.
#' @export
tunable.ai_model <- function(x, ...) {
  tibble::tibble(name = "examples", call_info = list(list(pkg = "functai", fun = "worked_examples")),
                 source = "model_spec", component = "ai_model", component_id = "main")
}

#' How many worked examples, as a tuning parameter
#'
#' The dials parameter for [ai_model()]'s `examples`.
#' @param range The smallest and largest number of examples.
#' @param trans Unused (the scale is the count itself).
#' @return A dials quantitative parameter.
#' @examples
#' \dontrun{worked_examples(c(0, 8))}
#' @export
worked_examples <- function(range = c(0L, 16L), trans = NULL) {
  rlang::check_installed("dials")
  dials::new_quant_param(type = "integer", range = range, inclusive = c(TRUE, TRUE), trans = trans,
                         label = c(examples = "Worked examples"), finalize = NULL)
}

# ---------------------------------------------------------------- fit

#' Fit an AI model (parsnip's fit function for the "functai" engine)
#'
#' Called by [parsnip::fit()]; use that. Returns an AI function (see [ai()]):
#' `parsnip::extract_fit_engine()` gives it back, callable on columns.
#' @param formula,data The outcome and predictors, and the training data.
#' @param description,examples,name See [ai_model()].
#' @param method `"labeled"`, `"bootstrap"` or `"gepa"`.
#' @param budget Calls `"gepa"` may make.
#' @param teacher The model that writes instructions (`"gepa"`) or examples (`"bootstrap"`).
#' @param samples Answers per row at prediction time.
#' @param seed The random seed for choosing examples.
#' @param ... Settings, as in [ai_config()] (`lm`, `temperature`, ...).
#' @return An AI function.
#' @keywords internal
#' @export
ai_model_fit <- function(formula, data, description = NULL, examples = 0L, name = NULL, method = "labeled",
                         samples = 1L, seed = 0L, budget = 300L, teacher = NULL, ...) {
  if (is.null(description) || !nzchar(description))
    cli::cli_abort(c("an AI model needs a description of its task", i = "{.code ai_model(description = \"Which team should answer this message?\")}"))
  tt <- stats::terms(formula, data = data)
  outcome <- all.vars(formula[[2L]])
  predictors <- attr(tt, "term.labels")
  bad <- setdiff(predictors, names(data))
  if (length(outcome) != 1L || length(bad)) cli::cli_abort("an AI model's formula names columns: {.code outcome ~ text + other}, not {.code {bad}}")
  data <- data[!is.na(data[[outcome]]), c(predictors, outcome), drop = FALSE]
  hidden <- startsWith(outcome, "..")               # workflows name the outcome ..y
  settings <- list(...)
  if (length(settings)) names(settings) <- paste0(".", names(settings))
  if (hidden && is.null(name)) settings$.include_fn_name <- FALSE
  # the same function ai() writes from this formula, its types read from the data
  fn <- do.call(ai, c(list(stats::reformulate(predictors, response = outcome), description, .data = data,
                           .name = name %||% if (hidden) "ai_model" else outcome), settings))
  train <- data
  examples <- as.integer(examples %||% 0L)
  if (!method %in% c("labeled", "bootstrap", "gepa"))
    cli::cli_abort("{.arg method} is \"labeled\", \"bootstrap\" or \"gepa\", not {.val {method}}")
  if (method == "gepa") fn <- gepa(fn, train, budget = budget, teacher = teacher, seed = seed)
  if (examples > 0L) {
    fn <- if (method == "bootstrap")
      bootstrap_few_shot(fn, train, max_bootstrapped = examples, max_labeled = examples, teacher = teacher, seed = seed)
    else labeled_few_shot(fn, train, k = examples, seed = seed)            # after gepa: its instruction, and examples
  }
  core <- core_of(fn)
  core$samples <- as.integer(samples)
  core$memo <- new.env()
  make_fn(core)
}

#' @rdname ai_model_fit
#' @param object A fitted AI function.
#' @param new_data A data frame.
#' @param type `"class"`, `"prob"` or `"numeric"`.
#' @keywords internal
#' @export
ai_model_predict <- function(object, new_data, type) {
  core <- core_of(object)
  samples <- core$samples %||% 1L
  if (type == "prob" && samples < 2L && measures_probabilities(core)) {
    # the model measures its probabilities (TypeSafe's Jev): the class prediction's calls gave them
    return(predict.functai_fn(object, new_data, type = "prob"))
  }
  if (type == "prob" && samples < 2L) {
    # parsnip's augment() asks every classification model for probabilities;
    # one answer a row measures none, so they are NA (never a made-up number)
    the$warned[["prob-without-samples"]] <- NULL                  # every time: a silent NA would mislead
    warn_once("prob-without-samples", c("probabilities are NA: one answer per row measures no probability",
      i = "for them, answer each row several times ({.code set_engine(\"functai\", samples = 5)}, 5 calls a row), or use a model that measures them ({.code lm = \"jev-latest\"})"))
    lv <- single_field(core)$levels
    out <- tibble::as_tibble(stats::setNames(rep(list(rep(NA_real_, nrow(new_data))), length(lv)), lv))
    return(out)
  }
  if (type == "prob") return(predict.functai_fn(object, new_data, type = "prob", samples = samples))
  p <- predict.functai_fn(object, new_data, type = "class", samples = samples)
  if (type == "class") p$.pred_class else p$.pred
}

# ---------------------------------------------------------------- registration

register_parsnip <- function() {
  if (!requireNamespace("parsnip", quietly = TRUE)) return(invisible(FALSE))
  env <- parsnip::get_model_env()
  if ("ai_model" %in% env$models) return(invisible(TRUE))
  parsnip::set_new_model("ai_model")
  for (mode in c("classification", "regression")) {
    parsnip::set_model_mode("ai_model", mode)
    parsnip::set_model_engine("ai_model", mode, "functai")
    parsnip::set_dependency("ai_model", "functai", "functai", mode = mode)
    parsnip::set_fit(model = "ai_model", eng = "functai", mode = mode, value = list(
      interface = "formula", protect = c("formula", "data"),
      func = c(pkg = "functai", fun = "ai_model_fit"), defaults = list()))
    parsnip::set_encoding(model = "ai_model", eng = "functai", mode = mode, options = list(
      predictor_indicators = "none", compute_intercept = FALSE, remove_intercept = FALSE, allow_sparse_x = FALSE))
  }
  for (arg in c("description", "examples", "name"))
    parsnip::set_model_arg(model = "ai_model", eng = "functai", parsnip = arg, original = arg,
                           func = list(pkg = "functai", fun = if (arg == "examples") "worked_examples" else arg), has_submodel = FALSE)
  pred <- function(type) list(pre = NULL, post = NULL, func = c(pkg = "functai", fun = "ai_model_predict"),
                              args = list(object = quote(object$fit), new_data = quote(new_data), type = type))
  parsnip::set_pred(model = "ai_model", eng = "functai", mode = "classification", type = "class", value = pred("class"))
  parsnip::set_pred(model = "ai_model", eng = "functai", mode = "classification", type = "prob", value = pred("prob"))
  parsnip::set_pred(model = "ai_model", eng = "functai", mode = "regression", type = "numeric", value = pred("numeric"))
  invisible(TRUE)
}

.onLoad <- function(libname, pkgname) {
  # parsnip learns about ai_model when both are loaded, in either order
  if (isNamespaceLoaded("parsnip")) register_parsnip()
  setHook(packageEvent("parsnip", "onLoad"), function(...) register_parsnip())
}
