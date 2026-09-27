# The contract every language shares (../contract), held by the R package:
# the same cases Python and TypeScript pass.

cases <- function(folder) {
  files <- sort(list.files(file.path(contract_root(), "cases", folder), pattern = "\\.json$", full.names = TRUE), method = "radix")
  stats::setNames(lapply(files, read_json_file), sub("\\.json$", "", basename(files)))
}

test_that("the contract data this package carries is the contract's", {
  root <- contract_root()
  for (p in c("layouts/xml.json", "layouts/chat.json", "layouts/json.json", "models.json", "unicode/casefold.json"))
    expect_identical(readLines(system.file("contract", p, package = "functai")), readLines(file.path(root, p)), info = p)
})

test_that("capabilities follow the contract's table", {
  table <- read_json_file(file.path(contract_root(), "models.json"))
  for (p in unlist(table$native$providers)) {
    caps <- model_capabilities(p, "some-model")
    expect_identical(caps$stop_sequences, !p %in% unlist(table$native$no_stop_sequences))
  }
  expect_true(model_capabilities("anthropic", "claude-sonnet-4-5")$native_reasoning)
  expect_true(model_capabilities("anthropic", "claude-3-5-haiku")$assistant_prefill)
  expect_identical(model_capabilities("claude-code", "claude-sonnet-4-5"), model_capabilities("anthropic", "claude-sonnet-4-5"))
  expect_false(model_capabilities("ollama", "x")$native_function_calling)
  expect_identical(model_capabilities("typesafe", "x"), list(native_structured_output = TRUE))
})

test_that("a sampling the model does not take is left out, with one warning", {
  the <- functai:::the
  the$warned <- list()
  expect_identical(adjust_settings(list(temperature = 1), "openai", "gpt-6-luna"), list(temperature = 1))
  expect_warning(out <- adjust_settings(list(temperature = 0, top_p = 0.5), "openai", "gpt-6-luna"), "does not take temperature, top_p")
  expect_null(out$temperature); expect_null(out$top_p)
  expect_no_warning(adjust_settings(list(temperature = 0, top_p = 0.5), "openai", "gpt-6-luna"))
  expect_null(suppressWarnings(adjust_settings(list(temperature = 0), "claude-code", "claude-sonnet-5"))$temperature)
  expect_identical(adjust_settings(list(temperature = 0), "anthropic", "claude-haiku-4-5"), list(temperature = 0))
  expect_identical(refused_settings("gemini", "gemini-3.8-flash"), character(0))
})

# ---------------------------------------------------------------- functions

# A contract definition written the way an R user writes it (fields as JSON).
r_function <- function(d) {
  field <- function(f) { x <- field_from_shape(f$shape, f$desc); x }
  inputs <- stats::setNames(lapply(d$inputs, field), vapply(d$inputs, function(f) f$name, ""))
  outputs <- stats::setNames(lapply(d$outputs, field), vapply(d$outputs, function(f) f$name, ""))
  settings <- list()
  if (!is.null(d$settings$adapter)) settings$.adapter <- d$settings$adapter
  if (!is.null(d$settings$module)) settings$.module <- d$settings$module
  if (isFALSE(d$settings$include_fn_name_in_instructions)) settings$.include_fn_name <- FALSE
  tools <- lapply(d$tools, function(t) structure(list(name = t$name, description = t$description, parameters = t$parameters, fn = function(...) ""), class = "functai_tool"))
  fn <- do.call(ai, c(list(.name = d$name, .description = d$description), inputs, list(.outputs = outputs, .tools = tools), settings))
  if (!is.null(d$state$instructions)) fn <- with_instructions(fn, d$state$instructions)
  if (length(d$state$demos)) fn <- with_demos(fn, d$state$demos)
  fn
}

without_type <- function(sig) list(instructions = sig$instructions, fields = lapply(sig$fields, function(f) {
  f$type <- NULL
  f <- Filter(Negate(is.null), f)
  f$purpose <- f$purpose %||% "plain"
  f[order(names(f), method = "radix")]
}))

for (name in names(cases("functions"))) {
  test_that(paste("function case", name), {
    c <- cases("functions")[[name]]
    fn <- r_function(c$definition)
    core <- core_of(fn)
    expect_identical(lmcc::canonical_json(without_type(ai_signature(fn))), lmcc::canonical_json(without_type(c$expect$signature)))
    expect_identical(lmcc::canonical_json(sample_inputs(signature_of(core, effective(core$own)))), lmcc::canonical_json(c$expect$sample))
    expect_identical(lmcc::canonical_json(probe_request(core, c$expect$sample)), lmcc::canonical_json(c$expect$request))
    expect_identical(request_hash(core, c$expect$sample), c$expect$request_hash)
    expect_identical(ai_version(fn), c$expect$version)
    expect_identical(ai_signature_id(fn), c$expect$signature_id)
  })
}

test_that("the contract has function cases", expect_gte(length(cases("functions")), 10L))

# ---------------------------------------------------------------- scores

for (name in names(cases("scores"))) {
  test_that(paste("score case", name), {
    c <- cases("scores")[[name]]
    if (identical(c$kind, "interval")) {
      got <- score_interval(as.numeric(unlist(c$values) %||% numeric(0)))
      for (k in c("mean", "low", "high")) {
        if (is.null(c$expect[[k]])) expect_true(is.na(got[[k]]), info = k)
        else expect_lt(abs(got[[k]] - as.numeric(c$expect[[k]])), 1e-12)
      }
    } else {
      keys <- intersect(names(c$prediction), names(c$answers))
      got <- exact_match(c$answers, c$prediction[keys])
      expect_identical(as.list(got), lapply(c$expect, as.numeric))
    }
  })
}

# ---------------------------------------------------------------- rated

for (name in names(cases("rated"))) {
  test_that(paste("rated case", name), {
    c <- cases("rated")[[name]]
    calls <- Filter(function(r) !is.null(r$functai_call), c$records)
    ratings <- Filter(function(r) !is.null(r$functai_rating), c$records)
    got <- rated_rows(calls, ratings, name = c$rated$name, module = c$rated$module, signature = c$rated$signature, by = c$rated$by)
    expect_identical(vapply(got$rows, lmcc::canonical_json, ""), vapply(c$expect$rows, lmcc::canonical_json, ""))
    expect_identical(lmcc::canonical_json(got$left_out), lmcc::canonical_json(c$expect$left_out))
  })
}

# ---------------------------------------------------------------- saved

for (name in names(cases("saved"))) {
  test_that(paste("saved case", name), {
    c <- cases("saved")[[name]]
    if (!is.null(c$expect$refuses)) {
      err <- tryCatch(from_manifest(c$manifest, c$node), functai_load_refused = function(e) e)
      expect_s3_class(err, "functai_load_refused")
      expect_identical(err$code, c$expect$refuses)
    } else {
      fn <- from_manifest(c$manifest, c$node)
      core <- core_of(fn)
      expect_identical(core$definition$name, c$expect$loads$name)
      expect_identical(core$module, c$expect$loads$module)
      expect_identical(ai_version(fn), c$expect$loads$version)
      expect_identical(ai_signature_id(fn), c$expect$loads$signature_id)
    }
  })
}
