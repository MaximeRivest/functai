# The contract every language shares (../contract), held by the R package:
# the same cases Python and TypeScript pass.

cases <- function(folder) {
  files <- sort(list.files(file.path(contract_root(), "cases", folder), pattern = "\\.json$", full.names = TRUE), method = "radix")
  stats::setNames(lapply(files, read_json_file), sub("\\.json$", "", basename(files)))
}

test_that("the contract data this package carries is the contract's", {
  root <- contract_root()
  for (p in c("layouts/xml.json", "layouts/chat.json", "layouts/json.json", "models.json", "unicode/casefold.json",
              file.path("schema", c("call.schema.json", "rating.schema.json", "saved.schema.json", "interface.schema.json", "event.schema.json"))))
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

# A contract definition written the way an R user writes it (fields as JSON;
# an optional input with its default, as defaults_to() gives one).
r_function <- function(d) {
  field <- function(f) {
    if (!isTRUE(f$optional)) return(field_from_shape(f$shape, f$desc))
    x <- field_from_shape(data_shape(f$shape), f$desc)
    if (!"default" %in% names(f$shape)) { x$optional <- TRUE; return(x) }
    defaults_to(default_value(within_default(x, f$shape[["default"]])), x)
  }
  inputs <- stats::setNames(lapply(d$inputs, field), vapply(d$inputs, function(f) f$name, ""))
  outputs <- stats::setNames(lapply(d$outputs, field), vapply(d$outputs, function(f) f$name, ""))
  settings <- list()
  if (!is.null(d$settings$adapter)) settings$.adapter <- d$settings$adapter
  if (!is.null(d$settings$module)) settings$.module <- d$settings$module
  if (isFALSE(d$settings$include_fn_name_in_instructions)) settings$.include_fn_name <- FALSE
  tools <- lapply(d$tools, function(t) structure(list(name = t$name, description = t$description, parameters = t$parameters, fn = function(...) ""), class = "functai_tool"))
  single <- identical(names(outputs), "result")
  lhs <- if (single) d$name else names(outputs)
  if (single) names(outputs) <- d$name
  formula <- stats::reformulate(names(inputs), response = str2lang(paste(lhs, collapse = " + ")))
  fn <- do.call(ai, c(list(formula, d$description), inputs, outputs, list(.name = d$name, .tools = tools), settings))
  if (!is.null(d$state$instructions)) fn <- with_instructions(fn, d$state$instructions)
  if (length(d$state$demos)) fn <- with_demos(fn, d$state$demos)
  fn
}

within_default <- function(x, default) { x$shape["default"] <- list(default); x }

plain <- function(x) lmcc::canonical_json(unclass(x))

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
    got <- rated_rows(calls, ratings, name = c$rated$name, module = c$rated$module, signature = c$rated$signature,
                      by = c$rated$by, interface = c$rated$interface)
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
      probes <- c$manifest$nodes[[c$node %||% c$manifest$entry]]$ai$probes
      expect_identical(vapply(probes, function(p) request_hash(core, p), ""), unlist(c$expect$loads$requests))
      for (send in c$expect$sends)                                # left out, an input is sent with its default
        expect_identical(request_hash(core, bound_row(fn, send$inputs)), send$request_hash, info = plain(send$inputs))
    }
    # describing the node without loading it, from its folder
    dir <- withr::local_tempdir()
    writeLines(lmcc::json_text(c$manifest), file.path(dir, "functai.json"), useBytes = TRUE)
    got <- tryCatch(ai_interface(dir, node = c$node), functai_refusal = function(e) e)
    if (!is.null(c$expect$describe$refuses)) {
      expect_s3_class(got, "functai_refusal")
      expect_identical(got$code, c$expect$describe$refuses)
    } else {
      expect_s3_class(got, "functai_interface")
      expect_identical(plain(got), plain(c$expect$describe$interface))
    }
  })
}

test_that("saved cases describe as well as load", expect_gte(sum(vapply(cases("saved"), function(c) !is.null(c$expect$describe), NA)), 16L))

# ---------------------------------------------------------------- programs (an AI function's interface)

# R has AI functions, not modules (contract/README.md, "Which cases each
# language passes"): the `ai` cases define an AI function from the
# definition; the `module` and `same-data` cases wait for modules.
for (name in names(Filter(function(c) identical(c$program, "ai"), cases("programs")))) {
  test_that(paste("program case", name), {
    c <- cases("programs")[[name]]
    got <- tryCatch(r_function(c$definition), functai_refusal = function(e) e)
    if (!is.null(c$expect$refuses)) {
      expect_s3_class(got, "functai_interface_malformed")
      expect_identical(got$code, c$expect$refuses)
      expect_identical(got$field, c$expect$field)
      return()
    }
    expect_s3_class(got, "functai_fn")
    iface <- ai_interface(got)
    expect_identical(plain(iface), plain(c$expect$interface))
    expect_identical(interface_signature(iface), c$expect$signature)
    expect_identical(ai_signature_id(got), c$expect$signature_id)
    expect_identical(program_of(core_of(got))()$interface, c$expect$signature)
    for (b in c$binds) expect_identical(plain(bound_row(got, b$inputs)), plain(b$expect$inputs), info = plain(b$inputs))
  })
}

# The `definitions` cases are assigned to languages with modules; R runs
# them through the checker it uses to define an AI function (`ai: true`) and
# to describe a saved node (a module's too), which is the same rule.
for (name in names(Filter(function(c) identical(c$program, "definitions"), cases("programs")))) {
  test_that(paste("program case", name, "(the interface checker)"), {
    c <- cases("programs")[[name]]
    for (x in c$interfaces) {
      problem <- interface_problem(x$interface, ai = isTRUE(x$ai))
      got <- if (is.null(problem)) list(signature = interface_signature(x$interface))
             else list(refuses = "interface-malformed", field = problem$field)
      expect_identical(plain(got), plain(x$expect), info = plain(x$interface))
    }
  })
}

test_that("the contract has AI program cases", expect_gte(sum(vapply(cases("programs"), function(c) identical(c$program, "ai"), NA)), 6L))

# ---------------------------------------------------------------- content

# The function a content case's fields are the fields of: two inputs, two
# outputs, reasoning added (module cot), and the tool calls with tools.
content_function <- function(fields, own) {
  tools <- if ("calls" %in% fields$added) list(ai_tool(function(job) "", "Read a CI log.", .name = "ci_log")) else NULL
  args <- list(summary + result ~ transcript + question, "Is the build broken?", .name = "triage_build", .defined_in = "ci",
               .module = "cot", .tools = tools, .log_calls = FALSE)
  if (!is.null(own)) args$.log_content <- own
  do.call(ai, args)
}

# Runs `code` inside the layers of a case: blocks and ai_config() around it.
within_layers <- function(layers, code) {
  settings <- functai:::the
  old <- settings$config
  on.exit(settings$config <- old)
  wrap <- function(i) {
    if (i < 1L) return(force(code))
    l <- layers[[i]]
    switch(l$where,
      own = wrap(i - 1L),
      configure = { ai_config(log_content = l$log_content); wrap(i - 1L) },
      block = with_ai_config(wrap(i - 1L), log_content = l$log_content))
  }
  wrap(length(layers))                                             # outermost first
}

for (name in names(cases("content"))) {
  test_that(paste("content case", name), {
    c <- cases("content")[[name]]
    if (is.null(c$environment)) withr::local_envvar(FUNCTAI_LOG_CONTENT = NA) else withr::local_envvar(FUNCTAI_LOG_CONTENT = c$environment)
    own <- Find(function(l) identical(l$where, "own"), c$layers)
    run <- function() within_layers(c$layers, {
      fn <- content_function(c$fields, own$log_content)
      core <- core_of(fn)
      fields <- call_fields(core, effective(core$own))
      expect_identical(fields$inputs, unlist(c$fields$inputs))
      expect_identical(fields$outputs, unlist(c$fields$outputs))
      expect_identical(fields$added, unlist(c$fields$added))
      keep <- content_kept(fields, content_layers(core))
      kept_record(c$record, fields, keep)
    })
    if (!is.null(c$expect$refuses)) {
      err <- tryCatch(run(), functai_refusal = function(e) e)
      expect_s3_class(err, "functai_log_content_field")
      expect_identical(err$code, c$expect$refuses)
      expect_identical(err$field, c$expect$field)
    } else {
      expect_identical(plain(run()), plain(c$expect$record))
    }
  })
}

# ---------------------------------------------------------------- saw

for (name in names(Filter(function(c) identical(c$kind, "read"), cases("saw")))) {
  test_that(paste("saw case", name), {
    c <- cases("saw")[[name]]
    for (q in c$queries) {
      got <- read_saw(c$records, q$call)
      expect <- if (!is.null(got$unknown)) list(unknown = got$unknown$code, call = got$unknown$call) else list(saw = got$saw)
      expect_identical(plain(expect), plain(q$expect), info = q$call)
      expect_identical(plain(got$keeps), plain(q$keeps), info = q$call)
    }
  })
}

# ---------------------------------------------------------------- the schemas R reads

test_that("the schema reader knows every keyword the contract's schemas use", {
  keywords <- character(0)
  walk <- function(x, key = "") {
    if (is_obj(x)) {
      if (!key %in% c("properties", "$defs")) keywords <<- c(keywords, names(x))
      for (k in names(x)) walk(x[[k]], if (key %in% c("properties", "$defs")) "" else k)
    } else if (is_arr(x)) for (v in x) walk(v)
  }
  for (f in c("call.schema.json", "rating.schema.json", "saved.schema.json", "interface.schema.json")) walk(contract_schema(f))
  expect_identical(setdiff(unique(keywords), c(SCHEMA_KEYWORDS, SCHEMA_WORDS)), character(0))
})

test_that("every record in the contract's cases passes its schema here, and a leaking one does not", {
  for (folder in c("rated", "saw")) for (c in cases(folder)) for (r in c$records) {
    if (is_format(r$functai_rating, RATING_FORMAT)) expect_null(schema_fault(r, "rating.schema.json"))
    if (is_format(r$functai_call, CALL_FORMATS)) expect_null(schema_fault(r, "call.schema.json"))
  }
  for (c in cases("content")) if (!is.null(c$expect$record)) expect_null(schema_fault(c$expect$record, "call.schema.json"))
  for (c in cases("saved")) if (!identical(c$expect$refuses, "saved-format")) expect_null(schema_fault(c$manifest, "saved.schema.json"))
  leaking <- cases("content")[["03-all-but-one-input"]]$expect$record
  leaking$exchanges[[1L]]$request <- list(messages = list())       # a request kept beside a dropped value
  expect_false(is.null(schema_fault(leaking, "call.schema.json")))
  named <- cases("saved")[["16-an-optional-input"]]$manifest
  named$nodes[["shop:reply"]]$interface$inputs[[1L]]$name <- "message\n"   # a name with a newline after it
  expect_false(is.null(schema_fault(named, "saved.schema.json")))
})
