# Loading and describing saved folders (contract/saved.md): what is refused,
# with which code and field, and what is never taken on trust.

saved_folder <- function(m) {
  dir <- withr::local_tempdir(.local_envir = parent.frame())
  writeLines(lmcc::json_text(m), file.path(dir, "functai.json"), useBytes = TRUE)
  dir
}

refusal_of <- function(expr) tryCatch(expr, functai_refusal = identity)

test_that("an interface read from a saved folder is refused with its field, loaded or described", {
  f <- ai(answer ~ n, "x", n = defaults_to(10L, json_shape(list(type = "integer", minimum = 10))))
  m <- to_manifest(f)
  m$nodes[[1L]]$interface$inputs[[1L]]$shape$default <- 5L
  dir <- saved_folder(json_normal(m))
  for (err in list(refusal_of(read_ai(dir)), refusal_of(ai_interface(dir)))) {
    expect_s3_class(err, "functai_interface_malformed")
    expect_identical(err$code, "interface-malformed")
    expect_identical(err$field, "n")
    expect_match(conditionMessage(err), "n: 5 is less than 10", fixed = TRUE)
  }
  # the loaded function's own log_content map, refused with the key it names
  m <- to_manifest(f)
  m$nodes[[1L]]$ai$settings$log_content <- list(nope = FALSE)
  err <- refusal_of(read_ai(saved_folder(json_normal(m))))
  expect_identical(err$code, "log-content-field")
  expect_identical(err$field, "nope")
  expect_s3_class(err, "functai_load_refused")
})

test_that("a folder written before interfaces is described by its signature, checked by the same rules", {
  f <- ai(answer ~ n, "x", n = integer())
  m <- to_manifest(f)
  m$nodes[[1L]]$interface <- NULL
  expect_identical(ai_interface(saved_folder(json_normal(m)))$inputs[[1L]]$name, "n")         # a sound one is described
  m$nodes[[1L]]$ai$signature$fields[[1L]]$shape$minimum <- "ten"
  dir <- saved_folder(json_normal(m))
  for (err in list(refusal_of(ai_interface(dir)), refusal_of(read_ai(dir)))) {
    expect_identical(err$code, "interface-malformed")
    expect_identical(err$field, "n")
  }
})

test_that("every probe is checked against its own fingerprint: one missing or one too many refuses", {
  f <- ai(answer ~ n, "x", n = integer())
  m <- to_manifest(f)
  m$nodes[[1L]]$ai$version <- NULL
  expect_s3_class(read_ai(saved_folder(json_normal(m))), "functai_fn")
  for (requests in list(list(), c(m$nodes[[1L]]$ai$fingerprints$requests, m$nodes[[1L]]$ai$fingerprints$requests))) {
    m2 <- m
    m2$nodes[[1L]]$ai$fingerprints$requests <- requests
    err <- refusal_of(read_ai(saved_folder(json_normal(m2))))
    expect_identical(err$code, "saved-differs")
  }
  m2 <- m                                                          # a worked example's probe with no fingerprint
  m2$nodes[[1L]]$ai$probes <- c(m$nodes[[1L]]$ai$probes, list(list(n = 7L)))
  expect_identical(refusal_of(read_ai(saved_folder(json_normal(m2))))$code, "saved-differs")
})

test_that("a saved object shape is a tibble column only when a tibble holds its every value exactly", {
  kind <- function(...) field_from_shape(list(type = "object", properties = list(a = list(type = "string"), b = list(type = "integer")), ...))$kind
  expect_identical(kind(required = list("a", "b")), "json")                                    # open: a member it does not name
  expect_identical(kind(required = list("a", "b"), additionalProperties = FALSE), "record")
  expect_identical(kind(required = list("a"), additionalProperties = FALSE), "record")         # b left out: NA, as it takes no null
  expect_identical(kind(required = list("a"), additionalProperties = TRUE), "json")
  nullable_b <- list(a = list(type = "string"), b = list(anyOf = list(list(type = "integer"), list(type = "null"))))
  expect_identical(field_from_shape(list(type = "object", properties = nullable_b, required = list("a"), additionalProperties = FALSE))$kind, "json")
  expect_identical(field_from_shape(list(type = "object", properties = nullable_b, required = list("a", "b"), additionalProperties = FALSE))$kind, "record")
  closed <- list(type = "object", properties = list(a = list(type = "string")), required = list("a"), additionalProperties = FALSE)
  expect_identical(field_from_shape(list(anyOf = list(closed, list(type = "null"))))$kind, "json")   # a null record would be a row of NAs
  expect_identical(field_from_shape(list(type = "array", items = closed))$item$kind, "record")
})

test_that("an optional input with no default of its own is described as optional, and no default is made up", {
  m <- list(functai_saved = 1L, entry = "x:mod", nodes = list(`x:mod` = list(kind = "module", module = "x", name = "mod",
    interface = list(description = "", inputs = list(list(name = "x", shape = list(type = "string"), optional = TRUE)),
                     outputs = list(list(name = "result", shape = list(type = "string")))))))
  iface <- ai_interface(saved_folder(m))
  expect_false(has_key(iface$inputs[[1L]]$shape, "default"))
  out <- paste(utils::capture.output(print(iface)), collapse = "\n")
  expect_match(out, "(optional)", fixed = TRUE)
  expect_false(grepl("default null", out, fixed = TRUE))
})
