# Binding a call's inputs: what an input left out is sent with, what a
# missing value is, which values fit (contract/programs.md, "Checking
# values"), and how many calls a table makes. Each test calls the function
# itself, against a fake provider.

ok_router <- function() fake_router(responder = function(req, i) "<result>\nok\n</result>")

test_that("an input left out is sent with its default exactly as the interface holds it, before and after saving", {
  shape <- list(type = "object", properties = list(a = list(type = "string"), b = list(type = "integer")), required = list("a"))
  f <- ai(answer ~ message + options, "x", options = defaults_to(list(a = "x"), json_shape(shape)), .lm = "gpt-4.1-mini")
  dir <- withr::local_tempdir()
  write_ai(f, dir)
  g <- read_ai(dir)
  bound <- list(message = "Hi", options = list(a = "x"))              # no `b`: absent is not null
  before <- expect_sends(f, list(message = "Hi"), bound = bound)
  after <- expect_sends(g, list(message = "Hi"), bound = bound)
  expect_identical(after$exchanges[[1L]]$request_hash, before$exchanges[[1L]]$request_hash)
  expect_identical(plain(after$exchanges[[1L]]$request), plain(before$exchanges[[1L]]$request))
  expect_identical(plain(ai_interface(g)$inputs[[2L]]$shape$default), '{"a":"x"}')
  # the loaded field is JSON, not a record that would fill `b` with a null
  expect_identical(core_of(g)$definition$inputs$options$kind, "json")
})

test_that("defaults of every kind are sent as they are: extra members, arrays, nulls, records, choices", {
  cases <- list(
    list(field = defaults_to(list(a = "x", extra = 1L), json_shape(list(type = "object", properties = list(a = list(type = "string")), required = list("a")))),
         json = '{"a":"x","extra":1}'),
    list(field = defaults_to(list(1L, 2L), json_shape(list(type = "array", items = list(type = "integer")))), json = "[1,2]"),
    list(field = defaults_to(list("a", "b"), vctrs::list_of(.ptype = character())), json = '["a","b"]'),
    list(field = defaults_to(NA, optional(integer())), json = "null"),
    list(field = defaults_to(tibble::tibble(a = "x", b = 1L)), json = '{"a":"x","b":1}'),
    list(field = defaults_to("brief", choice("kind", "brief")), json = '"brief"'),
    list(field = defaults_to(2.5), json = "2.5"))
  for (x in cases) {
    f <- ai(answer ~ message + extra, "x", extra = x$field, .lm = "gpt-4.1-mini")
    dir <- withr::local_tempdir()
    write_ai(f, dir)
    for (fn in list(f, read_ai(dir))) {
      rec <- expect_sends(fn, list(message = "Hi"))
      expect_identical(lmcc::canonical_json(rec$inputs$extra), x$json, info = x$json)
    }
  }
})

test_that("a default of null is sent, left out or given, and never skips the call", {
  tone <- defaults_to(NULL, json_shape(list(anyOf = list(list(type = "string"), list(type = "null")))))
  r <- ok_router()
  f <- ai(answer ~ message + tone, "x", tone = tone, .router = r, .lm = "gpt-4.1-mini", .log_calls = FALSE)
  expect_identical(unlist(f("Hi")), "ok")
  expect_length(r$env$requests, 1L)
  expect_sends(f, list(message = "Hi"), bound = list(message = "Hi", tone = NULL))
  expect_sends(f, list(message = "Hi", tone = NA), bound = list(message = "Hi", tone = NULL))       # R's missing value is JSON's null
  expect_sends(f, list(message = "Hi", tone = "brief"), bound = list(message = "Hi", tone = "brief"))
  p <- predict(f, tibble::tibble(message = c("a", "b"), tone = list(NULL, "brief")))
  expect_identical(nrow(p), 2L)
  expect_identical(p$.error, c(NA_character_, NA_character_))
  expect_length(r$env$requests, 3L)
  dir <- withr::local_tempdir()
  write_ai(update(f, router = NULL), dir)
  expect_sends(read_ai(dir), list(message = "Hi"), bound = list(message = "Hi", tone = NULL))
})

test_that("a missing value (NA) in an input whose type takes no null makes no call, required or optional", {
  r <- ok_router()
  f <- ai(answer ~ message + tone, "x", tone = defaults_to("kind"), .router = r, .lm = "gpt-4.1-mini", .log_calls = FALSE)
  expect_identical(f(c("a", "b"), tone = c("brief", NA)), c("ok", NA))
  expect_identical(f(c("a", NA)), c("ok", NA))
  expect_length(r$env$requests, 2L)
})

test_that("a table's row count decides the calls, even when every input is left out", {
  r <- ok_router()
  f <- ai(answer ~ tone, "x", tone = defaults_to("kind"), .router = r, .lm = "gpt-4.1-mini", .log_calls = FALSE)
  p <- predict(f, tibble::tibble(id = 1:3))
  expect_identical(nrow(p), 3L)
  expect_length(r$env$requests, 3L)
  expect_identical(length(unique(p$.call)), 3L)                      # three calls, not one recycled
  p <- predict(f, tibble::tibble(id = integer()))
  expect_identical(nrow(p), 0L)
  expect_length(r$env$requests, 3L)                                  # an empty table costs nothing
  a <- augment(f, tibble::tibble(id = 1:2))
  expect_identical(nrow(a), 2L)
  expect_length(r$env$requests, 5L)
  expect_identical(f(), "ok")                                       # a direct call with nothing given: one call
  expect_length(r$env$requests, 6L)
})

test_that("a given input that does not fit its type fails before any request, and its record says why", {
  r <- ok_router()
  folder <- withr::local_tempdir()
  f <- ai(answer ~ n, "x", n = json_shape(list(type = "integer", minimum = 10)), .router = r, .lm = "gpt-4.1-mini", .log_calls = folder)
  err <- tryCatch(f(5L), functai_refusal = identity)
  expect_s3_class(err, "functai_interface_input")
  expect_identical(err$code, "interface-input")
  expect_identical(err$field, "n")
  expect_length(r$env$requests, 0L)
  rec <- log_lines(folder)[[1L]]
  expect_identical(rec$error$type, "InterfaceError")
  expect_identical(rec$error$code, "interface-input")
  expect_identical(rec$exchanges, list())
  expect_null(rec$outputs)
  expect_null(schema_fault(rec, "call.schema.json"))
  # one row of several: that row fails, the others are answered
  expect_warning(out <- f(c(5L, 12L)), "1 of 2 calls")
  expect_identical(out, c(NA, "ok"))
  expect_length(r$env$requests, 1L)
  expect_identical(ai_problems()$row, 1L)
})

test_that("values are checked by every keyword of the vocabulary, and a number is never truncated to fit", {
  r <- ok_router()
  refused <- function(field, value) {
    f <- ai(answer ~ v, "x", v = field, .router = r, .lm = "gpt-4.1-mini", .log_calls = FALSE)
    inherits(tryCatch(f(value), functai_interface_input = identity), "functai_interface_input")
  }
  expect_true(refused(integer(), 5.5))                               # not sent as 5
  expect_false(refused(integer(), 5))                                # 5.0 is an integer
  expect_false(refused(double(), 5L))                                # every integer is a number
  expect_true(refused(choice("a", "b"), "c"))
  obj <- list(type = "object", properties = list(a = list(type = "string")), required = list("a"), additionalProperties = FALSE)
  expect_true(refused(json_shape(obj), list(list(a = "x", z = 1L))))  # a member the shape does not allow
  expect_true(refused(json_shape(obj), list(list(z = "x"))))          # a required member missing
  expect_false(refused(json_shape(obj), list(list(a = "x"))))
  expect_true(refused(json_shape(list(const = "yes")), "no"))
  expect_true(refused(json_shape(list(type = "array", minItems = 2)), list(list(1L))))
  expect_true(refused(json_shape(list(type = "array", uniqueItems = TRUE)), list(list(1L, 1L))))
  expect_true(refused(json_shape(list(type = "string", maxLength = 1)), "ab"))
  expect_false(refused(json_shape(list(type = "string", maxLength = 1)), "\u00e9"))   # one code point, two bytes
  defs <- list(`$defs` = list(Age = list(type = "integer", minimum = 0)), `$ref` = "#/$defs/Age")
  expect_true(refused(json_shape(defs), -1L))
  expect_false(refused(json_shape(defs), 3L))
  # keywords that are lmcc's are carried, never checked here
  expect_false(refused(json_shape(list(type = "string", pattern = "^[a-z]+$")), "ABC"))
  # a value given to a text input that is not text is sent as text, and fits
  expect_false(refused(character(), 5L))
  expect_length(r$env$requests, 7L)
})

test_that("a reply whose value does not fit its type is unreadable: the model is asked again", {
  r <- fake_router(list("<result>\n5\n</result>", "<result>\n12\n</result>"))
  f <- ai(answer ~ text, "x", answer = json_shape(list(type = "integer", minimum = 10)), .router = r, .lm = "gpt-4.1-mini", .log_calls = FALSE)
  expect_identical(unlist(f("Hi")), 12L)
  expect_length(r$env$requests, 2L)
  expect_match(last_text(r$env$requests[[2L]]), "could not be read")
  r <- fake_router(list('<result>\n{"a": "x", "z": 1}\n</result>', '<result>\n{"a": "x"}\n</result>'))
  shape <- list(type = "object", properties = list(a = list(type = "string")), required = list("a"), additionalProperties = FALSE)
  f <- ai(answer ~ text, "x", answer = json_shape(shape), .router = r, .lm = "gpt-4.1-mini", .log_calls = FALSE)
  expect_identical(f("Hi")[[1L]], list(a = "x"))
  expect_length(r$env$requests, 2L)
  r <- fake_router(list("<result>\nABC\n</result>"))                    # an lmcc keyword is not FunctAI's to check
  f <- ai(answer ~ text, "x", answer = json_shape(list(type = "string", pattern = "^[a-z]+$")), .router = r, .lm = "gpt-4.1-mini", .log_calls = FALSE)
  expect_identical(unlist(f("Hi")), "ABC")
})

test_that("a default is of its type, as vctrs casts: nothing is lost on the way", {
  expect_error(defaults_to(2.5, integer()), "is not whole number")
  expect_error(defaults_to(TRUE, character()), "is not text")
  expect_identical(defaults_to(2L, double())$shape$default, 2)
  n <- defaults_to(3L, "how many suggestions to give")               # a sentence describes the value's own type
  expect_identical(n$kind, "integer")
  expect_identical(n$desc, "how many suggestions to give")
  expect_identical(n$shape$default, 3L)
  d <- defaults_to(as.Date("2026-09-28"))                            # a date is text, as a column of dates is
  expect_identical(d$kind, "string")
  expect_identical(d$shape$default, "2026-09-28")
  expect_identical(as_field(as.Date(character()))$kind, "string")
  expect_identical(defaults_to(factor("b", levels = c("a", "b")))$shape$default, "b")
  expect_error(defaults_to(c(1L, 2L)), "not one value")
})

test_that("the first field at fault is named, inputs before outputs, whatever its fault", {
  err <- tryCatch(ai(answer ~ n, "x", n = defaults_to(5L, json_shape(list(type = "integer", minimum = 10))), answer = defaults_to("x")),
                  functai_refusal = identity)
  expect_identical(err$code, "interface-malformed")
  expect_identical(err$field, "n")
  err <- tryCatch(ai(answer ~ n, "x", answer = defaults_to("x")), functai_refusal = identity)
  expect_identical(err$field, "result")
  expect_match(conditionMessage(err), "only an input has a default")
  expect_match(conditionMessage(err), "answer, as the formula names it")
  expect_identical(err$call[[1L]], quote(ai))                        # the error names the call that was refused
})
