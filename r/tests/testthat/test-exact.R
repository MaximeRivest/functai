# Loading a saved function changes no value: what a call sends, and which
# values it refuses, are the same before write_ai() and after read_ai(); an
# answer comes back whole. Each function is called for real, against a fake
# provider that keeps the requests it receives, and those requests are
# compared.

# A function three ways: as defined; saved and loaded by R (which reads back
# the R types it saved: the interface's `type`); and the same folder as
# another language's (no R types), whose fields R reads as the types that
# hold their shapes' values exactly.
three_ways <- function(f) {
  dir <- withr::local_tempdir()
  write_ai(f, dir)
  m <- read_json_file(file.path(dir, "functai.json"))
  m$language <- "python"
  for (k in names(m$nodes)) for (side in c("inputs", "outputs"))
    m$nodes[[k]]$interface[[side]] <- lapply(m$nodes[[k]]$interface[[side]], function(x) x[names(x) != "type"])
  other <- withr::local_tempdir()
  writeLines(lmcc::json_text(m), file.path(other, "functai.json"), useBytes = TRUE)
  list(defined = f, loaded = read_ai(dir), foreign = read_ai(other))
}

# Calls each function with `args` under the probe facts; each must bind,
# refuse and send exactly as the first. Returns the first's record and
# requests.
sends_alike <- function(fns, args) {
  got <- lapply(fns, probe_call, args = args)
  first <- got[[1L]]
  for (w in names(got)[-1L]) {
    g <- got[[w]]
    expect_identical(plain(g$record$inputs), plain(first$record$inputs), info = w)
    expect_identical(plain(g$record$error), plain(first$record$error), info = w)
    expect_identical(length(g$requests), length(first$requests), info = w)
    if (length(first$requests) && length(g$requests))
      expect_identical(request_json(g$requests[[1L]]), request_json(first$requests[[1L]]), info = w)
  }
  first
}

refused_input <- function(x) {
  expect_identical(x$record$error$code, "interface-input")
  expect_length(x$requests, 0L)
}

open_obj <- list(type = "object", properties = list(a = list(type = "string")), required = list("a"))

test_that("a member an open object shape does not name is sent as given, before and after loading", {
  fns <- three_ways(ai(answer ~ options, "x", options = json_shape(open_obj), .lm = "gpt-4.1-mini"))
  x <- sends_alike(fns, list(options = list(list(a = "x", extra = "DO_NOT_DROP"))))
  expect_identical(plain(x$record$inputs$options), '{"a":"x","extra":"DO_NOT_DROP"}')
  expect_length(x$requests, 1L)
  for (fn in fns) expect_sends(fn, list(options = list(list(a = "x", extra = "DO_NOT_DROP"))),
                               bound = list(options = list(a = "x", extra = "DO_NOT_DROP")))
})

test_that("a required member left out is refused, before and after loading, even where it takes null", {
  shape <- list(type = "object", properties = list(a = list(anyOf = list(list(type = "string"), list(type = "null")))), required = list("a"))
  fns <- three_ways(ai(answer ~ options, "x", options = json_shape(shape), .lm = "gpt-4.1-mini"))
  refused_input(sends_alike(fns, list(options = list(lmcc::jobj()))))
  x <- sends_alike(fns, list(options = list(list(a = NULL))))           # given as null: sent as null
  expect_identical(plain(x$record$inputs$options), '{"a":null}')
})

test_that("a member a closed shape does not allow is refused, before and after loading", {
  shape <- c(open_obj, list(additionalProperties = FALSE))
  fns <- three_ways(ai(answer ~ options, "x", options = json_shape(shape), .lm = "gpt-4.1-mini"))
  expect_identical(core_of(fns$foreign)$definition$inputs$options$kind, "record")      # a closed record: a tibble
  refused_input(sends_alike(fns, list(options = list(list(a = "x", extra = "BAD")))))
  refused_input(sends_alike(fns, list(options = tibble::tibble(a = "x", extra = "BAD"))))
  x <- sends_alike(fns, list(options = tibble::tibble(a = "x")))
  expect_identical(plain(x$record$inputs$options), '{"a":"x"}')
})

test_that("a record()'s tibble is sent as it is: extra columns kept, a missing column refused", {
  fns <- three_ways(ai(answer ~ person, "x", person = record(name = character(), age = integer()), .lm = "gpt-4.1-mini"))
  x <- sends_alike(fns, list(person = tibble::tibble(name = "Ann", age = 3, id = 7L)))
  expect_identical(plain(x$record$inputs$person), '{"age":3,"id":7,"name":"Ann"}')
  refused_input(sends_alike(fns, list(person = tibble::tibble(name = "Ann"))))
  refused_input(sends_alike(fns, list(person = tibble::tibble(name = "Ann", age = NA_integer_))))   # null is not a whole number
  refused_input(sends_alike(fns, list(person = tibble::tibble(name = "Ann", age = 2.5))))
})

test_that("objects inside lists are sent as given, and checked where they are", {
  shape <- list(type = "array", items = open_obj)
  fns <- three_ways(ai(answer ~ items, "x", items = json_shape(shape), .lm = "gpt-4.1-mini"))
  x <- sends_alike(fns, list(items = list(list(list(a = "x", more = 1L), list(a = "y")))))
  expect_identical(plain(x$record$inputs$items), '[{"a":"x","more":1},{"a":"y"}]')
  refused_input(sends_alike(fns, list(items = list(list(list(a = "x"), list(b = "y"))))))
  closed <- list(type = "array", items = c(open_obj, list(additionalProperties = FALSE)))
  fns <- three_ways(ai(answer ~ items, "x", items = json_shape(closed), .lm = "gpt-4.1-mini"))
  refused_input(sends_alike(fns, list(items = list(list(list(a = "x", more = 1L))))))
  # one item, from a vector, is still a list
  fns <- three_ways(ai(answer ~ items, "x", items = json_shape(list(type = "array", items = list(type = "string"))), .lm = "gpt-4.1-mini"))
  x <- sends_alike(fns, list(items = list("only")))
  expect_identical(plain(x$record$inputs$items), '["only"]')
})

test_that("in a record, NA for a member it may leave out and that takes no null is left out", {
  shape <- list(type = "object", additionalProperties = FALSE, required = list("a", "note"),
                properties = list(a = list(type = "string"), n = list(type = "integer"), note = list(anyOf = list(list(type = "string"), list(type = "null")))))
  fns <- three_ways(ai(answer ~ o, "x", o = json_shape(shape), .lm = "gpt-4.1-mini"))
  expect_identical(core_of(fns$foreign)$definition$inputs$o$kind, "record")
  x <- sends_alike(fns, list(o = tibble::tibble(a = "x", n = NA_integer_, note = NA_character_)))
  expect_identical(plain(x$record$inputs$o), '{"a":"x","note":null}')          # n left out; note, required, is null
  y <- sends_alike(fns, list(o = list(list(a = "x", note = NULL))))
  expect_identical(request_json(y$requests[[1L]]), request_json(x$requests[[1L]]))
})

test_that("an answer comes back whole: an open object keeps every member, before and after loading", {
  reply <- '<result>{"a":"x","extra":"DO_NOT_DROP"}</result>'
  fns <- three_ways(ai(answer ~ text, "x", answer = json_shape(open_obj), .lm = "gpt-4.1-mini"))
  for (w in names(fns)) {
    got <- update(fns[[w]], router = fake_router(list(reply)), log_calls = FALSE)("hi")
    expect_identical(plain(got[[1L]]), '{"a":"x","extra":"DO_NOT_DROP"}', info = w)
  }
})

test_that("a record()'s answer is a tibble before and after R loads it; another language's open record is JSON", {
  reply <- '<result>{"name":"Ann","age":3}</result>'
  fns <- three_ways(ai(person ~ text, "x", person = record(name = character(), age = integer()), .lm = "gpt-4.1-mini"))
  call <- function(fn) update(fn, router = fake_router(list(reply)), log_calls = FALSE)("hi")
  expect_identical(call(fns$defined), call(fns$loaded))
  expect_s3_class(call(fns$loaded), "tbl_df")
  expect_identical(core_of(fns$loaded)$definition$outputs$result$kind, "record")
  # no R type to go by: an open record may have members a tibble has no column for
  foreign <- call(fns$foreign)
  expect_type(foreign, "list")
  expect_identical(plain(foreign[[1L]]), '{"age":3,"name":"Ann"}')
})

test_that("a closed record's answer is a tibble that gives back the JSON it came from", {
  shape <- list(type = "object", additionalProperties = FALSE, required = list("a", "note"),
                properties = list(a = list(type = "string"), n = list(type = "integer"), note = list(anyOf = list(list(type = "string"), list(type = "null")))))
  fns <- three_ways(ai(answer ~ text, "x", answer = json_shape(shape), .lm = "gpt-4.1-mini"))
  f <- fns$foreign
  answer <- core_of(f)$definition$outputs$result
  expect_identical(answer$kind, "record")
  for (json in c('{"a":"x","note":null}', '{"a":"x","n":2,"note":"y"}')) {
    got <- update(f, router = fake_router(list(paste0("<result>", json, "</result>"))), log_calls = FALSE)("hi")
    expect_s3_class(got, "tbl_df")
    expect_identical(plain(to_json(answer, element(got, 1L))), plain(lmcc::parse_json(json)), info = json)
  }
})

test_that("R reads back the R types it saved, and another language's names are its own", {
  f <- ai(a + b + c + d ~ x, "x", .name = "f",
          a = record(p = choice("u", "v"), q = vctrs::list_of(.ptype = integer())),
          b = json_shape(list(type = "string")),
          c = optional(double()),
          d = vctrs::list_of(.ptype = tibble::tibble(k = logical())))
  fns <- three_ways(f)
  kinds <- function(fn) vapply(core_of(fn)$definition$outputs, r_type_of, "")
  expect_identical(kinds(fns$loaded), kinds(fns$defined))
  expect_identical(unname(kinds(fns$defined)), c("tibble(p = factor, q = list_of(integer))", "list", "double", "list_of(tibble(k = logical))"))
  # a record() is open: from another language's folder it is JSON; the rest is as the shape says
  expect_identical(unname(kinds(fns$foreign)), c("list", "character", "double", "list_of(list)"))
  # a type that does not fit the shape is not believed
  expect_null(declared_field(list(type = "string"), quote(integer), list()))
  expect_identical(field_from_shape(list(type = "string"), type = quote(tibble(a = character)))$kind, "string")
  expect_identical(field_from_shape(open_obj, type = quote(tibble(a = integer)))$kind, "json")          # a member's type does not fit
  expect_identical(field_from_shape(open_obj, type = quote(tibble(a = character)))$kind, "record")
  expect_identical(field_from_shape(list(type = "string"), type = read_r_type("system('x')"))$kind, "string")   # read, never run
})
