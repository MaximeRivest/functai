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

# A record: the members it names and no other (records are closed).
rec_obj <- list(type = "object", properties = list(a = list(type = "string")), required = list("a"))
# An object that names a member and allows others.
open_obj <- c(rec_obj, list(additionalProperties = TRUE))

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

test_that("a member a record does not name is refused, before and after loading, whether it says so or not", {
  for (shape in list(rec_obj, c(rec_obj, list(additionalProperties = FALSE)))) {
    fns <- three_ways(ai(answer ~ options, "x", options = json_shape(shape), .lm = "gpt-4.1-mini"))
    expect_identical(core_of(fns$foreign)$definition$inputs$options$kind, "record")      # a record: a tibble
    refused_input(sends_alike(fns, list(options = list(list(a = "x", extra = "BAD")))))
    refused_input(sends_alike(fns, list(options = tibble::tibble(a = "x", extra = "BAD"))))
    x <- sends_alike(fns, list(options = tibble::tibble(a = "x")))
    expect_identical(plain(x$record$inputs$options), '{"a":"x"}')
  }
})

test_that("a record()'s tibble is sent as it is, and refused with a column the record does not name or without one it does", {
  fns <- three_ways(ai(answer ~ person, "x", person = record(name = character(), age = integer()), .lm = "gpt-4.1-mini"))
  x <- sends_alike(fns, list(person = tibble::tibble(name = "Ann", age = 3)))
  expect_identical(plain(x$record$inputs$person), '{"age":3,"name":"Ann"}')
  x <- sends_alike(fns, list(person = tibble::tibble(name = "Ann", age = 3, email = "ann@example.com")))
  refused_input(x)
  expect_match(x$record$error$message, "no member email is allowed", fixed = TRUE)
  router <- fake_router()
  e <- tryCatch(update(fns$defined, router = router, log_calls = FALSE)(person = tibble::tibble(name = "Ann", age = 3, email = "ann@example.com")),
                functai_interface_input = function(e) e)
  expect_identical(e$field, "person")
  expect_length(router$env$requests, 0L)
  # inside a list, and inside a record in a list
  fns <- three_ways(ai(answer ~ people, "x", people = vctrs::list_of(.ptype = tibble::tibble(name = character(), pet = tibble::tibble(kind = character()))),
                       .lm = "gpt-4.1-mini"))
  ok <- tibble::tibble(name = "Ann", pet = tibble::tibble(kind = "cat"))
  expect_length(sends_alike(fns, list(people = list(ok)))$requests, 1L)
  refused_input(sends_alike(fns, list(people = list(tibble::tibble(name = "Ann", pet = tibble::tibble(kind = "cat", chip = 1L))))))
  # a member it requires but lacks
  fns <- three_ways(ai(answer ~ person, "x", person = record(name = character(), age = integer()), .lm = "gpt-4.1-mini"))
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
  closed <- list(type = "array", items = rec_obj)
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

test_that("a record's answer is a tibble, three ways; an answer with a member it does not name is re-asked", {
  reply <- '<result>{"name":"Ann","age":3}</result>'
  fns <- three_ways(ai(person ~ text, "x", person = record(name = character(), age = integer()), .lm = "gpt-4.1-mini"))
  call <- function(fn, replies = list(reply)) update(fn, router = fake_router(replies), log_calls = FALSE)("hi")
  for (w in names(fns)) {
    expect_identical(core_of(fns[[w]])$definition$outputs$result$kind, "record", info = w)
    expect_identical(call(fns[[w]]), call(fns$defined), info = w)
    expect_s3_class(call(fns[[w]]), "tbl_df")
    # a member the record does not name is not dropped: the reply is unreadable, and the model asked again
    expect_identical(call(fns[[w]], list('<result>{"name":"Ann","age":3,"id":7}</result>', reply)), call(fns$defined), info = w)
  }
  # an open object with the same members stays JSON: a tibble has no column for another member
  fns <- three_ways(ai(answer ~ text, "x", answer = json_shape(open_obj), .lm = "gpt-4.1-mini"))
  expect_identical(core_of(fns$foreign)$definition$outputs$result$kind, "json")
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
  # from another language's folder, the types the shapes hold exactly: records are closed, so tibbles
  expect_identical(unname(kinds(fns$foreign)), c("tibble(p = factor, q = list_of(integer))", "character", "double", "list_of(tibble(k = logical))"))
  # a type that does not fit the shape is not believed
  expect_null(declared_field(list(type = "string"), quote(integer), list()))
  expect_identical(field_from_shape(list(type = "string"), type = quote(tibble(a = character)))$kind, "string")
  expect_identical(field_from_shape(open_obj, type = quote(tibble(a = integer)))$kind, "json")          # a member's type does not fit
  expect_identical(field_from_shape(open_obj, type = quote(tibble(a = character)))$kind, "record")
  expect_identical(field_from_shape(list(type = "string"), type = read_r_type("system('x')"))$kind, "string")   # read, never run
})

# Each answer (JSON), given by each of `fns` and then as the input of a
# function whose input is `field`, sends what the answer's own JSON sends
# there; the answers send different requests (none is read as another).
answers_round_trip <- function(fns, field, answers, name) {
  consumer <- ai(answer ~ value, "x", value = field, .lm = "gpt-4.1-mini")
  sent <- character(0)
  for (json in answers) {
    want <- probe_call(consumer, list(value = list(lmcc::parse_json(json))))
    expect_length(want$requests, 1L)
    sent[[json]] <- request_json(want$requests[[1L]])
    for (w in names(fns)) {
      answer <- update(fns[[w]], router = fake_router(list(paste0("<result>", json, "</result>"))), log_calls = FALSE)("hi")
      got <- probe_call(consumer, list(value = answer))
      info <- paste(name, w, json)
      expect_null(got$record$error$code, info = info)
      expect_length(got$requests, 1L)
      if (length(got$requests)) expect_identical(request_json(got$requests[[1L]]), sent[[json]], info = info)
      expect_identical(plain(to_json(core_of(fns[[w]])$definition$outputs$result, element(answer, 1L))), plain(lmcc::parse_json(json)), info = info)
    }
  }
  expect_false(anyDuplicated(unname(sent)) > 0L, info = name)
}

# An answer given back as an input sends what the answer's own JSON sends:
# a member the record may leave out comes back left out when it was left
# out, and there (with its nulls) when it was there. The consumer is called
# for real, and the requests it sends are compared.
test_that("an answer is an input that sends what its JSON sends: optional lists, objects and records, left out or there", {
  nullable_text <- list(anyOf = list(list(type = "string"), list(type = "null")))
  closed <- function(props, required) list(type = "object", properties = props, required = as.list(required), additionalProperties = FALSE)
  cases <- list(
    `an optional list` = list(shape = closed(list(tag = list(type = "string"), child = list(type = "array", items = list(type = "string"))), "tag"),
                              kind = "json", answers = c('{"tag":"x"}', '{"tag":"x","child":[]}', '{"tag":"x","child":["a"]}')),
    `an optional object` = list(shape = closed(list(tag = list(type = "string"), child = list(type = "object")), "tag"),
                                kind = "json", answers = c('{"tag":"x"}', '{"tag":"x","child":{}}', '{"tag":"x","child":{"k":null}}')),
    `an optional record` = list(shape = closed(list(tag = list(type = "string"), child = closed(list(x = nullable_text), "x")), "tag"),
                                kind = "json", answers = c('{"tag":"x"}', '{"tag":"x","child":{"x":null}}', '{"tag":"x","child":{"x":"y"}}')),
    `an optional choice` = list(shape = closed(list(tag = list(type = "string"), child = list(enum = list("a", "b"), type = "string")), "tag"),
                                kind = "record", answers = c('{"tag":"x"}', '{"tag":"x","child":"b"}')),
    # a required list and record, an optional number: a tibble holds it exactly
    `required structures` = list(shape = closed(list(tag = list(type = "string"), items = list(type = "array", items = list(type = "string")),
                                                     inner = closed(list(x = nullable_text), "x"), n = list(type = "number")), c("tag", "items", "inner")),
                                 kind = "record", answers = c('{"tag":"x","items":[],"inner":{"x":null}}', '{"tag":"x","items":["a"],"inner":{"x":"y"},"n":2.5}')))
  for (name in names(cases)) {
    case <- cases[[name]]
    fns <- three_ways(ai(answer ~ text, "x", answer = json_shape(case$shape), .lm = "gpt-4.1-mini"))
    expect_identical(core_of(fns$foreign)$definition$outputs$result$kind, case$kind, info = name)
    answers_round_trip(fns, json_shape(case$shape), case$answers, name)
  }
  # R's own saved type is not believed where a tibble would lose a member's presence
  shape <- cases[["an optional list"]]$shape
  expect_identical(field_from_shape(shape, type = quote(tibble(tag = character, child = list_of(character))))$kind, "json")
  expect_identical(field_from_shape(cases[["an optional choice"]]$shape, type = quote(tibble(tag = character, child = factor)))$kind, "record")
})

test_that("a default is sent in the order its members were written, before and after loading", {
  f <- ai(answer ~ q + p + o, "x", q = character(),
          p = defaults_to(tibble::tibble(name = "Z", age = 1L, pet = tibble::tibble(zeta = "a", alpha = 2L)),
                          record(name = character(), age = integer(), pet = record(zeta = character(), alpha = integer()))),
          o = defaults_to(list(b = 1L, a = list(d = "x", c = 2.5)), json_shape(list(type = "object"))),
          .lm = "gpt-4.1-mini")
  fns <- three_ways(f)
  for (w in names(fns)) {
    inputs <- ai_interface(fns[[w]])$inputs
    expect_identical(lmcc::json_text(inputs[[2L]]$shape$default), '{"name":"Z","age":1,"pet":{"zeta":"a","alpha":2}}', info = w)
    expect_identical(lmcc::json_text(inputs[[3L]]$shape$default), '{"b":1,"a":{"d":"x","c":2.5}}', info = w)
    expect_identical(lmcc::json_text(core_of(fns[[w]])$definition$inputs$p$shape$default), lmcc::json_text(inputs[[2L]]$shape$default), info = w)
  }
  x <- sends_alike(fns, list(q = "hi"))
  expect_length(x$requests, 1L)
  text <- last_text(x$requests[[1L]])
  expect_match(text, '<p>\n{\n  "name": "Z",\n  "age": 1,\n  "pet": {\n    "zeta": "a",\n    "alpha": 2\n  }\n}\n</p>', fixed = TRUE)
  expect_match(text, '<o>\n{\n  "b": 1,\n  "a": {\n    "d": "x",\n    "c": 2.5\n  }\n}\n</o>', fixed = TRUE)
})

test_that("R reads back a tibble's type whatever its column names: backslashes and backticks too", {
  for (name in c("a\\b", "a\\q", "a`b", "a b", "a\"b")) {
    proto <- tibble::new_tibble(stats::setNames(list(character()), name), nrow = 0L)
    fns <- three_ways(ai(answer ~ text, "x", answer = proto, .lm = "gpt-4.1-mini"))
    expect_identical(r_type_of(core_of(fns$loaded)$definition$outputs$result), r_type_of(core_of(fns$defined)$definition$outputs$result), info = name)
    expect_identical(core_of(fns$loaded)$definition$outputs$result$kind, "record", info = name)
    # the saved type itself reads back (a closed record would be a tibble from its shape alone too)
    saved <- to_manifest(fns$defined)$nodes[[1L]]$interface$outputs[[1L]]
    expect_identical(names(as.list(read_r_type(saved$type))[-1L]), name, info = name)
    expect_false(is.null(declared_field(saved$shape, read_r_type(saved$type), saved$shape)), info = name)
    reply <- paste0("<result>", lmcc::json_text(stats::setNames(list("value"), name)), "</result>")
    call <- function(fn) update(fn, router = fake_router(list(reply)), log_calls = FALSE)("x")
    expect_identical(call(fns$loaded), call(fns$defined), info = name)
  }
})

test_that("an R folder names each field's R type by the field's name", {
  f <- ai(a + b ~ x + y, "x", x = integer(), y = record(k = logical()), a = double(), b = vctrs::list_of(.ptype = character()), .name = "f", .lm = "gpt-4.1-mini")
  iface <- to_manifest(f)$nodes[[1L]]$interface
  types <- function(fields) stats::setNames(vapply(fields, function(g) g$type, ""), vapply(fields, function(g) g$name, ""))
  expect_identical(types(iface$inputs), c(x = "integer", y = "tibble(k = logical)"))
  expect_identical(types(iface$outputs), c(a = "double", b = "list_of(character)"))
})

test_that("an optional record() tells null from a record of nulls, and an answer of either is an input that sends it", {
  # a member that is never null: a row of NA is null (vctrs's missing row), and the record stays a tibble
  field <- optional(record(name = character(), note = optional(character())))
  expect_identical(field$kind, "record")
  fns <- three_ways(ai(answer ~ text, "x", answer = field, .lm = "gpt-4.1-mini"))
  expect_identical(core_of(fns$loaded)$definition$outputs$result$kind, "record")
  answers_round_trip(fns, field, c('null', '{"name":"A","note":null}', '{"name":"A","note":"b"}'), "a member never null")
  consumer <- ai(answer ~ value, "x", value = field, .lm = "gpt-4.1-mini")
  x <- probe_call(consumer, list(value = tibble::tibble(name = NA_character_, note = NA_character_)))
  expect_identical(plain(x$record$inputs), '{"value":null}')
  # where the record may not be null, a row of NA is a record, sent as it is (and refused)
  refused_input(probe_call(ai(answer ~ value, "x", value = record(name = character()), .lm = "gpt-4.1-mini"),
                           list(value = tibble::tibble(name = NA_character_))))
  # every member may be null: a row of NA would be a record too, so the record is a list column
  field <- optional(record(name = optional(character())))
  expect_identical(field$kind, "json")
  fns <- three_ways(ai(answer ~ text, "x", answer = field, .lm = "gpt-4.1-mini"))
  answers_round_trip(fns, field, c('null', '{"name":null}', '{"name":"A"}'), "every member may be null")
  # inside a record
  field <- record(tag = character(), inner = optional(record(x = character())), loose = optional(record(y = optional(integer()))))
  fns <- three_ways(ai(answer ~ text, "x", answer = field, .lm = "gpt-4.1-mini"))
  expect_identical(core_of(fns$loaded)$definition$outputs$result$kind, "record")
  answers_round_trip(fns, field, c('{"tag":"t","inner":null,"loose":null}', '{"tag":"t","inner":{"x":"y"},"loose":{"y":null}}',
                                   '{"tag":"t","inner":{"x":"y"},"loose":{"y":2}}'), "inside a record")
  # R's saved type is not believed for a nullable record whose row of NA would be a record too
  shape <- list(anyOf = list(list(type = "object", properties = list(y = list(anyOf = list(list(type = "integer"), list(type = "null")))), required = list("y")), list(type = "null")))
  expect_identical(field_from_shape(shape, type = quote(tibble(y = integer)))$kind, "json")
})

# The request a consumer sends for a JSON value, rendered from that JSON
# itself: not through the writer that turns R values into JSON, so a writer
# that changed the value would not change what it is compared against.
intended_request <- function(fn, value_json) {
  core <- core_of(update(fn, lm = "probe-model", capabilities = probe_capabilities()))
  request_json(lmcc::lm15_request(probe_render(core, list(value = lmcc::parse_json(value_json))), "probe-model", config_of(effective(core$own))))
}

test_that("a record whose leaves are all null is sent as that record, never as null: given, and as an answer given back", {
  closed <- function(props) list(type = "object", properties = props, required = as.list(names(props)), additionalProperties = FALSE)
  nullable <- function(s) list(anyOf = list(s, list(type = "null")))
  s <- nullable(closed(list(child = closed(list(x = nullable(list(type = "string")))))))
  cases <- list(
    `a nullable record of a record` = list(field = json_shape(s), json = '{"child":{"x":null}}', null = 'null'),
    `an optional record() of a record()` = list(field = optional(record(child = record(x = optional(character())))), json = '{"child":{"x":null}}', null = 'null'),
    `a record without additionalProperties` = list(field = json_shape(nullable(list(type = "object", properties = list(child = list(type = "object", properties = list(x = nullable(list(type = "string"))), required = list("x"))), required = list("child")))),
                                                   json = '{"child":{"x":null}}', null = 'null'),
    `inside a required record` = list(field = json_shape(closed(list(tag = list(type = "string"), obj = s))), json = '{"tag":"t","obj":{"child":{"x":null}}}', null = '{"tag":"t","obj":null}'),
    `inside a list` = list(field = json_shape(list(type = "array", items = s)), json = '[{"child":{"x":null}},null]', null = '[null,null]'))
  for (name in names(cases)) {
    case <- cases[[name]]
    producers <- three_ways(ai(answer ~ text, "x", answer = case$field, .lm = "gpt-4.1-mini"))
    consumers <- three_ways(ai(answer ~ value, "x", value = case$field, .lm = "gpt-4.1-mini"))
    want <- intended_request(consumers$defined, case$json)
    expect_false(identical(want, intended_request(consumers$defined, case$null)), info = name)
    for (p in names(producers)) {
      answer <- update(producers[[p]], router = fake_router(list(paste0("<result>", case$json, "</result>"))), log_calls = FALSE)("hi")
      for (c in names(consumers)) {
        info <- paste(name, p, c)
        for (given in list(answer, list(lmcc::parse_json(case$json)))) {
          got <- probe_call(consumers[[c]], list(value = given))
          expect_length(got$requests, 1L)
          if (length(got$requests)) expect_identical(request_json(got$requests[[1L]]), want, info = info)
          expect_identical(plain(got$record$inputs$value), plain(lmcc::parse_json(case$json)), info = info)
        }
      }
    }
  }
  # a structured member given NA is null there, not a sign that the whole record is missing
  consumer <- ai(answer ~ value, "x", value = json_shape(s), .lm = "gpt-4.1-mini")
  refused_input(probe_call(consumer, list(value = list(list(child = NA)))))
  # R's own values for it: a record of NA, where no member is a witness, is a record of nulls
  consumer <- ai(answer ~ value, "x", value = optional(record(child = record(x = optional(character())))), .lm = "gpt-4.1-mini")
  for (given in list(list(list(child = list(x = NA))), list(list(child = list(x = NULL))), tibble::tibble(child = tibble::tibble(x = NA_character_)))) {
    got <- probe_call(consumer, list(value = given))
    expect_identical(request_json(got$requests[[1L]]), intended_request(consumer, '{"child":{"x":null}}'))
  }
})

test_that("a row of NA is null only where a member the record requires is one value, never null, and NA", {
  field <- optional(record(name = character(), pet = record(x = optional(character()))))
  expect_identical(field$kind, "record")
  fns <- three_ways(ai(answer ~ value, "x", value = field, .lm = "gpt-4.1-mini"))
  null_row <- tibble::tibble(name = NA_character_, pet = tibble::tibble(x = NA_character_))
  x <- sends_alike(fns, list(value = null_row))
  expect_identical(plain(x$record$inputs$value), "null")
  expect_identical(request_json(x$requests[[1L]]), intended_request(fns$defined, "null"))
  # a row with a value in it is a record, sent as it is (and refused: name is not null)
  refused_input(sends_alike(fns, list(value = tibble::tibble(name = NA_character_, pet = tibble::tibble(x = "a")))))
  # JSON's null for the witness is JSON, sent as given (and refused), not read as R's missing row
  refused_input(sends_alike(fns, list(value = list(lmcc::parse_json('{"name":null,"pet":{"x":null}}')))))
  # a json_shape() says the same, so it is read the same: null
  fns <- three_ways(ai(answer ~ value, "x", value = json_shape(field$shape), .lm = "gpt-4.1-mini"))
  expect_identical(plain(sends_alike(fns, list(value = null_row))$record$inputs$value), "null")
})

test_that("a row of NA with a member the record does not name is refused, never sent as null, and the refusal names that member", {
  s <- optional(record(name = character()))
  bad <- list(
    `an NA column` = list(value = tibble::tibble(name = NA_character_, email = NA_character_), member = "email"),
    `an NA member` = list(value = list(list(name = NA_character_, email = NA)), member = "email"),
    `a NULL member` = list(value = list(list(name = NA_character_, id = NULL)), member = "id"),
    `a record of NULL` = list(value = list(list(name = NA_character_, email = list(address = NULL))), member = "email"),
    `a value in it` = list(value = tibble::tibble(name = NA_character_, email = "BAD"), member = "email"),
    `a record with a value` = list(value = tibble::tibble(name = "Ann", email = NA_character_), member = "email"))
  for (field in list(s, json_shape(s$shape), json_shape(list(anyOf = list(c(s$shape$anyOf[[1L]], list(additionalProperties = FALSE)), list(type = "null")))))) {
    fns <- three_ways(ai(answer ~ value, "x", value = field, .lm = "gpt-4.1-mini"))
    for (name in names(bad)) {
      x <- sends_alike(fns, list(value = bad[[name]]$value))
      refused_input(x)
      expect_match(x$record$error$message, sprintf("value: no member %s is allowed", bad[[name]]$member), fixed = TRUE, info = name)
    }
    # the row of NA it names is null; a record is sent as it is
    expect_identical(plain(sends_alike(fns, list(value = tibble::tibble(name = NA_character_)))$record$inputs$value), "null")
    expect_identical(plain(sends_alike(fns, list(value = tibble::tibble(name = "Ann")))$record$inputs$value), '{"name":"Ann"}')
  }
  # inside a list, and inside a required record
  for (field in list(json_shape(list(type = "array", items = s$shape)), record(child = s))) {
    given <- if (field$kind == "record") list(list(child = list(name = NA_character_, extra = NA))) else list(list(list(name = NA_character_, extra = NA)))
    x <- sends_alike(three_ways(ai(answer ~ value, "x", value = field, .lm = "gpt-4.1-mini")), list(value = given))
    refused_input(x)
    expect_match(x$record$error$message, "no member extra is allowed", fixed = TRUE)
  }
  # a member not named inside a record the row names: the row is not null either
  fns <- three_ways(ai(answer ~ value, "x", value = optional(record(name = character(), pet = record(kind = character()))), .lm = "gpt-4.1-mini"))
  x <- sends_alike(fns, list(value = tibble::tibble(name = NA_character_, pet = tibble::tibble(kind = NA_character_, chip = NA_integer_))))
  refused_input(x)
  expect_match(x$record$error$message, "value.pet: no member chip is allowed", fixed = TRUE)
  expect_identical(plain(sends_alike(fns, list(value = tibble::tibble(name = NA_character_, pet = tibble::tibble(kind = NA_character_))))$record$inputs$value), "null")
  # and inside a list in it: the member the record does not name is the one the refusal names
  fns <- three_ways(ai(answer ~ value, "x", value = optional(record(name = character(), pets = vctrs::list_of(.ptype = tibble::tibble(kind = character())))),
                       .lm = "gpt-4.1-mini"))
  x <- sends_alike(fns, list(value = tibble::tibble(name = NA_character_, pets = list(tibble::tibble(kind = "cat", chip = 1L)))))
  refused_input(x)
  expect_match(x$record$error$message, "value.pets[0]: no member chip is allowed", fixed = TRUE)
  # nor is a default: refused when the function is defined
  expect_error(ai(answer ~ value, "x", value = defaults_to(tibble::tibble(name = NA_character_, extra = NA), s), .lm = "gpt-4.1-mini"),
               "no member extra is allowed", class = "functai_interface_malformed")
})

test_that("a row of NA is null only where the shape takes null", {
  # a record that may not be null: a row of NA is the record of nulls it holds, and refused as that
  fns <- three_ways(ai(answer ~ value, "x", value = record(name = character()), .lm = "gpt-4.1-mini"))
  x <- sends_alike(fns, list(value = tibble::tibble(name = NA_character_)))
  refused_input(x)
  expect_identical(plain(x$record$inputs$value), '{"name":null}')
  expect_match(x$record$error$message, "value.name: null is not string", fixed = TRUE)
})

test_that("a record that may be null by a list of types is saved, loaded and called", {
  s <- list(type = list("object", "null"), properties = list(x = list(type = "string")), required = list("x"))
  fns <- three_ways(ai(answer ~ value, "x", value = json_shape(s), .lm = "gpt-4.1-mini"))
  for (json in c('{"x":"a"}', "null")) {
    x <- sends_alike(fns, list(value = list(lmcc::parse_json(json))))
    expect_identical(request_json(x$requests[[1L]]), intended_request(fns$defined, json))
  }
  refused_input(sends_alike(fns, list(value = list(list(x = "a", y = "b")))))
  # its answer comes back, three ways
  producers <- three_ways(ai(answer ~ text, "x", answer = json_shape(s), .lm = "gpt-4.1-mini"))
  answers_round_trip(producers, json_shape(s), c("null", '{"x":"a"}'), "a list of types")
})

test_that("a record as Python writes a dataclass is a tibble from its folder, and gives back what it came as", {
  # Person(name: str, age: int, nick: Optional[str] = None, tags: list[str]): every member required
  text <- list(type = "string")
  person <- list(type = "object", properties = list(name = text, age = list(type = "integer"),
                                                    nick = list(anyOf = list(text, list(type = "null")), default = NULL),
                                                    tags = list(type = "array", items = text)),
                 required = list("name", "age", "nick", "tags"))
  fns <- three_ways(ai(answer ~ text, "x", answer = json_shape(person), .lm = "gpt-4.1-mini"))
  expect_identical(core_of(fns$defined)$definition$outputs$result$kind, "json")         # json_shape(): a list column, as written
  expect_identical(r_type_of(core_of(fns$foreign)$definition$outputs$result), "tibble(name = character, age = integer, nick = character, tags = list_of(character))")
  answers_round_trip(fns, json_shape(person), c('{"name":"A","age":1,"nick":null,"tags":[]}', '{"name":"A","age":1,"nick":"a","tags":["x"]}'), "a dataclass")
  # an optional one, which requires a member that is one value and never null: a row of NA is null
  fns <- three_ways(ai(answer ~ text, "x", answer = json_shape(list(anyOf = list(person, list(type = "null")))), .lm = "gpt-4.1-mini"))
  expect_identical(core_of(fns$foreign)$definition$outputs$result$kind, "record")
  answers_round_trip(fns, json_shape(list(anyOf = list(person, list(type = "null")))), c('null', '{"name":"A","age":1,"nick":null,"tags":[]}'), "an optional dataclass")
})

test_that("an optional record held as a list column takes a one-row tibble for its default, and says it is a record", {
  field <- optional(record(name = optional(character())))
  expect_identical(field$kind, "json")
  expect_identical(type_label(field), "optional record of name (list column)")
  f <- ai(answer ~ q + value, "x", q = character(), value = defaults_to(tibble::tibble(name = "a"), field), .lm = "gpt-4.1-mini")
  expect_identical(lmcc::json_text(ai_interface(f)$inputs[[2L]]$shape$default), '{"name":"a"}')
  fns <- three_ways(f)
  x <- sends_alike(fns, list(q = "hi"))
  expect_identical(plain(x$record$inputs$value), '{"name":"a"}')
  # a one-row tibble is not the default of a value that is not a record
  expect_error(defaults_to(tibble::tibble(name = "a"), json_shape(list(type = "object"))), "default of a record")
})
