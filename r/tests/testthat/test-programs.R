# Programs (contract/programs.md): the contract's module, definitions,
# same-data and message cases, run through a program defined from an
# interface; and programs written the R way, with a formula.

program_cases <- function(kinds) {
  files <- sort(list.files(file.path(contract_root(), "cases", "programs"), pattern = "\\.json$", full.names = TRUE), method = "radix")
  all <- stats::setNames(lapply(files, read_json_file), sub("\\.json$", "", basename(files)))
  Filter(function(c) c$program %in% kinds, all)
}

# A stand-in for a value with no JSON form (cases/README.md): an R object of
# the class its `$type` names, printed as its `$repr`, with text of its own
# (a format method) when it has `$text`.
registerS3method("print", "functai_standin", function(x, ...) cat(attr(x, "repr")), envir = globalenv())
registerS3method("print", "functai_standin_text", function(x, ...) cat(attr(x, "repr")), envir = globalenv())
registerS3method("format", "functai_standin_text", function(x, ...) attr(x, "text"), envir = globalenv())
standin <- function(v) {
  if (is.list(v) && !is.null(names(v)) && all(c("$type", "$repr") %in% names(v)) && length(setdiff(names(v), c("$type", "$repr", "$text"))) == 0L)
    return(structure(list(), class = c(v[["$type"]], if (is.null(v[["$text"]])) "functai_standin" else "functai_standin_text"),
                     repr = v[["$repr"]], text = v[["$text"]]))
  v
}
# One value that is a list is one row of a list column, as an R user gives one; a stand-in is one value.
as_given <- function(v) { v <- standin(v); if (is.list(v) && !is.object(v)) list(v) else v }

sample_given <- function(iface) {
  out <- list()
  for (f in iface$inputs) if (!isTRUE(f$optional)) out[[f$name]] <- if (isTRUE(f$opaque)) 1L else sample_value(f$shape)
  out
}

# Call a program defined from `iface` with `inputs` (JSON), its code
# returning `returned`; what its code got, its outputs, or the refusal.
run_module <- function(iface, inputs, returned = NULL, log_content = NULL) {
  folder <- withr::local_tempdir()
  got <- NULL
  if (missing(returned)) {
    outs <- iface$outputs
    sample <- function(f) if (isTRUE(f$opaque)) 1L else sample_value(f$shape)
    returned <- if (length(outs) == 1L) sample(outs[[1L]]) else stats::setNames(lapply(outs, sample), vapply(outs, function(f) f$name, ""))
  }
  p <- program_from_interface(iface, function(...) { got <<- list(...); standin(returned) }, name = "case")
  if (!is.null(log_content)) { core <- program_core(p); core$own$log_content <- normalize_log_content(log_content); p <- make_program(core) }
  opaque <- vapply(iface$inputs, function(f) isTRUE(f$opaque), NA)
  names(opaque) <- vapply(iface$inputs, function(f) f$name, "")
  given <- lapply(stats::setNames(nm = names(inputs)), function(k) if (isTRUE(opaque[k])) standin(inputs[[k]]) else as_given(inputs[[k]]))
  err <- tryCatch({ with_ai_config(do.call(p, given), log_calls = folder); NULL }, error = identity)
  rec <- log_lines(folder)
  list(error = err, record = if (length(rec)) rec[[1L]] else NULL, got = got)
}

for (name in names(program_cases("module"))) {
  test_that(paste("program case", name), {
    c <- program_cases("module")[[name]]
    p <- tryCatch(program_from_interface(c$interface, function(...) NULL), functai_refusal = identity)
    if (!is.null(c$expect$refuses)) {
      expect_s3_class(p, "functai_interface_malformed"); expect_identical(p$field, c$expect$field)
      return()
    }
    expect_identical(interface_signature(c$interface), c$expect$signature)
    good <- Filter(function(k) !is.null(k$inputs) && !is.null(k$expect$inputs), c$checks)
    valid <- if (length(good)) good[[1L]]$inputs else sample_given(c$interface)
    fine <- Filter(function(k) "returned" %in% names(k) && !is.null(k$expect$outputs), c$checks)
    for (check in c$checks) {
      out <- if ("returned" %in% names(check)) run_module(c$interface, valid, check$returned)
        else if (length(fine)) run_module(c$interface, check$inputs, fine[[1L]]$returned) else run_module(c$interface, check$inputs)
      want <- check$expect
      if (!is.null(want$refuses)) {
        expect_s3_class(out$error, "functai_refusal")
        expect_identical(out$error$code, want$refuses, info = plain(check))
        expect_identical(out$error$field, want$field, info = plain(check))
        expect_identical(out$record$error$type, "InterfaceError")
        expect_null(out$record$outputs)
        if (identical(want$refuses, "interface-input")) expect_null(out$got)          # its code did not run
      } else {
        expect_null(out$error)
        if (!is.null(want$inputs)) expect_identical(plain(out$record$inputs), plain(want$inputs), info = plain(check))
        if (!is.null(want$outputs)) expect_identical(plain(out$record$outputs), plain(want$outputs), info = plain(check))
      }
    }
  })
}

for (name in names(program_cases("same-data"))) {
  test_that(paste("program case", name), {
    c <- program_cases("same-data")[[name]]
    expect_identical(vapply(c$interfaces, interface_signature, ""), unlist(c$expect$signatures))
    for (check in c$checks) for (i in seq_along(c$interfaces)) {
      out <- run_module(c$interfaces[[i]], check$inputs)
      want <- check$expect[[i]]
      if (!is.null(want$refuses)) { expect_identical(out$error$code, want$refuses); expect_identical(out$error$field, want$field) }
      else expect_identical(plain(out$record$inputs), plain(want$inputs))
    }
  })
}

for (name in names(program_cases("message"))) {
  test_that(paste("program case", name), {
    c <- program_cases("message")[[name]]
    for (check in c$checks) {
      out <- run_module(c$interface, check$inputs, log_content = check$log_content)
      want <- check$expect
      expect_identical(out$error$code, want$refuses); expect_identical(out$error$field, want$field)
      msg <- conditionMessage(out$error)
      if (!is.null(want$quotes)) expect_true(grepl(want$quotes, msg, fixed = TRUE), info = msg)
      else {
        v <- check$inputs[[want$field]]
        bits <- c(lmcc::canonical_json(v), if (is.list(v)) v[["$repr"]], if (is_str(v)) v)
        for (b in bits) expect_false(grepl(b, msg, fixed = TRUE), info = msg)
      }
    }
  })
}

for (name in names(program_cases("definitions"))) {
  test_that(paste("program case", name, "(defining a program)"), {
    c <- program_cases("definitions")[[name]]
    for (x in c$interfaces) {
      if (isTRUE(x$ai)) next
      got <- tryCatch({ p <- program_from_interface(x$interface, function(...) NULL); list(signature = interface_signature(ai_interface(p))) },
                      functai_refusal = function(e) list(refuses = e$code, field = e$field))
      expect_identical(plain(got), plain(x$expect), info = plain(x$interface))
    }
  })
}

# ---------------------------------------------------------------- programs written in R

test_that("a program is one call: the AI functions it calls are steps of it, in its tree", {
  folder <- withr::local_tempdir()
  team <- ai(team ~ message, "Which team?", team = choice("billing", "shipping"))
  answer <- ai(reply ~ message + team, "Answer as that team.")
  support <- ai_program(reply ~ message, "Answer a customer's message.", function(message) answer(message, team(message)))
  r <- fake_router(list("<result>\nbilling\n</result>", "<result>\nRefunded.\n</result>"))
  seen <- list()
  out <- with_ai_config(support("Charged twice."), lm = "gpt-4.1-mini", router = r, log_calls = folder,
                        observers = list(function(e) seen[[length(seen) + 1L]] <<- e))
  expect_identical(out, "Refunded.")
  recs <- log_lines(folder)
  expect_length(recs, 3L)
  outer <- Filter(function(x) x$program$kind == "module", recs)[[1L]]
  inner <- Filter(function(x) x$program$kind == "ai", recs)
  expect_true(all(vapply(inner, function(x) identical(x$parent, outer$id) && identical(x$root, outer$id), NA)))
  expect_identical(outer$outputs, list(result = "Refunded."))
  expect_identical(outer$inputs, list(message = "Charged twice."))
  expect_identical(vapply(seen, function(e) e$kind, "")[c(1L, length(seen))], c("started", "done"))
  expect_true(all(vapply(seen, function(e) identical(e$tree, outer$id), NA)))
  expect_identical(seen[[length(seen)]]$value, "Refunded.")
  expect_match(ai_version(support), "^sha256:")
  expect_false(identical(ai_version(support), ai_version(ai_program(reply ~ message, "Answer.", function(message) answer(message, "billing")))))
})

test_that("a program checks what its code returns, and binds what it is given", {
  p <- ai_program(n ~ text, "Count.", function(text) "many", n = integer())
  expect_error(p("abc"), class = "functai_interface_output")
  q <- ai_program(n ~ k, "Double.", function(k) k * 2L, k = integer(), n = integer())
  expect_identical(q("5"), 10L)
  expect_error(q(2.5), class = "functai_interface_input")
  several <- ai_program(a + b ~ x, "Two.", function(x) list(a = x, b = nchar(x)), b = integer(), .name = "two")
  expect_identical(several(c("hi", "hey")), tibble::tibble(a = c("hi", "hey"), b = c(2L, 3L)))
})

test_that("ai_stream() watches a program's call and every call inside it", {
  team <- ai(team ~ message, "Which team?", team = choice("billing", "shipping"))
  support <- ai_program(reply ~ message, "Answer.", function(message) paste("Team:", as.character(team(message))))
  r <- fake_router(list("<result>\nbilling\n</result>"))
  s <- with_ai_config(ai_stream(support, "Charged twice.", .show = FALSE), lm = "gpt-4.1-mini", router = r)
  expect_identical(s$value, "Team: billing")
  kinds <- vapply(s$events, function(e) e$kind, "")
  expect_identical(kinds, c("started", "started", "request", "text", "done", "done"))
  expect_null(s$events[[1L]]$after)
  expect_identical(ai_text(s), "")      # the program's answer has no text events: its helper's does
})

test_that("stop_stream() cancels the call it watches", {
  f <- ai(reply ~ message, "Answer.")
  r <- fake_router(list("<result>\nhello\n</result>"))
  s <- with_ai_config(ai_stream(f, "Hi", .each = function(e) if (e$kind == "request") stop_stream(), .show = FALSE), lm = "gpt-4.1-mini", router = r)
  expect_s3_class(s$error, "functai_cancelled")
  expect_identical(s$events[[length(s$events)]]$kind, "failed")
  expect_identical(s$events[[length(s$events)]]$error$type, "Cancelled")
})

test_that("a program is predicted, evaluated and rated like an AI function", {
  folder <- withr::local_tempdir()
  team <- ai(team ~ message, "Which team?", team = choice("billing", "shipping"))
  route <- ai_program(team ~ message, "Route a message.", function(message) as.character(team(message)), .name = "route")
  r <- fake_router(responder = function(request, i) if (grepl("parcel", last_text(request))) "<result>\nshipping\n</result>" else "<result>\nbilling\n</result>")
  data <- tibble::tibble(message = c("Charged twice", "Where is my parcel?"), team = c("billing", "billing"))
  with_ai_config(lm = "gpt-4.1-mini", router = r, log_calls = folder, {
    p <- predict(route, data)
    expect_identical(p$.pred, c("billing", "shipping"))
    ev <- evaluate(route, data)
    expect_equal(ev$score, 0.5)
  })
  rate(p$.call[[2L]], "wrong", answer = "billing", folder = folder)
  rows <- rated(route, folder = folder)
  expect_identical(rows$team, "billing")
  expect_identical(rows$message, "Where is my parcel?")
})
