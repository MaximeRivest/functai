# Conversations, tools that ask first, plugins: R's own tests, with a fake model.

reply_text <- function(text) sprintf("<result>\n%s\n</result>", text)
msg_count <- function(request) length(lmcc::lm15_plain(lm15::as_dict(request))$messages)

test_that("a conversation's turns remember each other, are kept, and branch", {
  tutor <- ai(reply ~ message, "Tutor.", .name = "tutor")
  r <- fake_router(responder = function(request, i) reply_text(sprintf("answer %d", i)))
  folder <- withr::local_tempdir(); logs <- withr::local_tempdir()
  with_ai_config(lm = "gpt-4.1-mini", router = r, log_calls = logs, {
    chat <- ai_conversation(tutor, "alex", store = folder)
    expect_identical(chat("Hi, I'm Alex."), "answer 1")
    expect_identical(chat("What is 1/2 + 1/3?"), "answer 2")
    expect_identical(msg_count(r$env$requests[[2L]]), 3L)                 # the first turn, then the question
    t <- ai_turns(chat)
    expect_identical(t$state, c("done", "done"))
    expect_identical(t$parent, c(NA, t$turn[[1L]]))
    # the turn's id is its call's id in the call log; its record says so
    recs <- log_lines(logs)
    second <- Filter(function(x) identical(x$id, t$turn[[2L]]), recs)[[1L]]
    expect_identical(second$conversation$id, "alex")
    expect_identical(second$saw, list(list(call = t$turn[[1L]], steps = TRUE)))
    expect_false(is.null(second$steps))
    # the same line tomorrow opens it again
    again <- ai_conversation(tutor, "alex", store = folder)
    expect_identical(nrow(ai_turns(again)), 2L)
    other <- continue_from(again, 1L)
    expect_identical(other("Another question"), "answer 3")
    expect_identical(msg_count(r$env$requests[[3L]]), 3L)                 # only the first turn
    expect_identical(nrow(ai_turns(again, all = TRUE)), 3L)
    expect_identical(ai_turns(other)$parent[[2L]], t$turn[[1L]])
    # last_turns(0): nothing earlier
    fresh <- ai_conversation(tutor, "alex", store = folder, context = last_turns(0))
    fresh("Hello?")
    expect_identical(msg_count(r$env$requests[[4L]]), 1L)
    expect_length(ai_render(chat, "next?")$messages, 5L)          # both turns, then the question
  })
  expect_error(ai_conversation(tutor, "a/b"), class = "functai_conversation_id")
})

test_that("a turn waits for a person's answer, then goes on paying for nothing twice", {
  refunds <- character(0)
  refund <- function(order) { refunds <<- c(refunds, order); "refunded" }
  helper <- ai(reply ~ message, "Help.", .name = "helper", .tools = list(ai_tool(refund, "Refund an order.", .effects = "changes")))
  r <- fake_router(responder = function(request, i) {
    if (msg_count(request) == 1L) list(calls = list(list(id = "c1", name = "refund", input = list(order = "A-1042"))))
    else reply_text("Done: refunded.")
  })
  folder <- withr::local_tempdir()
  with_ai_config(lm = "gpt-4.1-mini", router = r, {
    chat <- ai_conversation(helper, "shop", store = folder, approve = "changes")
    w <- tryCatch(chat("Refund A-1042 please."), functai_waiting = identity)
    expect_s3_class(w, "functai_waiting")
    expect_identical(w$turn$state, "waiting")
    expect_identical(w$approvals[[1L]]$name, "refund")
    expect_length(refunds, 0L)
    expect_error(chat("another?"), class = "functai_conversation_busy")
    # later, from another conversation object on the same store
    t <- ai_turn(ai_conversation(helper, "shop", store = folder, approve = "changes"))
    expect_identical(approve(t), "Done: refunded.")
    expect_identical(refunds, "A-1042")
    expect_length(r$env$requests, 2L)                                        # the first reply was kept: not asked again
    expect_identical(ai_turn(chat)$state, "done")
  })
  # denied: the model is told, and may answer otherwise
  r2 <- fake_router(responder = function(request, i) {
    if (msg_count(request) == 1L) list(calls = list(list(id = "c1", name = "refund", input = list(order = "B-1"))))
    else reply_text(sprintf("seen: %s", grepl("did not allow", paste(unlist(lmcc::lm15_plain(lm15::as_dict(request))), collapse = " "))))
  })
  with_ai_config(lm = "gpt-4.1-mini", router = r2, {
    chat <- ai_conversation(helper, "shop2", store = folder, approve = "changes")
    w <- tryCatch(chat("Refund B-1."), functai_waiting = identity)
    expect_identical(deny(w$turn, reason = "not today"), "seen: TRUE")
  })
  expect_identical(refunds, "A-1042")
  # a plain call has nobody to ask
  r3 <- fake_router(responder = function(request, i) list(calls = list(list(id = "c1", name = "refund", input = list(order = "C-1")))))
  expect_error(with_ai_config(helper("Refund C-1."), lm = "gpt-4.1-mini", router = r3, approve = "changes", tool_errors = "raise"), class = "functai_approval_required")
  # a function answers at once
  r4 <- fake_router(responder = function(request, i) if (msg_count(request) == 1L) list(calls = list(list(id = "c1", name = "refund", input = list(order = "D-1")))) else reply_text("ok"))
  expect_identical(with_ai_config(helper("Refund D-1."), lm = "gpt-4.1-mini", router = r4, approve = function(a) a$input$order == "D-1"), "ok")
  expect_identical(refunds, c("A-1042", "D-1"))
})

test_that("plugins change calls as data, recorded", {
  logs <- withr::local_tempdir()
  careful <- ai_plugin("careful", version = "1.2.0", before_call = function(call) ai_change(sections = "Only point out problems."))
  f <- ai(reply ~ message, "Review.", .name = "review")
  r <- fake_router(responder = function(request, i) reply_text("fine"))
  with_ai_config(f("x"), lm = "gpt-4.1-mini", router = r, plugins = list(careful), log_calls = logs)
  sys <- lmcc::lm15_plain(lm15::as_dict(r$env$requests[[1L]]))$system
  expect_match(paste(unlist(sys), collapse = " "), "Only point out problems.", fixed = TRUE)
  rec <- log_lines(logs)[[1L]]
  expect_identical(rec$changes[[1L]]$plugin, "careful")
  expect_identical(rec$changes[[1L]]$change$sections, list("Only point out problems."))
  expect_error(ai_plugin("Bad Name"), class = "functai_plugin_name")
  expect_error(on_hook(careful, "befor_call", function(e) NULL), class = "functai_plugin_hook")
  bad <- ai_plugin("bad", before_call = function(call) stop("boom"))
  expect_error(with_ai_config(f("x"), lm = "gpt-4.1-mini", router = r, plugins = list(bad)), "plugin bad failed")
  guard <- ai_plugin("guard", tool_call = function(t) if (t$name == "wipe") ai_change(block = "not here"))
  wiped <- FALSE
  g <- ai(reply ~ message, "Act.", .name = "act", .tools = list(ai_tool(function() { wiped <<- TRUE; "gone" }, "Wipe.", .name = "wipe")))
  r2 <- fake_router(responder = function(request, i) if (msg_count(request) == 1L) list(calls = list(list(id = "c1", name = "wipe", input = list()))) else reply_text("ok"))
  expect_identical(with_ai_config(g("go"), lm = "gpt-4.1-mini", router = r2, plugins = list(guard)), "ok")
  expect_false(wiped)
})

test_that("a program's conversation gives its code the conversation so far", {
  seen <- list()
  echo <- ai(reply ~ message, "Echo.", .name = "echo")
  bot <- ai_program(reply ~ message, "Answer.", function(message) { seen[[length(seen) + 1L]] <<- earlier(); echo(message) })
  r <- fake_router(responder = function(request, i) reply_text(sprintf("echo %d", i)))
  with_ai_config(lm = "gpt-4.1-mini", router = r, {
    chat <- ai_conversation(bot, "p")
    chat("one"); chat("two")
  })
  expect_length(seen[[1L]], 0L)
  expect_identical(seen[[2L]][[1L]]$message, "one")
  expect_identical(seen[[2L]][[1L]]$result, "echo 1")
  expect_identical(msg_count(r$env$requests[[2L]]), 1L)        # the helper remembers nothing unless told
})
