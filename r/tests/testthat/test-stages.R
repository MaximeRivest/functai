# The contract's cases of stages 1.2 to 5 and plugins (contract/README.md,
# "Which cases each language passes"): views, the reply cache's key,
# conversations, tools, context, plugins and baked examples.

stage_cases <- function(folder) {
  files <- sort(list.files(file.path(contract_root(), "cases", folder), pattern = "\\.json$", full.names = TRUE), method = "radix")
  stats::setNames(lapply(files, read_json_file), sub("\\.json$", "", basename(files)))
}

for (name in names(stage_cases("views"))) {
  test_that(paste("views case", name), {
    c <- stage_cases("views")[[name]]
    got <- outside_view(c$events, answer_from = c$answer_from)
    expect_identical(plain(lapply(got, unclass)), plain(c$expect$events))
  })
}

for (name in names(stage_cases("replies"))) {
  test_that(paste("replies case", name), {
    c <- stage_cases("replies")[[name]]
    expect_identical(reply_key(c$request, c$replicate), c$expect$key)
  })
}

# ---------------------------------------------------------------- the reply cache (stage 1.2)

test_that("a reply is kept once read, and the same request is answered from it", {
  f <- ai(reply ~ message, "Answer.")
  r <- fake_router(responder = function(request, i) sprintf("<result>\nanswer %d\n</result>", i))
  folder <- withr::local_tempdir()
  db <- file.path(withr::local_tempdir(), "replies.sqlite")
  with_ai_config(lm = "gpt-4.1-mini", router = r, cache_replies = db, log_calls = folder, {
    expect_identical(f(c("a", "b", "a")), c("answer 1", "answer 2", "answer 1"))   # one flight per key: "a" asked once
    expect_identical(f("a"), "answer 1")
    expect_identical(with_ai_config(f("a"), replicate = 1L), "answer 3")          # the second independent answer
  })
  expect_length(r$env$requests, 3L)
  recs <- log_lines(folder)
  cached <- vapply(recs, function(x) isTRUE(x$exchanges[[1L]]$cached), NA)
  expect_identical(sum(cached), 2L)
  expect_equal(Filter(function(x) isTRUE(x$exchanges[[1L]]$cached), recs)[[1L]]$exchanges[[1L]]$seconds, 0)
  # another session reads the same file
  clear_replies()
  r2 <- fake_router(list("<result>\nnew\n</result>"))
  expect_identical(with_ai_config(f("b"), lm = "gpt-4.1-mini", router = r2, cache_replies = db), "answer 2")
  expect_length(r2$env$requests, 0L)
})

test_that("an unreadable reply is never kept; a call whose log drops a field is never kept on disk", {
  f <- ai(n ~ text, "Count.", n = integer())
  r <- fake_router(list("<result>\nmany\n</result>", "<result>\n3\n</result>", "<result>\n4\n</result>"))
  db <- file.path(withr::local_tempdir(), "replies.sqlite")
  with_ai_config(lm = "gpt-4.1-mini", router = r, cache_replies = db, expect_identical(f("abc"), 3L))
  store <- reply_store(db)
  expect_identical(store$size(), 1L)
  g <- ai(n ~ text, "Count.", n = integer(), .log_content = FALSE)
  clear_replies()
  with_ai_config(lm = "gpt-4.1-mini", router = r, cache_replies = db, expect_identical(g("xyz"), 4L))
  expect_identical(store$size(), 1L)
})

test_that("quotes_found() checks a judge's evidence, word for word", {
  source <- "The parcel left Leeds on Monday. It was delayed by snow\u2014badly."
  expect_identical(quotes_found(source, c("\u201cIt was delayed by snow-badly.\u201d", "it  was DELAYED", "It was lost", "")), c(TRUE, TRUE, FALSE, FALSE))
})

test_that("prune_calls() deletes old days and keeps what ratings need", {
  folder <- withr::local_tempdir()
  f <- ai(reply ~ message, "Answer.")
  r <- fake_router(responder = function(request, i) "<result>\nok\n</result>")
  out <- with_ai_config(augment(f, tibble::tibble(message = c("a", "b"))), lm = "gpt-4.1-mini", router = r, log_calls = folder)
  rate(out$.call[[1L]], "right", folder = folder)
  day <- list.files(folder)
  file.rename(file.path(folder, day), file.path(folder, "2020-01-01"))
  before_rows <- rated(f, folder = folder)
  got <- prune_calls("1d", folder = folder)
  expect_identical(got$days, 1L); expect_identical(got$kept, 1L); expect_identical(got$calls, 1L)
  expect_false(dir.exists(file.path(folder, "2020-01-01")))
  expect_identical(rated(f, folder = folder), before_rows)
})

# ---------------------------------------------------------------- conversations (stages 2 to 4)

for (name in names(stage_cases("conversations"))) {
  test_that(paste("conversations case", name), {
    c <- stage_cases("conversations")[[name]]
    now <- unix_of(c$now)
    old <- the$conversation_clock
    the$conversation_clock <- function() now
    on.exit(the$conversation_clock <- old)
    if (c$kind == "state") {
      log <- apply_records(new_conv_log(), c$records)
      e <- c$expect
      states <- lapply(log$turns, turn_state_of)
      expect_identical(plain(states[order(names(states))]), plain(e$states[order(names(e$states))]))
      expect_identical(log$head, e$head)
      head <- if (is.null(log$head)) NULL else log$turns[[log$head]]
      state <- if (is.null(head)) NULL else turn_state_of(head)
      if (!is.null(e$`next`$refuses)) expect_identical(state, "waiting")
      else if (!is.null(e$`next`$waits)) { expect_identical(state, "running"); expect_identical(log$head, e$`next`$waits) }
      else expect_identical(done_on(log, log$head), e$`next`$parent)
      waiting <- Filter(length, lapply(log$turns, function(st) lapply(unanswered(st), function(a) a$invocation)))
      expect_identical(plain(if (length(waiting)) waiting else lmcc::jobj()), plain(e$waiting))
      unf <- Filter(length, lapply(log$turns, function(st) lapply(unfinished(st), function(x) x$invocation)))
      expect_identical(plain(if (length(unf)) unf else lmcc::jobj()), plain(e$unfinished))
    } else {
      tutor <- ai(result ~ message, "Tutor.", .name = "tutor")
      store <- memory_conversations()
      store$append("c", lapply(c$records, function(r) { r$seq <- NULL; r }))
      rule <- c$rule
      with_ai_config(router = fake_router(), lm = "gpt-4.1-mini", {
        chat <- ai_conversation(tutor, "c", store = store, context = structure(list(last = rule$last, without = as.character(unlist(rule$without))), class = "functai_context_rule"))
        conv <- conv_of(chat)
        log <- read_conv(conv)
        for (v in names(log$programs)) log$programs[[v]]$signature <- ai_signature_id(tutor)
        ctx <- turn_context(conv, log, c$parent)
        entries <- lapply(seq_along(ctx$ids), function(i) { e <- list(call = ctx$ids[[i]]); if (length(ctx$turns[[i]]$steps)) e$steps <- TRUE; e })
        expect_identical(plain(ctx$finish(entries)), plain(c$expect$saw))
        expect_identical(plain(ctx$rows), plain(c$expect$rows))
      })
    }
  })
}

for (name in names(stage_cases("tools"))) {
  test_that(paste("tools case", name), {
    c <- stage_cases("tools")[[name]]
    if (c$kind == "asks") {
      rule <- if (identical(c$rule, "function")) function(a) TRUE else if (is.list(c$rule)) check_approve(as.character(unlist(c$rule))) else c$rule
      got <- vapply(c$approvals, function(a) asks(rule, list(name = a$name, path = a$path, effects = a$effects)), NA)
      expect_identical(got, unlist(c$expect$asks))
    } else expect_identical(denial(c$reason), c$expect$output)
  })
}

for (name in names(stage_cases("plugins"))) {
  test_that(paste("plugins case", name), {
    c <- stage_cases("plugins")[[name]]
    if (c$kind == "order") {
      made <- list()
      layers <- lapply(c$layers, function(l) {
        ps <- lapply(unlist(l$plugins), function(n) { if (is.null(made[[n]])) made[[n]] <<- ai_plugin(n); made[[n]] })
        s <- list(plugins = ps)
        if (l$where == "configure" && isFALSE(c$program_plugins)) s$program_plugins <- FALSE
        list(where = l$where, s = s)
      })
      expect_identical(vapply(plugins_in_order(layers), function(p) p$name, ""), as.character(unlist(c$expect$order)))
      return()
    }
    hook <- c$hook; start <- c$start; want <- c$expect
    ran <- 0L
    plugins <- lapply(seq_along(c$changes), function(i) {
      ch <- c$changes[[i]]
      change <- if (is.null(ch)) NULL else do.call(ai_change, ch)
      do.call(ai_plugin, c(list(paste0("p", i)), stats::setNames(list(function(e) { ran <<- ran + 1L; change }), hook)))
    })
    read_tool <- ai_tool(function(...) "r", "Read.", .name = "read")
    write_tool <- ai_tool(function(...) "w", "Write.", .name = "write")
    f <- ai(result ~ x, "Answer.", .name = "f", .tools = list(read_tool, write_tool))
    with_ai_config(plugins = plugins, router = fake_router(), lm = "gpt-4.1-mini", {
      if (hook == "before_call") {
        core <- core_of(f)
        s <- effective(core$own); s$lm <- start$lm
        sig <- lmcc::signature_to_list(signature_of(core, s)); sig$instructions <- start$instruction
        plan <- bind_layout(NULL, NULL, lmcc::signature_from_list(sig), probe_capabilities(), "openai")
        call <- new.env(); call$site <- "f#1"; call$changes <- list(); call$saw <- list()
        got <- before_call_hooks(core, call, s, list(x = "1"), plan, list())
        instr <- lmcc::signature_to_list((got$plan %||% plan)$signature)$instructions
        expect_identical(instr, paste(c(want$instruction, unlist(want$sections)), collapse = "\n\n"))
        expect_identical((got$settings %||% s)$lm, want$lm)
        expect_identical(if (is.null(got$tools)) NULL else vapply(got$tools, function(t) t$name, ""), if (is.null(want$tools)) NULL else as.character(unlist(want$tools)))
        for (k in names(want$settings)) expect_equal((got$settings %||% s)[[k]] %||% (got$settings %||% s)$config[[k]], want$settings[[k]])
      } else if (hook == "context") {
        store <- memory_conversations()
        desc <- program_record(f); desc$version <- "v"
        recs <- list(desc); parent <- NULL
        for (t in unlist(start$keep)) {
          tr <- list(functai_conversation = 1L, kind = "turn", at = "2026-09-30T10:00:00.000000Z", turn = t); tr["parent"] <- list(parent); tr$program <- "v"; tr$inputs <- list(x = t)
          recs <- c(recs, list(tr, list(functai_conversation = 1L, kind = "ended", at = "2026-09-30T10:00:00.000000Z", turn = t, state = "done", outputs = list(result = "r", photo = "P", notes = "N"))))
          parent <- t
        }
        store$append("c", recs)
        chat <- ai_conversation(f, "c", store = store)
        got <- shown_turns_of(conv_of(chat), read_conv(conv_of(chat)), parent, list())
        expect_identical(vapply(got$picked, turn_id, ""), as.character(unlist(want$keep)))
        expect_identical(got$sections, as.character(unlist(want$sections)))
        expect_identical(plain(if (length(got$without)) lapply(got$without, as.list) else lmcc::jobj()), plain(want$without))
      } else if (hook == "turn_start") {
        g <- ai(result ~ message + tone, "G.", .name = "g")
        got <- turn_start_hooks(conv_of(ai_conversation(g)), start$inputs, list())
        expect_identical(plain(got$given), plain(want$inputs))
      } else {
        job <- new.env(); job$core <- core_of(f); job$settings <- list()
        call <- new.env(); call$site <- "f#1"; call$name <- "f"; call$id <- "c"; call$changes <- list(); call$invocations <- 1L
        call$tree <- new_tree_log("c"); job$call <- call
        cc <- list(id = "t1", name = "send", input = start$inputs %||% lmcc::jobj())
        if (hook == "tool_call") {
          got <- tool_gate(job, list(name = "send", effects = "changes"), cc, 1L)
          if (!is.null(want$block)) expect_true(grepl(want$block, got$output, fixed = TRUE))
          else expect_identical(plain(got$input), plain(want$inputs))
        } else expect_identical(tool_result_hooks(job, NULL, cc, list(), start$output), want$output)
      }
    })
    expect_identical(ran, as.integer(want$ran))
  })
}
