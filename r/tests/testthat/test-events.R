# The contract's events/ cases (contract/streaming.md, format 2): replaying
# and following logs, their kept form, the rules a store keeps, a writer
# keeping a log in a journal that fails, and which observers and journal a
# tree gets from the layers around it.

event_cases <- function() {
  files <- sort(list.files(file.path(contract_root(), "cases", "events"), pattern = "\\.json$", full.names = TRUE), method = "radix")
  stats::setNames(lapply(files, read_json_file), sub("\\.json$", "", basename(files)))
}
positions <- function(evs) lapply(evs, function(e) list(writer = e$writer, seq = e$seq))
pos_json <- function(p) if (is.null(p)) NULL else list(writer = p$writer, seq = p$seq)
read_answer <- function(f) tryCatch(list(events = positions(f())), functai_store_refusal = function(e) list(refuses = e$code))
same <- function(a, b) expect_identical(plain(a), plain(b))

# A journal whose transport follows a script (cases/README.md, `journal`).
scripted_store <- function(script) {
  inner <- memory_store()
  s <- new.env()
  s$inner <- inner; s$script <- script; s$trace <- list()
  next_word <- function(tree) {
    repeat {
      word <- if (length(s$script)) s$script[[1L]] else "ok"
      if (length(s$script)) s$script <- s$script[-1L]
      if (!word %in% c("claimed", "ended")) return(word)
      claim <- tryCatch(inner$claim(tree), functai_store_refusal = function(e) NULL)
      s$trace[[length(s$trace) + 1L]] <- list(other = word, writer = claim$writer)
      if (word == "ended" && !is.null(claim)) {
        kept <- inner$read(tree)
        seq <- claim$after$seq + 1L
        at <- sprintf("2026-09-28T10:00:%02d.%06dZ", 50L + seq %/% 100L, (seq %% 100L) * 10000L)
        name <- Filter(function(e) e$kind == "started" && e$call == tree, kept)[[1L]][["function"]]
        inner$append(new_event("failed", tree, claim$writer, seq, claim$after, at, tree, name, list(error = list(type = "Cancelled"))))
      }
    }
  }
  s$append <- function(e) {
    word <- next_word(e$tree)
    if (word == "down") { s$trace[[length(s$trace) + 1L]] <- list(seq = e$seq, transport = word, answer = NULL); stop("unreachable") }
    if (word == "conflict") {
      s$trace[[length(s$trace) + 1L]] <- list(seq = e$seq, transport = word, answer = "event-conflict")
      store_refuse("event-conflict", "another writer has the log", list(writer = e$writer, seq = e$seq))
    }
    answer <- tryCatch(inner$append(e), functai_store_refusal = function(err) err)
    s$trace[[length(s$trace) + 1L]] <- list(seq = e$seq, transport = word, answer = if (word == "lost") NULL else if (inherits(answer, "condition")) answer$code else answer)
    if (word == "lost") stop("the answer was lost")
    if (inherits(answer, "condition")) stop(answer)
    answer
  }
  s$claim <- function(tree) inner$claim(tree)
  s$read <- function(tree, after = NULL) inner$read(tree, after)
  s
}

case_seconds <- function(at) as.numeric(as.POSIXct(substr(at, 1, 19), format = "%Y-%m-%dT%H:%M:%S", tz = "UTC")) + as.numeric(substr(at, 21, 26)) / 1e6

run_journal_case <- function(c) {
  events <- lapply(c$events, as_event)
  start <- events[[1L]]; outcome <- events[[length(events)]]
  tree <- start$tree
  j <- scripted_store(c$script)
  times <- vapply(events, function(e) case_seconds(e$at), 0)
  tap <- new.env(); tap$events <- list()
  dir <- withr::local_tempdir()
  program <- start$program
  fields <- list(inputs = names(start$inputs), outputs = program$answer, added = character(0))
  keep <- stats::setNames(rep(TRUE, length(fields$inputs) + 1L), c(fields$inputs, fields$outputs))
  raised <- NULL
  with_ai_config(journal = ai_journal(j, required = c$mode == "required", retries = c$retries), log_calls = dir, {
    old <- the$scripted
    the$scripted <- list(id = tree, clock = function(seq) times[[seq]], tap = tap)
    on.exit(the$scripted <- old)
    call <- start_call(function() program, effective(), start$inputs, fields, keep)
    raised <- tryCatch({
      run_call(call, function(call) {
        for (e in events[-c(1L, length(events))]) {
          data <- event_data(e)
          if (e$kind == "tool_call") tool_called(call, data) else call_emit(call, e$kind, data)
        }
        if (outcome$kind == "done") outcome$value
        else stop(structure(class = c(outcome$error$type, "error", "condition"), list(message = outcome$error$message, call = NULL)))
      })
      NULL
    }, error = identity)
  })
  rec <- log_lines(dir)[[1L]]
  record <- list(error = if (is.null(rec$error)) NULL else rec$error[names(rec$error) != "message"])
  if (!is.null(rec$journal)) record$journal <- rec$journal
  settled <- NULL
  caller <- if (inherits(raised, "functai_journal_error") && identical(raised$code, "journal-end")) {
    if (identical(raised$journal, "unknown")) settled <- settle(raised)
    out <- if (!is.null(raised$outcome$done) || "done" %in% names(raised$outcome)) list(done = raised$outcome$done)
      else list(failed = error_json(raised$outcome$failed)[c("type", "code")])
    out$failed <- if (!is.null(out$failed)) Filter(Negate(is.null), out$failed) else NULL
    list(raises = list(type = "JournalError", code = raised$code, journal = raised$journal, event = pos_json(raised$event), outcome = out))
  } else if (is.null(raised)) list(returns = outcome$value)
  else list(raises = Filter(Negate(is.null), error_json(raised)[c("type", "code")]))
  kept <- if (tree %in% j$inner$trees()) j$inner$read(tree) else list()
  shown <- tap$shown
  got <- list(log = lapply(tap$events, unclass), trace = j$trace, kept = list(events = positions(kept), finished = j$inner$finished(tree)),
              caller = caller, record = record)
  if (!is.null(settled)) got$settled <- settled
  for (k in setdiff(names(c$expect), "shown")) same(got[[k]], c$expect[[k]])
}

run_receivers_scenario <- function(scenario) {
  stores <- list()
  journal <- function(x) if (is.null(x)) FALSE else {
    if (is.null(stores[[x$name]])) stores[[x$name]] <<- memory_store(x$name)
    ai_journal(stores[[x$name]], required = x$mode == "required")
  }
  seen <- new.env()
  observer <- function(name) { force(name); function(e) seen[[name]] <- c(seen[[name]], list(e)) }
  layer_settings <- function(where) {
    l <- Filter(function(l) l$where == where, scenario$layers)
    if (!length(l)) return(list())
    l <- l[[1L]]; out <- list()
    if (!is.null(l$observers)) out$observers <- lapply(l$observers, observer)
    if ("journal" %in% names(l)) out["journal"] <- list(journal(l$journal))
    if (!is.null(l$program_observers)) out$program_observers <- l$program_observers
    out
  }
  own <- layer_settings("own"); block <- layer_settings("block"); conf <- layer_settings("configure")
  if (length(own)) names(own) <- paste0(".", names(own))
  f <- do.call(ai, c(list(team ~ message, "Which team handles this?"), own))
  old <- the$config
  on.exit(the$config <- old)
  do.call(ai_config, conf)
  want <- scenario$expect
  r <- fake_router(list("<result>\nbilling\n</result>"))
  err <- rlang::inject(with_ai_config(tryCatch({ f("I was charged twice."); NULL }, error = identity), !!!block, lm = "gpt-4.1-mini", router = r))
  if (!is.null(want$refuses)) {
    expect_s3_class(err, "functai_journal_error"); expect_identical(err$code, want$refuses)
    expect_length(r$env$requests, 0L)
  } else expect_null(err)
  observed <- sort(ls(seen))
  expect_identical(observed, sort(unlist(want$observers) %||% character(0)))
  for (n in unlist(want$observers)) {
    kinds <- vapply(seen[[n]], function(e) e$kind, "")
    expect_identical(kinds[[1L]], "started"); expect_identical(kinds[[length(kinds)]], if (is.null(want$refuses)) "done" else "failed")
  }
  for (n in names(stores)) {
    logged <- stores[[n]]$trees()
    if (!is.null(want$journal) && identical(want$journal$name, n)) {
      log <- stores[[n]]$read(logged[[1L]])
      if (!is.null(want$refuses)) {
        expect_identical(vapply(log, function(e) e$kind, ""), c("started", "failed"))
        expect_identical(log[[2L]]$error$code, "journal-policy")
      } else expect_identical(log[[length(log)]]$kind, "done")
    } else expect_length(logged, 0L)
  }
}

for (name in names(event_cases())) {
  test_that(paste("events case", name), {
    c <- event_cases()[[name]]
    kind <- sub("-.*$", "", name)
    if (kind == "replay") {
      state <- new_log_state(); views <- list()
      for (e in c$events) { state <- replay_event(state, as_event(e)); views[[length(views) + 1L]] <- list(calls = state_json(state)$calls) }
      same(list(views = views, finished = state$finished), c$expect)
      for (r in c$resume) same(read_answer(function() resume_events(c$events, r$after)), r$expect)
    } else if (kind == "follow") {
      f <- follower(if (is.null(c$recover)) "kept" else c$recover$reader)
      results <- character(0)
      for (e in c$received) { r <- follow_event(f, e); results <- c(results, r); if (r == "unknown-format") break }
      expect_identical(results, unlist(c$expect$results))
      states <- lapply(stats::setNames(nm = names(f$trees)), function(t) follow_state(f, t))
      same(states, c$expect$state)
      if (!is.null(c$recover)) {
        tree <- names(f$trees)[[1L]]
        from <- if (identical(c$recover$from, "store")) "store" else as.integer(c$recover$from)
        reads <- follow_recover(f, tree, c$recover$source, from)
        got <- lapply(reads, function(r) list(after = pos_json(r$after), expect = if (inherits(r$answer, "condition")) list(refuses = r$answer$code) else list(events = positions(r$answer))))
        same(got, c$expect$recover$reads)
        same(follow_state(f, tree), c$expect$recover$state)
      }
    } else if (kind == "kept") {
      keeps <- lapply(c$kept, function(k) new_keep(k$inputs, k$outputs))
      same(lapply(kept_log(c$events, keeps), unclass), c$expect$events)
    } else if (kind == "store") {
      store <- memory_store()
      for (step in c$steps) {
        got <- if (!is.null(step$append)) tryCatch(store$append(step$append), functai_store_refusal = function(e) e$code)
          else if (!is.null(step$batch)) tryCatch(store$append(step$batch), functai_store_refusal = function(e) list(refuses = e$code, event = pos_json(e$event)))
          else tryCatch({ cl <- store$claim(step$claim); list(writer = cl$writer, after = pos_json(cl$after)) }, functai_store_refusal = function(e) list(refuses = e$code))
        same(got, step$expect)
      }
      for (r in c$reads) same(read_answer(function() store$read(r$tree, r$after)), r$expect)
      logs <- lapply(stats::setNames(nm = store$trees()), function(t) list(events = positions(store$read(t)), writer = store$writer_of(t), finished = store$finished(t)))
      same(if (length(logs)) logs else lmcc::jobj(), c$expect$logs)
    } else if (kind == "journal") {
      if (c$mode == "best-effort" && any(unlist(c$script) != "ok")) expect_warning(run_journal_case(c), "best-effort journal")
      else run_journal_case(c)
    } else if (kind == "receivers") {
      for (scenario in c$scenarios) run_receivers_scenario(scenario)
    }
  })
}

test_that("the contract has events cases of every kind", {
  kinds <- unique(sub("-.*$", "", names(event_cases())))
  expect_true(all(c("replay", "follow", "kept", "store", "journal", "receivers") %in% kinds))
})
