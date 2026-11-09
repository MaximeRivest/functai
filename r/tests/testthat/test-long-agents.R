# Long agents and pipelines, setups and declared notes, on a fake model: tools
# that answer with pictures, the step hook and its built-ins (limits,
# keep_pictures), the call log in a store of one's own, results that come
# later, notes from code (declared and checked), what a program runs on, what
# it may reach, and what code spent.

PNG <- openssl::base64_encode(as.raw(c(0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x66, 0x61, 0x6b, 0x65)))
LM <- "gpt-4.1-mini"
la_xml <- function(v) sprintf("<result>\n%s\n</result>", v)
request_plain <- function(req) lmcc::lm15_plain(lm15::as_dict(req))
tool_results <- function(req) Filter(function(p) identical(p$type, "tool_result"), unlist(lapply(request_plain(req)$messages, function(m) m$parts), recursive = FALSE))
pictures_in <- function(req) sum(vapply(tool_results(req), function(p) sum(vapply(p$content, function(x) identical(x$type, "image"), NA)), 0L))
texts_of_result <- function(p) vapply(Filter(function(x) identical(x$type, "text"), p$content), function(x) x$text, "")
offered <- function(req) length(request_plain(req)$tools)

# A model that asks for these tool calls, one per reply while it is offered tools, then answers.
script <- function(steps, answer = function(req) "done") fake_router(responder = function(request, i) {
  n <- length(tool_results(request))
  if (n < length(steps) && offered(request)) list(calls = list(list(id = paste0("c", n + 1L), name = steps[[n + 1L]]$name, input = steps[[n + 1L]]$input)))
  else la_xml(answer(request))
})
looks <- function(k) lapply(seq_len(k) - 1L, function(i) list(name = "look", input = list(page = as.character(i))))

look <- function(page) list(paste("page", page), lm15::image_part(data = PNG, media_type = "image/png"))
browse <- ai(result ~ task, "Look at pages to do the task.", .name = "browse",
             .tools = list(ai_tool(look, "A screenshot of a page, with its title.", .effects = "reads")))

test_that("a tool answers with a picture: the model sees it, the event has its parts", {
  r <- script(looks(1L))
  s <- with_ai_config(ai_stream(browse, "find the price", .show = FALSE), router = r, lm = LM)
  expect_identical(s$value, "done")
  last <- tool_results(r$env$requests[[length(r$env$requests)]])
  expect_identical(vapply(last[[1L]]$content, function(x) x$type, ""), c("text", "image"))
  expect_identical(last[[1L]]$content[[2L]]$data, PNG)
  res <- Filter(function(e) e$kind == "tool_result", s$events)[[1L]]
  expect_identical(res$output, "page 0")
  expect_identical(plain(res$parts), plain(list(list(type = "text", text = "page 0"), list(type = "image", media_type = "image/png", data = PNG))))
  kept <- kept_event(res, new_keep(list(task = FALSE), list(result = TRUE)), list(kind = "ai", answer = "result"))
  expect_null(kept$parts); expect_null(kept$output)
})

test_that("a picture from a file is given its bytes; a value is its JSON, as before", {
  file <- withr::local_tempfile(fileext = ".png")
  writeBin(as.raw(c(0x89, 0x50, 0x4e, 0x47, 0x66, 0x69, 0x6c, 0x65)), file)
  screenshot <- function() lm15::image_part(path = file, media_type = "image/png")
  facts <- function() list(a = 1L, list = list("x"))
  watch <- ai(result ~ task, "Watch.", .name = "watch",
              .tools = list(ai_tool(screenshot, "The screen."), ai_tool(facts, "Facts.")))
  r <- script(list(list(name = "screenshot", input = list()), list(name = "facts", input = list())))
  s <- with_ai_config(ai_stream(watch, "look", .show = FALSE), router = r, lm = LM)
  expect_identical(s$value, "done")
  got <- tool_results(r$env$requests[[length(r$env$requests)]])
  expect_identical(got[[1L]]$content[[1L]]$data, openssl::base64_encode(readBin(file, "raw", 8L)))
  big <- withr::local_tempfile(fileext = ".png")
  writeBin(as.raw(rep(0:255, 4L)), big)                    # 1 KB: a picture providers read only as base64 on one line
  wide <- result_of(lm15::image_part(path = big, media_type = "image/png"))[[1L]]$data
  expect_false(grepl("\n", wide, fixed = TRUE))
  expect_identical(openssl::base64_decode(wide), as.raw(rep(0:255, 4L)))
  expect_null(got[[1L]]$content[[1L]]$path)
  res <- Filter(function(e) e$kind == "tool_result", s$events)
  expect_identical(res[[2L]]$output, '{"a":1,"list":["x"]}')
  expect_null(res[[2L]]$parts)
})

test_that("a plugin sees a picture result and changes it", {
  seen <- list()
  p <- ai_plugin("caption", tool_result = function(t) { seen[[length(seen) + 1L]] <<- t$output; ai_change(output = c(t$output, list("(the price is at the top right)"))) })
  r <- script(looks(1L))
  expect_identical(with_ai_config(update(browse, plugins = list(p))("find"), router = r, lm = LM), "done")
  expect_identical(seen[[1L]][[2L]]$type, "image")
  expect_identical(vapply(tool_results(r$env$requests[[2L]])[[1L]]$content, function(x) x$type, ""), c("text", "image", "text"))
})

test_that("a turn keeps a picture result, and goes on with it", {
  clicked <- character(0)
  click <- function(where) { clicked <<- c(clicked, where); list("clicked", lm15::image_part(data = PNG, media_type = "image/png")) }
  agent <- ai(result ~ task, "Do it.", .name = "agent", .tools = list(ai_tool(click, "Click, and see the screen.", .effects = "changes")))
  store <- memory_conversations()
  with_ai_config(router = script(list(list(name = "click", input = list(where = "buy")))), lm = LM, {
    chat <- ai_conversation(agent, "pictures", store = store, approve = "click")
    expect_s3_class(tryCatch(chat("buy it"), functai_waiting = identity), "functai_waiting")
    expect_identical(approve(ai_turn(chat)), "done")
  })
  expect_identical(clicked, "buy")
  done <- Filter(function(x) identical(x$kind, "tool") && identical(x$state, "done"), store$read("pictures", 0L))
  expect_identical(plain(done[[1L]]$output[[2L]]), plain(list(type = "image", media_type = "image/png", data = PNG)))
})

# ---------------------------------------------------------------- the step hook

test_that("the step hook sees each step, and wraps up", {
  seen <- list()
  p <- ai_plugin("wrap", step = function(s) {
    seen[[length(seen) + 1L]] <<- list(s$step, s$max_steps, lapply(s$results, function(x) x$invocation), as.list(s$tools), s$spent$requests)
    if (s$step == 3L) ai_change(tools = character(0), reminder = "Answer now.")
  })
  r <- script(looks(10L))
  expect_identical(with_ai_config(update(browse, plugins = list(p), max_steps = 5L)("find"), router = r, lm = LM), "done")
  expect_identical(plain(seen), plain(list(list(2L, 5L, list(1L), list("look"), 1L), list(3L, 5L, list(1L, 2L), list("look"), 2L))))
  last <- r$env$requests[[length(r$env$requests)]]
  expect_identical(offered(last), 0L)                                   # no tool offered: the model must answer
  results <- tool_results(last)
  expect_identical(utils::tail(texts_of_result(results[[length(results)]]), 1L), "Answer now.")
})

test_that("earlier results shown otherwise: the record says so, and the call stays replayable", {
  p <- ai_plugin("forget", version = "2.0.0", step = function(s) {
    old <- Filter(function(x) any(vapply(x$output, function(q) is.list(q) && identical(q$type, "image"), NA)), utils::head(s$results, -1L))
    if (length(old)) ai_change(replace = stats::setNames(rep(list("(an earlier screenshot)"), length(old)), vapply(old, function(x) as.character(x$invocation), "")))
  })
  r <- script(looks(3L))
  logs <- withr::local_tempdir()
  expect_identical(with_ai_config(update(browse, plugins = list(p))("find"), router = r, lm = LM, log_calls = logs), "done")
  expect_identical(vapply(r$env$requests, pictures_in, 0L), c(0L, 1L, 1L, 1L))            # only the newest picture is sent
  rec <- read_log(logs)$calls[[1L]]
  steps <- Filter(function(ch) ch$hook == "step", rec$changes)
  expect_identical(plain(steps[[1L]]), plain(list(plugin = "forget", version = "2.0.0", hook = "step", change = list(replace = list(`1` = "(an earlier screenshot)")))))
  expect_true(all(vapply(rec$exchanges, function(ex) !is.null(ex$request_hash), NA)))     # nothing was rewritten
})

test_that("a step handler stops the call; a change step cannot make is refused", {
  brake <- ai_plugin("brake", step = function(s) if (s$step >= 3L) ai_change(stop = "too many pages"))
  err <- tryCatch(with_ai_config(update(browse, plugins = list(brake))("find"), router = script(looks(10L)), lm = LM), error = identity)
  expect_s3_class(err, "functai_plugin_stopped")
  expect_match(conditionMessage(err), "too many pages")
  bad <- ai_plugin("bad", step = function(s) ai_change(tools = "fly"))
  err <- tryCatch(with_ai_config(update(browse, plugins = list(bad))("find"), router = script(looks(2L)), lm = LM), error = identity)
  expect_identical(err$code, "plugin-change")
  worse <- ai_plugin("worse", step = function(s) ai_change(sections = "no"))
  err <- tryCatch(with_ai_config(update(browse, plugins = list(worse))("find"), router = script(looks(2L)), lm = LM), error = identity)
  expect_identical(err$code, "plugin-change")
})

# ---------------------------------------------------------------- limits and keep_pictures

test_that("limits: the last step answers instead of failing; a token limit wraps up, then stops the next call", {
  r <- script(looks(10L))
  expect_identical(with_ai_config(update(browse, plugins = list(limits(seconds = 3600)), max_steps = 3L)("find"), router = r, lm = LM), "done")
  expect_length(r$env$requests, 3L)
  last <- r$env$requests[[3L]]
  expect_identical(offered(last), 0L)
  results <- tool_results(last)
  expect_identical(utils::tail(texts_of_result(results[[length(results)]]), 1L), LAST_STEP)
  r <- script(looks(10L))
  limit <- limits(tokens = 1)                                          # reached after the first request
  expect_identical(with_ai_config(update(browse, plugins = list(limit))("find"), router = r, lm = LM), "done")
  expect_length(r$env$requests, 2L)
  results <- tool_results(r$env$requests[[2L]])
  expect_match(utils::tail(texts_of_result(results[[length(results)]]), 1L), "^This task has reached its limit \\(1 tokens\\)")
  twice <- ai_program(result ~ task, "Twice.", function(task) paste(browse(task), browse(task)))
  err <- tryCatch(with_ai_config(twice("find"), router = script(looks(10L)), lm = LM, plugins = list(limit)), error = identity)
  expect_s3_class(err, "functai_plugin_stopped")
  expect_match(conditionMessage(err), "limit reached: 1 tokens", fixed = TRUE)
})

test_that("a cost limit needs prices, and counts a model with no price as reached", {
  expect_error(limits(cost = 1))
  expect_error(limits())
  limit <- limits(cost = 1, prices = list(`some-other-model` = list(input_per_million = 1, output_per_million = 2)))
  err <- tryCatch(with_ai_config(update(browse, plugins = list(limit))("find"), router = script(looks(1L)), lm = LM), error = identity)
  expect_match(conditionMessage(err), "no price is known for gpt-4.1-mini", fixed = TRUE)    # refused up front
  flat <- list(input_per_million = 1, output_per_million = 2, cache_read_per_million = 1, cache_write_per_million = 1)
  priced <- limits(cost = 1000, prices = stats::setNames(list(flat), LM), then = "stop")
  expect_identical(with_ai_config(update(browse, plugins = list(priced))("find"), router = script(looks(1L)), lm = LM), "done")
  s <- ai_spent(list(list("m", list(input_tokens = 1e6, output_tokens = 1e6, total_tokens = 2e6, cache_read_tokens = 0, cache_write_tokens = 0))), 1)
  expect_equal(ai_cost(s, list(m = list(input_per_million = 3, output_per_million = 15))), 18)
  expect_null(ai_cost(s, list(other = list(input_per_million = 3, output_per_million = 15))))
  expect_equal(ai_spent()$tokens, 0)
})

test_that("keep_pictures shows only the newest, dropping them in batches when asked", {
  r <- script(looks(5L))
  expect_identical(with_ai_config(update(browse, plugins = list(keep_pictures(2)))("find"), router = r, lm = LM), "done")
  expect_identical(vapply(r$env$requests, pictures_in, 0L), c(0L, 1L, 2L, 2L, 2L, 2L))
  expect_identical(texts_of_result(tool_results(r$env$requests[[6L]])[[1L]]), c("page 0", PICTURE_GONE))
  r <- script(looks(7L))
  expect_identical(with_ai_config(update(browse, plugins = list(keep_pictures(2, at_once = 3)), max_steps = 10L)("find"), router = r, lm = LM), "done")
  expect_identical(vapply(r$env$requests, pictures_in, 0L), c(0L, 1L, 2L, 3L, 4L, 2L, 3L, 4L))
})

# ---------------------------------------------------------------- the call log in a store of one's own

test_that("calls go to a store of one's own, and are read back, rated", {
  lines <- character(0)
  store <- list(append = function(record) lines <<- c(lines, lmcc::json_text(record)), read = function() lapply(lines, lmcc::parse_json))
  team <- ai(team ~ message, "Which team answers this message.")
  with_ai_config(router = fake_router(list(la_xml("billing"))), lm = LM, log_calls = store, {
    team("I was charged twice.")
    rate(calls(team, folder = store)$id, "right", folder = store)
  })
  recs <- store$read()
  expect_identical(vapply(recs, function(x) !is.null(x$functai_call), NA), c(TRUE, FALSE))
  expect_identical(nrow(rated(team, folder = store)), 1L)
  expect_error(prune_calls(folder = store), "store of one's own")
  broken <- list(append = function(record) stop("database away"), read = function() list())
  expect_warning(out <- with_ai_config(team("x"), router = fake_router(list(la_xml("billing"))), lm = LM, log_calls = broken), "could not log")
  expect_identical(out, "billing")
  expect_error(ai_config(log_calls = 1), "log_calls")
  mem <- memory_calls()
  with_ai_config(router = fake_router(list(la_xml("a"), la_xml("b"))), lm = LM, log_calls = mem, { team("1"); team("2") })
  expect_identical(vapply(mem$read(), function(x) x$outputs$result, ""), c("a", "b"))
})

# ---------------------------------------------------------------- results that come later

jobs <- new.env(); jobs$started <- character(0)
start_harvest <- function(url) { jobs$started <- c(jobs$started, url); ai_later(paste0("harvest:", length(jobs$started))) }
weather <- function(city) paste("sunny in", city)
researcher <- ai(result ~ task, "Research the site.", .name = "researcher",
                 .tools = list(ai_tool(start_harvest, "Visit a site; its report comes back later.", .effects = "changes"),
                               ai_tool(weather, "Today's weather.", .effects = "reads")))

test_that("a turn waits for a result that comes later, and goes on when it is delivered", {
  jobs$started <- character(0)
  store <- memory_conversations()
  r <- script(list(list(name = "start_harvest", input = list(url = "a.com")), list(name = "weather", input = list(city = "Oslo"))),
              answer = function(req) paste("report:", paste(texts_of_result(tool_results(req)[[1L]]), collapse = "")))
  with_ai_config(router = r, lm = LM, {
    chat <- ai_conversation(researcher, "later-1", store = store)
    w <- tryCatch(chat("look at a.com"), functai_waiting = identity)
    expect_s3_class(w, "functai_waiting")
    expect_identical(vapply(w$later, function(x) x$key, ""), "harvest:1")
    expect_length(w$approvals, 0L)
    expect_identical(ai_turn(chat)$state, "waiting")
    asked <- length(r$env$requests)
    again <- ai_conversation(researcher, "later-1", store = store)            # the webhook, as another process
    expect_identical(deliver(again, "harvest:1", list("3 pages", lm15::image_part(data = PNG, media_type = "image/png"))), "report: 3 pages")
    expect_identical(jobs$started, "a.com")                                    # the tool did not run again
    expect_identical(length(r$env$requests) - asked, 2L)                      # no model answer paid twice: weather, then the answer
    expect_null(deliver(again, "harvest:1", "late duplicate"))                # a webhook sent twice changes nothing
  })
  kinds <- vapply(Filter(function(x) x$kind %in% c("tool", "delivered"), store$read("later-1", 0L)), function(x) paste(x$kind, x$state %||% ""), "")
  expect_identical(kinds, c("tool started", "tool later", "delivered ", "tool done", "tool done"))
})

test_that("a result delivered before the turn waits is used at once; outside a turn, refused", {
  jobs$started <- character(0)
  store <- memory_conversations()
  with_ai_config(router = script(list(list(name = "start_harvest", input = list(url = "b.com"))),
                                 answer = function(req) { rs <- tool_results(req); paste("report:", paste(texts_of_result(rs[[length(rs)]]), collapse = "")) }), lm = LM, {
    chat <- ai_conversation(researcher, "later-2", store = store)
    expect_null(deliver(chat, "harvest:1", "fast", resume = FALSE))         # the webhook came first
    expect_identical(chat("look"), "report: fast")
  })
  err <- tryCatch(with_ai_config(researcher("look"), router = script(list(list(name = "start_harvest", input = list(url = "c.com")))), lm = LM), error = identity)
  expect_identical(err$code, "later-unavailable")
  store <- memory_conversations()
  with_ai_config(router = script(list(list(name = "start_harvest", input = list(url = "d.com")))), lm = LM, {
    chat <- ai_conversation(researcher, "later-3", store = store)
    expect_s3_class(tryCatch(chat("look"), functai_waiting = identity), "functai_waiting")
    err <- tryCatch(resume_turn(ai_turn(chat)), error = identity)
    expect_identical(err$code, "turn-state")
    expect_match(conditionMessage(err), "harvest:3", fixed = TRUE)
  })
})

# ---------------------------------------------------------------- notes

write_up <- ai(result ~ x, "Write.", .name = "write_up")

test_that("code says what happened; declared notes are checked where they are made", {
  pipeline <- ai_program(result ~ url, "Pipeline.", function(url) { ai_note("screen captured", list(screen = 4L)); write_up(url) }, .name = "pipeline")
  s <- with_ai_config(ai_stream(pipeline, "a.com", .show = FALSE), router = fake_router(list(la_xml("ok"))), lm = LM)
  expect_identical(s$value, "ok")
  n <- Filter(function(e) e$kind == "note", s$events)[[1L]]
  expect_identical(plain(list(n$name, n$data, n[["function"]])), plain(list("screen captured", list(screen = 4L), "pipeline")))
  expect_null(ai_note("nothing watches"))
  expect_error(ai_note(" "))
  walk <- ai_program(result ~ url, "Walk a site.", function(url) {
    ai_note("stage", "walking"); ai_note("page captured", list(url = url, title = "Home")); ai_note("stage", "writing"); write_up(url)
  }, .notes = list(stage = note_type(choice("walking", "writing"), latest = TRUE, outside = TRUE),
                   `page captured` = note_type(json_shape(list(type = "object", properties = list(url = list(type = "string"), title = list(type = "string")),
                                                               required = list("url", "title"), additionalProperties = FALSE)))), .name = "walk")
  iface <- ai_interface(walk)
  expect_identical(vapply(iface$notes, function(n) n$name, ""), c("stage", "page captured"))
  s <- with_ai_config(ai_stream(walk, "a.com", .show = FALSE), router = fake_router(list(la_xml("ok"))), lm = LM)
  expect_identical(s$value, "ok")
  outside <- outside_view(s$events, notes = outside_notes(walk))
  expect_identical(vapply(Filter(function(e) e$kind == "note", outside), function(e) e$name, ""), c("stage", "stage"))
  state <- replay_log(s$events)
  expect_identical(vapply(state$calls[[s$events[[1L]]$call]]$notes, function(n) n$name, ""), c("stage", "page captured", "stage"))
  strict <- ai_program(result ~ x, "Strict.", function(x) { ai_note(x, "walk"); "done" }, .notes = list(stage = choice("walking", "writing")))
  err <- tryCatch(strict("stage"), functai_refusal = identity)
  expect_identical(c(err$code, err$field), c("interface-note", "stage"))
  err <- tryCatch(strict("stage changed"), functai_refusal = identity)
  expect_identical(c(err$code, err$field), c("interface-note", "stage changed"))
  # an AI function's notes are its tools': two tools that declare one name differently refuse it
  a <- function() "" ; b <- function() ""
  t1 <- ai_tool(a, "A.", .notes = list(page = character()))
  t2 <- ai_tool(b, "B.", .notes = list(page = integer()))
  one <- ai(result ~ x, "One.", .name = "one", .tools = list(t1))
  expect_identical(ai_interface(one)$notes[[1L]]$name, "page")
  err <- tryCatch(ai(result ~ x, "Two.", .name = "two", .tools = list(t1, t2)), functai_refusal = identity)
  expect_identical(c(err$code, err$field), c("interface-malformed", "page"))
})

test_that("the answer so far is read by an AI function's answer's shape", {
  insights <- ai(result ~ page, "Read the page.", .name = "insights",
                 result = record(summary = character(), severity = choice("low", "medium", "high")))
  got <- answer_so_far('{"summary": "Check', insights)
  expect_identical(plain(got), plain(list(summary = "Check")))
  expect_identical(plain(answer_so_far('{"summary": "x", "severity": "med', insights)), plain(list(summary = "x")))
  expect_null(answer_so_far("Sure!", insights))
})

# ---------------------------------------------------------------- setups (calls.md, Setups)

test_that("a setup: what a program runs on, beside what it is", {
  shipped <- function(order) "shipped"
  lost <- function(order) "LOST"
  a <- ai(result ~ order, "Track.", .name = "track", .tools = list(ai_tool(shipped, "Where is the order?", .name = "track_order")))
  b <- ai(result ~ order, "Track.", .name = "track", .tools = list(ai_tool(lost, "Where is the order?", .name = "track_order")))
  with_ai_config(lm = LM, {
    expect_identical(ai_version(a), ai_version(b))
    expect_false(identical(ai_setup(a), ai_setup(b)))                    # rewriting a tool: another setup, the same version
    expect_identical(ai_setup_document(a)$tools$track_order$code$language, "r")
    expect_false(identical(ai_setup(a), with_ai_config(ai_setup(a), lm = "claude-haiku-4-5")))   # another model
    v1 <- ai(result ~ order, "Track.", .name = "track", .tools = list(ai_tool(shipped, "Where is the order?", .name = "track_order", .code_id = "v1")))
    v2 <- ai(result ~ order, "Track.", .name = "track", .tools = list(ai_tool(shipped, "Where is the order?", .name = "track_order", .code_id = "v2")))
    expect_false(identical(ai_setup(v1), ai_setup(v2)))
    outer <- ai(result ~ q, "Outer.", .name = "outer", .tools = list(ai_tool(write_up, "Write.")))
    expect_identical(ai_setup_document(outer)$tools$write_up$setup, ai_setup(write_up))
    expect_false(identical(ai_setup(update(a, plugins = list(keep_pictures(2)))), ai_setup(update(a, plugins = list(keep_pictures(3))))))
    walk <- ai_program(result ~ url, "Walk.", function(url) write_up(url))
    doc <- ai_setup_document(walk)
    expect_identical(names(doc$ai), "__main__:write_up")
    expect_identical(doc$reach, "open")
    # code a version cannot read, named by the host: in every setup worked out under it, a program's too
    expect_null(doc$code_id)
    expect_identical(with_ai_config(ai_setup_document(walk)$code_id, code_id = "sha256:aaaa"), "sha256:aaaa")
    expect_identical(with_ai_config(ai_setup_document(a)$code_id, code_id = "sha256:aaaa"), "sha256:aaaa")
    expect_false(identical(ai_setup(walk), with_ai_config(ai_setup(walk), code_id = "sha256:aaaa")))
    expect_false(identical(with_ai_config(ai_setup(a), code_id = "a"), with_ai_config(ai_setup(a), code_id = "b")))
    expect_identical(ai_version(a), with_ai_config(ai_version(a), code_id = "a"))
  })
  for (bad in list("", "   ", strrep("x", 201), 7)) expect_error(with_ai_config(1, code_id = bad), "code_id")
})

test_that("reach: open, or behind a wall; a wall that lets through less refuses before any request", {
  fetch_page <- function(url) "page"
  walker <- ai(result ~ url, "Walk.", .name = "walker",
               .tools = list(ai_tool(fetch_page, "Fetch a page.", .needs = list(domains = "API.Browser-Use.com", secrets = "BROWSER_USE_API_KEY"))))
  with_ai_config(lm = LM, {
    expect_identical(ai_setup_document(walker)$reach, "open")
    expect_identical(plain(with_ai_config(ai_setup_document(walker)$reach, wall = "srt")),
                     plain(list(domains = list("api.browser-use.com"), secrets = list("BROWSER_USE_API_KEY"))))
  })
  r <- fake_router(list(la_xml("x")))
  logs <- withr::local_tempdir()
  err <- tryCatch(with_ai_config(walker("a.com"), router = r, lm = LM, log_calls = logs,
                                 wall = list(name = "bubblewrap", domains = "*.example.com", secrets = character(0))), error = identity)
  expect_identical(err$code, "needs-unmet")
  expect_identical(unlist(err$missing$domains), "api.browser-use.com")
  expect_length(r$env$requests, 0L)
  rec <- read_log(logs)$calls[[1L]]
  expect_identical(rec$error$code, "needs-unmet")
  expect_identical(rec$placement$wall, "bubblewrap")
  expect_identical(with_ai_config(walker("a.com"), router = fake_router(list(la_xml("x"))), lm = LM,
                                  wall = list(name = "bubblewrap", domains = "*.browser-use.com", secrets = "BROWSER_USE_API_KEY")), "x")
})

test_that("a record says its setup, where it ran, and what code spent; limits count it", {
  paid_visit <- function(url) { ai_spend(0.05); paste("visited", url) }
  visitor <- ai(result ~ url, "Visit the site.", .name = "visitor", .tools = list(ai_tool(paid_visit, "Visit with a paid browser.", .name = "visit", .effects = "reads")))
  logs <- withr::local_tempdir()
  with_ai_config(router = script(list(list(name = "visit", input = list(url = "a.com")))), lm = LM, log_calls = logs,
                 base_url = "https://gateway.example/v1", expect_identical(visitor("a.com"), "done"))
  rec <- read_log(logs)$calls[[1L]]
  expect_identical(rec$program$setup, with_ai_config(ai_setup(visitor), lm = LM))
  expect_identical(plain(rec$placement), plain(list(base_url = "https://gateway.example/v1")))
  expect_equal(rec$spent$usd, 0.05)
  expect_null(schema_fault(rec, "call.schema.json"))
  flat <- list(input_per_million = 0, output_per_million = 0, cache_read_per_million = 0, cache_write_per_million = 0)
  twice <- ai_program(result ~ url, "Twice.", function(url) paste(visitor(url), visitor(url)))
  err <- tryCatch(with_ai_config(twice("a.com"), router = script(list(list(name = "visit", input = list(url = "a.com")))), lm = LM,
                                 plugins = list(limits(cost = 0.04, prices = stats::setNames(list(flat), LM)))), error = identity)
  expect_s3_class(err, "functai_plugin_stopped")
  expect_match(conditionMessage(err), "cost 0.04", fixed = TRUE)
})

test_that("a saved folder keeps each setup, and two are compared; an agent is saved with its tools as data", {
  walk <- ai_program(result ~ url, "Walk.", function(url) write_up(url), .name = "walk")
  a <- withr::local_tempdir(); b <- withr::local_tempdir(); c <- withr::local_tempdir()
  with_ai_config(write_ai(walk, a), lm = LM)
  with_ai_config(write_ai(walk, b), lm = "claude-haiku-4-5")
  m <- read_json_file(file.path(a, "functai.json"))
  expect_null(schema_fault(m, "saved.schema.json"))
  node <- m$nodes[["__main__:walk"]]
  expect_identical(node$kind, "module")
  expect_identical(node$version, ai_version(walk))
  expect_false(is.null(m$nodes[["__main__:write_up"]]$setup))
  d <- ai_setup_diff(a, b)
  expect_identical(paste(d$program, d$field), c("__main__:write_up model.model", "__main__:write_up model.provider"))
  with_ai_config(write_ai(browse, c), lm = LM)
  m <- read_json_file(file.path(c, "functai.json"))
  t <- m$nodes[["__main__:browse"]]$ai$tools[[1L]]
  expect_identical(c(t$tool$name, t$effects), c("look", "reads"))
  expect_match(t$code$code, "^sha256:")
  err <- tryCatch(read_ai(c), error = identity)
  expect_identical(err$code, "saved-tools")
})
