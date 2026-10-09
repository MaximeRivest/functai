# Calls, offline: a fake model stands in for every provider.

mood_of <- function(router, ...) {
  settings <- list(.lm = "gpt-4.1-mini", .router = router, .log_calls = FALSE)
  more <- list(...)
  settings[names(more)] <- more
  do.call(ai, c(list(mood ~ review, "How does the customer feel about what they bought?",
    mood = choice("happy", "unhappy", "mixed")), settings))
}

guess <- function(request, i) {
  t <- last_text(request)
  sprintf("<result>\n%s\n</result>", if (grepl("Love|Great", t)) "happy" else if (grepl("late", t)) "mixed" else "unhappy")
}

test_that("a call sends the contract's layout and returns the declared type", {
  r <- fake_router(list("<result>\nunhappy\n</result>"))
  mood <- mood_of(r)
  out <- mood("Broke after a day.")
  expect_identical(out, factor("unhappy", levels = c("happy", "unhappy", "mixed")))
  req <- lmcc::lm15_plain(lm15::as_dict(r$env$requests[[1L]]))
  expect_match(req$system, "^Function: mood\n\nHow does the customer feel")
  expect_identical(message_text(r$env$requests[[1L]], 1L), "<review>\nBroke after a day.\n</review>\n")
  expect_null(req$config$stop)                                 # OpenAI's Responses API takes no stop sequences
  expect_s3_class(mood, "functai_fn")
  expect_identical(names(formals(mood)), "review")
})

test_that("a request to a provider that keeps requests says store: false, unless the settings say", {
  r <- fake_router(responder = guess)
  mood_of(r)("Broke.")
  mood_of(r, .config = list(store = TRUE))("Broke.")
  mood_of(r, .lm = "anthropic:claude-3-5-haiku")("Broke.")
  stores <- lapply(r$env$requests, function(q) lmcc::lm15_plain(lm15::as_dict(q))$config$store)
  expect_identical(stores, list(FALSE, TRUE, NULL))
})

test_that("a reply the provider stopped is never an answer, and is not asked again", {
  r <- fake_router(list(list(text = "", finish = "content_filter"), "<result>\nhappy\n</result>"))
  err <- tryCatch(mood_of(r, .retries = 2L, .on_error = "stop")("x"), error = function(e) e)
  expect_identical(err$code, "parse-filtered")
  expect_match(conditionMessage(err), "safety filter, or the model declining")
  expect_length(r$env$requests, 1L)
  r <- fake_router(list(list(text = "<result>\nhappy\n</result>", finish = "content_filter")))
  expect_error(mood_of(r, .on_error = "stop")("x"), "stopped the reply")
})

test_that("a model declining in its own words is lmcc's parse-filtered, and is not asked again", {
  declined <- lm15::response("gpt-4.1-mini", lm15::message_assistant(list(lm15::refusal_part("I can't."))), "stop",
                             usage = lm15::usage(input_tokens = 1L, output_tokens = 1L, total_tokens = 2L))
  r <- fake_router(list(declined, "<result>\nhappy\n</result>"))
  err <- tryCatch(mood_of(r, .retries = 2L, .on_error = "stop")("x"), error = function(e) e)
  expect_identical(err$code, "parse-filtered")
  expect_match(conditionMessage(err), "I can't.", fixed = TRUE)
  expect_length(r$env$requests, 1L)
})

test_that("an interrupted reply is sent again as it was: no correction, no larger budget", {
  r <- fake_router(list(list(text = "<result>\nhap", finish = "error"), "<result>\nhappy\n</result>"))
  expect_identical(as.character(mood_of(r)("x")), "happy")
  expect_length(r$env$requests, 2L)
  d <- lapply(r$env$requests, function(q) lmcc::lm15_plain(lm15::as_dict(q)))
  expect_identical(d[[2]]$messages, d[[1]]$messages)
  expect_identical(d[[2]]$config, d[[1]]$config)
  r <- fake_router(list(list(text = "<result>\nhap", finish = "error"), list(text = "<result>\nha", finish = "error")))
  err <- tryCatch(mood_of(r, .on_error = "stop")("x"), error = function(e) e)
  expect_identical(err$code, "parse-interrupted")
  expect_length(r$env$requests, 2L)
})

test_that("a picture given by its path is sent as its bytes, so a changed file is a new request", {
  dir <- withr::local_tempdir()
  png <- file.path(dir, "now.png")
  a <- openssl::base64_decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")
  b <- openssl::base64_decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")
  writeBin(a, png)
  r <- fake_router(responder = function(req, i) "<result>\nyellow\n</result>")
  colour <- ai(colour ~ picture, "The main colour.", picture = json_shape(list(media = "image")),
               .lm = "gpt-4.1-mini", .router = r, .log_calls = FALSE, .cache_replies = file.path(dir, "replies.sqlite"))
  picture <- list(list(media_type = "image/png", path = png))
  colour(picture); colour(picture)
  expect_length(r$env$requests, 1L)                                  # the same picture: from the cache
  parts <- lmcc::lm15_plain(lm15::as_dict(r$env$requests[[1L]]))$messages[[1L]]$parts
  image <- Filter(function(p) identical(p$type, "image"), parts)[[1L]]
  expect_identical(image$data, openssl::base64_encode(a))
  expect_null(image$path)
  writeBin(b, png)
  colour(picture)
  expect_length(r$env$requests, 2L)                                  # another picture, same name: asked
  err <- tryCatch(ai_render(colour, list(list(media_type = "image/png", path = file.path(dir, "gone.png")))),
                  error = function(e) e)
  expect_identical(err$code, "interface-input")
  expect_match(conditionMessage(err), "the file .*gone[.]png cannot be read")
  expect_length(r$env$requests, 2L)
})

test_that("a provider asking for a longer wait than api_max_wait is not waited: raised at once, saying until when", {
  spent <- lm15::lm15_error("window spent", code = "rate_limit", retry_after = 11270)
  r <- fake_router(list(spent, "<result>\nhappy\n</result>"))
  started <- Sys.time()
  expect_error(mood_of(r)("x"), "wait 3 h 7 min \\(until [0-9-]+ [0-9:]+ UTC\\).*api_max_wait=60 s.*api_max_wait = 11270")
  expect_length(r$env$requests, 1L)                           # no second request
  expect_lt(as.numeric(Sys.time() - started, units = "secs"), 5)
  expect_error(ai_config(api_max_wait = -1), "api_max_wait")
  expect_identical(vapply(c(45, 125, 720, 3600, 11270), functai:::wait_duration, ""),
                   c("45 s", "2 min 5 s", "12 min", "1 h", "3 h 7 min"))
})

test_that("functions are vectorised: one call per row, recycled, NA in gives NA out", {
  r <- fake_router(responder = guess)
  mood <- mood_of(r)
  out <- mood(c("Love it", "Broke", NA, "Arrived late"))
  expect_identical(as.character(out), c("happy", "unhappy", NA, "mixed"))
  expect_length(r$env$requests, 3L)                           # no call for the missing review
  expect_identical(mood(character(0)), factor(character(0), levels = c("happy", "unhappy", "mixed")))
})

test_that("they are columns in dplyr: mutate, and several outputs splice in as columns", {
  skip_if_not_installed("dplyr")
  r <- fake_router(responder = guess)
  mood <- mood_of(r)
  reviews <- tibble::tibble(id = 1:3, review = c("Love it", "Broke", "Arrived late"))
  got <- dplyr::mutate(reviews, mood = mood(review))
  expect_identical(as.character(got$mood), c("happy", "unhappy", "mixed"))
  expect_s3_class(got$mood, "factor")

  triage <- ai(summary + minutes ~ ticket, "Read the ticket.", minutes = described(integer(), "minutes to fix"),
    .name = "triage",
    .lm = "gpt-4.1-mini", .log_calls = FALSE,
    .router = fake_router(responder = function(req, i) "<summary>\nCharged twice\n</summary>\n<minutes>\n30\n</minutes>"))
  spliced <- dplyr::mutate(tibble::tibble(ticket = c("a", "b")), triage(ticket))
  expect_identical(names(spliced), c("ticket", "summary", "minutes"))
  expect_identical(spliced$minutes, c(30L, 30L))
  expect_match(ai_instructions(triage), "Output guidance:\n- minutes: minutes to fix$")
})

test_that("records come back as tibble columns; lists as list_of", {
  person <- ai(person ~ text, "Who is described?",
    person = tibble::tibble(name = character(), age = integer()), .adapter = "json",
    .lm = "gpt-4.1-mini", .log_calls = FALSE,
    .router = fake_router(responder = function(req, i) if (i == 1L) '{"result": {"name": "Ana", "age": 31}}' else '{"result": {"name": "Bo", "age": 50}}'))
  got <- person(c("Ana, 31.", "Bo, 50."))
  expect_s3_class(got, "tbl_df")
  expect_identical(got$name, c("Ana", "Bo"))
  expect_identical(got$age, c(31L, 50L))
  tags <- ai(tags ~ text, "Tags.", tags = vctrs::list_of(.ptype = character()),
    .lm = "gpt-4.1-mini", .log_calls = FALSE, .router = fake_router(list('<result>\n["a", "b"]\n</result>')))
  out <- tags("x")
  expect_s3_class(out, "vctrs_list_of")
  expect_identical(out[[1L]], c("a", "b"))
})

test_that("an unreadable reply is asked again once, in the contract's words", {
  r <- fake_router(list("They seem sad.", "<result>\nunhappy\n</result>"))
  expect_identical(as.character(mood_of(r)("Broke.")), "unhappy")
  expect_length(r$env$requests, 2L)
  expect_match(last_text(r$env$requests[[2L]]), "^Your reply could not be read: .*\\. Reply again, in exactly the form the instructions give\\.$")
})

# functions.md, "When the reply cannot be read": a cut reply, all thinking and no answer.
cut_off_reply <- function(request, adaptations = list()) {
  r <- lm15::response(request$model, lm15::message_assistant(list(lm15::thinking_part("still thinking"))), "length",
                      usage = lm15::usage(input_tokens = 3L, output_tokens = 900L, total_tokens = 903L, reasoning_tokens = 900L))
  if (length(adaptations)) r["adaptations"] <- list(adaptations)
  r
}

test_that("a cut-off reply: twice a set budget, never a guessed one; the refusal says what happened", {
  r <- fake_router(responder = function(req, i) cut_off_reply(req))
  err <- tryCatch(mood_of(r, .max_tokens = 500L, .retries = 2L)("x"), lmcc_refusal = identity)
  expect_identical(vapply(r$env$requests, function(q) as.integer(lmcc::lm15_plain(lm15::as_dict(q))$config$max_tokens), 1L), c(500L, 1000L, 2000L))
  expect_true(endsWith(err$hint, "; the model spent 900 of its 900 output tokens thinking; raise max_tokens (it was 2000) or ask for less"))
  # no budget: the reply had the most the call allows; the old rule re-sent it with 2048 after 128000
  notes <- list(
    lm15::adaptation("config.max_tokens", "defaulted",
                     "the Messages API requires max_tokens and none was set; the model's output ceiling was used", applied = 128000L),
    lm15::adaptation("config.reasoning.thinking_budget", "dropped", "budget_tokens is rejected by the API", asked = 32000L))
  r <- fake_router(responder = function(req, i) cut_off_reply(req, notes))
  err <- tryCatch(mood_of(r, .retries = 2L)("x"), lmcc_refusal = identity)
  expect_length(r$env$requests, 1L)
  expect_true(endsWith(err$hint, paste0("; the model spent 900 of its 900 output tokens thinking; no max_tokens was set, and lm15 sent 128000, ",
    "the most it knows this model to allow: lower the reasoning effort or ask for less ",
    "(lm15 adapted the request: config.reasoning.thinking_budget dropped: budget_tokens is rejected by the API)")))
  r <- fake_router(responder = function(req, i) cut_off_reply(req))
  err <- tryCatch(mood_of(r, .retries = 2L)("x"), lmcc_refusal = identity)
  expect_length(r$env$requests, 1L)
  expect_true(endsWith(err$hint, "; no max_tokens was set, so the provider used its own maximum: lower the reasoning effort or ask for less"))
})

test_that("a value outside its type is unreadable too; one failing row is an error, many are NA and a warning", {
  r <- fake_router(list("<result>\nfurious\n</result>", "<result>\nunhappy\n</result>"))
  expect_identical(as.character(mood_of(r)("x")), "unhappy")
  expect_error(mood_of(fake_router(list("nope", "still nope")))("x"), class = "lmcc_refusal")
  r2 <- fake_router(responder = function(req, i) if (grepl("bad", message_text(req, 1L))) "nope" else "<result>\nhappy\n</result>")
  expect_warning(out <- mood_of(r2)(c("good", "bad", "good")), "1 of 3 calls of")
  expect_identical(as.character(out), c("happy", NA, "happy"))
  expect_identical(ai_problems()$row, 2L)
  expect_error(mood_of(r2, .on_error = "stop")(c("good", "bad")), class = "lmcc_refusal")
})

test_that("tools run until the model answers", {
  orders <- c("A-1" = "stuck at the carrier")
  lookup_order <- function(order) orders[[order]]
  lookup <- ai_tool(lookup_order, "Look up an order.")
  expect_identical(lookup$name, "lookup_order")
  r <- fake_router(list(list(calls = list(list(id = "c1", name = "lookup_order", input = list(order = "A-1")))),
                        "<result>\nIt is stuck at the carrier.\n</result>"))
  helper <- ai(helper ~ question, "Help.", .tools = list(lookup), .lm = "gpt-4.1-mini", .router = r, .log_calls = FALSE)
  expect_identical(helper("Where is A-1?"), "It is stuck at the carrier.")
  second <- lmcc::lm15_plain(lm15::as_dict(r$env$requests[[2L]]))
  expect_match(lmcc::json_text(second$messages), "stuck at the carrier")
  expect_identical(second$tools[[1L]]$name, "lookup_order")
})

test_that("the version follows what is sent, not where it runs", {
  mood <- mood_of(NULL)
  v <- ai_version(mood)
  expect_match(v, "^sha256:[0-9a-f]{64}$")
  expect_identical(ai_version(update(mood, lm = "claude-haiku-4-5", temperature = 0.3)), v)
  expect_false(identical(ai_version(update(mood, adapter = "json")), v))
  expect_false(identical(ai_version(update(mood, module = "cot")), v))
  taught <- with_demos(mood, tibble::tibble(review = "Great", mood = "happy"))
  expect_false(identical(ai_version(taught), v))
  expect_identical(ai_version(with_demos(taught, NULL)), v)
  expect_false(identical(ai_version(with_instructions(mood, "Say how they feel.")), v))
})

# ---------------------------------------------------------------- predict, augment, evaluate

reviews <- tibble::tibble(review = c("Broke in a day", "Love it", "Good but late", "Terrible"),
                          mood = c("unhappy", "happy", "mixed", "unhappy"))

test_that("predict and augment: tidymodels' columns, with call ids", {
  mood <- mood_of(fake_router(responder = guess))
  p <- predict(mood, reviews)
  expect_identical(names(p), c(".pred_class", ".call", ".error"))     # a choice is a class, as in tidymodels
  expect_identical(as.character(p$.pred_class), c("unhappy", "happy", "mixed", "unhappy"))
  a <- augment(mood, reviews)
  expect_identical(names(a), c("review", "mood", ".pred_class", ".call", ".error"))
  n <- ai(n ~ text, "Count.", n = integer(), .lm = "gpt-4.1-mini", .log_calls = FALSE,
          .router = fake_router(list("<result>\n3\n</result>")))
  expect_identical(names(predict(n, tibble::tibble(text = "x"))), c(".pred", ".call", ".error"))
  expect_error(predict(mood, tibble::tibble(x = 1)), "no column for input")
})

test_that("evaluate: the score, its range, broom's tidy, glance and augment", {
  mood <- mood_of(fake_router(responder = function(req, i) sprintf("<result>\n%s\n</result>", if (grepl("Love", last_text(req))) "happy" else "unhappy")))
  ev <- evaluate(mood, reviews)
  expect_identical(ev$score, 0.75)
  t <- tidy(ev)
  expect_identical(names(t), c("metric", "estimate", "conf.low", "conf.high", "n", "failed"))
  expect_true(t$conf.low < 0.75 && t$conf.high > 0.75)
  expect_identical(nrow(glance(ev)), 1L)
  expect_identical(augment(ev)$exact_match, c(1, 1, 0, 1))
  expect_output(print(ev), "exact_match: 0.75")
  truth <- dplyr::rename(reviews, feeling = mood)
  expect_identical(evaluate(mood, truth, expected = feeling)$score, 0.75)
  failing <- mood_of(fake_router(responder = function(req, i) "?"), .retries = 0L)
  bad <- suppressWarnings(evaluate(failing, reviews[1:2, ]))
  expect_identical(bad$score, 0)
  expect_identical(tidy(bad)$failed, 2L)
})

test_that("improving adds worked examples: a new version, the examples in the request", {
  r <- fake_router(responder = guess)
  mood <- mood_of(r)
  taught <- labeled_few_shot(mood, reviews, k = 2L)
  expect_length(ai_demos(taught), 2L)
  expect_length(ai_demos(mood), 0L)
  expect_length(ai_render(taught, "new")$messages, 5L)
  boot <- bootstrap_few_shot(mood, reviews, max_bootstrapped = 2L, max_labeled = 3L)
  demos <- ai_demos(boot)
  expect_identical(sum(vapply(demos, function(d) !is.null(d$steps), NA)), 2L)
  expect_length(demos, 3L)
  expect_length(ai_render(boot, "new")$messages, 7L)
})

# ---------------------------------------------------------------- the call log

test_that("every call is a line in the log; ratings make rows with known answers", {
  folder <- withr::local_tempdir()
  mood <- update(mood_of(fake_router(responder = guess)), log_calls = folder)
  p <- with_ai_config(predict(mood, reviews[1:2, ]), caller = list(kind = "test", user = "ana"))
  log <- calls(mood, folder = folder)
  expect_identical(nrow(log), 2L)
  expect_setequal(log$id, p$.call)
  expect_identical(unique(log$version), ai_version(mood))
  expect_identical(unique(log$language), "r")
  line <- log_lines(folder)[[1L]]
  expect_identical(line$program$signature, ai_signature_id(mood))
  expect_identical(line$program$module, "__main__")
  expect_identical(line$caller$user, "ana")
  expect_identical(line$sizes$outputs$result, 9L)
  rate(p$.call, c("right", "wrong"), answer = list(NULL, "mixed"), by = "ben", folder = folder)
  rows <- rated(mood, folder = folder)
  expect_identical(nrow(rows), 2L)
  expect_s3_class(rows$mood, "factor")                        # the answer's column is the formula's name
  expect_setequal(as.character(rows$mood), c("unhappy", "mixed"))
  expect_identical(attr(rows, "left_out"), list(other_signature = 0L, no_content = 0L, no_answer = 0L, no_context = 0L))
})

test_that("log_content = FALSE keeps sizes and tokens, never values; a failed call records its error", {
  folder <- withr::local_tempdir()
  mood <- mood_of(fake_router(list("nope", "still nope")), .log_calls = folder, .log_content = FALSE)
  expect_error(mood("secret"))
  line <- log_lines(folder)[[1L]]
  expect_false(line$content)
  expect_null(line$inputs)
  expect_identical(line$sizes$inputs$review, 8L)
  expect_identical(line$error$type, "Refusal")
  expect_null(line$error$message)
  expect_length(line$exchanges, 2L)
})

# ---------------------------------------------------------------- saved

test_that("a function written here loads back with the same version and requests", {
  dir <- withr::local_tempdir()
  mood <- with_demos(mood_of(NULL, .temperature = 0, .defined_in = "shop"), tibble::tibble(review = "Broke", mood = "unhappy"))
  write_ai(mood, dir)
  m <- read_json_file(file.path(dir, "functai.json"))
  expect_identical(m$language, "r")
  expect_identical(m$entry, "shop:mood")
  again <- read_ai(dir)
  expect_identical(ai_version(again), ai_version(mood))
  expect_identical(ai_signature_id(again), ai_signature_id(mood))
  expect_identical(lmcc::canonical_json(ai_render(update(again, router = fake_router()), "x")),
                   lmcc::canonical_json(ai_render(update(mood, router = fake_router()), "x")))
})

test_that("record() takes described and optional fields; a null comes back NA", {
  ref <- ai(order_ref ~ message, "The order.",
    order_ref = record(order = optional(character()), days = described(integer(), "whole days")),
    .lm = "gpt-4.1-mini", .log_calls = FALSE,
    .router = fake_router(list("<result>\n{\"order\": null, \"days\": 3}\n</result>")))
  expect_match(ai_instructions(ref), "Function: order_ref")
  shape <- ai_signature(ref)$fields[[2L]]$shape
  expect_identical(lmcc::canonical_json(shape$properties$order), '{"anyOf":[{"type":"string"},{"type":"null"}]}')
  expect_identical(shape$properties$days$description, "whole days")
  out <- ref("x")
  expect_identical(out$order, NA_character_)
  expect_identical(out$days, 3L)
})

# ---------------------------------------------------------------- stage 1: interfaces, defaults, what the log keeps

reply_of <- function(router = fake_router(responder = function(req, i) "<result>\nHello!\n</result>"), ...) {
  ai(reply ~ message + tone, "Answer the customer.",
     message = "the customer's own words",
     tone = defaults_to("kind", choice("kind", "brief", "formal")),
     .lm = "gpt-4.1-mini", .router = router, ...)
}

test_that("an input with a default: the R function's default argument, sent when left out", {
  r <- fake_router(responder = function(req, i) "<result>\nHello!\n</result>")
  folder <- withr::local_tempdir()
  reply <- reply_of(r, .log_calls = folder)
  expect_identical(formals(reply)$tone, "kind")
  expect_identical(reply("Hi"), "Hello!")
  expect_match(message_text(r$env$requests[[1L]], 1L), "<tone>\nkind\n</tone>")
  reply(c("Hi", "Yo"), tone = "brief")
  expect_match(message_text(r$env$requests[[3L]], 1L), "<tone>\nbrief\n</tone>")
  lines <- log_lines(folder)
  expect_identical(lines[[1L]]$inputs$tone, "kind")                  # the record holds the value sent
  expect_identical(lines[[1L]]$functai_call, 2L)
  iface <- ai_interface(reply)
  expect_s3_class(iface, "functai_interface")
  expect_identical(lmcc::canonical_json(unclass(iface)$inputs[[2L]]),
    '{"name":"tone","optional":true,"shape":{"default":"kind","enum":["kind","brief","formal"],"type":"string"},"type":"factor"}')
  expect_output(print(iface), "optional, default \"kind\"")
  expect_output(print(reply), "tone     one of kind, brief, formal = \"kind\"")
  # a default is behaviour, not data: the signature does not see it; the version does, by its logic (calls.md, Versions)
  other <- ai(reply ~ message + tone, "Answer the customer.", message = "the customer's own words",
              tone = defaults_to("brief", choice("kind", "brief", "formal")), .lm = "gpt-4.1-mini")
  expect_identical(ai_signature_id(other), ai_signature_id(reply))
  expect_false(identical(ai_version(other), ai_version(reply)))
  # a default written as a formula is computed at each call; the version counts its code, not today's value
  day <- "2026-09-28"
  a <- ai(reply ~ message + day, "x", day = defaults_to(~ day), .lm = "gpt-4.1-mini")
  day <- "2026-09-29"
  b <- ai(reply ~ message + day, "x", day = defaults_to(~ day), .lm = "gpt-4.1-mini")
  expect_identical(ai_version(a), ai_version(b))
  expect_identical(lines[[1L]]$program$interface, lines[[1L]]$program$signature)
  # predict and render take it too when the column is not there
  p <- predict(reply, tibble::tibble(message = "Hi"))
  expect_identical(p$.pred, "Hello!")
  expect_match(ai_render(update(reply, router = fake_router()), "Hi")$messages[[1L]]$parts[[1L]]$text, "kind")
})

test_that("interfaces are checked when a function is defined", {
  err <- tryCatch(ai(reply ~ message + tone, "x", tone = defaults_to("loud", choice("kind", "brief"))), functai_refusal = identity)
  expect_s3_class(err, "functai_interface_malformed")
  expect_identical(err$field, "tone")
  expect_match(conditionMessage(err), "tone's default does not fit its type (tone: \"loud\" is not one of", fixed = TRUE)
  err <- tryCatch(ai(team ~ my.message, "x"), lmcc_refusal = identity)      # lmcc checks a signature first
  expect_identical(err$code, "signature-malformed")
  err <- tryCatch(ai(n ~ items + at_least, "Count.", at_least = defaults_to(5L, json_shape(list(type = "integer", minimum = 10)))),
                  functai_refusal = identity)
  expect_identical(err$field, "at_least")
  expect_error(ai(team ~ message, "x", team = defaults_to("a")), class = "functai_interface_malformed")
  expect_error(record(a = defaults_to(1L)), "belongs to an input")
  expect_error(defaults_to(NULL), "needs its type")
  # an AI function's shape may carry lmcc's keywords; FunctAI never reads them
  tagged <- ai(tag ~ code, "Tag.", code = defaults_to("ABC", json_shape(list(type = "string", pattern = "^[a-z]+$"))))
  expect_identical(formals(tagged)$code, "ABC")
  expect_identical(bound_row(tagged, list()), list(code = "ABC"))
  # optional() keeps an input's own default at the top of its shape
  n <- defaults_to(NA, optional(integer()))
  expect_identical(lmcc::canonical_json(n$shape), '{"anyOf":[{"type":"integer"},{"type":"null"}],"default":null}')
  f <- ai(x ~ message + limit, "x", limit = n)
  expect_identical(formals(f)$limit, NA_integer_)
})

test_that("log_content keeps what no layer drops, per field; the record says what it left out", {
  folder <- withr::local_tempdir()
  guessy <- fake_router(responder = function(req, i) "<summary>\nCharged twice\n</summary>\n<result>\nbilling\n</result>")
  triage <- ai(summary + result ~ transcript + question, "Sort the ticket.", .name = "triage", .lm = "gpt-4.1-mini",
               .router = guessy, .log_calls = folder, .log_content = c(transcript = FALSE))
  triage("Ana: I was charged twice.", "Which team?")
  rec <- log_lines(folder)[[1L]]
  expect_false(rec$content)
  expect_identical(lmcc::canonical_json(rec$omitted), '{"inputs":["transcript"],"outputs":[]}')
  expect_identical(names(rec$inputs), "question")
  expect_identical(rec$outputs$result, "billing")
  expect_identical(rec$sizes$inputs$transcript, 27L)                 # the size is always kept
  expect_null(rec$exchanges[[1L]]$request)
  expect_null(rec$exchanges[[1L]]$request_hash)
  expect_null(schema_fault(rec, "call.schema.json"))
  # a block can only remove; true in the function's own setting brings nothing back
  whole <- update(triage, log_content = TRUE)
  with_ai_config(whole("a", "b"), log_content = c("question"))        # the names of the fields to keep
  rec <- log_lines(folder)[[2L]]
  expect_identical(lmcc::canonical_json(rec$omitted), '{"inputs":["transcript"],"outputs":["summary","result"]}')
  local({ local_ai_config(log_content = list(summary = FALSE)); with_ai_config(whole("a", "b"), log_content = c(result = FALSE)) })
  rec <- log_lines(folder)[[3L]]
  expect_identical(lmcc::canonical_json(rec$omitted), '{"inputs":[],"outputs":["summary","result"]}')
  whole("a", "b")                                                     # outside the blocks: whole again
  rec <- log_lines(folder)[[4L]]
  expect_true(rec$content)
  expect_null(rec$omitted)
  expect_match(rec$exchanges[[1L]]$request_hash, "^sha256:[0-9a-f]{64}$")
  expect_null(schema_fault(rec, "call.schema.json"))
  withr::with_envvar(c(FUNCTAI_LOG_CONTENT = " OFF "), whole("a", "b"))
  rec <- log_lines(folder)[[5L]]
  expect_false(rec$content)
  expect_null(rec$inputs); expect_null(rec$outputs)
  # a name that is not a field of the function refuses when it is set as its own
  err <- tryCatch(update(triage, log_content = c(transcrpit = FALSE)), functai_refusal = identity)
  expect_s3_class(err, "functai_log_content_field")
  expect_identical(err$field, "transcrpit")
  expect_error(ai(team ~ message, "x", .log_content = list(tools = FALSE)), class = "functai_log_content_field")
  # a host's map names fields of every function it runs: one that lacks it is unchanged
  with_ai_config(whole("a", "b"), log_content = c(notes = FALSE))
  expect_true(log_lines(folder)[[6L]]$content)
  # a key that is not a name refuses wherever it is set
  expect_error(with_ai_config(NULL, log_content = list(`#private` = FALSE)), class = "functai_log_content_field")
  expect_error(ai_config(log_content = c(`two words` = FALSE)), class = "functai_log_content_field")
})

test_that("a one-output function's answer is named in log_content by its formula's name or as result", {
  folder <- withr::local_tempdir()
  mood <- update(mood_of(fake_router(responder = guess)), log_calls = folder, log_content = c(mood = FALSE))
  expect_identical(core_of(mood)$own$log_content, list(result = FALSE))     # kept, and saved, by its field's name
  mood("Love it")
  expect_identical(lmcc::canonical_json(log_lines(folder)[[1L]]$omitted), '{"inputs":[],"outputs":["result"]}')
  whole <- update(mood, log_content = TRUE)
  with_ai_config(whole("Love it"), log_content = c("review"))           # keep only the review
  expect_identical(lmcc::canonical_json(log_lines(folder)[[2L]]$omitted), '{"inputs":[],"outputs":["result"]}')
  with_ai_config(whole("Love it"), log_content = c(mood = FALSE, result = TRUE))   # a drop wins
  expect_identical(lmcc::canonical_json(log_lines(folder)[[3L]]$omitted), '{"inputs":[],"outputs":["result"]}')
  expect_error(update(mood, log_content = c(moood = FALSE)), "its fields are review and mood")
})

test_that("the reasoning FunctAI adds goes whenever any field goes", {
  folder <- withr::local_tempdir()
  r <- fake_router(responder = function(req, i) "<reasoning>\nThey were charged twice.\n</reasoning>\n<result>\nbilling\n</result>")
  team <- ai(team ~ message + channel, "Which team?", .lm = "gpt-4.1-mini", .router = r, .log_calls = folder, .module = "cot")
  team("I was charged twice", "email")
  expect_identical(names(log_lines(folder)[[1L]]$outputs), c("reasoning", "result"))
  with_ai_config(team("I was charged twice", "email"), log_content = c(channel = FALSE))
  rec <- log_lines(folder)[[2L]]
  expect_identical(lmcc::canonical_json(rec$omitted), '{"inputs":["channel"],"outputs":["reasoning"]}')
  expect_identical(names(rec$outputs), "result")
})

test_that("records, ratings and saved folders R writes pass the contract's schemas", {
  folder <- withr::local_tempdir()
  mood <- update(mood_of(fake_router(responder = guess)), log_calls = folder)
  p <- predict(mood, reviews)
  expect_error(update(mood_of(fake_router(list("nope", "nope"))), log_calls = folder)("x"))
  rate(p$.call[1:2], c("right", "wrong"), answer = list(NULL, "mixed"), by = "ana", folder = folder)
  for (rec in log_lines(folder)) {
    file <- if (!is.null(rec$functai_rating)) "rating.schema.json" else "call.schema.json"
    expect_null(schema_fault(rec, file))
  }
  dir <- withr::local_tempdir()
  write_ai(reply_of(), dir)
  m <- read_json_file(file.path(dir, "functai.json"))
  expect_null(schema_fault(m, "saved.schema.json"))
  expect_identical(lmcc::canonical_json(m$nodes[["__main__:reply"]]$interface), lmcc::canonical_json(unclass(ai_interface(reply_of()))))
  expect_identical(lmcc::canonical_json(unclass(ai_interface(dir))), lmcc::canonical_json(unclass(ai_interface(reply_of()))))
  again <- read_ai(dir)
  expect_identical(formals(again)$tone, "kind")                      # optional inputs come back from the interface
  expect_identical(ai_version(again), ai_version(reply_of()))
  broken <- m
  broken$nodes[["__main__:reply"]]$interface$outputs[[1L]]$shape <- list(type = "integer")
  writeLines(lmcc::json_text(broken), file.path(dir, "functai.json"))
  expect_error(read_ai(dir), class = "functai_saved_differs")
  expect_error(ai_interface(dir), class = "functai_saved_differs")
  expect_error(ai_interface(42), "describes an AI function or a saved folder")
})

test_that("rated() pools calls that record the same data: reasoning turned on changes the signature, not the interface", {
  folder <- withr::local_tempdir()
  mood <- update(mood_of(fake_router(responder = guess)), log_calls = folder)
  thinking <- update(mood_of(fake_router(responder = function(req, i) "<reasoning>\nIt broke.\n</reasoning>\n<result>\nunhappy\n</result>")),
                     log_calls = folder, module = "cot")
  expect_false(identical(ai_signature_id(thinking), ai_signature_id(mood)))
  a <- predict(mood, reviews[1, ]); b <- predict(thinking, reviews[1, ])
  rate(c(a$.call, b$.call), "right", by = "ana", folder = folder)
  rows <- rated(mood, folder = folder)
  expect_identical(nrow(rows), 2L)
  expect_identical(attr(rows, "left_out")$other_signature, 0L)
  # a call whose inputs were not kept makes no row
  hidden <- update(mood, log_content = FALSE)
  c3 <- predict(hidden, reviews[2, ])$.call
  rate(c3, "right", by = "ana", folder = folder)
  expect_identical(attr(rated(mood, folder = folder), "left_out")$no_content, 1L)
  expect_identical(calls(mood, folder = folder)$content, c(TRUE, TRUE, FALSE))
  expect_identical(calls(mood, folder = folder)$saw[[1L]], list())
})

test_that("GEPA's own calls keep nothing when a layer in force drops a field of the function it improves", {
  improved <- function(...) core_of(ai(summary ~ transcript + question, "x", ...))
  meta <- meta_fn(new_instruction ~ cases, "x", "_reflect", NULL, improved(.log_content = list(transcript = FALSE)), "run")
  expect_false(core_of(meta)$own$log_content)
  meta <- meta_fn(new_instruction ~ cases, "x", "_reflect", NULL, improved(), "run")
  expect_null(core_of(meta)$own$log_content)
  # a host's map names the function's fields, which the meta function has not: it still holds
  meta <- with_ai_config(meta_fn(new_instruction ~ cases, "x", "_reflect", NULL, improved(), "run"), log_content = c(transcript = FALSE))
  expect_false(core_of(meta)$own$log_content)
  old <- the$config
  ai_config(log_content = c("question"))
  meta <- meta_fn(new_instruction ~ cases, "x", "_reflect", NULL, improved(), "run")
  the$config <- old
  expect_false(core_of(meta)$own$log_content)
  withr::local_envvar(FUNCTAI_LOG_CONTENT = "0")
  expect_false(core_of(meta_fn(new_instruction ~ cases, "x", "_reflect", NULL, improved(), "run"))$own$log_content)
})

test_that("on_unreadable = 'record' keeps an unreadable reply: NA, .error says why, recorded, and a conversation goes on", {
  logs <- withr::local_tempdir()
  r <- fake_router(list("no tags at all", list(text = "", finish = "content_filter")))
  keep <- mood_of(r, .retries = 0L, .on_unreadable = "record", .log_calls = logs)
  p <- predict(keep, data.frame(review = c("Where is it?", "x")))
  expect_true(all(is.na(p$.pred_class)))
  expect_match(p$.error[[1L]], "parse-missing-fields|could not", perl = TRUE)
  expect_match(p$.error[[2L]], "stopped the reply")
  recs <- log_lines(logs)
  expect_length(recs, 2L)
  for (rec in recs) {
    expect_identical(length(rec$outputs), 0L)
    expect_identical(rec$error$type, "Refusal")
  }
  expect_setequal(vapply(recs, function(x) x$error$code, ""), c("parse-missing-fields", "parse-filtered"))
  # no warning: the calls ended well, with no answer
  r2 <- fake_router(list("nope"))
  expect_no_warning(v <- mood_of(r2, .retries = 0L, .on_unreadable = "record")("x"))
  expect_true(is.na(v))
  # a conversation's turn is done with no outputs, and the next goes on
  tutor <- ai(reply ~ message, "Tutor.", .name = "tutor")
  r3 <- fake_router(list("nope", "<result>\nfine\n</result>"))
  with_ai_config(lm = "gpt-4.1-mini", router = r3, retries = 0L, on_unreadable = "record", log_calls = FALSE, {
    chat <- ai_conversation(tutor, "kept")
    expect_true(is.na(chat("first")))
    expect_identical(chat("second"), "fine")
    expect_identical(ai_turns(chat)$state, c("done", "done"))
  })
  # the default still fails the call; a setting that is neither is refused
  expect_error(mood_of(fake_router(list("nope")), .retries = 0L)("x"), class = "lmcc_refusal")
  expect_error(mood_of(r, .on_unreadable = "keep"), "raise.*record")
})

test_that("api_key and base_url: the function's own key and address; the address is saved, the key never", {
  skip_if_not_installed("httpuv")
  # a server in a process of its own (this one waits on the request, so a server here could not answer it),
  # noting the path and the key of each request that reaches it
  port <- httpuv::randomPort()
  seen_file <- withr::local_tempfile(); pid_file <- withr::local_tempfile(); script <- withr::local_tempfile(fileext = ".R")
  writeLines(c(
    sprintf("writeLines(as.character(Sys.getpid()), %s)", deparse(pid_file)),
    sprintf("invisible(httpuv::startServer('127.0.0.1', %dL, list(call = function(req) {", port),
    sprintf("  cat(req$PATH_INFO, if (is.null(req$HTTP_AUTHORIZATION)) '' else req$HTTP_AUTHORIZATION, '\\n', file = %s, append = TRUE)", deparse(seen_file)),
    "  list(status = 400L, headers = list('Content-Type' = 'application/json'), body = '{\"error\":{\"message\":\"seen\"}}')",
    "})))",
    "repeat httpuv::service(100)"), script)
  system2(file.path(R.home("bin"), "Rscript"), script, wait = FALSE, stdout = FALSE, stderr = FALSE)
  listening <- function() file.exists(pid_file) &&
    !inherits(try(suppressWarnings(close(socketConnection("127.0.0.1", port, timeout = 1))), silent = TRUE), "try-error")
  for (i in 1:100) if (listening()) break else Sys.sleep(0.1)
  on.exit(if (file.exists(pid_file)) tools::pskill(as.integer(readLines(pid_file))), add = TRUE)
  url <- sprintf("http://127.0.0.1:%d/v1", port)
  f <- ai(mood ~ review, "How?", mood = choice("happy", "unhappy"), .lm = "openai:qwen3", .api_key = "sk-local",
          .base_url = url, .api_retries = 0L, .log_calls = FALSE, .on_error = "stop")
  expect_error(f("x"))
  seen <- strsplit(trimws(readLines(seen_file)), " ")
  expect_length(seen, 1L)
  expect_match(seen[[1L]][[1L]], "^/v1/")                         # the request went to the function's own address
  expect_identical(seen[[1L]][-1L], c("Bearer", "sk-local"))      # with its own key
  folder <- withr::local_tempdir()
  write_ai(f, folder)
  saved <- paste(readLines(file.path(folder, "functai.json"), encoding = "UTF-8"), collapse = "\n")
  expect_match(saved, url, fixed = TRUE)
  expect_false(grepl("sk-local", saved, fixed = TRUE))
  expect_identical(core_of(read_ai(folder))$own$base_url, url)
  expect_error(ai(mood ~ review, "How?", .base_url = 3), "one string")
})

test_that("a record or a picture given alone, as a named list, is refused saying how to give it", {
  look <- ai(colour ~ picture, "The main colour.", picture = json_shape(list(media = "image")), .lm = "gpt-4.1-mini")
  expect_error(ai_render(look, list(media_type = "image/png", data = "iVBORw0KGgo=")),
               "list(list(media_type = ..., data = ...))", fixed = TRUE)
  req <- ai_render(look, list(list(media_type = "image/png", data = "iVBORw0KGgo=")))
  expect_identical(vapply(req$messages[[1L]]$parts, function(p) p$type, ""), c("text", "image", "text"))
})
