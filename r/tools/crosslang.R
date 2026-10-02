# The R half of ../../tools/crosslang.py (run by it, with its working folder):
# load what Python saved, write the same function here, log and rate into the
# same folder, and read the ratings back.
suppressPackageStartupMessages(library(functai))
work <- commandArgs(TRUE)[[1L]]
read_json <- function(p) lmcc::parse_json(paste(readLines(p, warn = FALSE, encoding = "UTF-8"), collapse = "\n"))
python <- read_json(file.path(work, "python.json"))
say <- function(...) cat("  ok    ", ..., "\n", sep = "")
stopifnot_eq <- function(a, b, what) if (!identical(a, b)) stop(sprintf("%s: %s != %s", what, format(a), format(b)), call. = FALSE)

# 1. what Python saved loads here and sends the same bytes
for (name in names(python)) {
  want <- python[[name]]
  if (name == "rounded") {
    err <- tryCatch(read_ai(file.path(work, "saved", name)), functai_load_refused = function(e) e)
    stopifnot_eq(err$code, "saved-code", name)
    say(name, ": refused in R (saved-code: it runs Python code of its own)")
    next
  }
  fn <- read_ai(file.path(work, "saved", name))
  stopifnot_eq(ai_version(fn), want$version, paste(name, "version"))
  stopifnot_eq(ai_signature_id(fn), want$signature, paste(name, "signature"))
  mine <- lmcc::canonical_json(do.call(ai_render, c(list(fn), want$inputs)))
  stopifnot_eq(mine, lmcc::canonical_json(want$request), paste(name, "request"))
  say(name, ": saved in Python, loaded in R, same version and same request")
}

# 2. the same function, written here
router <- list(resolve = function(model) list(provider = "openai", model = model),
               complete = function(request) lm15::response(request$model, lm15::message_assistant("<result>\nhappy\n</result>"), "stop",
                                                            usage = lm15::usage(input_tokens = 10L, output_tokens = 5L, total_tokens = 15L)))
mood <- ai(mood ~ review, "How does the customer feel about what they bought?",
           mood = choice("happy", "unhappy", "mixed"), .defined_in = "shop", .temperature = 0,
           .lm = "gpt-4.1-mini", .router = router, .log_calls = file.path(work, "log"))
stopifnot_eq(ai_version(mood), python$mood$version, "version")
stopifnot_eq(ai_signature_id(mood), python$mood$signature, "signature")
say("mood written in R has Python's version and signature")

# 3. one log: log and rate here, then read everyone's calls and ratings
p <- predict(mood, data.frame(review = "Five stars, would buy again."))
rate(p$.call, "right", by = "cleo", folder = file.path(work, "log"))
log <- functai:::read_log(file.path(work, "log"))
rows <- functai:::rated_rows(log$calls, log$ratings, name = "mood", module = "shop", signature = ai_signature_id(mood),
                             interface = functai:::interface_signature(unclass(ai_interface(mood))))$rows
writeLines(lmcc::json_text(rows), file.path(work, "r-rated.json"))

# 4. stages 1.2 to 5 with Python: continue Python's conversation, answer from
# Python's disk cache, call Python's server
tutor <- ai(result ~ message, "Tutor.", .name = "tutor", .defined_in = "shop", .lm = "gpt-4.1-mini")
echo <- list(resolve = function(model) list(provider = "openai", model = model),
             complete = function(request) {
               n <- length(lmcc::lm15_plain(lm15::as_dict(request))$messages)
               lm15::response(request$model, lm15::message_assistant(sprintf("<result>\nr %d\n</result>", n)), "stop",
                              usage = lm15::usage(input_tokens = 10L, output_tokens = 5L, total_tokens = 15L))
             })
chat <- ai_conversation(tutor, "lesson-r", store = file.path(work, "conversations"))
answer <- with_ai_config(chat("Is it 5/6?"), router = echo, log_calls = file.path(work, "log"))
stopifnot_eq(answer, "r 5", "the third turn saw both of Python's")
rate(ai_turn(chat)$id, "right", by = "ana", folder = file.path(work, "log"))
say("a conversation Python started in a folder continues here: the third turn saw Python's two")

no_calls <- list(resolve = function(model) list(provider = "openai", model = model),
                 complete = function(request) stop("r: the request should have been answered from Python's disk cache"))
kept <- with_ai_config(update(mood, router = no_calls)("Kept for later."), cache_replies = file.path(work, "replies.sqlite"))
stopifnot_eq(as.character(kept), "unhappy", "the cached reply")
say("a reply Python kept in the disk cache answers the same request here, with no model call")

served <- read_json(file.path(work, "served.json"))
far <- ai_remote(served$url)
got <- with_ai_config(far("I was charged twice."), log_calls = file.path(work, "log"))
stopifnot_eq(as.character(got), "mixed", "the served answer")
say("a program Python serves is called here with ai_remote (logged here, kind remote)")
