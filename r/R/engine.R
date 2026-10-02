# One call per row: lay it out (lmcc), send it (lm15), read the reply (lmcc),
# run tools until the model answers (contract/functions.md, "When the reply
# cannot be read"). Each row is a job that says what to send next; the
# scheduler keeps up to `concurrency` requests in flight over curl, so a
# column of 200 rows takes about as long as 200 / concurrency calls.
#
# Each row's call shows what it does as events (contract/streaming.md): a
# `request` before each send (a cached or resumed reply too), the reply's text
# (piece by piece when a live reader watches a call that is alone in flight,
# else one piece per field), `retry`, `tool_call` and `tool_result`.

REASK <- "Your reply could not be read: %s. Reply again, in exactly the form the instructions give."

new_job <- function(plan, past, inputs, settings, model, tools, call, core = NULL) {
  job <- new.env(parent = emptyenv())
  job$plan <- plan; job$past <- past; job$settings <- settings; job$model <- model; job$tools <- tools; job$call <- call
  job$core <- core
  job$inputs <- inputs
  job$state <- "send"; job$not_before <- 0; job$retries <- 0L; job$api_retries <- 0L; job$steps <- 1L
  job$responses <- list(); job$overrides <- list(); job$tool_calls <- list()
  values <- prepare_inputs(plan$signature, inputs)
  if (length(tools)) values$tools <- lapply(tools, function(t) list(name = t$name, description = t$description, parameters = t$parameters))
  job$turn <- lmcc::new_turn(plan, values)
  render_next(job)
  job
}

render_next <- function(job) {
  job$rendered <- lmcc::render(job$plan, job$turn, if (length(job$past)) job$past else NULL)
  job$asked <- lmcc::request_of(job$rendered)          # the lmcc request this exchange sends (calls.md, exchanges)
  job$request_hash <- lmcc::sha256_of(job$asked)
  job$request <- lmcc::lm15_request(job$rendered, job$model, config_of(job$settings, job$overrides))
  job$retries <- 0L
}

fail <- function(job, err) { job$state <- "failed"; job$error <- err; ended(job); finished(job) }

# A job is done, failed or waiting: whoever watches the batch hears of it (a progress line).
finished <- function(job) if (is.function(job$on_finish)) job$on_finish(job)

# A call's own time: from its first request to its last reply, not its
# batch's (rows wait their turn in the pool).
sending <- function(job) if (is.null(job$call$sent)) { job$call$sent <- TRUE; job$call$started <- as.numeric(Sys.time()) }
ended <- function(job) if (!is.null(job$call)) job$call$ended_at <- as.numeric(Sys.time())

LMCC_TRUNCATED_ADVICE <- "; raise max_tokens or ask for less"

# A token count as text: lm15 keeps a JSON number's own spelling.
count_text <- function(x) { x <- unclass(x); if (is.character(x)) x else format(x, scientific = FALSE, trim = TRUE) }

# A parse-truncated refusal that says what happened (functions.md, "When the
# reply cannot be read"): how much went to thinking, whose limit it was and
# whether it can be raised, and what lm15 changed in the request. `limit`: the
# max_tokens this request set, or NULL.
cut_off <- function(refusal, response, limit) {
  hint <- refusal$hint
  if (endsWith(hint, LMCC_TRUNCATED_ADVICE)) hint <- substr(hint, 1L, nchar(hint) - nchar(LMCC_TRUNCATED_ADVICE))
  thought <- as.numeric(unclass(response$usage$reasoning_tokens %||% 0))
  total <- as.numeric(unclass(response$usage$output_tokens %||% 0))
  if (thought > 0) hint <- paste0(hint, if (total >= thought)
    sprintf("; the model spent %s of its %s output tokens thinking", count_text(thought), count_text(total)) else
    sprintf("; the model spent %s tokens thinking", count_text(thought)))
  notes <- response$adaptations %||% list()
  chosen <- NULL
  for (a in notes) if (identical(a$field, "config.max_tokens") && identical(a$action, "defaulted")) { chosen <- a$applied; break }
  hint <- paste0(hint, if (!is.null(limit)) sprintf("; raise max_tokens (it was %s) or ask for less", count_text(limit))
    else if (!is.null(chosen)) sprintf(paste0("; no max_tokens was set, and lm15 sent %s, the most it knows this model to allow: ",
                                              "lower the reasoning effort or ask for less"), count_text(chosen))
    else "; no max_tokens was set, so the provider used its own maximum: lower the reasoning effort or ask for less")
  other <- vapply(Filter(function(a) !identical(a$field, "config.max_tokens"), notes),
                  function(a) paste0(a$field, " ", a$action, ": ", a$reason), "")
  if (length(other)) hint <- paste0(hint, " (lm15 adapted the request: ", paste(other, collapse = "; "), ")")
  lmcc::refusal(refusal$code, hint, fix = refusal$fix, partial = refusal$partial)
}

# The first output value that does not fit its shape, as a parse-value
# refusal (functions.md: a value that does not fit its type is an unreadable
# reply). Read by the keywords programs.md lists, as a default and an input
# are: the same shape accepts the same values wherever it is checked.
check_values <- function(plan, values) {
  for (f in lmcc::signature_to_list(plan$signature)$fields) {
    if (f$direction != "output" || (f$purpose %||% "plain") != "plain" || !f$name %in% names(values)) next
    shape <- f$shape
    if (!well_formed(shape, shape, carry = TRUE) || loops(shape)) next
    p <- shape_fault(values[[f$name]], shape, shape, f$name)
    if (!is.null(p)) stop(lmcc::refusal("parse-value", p))
  }
}

# ---------------------------------------------------------------- what a watcher is shown

emit_text <- function(call, field, text) {
  if (!nzchar(text)) return(invisible())
  call_emit(call, "text", list(field = field, answer = identical(field, call$program_json$answer), text = text))
}

# lmcc's stream events that show an output's text (tool calls are shown whole).
show_pieces <- function(call, batch) {
  for (ev in batch) if (identical(ev$kind, "field_delta") && !identical(ev$field, "calls")) emit_text(call, ev$field, ev$text)
}

# A reply that came whole, shown as one text piece per field (streaming.md,
# law 6), and the thinking no output reads.
show_whole <- function(job, response) {
  call <- job$call
  d <- plain_lm15(response)
  reads_thinking <- any(vapply(lmcc::signature_to_list(job$plan$signature)$fields, function(f) identical(f$purpose, "reasoning"), NA))
  if (!reads_thinking) for (p in d$message$parts %||% list()) if (identical(p$type, "thinking") && nzchar(p$text %||% ""))
    call_emit(call, "thinking", list(text = p$text))
  shown <- tryCatch({
    s <- lmcc::reply_stream(job$plan)
    out <- list()
    for (p in d$message$parts %||% list()) if (p$type %in% c("text", "data", "tool_call")) out <- c(out, lmcc::feed(s, p))
    c(out, lmcc::finish(s, d$finish_reason)$events)
  }, error = function(e) NULL)
  if (is.null(shown)) return(invisible())          # an unreadable reply: the re-ask says so
  texts <- list()
  for (ev in shown) if (identical(ev$kind, "field_delta") && !identical(ev$field, "calls")) texts[[ev$field]] <- paste0(texts[[ev$field]] %||% "", ev$text)
  for (f in names(texts)) emit_text(call, f, texts[[f]])
  invisible()
}

begin_request <- function(job) {
  call <- job$call
  if (call$tree$replaying) return(invisible())
  call$requests <- call$requests + 1L
  call_emit(call, "request", list(request = call$requests, model = job$model))
}

asked_again <- function(r) if (identical(r$code, "parse-truncated")) "the reply was cut off; asking again with a larger token budget" else
  sprintf("the reply could not be read (%s); asking again", r$hint)

# ---------------------------------------------------------------- one reply

on_response <- function(job, response, started, seconds, cached = FALSE, streamed = FALSE, first_delta = NULL) {
  call <- job$call
  exchange(call, job$model, job$sent %||% job$request, response, started, seconds, request_hash = job$sent_hash,
           cached = cached, streamed = streamed, first_delta = first_delta)
  if (!cached && !isTRUE(job$replayed)) reply_note(job, response)          # a stored turn keeps every reply
  if (!streamed && (wants_pieces(call) || length(call$tree$sinks))) show_whole(job, response)
  job$responses[[length(job$responses) + 1L]] <- response
  reading <- tryCatch({ r <- lmcc::lm15_read(job$plan, response); check_values(job$plan, r$values); r },
                      lmcc_refusal = function(e) e)
  if (inherits(reading, "lmcc_refusal")) {
    reply_forget(job)                                              # a kept reply that no longer reads is forgotten
    code <- reading$code
    # the budget this request set (functions.md: a cut reply is re-sent with twice it, only when one was set)
    limit <- job$overrides$max_tokens %||% job$settings$max_tokens
    if (code == "parse-truncated") reading <- cut_off(reading, response, limit)
    if (job$retries >= job$settings$retries || !(startsWith(code, "parse-") || code == "format-read-error")) return(fail(job, reading))
    if (code == "parse-truncated" && is.null(limit)) return(fail(job, reading))   # nothing larger to give
    job$retries <- job$retries + 1L
    call_emit(call, "retry", list(reason = asked_again(reading), wait = NULL))
    if (code == "parse-truncated") {
      job$overrides$max_tokens <- 2L * as.integer(limit)
      job$request <- lmcc::lm15_request(job$rendered, job$model, config_of(job$settings, job$overrides))
    } else {
      correction <- sprintf(REASK, reading$hint)
      d <- lm15::as_dict(job$request)
      d$messages <- c(d$messages, list(lm15::as_dict(response$message), lm15::as_dict(lm15::message_user(correction))))
      job$request <- lm15::from_dict(d, "request")
      # the re-ask's request_hash: the request it follows, then the reply's message and the correction
      job$asked$messages <- c(job$asked$messages, list(lm15::as_dict(response$message),
                                                       list(role = "user", parts = list(list(type = "text", text = correction)))))
      job$request_hash <- lmcc::sha256_of(job$asked)
    }
    return(invisible())
  }
  reply_keep(job, response)                                        # only a reply that was read is kept
  job$turn <- lmcc::lm15_step(job$rendered, response)
  calls <- reading$values$calls %||% list()
  job$tool_calls <- c(job$tool_calls, calls)
  if (!length(calls)) {
    job$turn <- lmcc::finish_turn(job$turn)
    # the turn's outputs, the fields FunctAI added included; `calls` is every
    # tool call the model asked for in the call, across steps (functions.md),
    # not the finished turn's last step's list (`[]` once the model answers)
    job$outputs <- lmcc::turn_to_list(job$turn)$outputs
    if (length(job$tools) && "calls" %in% names(job$outputs)) job$outputs["calls"] <- list(job$tool_calls)
    job$probabilities <- reading$probabilities %||% list()      # what the provider measured (TypeSafe's Jev), by output
    job$state <- "done"
    ended(job)
    finished(job)
    return(invisible())
  }
  for (c in calls) {
    out <- tryCatch(tool_step(job, c), functai_turn_waiting = function(w) w)
    if (inherits(out, "functai_turn_waiting")) { job$state <- "waiting"; job$waiting <- out; finished(job); return(invisible()) }
    job$turn <- lmcc::tool_result(job$turn, c$id, out)
  }
  job$steps <- job$steps + 1L
  if (job$steps > job$settings$max_steps)
    return(fail(job, structure(class = c("functai_step_limit", "error", "condition"),
      list(message = sprintf("no answer after %d model steps", job$settings$max_steps), call = NULL))))
  render_next(job)
}

on_error <- function(job, err, started, seconds, streamed = FALSE) {
  exchange(job$call, job$model, job$sent %||% job$request, NULL, started, seconds, error = err, request_hash = job$sent_hash, streamed = streamed)
  reply_drop(job)
  if (inherits(err, "functai_cancelled")) return(fail(job, err))
  if (lm15::retryable(err) && job$api_retries < job$settings$api_retries) {
    wait <- err$retry_after
    if (!is.numeric(wait) || length(wait) != 1L || wait <= 0) wait <- min(30, 2^job$api_retries) * (0.5 + stats::runif(1))
    job$api_retries <- job$api_retries + 1L
    job$not_before <- as.numeric(Sys.time()) + wait
    call_emit(job$call, "retry", list(reason = sprintf("the provider failed (%s); sending again in %.1f s", error_json(err)$type, wait), wait = wait))
    return(invisible())
  }
  fail(job, err)
}

# One tool call of the model: its `tool_call` event (a required journal's
# barrier before a tool that changes things), the approval plugins, the tool
# run (or its kept result, when a turn is resumed), its `tool_result`.
tool_step <- function(job, c) {
  call <- job$call
  call$invocations <- call$invocations + 1L
  n <- call$invocations
  tool <- Filter(function(t) identical(t$name, c$name), job$tools)
  tool <- if (length(tool)) tool[[1L]] else NULL
  input <- c$input %||% lmcc::jobj()
  tool_called(call, list(id = c$id, name = c$name, input = input, invocation = n), changes = tool_changes(tool))
  out <- tool_output(job, tool, c, n)
  call_emit(call, "tool_result", list(id = c$id, name = c$name, output = out, invocation = n))
  out
}

# What a tool call gives the model: a recorded result (a resumed turn), else
# the gate's answer (plugins, approval: run with these inputs, or a refusal),
# then the tool's own result, after the tool_result hooks.
tool_output <- function(job, tool, c, n) {
  call <- job$call
  recorded <- tool_recorded(call, n)
  if (!is.null(recorded)) return(recorded)
  if (is.null(tool)) {
    if (identical(job$settings$tool_errors, "raise")) stop(sprintf("the model called unknown tool %s", c$name), call. = FALSE)
    return(sprintf("error: there is no tool named \"%s\"", c$name))
  }
  gate <- tool_gate(job, tool, c, n)
  if (!is.null(gate$output)) return(gate$output)
  tree_frontier(call$tree)                                       # a tool that runs is something new
  tool_started(call, tool, c, n, gate$input)
  old <- list(current = the$current, invocation = the$invocation)
  the$current <- call; the$invocation <- n
  on.exit({ the$current <- old$current; the$invocation <- old$invocation })
  out <- tryCatch(do.call(tool$fn, gate$input %||% list()), functai_turn_waiting = function(w) stop(w), error = function(e) {
    if (identical(job$settings$tool_errors, "raise")) stop(e)
    sprintf("error: %s: %s", class(e)[[1L]], conditionMessage(e))
  })
  out <- if (is.character(out) && length(out) == 1L) out else lmcc::json_text(plain_json(out))
  out <- tool_result_hooks(job, tool, c, gate$input, out)
  tool_done(call, tool, c, n, out)
  out
}

# Whether a tool changes things: it says so, or says nothing (unknown effects
# are treated as changes: forgetting to declare is safe).
tool_changes <- function(tool) is.null(tool) || !identical(tool$effects, "reads")

# ---------------------------------------------------------------- sending

send_one <- function(router, request) {
  if (inherits(router, c("lm15_router", "lm15_lm", "lm15_fake_lm", "lm15_bound"))) return(lm15::complete(router, request))
  if (is.function(router$complete)) return(router$complete(request))
  cli::cli_abort("a router is an {.fn lm15::new_router}, or a list with {.code resolve} and {.code complete} functions")
}

can_stream <- function(router) (inherits(router, c("lm15_router", "lm15_lm", "lm15_fake_lm", "lm15_bound"))) || is.function(router$stream)

stream_one <- function(router, request, on_event) {
  if (is.function(router$stream) && !inherits(router, c("lm15_router", "lm15_lm"))) return(router$stream(request, on_event))
  lm15::stream(router, request, on_event)
}

# Is this request one curl can send as built? lm15 sends some requests its own
# way (a subscription that streams, a stop sequence honoured client-side,
# judgments scored by token): those go through lm15::complete.
wire_for <- function(router, request) {
  if (!inherits(router, "lm15_router") || !is.null(router$transport)) return(NULL)
  d <- lm15::as_dict(request)
  if (!is.null(d$config$probabilities)) return(NULL)
  res <- lm15::resolve(router, d$model)
  lm <- lm15::router_lm(router, d$model)
  if (!identical(lm$definition$access$backend %||% "api", "api")) return(NULL)
  d$model <- res$model
  routed <- lm15::from_dict(d, "request")
  wire <- lm15::build_request(lm, routed)
  fields <- vapply(wire$adaptations %||% list(), function(a) as.character(a$field %||% ""), "")
  if (any(grepl("stop", fields, fixed = TRUE))) return(NULL)
  list(wire = wire, lm = lm, request = routed)
}

# Before a request goes out: a stop asked for, the `request` hooks (which may
# replace it: then it has no request_hash), a reply already known (a turn being
# resumed, the reply cache: answered at once, as a cached exchange), and its
# `request` event. Returns TRUE when the job still has to send it.
before_send <- function(job) {
  call <- job$call
  stopped <- tryCatch({ check_cancelled(call); NULL }, error = identity)
  if (!is.null(stopped)) { fail(job, stopped); return(FALSE) }
  hooked <- tryCatch(request_hooks(job, job$request), error = identity)
  if (inherits(hooked, "error")) { fail(job, hooked); return(FALSE) }
  job$sent <- hooked$request
  job$sent_hash <- if (isTRUE(hooked$replaced)) NULL else job$request_hash
  if (isTRUE(hooked$replaced)) call$replayable <- FALSE
  hit <- reply_lookup(job, job$sent)
  if (inherits(hit, "functai_wait")) { job$not_before <- as.numeric(Sys.time()) + 0.05; return(FALSE) }   # another flight holds the key
  if (is.null(hit)) tree_frontier(call$tree)                     # a request with no kept reply is something new
  begin_request(job)
  if (!is.null(hit)) {
    sending(job)
    tryCatch(on_response(job, hit, as.numeric(Sys.time()), 0, cached = TRUE), error = function(e) fail(job, e))
    return(FALSE)
  }
  TRUE
}

run_jobs <- function(jobs, router, concurrency) {
  concurrency <- max(1L, as.integer(concurrency))
  now <- function() as.numeric(Sys.time())
  pending <- function() Filter(function(j) j$state == "send", jobs)
  pooled <- inherits(router, "lm15_router") && is.null(router$transport)
  # a live reader watching the only call in flight sees its text piece by piece; a column goes
  # through the pool, and each reply is shown whole (one piece per field)
  alone <- length(jobs) == 1L
  # a live reader watching the only call in flight streams it from the provider; any other call goes
  # through the pool, which lets a running conversation turn renew its lease and see a stop while it waits
  stream_alone <- alone && streams_live(jobs[[1L]]$call) && can_stream(router)
  if (!pooled || stream_alone) {
    repeat {                                   # one at a time: fakes, custom transports, a watched call
      ready <- Filter(function(j) j$not_before <= now(), pending())
      if (!length(ready)) {
        waiting <- pending(); if (!length(waiting)) break
        Sys.sleep(max(0, min(vapply(waiting, function(j) j$not_before, 0)) - now())); next
      }
      for (job in ready) if (job$state == "send") step_job(job, router, stream = alone && streams_live(job$call) && can_stream(router))
    }
    return(invisible())
  }
  pool <- curl::new_pool(total_con = concurrency, host_con = concurrency)
  in_flight <- 0L
  queue <- new.env(); queue$jobs <- jobs
  submit <- function(job) {
    if (!before_send(job)) return(FALSE)
    plan <- tryCatch(wire_for(router, job$sent), error = identity)
    if (inherits(plan, "error")) { on_error(job, plan, now(), 0); return(FALSE) }
    if (is.null(plan)) { send_job(job, router); return(FALSE) }
    sending(job)
    started <- now()
    h <- curl::new_handle(url = plan$wire$url)
    curl::handle_setopt(h, customrequest = plan$wire$method, followlocation = FALSE, timeout = 300, connecttimeout = 30)
    if (length(plan$wire$body)) curl::handle_setopt(h, postfields = plan$wire$body)
    curl::handle_setheaders(h, .list = plan$wire$headers)
    job$in_flight <- TRUE
    job$handle <- h
    in_flight <<- in_flight + 1L
    finish <- function(response, err = NULL) {
      if (!isTRUE(job$in_flight)) return(invisible())         # cancelled meanwhile
      in_flight <<- in_flight - 1L
      job$in_flight <- FALSE
      job$handle <- NULL
      tryCatch(if (is.null(err)) on_response(job, response, started, now() - started) else on_error(job, err, started, now() - started),
               functai_turn_waiting = function(w) { job$state <- "waiting"; job$waiting <- w; finished(job) },
               error = function(e) fail(job, e))
      fill()
    }
    curl::multi_add(h, pool = pool,
      done = function(res) {
        parsed <- tryCatch(lm15::parse_response(plan$lm, plan$request, res$content, status = res$status_code,
                                                headers = curl::parse_headers_list(res$headers)), error = identity)
        # what lm15 adapted in the request rides on the reply, as lm15::complete does (MAP-13),
        # so the call log and a cut-off refusal can say it
        noted <- if (identical(plan$lm$adaptations, "silent")) list() else plan$wire$adaptations
        if (!inherits(parsed, "error") && length(noted) && !length(parsed$adaptations)) parsed["adaptations"] <- list(noted)
        if (inherits(parsed, "error")) finish(NULL, parsed) else finish(parsed)
      },
      # curl's message can hold the address, and an address can hold a key: only its kind leaves
      fail = function(msg) finish(NULL, lm15::lm15_error("HTTP transfer failed or timed out; credential-bearing diagnostics are suppressed.", code = "transport")))
    TRUE
  }
  fill <- function() {
    while (in_flight < concurrency) {
      ready <- Filter(function(j) j$state == "send" && !isTRUE(j$in_flight) && j$not_before <= now(), queue$jobs)
      if (!length(ready)) return(invisible())
      submit(ready[[1L]])
    }
  }
  # every second while requests are in flight: a conversation turn's heartbeat (its lease renewed, a stop
  # asked from another process), and a call cancelled meanwhile stopped where it is
  tick <- function() for (job in queue$jobs) if (isTRUE(job$in_flight)) {
    if (!is.null(job$call$turn_run)) turn_check_stop(job$call$turn_run)
    if (call_cancelled(job$call)) {
      curl::multi_cancel(job$handle)
      in_flight <<- in_flight - 1L
      job$in_flight <- FALSE; job$handle <- NULL
      on_error(job, cancelled_error(), job$call$started %||% now(), 0)
    }
  }
  repeat {
    fill()
    if (in_flight > 0L) { curl::multi_run(timeout = 1, pool = pool); tick(); next }
    waiting <- pending()
    if (!length(waiting)) break
    Sys.sleep(max(0, min(vapply(waiting, function(j) j$not_before, 0)) - now()))
  }
  invisible()
}

# Whether a stream reads this call live (not only observers and journals,
# which get the kept form and are given whole replies as one piece per field
# when the call runs among others).
streams_live <- function(call) wants_pieces(call)

step_job <- function(job, router, stream = FALSE) {
  if (!before_send(job)) return(invisible())
  send_job(job, router, stream)
}

send_job <- function(job, router, stream = FALSE) {
  sending(job)
  started <- as.numeric(Sys.time())
  if (stream) return(stream_job(job, router, started))
  response <- tryCatch(send_one(router, job$sent), error = identity)
  seconds <- as.numeric(Sys.time()) - started
  tryCatch(if (inherits(response, "error")) on_error(job, response, started, seconds) else on_response(job, response, started, seconds),
           functai_turn_waiting = function(w) { job$state <- "waiting"; job$waiting <- w; finished(job) },
           error = function(e) fail(job, e))
}

# One request streamed: each piece of an output's text is shown as it comes
# (lmcc's stream reader, which never decides the call: a piece it cannot read
# stops the view, and the whole reply is read after), thinking no output reads
# too; a stream closed meanwhile stops it at its next piece.
stream_job <- function(job, router, started) {
  call <- job$call
  view <- lmcc::reply_stream(job$plan)
  viewing <- TRUE; first <- NULL; events <- list(); reason <- NULL
  reads_thinking <- any(vapply(lmcc::signature_to_list(job$plan$signature)$fields, function(f) identical(f$purpose, "reasoning"), NA))
  response <- tryCatch({
    stream_one(router, job$sent, function(e) {
      if (!is.null(call$turn_run)) turn_check_stop(call$turn_run)
      if (call_cancelled(call)) stop(cancelled_error())
      events[[length(events) + 1L]] <<- e
      ev <- lmcc::lm15_plain(e)
      if (identical(ev$type, "end")) reason <<- ev$finish_reason
      if (!identical(ev$type, "delta")) return(invisible())
      if (is.null(first)) first <<- as.numeric(Sys.time()) - started
      d <- ev$delta
      if (identical(d$type, "thinking") && !reads_thinking && nzchar(d$text %||% "")) call_emit(call, "thinking", list(text = d$text))
      if (viewing) {
        got <- tryCatch(lmcc::feed(view, d), error = function(err) NULL)
        if (is.null(got)) viewing <<- FALSE else show_pieces(call, got)
      }
    })
  }, error = identity)
  if (!inherits(response, "error") && viewing) tryCatch(show_pieces(call, lmcc::finish(view, reason)$events), error = function(e) NULL)
  if (!inherits(response, "error") && !inherits(response, "lm15_Response")) response <- tryCatch(lm15::materialize_response(events, job$sent), error = identity)
  seconds <- as.numeric(Sys.time()) - started
  tryCatch(if (inherits(response, "error")) on_error(job, response, started, seconds, streamed = TRUE)
           else on_response(job, response, started, seconds, streamed = TRUE, first_delta = first),
           functai_turn_waiting = function(w) { job$state <- "waiting"; job$waiting <- w; finished(job) },
           error = function(e) fail(job, e))
}
