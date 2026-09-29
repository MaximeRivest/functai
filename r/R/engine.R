# One call per row: lay it out (lmcc), send it (lm15), read the reply (lmcc),
# run tools until the model answers (contract/functions.md, "When the reply
# cannot be read"). Each row is a job that says what to send next; the
# scheduler keeps up to `concurrency` requests in flight over curl, so a
# column of 200 rows takes about as long as 200 / concurrency calls.

REASK <- "Your reply could not be read: %s. Reply again, in exactly the form the instructions give."

new_job <- function(plan, past, inputs, settings, model, tools, call) {
  job <- new.env(parent = emptyenv())
  job$plan <- plan; job$past <- past; job$settings <- settings; job$model <- model; job$tools <- tools; job$call <- call
  job$state <- "send"; job$not_before <- 0; job$retries <- 0L; job$api_retries <- 0L; job$steps <- 1L
  job$responses <- list(); job$overrides <- list()
  values <- prepare_inputs(plan$signature, inputs)
  if (length(tools)) values$tools <- lapply(tools, function(t) list(name = t$name, description = t$description, parameters = t$parameters))
  job$turn <- lmcc::new_turn(plan, values)
  render_next(job)
  job
}

render_next <- function(job) {
  job$rendered <- lmcc::render(job$plan, job$turn, if (length(job$past)) job$past else NULL)
  job$request_hash <- lmcc::sha256_of(lmcc::request_of(job$rendered))
  job$request <- lmcc::lm15_request(job$rendered, job$model, config_of(job$settings, job$overrides))
  job$retries <- 0L
}

fail <- function(job, err) { job$state <- "failed"; job$error <- err; ended(job) }

# A call's own time: from its first request to its last reply, not its
# batch's (rows wait their turn in the pool).
sending <- function(job) if (is.null(job$call$sent)) { job$call$sent <- TRUE; job$call$started <- as.numeric(Sys.time()) }
ended <- function(job) if (!is.null(job$call)) job$call$ended <- as.numeric(Sys.time())

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

on_response <- function(job, response, started, seconds) {
  exchange(job$call, job$model, job$request, response, started, seconds, request_hash = job$request_hash)
  job$responses[[length(job$responses) + 1L]] <- response
  reading <- tryCatch({ r <- lmcc::lm15_read(job$plan, response); check_values(job$plan, r$values); r },
                      lmcc_refusal = function(e) e)
  if (inherits(reading, "lmcc_refusal")) {
    code <- reading$code
    if (job$retries >= job$settings$retries || !(startsWith(code, "parse-") || code == "format-read-error")) return(fail(job, reading))
    job$retries <- job$retries + 1L
    if (code == "parse-truncated") {
      job$overrides$max_tokens <- 2L * as.integer(job$overrides$max_tokens %||% job$settings$max_tokens %||% 1024L)
      job$request <- lmcc::lm15_request(job$rendered, job$model, config_of(job$settings, job$overrides))
    } else {
      d <- lm15::as_dict(job$request)
      d$messages <- c(d$messages, list(lm15::as_dict(response$message), lm15::as_dict(lm15::message_user(sprintf(REASK, reading$hint)))))
      job$request <- lm15::from_dict(d, "request")
    }
    return(invisible())
  }
  job$turn <- lmcc::lm15_step(job$rendered, response)
  calls <- reading$values$calls %||% list()
  if (!length(calls)) {
    job$turn <- lmcc::finish_turn(job$turn)
    # the turn's outputs, the fields FunctAI added included: `calls` is what
    # lmcc's finished turn holds for it (its last model step's, `[]` once the
    # model answers); the calls made on the way are in the turn's steps and
    # the call's exchanges
    job$outputs <- lmcc::turn_to_list(job$turn)$outputs
    job$probabilities <- reading$probabilities %||% list()      # what the provider measured (TypeSafe's Jev), by output
    job$state <- "done"
    ended(job)
    return(invisible())
  }
  for (c in calls) job$turn <- lmcc::tool_result(job$turn, c$id, run_tool(job, c))
  job$steps <- job$steps + 1L
  if (job$steps > job$settings$max_steps)
    return(fail(job, structure(class = c("functai_step_limit", "error", "condition"),
      list(message = sprintf("no answer after %d model steps", job$settings$max_steps), call = NULL))))
  render_next(job)
}

on_error <- function(job, err, started, seconds) {
  exchange(job$call, job$model, job$request, NULL, started, seconds, error = err, request_hash = job$request_hash)
  if (lm15::retryable(err) && job$api_retries < job$settings$api_retries) {
    wait <- err$retry_after
    if (!is.numeric(wait) || length(wait) != 1L || wait <= 0) wait <- min(30, 2^job$api_retries) * (0.5 + stats::runif(1))
    job$api_retries <- job$api_retries + 1L
    job$not_before <- as.numeric(Sys.time()) + wait
    return(invisible())
  }
  fail(job, err)
}

run_tool <- function(job, call) {
  tool <- Filter(function(t) identical(t$name, call$name), job$tools)
  if (!length(tool)) {
    if (identical(job$settings$tool_errors, "raise")) stop(sprintf("the model called unknown tool %s", call$name), call. = FALSE)
    return(sprintf("error: there is no tool named \"%s\"", call$name))
  }
  old <- the$current; the$current <- job$call; on.exit(the$current <- old)
  out <- tryCatch(do.call(tool[[1L]]$fn, call$input %||% list()), error = function(e) {
    if (identical(job$settings$tool_errors, "raise")) stop(e)
    sprintf("error: %s: %s", class(e)[[1L]], conditionMessage(e))
  })
  if (is.character(out) && length(out) == 1L) out else lmcc::json_text(plain_json(out))
}

# ---------------------------------------------------------------- sending

send_one <- function(router, request) {
  if (inherits(router, c("lm15_router", "lm15_lm", "lm15_fake_lm", "lm15_bound"))) return(lm15::complete(router, request))
  if (is.function(router$complete)) return(router$complete(request))
  cli::cli_abort("a router is an {.fn lm15::new_router}, or a list with {.code resolve} and {.code complete} functions")
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

run_jobs <- function(jobs, router, concurrency) {
  concurrency <- max(1L, as.integer(concurrency))
  now <- function() as.numeric(Sys.time())
  pending <- function() Filter(function(j) j$state == "send", jobs)
  if (!inherits(router, "lm15_router") || !is.null(router$transport)) {
    repeat {                                   # one at a time: fakes, custom transports
      ready <- Filter(function(j) j$not_before <= now(), pending())
      if (!length(ready)) {
        waiting <- pending(); if (!length(waiting)) break
        Sys.sleep(max(0, min(vapply(waiting, function(j) j$not_before, 0)) - now())); next
      }
      for (job in ready) step_job(job, router)
    }
    return(invisible())
  }
  pool <- curl::new_pool(total_con = concurrency, host_con = concurrency)
  in_flight <- 0L
  queue <- new.env(); queue$jobs <- jobs
  submit <- function(job) {
    plan <- tryCatch(wire_for(router, job$request), error = identity)
    if (inherits(plan, "error")) { on_error(job, plan, now(), 0); return(FALSE) }
    if (is.null(plan)) { step_job(job, router); return(FALSE) }
    sending(job)
    started <- now()
    h <- curl::new_handle(url = plan$wire$url)
    curl::handle_setopt(h, customrequest = plan$wire$method, followlocation = FALSE, timeout = 300, connecttimeout = 30)
    if (length(plan$wire$body)) curl::handle_setopt(h, postfields = plan$wire$body)
    curl::handle_setheaders(h, .list = plan$wire$headers)
    job$in_flight <- TRUE
    in_flight <<- in_flight + 1L
    finish <- function(response, err = NULL) {
      in_flight <<- in_flight - 1L
      job$in_flight <- FALSE
      tryCatch(if (is.null(err)) on_response(job, response, started, now() - started) else on_error(job, err, started, now() - started),
               error = function(e) fail(job, e))
      fill()
    }
    curl::multi_add(h, pool = pool,
      done = function(res) {
        parsed <- tryCatch(lm15::parse_response(plan$lm, plan$request, res$content, status = res$status_code,
                                                headers = curl::parse_headers_list(res$headers)), error = identity)
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
  repeat {
    fill()
    if (in_flight > 0L) { curl::multi_run(pool = pool); next }
    waiting <- pending()
    if (!length(waiting)) break
    Sys.sleep(max(0, min(vapply(waiting, function(j) j$not_before, 0)) - now()))
  }
  invisible()
}

step_job <- function(job, router) {
  sending(job)
  started <- as.numeric(Sys.time())
  response <- tryCatch(send_one(router, job$request), error = identity)
  seconds <- as.numeric(Sys.time()) - started
  tryCatch(if (inherits(response, "error")) on_error(job, response, started, seconds) else on_response(job, response, started, seconds),
           error = function(e) fail(job, e))
}
