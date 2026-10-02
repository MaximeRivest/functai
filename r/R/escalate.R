# Escalation (contract/streaming.md, law 4): when the first model is less sure
# of its answer than `escalate_below` (default 0.9), another model, or another
# AI function, answers instead. The call shows a `retry` saying why, then the
# other's request (a model) or its call as a child of this one (an AI
# function); the record says `escalated`. A first model that measures no
# confidence (only baked models and TypeSafe's Jev do) cannot escalate.

escalate <- function(core, job, s) {
  target <- if (isTRUE(the$escalating)) core$own$escalate_to else s$escalate_to
  if (is.null(target)) return(job)
  call <- job$call
  conf <- confidence_of(job$outputs, job$probabilities)
  if (is.null(conf)) stop(sprintf("%s: escalate_to needs a first model that measures its confidence (a baked model, TypeSafe's Jev); %s gave no probabilities",
                                  core$definition$name, job$model), call. = FALSE)
  threshold <- as.numeric(s$escalate_below %||% 0.9)
  if (conf >= threshold) return(job)
  who <- if (inherits(target, "functai_fn")) core_of(target)$definition$name else if (inherits(target, "functai_baked")) target$model else as.character(target)
  call_emit(call, "retry", list(reason = sprintf("the first model was %d%% sure (less than %d%%); %s answers instead", round(100 * conf), round(100 * threshold), who), wait = NULL))
  call$escalated <- TRUE
  if (inherits(target, "functai_fn")) {
    # the target follows its own escalate_to (a longer chain), never one around this call
    tcore <- core_of(target)
    old <- list(current = the$current, escalating = the$escalating)
    the$current <- call; the$escalating <- TRUE
    on.exit({ the$current <- old$current; the$escalating <- old$escalating })
    given <- job$inputs[intersect(names(tcore$definition$inputs), names(job$inputs))]
    rows <- input_rows(tcore, given, n = 1L)
    got <- run_rows(tcore, rows)[[1L]]
    if (!is.null(got$error)) stop(got$error)
    out <- new.env(parent = emptyenv())
    for (k in ls(job, all.names = TRUE)) assign(k, get(k, envir = job), envir = out)
    out$outputs <- got$outputs[intersect(names(core$definition$outputs), names(got$outputs))]
    out$probabilities <- got$probabilities %||% list(); out$turn <- got$turn; out$model <- got$model
    return(out)
  }
  # this call only: the other model, and no second escalation from it
  s2 <- s; s2$lm <- target; s2$escalate_to <- NULL
  r <- route(s2)
  s2 <- adjust_settings(s2, r$provider, r$wire)
  plan <- bind_layout(s2$adapter, s2$template, signature_of(core, s2), call_capabilities(r$provider, r$wire, s2), r$provider)
  job2 <- new_job(plan, past_turns(core, plan), job$inputs, s2, r$model, core$tools, call, core)
  job2$inputs <- job$inputs
  old <- the$current; the$current <- call
  on.exit(the$current <- old)
  run_jobs(list(job2), r$router, 1L)
  if (!identical(job2$state, "done")) stop(job2$error)
  job2
}
