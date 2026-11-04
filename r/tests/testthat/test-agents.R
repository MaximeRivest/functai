# The contract's cases for long agents and pipelines (contract/cases/agents),
# the answer so far (cases/partial) and notes a program declares
# (cases/programs of kind notes): the same cases Python and TypeScript pass.
# The setup cases are in test-contract.R, with the functions they define.

agent_cases <- function(folder) {
  files <- sort(list.files(file.path(contract_root(), "cases", folder), pattern = "\\.json$", full.names = TRUE), method = "radix")
  stats::setNames(lapply(files, read_json_file), sub("\\.json$", "", basename(files)))
}

# What a tool written in R returns for the case's data: a text, lm15's part, or a list of them.
native_result <- function(x) {
  if (is.list(x) && !is.null(names(x)) && is_str(x$type)) return(lm15::from_dict(x, "part"))
  if (is.list(x)) return(lapply(x, native_result))
  x
}

for (name in names(agent_cases("agents"))) {
  test_that(paste("agents case", name), {
    c <- agent_cases("agents")[[name]]
    if (c$kind == "result") {
      result <- result_of(native_result(c$returned))
      expect_identical(plain(result), plain(c$expect$result))
      event <- list(output = result_text(result))
      if (!is_str(result)) event$parts <- result
      expect_identical(plain(event), plain(c$expect$event))
    } else if (c$kind == "state") {
      now <- unix_of(c$now)
      old <- the$conversation_clock
      the$conversation_clock <- function() now
      on.exit(the$conversation_clock <- old)
      log <- apply_records(new_conv_log(), c$records)
      e <- c$expect
      states <- lapply(log$turns, turn_state_of)
      expect_identical(plain(states[order(names(states))]), plain(e$states[order(names(e$states))]))
      later <- Filter(length, lapply(log$turns, function(st) lapply(undelivered(st, log$delivered), function(t) t$key)))
      expect_identical(plain(if (length(later)) later else lmcc::jobj()), plain(e$later))
      unf <- Filter(length, lapply(log$turns, function(st) lapply(unfinished(st), function(x) x$invocation)))
      expect_identical(plain(if (length(unf)) unf else lmcc::jobj()), plain(e$unfinished))
      delivered <- lapply(log$delivered, function(r) r$output)
      expect_identical(plain(if (length(delivered)) delivered else lmcc::jobj()), plain(e$delivered))
      waiting <- Filter(function(st) identical(turn_state_of(st), "waiting"), log$turns)
      resumes <- lapply(waiting, function(st) if (!length(unanswered(st)) && !length(undelivered(st, log$delivered))) TRUE else list(refuses = "turn-state"))
      expect_identical(plain(if (length(resumes)) resumes else lmcc::jobj()), plain(e$resumes))
    } else if (c$kind == "kept") {
      keeps <- lapply(c$kept, function(k) new_keep(k$inputs, k$outputs))
      got <- kept_log(c$events, keeps)
      expect_identical(plain(lapply(got, unclass)), plain(c$expect$events))
    } else if (c$kind == "limits") {
      plugin <- do.call(limits, c$limits)
      sp <- c$spent
      spent <- ai_spent(sp$exchanges, sp$seconds, sp$reported %||% 0)
      at <- c$at
      handler <- plugin$handlers[[at$hook]][[1L]]
      got <- if (at$hook == "before_call") handler(list(spent = spent, lm = at$lm))
             else handler(list(spent = spent, step = at$step, max_steps = at$max_steps, tools = as.character(unlist(at$tools)),
                               last_step = at$step >= at$max_steps))
      if (is.null(c$expect$change)) expect_null(got)
      else expect_identical(plain(change_record(got)), plain(c$expect$change))
    } else if (c$kind == "spent") {
      s <- ai_spent(c$exchanges, 0, c$reported %||% 0)
      expect_identical(plain(s$usage), plain(c$expect$usage))
      expect_equal(s$tokens, c$expect$tokens)
      got <- ai_cost(s, c$prices)
      if (is.null(c$expect$cost)) expect_null(got) else expect_lt(abs(got - c$expect$cost), 1e-12)
    } else if (c$kind == "pictures") {
      got <- pictures_to_drop(lapply(c$results, function(r) list(invocation = r$invocation, name = "look", input = list(), output = r$output)),
                              c$last, c$at_once)
      expect_identical(plain(if (length(got)) got else lmcc::jobj()), plain(c$expect$replace))
    } else fail(paste("a kind of agents case this harness does not know:", c$kind))
  })
}
test_that("the contract has agents cases", expect_gte(length(agent_cases("agents")), 37L))

for (name in names(agent_cases("partial"))) {
  test_that(paste("partial case", name), {
    c <- agent_cases("partial")[[name]]
    for (check in c$checks) {
      got <- answer_so_far(check$text, c$shape)
      expect_identical(plain(if (is.null(got)) list(nothing = TRUE) else list(value = got)), plain(check$expect), info = check$text)
    }
  })
}

for (name in names(Filter(function(c) identical(c$program, "notes"), agent_cases("programs")))) {
  test_that(paste("program case", name, "(notes)"), {
    c <- agent_cases("programs")[[name]]
    now <- new.env()
    # one row: a list (a record, a list of records) is one row of a list column, as an R user gives one
    sample_of <- function(iface) stats::setNames(lapply(iface$inputs, function(f) { v <- sample_value(f$shape); if (is.list(v)) list(v) else v }),
                                                 vapply(iface$inputs, function(f) f$name, ""))
    chain <- c$chain
    program <- program_from_interface(chain[[length(chain)]], function(...) { ai_note(now$name, now$data); "ok" }, name = paste0("p", length(chain)))
    for (i in rev(seq_len(length(chain) - 1L))) {
      inner <- program; inner_iface <- chain[[i + 1L]]
      program <- local({
        inner <- inner; inner_iface <- inner_iface
        program_from_interface(chain[[i]], function(...) do.call(inner, sample_of(inner_iface)), name = paste0("p", i))
      })
    }
    for (check in c$checks) {
      now$name <- check$name; now$data <- check$data
      got <- tryCatch({ with_ai_config(do.call(program, sample_of(chain[[1L]])), on_error = "stop"); list(ok = TRUE) },
                      functai_refusal = function(e) list(refuses = e$code, field = e$field))
      expect_identical(plain(got), plain(check$expect), info = check$name)
    }
  })
}
