# Plugins (contract/plugins.md): code that changes what programs do, through
# seven hooks, with every change returned as data and recorded, so a rated
# call is asked again as it was. FunctAI's own features are plugins written
# with these hooks: `approve` is the `approval` plugin; compaction() and
# delegate() are in builtins.R.

PLUGIN_API <- 1L
HOOKS <- list(turn_start = "inputs", context = c("keep", "without", "sections"),
              before_call = c("instruction", "sections", "lm", "settings", "tools"), request = character(0),
              tool_call = c("inputs", "block"), tool_result = "output", turn_end = character(0))
CHANGE_FIELDS <- c("instruction", "sections", "lm", "settings", "tools", "keep", "without", "inputs", "block", "output")

plugin_error <- function(code, message, plugin = NULL, hook = NULL) {
  rlang::error_cnd(c("functai_plugin_error", paste0("functai_", gsub("-", "_", code)), "functai_refusal"), code = code,
                   plugin = plugin, hook = hook, message = message, functai_type = "PluginError")
}

#' A plugin: hooks that change what programs do
#'
#' A named, versioned set of hooks: functions FunctAI calls at fixed points
#' of a call or a conversation's turn. Each is given an event and returns an
#' [ai_change()] (or `NULL`: no opinion), and every change is recorded in the
#' call's record, so a rated call is asked again as it was.
#'
#' | hook | when | it may change |
#' |---|---|---|
#' | `turn_start` | a conversation's turn is sent, before it is recorded | `inputs` |
#' | `context` | the turn's earlier turns are chosen | `keep`, `without`, `sections` |
#' | `before_call` | an AI function is about to be asked (a program's steps too) | `instruction`, `sections`, `lm`, `settings`, `tools` |
#' | `request` | a provider request is about to be sent | returns another lm15 request |
#' | `tool_call` | a tool is about to run | `inputs`, `block`; or [ai_ask()] a person |
#' | `tool_result` | a tool ran | `output` |
#' | `turn_end` | a turn ended (hears only; may [keep_entry()] entries) | |
#'
#' Plugins are set as settings: an AI function's or a program's own
#' (`.plugins`), a conversation's, a block's (`with_ai_config(plugins = )`),
#' or the session's (`ai_config(plugins = )`). The program's own run first,
#' then each layer out to the session's, so the host's handlers see what the
#' program's did and have the last word. A host's `program_plugins = FALSE`
#' drops the program's own. A handler runs in the call's own R session, with
#' all its power: a plugin is code you trust.
#' @param name Lower-case letters, digits, `_` and `-`, starting with a
#'   letter: its changes and entries are named by it.
#' @param version Its own version, recorded with each change.
#' @param ... Handlers by hook: `before_call = function(call) ...`.
#' @param description What it does.
#' @param api The plugin API it was written for (1).
#' @return A plugin.
#' @examples
#' careful <- ai_plugin("careful", version = "1.0.0",
#'   before_call = function(call) ai_change(sections = "Only point out problems."),
#'   tool_call = function(t) if (t$name == "delete_file") ai_change(block = "not here"))
#' careful
#' @export
ai_plugin <- function(name, version = "0.0.0", ..., description = "", api = 1L) {
  if (!is_str(name) || !grepl("^[a-z][a-z0-9_-]{0,63}$", name))
    stop(plugin_error("plugin-name", sprintf("a plugin's name is lower-case letters, digits, '_' or '-' (at most 64), starting with a letter; not %s", short_json(name))))
  if (!identical(as.integer(api), PLUGIN_API)) stop(plugin_error("plugin-api", sprintf("plugin %s is written for plugin API %s; this functai implements %d", name, api, PLUGIN_API), plugin = name))
  p <- structure(list(name = name, version = as.character(version), api = PLUGIN_API, description = description, handlers = list()), class = "functai_plugin")
  hooks <- list(...)
  for (h in names(hooks)) p <- on_hook(p, h, hooks[[h]])
  p
}

#' @rdname ai_plugin
#' @param plugin A plugin.
#' @param hook A hook's name.
#' @param handler A function of one event.
#' @export
on_hook <- function(plugin, hook, handler) {
  if (!is_str(hook) || !hook %in% names(HOOKS))
    stop(plugin_error("plugin-hook", sprintf("%s: there is no hook %s; hooks: %s", plugin$name, short_json(hook), paste(names(HOOKS), collapse = ", ")), plugin = plugin$name, hook = hook))
  if (!is.function(handler)) cli::cli_abort("{plugin$name}.{hook}: a handler is a function of one event")
  plugin$handlers[[hook]] <- c(plugin$handlers[[hook]], list(handler))
  plugin
}

#' @export
print.functai_plugin <- function(x, ...) {
  cat(sprintf("<ai plugin> %s %s: %s\n", x$name, x$version, if (length(x$handlers)) paste(names(x$handlers), collapse = ", ") else "no hooks"))
  invisible(x)
}

#' What a hook changes
#'
#' A change, as data: each hook may change some fields (see [ai_plugin()]);
#' another refuses `plugin-change`.
#' @param instruction The instruction itself, for this call.
#' @param sections Text added after the instruction, in order.
#' @param lm The model for this call.
#' @param settings lm15 settings for it (`temperature`, `max_tokens`, ...).
#' @param tools The names of the tools offered, from the function's own.
#' @param keep The ids of the earlier turns shown.
#' @param without Fields left out of earlier turns: names for every turn, or
#'   a named list by turn id.
#' @param inputs Inputs replaced, by name (a turn's, or a tool's).
#' @param block The tool may not run, and why.
#' @param output The tool's result as the model is shown it.
#' @return A change.
#' @export
ai_change <- function(instruction = NULL, sections = NULL, lm = NULL, settings = NULL, tools = NULL, keep = NULL, without = NULL,
                      inputs = NULL, block = NULL, output = NULL) {
  out <- list(instruction = instruction, sections = sections, lm = lm, settings = settings, tools = tools, keep = keep,
              without = without, inputs = inputs, block = block, output = output)
  structure(Filter(Negate(is.null), out), class = "functai_change")
}

#' @export
print.functai_change <- function(x, ...) { cat("<ai change> ", paste(sprintf("%s = %s", names(x), vapply(unclass(x), short_json, "")), collapse = ", "), "\n", sep = ""); invisible(x) }

# A change as the record keeps it.
change_record <- function(ch) {
  out <- list()
  for (k in names(ch)) {
    v <- ch[[k]]
    out[[k]] <- if (k %in% c("sections", "tools", "keep")) as.list(as.character(v)) else if (k %in% c("settings", "inputs")) lapply(v, value_json) else value_json(v)
  }
  out
}

# A plugins setting: a list of plugins (or files that define one).
check_plugins <- function(x, call = NULL) {
  if (is.null(x)) return(NULL)
  if (inherits(x, "functai_plugin") || is_str(x)) x <- list(x)
  if (!is.list(x) || !all(vapply(x, function(p) inherits(p, "functai_plugin") || is_str(p), NA)))
    cli::cli_abort("{.arg plugins} is a list of plugins ({.fn ai_plugin}), or files that define one", call = call)
  x
}

#' A plugin from a file
#'
#' Runs an R file that defines `plugin` (an [ai_plugin()]) in an environment
#' of its own, and gives it back. Loading runs the file: load only code you
#' trust. A `plugins` setting may name the file instead.
#' @param path The file.
#' @return A plugin.
#' @export
load_plugin <- function(path) {
  p <- normalizePath(path.expand(path), mustWork = FALSE)
  if (!file.exists(p)) stop(plugin_error("plugin-load", sprintf("%s: no such file", p)))
  key <- paste0(p, "@", as.numeric(file.mtime(p)))
  if (!is.null(the$loaded_plugins[[key]])) return(the$loaded_plugins[[key]])
  env <- new.env(parent = globalenv())
  got <- tryCatch({ sys.source(p, envir = env, keep.source = FALSE); NULL }, error = identity)
  if (!is.null(got)) stop(plugin_error("plugin-load", sprintf("%s: %s", p, conditionMessage(got))))
  if (!exists("plugin", envir = env, inherits = FALSE)) stop(plugin_error("plugin-load", sprintf("%s defines no `plugin <- ai_plugin(...)`", p)))
  if (!inherits(env$plugin, "functai_plugin")) stop(plugin_error("plugin-load", sprintf("%s: `plugin` is not a plugin", p)))
  the$loaded_plugins[[key]] <- env$plugin
  env$plugin
}

resolve_plugin <- function(e) if (is_str(e)) load_plugin(e) else e

# The setting layers around a call, closest first: its own, `extra` (a
# conversation's), each block from the closest out, the session's.
setting_layers <- function(own = list(), extra = list()) {
  c(list(list(where = "own", s = own)), lapply(extra, function(x) list(where = "block", s = x)),
    lapply(rev(the$blocks %||% list()), function(b) list(where = "block", s = b)), list(list(where = "configure", s = the$config %||% list())))
}

# The plugins around a call, in the order their handlers run (plugins.md,
# "Order"): the program's own first, then each layer out; one set in several
# layers runs once, in its outermost; a host's program_plugins = FALSE drops
# the program's own.
plugins_in_order <- function(layers) {
  vetoed <- any(vapply(layers, function(l) l$where != "own" && isFALSE(l$s$program_plugins), NA))
  seen <- character(0); kept <- list()
  for (l in rev(layers)) {
    mine <- list()
    if (!(vetoed && l$where == "own")) for (e in l$s$plugins %||% list()) {
      p <- resolve_plugin(e)
      if (p$name %in% seen) next
      seen <- c(seen, p$name); mine[[length(mine) + 1L]] <- p
    }
    kept[[length(kept) + 1L]] <- mine
  }
  unlist(rev(kept), recursive = FALSE) %||% list()
}

plugins_around <- function(own = list(), extra = list()) plugins_in_order(setting_layers(own, extra))
has_hook <- function(plugins, hook) any(vapply(plugins, function(p) length(p$handlers[[hook]]) > 0L, NA))

# Errors a handler raises that are never a plugin's failure: the call stops,
# or waits, as it would anyway.
passes_through <- function(err) inherits(err, c("functai_refusal", "functai_cancelled", "functai_waiting", "functai_turn_waiting", "interrupt"))

# Every handler of `hook`, in order: each given the event as the changes
# before it left it, its change checked, applied and recorded (in `applied`,
# an environment with `items`). A handler that fails stops the call.
run_hook <- function(hook, plugins, event, applied, apply) {
  allowed <- HOOKS[[hook]]
  for (p in plugins) for (f in p$handlers[[hook]]) {
    event$plugin <- p
    got <- tryCatch(f(event), error = function(err) {
      if (passes_through(err)) stop(err)
      stop(plugin_error("plugin-failed", sprintf("plugin %s failed in %s: %s", p$name, hook, conditionMessage(err)), plugin = p$name, hook = hook))
    })
    event$plugin <- NULL
    if (is.null(got)) next
    if (!inherits(got, "functai_change")) stop(plugin_error("plugin-change", sprintf("plugin %s: %s returns ai_change() or NULL", p$name, hook), plugin = p$name, hook = hook))
    wrong <- setdiff(names(got), allowed)
    if (length(wrong)) stop(plugin_error("plugin-change", sprintf("plugin %s: %s cannot change %s (it may change %s)", p$name, hook,
      paste(wrong, collapse = ", "), if (length(allowed)) paste(allowed, collapse = ", ") else "nothing"), plugin = p$name, hook = hook))
    if (!length(got)) next
    tryCatch(apply(got), error = function(err) {
      if (inherits(err, "functai_plugin_error") || passes_through(err)) stop(err)
      stop(plugin_error("plugin-change", sprintf("plugin %s: %s: %s", p$name, hook, conditionMessage(err)), plugin = p$name, hook = hook))
    })
    applied$items[[length(applied$items) + 1L]] <- list(plugin = p$name, version = p$version, hook = hook, change = change_record(got))
  }
  invisible()
}

new_applied <- function() { a <- new.env(parent = emptyenv()); a$items <- list(); a }

hook_event <- function(kind, ...) {
  e <- new.env(parent = emptyenv())
  fields <- list(...)
  for (k in names(fields)) assign(k, fields[[k]], envir = e)
  e$plugin <- NULL
  structure(e, class = c(paste0("functai_", kind, "_event"), "functai_hook_event"))
}

#' @export
print.functai_hook_event <- function(x, ...) {
  shown <- setdiff(ls(x), c("plugin", "turn_run", "call", "program", "conversation_obj"))
  cat("<", sub("^functai_(.*)_event$", "\\1", class(x)[[1L]]), " event> ", paste(shown, collapse = ", "), "\n", sep = "")
  invisible(x)
}

texts_of <- function(sections) {
  if (!is.character(sections) && !(is.list(sections) && all(vapply(sections, is_str, NA)))) stop("sections are texts")
  s <- as.character(unlist(sections))
  s[nzchar(trim_white(s))]
}

# A call's place in its tree by names (`support/answer`): what a host's rules read.
names_path <- function(call) if (is.null(call)) "" else paste(sub("#.*$", "", strsplit(call$site, "/", fixed = TRUE)[[1L]]), collapse = "/")

# ---------------------------------------------------------------- before_call

LM15_SETTINGS <- c("temperature", "max_tokens", "top_p", "stop", "seed", "reasoning", "tool_choice", "response_format",
                   "frequency_penalty", "presence_penalty", "extensions", "logprobs", "top_logprobs", "parallel_tool_calls")

# The before_call hooks for one row of an AI function: the sections its
# context gives first (its conversation's turn, a row asked again), then each
# handler's. Returns what changes the job: its plan (instruction, sections,
# the tools offered), settings, model.
before_call_hooks <- function(core, call, s, row, plan, past) {
  given <- context_sections(call)
  shown <- context_turns(core, call, plan, s)
  call$sections <- given
  if (!is.null(shown)) { call$saw <- shown$saw; past <- c(past, shown$turns) }
  plugins <- plugins_around(core$own, turn_settings_layers(call))
  out <- list(past = past)
  if (!has_hook(plugins, "before_call") && !length(given)) return(out)
  base <- lmcc::signature_to_list(plan$signature)$instructions
  all_tools <- vapply(core$tools, function(t) t$name, "")
  run <- call$turn_run
  ev <- hook_event("before_call", instruction = base, `function` = core$definition$name, program = core, inputs = row,
                   lm = s$lm, settings = s[intersect(names(s), c("temperature", "max_tokens", "top_p", "seed"))], tools = all_tools,
                   all_tools = all_tools, sections = given, path = names_path(call),
                   conversation = if (!is.null(run)) run$conv$id else NULL, turn = if (!is.null(run)) run$turn else NULL, turn_run = run, call = call)
  applied <- new_applied()
  settings <- s; instruction <- NULL; offered <- NULL
  if (has_hook(plugins, "before_call")) run_hook("before_call", plugins, ev, applied, function(ch) {
    if (!is.null(ch$instruction)) { if (!is_str(ch$instruction) || !nzchar(trim_white(ch$instruction))) stop("instruction is the text of an instruction"); instruction <<- ev$instruction <- ch$instruction }
    if (!is.null(ch$sections)) ev$sections <- c(ev$sections, texts_of(ch$sections))
    if (!is.null(ch$lm)) { if (!is_str(ch$lm)) stop("lm is a model's name"); settings$lm <<- ev$lm <- ch$lm }
    if (!is.null(ch$settings)) {
      bad <- setdiff(names(ch$settings), LM15_SETTINGS)
      if (length(bad)) stop(sprintf("settings are lm15 settings, not %s", paste(bad, collapse = ", ")))
      for (k in names(ch$settings)) {
        if (k %in% c("temperature", "max_tokens", "top_p", "seed", "stop")) settings[[k]] <<- ch$settings[[k]]
        else settings$config[[k]] <<- ch$settings[[k]]
        ev$settings[[k]] <- ch$settings[[k]]
      }
    }
    if (!is.null(ch$tools)) {
      names_ <- as.character(unlist(ch$tools))
      unknown <- setdiff(names_, all_tools)
      if (length(unknown)) stop(sprintf("%s has no tool %s (its tools: %s)", core$definition$name, unknown[[1L]], if (length(all_tools)) paste(all_tools, collapse = ", ") else "none"))
      offered <<- ev$tools <- all_tools[all_tools %in% names_]
    }
  })
  call$changes <- c(call$changes, applied$items)
  sections <- ev$sections
  if (!is.null(instruction) || length(sections)) {
    if (!layout_writes_instruction(plan)) stop(plugin_error("plugin-change", sprintf(
      "%s: plugins changed its instruction, and its layout never writes {instruction}: the change would not reach the model", core$definition$name)))
    sig <- lmcc::signature_to_list(plan$signature)
    sig$instructions <- paste(c(instruction %||% sig$instructions, sections), collapse = "\n\n")
    plan <- rebind_plan(plan, lmcc::signature_from_list(sig), settings)
    out$plan <- plan
  }
  if (!identical(settings$lm, s$lm)) {
    r <- route(settings)
    out$model <- r$model
    settings <- adjust_settings(settings, r$provider, r$wire)
    out$plan <- rebind_plan(out$plan %||% plan, (out$plan %||% plan)$signature, settings, r)
    call$provider <- r$provider
  }
  out$settings <- settings
  if (!is.null(offered)) out$tools <- Filter(function(t) t$name %in% offered, core$tools)
  out
}

# Whether a plan's layout writes the instruction (a template without
# `{instruction}` does not, and would drop what plugins add).
layout_writes_instruction <- function(plan) {
  probe <- "\u2063functai-probe\u2063"
  sig <- lmcc::signature_to_list(plan$signature); sig$instructions <- probe
  tryCatch({
    p <- rebind_plan(plan, lmcc::signature_from_list(sig))
    grepl(probe, lmcc::canonical_json(lmcc::request_of(lmcc::render(p, sample_inputs(p$signature)), "probe")), fixed = TRUE)
  }, error = function(e) TRUE)
}

# The same layout and facts, with another signature (or settings and route).
rebind_plan <- function(plan, sig, settings = NULL, r = NULL) {
  b <- plan$functai_binding
  if (is.null(b)) stop("this plan cannot be laid out again")
  if (!is.null(r)) { b$caps <- call_capabilities(r$provider, r$wire, settings); b$provider <- r$provider }
  out <- bind_layout(b$adapter, b$template, sig, b$caps, b$provider)
  out
}

# The `request` hook (the escape hatch): the request to send, and whether a
# handler replaced it (then no one can rebuild it: no request_hash).
request_hooks <- function(job, request) {
  call <- job$call
  plugins <- plugins_around(job$core$own %||% list(), turn_settings_layers(call))
  if (!has_hook(plugins, "request")) return(list(request = request, replaced = FALSE))
  ev <- hook_event("request", request = request, `function` = call$name, path = names_path(call), turn_run = call$turn_run)
  changed <- FALSE
  for (p in plugins) for (f in p$handlers$request) {
    ev$plugin <- p
    got <- tryCatch(f(ev), error = function(err) {
      if (passes_through(err)) stop(err)
      stop(plugin_error("plugin-failed", sprintf("plugin %s failed in request: %s", p$name, conditionMessage(err)), plugin = p$name, hook = "request"))
    })
    if (is.null(got) || identical(got, ev$request)) next
    if (!inherits(got, "lm15_Request") && !inherits(got, "lm15_value")) stop(plugin_error("plugin-change", sprintf("plugin %s: request returns an lm15 request or NULL", p$name), plugin = p$name, hook = "request"))
    ev$request <- got; changed <- TRUE
    call$changes[[length(call$changes) + 1L]] <- list(plugin = p$name, version = p$version, hook = "request", change = list(request = "replaced"))
  }
  list(request = ev$request, replaced = changed)
}

# ---------------------------------------------------------------- tools: tool_call, approval, tool_result

DENIED <- "The person did not allow this call."
denial <- function(reason) if (is.null(reason) || !nzchar(reason)) DENIED else paste0(DENIED, " Reason: ", reason)

# A tool call a person is asked about.
new_approval <- function(call, n, c, tool, plugin = "approval", question = NULL) {
  path <- paste(c(sub("#.*$", "", strsplit(call$site, "/", fixed = TRUE)[[1L]]), c$name), collapse = "/")
  structure(list(call = call$id, invocation = n, id = c$id, name = c$name, input = c$input %||% lmcc::jobj(),
                 effects = if (is.null(tool)) NULL else tool$effects, path = path, site = call$site, plugin = plugin, question = question),
            class = "functai_approval")
}

#' @export
print.functai_approval <- function(x, ...) { cat(sprintf("<approval> %s #%d (%s)%s\n", x$path, x$invocation, x$effects %||% "effects unknown", if (!is.null(x$question)) paste0(": ", x$question) else "")); invisible(x) }

approval_json <- function(a) {
  out <- list(call = a$call, invocation = a$invocation, id = a$id, name = a$name, input = value_json(a$input))
  out["effects"] <- list(a$effects)
  out$path <- a$path; out$site <- a$site; out$plugin <- a$plugin
  if (!is.null(a$question)) out$question <- a$question
  out
}
approval_from <- function(d) structure(list(call = d$call, invocation = as.integer(d$invocation), id = d$id, name = d$name, input = d$input,
                                            effects = d$effects, path = d$path %||% d$name, site = d$site %||% "", plugin = d$plugin %||% "approval",
                                            question = d$question), class = "functai_approval")

# An approve setting: a function (asked at once), "changes", "all", or tool names and paths.
check_approve <- function(x, call = NULL) {
  if (is.null(x) || is.function(x)) return(x)
  if (is_str(x) && x %in% c("changes", "all")) return(x)
  if (is.character(x) && length(x) && all(nzchar(x))) return(structure(x, class = "functai_approve_list"))
  cli::cli_abort("{.arg approve} is a function, {.val changes}, {.val all}, or tool names and approval paths", call = call)
}

# Whether a rule asks a person about this tool call (tools.md).
asks <- function(rule, a) {
  if (is.null(rule)) return(FALSE)
  if (is.function(rule) || identical(unclass(rule), "changes")) return(!identical(a$effects, "reads"))
  if (identical(unclass(rule), "all")) return(TRUE)
  any(vapply(unclass(rule), function(e) identical(e, a$name) || identical(e, a$path) || endsWith(a$path, paste0("/", e)), NA))
}

verdict_of <- function(answer) {
  if (isTRUE(answer)) return(list(allowed = TRUE, reason = NULL))
  if (isFALSE(answer) || is.null(answer)) return(list(allowed = FALSE, reason = NULL))
  if (is_str(answer)) return(list(allowed = FALSE, reason = answer))
  cli::cli_abort("an approve function answers TRUE, FALSE, or a reason to refuse (a text)")
}

# The built-in approval plugin: `approve` as a rule or a function.
approval_plugin <- function() ai_plugin("approval", version = "1.0.0", description = "approve: ask before tools run, as a rule says",
  tool_call = function(t) {
    rule <- t$settings$approve
    if (!asks(rule, t$approval)) return(NULL)
    ai_ask(t, decide = if (is.function(rule)) rule else NULL)
    NULL
  })

# The tool_call hooks for one tool call, then the approval plugin last, on the
# input they left: list(input) to run it, or list(output) to show instead.
tool_gate <- function(job, tool, c, n) {
  call <- job$call
  plugins <- c(plugins_around(job$core$own %||% list(), turn_settings_layers(call)), list(the$approval_plugin %||% (the$approval_plugin <- approval_plugin())))
  a <- new_approval(call, n, c, tool)
  ev <- hook_event("tool_call", name = c$name, input = c$input %||% lmcc::jobj(), effects = a$effects, path = a$path, invocation = n,
                   id = c$id, `function` = call$name, approval = a, settings = job$settings, turn_run = call$turn_run, call = call, refused = NULL)
  applied <- new_applied()
  blocked <- NULL
  apply <- function(ch) {
    if (!is.null(ch$inputs)) {
      if (!is.list(ch$inputs) || is.null(names(ch$inputs))) stop("inputs is a named list of the tool's inputs")
      base <- if (is.list(ev$input)) ev$input else list()
      for (k in names(ch$inputs)) base[k] <- list(value_json(ch$inputs[[k]]))
      ev$input <- base
    }
    if (!is.null(ch$block)) { if (!is_str(ch$block)) stop("block is the reason, a text"); blocked <<- ch$block }
  }
  out <- tryCatch({
    for (p in plugins) { if (!is.null(blocked) || !is.null(ev$refused)) break; run_hook("tool_call", list(p), ev, applied, apply) }
    NULL
  }, functai_plugin_failed = function(err) {
    warn_once(paste0("tool-call:", err$plugin), sprintf("%s: the tool does not run", conditionMessage(err)))
    blocked <<- sprintf("a check on this tool call failed (%s)", err$plugin)
    applied$items[[length(applied$items) + 1L]] <- list(plugin = err$plugin, version = "", hook = "tool_call", change = list(block = blocked))
    NULL
  })
  call$changes <- c(call$changes, applied$items)
  if (!is.null(blocked)) {
    who <- if (length(applied$items)) applied$items[[length(applied$items)]]$plugin else "?"
    return(list(output = sprintf("This call was blocked (%s): %s", who, blocked)))
  }
  if (!is.null(ev$refused)) return(list(output = ev$refused))
  list(input = ev$input)
}

#' Ask a person whether a tool may run
#'
#' Inside a `tool_call` hook: in a conversation the turn waits, saved, and
#' goes on when someone answers ([approve()]); on a stream, `.each` is given
#' the `approval` event and may answer it with [approve()] or [deny()], and
#' an interactive session is asked at the console otherwise; a plain call
#' refuses (`approval-required`). `decide` answers in place of a person.
#' @param event The `tool_call` event.
#' @param reason Why it asks (shown with the question).
#' @param decide A function of the approval giving `TRUE`, `FALSE` or a
#'   reason to refuse.
#' @return Whether it may run (the model is shown the refusal otherwise).
#' @export
ai_ask <- function(event, reason = NULL, decide = NULL) {
  a <- event$approval
  a$input <- event$input
  a$plugin <- if (is.null(event$plugin)) "approval" else event$plugin$name
  a$question <- reason
  got <- ask_person(event$call, a, decide)
  if (!got$allowed) event$refused <- denial(got$reason)
  got$allowed
}

# Ask whether a tool call may run: decide; a resumed turn's recorded answer;
# else a conversation's turn waits; a watched call asks its stream (then the
# console, when interactive); a plain call refuses approval-required.
ask_person <- function(call, a, decide = NULL) {
  run <- call$turn_run
  if (!is.null(run)) {
    known <- recorded_approval(run, call, a)
    if (!is.null(known)) {
      if (known$fresh) { tree_frontier(call$tree); emit_approved(call, a, known$allowed, known$by, known$reason) }
      return(known)
    }
  }
  tree_frontier(call$tree)
  data <- list(id = a$id, invocation = a$invocation, name = a$name, input = value_json(a$input))
  data["effects"] <- list(a$effects)
  data$path <- a$path; data$to <- the$approvals_to %||% "owner"; data$plugin <- a$plugin
  if (!is.null(a$question)) data$question <- a$question
  asked <- call_emit(call, "approval", data)
  got <- if (!is.null(decide)) c(verdict_of(decide(a)), list(by = NULL))
    else if (!is.null(run)) stop(turn_waiting(a))
    else answer_here(call, a, asked)
  emit_approved(call, a, got$allowed, got$by, got$reason)
  if (!is.null(run)) note_approval(run, call, a, got$allowed, got$reason, got$by)
  got
}

emit_approved <- function(call, a, allowed, by, reason) {
  data <- list(id = a$id, invocation = a$invocation, verdict = if (allowed) "yes" else "no")
  data["by"] <- list(by); data$plugin <- a$plugin
  if (!is.null(reason)) data$reason <- reason
  call_emit(call, "approved", data)
}

approval_required <- function(a) rlang::error_cnd(c("functai_approval_required", "functai_refusal"), code = "approval-required", approval = a,
  functai_type = "ApprovalError",
  message = sprintf("%s: this tool call needs a person's answer (%s), and a plain call has nobody to ask. Give approve a function, watch the call with ai_stream() and answer in .each, or use a conversation, where the turn waits", a$path, a$plugin))

# The answer of whoever watches the call in this session: a stream's `.each`
# answering the `approval` event (approve()/deny() on it), else the console.
answer_here <- function(call, a, asked) {
  key <- paste(a$call, a$invocation, a$plugin)
  ans <- the$answers_here[[key]]
  if (!is.null(ans)) { the$answers_here[[key]] <- NULL; return(ans) }
  watched <- length(Filter(function(s) inside(call$tree, call$id, s$call), call$tree$streams)) > 0L
  if (watched && interactive()) {
    prompt <- sprintf("%s(%s)%s: allow? [y/N] ", a$name, short_text(lmcc::canonical_json(value_json(a$input))), if (!is.null(a$question)) paste0(" (", a$question, ")") else "")
    yes <- tolower(trim_white(readline(prompt))) %in% c("y", "yes")
    return(list(allowed = yes, reason = NULL, by = NULL))
  }
  stop(approval_required(a))
}

#' Answer a person's question about a tool call
#'
#' `approve()` says yes, `deny()` says no (the model is told the person did
#' not allow the call, and why). Give it the `approval` event a stream's
#' `.each` received (the call waiting in this session goes on), or a turn
#' that waits: a conversation's turn goes on when nothing else waits for an
#' answer, here, and its answer is returned (`resume = FALSE`: later, with
#' [resume_turn()]).
#' @param x An `approval` event, or a waiting turn.
#' @param approval Which approval a turn waits for (`turn$waiting[[1]]`, its
#'   invocation number); `NULL` for the only one.
#' @param reason Why not.
#' @param by Who answers.
#' @param resume Whether the turn goes on now.
#' @return The turn's answer, when it went on; else nothing.
#' @export
approve <- function(x, approval = NULL, by = NULL, resume = TRUE) answer_approval(x, approval, TRUE, NULL, by, resume)

#' @rdname approve
#' @export
deny <- function(x, approval = NULL, reason = NULL, by = NULL, resume = TRUE) answer_approval(x, approval, FALSE, reason, by, resume)

answer_approval <- function(x, approval, allowed, reason, by, resume) {
  if (inherits(x, "functai_event")) {
    if (x$kind != "approval") cli::cli_abort("{.fn approve} and {.fn deny} answer an {.val approval} event, or a waiting turn")
    the$answers_here[[paste(x$call, x$invocation, x$plugin %||% "approval")]] <- list(allowed = allowed, reason = reason, by = by)
    return(invisible())
  }
  if (inherits(x, "functai_turn")) return(answer_turn(x, approval, allowed, reason, by, resume))
  cli::cli_abort("{.fn approve} and {.fn deny} answer an {.val approval} event, or a waiting turn")
}

# tool_result: what the model is shown of a tool's result.
tool_result_hooks <- function(job, tool, c, input, out) {
  call <- job$call
  plugins <- plugins_around(job$core$own %||% list(), turn_settings_layers(call))
  if (!has_hook(plugins, "tool_result")) return(out)
  ev <- hook_event("tool_result", name = c$name, input = input, output = out, path = paste(names_path(call), c$name, sep = "/"),
                   invocation = call$invocations, `function` = call$name, turn_run = call$turn_run)
  applied <- new_applied()
  run_hook("tool_result", plugins, ev, applied, function(ch) { if (!is_str(ch$output)) stop("output is the text the model is shown"); ev$output <- ch$output })
  call$changes <- c(call$changes, applied$items)
  ev$output
}
