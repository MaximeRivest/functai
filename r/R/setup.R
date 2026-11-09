# What a program runs on (contract/calls.md, "Setups"; design/18).
#
# A version says what a program is: its instruction, examples, layout, code.
# Its setup says what it runs on: which model answers each step, with which
# settings and plugins, with which tool code, able to reach what. Two runs of
# one version on two models have two setups, so a score, a comparison or a
# release belongs to exactly one thing that ran.

# FunctAI's settings that change what the model is asked or what comes back, with their defaults
SETUP_SETTINGS <- list(max_steps = 8L, retries = 1L, tool_errors = "report", on_unreadable = "raise")

# ---------------------------------------------------------------- the model

# A baked model by its files (weights included): sha256 of its baked.json's hashes.
baked_id <- function(b) baked_id_of(b$meta)

baked_id_of <- function(meta) {
  hashes <- meta$hashes %||% lmcc::jobj()
  w <- meta$weights
  # weights on a service (not in the folder): named by their URI too (calls.md, Setups)
  if (identical(w$form, "remote")) return(lmcc::sha256_of(lmcc::jobj(hashes = hashes, uri = w$uri)))
  lmcc::sha256_of(hashes)
}

# A model as a setup names it: list(provider, model) as lm15 routes it (a bare
# name and the name with its provider are one model), list(baked = id),
# list(provider = NULL, model = name) for a name lm15 cannot route, or NULL
# when none can be named.
setup_model <- function(lm, s) {
  if (inherits(lm, "functai_baked")) return(list(baked = baked_id(lm)))
  if (is.null(lm) && is.null(s$router)) lm <- default_model()
  if (is.null(lm)) return(NULL)
  s$lm <- lm
  r <- tryCatch(route(s), error = function(e) NULL)
  if (is.null(r)) { out <- list(); out["provider"] <- list(NULL); out$model <- model_string(lm); return(out) }
  list(provider = r$provider, model = r$wire)
}

# The lm15 settings set in any layer, as lm15's canonical JSON; NULL when none is.
setup_config <- function(s) {
  cfg <- config_of(s)
  if (is.null(cfg)) return(NULL)
  d <- plain_lm15(cfg)
  if (length(d)) d else NULL
}

# ---------------------------------------------------------------- tools' code

# A tool as a setup counts it: its code (as R reads a program's own: the
# function deparsed, without its source references), or the setup of the
# program it is.
tool_entry <- function(t) {
  p <- t$program
  if (inherits(p, "functai_fn") || (inherits(p, "functai_program") && is.null(program_core(p)$remote))) return(list(setup = ai_setup(p)))
  if (inherits(p, "functai_program")) return(list(described = TRUE))     # a program served elsewhere: its server's
  code <- list(language = "r", code = code_hash(t$fn))
  if (!is.null(t$code_id)) code$id <- t$code_id
  list(code = code)
}

# ---------------------------------------------------------------- reach (contract/calls.md, "Reach")

# What a program's code needs to reach: its own, its tools', and (a program)
# every AI function's and program's it reaches.
needs_of <- function(x, seen = character(0)) {
  core <- program_core_of(x)
  key <- paste(core$module, core$definition$name, sep = ":")
  if (key %in% seen) return(NULL)
  seen <- c(seen, key)
  if (is_program(x)) {
    parts <- program_values(core)
    return(do.call(merged_needs, c(list(core$needs), lapply(parts, function(p) needs_of(p, seen)))))
  }
  do.call(merged_needs, lapply(core$tools %||% list(), function(t)
    if (inherits(t$program, c("functai_fn", "functai_program"))) needs_of(t$program, seen) else t$needs))
}

wall_of <- function(s) { w <- s$wall; if (is.null(w)) NULL else if (is_str(w)) list(name = w) else w }

# "open" with no wall; else what the wall lets through (or the program's needs), and its limits.
reach_of <- function(program, s) {
  wall <- wall_of(s)
  if (is.null(wall)) return("open")
  needs <- needs_of(program) %||% list()
  pick <- function(k) as.list(sort_code_points(unique(as.character(unlist(if (!is.null(wall[[k]])) wall[[k]] else needs[[k]])))))
  out <- list(domains = pick("domains"), secrets = pick("secrets"))
  if (!is.null(wall$limits)) out$limits <- wall$limits
  out
}

lets_through <- function(granted, domain) any(vapply(as.character(unlist(granted)), function(g)
  identical(g, domain) || (startsWith(g, "*.") && endsWith(domain, substring(g, 2L))), NA))

reach_error <- function(message, missing) rlang::error_cnd(c("functai_reach_error", "functai_needs_unmet", "functai_refusal"), code = "needs-unmet",
                                                          missing = missing, functai_type = "ReachError", message = message)

# Refuse a program that needs what its wall does not let through (needs-unmet), before its first call.
check_reach <- function(program, s) {
  wall <- wall_of(s)
  if (is.null(wall) || (is.null(wall$domains) && is.null(wall$secrets))) return(invisible())
  needs <- needs_of(program) %||% list()
  missing <- list(domains = list(), secrets = list())
  if (!is.null(wall$domains)) missing$domains <- Filter(function(d) !lets_through(wall$domains, d), needs$domains %||% list())
  if (!is.null(wall$secrets)) missing$secrets <- Filter(function(x) !x %in% unlist(wall$secrets), needs$secrets %||% list())
  if (!length(missing$domains) && !length(missing$secrets)) return(invisible())
  what <- paste(c(unlist(missing$domains), paste("the secret", unlist(missing$secrets))[length(missing$secrets) > 0L]), collapse = ", ")
  stop(reach_error(sprintf("%s needs %s, which the wall \"%s\" does not let through", program_name(program), what, wall$name), missing))
}

# Where a call runs (its record's placement): the wall's name, and an AI function's server address.
placement_of <- function(s, ai) {
  out <- list()
  wall <- tryCatch(wall_of(s), error = function(e) NULL)
  if (!is.null(wall)) out$wall <- wall$name
  if (isTRUE(ai) && is_str(s$base_url)) out$base_url <- s$base_url
  if (length(out)) out else NULL
}

# The wall setting as data: a name, or list(name, domains?, secrets?, limits?).
check_wall <- function(w, call = NULL) {
  if (is.null(w) || (is_str(w) && nzchar(trimws(w)))) return(w)
  if (!is.list(w) || !is_str(w$name) || !nzchar(trimws(w$name)) || length(setdiff(names(w), c("name", "domains", "secrets", "limits"))))
    cli::cli_abort("{.arg wall} is the name of the wall programs run behind ({.val srt}), or {.code list(name = ..., domains = ..., secrets = ..., limits = ...)}: what it lets through, and its limits", call = call)
  out <- list(name = w$name)
  lets <- needs_value(w[intersect(names(w), c("domains", "secrets"))], "wall")
  for (k in c("domains", "secrets")) if (!is.null(w[[k]])) out[[k]] <- lets[[k]]
  if (!is.null(w$limits)) {
    if (!is.list(w$limits) || is.null(names(w$limits))) cli::cli_abort("{.arg wall}'s limits is a named list: memory, processor, seconds, as the host says them", call = call)
    out$limits <- plain_json(w$limits)
  }
  out
}

# ---------------------------------------------------------------- the document

plugin_entry <- function(p) {
  out <- list(name = p$name, version = p$version)
  if (!is.null(p$config)) out$config <- p$config
  out
}

ai_document <- function(fn) {
  core <- core_of(fn)
  s <- effective(core$own)
  doc <- list(version = version_of(core))
  doc["model"] <- list(setup_model(s$lm, s))
  cfg <- setup_config(s)
  if (!is.null(cfg)) doc$config <- cfg
  settings <- lapply(names(SETUP_SETTINGS), function(k) { v <- s[[k]]; if (is.null(v)) SETUP_SETTINGS[[k]] else if (is.numeric(v)) as.integer(v) else v })
  names(settings) <- names(SETUP_SETTINGS)
  if (!is.null(s$capabilities)) settings$capabilities <- plain_json(s$capabilities)
  doc$settings <- settings
  if (!is.null(s$escalate_to)) {
    to <- if (inherits(s$escalate_to, "functai_fn")) list(setup = ai_setup(s$escalate_to)) else setup_model(s$escalate_to, s)
    doc$escalate <- list(to = to, below = s$escalate_below %||% 0.9)
  }
  running <- plugins_around(core$own)
  if (length(running)) doc$plugins <- lapply(running, plugin_entry)
  if (length(core$tools)) doc$tools <- stats::setNames(lapply(core$tools, tool_entry), vapply(core$tools, function(t) t$name, ""))
  doc$reach <- reach_of(fn, s)
  doc
}

# The AI functions and programs a program's code names, as values.
program_values <- function(core) {
  if (is.null(core$body)) return(list())
  env <- environment(core$body) %||% globalenv()
  out <- list()
  for (n in unique(all.names(body(core$body)))) {
    v <- tryCatch(get(n, envir = env), error = function(e) NULL)
    if (inherits(v, "functai_fn") || (inherits(v, "functai_program") && !identical(program_core(v)$body, core$body))) out[[n]] <- v
  }
  out
}

# The AI functions a program reaches, by the keys its version uses (module:name).
reached_ai <- function(p, seen = character(0)) {
  core <- program_core(p)
  out <- list()
  for (n in names(program_values(core))) {
    v <- program_values(core)[[n]]
    if (inherits(v, "functai_fn")) out[[paste0(core_of(v)$module, ":", n)]] <- v
    else {
      k <- paste0(program_core(v)$module, ":", n)
      if (!k %in% seen) out <- c(out, reached_ai(v, c(seen, k)))
    }
  }
  out[!duplicated(names(out))]
}

module_document <- function(p) {
  core <- program_core(p)
  ai <- reached_ai(p)
  ai <- ai[sort_code_points(names(ai))]
  list(version = program_version(core), ai = if (length(ai)) lapply(ai, ai_setup) else lmcc::jobj(), reach = reach_of(p, effective(core$own)))
}

#' What a program runs on
#'
#' A version ([ai_version()]) says what a program is; its setup says what it
#' runs on (contract/calls.md, "Setups"): for an AI function, its version, the
#' model it would ask (as lm15 routes it), the lm15 settings set, FunctAI's
#' settings that change what the model is asked or what comes back, the
#' backup model, its plugins in the order they run (with their config), its
#' tools' code, and what it may reach; for a program, its version, the setup
#' of every AI function it reaches, and its reach. Worked out where it runs,
#' from the settings in force there, before its first call and without the
#' network. Every call records it (`program.setup`), so a score, a comparison
#' or a release belongs to exactly one thing that ran; ratings still pool by
#' interface.
#'
#' A tool's code counts in the setup, not the version, so a program loaded in
#' another language keeps its version: R reads a tool's function as written
#' (deparsed), never the functions it calls or what a closure captured;
#' `ai_tool(..., .code_id = )` says what identifies it beyond that.
#' `ai_setup_diff()` says what differs between two setups, step by step:
#' programs here, or saved folders ([write_ai()]).
#' @param fn An AI function or a program.
#' @param a,b Programs, or saved folders.
#' @return `ai_setup()`: `"sha256:"` and a hash; `ai_setup_document()`: the
#'   document, as a list; `ai_setup_diff()`: a tibble (`program`, `field`,
#'   `before`, `after`).
#' @examples
#' mood <- ai(mood ~ review, "How does the customer feel?", mood = choice("happy", "unhappy"))
#' with_ai_config(ai_setup_document(mood), lm = "openai:gpt-4.1-mini")
#' @export
ai_setup <- function(fn) lmcc::sha256_of(ai_setup_document(fn))

#' @rdname ai_setup
#' @export
ai_setup_document <- function(fn) {
  core <- program_core_of(fn)
  if (!is.null(core$remote)) cli::cli_abort("{core$definition$name} is served elsewhere: what it runs on is its server's")
  key <- paste(core$module, core$definition$name, sep = ":")
  working <- the$setup_working %||% character(0)
  if (key %in% working) return(list(cycle = core$definition$name))
  the$setup_working <- c(working, key)
  on.exit(the$setup_working <- working)
  if (is_program(fn)) module_document(fn) else ai_document(fn)
}

# A call record's program.setup: what it runs on, when it can be worked out (a setup never stops a call).
known_setup <- function(fn) tryCatch(ai_setup(fn), error = function(e) {
  warn_once(paste0("setup:", program_name(fn), conditionMessage(e)),
            sprintf("the setup of %s could not be worked out (%s); its records name none", program_name(fn), conditionMessage(e)))
  NULL
})

setup_documents <- function(source) {
  if (inherits(source, c("functai_fn", "functai_program"))) {
    out <- stats::setNames(list(ai_setup_document(source)), program_name(source))
    if (is_program(source)) { ai <- reached_ai(source); for (k in names(ai)) out[[k]] <- ai_setup_document(ai[[k]]) }
    return(out)
  }
  if (!is_str(source)) cli::cli_abort("{.fn ai_setup_diff} compares programs, or saved folders (their functai.json)")
  path <- if (dir.exists(source)) file.path(source, "functai.json") else source
  m <- lmcc::parse_json(paste(readLines(path, warn = FALSE, encoding = "UTF-8"), collapse = "\n"))
  out <- list()
  for (key in names(m$nodes)) {
    node <- m$nodes[[key]]
    if (is.null(node$setup)) next
    out[[if (identical(key, m$entry)) node$name else key]] <- node$setup
  }
  out
}

setup_differences <- function(a, b, path, out) {
  if (is.list(a) && !is.null(names(a)) && is.list(b) && !is.null(names(b))) {
    for (k in sort_code_points(union(names(a), names(b)))) out <- setup_differences(a[[k]], b[[k]], if (nzchar(path)) paste0(path, ".", k) else k, out)
    return(out)
  }
  if (!same_json(a, b)) out[[length(out) + 1L]] <- list(field = path, before = a, after = b)
  out
}

#' @rdname ai_setup
#' @export
ai_setup_diff <- function(a, b) {
  da <- setup_documents(a); db <- setup_documents(b)
  rows <- list()
  for (p in sort_code_points(union(names(da), names(db)))) {
    strip <- function(d) if (is.null(d)) lmcc::jobj() else d[setdiff(names(d), "ai")]
    for (x in setup_differences(strip(da[[p]]), strip(db[[p]]), "", list()))
      rows[[length(rows) + 1L]] <- list(program = p, field = x$field, before = list(x$before), after = list(x$after))
  }
  tibble::tibble(program = vapply(rows, function(r) r$program, ""), field = vapply(rows, function(r) r$field, ""),
                 before = lapply(rows, function(r) r$before[[1L]]), after = lapply(rows, function(r) r$after[[1L]]))
}
