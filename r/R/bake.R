# Baking (contract/baked.md): what a generative student is trained on and
# called with, so a model trained by any language, or by any trainer from
# examples R wrote, is called correctly by every language. Training itself is
# not here: R writes the examples every trainer reads, and runs the weights
# wherever they are served (vLLM, SGLang, TGI, llama.cpp); Python's
# functai.bake trains them, or any trainer does from the exported table.

BAKE_CAPABILITIES <- list(instruct = TRUE)
BAKED_FORMAT <- 2L
EXAMPLES_FORMAT <- 1L

bake_error <- function(code, message) rlang::error_cnd(c("functai_bake_error", paste0("functai_", gsub("-", "_", code)), "functai_refusal"),
                                                      code = code, message = message, functai_type = "BakeError")

value_hash <- function(v) lmcc::sha256_of(v)

# The function's full signature, as a student reads it before inputs are
# left out: its instruction and fields, reasoning only when the bake keeps it.
bake_signature <- function(core, cot) {
  s <- effective(core$own)
  lmcc::signature_from_list(list(instructions = instructions_of(core$definition, s$include_fn_name %||% TRUE, core$state$instructions),
                                 fields = fields_of(core$definition, cot, FALSE)))
}

reduced_signature <- function(sig, leave_out) {
  if (!length(leave_out)) return(sig)
  d <- lmcc::signature_to_list(sig)
  d$fields <- Filter(function(f) !(f$direction == "input" && f$name %in% leave_out), d$fields)
  lmcc::signature_from_list(d)
}

# The layout a student is trained and called with: its function's, with
# replies written from values.
student_layout <- function(core, layout = NULL) {
  s <- effective(core$own)
  adapter <- if (!is.null(layout)) { if (is.list(layout)) template_adapter(layout) else resolve_adapter(layout) }
    else if (!is.null(s$template)) template_adapter(s$template) else resolve_adapter(s$adapter %||% "xml")
  art <- lmcc::dump_adapter(adapter, registry())
  art$replay <- "values"
  art
}

new_entry <- function(name, fingerprint, signature, layout, outputs, reasoning, fixed, derived, capabilities = BAKE_CAPABILITIES)
  structure(list(name = name, fingerprint = fingerprint, signature = signature, layout = layout, outputs = outputs, reasoning = reasoning,
                 fixed = fixed, derived = derived, capabilities = capabilities), class = "functai_bake_entry")

left_out_of <- function(e) c(names(e$fixed), names(e$derived))

# What a student of `fn` reads (baked.md, "The examples"), its rows checked
# against `fixed` (one value in every row) and `derived` (decided by another input).
bake_entry <- function(fn, layout = NULL, reasoning = FALSE, fixed = list(), derived = list(), rows = list()) {
  core <- core_of(fn)
  if (length(core$tools)) stop(bake_error("bake-rows", sprintf("%s uses tools; training tool-calling students is not supported yet", core$definition$name)))
  cot <- isTRUE(reasoning) && identical(effective(core$own)$module, "cot")
  full <- bake_signature(core, cot)
  fields <- lmcc::signature_to_list(full)$fields
  names_ <- vapply(Filter(function(f) f$direction == "input" && (f$purpose %||% "plain") == "plain", fields), function(f) f$name, "")
  for (n in c(names(fixed), names(derived))) if (!n %in% names_)
    stop(bake_error("bake-rows", sprintf("%s has no input %s to leave out (its inputs: %s)", core$definition$name, n, paste(names_, collapse = ", "))))
  both <- intersect(names(fixed), names(derived))
  if (length(both)) stop(bake_error("bake-rows", sprintf("%s cannot be both fixed and derived", paste(both, collapse = ", "))))
  for (n in names(derived)) { src <- derived[[n]]
    if (!src %in% names_ || src %in% c(names(fixed), names(derived))) stop(bake_error("bake-rows", sprintf("derived %s = %s: %s must be another input the student still reads", n, src, src))) }
  prepared <- function(vals) prepare_inputs(full, vals)
  fixed_hashes <- lapply(stats::setNames(nm = names(fixed)), function(n) value_hash(prepared(stats::setNames(list(fixed[[n]]), n))[[n]]))
  for (i in seq_along(rows)) for (n in names(fixed_hashes)) if (n %in% names(rows[[i]]) &&
      value_hash(prepared(stats::setNames(list(rows[[i]][[n]]), n))[[n]]) != fixed_hashes[[n]])
    stop(bake_error("bake-rows", sprintf("fixed %s: row %d gives another value; a fixed input has one value in every row", n, i)))
  table <- list()
  for (n in names(derived)) {
    src <- derived[[n]]; values <- list()
    for (i in seq_along(rows)) {
      row <- rows[[i]]
      if (!all(c(n, src) %in% names(row))) stop(bake_error("bake-rows", sprintf("derived %s = %s: row %d lacks %s or %s", n, src, i, n, src)))
      p <- prepared(row[c(n, src)])
      k <- value_hash(p[[src]]); v <- value_hash(p[[n]])
      if (!is.null(values[[k]]) && !identical(values[[k]], v))
        stop(bake_error("bake-rows", sprintf("derived %s = %s: two rows with the same %s give %s different values, so %s does not decide it. Keep %s as an input", n, src, src, n, src, n)))
      values[[k]] <- v
    }
    table[[n]] <- list(from = src, values = if (length(values)) values else lmcc::jobj())
  }
  sig <- reduced_signature(full, c(names(fixed), names(derived)))
  outs <- vapply(Filter(function(f) f$direction == "output" && !identical(f$purpose, "tools.calls"), fields), function(f) f$name, "")
  new_entry(core$definition$name, lmcc::signature_fingerprint(full), sig, student_layout(core, layout), outs, cot,
            if (length(fixed_hashes)) fixed_hashes else lmcc::jobj(), if (length(table)) table else lmcc::jobj())
}

student_plan <- function(e) lmcc::lmcc_bind(lmcc::load_adapter(e$layout, registry()), e$signature, e$capabilities, registry())

# An lm15 request (canonical JSON) as chat-template messages: the system text
# first, a developer message as a system one, each message's text parts joined.
chat_messages <- function(request) {
  out <- list()
  system <- request$system
  if (!is.null(system) && length(system)) out[[1L]] <- list(role = "system", content = if (is_str(system)) system else paste(vapply(system, function(p) p$text, ""), collapse = ""))
  for (m in request$messages) {
    if (!m$role %in% c("user", "assistant", "system", "developer")) stop(bake_error("bake-rows", sprintf("a generative student reads text chats; the request has a %s message", m$role)))
    texts <- vapply(m$parts, function(p) {
      if (!identical(p$type, "text")) stop(bake_error("bake-rows", sprintf("a generative student reads text; the request carries a %s part", p$type %||% "?")))
      p$text
    }, "")
    out[[length(out) + 1L]] <- list(role = if (m$role == "developer") "system" else m$role, content = paste(texts, collapse = ""))
  }
  out
}

# One row's chat messages as the student sees them, and the reply the layout
# writes for its outputs (NULL without outputs).
student_messages <- function(e, inputs, outputs = NULL) {
  plan <- student_plan(e)
  values <- prepare_inputs(e$signature, inputs[!names(inputs) %in% left_out_of(e)])
  msgs <- chat_messages(lmcc::request_of(lmcc::render(plan, lmcc::new_turn(plan, values)), "student"))
  if (is.null(outputs)) return(list(messages = msgs, reply = NULL))
  example <- lmcc::example_turn(plan, values, lapply(outputs[intersect(e$outputs, names(outputs))], value_json))
  both <- chat_messages(lmcc::request_of(lmcc::render(plan, lmcc::new_turn(plan, values), list(example)), "student"))
  replies <- Filter(function(m) m$role == "assistant", both)
  list(messages = msgs, reply = replies[[length(replies)]]$content)
}

entry_meta <- function(e) list(name = e$name, fingerprint = e$fingerprint, signature = lmcc::signature_to_list(e$signature), layout = e$layout,
                               outputs = as.list(e$outputs), reasoning = e$reasoning, fixed = e$fixed, derived = e$derived, capabilities = e$capabilities)
entry_from_meta <- function(d) new_entry(d$name, d$fingerprint, lmcc::signature_from_list(d$signature), d$layout, as.character(unlist(d$outputs)),
                                         isTRUE(d$reasoning), d$fixed %||% lmcc::jobj(), d$derived %||% lmcc::jobj(), d$capabilities %||% BAKE_CAPABILITIES)

# Rows of a data frame (or a list of named lists) as named lists, the
# formula's output column read under its field's name.
bake_rows <- function(core, rows) {
  cols <- columns_of(core)
  if (is.data.frame(rows)) rows <- lapply(seq_len(nrow(rows)), function(i) lapply(rows, element, i = i))
  lapply(rows, function(r) { for (k in names(cols)) if (!k %in% names(r) && cols[[k]] %in% names(r)) r[k] <- list(r[[cols[[k]]]]); r })
}

#' The training conversations of an AI function
#'
#' `bake_examples()` turns rows with known answers into the conversations a
#' generative student learns (contract/baked.md, "The examples table"): one
#' row per conversation, `function`, `messages` (the prompt, then the reply:
#' the loss belongs on the reply only), `tag`, `weight`, `row_id`, `split`
#' (`"train"` or `"validation"`) and `source`. They are the messages the
#' student is called with, in every language. `export_examples()` writes them
#' as JSON lines, with `<path>.meta.json` beside it saying what made them, for
#' any trainer (TRL's `SFTTrainer`, Axolotl, Unsloth, a service's upload).
#' Python's `functai.bake` trains on them itself.
#' @param fn An AI function.
#' @param rows Rows with known answers: a data frame with its input and
#'   output columns (from [rated()], say).
#' @param fixed Inputs with one value in every row, left out of what the
#'   student reads: `list(guidance = "Keep every number.")`.
#' @param derived Inputs decided by another input the student still reads:
#'   `list(guidance = "section")`.
#' @param reasoning Whether the student learns the reasoning too (a function
#'   with `.module = "cot"`).
#' @param layout The layout it is trained with (default: the function's).
#' @param validation The share of rows held out for validation.
#' @param seed The seed choosing them.
#' @param weight,tag Columns of `rows` giving each row's weight and tag.
#' @return A tibble (`bake_examples()`), or the path (`export_examples()`).
#' @export
bake_examples <- function(fn, rows, fixed = list(), derived = list(), reasoning = FALSE, layout = NULL, validation = 0.1, seed = 0L,
                          weight = NULL, tag = NULL) {
  core <- core_of(fn)
  data <- bake_rows(core, rows)
  e <- bake_entry(fn, layout, reasoning, fixed, derived, data)
  n <- length(data)
  held <- if (n > 1L) withr::with_seed(seed, sample.int(n, min(n - 1L, round(n * validation)))) else integer(0)
  ins <- names(core$definition$inputs)
  out <- lapply(seq_len(n), function(i) {
    row <- data[[i]]
    inputs <- Filter(Negate(is_missing), row[intersect(ins, names(row))])
    for (k in names(fixed)) if (!k %in% names(inputs)) inputs[k] <- list(fixed[[k]])
    outs <- Filter(Negate(is_missing), row[intersect(e$outputs, names(row))])
    if (!length(outs)) stop(bake_error("bake-rows", sprintf("row %d has no answer for %s: a student learns from rows with known answers", i, paste(e$outputs, collapse = ", "))))
    got <- student_messages(e, lapply(inputs, value_json), outs)
    list(messages = c(got$messages, list(list(role = "assistant", content = got$reply))),
         tag = if (is.null(tag)) NA_character_ else as.character(row[[tag]] %||% NA), weight = if (is.null(weight)) 1 else as.numeric(row[[weight]]),
         row_id = i - 1L, split = if (i %in% held) "validation" else "train")
  })
  structure(tibble::tibble(`function` = rep(e$name, n), messages = lapply(out, function(o) o$messages), tag = vapply(out, function(o) o$tag, ""),
                           weight = vapply(out, function(o) o$weight, 0), row_id = vapply(out, function(o) o$row_id, 0L),
                           split = vapply(out, function(o) o$split, ""), source = rep("data", n)), entry = e)
}

#' @rdname bake_examples
#' @param path The JSON lines file to write.
#' @param ... Options of `bake_examples()`.
#' @export
export_examples <- function(path, fn, rows, ...) {
  table <- bake_examples(fn, rows, ...)
  e <- attr(table, "entry")
  dir.create(dirname(normalizePath(path, mustWork = FALSE)), recursive = TRUE, showWarnings = FALSE)
  lines <- vapply(seq_len(nrow(table)), function(i) {
    r <- list(`function` = table$`function`[[i]], messages = table$messages[[i]])
    r["tag"] <- list(if (is.na(table$tag[[i]])) NULL else table$tag[[i]])
    r$weight <- table$weight[[i]]; r$row_id <- table$row_id[[i]]; r$split <- table$split[[i]]; r$source <- table$source[[i]]
    lmcc::json_text(r)
  }, "")
  writeLines(lines, path, useBytes = TRUE)
  meta <- list(functai_examples = EXAMPLES_FORMAT); meta["student"] <- list(NULL); meta["template"] <- list(NULL); meta$functions <- list(entry_meta(e))
  writeLines(lmcc::json_text(meta), paste0(path, ".meta.json"), useBytes = TRUE)
  invisible(path)
}

# ---------------------------------------------------------------- a baked model, used

#' A model baked anywhere, served by an OpenAI-compatible server
#'
#' A folder baked by Python's `functai.bake` (or by a trainer that wrote
#' `baked.json` format 2), its weights served at `url` by an OpenAI-compatible
#' server (`vllm serve <folder>/model` serves `/v1/chat/completions`).
#' `update(fn, lm = baked(...))` runs `fn` on it, laid out exactly as it was
#' trained: the student's signature and layout, no worked examples, thinking
#' off. A call is refused when `fn` changed since it was baked
#' (`baked-changed`), or gives a fixed input another value (`baked-fixed`) or
#' a derived one a pair the student never saw (`baked-derived`).
#' @param folder The baked folder (it holds `baked.json`).
#' @param url The server's address (`http://localhost:8000/v1`).
#' @param model The model's name on the server (default: the folder's).
#' @param api_key The server's key, when it asks for one.
#' @param timeout Seconds to wait for an answer.
#' @return A baked model, for `lm`.
#' @export
baked <- function(folder, url, model = NULL, api_key = NULL, timeout = 120) {
  path <- file.path(folder, "baked.json")
  if (!file.exists(path)) stop(bake_error("baked-format", sprintf("%s has no baked.json", folder)))
  meta <- read_json_file_(path)
  if (!identical(as.integer(meta$functai_baked %||% 0L), BAKED_FORMAT))
    stop(bake_error("baked-format", sprintf("%s is baked.json format %s; this reader reads format %d (a format 1 folder: bake it again)", folder, short_json(meta$functai_baked), BAKED_FORMAT)))
  if (!identical(meta$kind, "generative")) stop(bake_error("baked-format", sprintf("%s is a %s model; R runs generative students (a head runs in Python)", folder, meta$kind %||% "?")))
  entries <- lapply(meta$functions, entry_from_meta); names(entries) <- vapply(entries, function(e) e$name, "")
  structure(list(folder = normalizePath(folder), meta = meta, entries = entries, url = sub("/+$", "", url),
                 model = model %||% meta$name %||% basename(normalizePath(folder)), api_key = api_key, timeout = timeout), class = "functai_baked")
}

#' @export
print.functai_baked <- function(x, ...) { cat(sprintf("<baked model %s at %s> %s\n", x$model, x$url, paste(names(x$entries), collapse = ", "))); invisible(x) }

read_json_file_ <- function(path) lmcc::parse_json(paste(readLines(path, warn = FALSE, encoding = "UTF-8"), collapse = "\n"))

entry_for <- function(b, core) {
  e <- b$entries[[core$definition$name]] %||% (if (length(b$entries) == 1L) b$entries[[1L]])
  if (is.null(e)) stop(bake_error("baked-changed", sprintf("this baked model answers %s, not %s", paste(names(b$entries), collapse = ", "), core$definition$name)))
  full <- bake_signature(core, e$reasoning && identical(effective(core$own)$module, "cot"))
  if (!identical(lmcc::signature_fingerprint(full), e$fingerprint))
    stop(bake_error("baked-changed", sprintf("%s has changed since it was baked (its inputs, outputs, types or instruction); bake it again", core$definition$name)))
  e
}

# The inputs the student reads, after checking fixed inputs have their baked
# values and derived ones follow their source.
student_inputs <- function(e, sig, inputs) {
  if (!length(left_out_of(e))) return(inputs)
  p <- prepare_inputs(sig, inputs)
  for (n in names(e$fixed)) if (n %in% names(p) && value_hash(p[[n]]) != e$fixed[[n]])
    stop(bake_error("baked-fixed", sprintf("%s: this baked model was trained with %s fixed to one value, and this call gives another. The student never learned to read %s: call it with the baked value, or bake again with this one", e$name, n, n)))
  for (n in names(e$derived)) {
    src <- e$derived[[n]]$from
    if (!src %in% names(p)) next
    want <- e$derived[[n]]$values[[value_hash(p[[src]])]]
    if (is.null(want)) stop(bake_error("baked-derived", sprintf("%s: this baked model never saw this %s in training, so it does not know the %s that goes with it", e$name, src, n)))
    if (n %in% names(p) && value_hash(p[[n]]) != want) stop(bake_error("baked-derived", sprintf("%s: %s is not the one the baked model learned for this %s; bake again with this pair", e$name, n, src)))
  }
  inputs[!names(inputs) %in% left_out_of(e)]
}

# A router for a baked student: the request's messages as chat messages, to
# an OpenAI-compatible chat server, thinking off.
baked_router <- function(b) {
  complete <- function(request) {
    d <- plain_lm15(request)
    gen <- b$meta$generation %||% list()
    body <- list(model = b$model, messages = chat_messages(d), temperature = 0, chat_template_kwargs = list(enable_thinking = FALSE))
    max_new <- d$config$max_tokens %||% gen$max_new_tokens
    if (!is.null(max_new)) body$max_tokens <- as.integer(max_new)
    h <- curl::new_handle(timeout = b$timeout, customrequest = "POST", postfields = lmcc::json_text(body))
    headers <- list(`Content-Type` = "application/json")
    if (!is.null(b$api_key)) headers$Authorization <- paste("Bearer", b$api_key)
    curl::handle_setheaders(h, .list = headers)
    resp <- curl::curl_fetch_memory(paste0(b$url, "/chat/completions"), handle = h)
    text <- rawToChar(resp$content); Encoding(text) <- "UTF-8"
    if (resp$status_code >= 400L) stop(sprintf("the baked model's server at %s answered %d: %s", b$url, resp$status_code, substr(text, 1L, 300L)), call. = FALSE)
    out <- lmcc::parse_json(text)
    choice <- out$choices[[1L]]
    u <- out$usage %||% list()
    lm15::response(out$model %||% b$model, lm15::message_assistant(list(lm15::text_part(choice$message$content %||% ""))),
                   if (identical(choice$finish_reason, "length")) "length" else "stop",
                   usage = lm15::usage(input_tokens = u$prompt_tokens, output_tokens = u$completion_tokens))
  }
  list(resolve = function(model) list(provider = "functai-baked-lm", model = model), complete = complete)
}
