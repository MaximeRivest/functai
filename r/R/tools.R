#' A tool the model may call
#'
#' An R function the model can ask for while it answers: FunctAI runs it and
#' sends the result back, until the model gives its answer. The tool's
#' inputs are the function's arguments; its name is the function's.
#'
#' A tool returns text, a value (shown to the model as its JSON), a picture,
#' a sound, a video or a document (an lm15 part: `lm15::image_part(path =
#' "page.png")`), a list holding one (a picture with its caption), or
#' [ai_later()] when its result comes later. A part read from a file is given
#' its bytes, so the result is whole wherever it is read again.
#' @param .fn The function. It is called with its arguments by name.
#' @param .description What it does, for the model.
#' @param ... Its arguments' types, as in [ai()]: nothing (text), a sentence
#'   (text, described) or a type.
#' @param .name The tool's name, as the model sees it. Default: the name
#'   `.fn` is given by (`ai_tool(lookup_order, ...)` is `lookup_order`).
#' @param .effects What it does to the world: `"reads"` (it only looks: a
#'   search, a lookup) or `"changes"` (it writes, sends or pays). Left out, it
#'   is unknown, which every approval rule treats as `"changes"`: forgetting
#'   to declare is safe. An AI function given as a tool reads, unless one of
#'   its own tools changes things or declares nothing.
#' @param .needs What its code reaches: `list(domains = c("api.example.com",
#'   "*.example.com"), secrets = "EXAMPLE_API_KEY")`, host names and the
#'   names of the secrets it uses, never their values. Behind a wall that
#'   lets through less (the `wall` setting), the function that has it is
#'   refused before its first call (`needs-unmet`), and what it may reach
#'   counts in its setup ([ai_setup()]).
#' @param .notes The notes its code makes ([ai_note()]), by name: a type, or
#'   [note_type()] (`list(stage = note_type(choice("walking", "writing"),
#'   latest = TRUE))`). Every AI function that has the tool declares them,
#'   and each note is checked where it is made.
#' @param .code_id What identifies its code beyond the function's own text
#'   (a revision, a bundle's hash): a value a closure captured, or code the
#'   function calls, is not in its text.
#' @return A tool, for `ai(..., .tools = list(...))`.
#' @examples
#' orders <- c("A-1042" = "stuck at the carrier since Monday")
#' lookup_order <- function(order) if (order %in% names(orders)) orders[[order]] else "unknown order"
#' ai_tool(lookup_order, "Look up where an order is.", order = "a letter, a dash and four digits")
#' @export
ai_tool <- function(.fn, .description = "", ..., .name = NULL, .effects = NULL, .needs = NULL, .notes = NULL, .code_id = NULL) {
  if (!is.null(.effects) && !(is_str(.effects) && .effects %in% c("reads", "changes")))
    cli::cli_abort("{.arg .effects} is {.val reads} or {.val changes} (or left out: unknown, which counts as changes)")
  if (is.null(.effects) && inherits(.fn, "functai_fn"))
    .effects <- if (all(vapply(core_of(.fn)$tools, function(t) identical(t$effects, "reads"), NA))) "reads" else NULL
  if (!is.function(.fn)) cli::cli_abort("{.arg .fn} is the R function the tool runs")
  if (!is.null(.code_id) && !is_str(.code_id)) cli::cli_abort("{.arg .code_id} is one string: what identifies the tool's code")
  given <- substitute(.fn)
  name <- .name %||% if (is.symbol(given)) as.character(given) else
    cli::cli_abort(c("a tool written in place needs a name", i = "{.code ai_tool(function(order) ..., \"...\", .name = \"lookup_order\")}"))
  args <- setdiff(names(formals(.fn)), "...")
  specs <- list(...)
  stray <- setdiff(names(specs), args)
  if (length(specs) && (is.null(names(specs)) || any(!nzchar(names(specs))) || length(stray)))
    cli::cli_abort(c("{.fn ai_tool} types the function's arguments by name ({.field {args}})", x = if (length(stray)) "{.field {stray}} is not one of them"))
  inputs <- stats::setNames(lapply(args, function(a) as_field(specs[[a]] %||% character())), args)
  # no additionalProperties: Gemini refuses the keyword in function declarations
  params <- list(type = "object", properties = if (length(inputs)) lapply(inputs, described_shape) else lmcc::jobj(),
                 required = as.list(args))
  structure(list(name = name, description = .description, parameters = params, fn = .fn, effects = .effects,
                 needs = needs_value(.needs, sprintf("tool %s", name)), notes = notes_of(.notes, sprintf("tool %s", name)),
                 code_id = .code_id, program = if (inherits(.fn, c("functai_fn", "functai_program"))) .fn else NULL),
            class = "functai_tool")
}

# A domain code needs (contract/calls.md, "Reach"): a host name, or "*." and one (every name under it), in lower case.
DOMAIN_RE <- "^(\\*\\.)?[a-z0-9]([a-z0-9-]*[a-z0-9])?(\\.[a-z0-9]([a-z0-9-]*[a-z0-9])?)*$"

# What code needs to reach, as data (contract/tools.md, "What a tool needs"):
# list(domains, secrets), domains in lower case, each sorted and once each;
# NULL when nothing was said.
needs_value <- function(x, where = "needs") {
  if (is.null(x)) return(NULL)
  if (!is.list(x) || is.null(names(x)) || length(setdiff(names(x), c("domains", "secrets"))))
    cli::cli_abort("{where}: needs is {.code list(domains = c(...), secrets = c(...))}: the host names its code reaches and the names of the secrets it uses")
  out <- list()
  for (k in c("domains", "secrets")) {
    v <- x[[k]] %||% character(0)
    if (is.list(v)) v <- unlist(v)
    if (!is.character(v) || anyNA(v) || any(!nzchar(trimws(v)))) cli::cli_abort("{where}: needs${k} is a vector of names")
    v <- trimws(v)
    if (k == "domains") {
      v <- tolower(v)
      bad <- v[!grepl(DOMAIN_RE, v)]
      if (length(bad)) cli::cli_abort("{where}: {.val {bad[[1L]]}} is not a host name (api.example.com) or *. and one (*.example.com)")
    }
    out[[k]] <- as.list(sort_code_points(unique(v)))
  }
  out
}

# Several programs' needs as one (each sorted, once each); NULL when none says any.
merged_needs <- function(...) {
  said <- Filter(Negate(is.null), list(...))
  if (!length(said)) return(NULL)
  pick <- function(k) as.list(sort_code_points(unique(as.character(unlist(lapply(said, function(n) n[[k]]))))))
  list(domains = pick("domains"), secrets = pick("secrets"))
}

#' A result that comes later
#'
#' What a tool returns when its result comes later: it started something
#' that ends elsewhere (a job, a cloud browser, a person's reply) and
#' whoever learns the result delivers it under `key` (put an id in it:
#' `"harvest:8f2c"`).
#'
#' Only in a conversation's turn: the turn runs the step's other tool calls,
#' then stops, saved, and waits (a `functai_waiting` error whose `later`
#' lists the results awaited); [deliver()], from any process that opens the
#' conversation, gives the tool its result and the turn goes on. The tool
#' does not run again. Anywhere else nothing could wait for it, and the call
#' refuses `later-unavailable`. A result delivered before the turn waits for
#' it is used at once.
#' @param key Names the result in its conversation.
#' @return A later result, for a tool to return.
#' @examples
#' start_harvest <- function(url) ai_later(paste0("harvest:", nchar(url)))
#' start_harvest("a.com")
#' @export
ai_later <- function(key) {
  if (!is_str(key) || !nzchar(trimws(key))) cli::cli_abort("a later result's key is a non-empty text (put an id in it: {.val harvest:8f2c})")
  structure(list(key = key), class = "functai_later")
}

#' @export
print.functai_later <- function(x, ...) { cat(sprintf("<later result> %s\n", x$key)); invisible(x) }

# ---------------------------------------------------------------- what a tool returns (contract/tools.md)

RESULT_PARTS <- c("text", "image", "audio", "video", "document")
is_part <- function(x) inherits(x, "lm15_Part")

# An lm15 part as its JSON form; a part read from a file holds the file's bytes.
part_json <- function(p) {
  d <- plain_lm15(p)
  if (!d$type %in% RESULT_PARTS) cli::cli_abort("a tool's result holds text, pictures, sounds, videos and documents; not a {d$type} part")
  if (!is.null(d$path) && is.null(d$data)) {
    d$data <- openssl::base64_encode(lm15::media_bytes(p))           # one line: base64 that wraps is refused by providers
    d$path <- NULL
  }
  d
}

as_result_text <- function(v) if (is.character(v) && length(v) == 1L && !is.na(v)) v else lmcc::json_text(plain_json(v))

# What a tool returned, as the model is shown it (contract/tools.md, "What a
# tool returns"): text (a text; any other value, as JSON), or, when it returned
# a part or a list holding one, a list of lm15 parts in their JSON form.
result_of <- function(out) {
  if (is_part(out)) return(list(part_json(out)))
  if (is.list(out) && !is.data.frame(out) && is.null(names(out)) && any(vapply(out, is_part, NA)))
    return(lapply(out, function(x) if (is_part(x)) part_json(x) else list(type = "text", text = as_result_text(x))))
  as_result_text(out)
}

# A tool result's text: itself, or its text parts joined by a new line.
result_text <- function(output) {
  if (is_str(output)) return(output)
  paste(vapply(Filter(function(p) identical(p$type, "text"), output), function(p) p$text %||% "", ""), collapse = "\n")
}

# A tool result given as data (a plugin's change, a result given for a tool
# that may have run, a result delivered later): text, or a list of lm15 parts.
check_result <- function(output, where) {
  if (is_str(output)) return(output)
  if (is_part(output)) return(list(part_json(output)))
  if (is.list(output) && length(output) && !is.data.frame(output) && (is.null(names(output)) || inherits(output, "lm15_Part")))
    return(lapply(output, function(p) {
      if (is_part(p)) part_json(p)
      else if (is.list(p) && is_str(p$type) && p$type %in% RESULT_PARTS) p
      else if (is_str(p)) list(type = "text", text = p)
      else cli::cli_abort("{where}: a part of a tool's result is text, a picture, a sound, a video or a document (an lm15 part)")
    }))
  cli::cli_abort("{where} is text, or a list of lm15 parts (text, pictures, sounds, videos, documents)")
}

#' @export
print.functai_tool <- function(x, ...) {
  cat(sprintf("<ai tool> %s(%s): %s%s\n", x$name, paste(names(x$parameters$properties), collapse = ", "), x$description,
              if (is.null(x$effects)) "" else sprintf(" [%s]", x$effects)))
  invisible(x)
}
