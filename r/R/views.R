# Views (contract/streaming.md, "Views"): what one kind of reader may see of
# a call tree's log. `full` (every event and value, only in the process
# running the tree), `kept` (what log_content lets be kept: a store's form),
# and `outside`: a caller who sees only the program's boundary (a served
# program's customer): its `started` (its program without file and line), its
# answer's text as it is written, the approvals addressed to it, its `done`,
# or its `failed` with the error's type and code. Never a helper's answer, a
# tool call or result, a thinking, or why a request was retried.

OUTSIDE_PROGRAM_KEYS <- c("name", "kind", "module", "version", "signature", "interface", "answer")

new_view <- function(name = c("outside", "full", "kept"), answer_from = NULL) {
  v <- new.env(parent = emptyenv())
  v$name <- match.arg(name)
  v$answer_from <- if (is.null(answer_from)) NULL else if (is.character(answer_from)) answer_from
    else if (inherits(answer_from, "functai_fn")) core_of(answer_from)$definition$name else as.character(answer_from)
  v$root <- NULL; v$root_function <- NULL; v$root_kind <- "ai"; v$answer <- "result"
  v$forwarded <- list(); v$requests <- 0L; v$last <- NULL; v$caller_asked <- character(0)
  v
}

view_link <- function(v, e) { out <- relinked(e, v$last); v$last <- event_position(out); out }
with_kind <- function(e, kind, call, fn, data) new_event(kind, e$tree, e$writer, e$seq, e$after, e$at, call, fn, data)

# The event as the view shows it, or NULL.
view_event <- function(v, e) {
  if (v$name %in% c("full", "kept")) return(view_link(v, e))
  if (is.null(v$root)) {
    if (e$kind != "started") return(NULL)
    v$root <- e$call; v$root_function <- e[["function"]]
    v$root_kind <- e$program$kind %||% "ai"; v$answer <- e$program$answer %||% "result"
  }
  if (identical(e$call, v$root)) view_root(v, e) else view_inside(v, e)
}

view_root <- function(v, e) {
  data <- event_data(e)
  switch(e$kind,
    started = { data$program <- (data$program %||% list())[names(data$program %||% list()) %in% OUTSIDE_PROGRAM_KEYS]; data$invocation <- NULL
                view_link(v, with_data(e, data)) },
    request = if (identical(v$root_kind, "ai")) { v$requests <- max(v$requests, as.integer(data$request %||% 0L)); view_link(v, e) },
    retry = if (identical(v$root_kind, "ai")) { data$reason <- NULL; data$content <- FALSE; view_link(v, with_data(e, data)) },
    text = if (isTRUE(data$answer)) view_link(v, e),
    approval = , approved = view_approval(v, e),
    done = view_link(v, e),
    failed = { err <- data$error %||% list(); data$error <- err[names(err) %in% c("type", "code")]; data$content <- FALSE
               view_link(v, with_data(e, data)) },
    NULL)
}

view_approval <- function(v, e) {
  key <- paste(e$call, e$invocation %||% "")
  if (e$kind == "approval") { if (!identical(e$to, "caller")) return(NULL); v$caller_asked <- c(v$caller_asked, key) }
  else if (!key %in% v$caller_asked) return(NULL)
  view_link(v, with_kind(e, e$kind, v$root, v$root_function, event_data(e)))
}

view_inside <- function(v, e) {
  if (e$kind %in% c("approval", "approved")) return(view_approval(v, e))
  if (identical(v$root_kind, "ai") || is.null(v$answer_from)) return(NULL)
  if (e$kind == "started") {
    v$forwarded[[e$call]] <- identical(e[["function"]], v$answer_from)
    return(if (v$forwarded[[e$call]]) view_as_request(v, e))
  }
  if (!isTRUE(v$forwarded[[e$call]])) return(NULL)
  if (e$kind %in% c("request", "retry")) return(view_as_request(v, e))
  if (e$kind == "text" && isTRUE(e$answer)) {
    data <- event_data(e); data$field <- v$answer
    return(view_link(v, with_kind(e, "text", v$root, v$root_function, data)))
  }
  NULL
}

view_as_request <- function(v, e) {
  v$requests <- v$requests + 1L
  data <- list(request = v$requests); data["model"] <- list(NULL)
  view_link(v, with_kind(e, "request", v$root, v$root_function, data))
}

#' What a caller outside a program sees of its log
#'
#' A whole log's events as the outside view shows them (contract/
#' streaming.md, "Views"): what a served program's caller may see.
#' @param events A call tree's events (its whole log, or its kept form).
#' @param answer_from The AI function (or its name) whose answer is a
#'   program's answer, shown as the program's as it is written.
#' @return The events of the view.
#' @export
outside_view <- function(events, answer_from = NULL) {
  v <- new_view("outside", answer_from)
  Filter(Negate(is.null), lapply(events, function(e) view_event(v, as_event(e))))
}
