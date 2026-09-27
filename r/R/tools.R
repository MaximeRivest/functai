#' A tool the model may call
#'
#' An R function the model can ask for while it answers: FunctAI runs it and
#' sends the result back, until the model gives its answer. The tool's
#' inputs are the function's arguments; its name is the function's.
#' @param .fn The function. It is called with its arguments by name.
#' @param .description What it does, for the model.
#' @param ... Its arguments' types, as in [ai()]: nothing (text), a sentence
#'   (text, described) or a type.
#' @param .name The tool's name, as the model sees it. Default: the name
#'   `.fn` is given by (`ai_tool(lookup_order, ...)` is `lookup_order`).
#' @return A tool, for `ai(..., .tools = list(...))`.
#' @examples
#' orders <- c("A-1042" = "stuck at the carrier since Monday")
#' lookup_order <- function(order) if (order %in% names(orders)) orders[[order]] else "unknown order"
#' ai_tool(lookup_order, "Look up where an order is.", order = "a letter, a dash and four digits")
#' @export
ai_tool <- function(.fn, .description = "", ..., .name = NULL) {
  if (!is.function(.fn)) cli::cli_abort("{.arg .fn} is the R function the tool runs")
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
  structure(list(name = name, description = .description, parameters = params, fn = .fn), class = "functai_tool")
}

#' @export
print.functai_tool <- function(x, ...) {
  cat(sprintf("<ai tool> %s(%s): %s\n", x$name, paste(names(x$parameters$properties), collapse = ", "), x$description))
  invisible(x)
}
