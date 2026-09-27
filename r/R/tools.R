#' A tool the model may call
#'
#' An R function the model can ask for while it answers: FunctAI runs it and
#' sends the result back, until the model gives its answer.
#' @param .fn The function. It is called with the inputs by name.
#' @param .name The tool's name, as the model sees it.
#' @param .description What it does.
#' @param ... Its inputs, as `name = prototype` (see [ai()]).
#' @return A tool, for `ai(..., .tools = list(...))`.
#' @examples
#' orders <- c("A-1042" = "stuck at the carrier since Monday")
#' lookup <- ai_tool(function(order) orders[[order]] %||% "unknown order",
#'   "lookup_order", "Look up where an order is.", order = character())
#' @export
ai_tool <- function(.fn, .name, .description = "", ...) {
  if (!is.function(.fn)) cli::cli_abort("{.arg .fn} is the R function the tool runs")
  inputs <- lapply(list(...), as_field)
  props <- lapply(inputs, function(f) if (is.null(f$desc)) f$shape else c(f$shape, list(description = f$desc)))
  # no additionalProperties: Gemini refuses the keyword in function declarations
  params <- list(type = "object", properties = if (length(props)) props else lmcc::jobj(), required = as.list(names(inputs)))
  structure(list(name = .name, description = .description, parameters = params, fn = .fn), class = "functai_tool")
}

#' @export
print.functai_tool <- function(x, ...) {
  cat(sprintf("<ai tool> %s(%s): %s\n", x$name, paste(names(x$parameters$properties), collapse = ", "), x$description))
  invisible(x)
}
