#' @keywords internal
#' @importFrom stats predict update
"_PACKAGE"

#' Customer support messages, with the team each belongs to
#'
#' 80 messages to a small homeware shop. Each goes to one of four teams
#' (`category`: shipping, billing, product, account); the shop has house
#' rules about which one, and about how order numbers are written
#' (`order_id`). The same table as Python's `functai.datasets.tickets()`.
#' @format A tibble with `id`, `message`, `channel`, `category`, `order_id`.
"tickets"

#' Bird survey field notes, with species, count and behaviour
#'
#' 60 notes from a bird survey. The same table as Python's
#' `functai.datasets.field_notes()`.
#' @format A tibble with `id`, `site`, `date`, `note`, `species`, `count`, `behaviour`.
"field_notes"
