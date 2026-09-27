#' @keywords internal
#' @importFrom stats predict update
"_PACKAGE"

#' Customer support messages, with the team each belongs to
#'
#' 80 messages to a small homeware shop. Each goes to one of four teams
#' (`category`); the shop has house rules about which one, and about how
#' order numbers are written (`order_id`). The same table as Python's
#' `functai.datasets.tickets()`.
#'
#' @section House rules: The labels follow them:
#' * Anything wrong with the delivery itself (late, lost, sent to the wrong
#'   place, the wrong item, something missing, or **broken when it
#'   arrived**) is **shipping**: the carrier pays.
#' * Anything about money (charges, invoices, coupons, cards, and **every
#'   request for money back**, whatever the reason) is **billing**.
#' * Problems that appear while using a product, and questions about
#'   products, are **product**.
#' * Signing in, passwords, profile details, personal data and emails from
#'   the shop are **account**.
#'
#' An order number is a letter, a dash and four digits (`"A-1042"`).
#' @format A tibble with 80 rows: `id`, `message` (what the customer wrote),
#'   `channel` (`"email"` or `"chat"`), `category` (the team: `"shipping"`,
#'   `"billing"`, `"product"` or `"account"`), `order_id` (`NA` when there is none).
"tickets"

#' Bird survey field notes, with species, count and behaviour
#'
#' 60 notes from a bird survey. The same table as Python's
#' `functai.datasets.field_notes()`.
#' @format A tibble with `id`, `site`, `date`, `note`, `species`, `count`, `behaviour`.
"field_notes"
