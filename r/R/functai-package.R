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

#' Refund requests, with the decision the shop's rules give
#'
#' 120 refund requests to the homeware shop of [tickets]: what the customer
#' wrote, what the order system knows, the item's state, and the decision.
#' The same table as Python's `functai.datasets.refunds()`.
#'
#' @section Refund rules: `decision` follows them exactly:
#' * Damaged on arrival, or the wrong item (or part of the order missing):
#'   a refund within **60 days** of delivery, final sale or not.
#' * Faulty (it failed in normal use): a refund within **365 days**, final
#'   sale or not.
#' * Unopened, or opened but not used, and no longer wanted: a refund within
#'   **30 days**, and **never for a final-sale item**.
#' * Used and no longer wanted: **no refund**.
#'
#' The messages were written by a language model from each row's facts, in
#' varied tones and lengths; some say what happened only indirectly. The
#' facts, and so the decisions, were drawn first (`data-raw/refunds.R`).
#' @format A tibble with 120 rows: `id`, `message` (what the customer wrote),
#'   `item`, `price` (dollars), `days_since_delivery` (from the order system),
#'   `final_sale` (logical), `state` (a factor: `unopened`, `opened_unused`,
#'   `used`, `damaged`, `wrong_item`, `faulty`) and `decision` (a factor:
#'   `approve`, `deny`).
"refunds"
