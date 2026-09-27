# The refunds dataset: 120 refund requests to the homeware shop of `tickets`,
# with the facts the shop's order system knows, what state the item is in, and
# the decision the shop's refund rules give. Written once, into
# python/functai/datasets/refunds.csv (which both packages ship):
#
#   cd r && set -a && . ../../lm15-dev/.env && set +a
#   R_LIBS=.lib Rscript data-raw/refunds.R      (then data-raw/datasets.R)
#
# The facts and states are drawn here, with a seed. A model writes each
# customer's message from its row (claude-sonnet-5; a model's writing is not
# repeatable, so the CSV, not this script, is the dataset). The decision is
# computed from the facts by the rules below, never by a model.

library(functai)

set.seed(20260927)
n <- 120
states <- c(unopened = 14, opened_unused = 16, used = 22, damaged = 20, wrong_item = 14, faulty = 34)
items <- c("kettle", "table lamp", "wool throw", "frying pan", "set of four mugs", "glass vase", "stand mixer",
           "linen duvet cover", "floor rug", "cast-iron casserole", "knife block", "desk chair", "bath towels",
           "coffee grinder", "ceramic planter", "bedside table", "toaster", "wall clock", "curtains", "dinner plates",
           "blender", "pillow pair", "salad bowl", "reading lamp", "shoe rack", "espresso machine", "doormat", "teapot")
state <- sample(rep(names(states), states))
# days since delivery: most requests come early; some sit near the 30- and 60-day lines, some late
days <- vapply(state, function(s) {
  r <- stats::runif(1)
  d <- if (r < 0.45) sample(1:28, 1) else if (r < 0.65) sample(26:36, 1) else if (r < 0.80) sample(52:68, 1)
       else if (r < 0.93) sample(70:300, 1) else sample(340:420, 1)
  as.integer(d)
}, 1L)
price <- round(exp(stats::runif(n, log(9), log(480))), 2)
final_sale <- stats::runif(n) < 0.18
item <- sample(items, n, replace = TRUE)

#' The shop's refund rules (as ?refunds states them)
decide <- function(state, days, final_sale) {
  ok <- ifelse(state %in% c("damaged", "wrong_item"), days <= 60,
        ifelse(state == "faulty", days <= 365,
        ifelse(state %in% c("unopened", "opened_unused"), days <= 30 & !final_sale,
        FALSE)))
  factor(ifelse(ok, "approve", "deny"), levels = c("approve", "deny"))
}

about <- c(
  unopened = "The item is still sealed in its box, never opened. They simply no longer want it (changed their mind, bought another, a gift they don't need...).",
  opened_unused = "They opened the packaging and looked at it, but never used it. They no longer want it (wrong size or colour for their room, changed their mind...). Nothing is wrong with it.",
  used = "They have used it for a while (days or weeks) and it works fine, but they no longer want it (don't like it, doesn't suit them, found better). Nothing is wrong with it.",
  damaged = "It arrived broken, cracked, dented or damaged in transit: the damage was there when they opened the parcel.",
  wrong_item = "The shop sent the wrong item (another product, colour or size than ordered), or part of the order is missing.",
  faulty = "It worked at first, then developed a fault in normal use (stopped working, broke, leaks, a part failed)."
)

tone <- sample(c("polite", "annoyed", "chatty", "terse", "anxious", "formal", "casual, lowercase, a typo or two", "blunt"), n, replace = TRUE)
length <- sample(c("one short line", "one or two sentences", "two or three sentences", "a short paragraph"), n, replace = TRUE,
                 prob = c(0.2, 0.3, 0.3, 0.2))
names_ <- c("Aiko", "Ben", "Carmen", "Dev", "Elif", "Farid", "Grace", "Hugo", "Ines", "Jonas", "Kofi", "Lena", "Mateo",
            "Nadia", "Omar", "Priya", "Quinn", "Rosa", "Sven", "Tariq", "Uma", "Viktor", "Wen", "Yusuf", "Zoe")
sign <- ifelse(stats::runif(n) < 0.5, sample(names_, n, replace = TRUE), "no signature")
subtle <- ifelse(stats::runif(n) < 0.3, "indirect: the state is clear only from details (what they did with it, what they noticed and when), never stated outright",
                 "plain: say what happened in their own words")

writer <- ai("refund_request",
  "Write the message a real customer of a small online homeware shop sends to ask for their money back.
Follow the tone, length, signature and way of telling given. Make the item's state clear to a careful reader
without naming it as a category. Avoid stock phrases (no 'let me know what you need from me', 'happy to send it back',
'nothing wrong with it at all'). Do not mention the shop's rules, the price, or a sale unless natural;
never contradict the facts given. Write only the message.",
  item = character(), state = described(character(), "what happened to the item: the customer's situation"),
  weeks_since_delivery = described(double(), "roughly how long ago it arrived, if they mention it at all"),
  tone = character(), length = character(), signature = described(character(), "a first name, or no signature"),
  telling = described(character(), "how directly they say what happened"),
  .returns = character(), .lm = "claude-sonnet-5", .concurrency = 8)

message <- writer(item, unname(about[state]), round(days / 7, 1), tone, length, sign, subtle)
stopifnot(!anyNA(message))

refunds <- tibble::tibble(
  id = seq_len(n), message = trimws(message), item = item, price = price,
  days_since_delivery = days, final_sale = final_sale,
  state = factor(state, levels = names(states)),
  decision = decide(state, days, final_sale))

utils::write.csv(refunds, file.path("..", "python", "functai", "datasets", "refunds.csv"), row.names = FALSE, fileEncoding = "UTF-8")
print(table(refunds$state, refunds$decision))
