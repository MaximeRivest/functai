# Writing a function: the formula, the codebook, types from a table. Offline.

test_that("the formula names the outputs and the inputs, in order", {
  f <- ai(decision ~ message + price, "Decide.")
  expect_identical(names(formals(f)), c("message", "price"))
  d <- core_of(f)$definition
  expect_identical(d$name, "decision")                             # one output: named after it
  expect_identical(names(d$outputs), "result")                     # and `result`, as in Python and TypeScript
  triage <- ai(summary + urgent ~ message, "Read it.", urgent = logical(), .name = "triage")
  expect_identical(names(core_of(triage)$definition$outputs), c("summary", "urgent"))
  expect_identical(core_of(triage)$definition$outputs$urgent$kind, "boolean")
})

test_that("the formula is checked, with the fix in the message", {
  expect_error(ai("team", "Which team?"), "starts with a formula")
  expect_error(ai(~ message, "Which team?"), "starts with a formula")
  expect_error(ai(team ~ log(message), "Which team?"), "reads the column itself")
  expect_error(ai(team ~ message * channel, "Which team?"), "interaction")
  expect_error(ai(team ~ message - 1, "Which team?"), "intercept")
  expect_error(ai(team ~ team, "Which team?"), "both sides")
  expect_error(ai(a + b ~ message, "Two."), "needs a name")
  expect_error(ai(team ~ message, "Which team?", mesage = "typo"), "not in the formula")
  expect_error(ai(team ~ message, "Which team?", team = c("billing", "shipping")), "choice\\(")
  expect_error(ai(team ~ message, "Which team?", .lmm = "x"), "unknown setting")
  expect_error(ai(team ~ message, "Which team?", .lm = "a", .lm = "b"), "given twice")
})

test_that("a sentence describes a text field; a type types it", {
  f <- ai(summary ~ message, "Summarise.", message = "the customer's own words", summary = described(character(), "one sentence"))
  expect_identical(core_of(f)$definition$inputs$message$desc, "the customer's own words")
  expect_identical(core_of(f)$definition$inputs$message$kind, "string")
  expect_match(ai_instructions(f), "- message: the customer's own words", fixed = TRUE)
  expect_match(ai_instructions(f), "- result: one sentence", fixed = TRUE)
})

test_that("types come from .data: a factor is a choice of its levels; . is every other column", {
  f <- ai(decision ~ message + price + days_since_delivery + final_sale, "Refund?", .data = refunds,
          price = "in dollars")
  d <- core_of(f)$definition
  expect_identical(vapply(d$inputs, function(x) x$kind, ""),
                   c(message = "string", price = "number", days_since_delivery = "integer", final_sale = "boolean"))
  expect_identical(d$inputs$price$desc, "in dollars")
  expect_identical(d$outputs$result$levels, levels(refunds$decision))
  all_but <- ai(decision ~ ., "Refund?", .data = refunds[c("message", "price", "decision")])
  expect_identical(names(formals(all_but)), c("message", "price"))
  expect_error(ai(decision ~ mesage, "Refund?", .data = refunds), "not a column")
  expect_error(ai(decision ~ ., "Refund?"), "give .*\\.data")
  here <- ai(decision ~ message, "Refund?", .data = refunds, decision = choice("approve", "deny", "review"))
  expect_identical(core_of(here)$definition$outputs$result$levels, c("approve", "deny", "review"))   # yours win
})

test_that("a function written from a formula is the one written from the contract", {
  a <- ai(mood ~ review, "How does the customer feel about what they bought?", mood = choice("happy", "unhappy", "mixed"))
  b <- ai(mood ~ review, "How does the customer feel about what they bought?",
          .data = tibble::tibble(review = character(), mood = factor(levels = c("happy", "unhappy", "mixed"))))
  expect_identical(ai_version(a), ai_version(b))
})

test_that("choice(): strings, vectors, a factor's levels; named levels say what they mean", {
  expect_identical(choice("a", "b")$levels, c("a", "b"))
  expect_identical(choice(c("a", "b"), "c")$levels, c("a", "b", "c"))
  expect_identical(choice(refunds$state)$levels, levels(refunds$state))
  s <- choice(approve = "the rules allow it", deny = "they do not")
  expect_identical(s$levels, c("approve", "deny"))
  expect_identical(field_desc(s), "approve: the rules allow it; deny: they do not")
  expect_identical(field_desc(described(s, "What to do.")), "What to do. approve: the rules allow it; deny: they do not")
  expect_identical(choice("keep", c(approve = "yes"))$levels, c("keep", "approve"))
  expect_error(choice(), "needs its answers")
  expect_error(choice("a", "a"), "twice")
  f <- ai(action ~ message, "Act.", action = s)
  expect_match(ai_instructions(f), "- result: approve: the rules allow it; deny: they do not", fixed = TRUE)
})

test_that("demos, evaluation and improvement read the answer from the formula's column", {
  r <- fake_router(responder = function(req, i) "<result>\nbilling\n</result>")
  team <- ai(category ~ message, "Which team?", category = choice("shipping", "billing"),
             .lm = "gpt-4.1-mini", .router = r, .log_calls = FALSE)
  rows <- tibble::tibble(message = c("Charged twice", "Parcel lost"), category = c("billing", "shipping"))
  expect_identical(evaluate(team, rows)$score, 0.5)                 # no `expected`: the formula said where
  taught <- labeled_few_shot(team, rows, k = 2L)
  expect_setequal(vapply(ai_demos(taught), function(d) d$outputs$result, ""), c("billing", "shipping"))
})

test_that("print reads like the definition: the formula, the words, a codebook", {
  f <- ai(state ~ message, "What state is the item in?", message = "the customer's own words",
          state = choice(unopened = "still sealed", faulty = "failed in normal use"), .lm = "gpt-4.1-mini")
  out <- paste(capture.output(print(f)), collapse = "\n")
  expect_match(out, "<ai function> state ~ message", fixed = TRUE)
  expect_match(out, "message  text  # the customer's own words", fixed = TRUE)
  expect_match(out, "unopened  still sealed", fixed = TRUE)
  t <- ai(summary + urgent ~ message, "Read it.", urgent = optional(logical()), .name = "triage")
  out <- paste(capture.output(print(t)), collapse = "\n")
  expect_match(out, "<ai function> triage: summary + urgent ~ message", fixed = TRUE)
  expect_match(out, "urgent   optional yes or no", fixed = TRUE)
})

test_that("ai_tool(): the function's arguments are its inputs, its name the tool's", {
  lookup_order <- function(order, verbose = FALSE) "stuck"
  t <- ai_tool(lookup_order, "Look up an order.", order = "a letter, a dash and four digits", verbose = logical())
  expect_identical(t$name, "lookup_order")
  expect_identical(t$parameters$properties$order$description, "a letter, a dash and four digits")
  expect_identical(t$parameters$properties$verbose$type, "boolean")
  expect_error(ai_tool(function(order) "x", "Look up."), "needs a name")
  expect_identical(ai_tool(function(order) "x", "Look up.", .name = "look")$name, "look")
  expect_error(ai_tool(lookup_order, "Look up.", ordr = "typo"), "not one of them")
})
