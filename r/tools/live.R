# functai for R against real models (costs cents). Not run by the tests.
#   set -a; source ~/Projects/lm15-dev/.env; set +a
#   R_LIBS=r/.lib Rscript r/tools/live.R [model ...]
suppressPackageStartupMessages({ library(functai); library(dplyr) })
models <- commandArgs(TRUE)
if (!length(models)) models <- c("gpt-4.1-mini", "claude-haiku-4-5", "gemini:gemini-2.5-flash")
failed <- 0L
check <- function(what, expr) {
  t0 <- Sys.time()
  out <- tryCatch(expr, error = function(e) e)
  s <- sprintf("%.1f s", as.numeric(Sys.time() - t0, units = "secs"))
  if (inherits(out, "error")) { failed <<- failed + 1L; cat(sprintf("  FAIL  %s: %s\n", what, conditionMessage(out))) }
  else cat(sprintf("  ok    %s (%s): %s\n", what, s, paste(utils::capture.output(print(out))[1:min(4, length(utils::capture.output(print(out))))], collapse = " | ")))
}
team <- ai(team ~ message, "Which team should answer this customer message?",
           team = choice("shipping", "billing", "product", "account"), .temperature = 0)
for (lm in models) {
  cat(lm, "\n")
  local_ai_config(lm = lm)
  check("a choice", team("I was charged twice for order B-2210."))
  check("a column in mutate (20 rows)", tickets |> slice(1:20) |> mutate(team = team(message)) |> count(team))
  check("evaluate", evaluate(team, slice(tickets, 1:20), expected = category) |> tidy())
  check("a record, json layout", ai(person ~ text, "Who is described?",
    person = record(name = character(), age = integer(), city = character()), .adapter = "json")("Ana, 31, lives in Lyon."))
  check("reasoning first", ai(solve ~ problem, "Solve the word problem.", solve = double(), .module = "cot", .max_tokens = 4000L)("A pen costs 3 dollars. How much do 7 pens cost?"))
  orders <- c("A-1042" = "stuck at the carrier since Monday")
  lookup_order <- function(order) if (order %in% names(orders)) orders[[order]] else "unknown order"
  lookup <- ai_tool(lookup_order, "Look up where an order is.")
  check("a tool", ai(support ~ message, "Answer the customer, looking up their order.", .tools = list(lookup))("Where is my order A-1042?"))
}
quit(status = if (failed) 1L else 0L)
