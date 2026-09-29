# What a call saw (contract/calls.md, "Saw"), read from a log that holds
# other things too: only call records of a format this reader knows are read,
# and an entry it cannot read in full is not known.

saw_record <- function(id, saw = list(), ...) {
  rec <- list(functai_call = 2L, id = id, content = TRUE, saw = saw,
              sizes = list(inputs = list(x = 1L), outputs = list(result = 1L)), inputs = list(x = "a"), outputs = list(result = "b"),
              exchanges = list())
  extra <- list(...)
  for (k in names(extra)) rec[k] <- list(extra[[k]])
  rec
}

test_that("a record of a format the reader does not know is skipped whole, as the call or as what it saw", {
  future <- saw_record("future"); future$functai_call <- 99L
  expect_identical(read_saw(list(future), "future")$unknown$code, "not-recorded")
  seen <- saw_record("seen"); seen$functai_call <- 99L
  got <- read_saw(list(seen, saw_record("now", list(list(call = "seen")))), "now")
  expect_identical(got$keeps$refuses, "missing-call")
  got <- read_saw(list(seen, saw_record("now", list(list(saw_of = "seen")))), "now")
  expect_identical(got$unknown$code, "missing-call")
  rating <- list(functai_rating = 1L, id = "r1", call = "now", verdict = "right", saw = list())
  expect_identical(read_saw(list(rating), "r1")$unknown$code, "not-recorded")
  # the contract's own record, with only its format changed
  path <- list.files(file.path(contract_root(), "cases", "saw"), pattern = "^01-", full.names = TRUE)[[1L]]
  r <- read_json_file(path)$records[[1L]]
  r$functai_call <- 99L
  expect_identical(read_saw(list(r), r$id)$unknown$code, "not-recorded")
})

test_that("two different records with one id are not guessed between; the same record twice is one", {
  a <- saw_record("x")
  b <- saw_record("x", list(list(call = "y")))
  expect_identical(read_saw(list(a, b), "x")$unknown$code, "not-recorded")
  expect_identical(read_saw(list(a, a), "x")$keeps, list(ok = TRUE))
  got <- read_saw(list(a, b, saw_record("z", list(list(call = "x")))), "z")
  expect_identical(got$keeps$refuses, "missing-call")
})

test_that("an entry whose known keys hold values of another kind is not known", {
  old <- saw_record("old")
  for (entry in list(list(call = "old", steps = "yes"), list(call = "old", steps = FALSE), list(call = "old", slot = 5L),
                     list(call = "old", slot = "not a name"), list(call = "old", without = list()), list(call = "old", without = "x"),
                     list(call = 5L), list(call = "old", without = list("x", "x")))) {
    got <- read_saw(list(old, saw_record("new", list(entry))), "new")
    expect_identical(got$unknown$code, "unknown-key", info = plain(entry))
  }
  bad <- saw_record("new"); bad$saw <- "everything"
  expect_identical(read_saw(list(bad), "new")$unknown$code, "unknown-key")
  ok <- read_saw(list(old, saw_record("new", list(list(call = "old", slot = "turns", without = list("x"))))), "new")
  expect_identical(ok$keeps, list(ok = TRUE))
})

test_that("saw_of stands only as the first entry, and steps need every request hash", {
  got <- read_saw(list(saw_record("a"), saw_record("b"), saw_record("c", list(list(call = "a"), list(saw_of = "b")))), "c")
  expect_identical(got$unknown$code, "unknown-key")
  shown <- saw_record("s", exchanges = list(list(model = "m", finish = "stop", response = list())))
  got <- read_saw(list(shown, saw_record("t", list(list(call = "s", steps = TRUE)))), "t")
  expect_identical(got$keeps, list(refuses = "not-kept", call = "s"))
  shown$exchanges[[1L]]$request_hash <- "sha256:00"
  got <- read_saw(list(shown, saw_record("t", list(list(call = "s", steps = TRUE)))), "t")
  expect_identical(got$keeps, list(ok = TRUE))
})
