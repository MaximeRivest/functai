# The contract's data this package carries (inst/contract, copied from the
# repository's contract/ by r/check): the layouts, the model table, case
# folding. tests/testthat/test-contract.R checks the copies are equal.

the <- new.env(parent = emptyenv())

# The layouts' text is cached and parsed on every read, so a caller can
# change what it gets without changing the contract's data. (Loaded
# adapters are cached apart, in `the$adapters`.)
read_contract <- function(path) {
  key <- paste0("text:", path)
  if (is.null(the[[key]])) {
    file <- system.file("contract", path, package = "functai", mustWork = TRUE)
    the[[key]] <- paste(readLines(file, warn = FALSE, encoding = "UTF-8"), collapse = "\n")
  }
  lmcc::parse_json(the[[key]])
}

contract_data <- function(what) {
  switch(what,
    layouts = list(xml = read_contract("layouts/xml.json"), chat = read_contract("layouts/chat.json"),
                   json = read_contract("layouts/json.json")),
    models = { if (is.null(the$models)) the$models <- read_contract("models.json"); the$models },
    casefold = { if (is.null(the$casefold)) the$casefold <- read_contract("unicode/casefold.json")$folds; the$casefold })
}

`%||%` <- function(x, y) if (is.null(x)) y else x
