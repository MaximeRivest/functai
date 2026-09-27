# Text rules the contract spells out (contract/scores.md): which characters
# are white space, and Unicode full case folding. R's trimws(), \\s and
# tolower() differ from both.

WHITE <- c(0x09:0x0d, 0x1c:0x20, 0x85, 0xa0, 0x1680, 0x2000:0x200a, 0x2028, 0x2029, 0x202f, 0x205f, 0x3000)

split_white <- function(text) {
  cps <- utf8ToInt(enc2utf8(text))
  if (!length(cps)) return(character(0))
  white <- cps %in% WHITE
  runs <- rle(white)
  ends <- cumsum(runs$lengths)
  starts <- ends - runs$lengths + 1L
  words <- character(0)
  for (i in seq_along(runs$values)) if (!runs$values[[i]]) words <- c(words, intToUtf8(cps[starts[[i]]:ends[[i]]]))
  words
}

trim_white <- function(text) {
  cps <- utf8ToInt(enc2utf8(text))
  keep <- which(!cps %in% WHITE)
  if (!length(keep)) return("")
  intToUtf8(cps[min(keep):max(keep)])
}

#' Unicode full case folding
#'
#' Folds text code point by code point with the contract's table (Unicode
#' 16.0, `CaseFolding.txt` statuses C and F): what Python's `str.casefold()`
#' does, so the German sharp s and `"SS"` fold alike.
#' @param text A character vector.
#' @return A character vector.
#' @export
casefold_full <- function(text) {
  folds <- contract_data("casefold")
  vapply(text, function(x) {
    if (is.na(x)) return(NA_character_)
    cps <- utf8ToInt(enc2utf8(x))
    paste0(vapply(cps, function(cp) {
      key <- toupper(sprintf("%04x", cp))
      folds[[key]] %||% intToUtf8(cp)
    }, ""), collapse = "")
  }, "", USE.NAMES = FALSE)
}

#' Text as exact_match compares it
#'
#' White space collapsed (the contract's white space, not R's) and case
#' folded.
#' @param text A character vector.
#' @return A character vector.
#' @export
normalize_text <- function(text) {
  vapply(text, function(x) if (is.na(x)) NA_character_ else casefold_full(paste(split_white(x), collapse = " ")), "",
         USE.NAMES = FALSE)
}
