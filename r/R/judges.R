# Checking a judge's evidence: a judge written as an ordinary AI function (a
# score with the quotes it rests on) is only as good as its quotes.

QUOTE_SAME <- c("\u2018" = "'", "\u2019" = "'", "\u201a" = "'", "\u201b" = "'", "\u2032" = "'", "`" = "'", "\u00b4" = "'",
                "\u201c" = '"', "\u201d" = '"', "\u201e" = '"', "\u201f" = '"', "\u2033" = '"', "\u00ab" = '"', "\u00bb" = '"',
                "\u2010" = "-", "\u2011" = "-", "\u2012" = "-", "\u2013" = "-", "\u2014" = "-", "\u2015" = "-", "\u2212" = "-",
                "\u2026" = "...", "\u00a0" = " ")
QUOTE_EDGES <- strsplit(" \t\n\"'.,;:!?\u2026\u201c\u201d\u2018\u2019\u00ab\u00bb()[]", "")[[1L]]

# Text as the check compares it: Unicode compatibility form, case folded,
# curly quotes straight, dashes one dash, white space one space.
quote_plain <- function(x) {
  x <- utf8::utf8_normalize(enc2utf8(as.character(x)), map_compat = TRUE)
  for (k in names(QUOTE_SAME)) x <- gsub(k, QUOTE_SAME[[k]], x, fixed = TRUE)
  casefold_full(trim_white(gsub("[[:space:]]+", " ", x, perl = TRUE)))
}

strip_edges <- function(x) {
  ch <- strsplit(x, "")[[1L]]
  while (length(ch) && ch[[1L]] %in% QUOTE_EDGES) ch <- ch[-1L]
  while (length(ch) && ch[[length(ch)]] %in% QUOTE_EDGES) ch <- ch[-length(ch)]
  paste(ch, collapse = "")
}

#' Whether a judge's quotes are really in the text
#'
#' White space, case, curly and straight quotes, dashes and a quote's own
#' quotation marks and final punctuation do not count; any other difference
#' does (a changed word, a paraphrase, an invented sentence). Deterministic,
#' and costs nothing: a check for a judge written as an ordinary AI function
#' that gives a score and the quotes it rests on.
#' @param text The source the quotes should come from.
#' @param quotes Quotes (a character vector, or a list column of them).
#' @return A logical vector, one per quote.
#' @examples
#' source <- "The parcel left Leeds on Monday. It was delayed by snow."
#' quotes_found(source, c("\u201cIt was delayed by snow\u201d", "It was lost"))
#' @export
quotes_found <- function(text, quotes) {
  haystack <- quote_plain(paste(text, collapse = "\n"))
  vapply(unlist(quotes), function(q) { needle <- strip_edges(quote_plain(q)); nzchar(needle) && grepl(needle, haystack, fixed = TRUE) }, NA, USE.NAMES = FALSE)
}
