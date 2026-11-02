# Token counts (contract/calls.md, *Token counts*). lm15 2 gives each counter
# one meaning on every provider: input includes cached tokens, output includes
# reasoning. A sum over model calls keeps a counter only when every call
# reported it; counts lm15 1 wrote are shown only when they cannot mislead.

# Token counts summed over model calls (a list of named lists of integers): a
# counter is in the sum only when every call reported it, so a sum is exact
# or absent, never a part of the total. No calls: an empty JSON object.
sum_usage <- function(parts) {
  total <- NULL
  for (part in parts) {
    part <- part %||% list()
    if (is.null(total)) {
      total <- as.list(part)
    } else {
      keep <- intersect(names(total), names(part))
      total <- stats::setNames(lapply(keep, function(k) as.integer(total[[k]] + part[[k]])), keep)
    }
  }
  if (length(total)) total else lmcc::jobj()
}

# Token counts an earlier run kept, with the meaning lm15 2 gives them.
# `lm15`: the lm15 that wrote them (a call record's process$lm15), or NULL
# when the writer did not say (then lm15 1). lm15 1 kept each provider's own
# counting (Anthropic's input left the cache out, Gemini's and xAI's output
# the reasoning): its counts are kept only when total_tokens is input_tokens
# + output_tokens and no cache count is above 0; otherwise every count is
# unknown (an empty object).
usage_read <- function(usage, lm15 = NULL) {
  counts <- Filter(function(v) is.numeric(v) && !is.logical(v) && length(v) == 1L && !is.na(v) && v == round(v),
                   as.list(usage %||% list()))
  counts <- lapply(counts, as.integer)
  empty <- lmcc::jobj()
  major <- if (is.character(lm15) && length(lm15) == 1L) suppressWarnings(as.integer(strsplit(lm15, ".", fixed = TRUE)[[1L]][[1L]])) else NA
  if (!is.na(major) && major >= 2L) return(if (length(counts)) counts else empty)
  i <- counts$input_tokens; o <- counts$output_tokens; t <- counts$total_tokens
  if (is.null(i) || is.null(o) || is.null(t) || t != i + o) return(empty)
  if ((counts$cache_read_tokens %||% 0L) > 0L || (counts$cache_write_tokens %||% 0L) > 0L) return(empty)
  counts
}
