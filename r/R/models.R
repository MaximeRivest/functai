# Which model, how to reach it, and what it can do (contract/functions.md,
# "Capabilities"). lm15 routes a model string to a provider; what the model
# can do comes from the contract's table, never from guessing.

probe_capabilities <- function() {
  p <- contract_data("models")$probe
  p[names(p) != "about"]
}

provider_sets <- function() {
  m <- contract_data("models")
  speaks <- m$speaks_as[names(m$speaks_as) != "about"]
  list(native = unlist(m$native$providers), no_stop = unlist(m$native$no_stop_sequences),
       prefill = unlist(m$native$assistant_prefill), prefixes = m$native$reasoning_prefixes,
       chat = unlist(m$chat_completions$providers), hosts = unlist(m$native_tool_hosts$providers),
       judgment = unlist(m$judgment_only$providers), speaks = speaks)
}

#' What FunctAI declares a model can do
#'
#' From the contract's table, by the provider lm15 routes the model to.
#' @param provider The lm15 provider (`"openai"`, `"anthropic"`, ...).
#' @param model The model's name at that provider.
#' @return A named list of `TRUE`/`FALSE` facts.
#' @examples
#' model_capabilities("anthropic", "claude-haiku-4-5")
#' @export
model_capabilities <- function(provider, model) {
  s <- provider_sets()
  if (provider %in% s$judgment) return(list(native_structured_output = TRUE))
  if (!is.null(s$speaks[[provider]])) return(model_capabilities(s$speaks[[provider]], model))
  caps <- list(instruct = TRUE)
  if (provider %in% s$native) {
    caps$native_function_calling <- TRUE
    caps$native_structured_output <- TRUE
    caps$stop_sequences <- !provider %in% s$no_stop
    caps$native_reasoning <- any(startsWith(model, unlist(s$prefixes[[provider]]) %||% character(0)))
    caps$assistant_prefill <- provider %in% s$prefill && !caps$native_reasoning
  } else if (provider %in% s$chat) {
    caps[c("native_function_calling", "native_structured_output", "stop_sequences", "native_reasoning")] <- list(TRUE, TRUE, FALSE, FALSE)
  } else {
    caps$native_function_calling <- provider %in% s$hosts
    caps$stop_sequences <- FALSE
  }
  caps
}

call_capabilities <- function(provider, model, settings) {
  caps <- model_capabilities(provider, model)
  s <- provider_sets()
  anthropic <- provider == "anthropic" || identical(s$speaks[[provider]], "anthropic")
  t <- settings$temperature
  if (anthropic && isTRUE(caps$native_reasoning) && !is.null(t) && t != 1) {
    caps$native_reasoning <- FALSE
    caps$assistant_prefill <- TRUE
  }
  for (k in names(settings$capabilities)) caps[[k]] <- settings$capabilities[[k]]
  caps
}

# Settings a model refuses are left out of its requests, with one warning per
# provider, instead of failing every call of a function configured for another
# model (contract/functions.md, "Sampling a model does not take").
SAMPLING <- c("temperature", "top_p")

refused_settings <- function(provider, model) {
  if (identical(provider, "openai-codex")) return(c(SAMPLING, "max_tokens"))   # no knobs, no output cap
  m <- contract_data("models")
  speaks <- m$speaks_as[[provider]] %||% provider
  prefixes <- unlist(m$fixed_sampling[[speaks]]) %||% character(0)
  if (any(startsWith(model, prefixes))) SAMPLING else character(0)
}

adjust_settings <- function(s, provider, model) {
  refused <- function(k) {
    v <- s[[k]]
    !is.null(v) && (!k %in% SAMPLING || identical(provider, "openai-codex") || !isTRUE(all.equal(as.numeric(v), 1)))
  }
  drop <- Filter(refused, refused_settings(provider, model))
  if (!length(drop)) return(s)
  warn_once(paste0("refused:", provider, ":", paste(drop, collapse = ",")),
            sprintf("%s:%s does not take %s; left out of its requests", provider, model, paste(drop, collapse = ", ")))
  for (k in drop) s[k] <- list(NULL)
  s
}

ALIASES <- c(claude = "claude-code", chatgpt = "openai-codex", copilot = "github-copilot", kimi = "kimi-code")

model_string <- function(lm) {
  head <- sub(":.*$", "", lm)
  if (grepl(":", lm, fixed = TRUE) && tolower(head) %in% names(ALIASES))
    return(paste0(ALIASES[[tolower(head)]], substring(lm, nchar(head) + 1L)))
  lm
}

default_model <- function() {
  picks <- c(OPENAI_API_KEY = "gpt-4.1-mini", ANTHROPIC_API_KEY = "claude-haiku-4-5",
             GEMINI_API_KEY = "gemini:gemini-2.5-flash", GOOGLE_API_KEY = "gemini:gemini-2.5-flash",
             GROQ_API_KEY = "groq:openai/gpt-oss-120b", OPENROUTER_API_KEY = "openrouter:openai/gpt-4.1-mini")
  for (k in names(picks)) if (nzchar(Sys.getenv(k))) return(picks[[k]])
  NULL
}

default_router <- function() {
  if (is.null(the$router)) the$router <- lm15::new_router()
  the$router
}

# provider and model name for a model string, through the router
resolve_model <- function(router, model) {
  if (inherits(router, "lm15_router")) return(lm15::resolve(router, model))
  if (is.function(router$resolve)) return(router$resolve(model))
  head <- sub(":.*$", "", model)
  if (grepl(":", model, fixed = TRUE)) list(provider = head, model = substring(model, nchar(head) + 2L))
  else list(provider = router$provider %||% "openai", model = model)
}
