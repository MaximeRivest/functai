# Layouts: how a call is written as messages and how the reply is read back
# (contract/functions.md, "The layout"). A layout is an lmcc adapter; the
# three named ones are the contract's artifacts, carried as data.

registry <- function() {
  if (is.null(the$registry)) {
    reg <- lmcc::lmcc_registry()
    lmcc::install_std(reg)
    lmcc::register_reader(reg, "functai_reply", reply_reader, version = "1.0.0", exist_ok = TRUE)
    the$registry <- reg
  }
  the$registry
}

# The whole reply is the one output's value: a template with no output pattern.
reply_reader <- function(spec) {
  if (length(setdiff(names(spec), "kind")))
    stop(lmcc::refusal("entry-malformed", "reader: functai_reply takes only 'kind'", fix = list(action = "edit-entry", path = "reader")))
  lmcc::new_reader(
    spec = spec,
    split = function(text, names) {
      if (length(names) != 1L)
        stop(lmcc::refusal("parse-ambiguous", sprintf("the template has no output pattern, so the reply can hold one output, not %d", length(names))))
      out <- list(gsub("^[ \\t\\n\\r\\f\\x0b]+|[ \\t\\n\\r\\f\\x0b]+$", "", text, perl = TRUE))   # lmcc's strip: six ASCII spaces
      names(out) <- names[[1L]]; out
    },
    join = function(spelled) paste(vapply(spelled, function(x) x[[2L]], ""), collapse = "\n"))
}

NAMED <- c(xml = "xml", default = "xml", tags = "xml", chat = "chat", chatadapter = "chat", json = "json", jsonadapter = "json")

resolve_adapter <- function(adapter) {
  if (is.null(adapter)) adapter <- "xml"
  if (is.character(adapter)) {
    key <- gsub("[-_ ]", "", tolower(adapter))
    if (!key %in% names(NAMED)) cli::cli_abort("unknown adapter {.val {adapter}}: use \"xml\", \"chat\", \"json\", or template = list(...)")
    name <- NAMED[[key]]
    if (is.null(the$adapters[[name]])) the$adapters[[name]] <- lmcc::load_adapter(contract_data("layouts")[[name]], registry())
    return(the$adapters[[name]])
  }
  if (inherits(adapter, "lmcc_adapter")) return(adapter)
  lmcc::load_adapter(adapter, registry())
}

template_adapter <- function(messages, reader = "derived") {
  msgs <- lapply(messages, function(m) if (!is.null(m$content) && is.null(m$text)) list(role = m$role, text = m$content) else m)
  if (!any(vapply(msgs, function(m) !is.null(m$directive), NA))) {
    users <- which(vapply(msgs, function(m) identical(m$role, "user"), NA))
    at <- if (length(users)) max(users) else length(msgs) + 1L
    msgs <- append(msgs, list(list(directive = "turns")), after = at - 1L)
  }
  xml <- contract_data("layouts")$xml
  lmcc::load_adapter(list(name = "functai_template", versions = xml$versions, template = msgs,
                          reader = list(kind = reader), transports = xml$transports, formats = xml$formats), registry())
}

judgment_layout <- function(sig) {
  data <- lmcc::signature_to_list(sig)
  inputs <- Filter(function(f) f$direction == "input" && (f$purpose %||% "plain") == "plain", data$fields)
  outputs <- Filter(function(f) f$direction == "output", data$fields)
  doc <- data$instructions
  data$fields <- lapply(data$fields, function(f) {
    if (f$direction != "output") return(f)
    if (is.null(f$desc) || !nzchar(f$desc)) f$desc <- if (nzchar(doc)) (if (length(outputs) == 1L) doc else sprintf("%s (%s)", doc, f$name)) else NULL
    f
  })
  body <- if (length(inputs) == 1L) sprintf("{%s}", inputs[[1L]]$name) else "{% for f in inputs %}<{f.name}>\n{f.value}\n</{f.name}>\n{% endfor %}"
  a <- lmcc::load_adapter(list(name = "functai_judgment", versions = list(kernel = contract_data("layouts")$xml$versions$kernel, vocab = list(`reader/json_object` = "0.2.0", `format/json` = "0.1.0")),
                               template = list(list(directive = "turns"), list(role = "user", text = body)),
                               reader = list(kind = "json_object"), formats = list(`*` = list(use = "json"))), registry())
  list(adapter = a, signature = lmcc::signature_from_list(data))
}

# Bind a layout to a signature for a model's facts; every refusal fires here.
bind_layout <- function(adapter, template, sig, caps, provider) {
  if (!is.null(template)) {
    plan <- tryCatch(lmcc::lmcc_bind(template_adapter(template), sig, caps, registry()),
      lmcc_refusal = function(e) if (identical(e$code, "not-readable") && identical(e$fix$path, "template")) NULL else stop(e))
    if (!is.null(plan)) return(plan)
    visible <- Filter(function(f) f$direction == "output" && (f$purpose %||% "plain") == "plain", lmcc::signature_to_list(sig)$fields)
    if (length(visible) != 1L) stop(lmcc::refusal("not-readable", sprintf("the template has no output pattern, so the reply can only be one output, but this function has %d", length(visible)), fix = list(action = "edit-template", path = "template")))
    return(lmcc::lmcc_bind(template_adapter(template, "functai_reply"), sig, caps, registry()))
  }
  if (is.null(adapter) && provider %in% provider_sets()$judgment) {
    j <- judgment_layout(sig)
    return(lmcc::lmcc_bind(j$adapter, j$signature, caps, registry()))
  }
  lmcc::lmcc_bind(resolve_adapter(adapter), sig, caps, registry())
}
