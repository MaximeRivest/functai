# Serving a program over HTTP (contract/serving.md), and using one served
# elsewhere (ai_remote()). What a caller sees is the program's boundary (the
# outside view): its answer, its answer's text, approvals addressed to it,
# and its end; never a helper's answer, a tool's input or output, a
# thinking, or an error's message. The owner watches everything in their log.

SERVE_FORMAT <- 1L
MAX_BODY <- 16 * 1024^2
LOCAL_HOSTS <- c("127.0.0.1", "::1", "localhost")
UUID_TEXT <- "^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"

serve_error <- function(code, message) rlang::error_cnd(c("functai_serve_error", paste0("functai_", gsub("-", "_", code)), "functai_refusal"),
                                                       code = code, message = message, functai_type = "ServeError")

json_reply <- function(status, data) list(status = as.integer(status), headers = list("Content-Type" = "application/json"), body = lmcc::json_text(data))

error_reply <- function(status, err, message = FALSE) {
  out <- list(type = error_json(err)$type)
  if (is_str(err$code)) out$code <- err$code
  if (is_str(err$field)) out$field <- err$field
  if (message) out$message <- conditionMessage(err)
  json_reply(status, list(error = out))
}

serve_keys <- function(keys) {
  if (is.null(keys)) return(character(0))
  if (is_str(keys) && file.exists(path.expand(keys))) {
    l <- trimws(readLines(path.expand(keys), warn = FALSE))
    return(l[nzchar(l) & !startsWith(l, "#")])
  }
  as.character(keys)
}

# Two keys compared in time that does not depend on where they differ.
same_secret <- function(a, b) {
  x <- charToRaw(enc2utf8(a)); y <- charToRaw(enc2utf8(b))
  n <- max(length(x), length(y))
  x <- c(x, raw(n - length(x))); y <- c(y, raw(n - length(y)))
  length(charToRaw(enc2utf8(a))) == length(charToRaw(enc2utf8(b))) && all(xor(x, y) == as.raw(0))
}

position_id <- function(e) sprintf("%d-%d", e$writer, e$seq)
parse_position <- function(text) {
  m <- if (is_str(text)) regmatches(text, regexec("^\\s*([0-9]+)-([0-9]+)\\s*$", text))[[1L]] else character(0)
  if (length(m)) position(as.integer(m[[2L]]), as.integer(m[[3L]])) else NULL
}
sse_text <- function(events) paste0(vapply(events, function(e) sprintf("id: %s\nevent: %s\ndata: %s\n\n", position_id(e), e$kind, lmcc::json_text(unclass(e))), ""), collapse = "")

#' Serve a program over HTTP
#'
#' A program as an HTTP service (contract/serving.md), to callers who see
#' only its boundary: its interface, calls, streams and conversations.
#' `ai_service()` makes the service (`serve_request(service, method, path,
#' headers, body)` answers one request, for testing or another server);
#' `ai_serve()` runs it on httpuv.
#'
#' | route | answer |
#' |---|---|
#' | `GET /interface` | the program, described (`{"functai_interface": 1, ...}`) |
#' | `GET /openapi.json` | the same, as OpenAPI 3.1; `GET /`: a form |
#' | `POST /call` `{"inputs": ...}` | `{"call", "outputs", "value"}`; a refused input `422` |
#' | `POST /stream` | Server-Sent Events of the outside view |
#' | `POST /conversations/<id>/turns` | a turn, saved before it runs (`after`, `request_id`, `wait`) |
#' | `GET /conversations/<id>/turns[/<turn>[/events]]` | the branch's turns, one turn, its events (`Last-Event-ID` resumes) |
#' | `POST .../turns/<turn>/stop`, `.../approvals/<invocation>` | stop it; answer an approval (`approvals = "caller"`) |
#'
#' Without `keys` it listens only on this machine (`serve-keys` otherwise).
#' A caller that is a functai call sends `FunctAI-Parent`: the served call
#' names it as its parent, so the two logs make one call tree.
#'
#' R answers one request at a time, as R runs one thing at a time: a long
#' call holds the others until it ends, and a stream's events are sent
#' together when its call ends (httpuv cannot send a response piece by
#' piece). A served program that must stream live, or take many callers at
#' once, is served by Python's or Julia's functai from the same saved folder.
#' @param program An AI function, a program, or a saved folder.
#' @param keys Bearer keys a caller must send: a vector, or a file of one per line.
#' @param store Where conversations are kept (as [ai_conversation()]'s).
#' @param lm The model every call uses.
#' @param approvals Who answers a tool call the program's `approve` rule asks
#'   about: `"owner"` (in their own session; the caller sees the turn
#'   waiting), or `"caller"`, through the approvals route.
#' @param host,port Where it listens.
#' @param block Whether to serve until interrupted (`FALSE`: returns the
#'   server; httpuv answers whenever R is idle; [httpuv::stopServer()] stops it).
#' @return `ai_service()`: a service; `ai_serve()`: the server, invisibly.
#' @export
ai_service <- function(program, keys = NULL, store = NULL, lm = NULL, approvals = c("owner", "caller")) {
  if (is_str(program)) program <- read_ai(program)
  if (!inherits(program, c("functai_fn", "functai_program"))) cli::cli_abort("serve an AI function, a program, or a saved folder")
  approvals <- match.arg(approvals)
  iface <- program_iface(program)
  opaque <- vapply(c(iface$inputs, iface$outputs), function(x) if (isTRUE(x$opaque)) x$name else NA_character_, "")
  opaque <- opaque[!is.na(opaque)]
  if (length(opaque)) stop(serve_error("serve-opaque", sprintf("%s cannot be served: %s may hold values with no JSON form, and only JSON crosses HTTP. Give %s a type",
                                                              program_name(program), paste(opaque, collapse = ", "), if (length(opaque) > 1L) "them" else "it")))
  if (!is.null(lm) && inherits(program, "functai_fn")) program <- update(program, lm = lm)
  structure(list(program = program, keys = serve_keys(keys), store = store, lm = lm, approvals = approvals), class = "functai_service")
}

#' @export
print.functai_service <- function(x, ...) { cat(sprintf("<service> %s%s\n", program_name(x$program), if (length(x$keys)) sprintf(", %d keys", length(x$keys)) else "")); invisible(x) }

#' @rdname ai_service
#' @export
ai_serve <- function(program, host = "127.0.0.1", port = 8080L, keys = NULL, store = NULL, lm = NULL, approvals = c("owner", "caller"), block = TRUE) {
  rlang::check_installed("httpuv", reason = "to serve a program over HTTP")
  s <- if (inherits(program, "functai_service")) program else ai_service(program, keys, store, lm, match.arg(approvals))
  if (!length(s$keys) && !host %in% LOCAL_HOSTS)
    stop(serve_error("serve-keys", sprintf("serving on %s lets anyone on the network call %s and spend your model budget: give keys (keys = \"keys.txt\"), or serve on 127.0.0.1", host, program_name(s$program))))
  app <- list(call = function(req) {
    body <- if (!is.null(req$rook.input)) req$rook.input$read() else raw()
    if (length(body) > MAX_BODY) return(list(status = 413L, headers = list(), body = ""))
    headers <- list()
    for (k in grep("^HTTP_", names(req), value = TRUE)) headers[[tolower(gsub("_", "-", sub("^HTTP_", "", k)))]] <- req[[k]]
    r <- serve_request(s, req$REQUEST_METHOD, paste0(req$PATH_INFO, if (nzchar(req$QUERY_STRING %||% "")) paste0("?", req$QUERY_STRING) else ""), headers, body)
    r$headers <- lapply(r$headers, as.character)
    r
  })
  server <- httpuv::startServer(host, as.integer(port), app)
  if (!block) return(invisible(server))
  on.exit(httpuv::stopServer(server))
  cli::cli_inform("serving {.fn {program_name(s$program)}} at {.url http://{host}:{port}} (interrupt to stop)")
  tryCatch(repeat httpuv::service(100), interrupt = function(e) NULL)
  invisible(server)
}

serve_describe <- function(s) {
  info <- program_info(s$program)
  out <- list(functai_interface = SERVE_FORMAT, name = program_name(s$program), kind = info$kind, version = info$version,
              interface = program_iface(s$program), answer = info$answer)
  out["model"] <- list(s$lm); out$approvals <- s$approvals
  out
}

serve_openapi <- function(s) {
  iface <- program_iface(s$program)
  props <- function(fs) { p <- lapply(fs, function(x) x$shape); names(p) <- vapply(fs, function(x) x$name, ""); if (length(p)) p else lmcc::jobj() }
  ins <- list(type = "object", properties = props(iface$inputs), required = lapply(Filter(function(x) !isTRUE(x$optional), iface$inputs), function(x) x$name))
  outs <- list(type = "object", properties = props(iface$outputs))
  schema <- list(type = "object", properties = list(inputs = ins), required = list("inputs"))
  body <- list(required = TRUE, content = list(`application/json` = list(schema = schema)))
  answer <- list(type = "object", properties = list(call = list(type = "string"), outputs = outs, value = lmcc::jobj()))
  list(openapi = "3.1.0", info = list(title = program_name(s$program), version = serve_describe(s)$version, description = iface$description),
       paths = list(`/call` = list(post = list(requestBody = body, responses = list(`200` = list(description = "the answer", content = list(`application/json` = list(schema = answer)))))),
                    `/stream` = list(post = list(requestBody = body, responses = list(`200` = list(description = "Server-Sent Events", content = list(`text/event-stream` = lmcc::jobj())))))),
       components = list(securitySchemes = list(key = list(type = "http", scheme = "bearer"))),
       security = if (length(s$keys)) list(list(key = list())) else list())
}

html_escape <- function(t) { t <- gsub("&", "&amp;", t, fixed = TRUE); t <- gsub("<", "&lt;", t, fixed = TRUE); t <- gsub(">", "&gt;", t, fixed = TRUE); gsub("\"", "&quot;", t, fixed = TRUE) }

serve_form <- function(s) {
  iface <- program_iface(s$program); name <- html_escape(program_name(s$program))
  rows <- paste(vapply(iface$inputs, function(x) sprintf('<label>%s<br><textarea name="%s" rows="3"></textarea></label><br>', html_escape(x$name), html_escape(x$name)), ""), collapse = "")
  paste0("<!doctype html><meta charset=utf-8><title>", name, "</title><style>body{font:16px system-ui;max-width:40em;margin:2em auto}textarea{width:100%}</style>",
         "<h1>", name, "</h1><p>", html_escape(iface$description), "</p><form id=f>", rows, "<label>key <input name=__key type=password></label> <button>Ask</button></form><pre id=out></pre><script>",
         "f.onsubmit=async e=>{e.preventDefault();const d=new FormData(f),inputs={};for(const[k,v]of d)if(k!='__key'){try{inputs[k]=JSON.parse(v)}catch{inputs[k]=v}}",
         "const r=await fetch('call',{method:'POST',headers:{'content-type':'application/json',authorization:'Bearer '+d.get('__key')},body:JSON.stringify({inputs})});",
         "out.textContent=JSON.stringify(await r.json(),null,2)}</script>")
}

authorized <- function(s, headers) {
  if (!length(s$keys)) return(TRUE)
  got <- headers[["authorization"]] %||% ""
  if (!startsWith(tolower(got), "bearer ")) return(FALSE)
  given <- trimws(substring(got, 8L))
  any(vapply(s$keys, function(k) same_secret(given, k), NA))
}

#' @rdname ai_service
#' @param service A service.
#' @param method,path The request's method and path.
#' @param headers Its headers, a named list (lower-case names).
#' @param body Its body (raw or text).
#' @export
serve_request <- function(service, method, path, headers = list(), body = raw()) {
  s <- service
  path <- paste0("/", gsub("^/+|/+$", "", sub("\\?.*$", "", path)))
  names(headers) <- tolower(names(headers))
  tryCatch({
    if (method == "GET" && path == "/") return(list(status = 200L, headers = list("Content-Type" = "text/html; charset=utf-8"), body = serve_form(s)))
    if (!authorized(s, headers)) return(json_reply(401L, list(error = list(type = "Unauthorized"))))
    if (method == "GET" && path == "/interface") return(json_reply(200L, serve_describe(s)))
    if (method == "GET" && path == "/openapi.json") return(json_reply(200L, serve_openapi(s)))
    text <- if (is.raw(body)) rawToChar(body) else as.character(body %||% "")
    data <- if (!nzchar(text)) lmcc::jobj() else tryCatch(lmcc::parse_json(text), error = function(e) NULL)
    if (is.null(data)) return(json_reply(400L, list(error = list(type = "BadRequest", message = "the body is not JSON"))))
    if (!is.list(data) || (length(data) && is.null(names(data)))) return(json_reply(400L, list(error = list(type = "BadRequest", message = "the body is a JSON object"))))
    parent <- headers[["functai-parent"]]
    if (!is_str(parent) || !grepl(UUID_TEXT, parent)) parent <- NULL
    parts <- strsplit(sub("^/", "", path), "/", fixed = TRUE)[[1L]]
    if (method == "POST" && path == "/call") return(serve_call(s, data, parent))
    if (method == "POST" && path == "/stream") return(serve_stream(s, data, parent))
    if (length(parts) >= 3L && parts[[1L]] == "conversations" && parts[[3L]] == "turns")
      return(serve_conversation(s, method, parts[[2L]], parts[-(1:3)], data, headers, parent))
    json_reply(404L, list(error = list(type = "NotFound")))
  }, functai_interface_input = function(e) error_reply(422L, e, message = TRUE),
     functai_interface_output = function(e) error_reply(422L, e),
     functai_conversation_error = function(e) error_reply(if (identical(e$code, "turn-unknown")) 404L else if (identical(e$code, "conversation-id")) 400L else 409L, e, message = TRUE),
     functai_refusal = function(e) error_reply(409L, e),
     error = function(e) error_reply(500L, e))
}

# The request's inputs, checked against the interface before anything runs
# (an input it does not have is interface-input, 422).
served_inputs <- function(s, data) {
  inputs <- data$inputs %||% lmcc::jobj()
  if (!is.list(inputs) || (length(inputs) && is.null(names(inputs))))
    stop(rlang::error_cnd(c("functai_interface_input", "functai_refusal"), code = "interface-input", field = NULL, message = "inputs is a JSON object of the program's inputs"))
  known <- vapply(program_iface(s$program)$inputs, function(x) x$name, "")
  extra <- setdiff(names(inputs), known)
  if (length(extra)) stop(rlang::error_cnd(c("functai_interface_input", "functai_refusal"), code = "interface-input", field = sort_code_points(extra)[[1L]],
                                           message = sprintf("%s: %s is not one of its inputs", program_name(s$program), sort_code_points(extra)[[1L]])))
  # one value each: a list is one value (a list column of one row)
  lapply(inputs, function(v) if (is.list(v)) list(v) else v)
}

in_service <- function(s, parent, code) {
  old <- list(to = the$approvals_to, parent = the$remote_parent)
  the$approvals_to <- s$approvals; the$remote_parent <- parent
  on.exit({ the$approvals_to <- old$to; the$remote_parent <- old$parent })
  caller <- c(caller_of(effective()), list(kind = "api"))
  settings <- list(caller = caller)
  if (!is.null(s$lm) && is_program(s$program)) settings$lm <- s$lm
  rlang::inject(with_ai_config(code, !!!settings))
}

# One served call, watched (so its id and events are known), run as calling it runs.
watched_call <- function(s, given, parent) {
  st <- new.env(parent = emptyenv()); st$events <- list(); st$calls <- character(0); st$closed <- FALSE; st$each <- NULL
  old <- the$stream_opening
  the$stream_opening <- st
  on.exit(the$stream_opening <- old)
  out <- tryCatch(list(value = in_service(s, parent, do.call(s$program, given))), error = identity)
  the$stream_opening <- old
  list(stream = st, out = out)
}

served_outputs <- function(s, st, value) {
  names_ <- vapply(program_iface(s$program)$outputs, function(x) x$name, "")
  done <- Filter(function(e) e$kind == "done" && length(st$calls) && identical(e$call, st$calls[[1L]]), st$events)
  v <- if (length(done)) done[[1L]]$value else value_json(value)
  if (length(names_) == 1L) stats::setNames(list(v), names_) else v
}

serve_call <- function(s, data, parent) {
  given <- served_inputs(s, data)
  got <- watched_call(s, given, parent)
  if (inherits(got$out, "condition")) stop(got$out)
  v <- got$out$value
  outputs <- served_outputs(s, got$stream, v)
  json_reply(200L, list(call = got$stream$calls[[1L]], outputs = outputs, value = if (length(outputs) == 1L) outputs[[1L]] else outputs))
}

serve_stream <- function(s, data, parent) {
  given <- served_inputs(s, data)
  got <- watched_call(s, given, parent)
  v <- new_view("outside", program_core_of(s$program)$answer_from)
  shown <- Filter(Negate(is.null), lapply(got$stream$events, function(e) view_event(v, e)))
  list(status = 200L, headers = list("Content-Type" = "text/event-stream", "Cache-Control" = "no-cache"), body = sse_text(shown))
}

turn_json <- function(s, t) {
  out <- list(turn = t$id); out["parent"] <- list(t$parent); out$state <- t$state; out$inputs <- t$inputs
  if (identical(t$state, "done")) {
    names_ <- vapply(program_iface(s$program)$outputs, function(x) x$name, "")
    o <- t$outputs; out$outputs <- o[names(o) %in% names_]; out["value"] <- list(t$value)
  }
  if (identical(t$state, "waiting") && s$approvals == "caller")
    out$waiting <- lapply(t$waiting, function(a) list(invocation = a$invocation, name = a$name, input = a$input, path = a$path))
  if (length(t$error)) out$error <- t$error[names(t$error) %in% c("type", "code")]
  out
}

serve_conversation <- function(s, method, cid, rest, data, headers, parent) {
  chat <- ai_conversation(s$program, cid, store = s$store)
  if (method == "POST" && !length(rest)) {
    given <- served_inputs(s, data)
    if (!is.null(data$after)) chat <- continue_from(chat, data$after)
    c <- conv_of(chat)
    err <- tryCatch({ in_service(s, parent, send_turn(c, given, list(), data$request_id)); NULL }, error = identity)
    if (inherits(err, c("functai_interface_input", "functai_conversation_busy", "functai_conversation_id"))) stop(err)
    log <- read_conv(c)
    tid <- if (!is.null(data$request_id) && !is.null(log$request_ids[[as.character(data$request_id)]])) log$request_ids[[as.character(data$request_id)]] else c$head
    out <- turn_json(s, turn_object(c, log$turns[[tid]])); out$conversation <- cid
    return(json_reply(201L, out))
  }
  if (method == "GET" && !length(rest)) {
    t <- ai_turns(chat)
    c <- conv_of(chat); log <- read_conv(c)
    return(json_reply(200L, list(conversation = cid, turns = lapply(t$turn, function(id) turn_json(s, turn_object(c, log$turns[[id]]))))))
  }
  t <- ai_turn(chat, rest[[1L]])
  if (method == "GET" && length(rest) == 1L) return(json_reply(200L, turn_json(s, t)))
  if (method == "GET" && identical(rest[-1L], "events")) {
    after <- parse_position(headers[["last-event-id"]])
    evs <- turn_events(t, after = after, view = "outside", timeout = 0)
    return(list(status = 200L, headers = list("Content-Type" = "text/event-stream", "Cache-Control" = "no-cache"), body = sse_text(evs)))
  }
  if (method == "POST" && identical(rest[-1L], "stop")) { stop_turn(t); return(json_reply(202L, list(turn = t$id, stopping = TRUE))) }
  if (method == "POST" && length(rest) == 3L && rest[[2L]] == "approvals") {
    if (s$approvals != "caller") return(json_reply(403L, list(error = list(type = "Forbidden", message = "approvals go to the owner"))))
    verdict <- data$verdict
    if (!verdict %in% c("yes", "no")) return(json_reply(400L, list(error = list(type = "BadRequest", message = "verdict is 'yes' or 'no'"))))
    inv <- as.integer(rest[[3L]])
    in_service(s, parent, {
      if (verdict == "yes") approve(t, inv, resume = FALSE) else deny(t, inv, reason = data$reason, resume = FALSE)
      again <- ai_turn(chat, t$id)
      if (!length(again$waiting)) later::later(function() tryCatch(in_service(s, parent, resume_turn(again)), error = function(e) NULL))
    })
    return(json_reply(202L, list(turn = t$id, verdict = verdict)))
  }
  json_reply(404L, list(error = list(type = "NotFound")))
}

# ---------------------------------------------------------------- a program served elsewhere

remote_request <- function(url, key = NULL, data = NULL, timeout = 120, parent = NULL) {
  h <- curl::new_handle(timeout = timeout)
  headers <- list(Accept = "application/json")
  if (!is.null(key)) headers$Authorization <- paste("Bearer", key)
  if (!is.null(parent)) headers$`FunctAI-Parent` <- parent
  if (!is.null(data)) { headers$`Content-Type` <- "application/json"; curl::handle_setopt(h, customrequest = "POST", postfields = lmcc::json_text(data)) }
  curl::handle_setheaders(h, .list = headers)
  resp <- curl::curl_fetch_memory(url, handle = h)
  text <- rawToChar(resp$content); Encoding(text) <- "UTF-8"
  if (resp$status_code >= 400L) {
    err <- tryCatch(lmcc::parse_json(text)$error, error = function(e) NULL) %||% list()
    code <- err$code %||% sprintf("remote-%d", resp$status_code)
    msg <- err$message %||% sprintf("%s answered %d (%s)", url, resp$status_code, err$type %||% "error")
    if (code %in% c("interface-input", "interface-output")) stop(rlang::error_cnd(c(paste0("functai_", gsub("-", "_", code)), "functai_refusal"), code = code, field = err$field, message = msg))
    stop(rlang::error_cnd(c("functai_remote_error", "functai_refusal"), code = code, status = resp$status_code, message = msg, functai_type = "RemoteError"))
  }
  lmcc::parse_json(text)
}

#' A program served elsewhere, used here
#'
#' A program served by [ai_serve()] (or by functai in any language), used
#' like a local one: its interface is the server's (`GET /interface`), its
#' inputs are bound and checked here before anything is sent, and what comes
#' back is checked against its outputs. Each call is logged here (`program.kind`
#' `"remote"`) and there; the server's record names this call as its parent,
#' so the two logs make one call tree. It runs over a column and evaluates as
#' a local program does.
#' @param url Where it is served.
#' @param key Its bearer key.
#' @param timeout Seconds to wait for an answer.
#' @return A program.
#' @export
ai_remote <- function(url, key = NULL, timeout = 120) {
  base <- sub("/+$", "", url)
  described <- remote_request(paste0(base, "/interface"), key, timeout = timeout)
  if (!identical(as.integer(described$functai_interface %||% 0L), SERVE_FORMAT))
    stop(rlang::error_cnd(c("functai_remote_error", "functai_refusal"), code = "remote-format", message = sprintf("%s does not describe a functai program this version reads", url)))
  iface <- described$interface
  names_ <- vapply(iface$outputs, function(x) x$name, "")
  body <- function(...) {
    inputs <- list(...)
    current <- the$current
    got <- remote_request(paste0(base, "/call"), key, list(inputs = lapply(inputs, value_json)), timeout, parent = if (is.null(current)) NULL else current$id)
    outputs <- got$outputs %||% list()
    if (length(names_) == 1L) outputs[[names_[[1L]]]] else outputs[names_]
  }
  p <- program_from_interface(iface, body, name = described$name, .defined_in = "functai.remote", ai = identical(described$kind, "ai"))
  core <- program_core(p)
  core$remote <- list(url = base, version = described$version, kind = described$kind %||% "module")
  make_program(core)
}
