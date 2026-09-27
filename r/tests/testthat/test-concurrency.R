# Rows run at once over the network: a fake OpenAI server in another R
# process answers each request after half a second.

test_that("a column's calls are in flight together, and each answer goes to its own row", {
  skip_on_cran()
  skip_if_not_installed("httpuv")
  skip_if(!nzchar(Sys.which("Rscript")), "no Rscript on PATH")
  port <- httpuv::randomPort()
  pidfile <- withr::local_tempfile()
  server <- sprintf('
    writeLines(as.character(Sys.getpid()), "%s")
    library(httpuv)
    reply <- function(text) sprintf(\'{"id":"r","object":"response","status":"completed","model":"gpt-4.1-mini","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":%%s}]}],"usage":{"input_tokens":3,"output_tokens":2,"total_tokens":5}}\', jsonlite::toJSON(text, auto_unbox = TRUE))
    runServer("127.0.0.1", %d, list(call = function(req) {
      body <- rawToChar(req$rook.input$read())
      n <- regmatches(body, regexpr("row [0-9]+", body))
      promises::promise(function(resolve, reject) later::later(function()
        resolve(list(status = 200L, headers = list("Content-Type" = "application/json"),
                     body = reply(sprintf("<result>\\n%%s\\n</result>", n)))), 0.5))
    }))', pidfile, port)
  script <- withr::local_tempfile(fileext = ".R")
  writeLines(server, script)
  system2("Rscript", script, stdout = FALSE, stderr = FALSE, wait = FALSE)
  withr::defer(if (file.exists(pidfile)) tools::pskill(as.integer(readLines(pidfile))))
  for (i in 1:100) { ok <- tryCatch({ curl::curl_fetch_memory(sprintf("http://127.0.0.1:%d/", port)); TRUE }, error = function(e) FALSE); if (ok) break; Sys.sleep(0.1) }
  skip_if(!ok, "the fake server did not start")

  router <- lm15::new_router(api_keys = list(openai = "sk-test"), base_urls = list(openai = sprintf("http://127.0.0.1:%d/v1", port)))
  log <- withr::local_tempdir()
  echo <- ai(echo ~ text, "Say which row this is.", .lm = "gpt-4.1-mini", .router = router, .log_calls = log, .concurrency = 8L)
  started <- Sys.time()
  out <- echo(sprintf("row %d", 1:8))
  took <- as.numeric(Sys.time() - started, units = "secs")
  expect_identical(out, sprintf("row %d", 1:8))
  expect_lt(took, 2)                       # one at a time would take 4 s
  started <- Sys.time()
  out <- update(echo, concurrency = 2L)(sprintf("row %d", 1:4))
  expect_identical(out, sprintf("row %d", 1:4))
  expect_gt(as.numeric(Sys.time() - started, units = "secs"), 0.9)   # two waves of half a second
  # each call's time is its own request's, not its wave's or its batch's
  lines <- log_lines(log)
  expect_length(lines, 12L)
  own <- vapply(lines, function(l) l$seconds - l$exchanges[[1L]]$seconds, 0)
  expect_true(all(abs(own) < 0.05), info = paste(round(own, 3), collapse = " "))
})
