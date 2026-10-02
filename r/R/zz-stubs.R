# temporary
request_hooks <- function(job, request) list(request = request, replaced = FALSE)
before_call_hooks <- function(core, call, s, row, plan, past) list()
tool_gate <- function(job, tool, c, n) list(input = c$input %||% list())
tool_result_hooks <- function(job, tool, c, input, out) out
recorded_reply <- function(job, request) NULL
reply_note <- function(job, response) invisible()
tool_recorded <- function(call, n) NULL
tool_started <- function(call, tool, c, n, input) invisible()
tool_done <- function(call, tool, c, n, out) invisible()
turn_check_stop <- function(run) invisible()
steps_of <- function(call, turn) NULL
escalate <- function(core, job, s) job
check_plugins <- function(x, call = NULL) x
check_approve <- function(x, call = NULL) x
