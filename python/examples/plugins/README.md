# Plugins ported from Pi and Chattering

Real extensions, each written again as a FunctAI plugin with the public hooks
only: the test that the plugin API covers what they need
(`tests/test_ported_plugins.py` runs every one).

| plugin | was | hooks |
|---|---|---|
| `modes.py` | Pi's and Chattering's `modes` | `before_call` (instruction, sections, tools), entries |
| `session_prompt.py` | Pi's `session-prompt` | `before_call` (instruction), entries |
| `auto_inject.py` | Pi's `auto-inject` | `turn_start` (inputs) |
| `prompt_capture.py` | Pi's `prompt-capture` | `before_call`, `request` (reading only) |
| `checkpoints.py` | Chattering's `checkpoints` | `tool_call`, `tool_result`, entries |
| `bulky_budget.py` | Chattering's `image-budget` | `context` (`without`) |

Delegation and compaction are built in (`functai.delegate`, `functai.compaction`).

What the originals do that a plugin does not: register commands, shortcuts,
status lines and editor text. Those belong to the host showing the
conversation (a terminal, Chattering's pages); a plugin offers the function
(`modes.switch`, `session_prompt.set`, `checkpoints.undo`) and the host binds
it to a command.
