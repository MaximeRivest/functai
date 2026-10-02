# Reference

## Writing AI functions

A typed function becomes a model call. The parts of the function are the parts of the prompt.

| | |
| --- | --- |
| [ai](ai.md#functai.ai) | Turn a typed Python function into an AI function. |
| [_ai](ai-sentinel.md#functai._ai) | The model's answer, inside an AI function's body. |
| [module](module.md#functai.module) | ``@module``: a plain Python function that calls @ai functions, optimized as one program. |
| [FunctAIFunc](FunctAIFunc.md#functai.FunctAIFunc) | A typed Python function whose body is a model call. Build with ``@ai``. |
| [Prediction](Prediction.md#functai.Prediction) | Everything one call produced. |
| [JSON](JSON.md#functai.JSON) | Any JSON value: annotate a module's input or output with it to say "data, |

## Models and settings

Which model answers, how it is reached, and every other setting.

| | |
| --- | --- |
| [configure](configure.md#functai.configure) | Set defaults for every AI function: the model, sampling, layout, and more. |
| [login](login.md#functai.login) | Sign in to a provider once; every later session uses it. |
| [logins](logins.md#functai.logins) | Everything you can use right now: logins, saved keys, and keys in the environment. |
| [logout](logout.md#functai.logout) | Forget a saved login or key, on this machine. |
| [login_methods](login_methods.md#functai.login_methods) | The ways you can sign in to a provider, and how far each is proven. |

## Prompt layouts

How each value is written into the prompt and read back. The default needs nothing; these write the conversation yourself.

| | |
| --- | --- |
| [system](system.md#lmcc.adapter.system) |  |
| [user](user.md#lmcc.adapter.user) |  |
| [assistant](assistant.md#lmcc.adapter.assistant) |  |
| [developer](developer.md#lmcc.adapter.developer) |  |
| [turns](turns.md#functai.turns) | A turn slot in messages form (kernel §3a): the slot's turns become |

## Evaluation

Run a program on rows with known answers and score it, with an honest interval.

| | |
| --- | --- |
| [evaluate](evaluate.md#functai.evaluate) | Run a program on rows with known answers, and score it. |
| [Evaluation](Evaluation.md#functai.Evaluation) | The result of ``evaluate``: a score, its uncertainty, and every answer. |
| [compare](compare.md#functai.compare) | Compare two evaluations of the same rows, row by row. |
| [runs](runs.md#functai.runs) | Every evaluation logged in a folder, as one table. |
| [exact_match](exact_match.md#functai.exact_match) | The default metric: every expected output equals the prediction. |

## Optimizers

Improve the instruction and the worked examples a function sends. Each returns an improved copy; the function is unchanged. The classes are for `fn.opt(rows, optimizer=...)`.

| | |
| --- | --- |
| [labeled_few_shot](labeled_few_shot.md#functai.labeled_few_shot) | An improved copy: up to ``k`` rows with known answers become worked examples. |
| [bootstrap_few_shot](bootstrap_few_shot.md#functai.bootstrap_few_shot) | An improved copy: the function (or a stronger ``teacher`` model) runs on |
| [gepa](gepa.md#functai.gepa) | An improved copy whose instruction a ``teacher`` model rewrote from the |
| [LabeledFewShot](LabeledFewShot.md#functai.LabeledFewShot) | Rows with known answers become worked examples, sent before every call. |
| [BootstrapFewShot](BootstrapFewShot.md#functai.BootstrapFewShot) | Run the program (or a ``teacher``: a stronger model name, or an AI function) |
| [BootstrapFewShotWithRandomSearch](BootstrapFewShotWithRandomSearch.md#functai.BootstrapFewShotWithRandomSearch) | Try several sets of demos and keep the one that scores best on the validation rows. |
| [InstructionSearch](InstructionSearch.md#functai.InstructionSearch) | Search instructions written by a model, with demo sets, and keep the best. |
| [GEPA](GEPA.md#functai.GEPA) | Rewrite the instruction from the function's mistakes: GEPA (Agrawal et |
| [Optimizer](Optimizer.md#functai.Optimizer) | The base class of optimizers: subclass it to write your own. |
| [ProgramState](ProgramState.md#functai.ProgramState) | What an optimizer tunes in one AI function: its instruction and its |

## Saving and shipping

Find everything a program depends on, save it to a folder, prove it runs elsewhere, load it back.

| | |
| --- | --- |
| [check](check.md#functai.check) | List everything a program depends on, and what would stop a clean save. |
| [Report](Report.md#functai.Report) | What ``functai.check`` found: everything a program depends on, and what |
| [Problem](Problem.md#functai.Problem) | One thing that keeps a program from being saved cleanly. |
| [save](save.md#functai.save) | Save a program to a folder, with everything it depends on. |
| [verify](verify.md#functai.verify) | Prove a saved program runs somewhere else, without calling a model. |
| [Verification](Verification.md#functai.Verification) | What ``verify`` found. |
| [load](load.md#functai.load) | Load a saved program, ready to call. |
| [describe](describe.md#functai.describe) | What a saved program takes and gives, without loading it or running |
| [file](file.md#functai.file) | A data file the program reads: ``open(functai.file("data/stopwords.txt"))``. |

## Streaming

Watch a call while it is made. `fn.stream(...)` returns a Stream; its events are in `functai.streaming`.

| | |
| --- | --- |
| [Stream](Stream.md#functai.Stream) | One call of an AI function or a module, watched while it is made. |
| [Cancelled](Cancelled.md#functai.Cancelled) | The stream was closed before its call ended. |

## Observers and journals

Every event of every call tree, given to your code as it happens, or kept in a store while the call runs; read back and followed live.

| | |
| --- | --- |
| [Journal](Journal.md#functai.Journal) | Where a call tree's kept log is written while it runs: a store, and how |
| [MemoryStore](MemoryStore.md#functai.MemoryStore) | A store kept in this process's memory, by the rules every store keeps |
| [Store](Store.md#functai.Store) | What a journal needs of a store (streaming.md, *The rules a store keeps*). |
| [Follower](Follower.md#functai.Follower) | A reader that follows a form of logs live, one state per tree |
| [flush](flush.md#functai.flush) | Wait (at most ``timeout`` seconds; None: for ever) until every observer |
| [Outcome](Outcome.md#functai.Outcome) | What a call's program did: its ``value``, or its ``error``. |

## Conversations

A program's calls that remember each other, kept in a store; branches, what the model sees, and helpers' memory inside a module.

| | |
| --- | --- |
| [conversations.Conversation](conversations.Conversation.md#functai.conversations.Conversation) | A program's conversation: its turns, kept in a store, called like the |
| [conversations.Turn](conversations.Turn.md#functai.conversations.Turn) | One turn of a conversation, as its records say now. |
| [last_turns](last_turns.md#functai.last_turns) | Show the model only the last ``n`` earlier turns. |
| [all_turns](all_turns.md#functai.all_turns) | Show the model every earlier turn of the conversation (the default). |
| [remember](remember.md#functai.remember) | What an AI function called inside a module's conversation remembers. |
| [earlier](earlier.md#functai.earlier) | The conversation so far, as data: inside a module's turn, one row per |
| [FolderStore](FolderStore.md#functai.FolderStore) | Conversations kept in a folder, shared by every process that opens it: |
| [MemoryConversations](MemoryConversations.md#functai.MemoryConversations) | Conversations kept in this process's memory (lost when it ends): the |
| [Waiting](Waiting.md#functai.Waiting) | A turn stopped to wait for a person's answer (code ``turn-waiting``): |
| [ConversationError](ConversationError.md#functai.ConversationError) | A conversation, or one of its turns, refused what was asked |

## Plugins

Hooks over turns, context, calls, requests and tools; every change is data and recorded. Approval, compaction and delegation are plugins too.

| | |
| --- | --- |
| [Plugin](Plugin.md#functai.Plugin) | A named, versioned set of hooks. |
| [Change](Change.md#functai.Change) | What a hook changes. Each hook accepts some fields (HOOKS); a field it |
| [PluginError](PluginError.md#functai.PluginError) | A plugin refused or failed (contract/plugins.md). ``code`` is one |
| [load_plugin](load_plugin.md#functai.load_plugin) | A plugin from a Python file that defines ``plugin`` (an |
| [compaction](compaction.md#functai.compaction) | Keep a long conversation short: older turns are folded into a summary. |
| [delegate](delegate.md#functai.delegate) | Another program as a tool: an assistant hands part of its work to it. |

## Tools that ask first

Tools say what they do to the world; a person can be asked before they run.

| | |
| --- | --- |
| [tool](tool.md#functai.tool) | Make a function a tool that says what it does to the world. |
| [Tool](Tool.md#functai.Tool) | A function the model may call, with what it does to the world. Made by |
| [Approval](Approval.md#functai.Approval) | One tool call waiting for a person's answer. |
| [ApprovalError](ApprovalError.md#functai.ApprovalError) | A tool call needs a person's answer and nobody can be asked (code |

## Serving

Serve a program over HTTP to callers who see only its boundary; use a served program like a local one (`remote`'s page also has `RemoteProgram` and `RemoteError`).

| | |
| --- | --- |
| [serve](serve.md#functai.serve) | Serve a program over HTTP: its interface, calls, streams and |
| [Service](Service.md#functai.Service) | A program as an HTTP service, independent of any server: ``handle`` |
| [remote](remote.md#functai.remote) | A program served elsewhere, used like a local one (contract/serving.md). |
| [ServeError](ServeError.md#functai.ServeError) | A program cannot be served as asked (code ``serve-opaque``: an input or |

## The call log

Keep every call on disk, mark answers right or wrong, and turn the corrections into rows with known answers.

| | |
| --- | --- |
| [calls](calls.md#functai.calls) | Every logged call, as a table. |
| [rate](rate.md#functai.rate) | Say whether a call's answer is right, and if not, what it should have been. |
| [rated](rated.md#functai.rated) | The calls people rated, as rows with known answers. |
| [split](split.md#functai.split) | Two tables, with every group of rows on one side: ``train, test = |
| [prune_calls](prune_calls.md#functai.prune_calls) | Delete the call log's day folders older than a time, keeping what |

## Long runs

Replies kept on disk, and a check on a judge's evidence. See also `fn.map` (a progress line, and resuming by running again) and `prune_calls`.

| | |
| --- | --- |
| [quotes_found](quotes_found.md#functai.quotes_found) | Whether each quote is in the text, word for word. |
| [clear_cache](clear_cache.md#functai.clear_cache) | Forget the replies the reply cache kept, so the next identical requests |

## Baking into weights

Train a small model that answers an AI function, then run the same function on it.

| | |
| --- | --- |
| [bake.bake](bake.bake.md#functai.bake.bake) | Train weights that answer an AI function (or several); returns the baked model. |
| [bake.examples](bake.examples.md#functai.bake.examples) | From an AI function and rows of data to training examples. |
| [bake.plan](bake.plan.md#functai.bake.plan) | What a generative bake would do, decided, with nothing spent. |
| [bake.Plan](bake.Plan.md#functai.bake.Plan) | A generative bake, decided before anything is spent: |
| [bake.Examples](bake.Examples.md#functai.bake.Examples) | The training conversations for a student, made by |
| [bake.adopt](bake.adopt.md#functai.bake.adopt) | A model trained elsewhere (TRL, Axolotl, Unsloth, by hand), as a baked |
| [bake.judge](bake.judge.md#functai.bake.judge) | Measure a generative student on test rows: ``functai.evaluate`` on the |
| [bake.runs](bake.runs.md#functai.bake.runs) | Every bake run on this machine, newest first. |
| [bake.run](bake.run.md#functai.bake.run) | A bake run, from its folder: reattach to it after a restart. |
| [bake.Run](bake.Run.md#functai.bake.Run) | A training run: a folder, a process of its own, and a model at the end. |
| [bake.Baked](bake.Baked.md#functai.bake.Baked) | A model trained to answer one or more AI functions: what ``bake`` |
| [bake.BakeReport](bake.BakeReport.md#functai.bake.BakeReport) |  |
| [bake.load](bake.load.md#functai.bake.load) | A baked model, from its folder. |
| [bake.is_baked](bake.is_baked.md#functai.bake.is_baked) | Whether ``obj`` is a baked model (``Baked``), without importing what runs one. |
| [bake.BakeError](bake.BakeError.md#functai.bake.BakeError) | A bake, or a baked model, cannot do what was asked; the message says |

## Datasets

Small labelled tables to learn with. Every example on this site runs on them.

| | |
| --- | --- |
| [datasets.tickets](datasets.tickets.md#functai.datasets.tickets) | Customer support messages to a small homeware shop, with the team each belongs to. |
| [datasets.refunds](datasets.refunds.md#functai.datasets.refunds) | Refund requests to the homeware shop of :func:`tickets`, with the decision its rules give. |
| [datasets.field_notes](datasets.field_notes.md#functai.datasets.field_notes) | Bird survey notes written by volunteers, with the species, count and behaviour of each. |

## Inspection

See exactly what was sent and what came back.

| | |
| --- | --- |
| [phistory](phistory.md#functai.phistory) | The last model calls, as readable text: what was sent, what came back. |
| [inspect_history](inspect_history.md#functai.inspect_history) | The last ``n`` requests FunctAI sent (or answered from its cache), oldest first. |
| [signature_text](signature_text.md#functai.signature_text) | A one-line summary of the signature. |

## Errors

What functai raises, and what each one tells you to do.

| | |
| --- | --- |
| [FunctAIError](FunctAIError.md#functai.FunctAIError) | The base of every error FunctAI raises with a code: ``err.code``. |
| [InterfaceError](InterfaceError.md#functai.InterfaceError) | A program's interface refused what it was given or what it gave back. |
| [LoginRequired](LoginRequired.md#functai.LoginRequired) | No usable credential for the model's provider: sign in or pass a key. |
| [StepLimit](StepLimit.md#functai.StepLimit) | A function with tools asked the model ``max_steps`` times (8 by |
| [LogContentError](LogContentError.md#functai.LogContentError) | A ``log_content`` map that cannot be honoured (code |
| [SawError](SawError.md#functai.SawError) | What a call saw cannot be known, or shown again (contract/calls.md, |
| [JournalError](JournalError.md#functai.JournalError) | A journal kept the call from going on, or could not confirm its end. |
| [EventRefused](EventRefused.md#functai.EventRefused) | A store's refusal of an append, a claim or a read (contract/streaming.md, |
| [Refused](Refused.md#functai.Refused) | ``functai.save`` found problems that keep the program from being saved |
| [LoadRefused](LoadRefused.md#functai.LoadRefused) | The saved program cannot be loaded as saved; ``.problems`` says why, and |

## Low-level helpers

The pieces `@ai` uses to read a function. You rarely need them directly.

| | |
| --- | --- |
| [flexiclass](flexiclass.md#functai.flexiclass) | Make a plain annotated class a dataclass, as ``@ai`` does for types. |
| [docments](docments.md#functai.docments) | Docments: documentation harvested from code (inline comments, docstrings), |
| [docstring](docstring.md#functai.docstring) | Get cleaned docstring for functions and classes. |
| [parse_docstring](parse_docstring.md#functai.parse_docstring) | Split a numpy-style docstring into its parts. |
| [compute_signature](compute_signature.md#functai.compute_signature) | The lmcc signature of an @ai function: inputs, outputs, instruction. |