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

## Models and settings

Which model answers, how it is reached, and every other setting.

| | |
| --- | --- |
| [configure](configure.md#functai.configure) | Set defaults for every AI function: the model, sampling, layout, and more. |
| [login](login.md#functai.login) | Sign in to a provider once; every later session uses it. |
| [logins](logins.md#functai.logins) | Everything you can use right now: logins, saved keys, and keys in the environment. |
| [logout](logout.md#functai.logout) | Forget a saved login or key, on this machine. |
| [login_methods](login_methods.md#functai.login_methods) | The ways lm15 can sign in to a provider, and whether each is proven. |

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

Improve the instruction and the worked examples a function sends. Used through `fn.opt(...)`.

| | |
| --- | --- |
| [LabeledFewShot](LabeledFewShot.md#functai.LabeledFewShot) | Up to ``k`` labeled examples become demos (a random sample, or the first ``k``). |
| [BootstrapFewShot](BootstrapFewShot.md#functai.BootstrapFewShot) | Run the program (or a ``teacher``: a stronger model name, or an AI function) |
| [BootstrapFewShotWithRandomSearch](BootstrapFewShotWithRandomSearch.md#functai.BootstrapFewShotWithRandomSearch) | Try several sets of demos and keep the one that scores best on the validation rows. |
| [InstructionSearch](InstructionSearch.md#functai.InstructionSearch) | Search instructions written by a model, with demo sets, and keep the best. |
| [Optimizer](Optimizer.md#functai.Optimizer) | The base class of optimizers. |

## Saving and shipping

Find everything a program depends on, save it to a folder, prove it runs elsewhere, load it back.

| | |
| --- | --- |
| [check](check.md#functai.check) | List everything a program depends on, and what would stop a clean save. |
| [Report](Report.md#functai.Report) | What ``functai.check`` found: the graph, the requirements, the problems. |
| [Problem](Problem.md#functai.Problem) | One thing that keeps a program from being saved cleanly. |
| [save](save.md#functai.save) | Save a program to a folder, with everything it depends on. |
| [verify](verify.md#functai.verify) | Prove a saved program runs somewhere else, without calling a model. |
| [Verification](Verification.md#functai.Verification) | What ``verify`` found. |
| [load](load.md#functai.load) | Load a saved program, ready to call. |
| [file](file.md#functai.file) | A data file the program reads: ``open(functai.file("data/stopwords.txt"))``. |

## The call log

Keep every call on disk, mark answers right or wrong, and turn the corrections into rows with known answers.

| | |
| --- | --- |
| [calls](calls.md#functai.calls) | Every logged call, as a table. |
| [rate](rate.md#functai.rate) | Say whether a call's answer is right, and if not, what it should have been. |
| [rated](rated.md#functai.rated) | The calls people rated, as rows with known answers. |

## Baking into weights

Train a small model that answers an AI function, then run the same function on it.

| | |
| --- | --- |
| [bake.bake](bake.bake.md#functai.bake.bake) | Train weights that answer ``fn``; returns the baked model (see ``Baked``). |
| [bake.Baked](bake.Baked.md#functai.bake.Baked) | A model trained for one AI function (see the module docstring). |
| [bake.BakeReport](bake.BakeReport.md#functai.bake.BakeReport) |  |
| [bake.load](bake.load.md#functai.bake.load) | A baked model from its folder (its file hashes are checked). |

## Datasets

Small labelled tables to learn with. Every example on this site runs on them.

| | |
| --- | --- |
| [datasets.tickets](datasets.tickets.md#functai.datasets.tickets) | Customer support messages to a small homeware shop, with the team each belongs to. |
| [datasets.field_notes](datasets.field_notes.md#functai.datasets.field_notes) | Bird survey notes written by volunteers, with the species, count and behaviour of each. |

## Inspection

See exactly what was sent and what came back.

| | |
| --- | --- |
| [phistory](phistory.md#functai.phistory) | The last model calls, as readable text: what was sent, what came back. |
| [inspect_history](inspect_history.md#functai.inspect_history) | The last ``n`` requests functai sent (or answered from its cache), oldest first. |
| [signature_text](signature_text.md#functai.signature_text) | A one-line summary of the signature. |
| [clear_cache](clear_cache.md#functai.clear_cache) | Forget every cached reply. |

## Errors

What functai raises, and what each one tells you to do.

| | |
| --- | --- |
| [LoginRequired](LoginRequired.md#functai.LoginRequired) | No usable credential for the model's provider: sign in or pass a key. |
| [StepLimit](StepLimit.md#functai.StepLimit) | The tool loop reached ``max_steps`` without an answer. ``.turn`` is the turn so far. |
| [Refused](Refused.md#functai.Refused) | ``save`` found errors; ``.report`` has them all. |
| [LoadRefused](LoadRefused.md#functai.LoadRefused) | The saved program cannot be loaded as saved; ``.problems`` says why. |

## Low-level helpers

The pieces `@ai` uses to read a function. You rarely need them directly.

| | |
| --- | --- |
| [flexiclass](flexiclass.md#functai.flexiclass) | Make a plain annotated class a dataclass, as ``@ai`` does for types. |
| [docments](docments.md#functai.docments) | Docments: documentation harvested from code (inline comments, docstrings), |
| [docstring](docstring.md#functai.docstring) | Get cleaned docstring for functions and classes. |
| [parse_docstring](parse_docstring.md#functai.parse_docstring) | Split a numpy-style docstring into its parts. |
| [compute_signature](compute_signature.md#functai.compute_signature) | The lmcc signature of an @ai function: inputs, outputs, instruction. |