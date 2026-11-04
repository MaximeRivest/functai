# 16 — Studio: seeing, judging and shaping AI pipelines

*Status, 2026-11-02, evening: **the first version is built** (Python,
`python/functai/studio/`, `functai studio …`; tests `python/tests/test_studio.py`);
"As built" below says what exists, what does not, and what was found on the way.
Read twice by Maxime before that. Agreed: the general idea; code as the only
source of truth (§4); contract additions in all four languages in one commit
(§21, §26), and none made yet; agent first (§23); each step's model chosen by
the person, the helper on the person's model (§24); the default model the
newest small one, checked against the providers' lists (`claude-haiku-5-5`).
The mockup, `design/16-studio-mockup/index.html`, is its fourth version.*

## As built (2026-11-02)

**What exists** (`functai studio --help`; the agents' method in
`python/functai/skill/studio.md`):

| part | file | what it does |
|---|---|---|
| the project | `studio/project.py` | finds the program in a file (`club.py`, `club.py:name`, or the only one in the folder), loads it as `functai serve` does, its steps, its examples (`examples/<name>.csv` or `.jsonl`), its folders (`.functai/calls`, `.functai/studio`), settings (default model, budget, limit, month size), the examples kept aside (40 %, by a stable hash of the example and a project salt) |
| the map | `studio/graph.py` | §6.1 and §21.1 for Python only, as Studio's own data (not the contract yet): the module body read with `ast`: data flow, conditions as connections, fixed values (with the steps of the other branch, to tell which path a run took), file values, loops; a deterministic layered layout |
| what happened | `studio/data.py` | the call log read into rows (example → root call → step calls), versions named by first appearance, judgements with their source (person, carried, step), scores with Wilson ranges, money from tokens × a dated price table (`studio/prices.json`, OpenRouter's list of 2026-11-02, 23 models), who did what |
| running | `studio/running.py` | examples (cost estimated first, budget, threads, the project's reply cache), a model for one step (a `before_call` plugin, recorded), what-if (§7.4) by standing in for a step in the module's globals and recording apart, `why`, `blame`, `compare` (before = a git revision or a file, or `--lm`), judgements of what-if answers kept apart (`whatif-judgements.jsonl`) |
| the code edit | `studio/edit.py` | a step's `@ai(lm="…")`, one decorator argument, the rest of the file untouched |
| the summary | `studio/page.py` | the one-page HTML (§6.4): six facts, an SVG of the map, three examples step by step; no script, no AI |
| the dashboard | `studio/server.py`, `studio/ui/` | stdlib HTTP server, JSON API over the same functions, jobs one at a time, 127.0.0.1 unless `--host` (then a key), a header other sites cannot send; the page: start (a prompt to give one's coding agent, starters including work, a ready example), Map, Summary (download, copy link), Table, Code (changes since the last commit in green), judging in place and ten at a time (keys), what-if and compare results that a person finishes judging, each step's model (try, then use), costs (per message, per month, a limit), who did what, "work with your coding agent" (requests to copy), an address per view, live refresh |
| the example | `studio/examples/club/` | the club inbox: 3 steps, an FAQ, 12 messages |

**Checked**: 16 offline tests (fake model) for every part; FunctAI's 920
Python tests; by hand on real models (Claude Haiku 5.5, Sonnet 5.5) with a
browser driven by a script: the example made, 12 messages run (13.5 s, 0.14¢),
judging, ten checks with the keyboard, a what-if that ran the real later steps,
a model tried for one step and then written into the code, Back and Forward,
reload, the start page.

**Not built, on purpose or not yet**:

- **Requirements and `check`** (§26): not built; the three answers (PASS, FAIL,
  NOT ENOUGH CHECKED) wait for the syntax design.
- **`propose`**: not built; agents change the code in their own way and the
  person keeps or drops it in their agent, as decided (§23).
- **The contract additions** (graph, interface lines, given answers, runner,
  `via`, requirements): none. Studio keeps its own map and its what-if stand-in
  in Python; the rest needs no contract change. When the design has proved
  itself, they go into all four languages in one commit.
- **Other languages**: Python programs only (the runner of §26 is not built).
- **The built-in helper and its ask box**: not built (agent first); the
  dashboard has "Work with your coding agent" instead.
- **Runs of agents** (§6.5: timeline, rewind, paths map): not built.
- **Pictures and files as example inputs**: examples are text, numbers, yes/no,
  JSON; FunctAI's media inputs are not read from the CSV yet.
- **The language toggle of the Code view**: not built.
- **An online demo**: not made (how the mockup was published is not on record).

**Trade-offs taken**:

- The what-if replaces a step in the module's globals for the length of one
  call, under a lock: it works for steps a module calls by name (the usual
  way), not for steps reached through attributes or other modules, and what-if
  runs are one at a time.
- A judgement carries to a later run only when the answer is the same, or
  exactly comparable; free text that changed waits for a person (in a
  comparison, side by side). Honest, and more clicks.
- An end result is counted wrong when a step on its way was judged wrong and
  nobody judged the end: usual, but not always true; it says so ("because kind
  was judged wrong") and a person's judgement of the end wins.
- A model change is not a new version (the contract's rule), so runs made with
  the old model stay current; the dashboard notices and offers to run them again.
- The money is list price × tokens: replies from the cache count 0, the
  provider's own caching discounts are not applied, and models not in the price
  table count tokens only, which the dashboard says.
- The dashboard serves one person on this computer; another address needs the
  key it prints, and anyone with the key can spend through the person's
  accounts.
- It speaks plain http. Browsers force names ending in `.ts.net` to https, so
  from another machine the link is the computer's name or address, or Tailscale
  gives it https (`tailscale serve --bg --https=<port> http://127.0.0.1:<port>`,
  tailnet only); `functai studio open --host 0.0.0.0` prints all three
  (found 2026-11-02: the first link given to Maxime used the `.ts.net` name
  over http and did not open on his laptop).

**After a third simulated run (2026-11-02, on mockup 2; checked against the
build)**: four findings still held in the built Studio and were fixed the same
day: actions other than "run" asked a model with no cost shown and no limit
(now every such button shows its cost, the limit applies to all, the command
line's budget too); nothing said whose account pays (now "Who pays" in Costs,
and a warning when a step's company has no key here); no history or undo (each
version's code is kept, listed with its score, and can be put back); step
names meant nothing to non-coders (each card now says what the step is asked).
The agents' method now says to name every fact the person has and ask where it
comes from. Two findings did not hold (the example overwriting work; a promised
score). Parent consent and the household layer stay later (§10.2, §11).

**After Maxime's third reading (2026-11-02): editing in the dashboard, and
reading what the models saw.** Working only through a coding agent felt rigid;
changing models was too hard; and the answers could not be read at a glance.
Built the same day, with the code still the only truth:

- **Edits written into the code** (`studio/edit.py`): a step's model (and every
  step's, in Models), its instruction (docstring), its kind of answer (return
  annotation, `Literal[...]` for choices, the import added), which value feeds
  each input (a name available at that call: the pipeline's inputs, earlier
  answers, the file's values, each labelled "what kind gave", "read from
  club_faq.md"), a fixed value, a file a value is read from, a new step after
  another (its call wraps the other's: `shorten(answer(...))`, so its answer goes
  on in its place) or at the end (it wraps every returned expression), taking a
  step out (its call replaced by its input of the same type as its answer), and
  the whole code. Each edit changes only its span, is refused when the result is
  not Python or when the file changed since the person looked (a hash of the
  text sent with the edit), keeps the text before it, and is undone in one click
  while the file is as the edit left it. Trade-off: the graph edits cover the
  common shapes (steps called by name in one `@module`); anything else is "in
  the code", which the dashboard now edits too.
- **Review** (`data.example_detail`, the Review view): per example, per step,
  the inputs as fields (long ones folded), the answer (all outputs when there
  are several), the model's reasoning when it gave any, and the exact request
  and reply; judging, "should be" and notes per step, by keyboard.

**Standalone, for everyone (2026-11-02, Maxime: "think about a user installing that
as a standalone, then what would they want? That is the priority").** Someone who
installs FunctAI, has no coding agent and no keys set, types `functai studio`, and
does everything in the page:

- **Accounts** (`studio/connect.py`): lm15's managed logins through FunctAI's
  (`functai.login(ui=...)`, new): subscriptions signed in through lm15's own
  flows, with the page as lm15's `AuthUI` (links, device codes, codes to paste,
  choices; one sign-in at a time, cancelled after 15 minutes); API keys saved
  and checked with a model list (a refused key is not kept); sign out. Decided
  with Maxime: lm15's shared credentials file, so the same connections serve
  scripts. Keys and sign-in answers only from this computer or an encrypted
  tunnel (a plain-http page from the network is refused, and the page says why).
  Models offered from connected accounts first; the default model for unnamed
  steps follows what is connected; subscription and local models cost nothing
  per use in the money shown.
- **Home and a new pipeline** (`studio/newpipe.py`, `studio/helper.py`): a home
  page listing the folder's pipelines; a pipeline planned from a sentence by a
  FunctAI program of Studio's own (`plan_pipeline`, typed: inputs, facts, steps,
  example inputs), edited by the person, and written as code by Studio, never by
  the model: one `@ai` per step, one `@module` calling them in order, each fact a
  file to fill in (flagged until filled), the examples a CSV. Trade-off: new
  pipelines are linear; branches come later (Map edits, Code, an agent).
- **Examples** (`edit.add_examples`, `paste_examples`, …): add, change, remove,
  paste (lines, or a spreadsheet table), upload, keep a run from elsewhere,
  have the AI write ten; all undoable.
- **Checked as a newcomer**, in a browser, with a fresh home folder (no keys, no
  sign-ins): paste a key, describe a bike shop's email helper, plan with AI (3
  steps, 2 facts to fill in), build, write examples with AI, run 18 (0.88¢),
  Review. Sign-in flows checked with a stand-in provider (device code and
  question; sign-in link and pasted address; cancel).

**Found on the way (in FunctAI, outside Studio)**:

- `graph.Origins` read every installed package's file list each time a
  program's graph was read (2.5 s here), once per thread on a module's first
  concurrent calls. Fixed in `functai/graph.py`: read once, again when a folder
  on `sys.path` changes. Python's tests went from 72 s to 16 s for Studio's.
- `calllog._module_graph` keeps a module's graph until a name it bound
  changes; a global replaced by something the analysis does not bind (a plain
  function) leaves the old graph, and so the old version, in place. Studio
  forgets the entry after a what-if; FunctAI itself is unchanged.
- `tests/test_saving.py::test_verify_in_a_fresh_environment` fails on this
  machine before and after these changes: FunctAI 1.2.0 on PyPI requires lm15
  2, the environment has lm15 1.1.0.
- The OpenAI key on this machine has no credit left (HTTP 429, "no credits
  remaining"), so the OpenAI models could not be tried live.


Sources: FunctAI's contract (`calls.md`, `programs.md`, `streaming.md`,
`scores.md`, `tools.md`, `plugins.md`, `saved.md`, `serving.md`),
`python/functai/graph.py` (`functai.check`, which already finds a
program's parts by reading its code), lmcc's README, lm15's README,
Chattering `design/74` (program pages, as built), `75` (making programs),
`76` (programs you ship), `design/06` here, and 35 papers found with the
Scholars Reading List search and read from their abstracts (references at
the end; claims are only what the abstracts say).

---

## 1. The job

*When I build something out of AI steps, show me what it actually does,
step by step, on real examples. Let me tell it when it is wrong, find which
step is at fault, fix that step and prove the fix helped. Then let me run
it on a whole table, ship it, and show other people how it works. It should
never trap my work in a tool.*

Who, and the version of that job each has:

| who | the job in their words | what they hire today |
|---|---|---|
| a curious 14-year-old | "I want to make an AI thing that does X, see if it's any good, and show my friends" | ChatGPT, Scratch, Teachable Machine, sometimes n8n tutorials |
| a student (school, university) | "I want to build a pipeline for my project, measure it, and put it in my report" | notebooks, LangChain tutorials, Langflow |
| a data scientist | "Run an AI step over 50,000 rows, measure it honestly, explain it to my manager" | notebooks, spreadsheets, slides |
| an AI engineer | "Debug why my agent did that, fix it, stop it from regressing, deploy it" | LangSmith / Langfuse / Phoenix, LangGraph Studio, homemade dashboards |
| a stakeholder (a manager, a domain expert, a teacher, a parent) | "What does this system do, how often is it right, and can I trust it with this?" | slides someone made, a demo |

One tool for all five only works if the **same object** serves them all.
In Studio that object is the program, as code, with its record of calls.
Each person gets a different view of it. Nobody gets a different copy.

## 2. What the market does, and where it stops

| family | examples | what they get right | where they stop |
|---|---|---|---|
| automation canvases | n8n, Make, Zapier, Node-RED | data visible at each node after a run; easy to start; triggers and connectors | the graph file *is* the program (lock-in); no measurement of whether answers are right; big graphs turn into spaghetti |
| LLM flow builders | Langflow, Flowise, Dify, Rivet, Vellum | drag-and-drop LLM steps; a playground | same lock-in (export-to-code is second class); evaluation bolted on; the prompt you see is not the bytes sent |
| node media tools | ComfyUI, Blender nodes, Max/MSP | a huge remix culture; ComfyUI embeds the graph in every output image (provenance for free) | graphs over a few dozen nodes are hard to read; no notion of "right" |
| graphs drawn from code | LangGraph Studio, Dagster's asset graph | the graph comes from your code, so no lock-in | you see the graph but cannot shape the code from it; little about quality |
| tracing and evals | LangSmith, Langfuse, Braintrust, Phoenix, Weave | traces, datasets, LLM judges, experiments | a separate place from where you build; made for engineers; judges are rarely checked against people |
| prompt workbenches | ChainForge, promptfoo, OpenAI playground | compare prompts × models × inputs in a grid | one step at a time, or a small chain; no deployment |
| data workflow tools | KNIME, Orange, Alteryx | tables flowing through widgets; Orange is used to teach | not built for LLM steps, agents or text quality |
| learning tools | Scratch, Teachable Machine | very low floor, playful, remixable, loved by kids | they stop well before real work: you can't take them to a job |
| notebooks | Jupyter, marimo, Observable, Hex | real code, real data, shareable | messy; no picture of the pipeline; no per-step judging; stakeholders don't read them |

**The gap nobody fills:** a place where the *picture*, the *code*, the
*examples* and the *measurements* are one thing; where a beginner starts in
seconds and a professional never hits the ceiling; and where leaving costs
nothing because the work was always plain code and plain files.

FunctAI is unusually well placed to fill it. Most of the hard parts exist:

- the program is typed code in four languages (`@ai`, modules);
- `functai.check` already finds a program's parts by *reading* its code
  (`python/functai/graph.py`);
- every call is a record in an open log, as a tree (`calls.md`), with what
  each step was shown (`saw`) and people's ratings;
- scores come with honest 95% ranges, and versions are compared on the
  same rows (`scores.md`, `evaluate`/`compare`);
- the exact request bytes are known (lmcc lays them out; lm15 sends them);
- streams, conversations, tools that ask first, plugins whose changes are
  recorded, serving, saved folders that load in any language, GEPA, bake.

Studio is mostly a **lens** over things FunctAI already records.

## 3. What the research says, as design rules

Each rule is a finding, and the decision it forces.

**R1. Show the steps and their in-between results; people then trust and
steer the system more.** Splitting an LLM task into steps whose
in-between results users can see and edit improved results and made the
system feel clearer and more controllable (AI Chains, 20 people) [1].
PromptChainer turned this into a visual tool for chains [2].
→ *Every wire shows a real value from a real call. Every in-between value
can be edited, and the pipeline reruns from there.*

**R2. Beginners prompt by trial and error, not system; they
overgeneralise from one example.** [3] Non-experts expected the model to
behave like a person being instructed and drew conclusions from single
outputs.
→ *Studio never shows one output alone when it can show several. "Try"
runs on a handful of examples by default, not one. Accuracy is always a
range, never one number from one try.*

**R3. You can't write your judging rules before seeing outputs (criteria
drift).** People need rules to grade outputs, but grading outputs is how
they find their rules; some rules only exist once you've seen the outputs
[4]. LLM judges inherit the faults of what they judge and must be checked
against people [4]. Defining criteria while looking at many outputs helped
developers find more problems (EvalLM) [5].
→ *Judging starts as one click on an output ("right / wrong, what should
it have said?"). Rules are written next to graded examples, never on an
empty form. An AI judge always shows how often it agrees with the person's
own grades, and with what range, before it's trusted.*

**R4. Many outputs at once need help to compare.** Showing many LLM
responses together, with their likenesses and differences, made tasks
doable that people had given up on [6]. ChainForge's grid of prompts ×
models × inputs let a range of people test their own hypotheses [7].
Interfaces built around objects you can copy, recombine and compare side
by side helped people experiment [8].
→ *The table view: every pipeline is also a spreadsheet. Rows are
examples; columns are the steps' outputs; versions sit side by side.*

**R5. Visual languages match the problem well but resist change and hide
dependencies; secondary notation matters.** The cognitive dimensions
framework found dataflow languages good at "closeness of mapping", but
warned about *viscosity* (hard to make a local change) and the value of
*secondary notation* (meaning carried by layout and colour) [9]. Scaling
visual languages to large programs is the classic open problem [10].
→ *Nobody drags wires to build logic. Layout is automatic, from the code.
Big programs zoom: a module is one box until you open it. Groups and notes
are allowed, as comments in the code.*

**R6. Blocks help beginners learn and stay interested, but text feels
"real".** In a 5-week high-school study, students with blocks learned more
and were more interested in more computing; students with text felt their
work was closer to what professionals do [11]. Hybrid tools are a middle
path [12].
→ *The picture and the code are always one click apart. Every button shows
the code it wrote ("show me the code"). A teen sees real Python grow from
their first click. Moving on to code isn't a jump to a different tool.*

**R7. Low floor, high ceiling, wide walls.** Tools for creative thinking
should be easy to start, have no limit to how far you can go, and support
many kinds of projects [13]. In Scratch, giving learners more powerful data
structures *caused* more use of them: wide walls increase engagement [14].
Scratch is built for self-directed learning through tinkering and sharing
with peers [15].
→ *The beginner and the professional use the same tool, with more of it
showing as needed. Starters span many kinds of projects (games, school,
art, data, agents), not one template.*

**R8. People invest attention only when the payoff is clear.** The
attention investment model explains first steps in programming as a
cost–benefit–risk judgment [16]. LLM tools for end users need to show *what
the system understood* in predictable words [17].
→ *Each new idea (types, examples, checks, versions, judges, optimizing,
baking, serving) shows up at the moment it pays off, with its cost said up
front ("this runs 20 examples, about 0.4¢, 15 s").*

**R9. Remixing spreads ideas but can reduce originality.** In the Scratch
community, works that are moderately complex and by prominent authors get
remixed more, and remixes are less original [18].
→ *"See inside" and "Remix" on everything shared, but starters ask "what
will you change?" and show what the remix changed (a diff), so remixing is
the start of making something, not a copy.*

**R10. Exploring and explaining pull against each other.** Notebooks are
personal, exploratory and messy; analysts use other media to share [19];
messes pile up [20]. Data science teams work with many stakeholders and
tools [21]; explanations in practice mostly serve engineers, not the people
affected [22].
→ *A separate view for explaining (Story) that is built from the same
records, so it never goes stale and costs no extra work to make.*

**R11. Agent traces are long; developers need overview, rewind and
edit.** Developers of multi-agent systems struggled to review long
conversations and wanted to edit and reset messages; resetting and
replaying was a key steering strategy (AGDebugger, 14 people) [23].
Visual summaries of agent behaviour help analysis [24]. Automatically
finding the failing agent and step is still unreliable: the best method
found the responsible agent 53.5% of the time and the decisive step 14.2%
[25].
→ *Rewind and edit any turn, then replay. "Where did it go wrong" is an
**experiment** the person can run (fix a step, replay downstream, see if
the end becomes right), not a guess from another model.*

**R12. Many runs, one picture.** Process mining builds a map of the paths
actually taken from event logs [26].
→ *For agents and branching pipelines: one map of the paths taken across
all runs. Line thickness is how often; colour is how often it was right.*

**R13. People in the loop are users, not oracles.** Interactive machine
learning works when the system is designed around how people actually give
feedback [27]; Teachable Machine made training approachable to people in
201 countries [28]; children can learn ML by teaching machines [29].
Amershi et al.'s guidelines ask AI systems to make clear what they can do,
how well, and why they did what they did [30].
→ *Every step says what it does (its instruction, in its own words), how
well (measured, with range), and why this time (the exact request, on
demand).*

**R14. AI literacy is a goal in itself.** Long and Magerko list
competencies such as knowing what AI can and can't do, how it learns from
data, and the ethics of it [31].
→ *Studio teaches as you use it: a "what leaves your computer" badge on
every step that sends data to a provider; the request bytes one click
away; honest uncertainty everywhere.*

**R15. Measure creativity support, not just speed.** The Creativity Support
Index measures exploration, expressiveness, immersion, enjoyment, results
worth effort and collaboration [32].
→ *Studio's user studies use the CSI as well as task times (§10).*

## 4. The one idea: your code is the program; Studio is a lens

Every flow builder on the market stores the picture as the program. That is
the source of lock-in, and it is why their "export to code" is never
faithful. Studio does the opposite:

```
           your code (Python · TypeScript · R · Julia)        ← the only source of truth
                │  read by functai.check (no running)          (a file you own, in git)
                ▼
  the program's map ──────── edits come back as code diffs ─────┐
                │                                               │
           runs (call log: JSONL, open contract)  ◄── ratings ──┤
                │                                               │
    Map · Table · Code · Story   (four views of the same thing, same selection)
```

What lives where (all in the user's folder, all open formats, all readable
without Studio):

| thing | where | format |
|---|---|---|
| the program | `club.py` (or `.ts`, `.R`, `.jl`) | the language itself |
| runs and ratings | FunctAI's call-log folder | `calls.md` JSONL, format 2 |
| example tables, answer keys | `examples/*.csv` or `.parquet` | plain tables (`rated` rows) |
| a saved, shippable version | `club.functai/` | `saved.md` folder |
| notes the picture adds (pinned layout, colours) | `.functai/studio.json` | optional; delete it and nothing breaks |

**The lock-in test.** Studio passes only if all seven hold, and the check
suite tests them:

1. Uninstall Studio: everything still runs, with the same answers.
2. Every Studio action has a code equivalent, and Studio shows it.
3. Every bit of state is in an open, documented format in the user's
   folder, and diffs cleanly in git.
4. Any provider or local model (lm15), any layout (lmcc).
5. Any of four languages; a saved program loads in the others.
6. Shipping needs nothing from Studio (`functai serve`, a container, a
   function on any cloud).
7. Studio itself is open source, and runs on your own machine.

## 5. Words

One vocabulary for the teen and the engineer. It must predict the
behaviour (lmcc's D-40 test: give the words and three examples to someone
new; every wrong guess is a naming bug).

| word on screen | what it is in FunctAI | why this word |
|---|---|---|
| **step** | an AI function, or a plain function a module calls | everyone knows "step"; "node" is jargon, "box" is childish |
| **pipeline** | a module (a program made of steps) | used by both data scientists and engineers |
| **agent** | a step with tools that loops | the word people already use |
| **run** | one call tree | "trace" is engineer-only |
| **example** | one row: inputs, and maybe the right answer | |
| **right / wrong** | a rating | "label" and "annotation" are jargon |
| **answer key** | the rows with known answers (`rated`) | a teacher's word everyone knows |
| **how often right** | the score, always with its range | "accuracy" shows in small print for professionals |
| **version** | v1, v2… (the fingerprint in small print) | never hashes on screen (as Chattering 74) |
| **leaves your computer** | a step that sends data to a provider | AI literacy |

## 6. Four views of one program

All four are open on the same selection: click a step in the Map, and the
Table jumps to its column, the Code to its definition, the Story to its
moment.

### 6.1 Map — what the pipeline is, and what it did

- **Drawn from the code**, by `functai.check` (static: which steps call
  which, with the branch conditions read from the code), plus the runs
  (dynamic: how often each path was taken). Automatic layered layout, left
  to right, inputs to outputs. Deterministic: the same code draws the same
  picture.
- **Each step is a card** with its name, its arrow signature
  (`message → question · idea · spam`, as Chattering 74), its model, and
  one line of status: **how often right** with its range, or "nobody has
  checked yet".
- **Wires carry real values.** Hover a wire: the last value that passed.
  Pick an example in the Table: every wire shows that example's values,
  like n8n's pinned data, but for any row.
- **Branches** (an `if` on a step's answer) are forks labelled with the
  condition from the code (`kind == "spam"`). Loops get a ↻ badge with how
  many times they ran.
- **Agents** are a step with a **tool belt** (its tools, each with its
  effects: reads, writes, sends, costs money). Possible calls are dashed;
  calls that happened are solid. Open an agent and its runs unroll as a
  timeline of turns (§6.5).
- **Zoom with meaning.** Far: the pipeline as five words. Middle: steps
  and wires. Close: a step's instruction, types, examples and the exact
  request (lmcc's layout, filled). Modules fold into one card until opened
  (R5).
- **Leaves-your-computer badges**: a small cloud icon with the provider's
  name on each step that sends data out, and a house icon on steps that
  run a local model (R14).

### 6.2 Table — every pipeline is also a spreadsheet

The most successful end-user programming tool ever made is the
spreadsheet. FunctAI is already table-native (`map`, dpyr, tidyverse,
DataFrames). So:

- **Rows** are examples (typed in, pasted, uploaded, or taken from real
  runs). **Columns** are the inputs, then each step's output, then the
  final answer, then the right answer when known.
- **Run a column** (one step on all rows), a row (one example through the
  whole pipeline), or everything. Before it runs: how many calls, about
  how much money, about how long (R8). Cached results come back at once
  and cost nothing (the reply cache, `replies.md`).
- **Judge in place**: a cell gets ✓ / ✗; ✗ asks "what should it have
  said?" with the allowed answers as buttons (Chattering 74's flow, now on
  every step, not just the final answer).
- **Versions side by side**: the column of v1 next to v2, with the rows
  where they differ first, and the paired comparison ("v2 fixed 6, broke
  1; v2 is better, likely by 4 to 15 points").
- **Scale** is the same table: 30 rows in a browser, 300,000 rows run by
  the user's own Python/R/Julia process through dpyr / DuckDB, with
  progress, resume and cache (FunctAI issue #5 is a prerequisite), the
  table showing a sample and the totals.

### 6.3 Code — the real file, in your language

- The program's actual file, in an editor, with the selected step
  highlighted. Edits in the Map or Table appear here first, as a diff to
  accept (never a silent rewrite).
- **Structured edits** the picture can make: add a step, rename it, change
  its instruction (docstring), its answer type or its choices, its model;
  add an example; send one step's output into another step's input (writes
  the call line in the module); add a branch on a step's answer. Python
  first (libcst keeps comments and formatting), TypeScript next (ts-morph).
  Anything else: "edit this in code", which is always possible.
- **Language toggle**: show this pipeline in Python, TypeScript, R or
  Julia. AI functions translate exactly (the contract: `saved.md`, loading
  in another language); plain code around them is translated by a model
  and marked "translated, check it".
- **Show me the code** on every button in Studio, including judging,
  comparing, optimizing, baking and serving: the 1–3 lines that do it
  outside Studio (R6, and the lock-in test §4.2).

### 6.4 Story — how it works, for people who don't read code

- Pick an example (or let Studio pick a typical one, one it got wrong, and
  one where it was unsure). Story walks it through the pipeline **one step
  at a time**, on the real recorded values: what came in, what the step was
  asked to do (its instruction, in its words), what it answered, and how
  often that step is right.
- Then the whole picture: how often the pipeline is right overall, where
  it fails most (the step that most often makes the end wrong, from
  "fix-this-step" experiments, §7), what it costs per use, which steps send
  data to which companies.
- **Share as one file**: a self-contained HTML page with the chosen
  examples embedded (no server, no account, opens in any browser, works in
  ten years). Before sharing: a check of which values it contains (names,
  emails), with the option to blur them. Or a live link, when Studio is
  hosted (Chattering).
- **People can judge from the Story**: a domain expert who opens it can
  mark answers right or wrong; their ratings come back as ordinary rating
  records (by their name). Experts are often the best judges (PromptHive
  brought subject experts into prompt work [33]).

### 6.5 Runs — for agents and long conversations

- A **timeline** of one run: turns, model calls, tool calls with their
  approvals (`tools.md`), plugin changes (recorded as data, `plugins.md`),
  streamed as they happen (`streaming.md`).
- **Rewind and edit**: change a turn's message or a tool's result, and
  replay from there as a new branch of the run (conversations already
  branch; cached steps cost nothing). The original stays (R11).
- **Paths map**: all runs of this agent drawn as one flow from start to
  finish: which tools, in which order, how often, and how often each path
  ended right (R12). It is the answer to "what does my agent usually do?"
  that no single trace gives.
- **Inside a helper**: a step that calls other AI programs shows them
  folded (the first layer, as `design/06` leaned); open to see deeper.

## 7. The loop, and what's new in each part

**Make** → **Try** → **Judge** → **Find the weak step** → **Improve** →
**Ship** → **Watch** → back to Judge.

1. **Make.** From a sentence ("sort my club's messages into questions,
   ideas and spam, and answer the questions from our FAQ"), a model writes
   *code* (Chattering 75's `program_draft`, extended to pipelines; the way
   InstructPipe drafts pipelines from instructions [34]), shown as Map +
   Code at once. Or start from a starter, or from an existing file. Or
   remix someone's shared pipeline.
2. **Try.** On a handful of examples, never one (R2). The draft's examples
   become the first rows, but the right answers stay hidden from the
   instruction (Chattering 75's rule: a test it was handed the answer to
   proves nothing).
3. **Judge.** Click ✓ / ✗ on any value at any step. Write rules next to
   graded rows. An AI judge is proposed only once there are ~10 graded rows
   to measure it against, and shows its agreement with the person (R3).
4. **Find the weak step — the new part.** On a wrong final answer: *"Which
   step went wrong?"* Studio offers the experiment: give a step the right
   output (typed by the person, or taken from the answer key), replay only
   the steps after it, and see whether the end becomes right. Run over all
   wrong rows, this gives each step a **blame count**: "if *answer* had been
   right, 9 of 14 wrong results would have been right". It is causal (an
   intervention, not a correlation), it is cheap (only downstream steps
   rerun; the rest come from cache), and it is honest where automatic
   attribution is not yet reliable [25]. In FunctAI terms: a `before_call`
   change that gives the answer instead of asking the model, recorded as
   given by a person, so it can never be counted as a model's answer
   (contract addition, §9).
5. **Improve.** Edit the weak step's instruction, add examples, change its
   model, split it in two, or let GEPA propose changes (`opt`); or bake it
   into a small model of your own. Every change is a new version, measured
   on the same rows as the old one, with the paired range (`compare`). A
   change that doesn't beat the old one is said so plainly.
6. **Ship.** "Use it elsewhere": the 3 lines to call it from Python / TS /
   R / Julia; `functai serve` (a web address with a key); a form for people
   with no code (Chattering 75/76); a container; the saved folder to put
   anywhere. Studio never sits in the path of a shipped program.
7. **Watch.** Real calls flow into the same log; a random sample is offered
   for judging each week ("check 20"); drift shows as a change in how often
   it's right, with ranges.

## 8. Growing with the person (low floor, high ceiling)

| level | what shows up | the moment it pays off |
|---|---|---|
| 0. Play | a starter with recorded runs: explore, judge, see the Story, remix. **No model key needed** (the starter ships its call log and reply cache) | first minute |
| 1. Make one step | sentence → step; try it on 5 examples | "it works… mostly" |
| 2. Choices and types | "should the answer be one of a few?" → `Literal`; numbers, lists, yes/no | when answers come back messy |
| 3. Judge | ✓ / ✗, the answer key, "how often right" with range | when "mostly" isn't enough |
| 4. Pipelines | a second step, a branch | when one step does too much |
| 5. Find and fix | blame counts; versions compared | when the end is wrong and it's unclear why |
| 6. Agents | tools with effects and approval; runs; paths map | when a step needs to look things up or act |
| 7. Scale | the table on 100,000 rows in your own process; cost before running | real data |
| 8. Optimize and own | GEPA; bake a small model; local models | when cost or quality matters |
| 9. Ship and watch | serve, forms, containers; weekly checks | when others depend on it |

Nothing at level 0–3 needs the words module, signature, adapter, trace,
metric or fine-tuning. Professionals can open everything at once
(settings: "show everything").

## 9. What FunctAI, lmcc and lm15 must add

Mechanism in the libraries; the picture in Studio.

**FunctAI contract**

1. **`graph.md`: a program's map as data.** Steps, their kinds (AI
   function, module, plain function, tool, baked model), edges (calls, with
   the branch condition's source text), loops, the source span of each
   (file, line, column) so the picture and the code point at each other.
   Every language emits the same map for equivalent programs; cases in
   `cases/graph/`. Python's `functai.check` already finds the nodes.
2. **The signature's plain data on each version** in the call log (asked
   for by Chattering 74: the allowed answers are now read back from the
   prompt).
3. **Given answers.** A `before_call` change field `answer`: the call is
   not sent; the record says the answer was given, by whom. Needed for
   "fix this step" (§7.4) and for teaching ("what if this step said…").
4. **Estimates before running**: requests × tokens laid out by lmcc ×
   prices. lm15 has no price table today; it needs one (dated, with its
   source), or Studio shows tokens only. A price table goes stale: the
   estimate says its date.
5. **Resumable runs over big tables with progress** (issue #5), since the
   Table view at scale depends on it.
6. **A Studio server**: `functai studio [folder]`, a local web app (like
   TensorBoard, marimo or Jupyter: one command, a browser tab, nothing to
   sign up for), reading the call log and the code, and running programs in
   the user's own process. Python first; the UI speaks only the contract,
   so TS/R/Julia servers can follow.

**lmcc**: the filled request for a step without sending it (exists:
`preview`); a stable "where in the request did this value go" mapping, so
the close-up zoom can highlight an input inside the exact bytes.

**lm15**: model prices (above); a local-model path that a beginner can
install (Ollama compat exists).

**Chattering**: hosts Studio's UI inside its program pages (design 74–76
grow into this; same log, same ratings), adds the classroom/household
layer (§11), and is where live links to Stories live.

## 10. The delight check: a smart 14-year-old

### 10.1 A cognitive walkthrough (inspection, not evidence)

A cognitive walkthrough asks, at each step of a task, four questions: will
the person try to do the right thing; will they see that the action is
there; will they connect the action with what they want; will they see
that it worked [35]. I did one for this design, on paper, as a check
before any building. **It is not a user study.** It finds problems; only
real teenagers can confirm delight.

*Maya, 14. Runs a Discord server for her school's robotics club. Knows a
little Scratch, has used ChatGPT, never written Python. Wants: "a bot that
sorts messages into questions, ideas and spam, and answers the questions
from our FAQ." Her parent set up Studio on the family computer (§11).*

| step | what she does | Q1–Q4 | problem found → design change |
|---|---|---|---|
| 1 | opens Studio; sees starters ("Sort messages", "Quiz from my notes", "Is this a fair trade?", "Describe my drawing") and "Make your own" | ✓ ✓ ✓ ✓ | none; starters must cover wide walls (games, school, art), not only business tasks |
| 2 | types her sentence in "Make your own" | ✓ ✓ ✓ ? | waiting 10 s with a spinner feels broken → **stream the draft**: the Map draws step by step as the code is written |
| 3 | sees 3 steps (*kind*, *answer*, *reply*) and Python on the side | ✓ ✓ ? ✓ | "FAQ" isn't there: where does it come from? → the draft asks one question: "Paste your club's FAQ", and shows it as an input of *answer* |
| 4 | clicks **Try** | ✓ ✓ ✓ ✓ | before running: "5 examples · about 0.1¢ · 6 s". She doesn't know what 0.1¢ means relative to her limit → show it against the household's allowance ("of your 2 € this month") |
| 5 | sees a table: 5 messages, 3 columns of answers | ✓ ✓ ✓ ✓ | one says spam for "can I bring my cousin?"; she wants to say it's wrong → ✗ on the cell, then "What should it have said?" with *question · idea · spam* as buttons |
| 6 | sees "how often right: 4 of 5 (likely between 38% and 99%)" | ✓ ? ? ✓ | a range of 38–99% confuses a 14-year-old → plain words first: "**Too few examples to know.** Check 15 more to find out." The numbers stay, in small print, one tap away |
| 7 | judges 15 more (j/k, 1/2/3 keys or taps) | ✓ ✓ ✓ ✓ | dots filling green and red (Chattering 74) feel like a game: keep |
| 8 | the final reply is wrong on 3 rows; clicks **Which step went wrong?** | ✓ ✓ ? ✓ | "replay downstream" is jargon → button text: "**What if this step had been right?**", result: "2 of 3 would have been right if *kind* had been right. Start with *kind*." |
| 9 | edits *kind*'s instruction in the card ("Questions about who can come are questions") | ✓ ✓ ✓ ✓ | she must see that this changed the code → the docstring in Code flashes; a small "v2" appears |
| 10 | Studio reruns the 20 examples on v2, compares | ✓ ✓ ✓ ✓ | "v2 fixed 2, broke 0" in words, with confetti only when it's really better (not when it's within the range: honesty over cheap delight) |
| 11 | wants to show her club | ✓ ? ✓ ✓ | "Ship" is a scary word → "**Use it**": as a link with a form, in Discord (a starter bot template), or "show how it works" (Story). Story checks for names in her examples and offers to blur them |

Problems this found that no amount of expert-thinking had: money must be
shown against something she understands; ranges must be words first; the
"fix this step" experiment needs a what-if question, not a technical
name; the draft must stream.

### 10.2 What a 14-year-old cannot do, and what that forces

- **No credit card, and provider terms.** Model API terms usually require
  an adult or a parent's permission; a teen can't sign up alone. So Studio
  needs: (a) **level 0 without any model** (starters with recorded runs
  and cache; exploration, judging, Story, remix all work offline);
  (b) **a household or classroom host** (§11) where an adult connects a
  model and gives allowances; (c) **local models** on the family computer
  (lm15's Ollama compat) where the hardware allows, with honesty about
  their lower quality.
- **Privacy.** Her friends' Discord messages go to a company when a cloud
  model runs. The "leaves your computer" badge names the company on every
  step, and the first run asks once: "These messages will be sent to
  OpenAI. OK?" (R14). A teacher can restrict a class to local models.
- **Safety.** Starters are age-appropriate; provider safety filters stay
  on; nothing a minor makes is published publicly without the adult's
  setting allowing it.

### 10.3 The real test (before calling it delightful)

- **Who:** 8–12 teenagers aged 13–16 (with parents' consent), 6 data
  scientists, 6 AI engineers, 4 stakeholders who don't code.
- **Tasks:** teens: make, judge and improve a pipeline of their choosing in
  45 minutes, thinking aloud. Data scientists: run and measure a step on
  2,000 rows and explain it to a stakeholder. Engineers: find why an agent
  fails on 3 known runs and fix it. Stakeholders: from a Story alone,
  answer "how often is it right, where does it fail, what leaves the
  company?"
- **Measures:** time to the first result (target under 2 minutes from
  opening), time to the first fixed mistake (target under 15 minutes for
  teens), Creativity Support Index [32], the four stakeholder questions
  answered correctly, whether they come back unasked within a week,
  and, for professionals, whether they would put it in their real work.
- **Baselines:** ChainForge for prompt testing, n8n or Langflow for
  building, a notebook for data scientists, LangSmith/Langfuse for agent
  debugging.
- **Bar:** Studio is not called delightful until teens finish the task
  without help and rate enjoyment and "results worth effort" above the
  baselines. Synthetic personas run by a model may be used to catch
  obvious problems early; they are not evidence of delight and are never
  reported as such.

## 11. Where it lives

Three places, one UI:

1. **On your machine:** `functai studio` opens a local tab, over your
   folder. No account. This is the default and the lock-in-free path.
2. **In Chattering:** the program pages (74–76) become Studio's views;
   the same UI component, the same log, plus Chattering's people and
   permissions, live links and phones.
3. **Household and classroom:** a Chattering host where an adult connects
   models, sets allowances per person, chooses which providers are allowed
   (local only, for example), and sees what the kids made. Mechanism in
   FunctAI (keys, caps per caller as a plugin), policy in the host (the
   standing principle: no default spending caps in FunctAI; the adult sets
   them for the children they're responsible for).

The UI is a separate web package that speaks only the contract (call log,
`graph.md`, interfaces, events, the studio server's routes). It never
imports a language's FunctAI.

## 12. Not doing, on purpose

- **No graph as the source of truth.** Nobody stores a pipeline in a Studio
  file. (This is the decision that makes everything else honest.)
- **No wiring logic by dragging.** Edits are structured and become code.
- **No new runtime, no hosted-only service.** Programs run where FunctAI
  runs.
- **No chat app.** Conversations are viewed and replayed, not used here;
  Chattering is the chat app.
- **No connectors marketplace (n8n-style triggers to 400 apps) at first.**
  Tools are code; a few starter tools (web page, file, Discord, sheet).
  Triggers belong to the host or the user's own code.

## 13. Phases, each useful alone

1. **See** (no editing): `functai studio` with Map (from `graph.md`) +
   runs overlaid, Table over the call log, Code read-only, judging on any
   step. Python. *Useful at once for every existing FunctAI user.*
2. **Judge and find:** answer keys per step, versions side by side,
   "what if this step had been right?" (given answers in the contract).
3. **Shape:** structured edits writing Python code as diffs; Make from a
   sentence; starters with recorded runs (level 0).
4. **Explain:** Story, shared as one file; ratings from Story readers.
5. **Agents:** Runs timeline, rewind and edit, paths map.
6. **Scale and ship:** big tables in the user's process with cost before
   running; "Use it" (serve, form, container); weekly checks.
7. **Everywhere:** TypeScript edits, R and Julia servers; Chattering's
   household and classroom layer.
8. **The real test** (§10.3) runs after phase 3 and again after 5, and
   can stop or reshape everything after it.

## 14. Trade-offs, stated

- **Code as the only truth limits what the picture can edit.** Users can't
  draw arbitrary logic. *Why accepted:* it's what makes no lock-in real
  and the picture never wrong. The cost: some beginners will want to drag
  wires and can't; they type a sentence or a line instead.
- **Automatic layout means users can't arrange things exactly.** Pinning
  is allowed in a sidecar file. Secondary notation [9] is weaker than in
  free canvases.
- **Four languages multiply work.** Edits come to Python first, then TS;
  R and Julia get viewing first. Parity is uneven for a long time.
- **Translation of the code around AI functions is done by a model** and
  can be wrong; it is marked as such.
- **Counterfactual blame needs a correct output from a person** (or an
  answer key per step). Without one, Studio can only say "this step looks
  suspicious", which it labels as a guess.
- **Cost estimates need a price table** that goes stale; shown with its
  date, or as tokens only.
- **Level 0 without a key** means shipping recorded runs with starters:
  more files, and the recordings show one model's behaviour at one time.
- **A 14-year-old needs an adult to connect a model.** This is a real
  barrier to "start in seconds"; local models lower it only on capable
  computers.
- **The walkthrough in §10.1 is my inspection, not users.** Its findings
  are hypotheses until §10.3 runs.
- **The research is read from abstracts only.** Full papers may qualify
  some of the findings used here.
- **Scope.** This is a large product. Phase 1 is deliberately read-only so
  that value arrives before the hard part (editing code from a picture).

## 15. Decisions for Maxime

1. **Code as the only source of truth** (§4, §12): agree? It is the
   decision everything else rests on.
2. **Where first:** a local `functai studio` command (recommended: lock-in
   free, useful to every FunctAI user, and Chattering embeds the same UI),
   or straight into Chattering?
3. **The name.** "Studio" is the working name. Others: *Bench*, *Loupe*,
   *Lens*, *Workshop* (taken on this machine), *FunctAI Atlas* (after
   Chattering's mockup).
4. **Contract additions** (§9): `graph.md`, signature data per version,
   given answers (`before_call` → `answer`). Approve the direction before
   any implementation?
5. **Teens and money:** build the household/classroom layer in Chattering
   (recommended), or leave teens to local models and starters only?
6. **The real test (§10.3):** do you have access to teenagers (with their
   parents' consent) and to a few outside professionals? Without it, the
   delight claim stays unproven.

## After Maxime's first reading (2026-11-02)

His reading, in short: the general idea is right; code as the only source
of truth, agreed; the contract additions, agreed in direction. The
interface is cluttered (too much information); big outputs and many odd
cases worry him; and something is missing: **a box where you ask a model
for a new pipeline or a change, so that the person, an AI helper, FunctAI
and Studio work together in one loop that keeps getting better.**

## 16. Decided

- **Code is the only source of truth** (§4, §12). Everything below keeps
  to it: the helper's work arrives as changes to the code, never as a
  picture-only edit.
- **The three contract additions**, in direction: a program's map
  (`graph.md`), the plain interface of each version in the call log, and
  given answers. Drafted in §21; implementation waits for the questions
  in §22.

## 17. The ask box, and the loop

### 17.1 One box

One box, at the bottom of every view, always the same: *"Ask, or say what
to change…"*. On an empty Studio it is the whole screen: *"What do you want
to make?"*. It takes four kinds of request:

| kind | examples | what comes back |
|---|---|---|
| make | "sort my club's messages and answer questions from our FAQ" | a new pipeline: code, drawn as it is written |
| change | "make replies shorter", "use a local model for *kind*", "split *kind* in two", "add a step that translates to French" | a **proposal**: a change to the code, already run on examples and measured |
| ask | "why is #2 wrong?", "which step costs the most?", "what does *answer* do with an empty FAQ?" | an **answer with its evidence**: the rows and steps it rests on (links that select them), and the experiment that shows it when one can |
| do | "find 10 tricky messages", "run everything on the new file", "check 20 answers with me" | rows in the Table, a run, or the judging flow |

People use AI coding help in two modes: they *know* what they want and
want it faster, or they *don't* and want to see options [36]. So a change
request gets **one** proposal ("make replies shorter"), and an open one
("what could I try?") gets **two or three**, run side by side on the same
rows, compared as columns.

### 17.2 What every answer is

Never prose alone when a claim can be checked. Every answer from the
helper is one or more of four things Studio already knows how to show:

1. **a proposal**: a change to the code (a diff), with what it did on
   examples and how it compares with the current version;
2. **an experiment**: a what-if (§7.4), or a column of new results;
3. **an explanation** that points at rows and steps (clicking a pointer
   selects them in every view);
4. **a question back**, when something is missing (the FAQ, what "right"
   means here).

"*kind* is the problem" comes with the what-if that shows it. "The FAQ has
no fees in it" comes with the row and the line of the FAQ. This also
teaches: the answer is sometimes "this isn't an AI problem, your data is
missing a fact" (AI literacy, R14).

### 17.3 The loop

```
              says what it wants · judges ✓/✗ · keeps or drops changes
                                ┌──────────┐
                                │  person  │
                                └──────────┘
                  proposals, with ▲      │ requests, ratings
                  their evidence  │      ▼
   ┌───────────────┐   tools    ┌──────────────┐
   │   AI helper   │ ◄────────► │    Studio    │  shows everything,
   │ Studio's own, │  rows, runs│              │  one click from every action
   │ or your agent │  what-ifs  └──────────────┘
   └───────────────┘                   │
           │ writes changes            │ runs, records, measures
           ▼                           ▼
       your code  ───────────►  FunctAI: runs it exactly, keeps every call,
       (in git)                 scores with honest ranges
```

Each does what it is best at:

| who | does | never does |
|---|---|---|
| the person | says what they want and what "right" means; judges; keeps or drops each change | has to write code (but always can) |
| the AI helper | writes and changes code; proposes; runs the boring experiments; explains, with evidence | decides what is right; keeps its own changes; judges its own work |
| FunctAI | runs the program exactly; records every call, rating, change and who made it; scores with ranges | guesses |
| Studio | shows all of it; turns every action into one click and its code | stores the program |

**Why it is a virtuous circle and not just a chat:** every turn leaves
something that makes the next turn better, and all of it is plain files:

- ratings become the **answer key**, so measurements get sharper;
- what-ifs become **blame counts**, so the next fix aims better;
- kept changes become **versions with measured gains**, so progress is
  real, not felt;
- the helper's own calls are FunctAI calls, judged in Studio like any
  program (a teen can mark the helper's answer wrong), so the helper
  gets better *for this person*, from their own judgements.

### 17.4 Rules that keep the loop honest

A loop with an AI in it can also spin the wrong way: the AI grades its own
work, fits the few examples it has seen, and the person clicks "keep" on
everything. Eight rules:

1. **Only people say what is right.** The helper may *guess* verdicts (an
   AI judge, as in §7.3); a guess is a call of a judge program, never a
   rating, shown as a guess, with how often it agrees with the person's
   own ratings (R3). Models favour their own outputs [37][38], so the
   helper never judges a change it wrote.
2. **The helper never keeps its own change.** Every change is a proposal;
   the person keeps it or not. (A person may let their own coding agent
   commit from their own terminal; Studio still shows who did what.)
3. **"Better" is measured on rows the helper did not see.** When the
   helper writes a proposal, it is shown a random part of the answer key;
   the rest is held back, and the proposal is measured there, paired with
   the current version, with the range. The split is recorded with the
   proposal. Too few held-back rows, and it says so in words: "better
   here, but 5 is too few to be sure: check 10 more".
4. **Think first, where the number is made.** In the random check that
   produces the official "how often right", the helper's guess stays
   hidden until the person has answered. Making people think first
   reduces over-reliance on AI suggestions, though people like it less
   [39]. Elsewhere, guesses show (speed). Trade-off stated in §22.
5. **See it run before keeping it.** A proposal is shown running on
   examples, not only as code: seeing runtime values made AI-written code
   cheaper to check and reduced both over- and under-reliance [40].
   Novices using AI code tools struggle to judge what it gave them [43],
   so each proposal also says what it changed in one plain sentence.
6. **Cost before, always.** A proposal that runs experiments says calls,
   money and time first. A session has a budget the person sets.
7. **Who did what is always visible.** Each change, rating, run and
   proposal names its author: the person, Studio's helper, or another
   agent by name.
8. **Offer, don't interrupt** (mixed initiative [41]). The helper offers
   at most three suggestions under the box, only when it has evidence
   ("Why is #2 wrong?" after #2 is judged wrong). It never pops up.

### 17.5 The helper is a FunctAI program, and so is your agent

- **Studio's helper** is an agent written with FunctAI: tools with effects
  (`tools.md`): *read* rows, runs and code; *run* experiments (costs
  money, asks within the budget); *propose* changes (writes nothing until
  kept). Its calls go to the log. It uses the model the person configured
  (lm15: any provider, or a local model).
- **Bring your own agent.** `functai studio` offers the same tools by
  command line and MCP: `functai studio rows`, `… what-if`, `… propose
  <patch>`, `… ask "…"`. Claude Code, Pi or Codex in a terminal read the
  same runs and ratings and propose changes, which appear in Studio's
  thread under their name. The other way too: "continue this in my
  agent" hands a proposal and its evidence to the person's coding agent.
  In Chattering, which already runs those agents live (Chattering design
  91), the thread *is* a conversation.
- **Files, so nothing locks in:** proposals are patch files in
  `.functai/proposals/` (git can apply them; delete them and nothing
  breaks); the thread is `.functai/studio/thread.jsonl`.

### 17.6 Making a pipeline from a sentence

1. The person writes what they want (or picks a starter, which fills the
   box).
2. The helper writes the code, streamed: the Map draws step by step as
   the code appears (the walkthrough's "a spinner feels broken", §10.1).
3. It asks for what is missing (the FAQ), never invents it.
4. It writes typical example inputs, never their right answers (Chattering
   75's rule: a test it was handed the answer to proves nothing).
5. Try (cost first) → the person judges → the loop of §17.3.

## 18. Calm by default

What went wrong in the first mockup: the same score was shown in four
places (top bar, Map card, side card, Table header); three side panels
were always open; every step card had five lines; banners and key hints
everywhere. About 40 separate things on the first screen.

Rules:

1. **One focus per screen:** the pipeline, with one example flowing
   through it. Everything else on demand.
2. **One place per fact.** A step's score lives in its drawer; the Map
   shows only a coloured dot; the top bar shows one status in words.
3. **Layers** (multi-layer interfaces: start with a small set of features,
   move up when needed [42]). First layer: Map, the example, the ask box,
   Try. The Table's numbers appear after a first run; percentages after
   ~10 judgements; Story when there is something to tell; Runs only for
   agents. Code stays one click away from the start (R6): the one
   exception.
4. **One drawer at a time**, opened by a click, closed with Esc.
5. **Words before numbers, numbers before ranges** (ranges one click
   away).
6. **Colour means one thing each:** grey not judged; green and red only a
   person's judgement; purple failed (not the same as wrong); blue the
   selection.
7. **Target:** at most ~12 separate things on the first screen; checked
   in the real test (§10.3) by first-click success.

## 19. Big outputs and odd cases

| case | what Studio shows |
|---|---|
| long text (an essay, 5,000 words) | Map: the first line and "text · 5,000 words". Table: two lines and the length. Drawer: a reader with scrolling and search. Versions: a word diff, and for many rows, what differs across them (sensemaking at scale [6]) |
| structured outputs (a record, a list of 300 items, nested) | Map: the shape in words ("list · 312 items"). Drawer: a tree folded at depth two; lists as a small table; judging per field (a rating's `outputs`) |
| pictures, sounds, documents, video | thumbnails, a play button, page thumbnails; never full size inline; big files by reference, as the log keeps them (`calls.md`, *Content*) |
| streaming | the value fills in as it arrives, marked unfinished, with Stop |
| failures: refused, unreadable reply, timeout, provider down, wrong kind of input | purple **failed**, the reason in words and the fix when known; counted apart from wrong; never inside "how often right"; "how often it fails" shown on its own |
| a step that reads text given a picture | failed: "*kind* reads text; this message is a picture"; the helper offers to add a picture input (FunctAI takes pictures in every language) |
| same input, different answers | "Try 3 times" shows agreement ("same answer 3 of 3", or the spread) |
| many rows (100,000) | the Table shows a sample and totals; runs in the person's own process, resumable (issue #5); filters by state; measurement only from random samples (a rating's `sample`) |
| many steps, deep modules | modules fold into one card; zoom; "show me where the price is computed" in the ask box finds and selects it |
| loops (an agent's 40 turns, retries) and fan-out (`map` over 300 items) | one card with "↻ 40" or "× 300"; open it for a timeline or small multiples; the paths map for agents (§6.5) |
| code the reader cannot follow (functions passed in, `getattr`, `eval`) | steps seen only in runs, drawn dashed: "seen in runs, not found in the code" (the graph's `unseen`, §21) |
| code that does not run (syntax or import error) | the last good drawing, greyed, the error in words with a link to its line; the helper offers a fix |
| code changed elsewhere (an editor, `git checkout`, a coding agent) | the file is watched; the Map redraws; runs and ratings stay with their versions (old runs labelled v1 when the code is v3); a proposal made against an older version is redone or marked out of date |
| two writers at once | proposals are patches against a version, never applied silently; if the file changed: "the file changed since; redo the proposal?" |
| private data in values | the person's own machine shows values as they are; sharing (Story) finds names, e-mails and keys first and offers to hide them |
| slow steps (a minute each) | progress per row; the Table fills as results come; the tab can close: runs belong to FunctAI, not to the page |
| cost | the session's budget; above it, Studio asks; cached results are free (the reply cache) |
| nothing yet | no runs: "Try it on 5 examples"; no judgements: "Nobody has checked yet"; no examples: the helper offers typical inputs, never right answers |

## 20. The mockup, second version

`design/16-studio-mockup/index.html` (made-up data; nothing runs; your
text is not read by a model: the mockup always builds the club example).
It opens on the empty Studio ("What do you want to make?"), builds the
pipeline from the sentence, asks for the FAQ, runs, and then shows:

- the calm Map: one example flowing, one dot per step, long values
  shortened with their length, the failed example in purple;
- the ask box with suggestions that change with what is selected:
  *Why is #2 wrong?* (an explanation with its what-if), *Fix the step that
  fails most* (a proposal with a diff, measured on held-back rows, Keep or
  Discard), *Make replies shorter*, *What could I try?* (three
  alternatives run side by side), *Why is #5 wrong?* (the FAQ is missing a
  fact: not an AI problem);
- who did what on every message in the thread;
- Code showing the lines a kept change added; Table and Story calmer.

## 21. The contract additions, drafted

Written here first; they move into `contract/` with every language's
implementation and their cases, in one commit (this repository's rule).

### 21.1 A program's map (`graph.md`, format 1)

```json
{"functai_graph": 1,
 "program": "club:club_inbox",
 "flows": true,
 "nodes": [
   {"id": "club:club_inbox", "kind": "module", "interface": "sha256:…",
    "source": {"file": "club.py", "line": 24, "end_line": 31}},
   {"id": "club:kind", "kind": "ai", "signature": "sha256:…", "interface": "sha256:…",
    "source": {"file": "club.py", "line": 7, "end_line": 9}},
   {"id": "club:FAQ", "kind": "constant", "source": {"file": "club.py", "line": 4, "end_line": 4}}],
 "edges": [
   {"from": "club:club_inbox", "to": "club:kind", "kind": "calls", "line": 25},
   {"from": "club:kind", "to": "club:answer", "kind": "flows", "when": "k == \"question\"", "line": 28},
   {"from": "club:FAQ", "to": "club:answer", "kind": "flows", "line": 28}],
 "unseen": [{"in": "club:club_inbox", "line": 30, "why": "calls a function passed in as an argument"}]}
```

- **Nodes** by kind: `module`, `ai`, `code` (a plain function of the
  person's), `tool`, `baked`, `constant`. Ids are `module:name`, as
  `functai.check` names them. `source` points into the file, so the
  picture and the code select each other.
- **Edges** by kind: `calls` (A's code calls B), `flows` (a value B gives
  reaches C, as read from the code), `tool` (an AI function offers B as a
  tool). `when` is the source text of the condition that guards the edge,
  as written: never evaluated, never translated. `loop: true` inside a
  loop; `each: true` for one call per item (fan-out).
- **`flows: false`** when an implementation cannot read flows (likely R
  and Julia at first): it gives `calls` edges only, and Studio draws the
  calls in order without value wires.
- **Static only.** How often each edge was taken comes from the call log,
  never stored in the map.
- **Across languages:** equivalent programs give the same nodes, kinds
  and edges; `when`, `source` and `line` are each language's own text.
  Cases compare all but those.
- Order: nodes and edges in source order, so the same code gives the same
  bytes, and the same picture.

### 21.2 A version's interface, in the call log (`calls.md`)

The log names a program's interface only by its hash today, so readers
(Chattering, Studio) guess the allowed answers back from the prompt
(Chattering design 74, trade-offs). A new line, once per writer file and
version, before that version's first call in the file:

```json
{"functai_program": 1, "version": "sha256:…", "name": "kind", "module": "club",
 "kind": "ai", "interface": {"description": "…", "inputs": […], "outputs": […]},
 "instruction": "What kind of message did the club get?"}
```

- `interface` is the plain interface of `programs.md`; `instruction`, an
  AI function's only (its whole instruction, as laid out).
- It is the program's definition, never a call's values. Open question in
  §22: whether `content: false` must leave out `instruction`, since a
  person could put private text in an instruction or in worked examples.
- Readers take the interface from here, and fall back to today's rule for
  logs written before.

### 21.3 Given answers (`plugins.md`, `calls.md`)

For "what if this step had been right?" (§7.4) and for teaching ("what if
it said *idea*?").

- **`before_call` gains a change field `answer`**: the AI function's
  outputs, by name, as JSON. The call is not sent; its outputs are those
  values (checked against its interface, as any answer is).
- **The call record says so:** `given: {"by": person}` (or `account`, as
  ratings do), and no exchanges, no tokens.
- **Given answers never count:** a given call is never a row of `rated`
  and never scored; rating it is refused (`rating-given`). A call whose
  tree holds a given call (the module around a what-if) is marked
  `given_inside: true` and is left out of `rated` and scores the same way.
  The real model calls downstream stay ordinary calls: they are honest
  answers to the inputs they got.
- Python sketch: `with functai.given(kind="question"): club_inbox(msg)`.
- Cases: a given call has no exchange; `rated` leaves out it and every
  call above it; rating it is refused; versions do not change.

## 22. Trade-offs of this round, and questions

Trade-offs:

- **Thinking first** (rule 4) is liked less [39]; it is used only in the
  random check that makes the official number.
- **Held-back rows** mean the helper sees fewer examples, so its first
  proposals are worse. Under ~10 judged rows a proposal cannot be measured
  at all; Studio says so, and the person keeps it on what they saw running.
- **The helper costs money and needs a model.** With no key (level 0), the
  box works only on starters' recorded runs, or a local model, which
  writes worse code.
- **One more folder**, `.functai/` (proposals, thread, pinned layout);
  removable.
- **Value wires need `flows`**, which some languages will not read for a
  while; their Map shows calls in order, without values on wires.
- **`given_inside` leaves whole what-if trees out of scores**: experiments
  never inflate accuracy, and never count as evidence for the module
  either.
- **Calm hides things:** a professional must click once more to see
  ranges and numbers; "show everything" in settings brings them back.
- **The mockup is scripted**: it shows the flow, not whether a model
  makes such good proposals; that is tested only with a real helper.

Questions for Maxime:

1. **The helper's model:** whatever the person configured, or a
   recommended default that Studio names?
2. **Instruction text in the log with `content: false`** (§21.2): keep it
   (it is code) or leave it out (it may hold private text)?
3. **An "autopilot"** where Studio's helper keeps its own changes when
   measured better on held-back rows: never, or as a setting off by
   default?
4. **Implementing the contract additions:** all four languages in one
   commit (the repository's rule), or Python and TypeScript first, with R
   and Julia reading the new lines and giving `flows: false`?

## After Maxime's second reading (2026-11-02)

- **The helper uses the AI the person set up.** And the person chooses the
  AI of **every step** (§24).
- **Contract additions go into all four languages, in one commit.**
- **Not decided yet:** instructions in the log with `content: false`;
  an autopilot. Both parked.
- **Order changed:** a helper inside Studio, yes, but later. First, the
  experience must work very well with **any coding agent** (Claude Code,
  Pi, Codex, Cursor…), through a **Studio command line**. People already
  ask their coding agent for things; they will ask it for pipelines. The
  agent builds and tests with the command line; the person works in the
  dashboard. The model is `workshop`: the agent writes `design.py`, runs
  `workshop verify`, and the person opens the result in FreeCAD or
  Blender, and it just works.

## 23. Agent first: the split of work

```
   person ──asks──► coding agent (theirs: Claude Code, Pi, Codex…, their model)
     │                    │ writes code        │ runs `functai studio …`
     │                    ▼                    ▼
     │               club.py  ◄──────►  FunctAI: runs, records, measures
     │               examples/            .functai/  (runs, ratings, report)
     │                                          │
     └──── looks, judges, decides ◄── Studio dashboard (same files)
```

| who | where | does |
|---|---|---|
| the person | their coding agent's chat, and the dashboard | says what they want; judges answers in the dashboard; keeps or drops changes (in their agent, as they already do for code) |
| the coding agent | the terminal | writes the pipeline and examples; runs, inspects, finds the weak step, measures changes, checks requirements; hands the dashboard over |
| FunctAI | the person's machine | runs exactly, records every call, rating and change, measures with ranges |
| the dashboard | a browser tab (or Chattering) | shows it all; judging; choosing each step's AI; Story for others |

What this changes from §17: the "helper" is the person's own coding agent,
on the model they already chose for it. Studio's job is to make that agent
excellent at pipelines (the command line, its skill) and to make the
person's side delightful (the dashboard). The ask box of §17 becomes, for
now, the person's own agent chat; the dashboard's box comes later and uses
the same commands.

**Why this order is right:** it meets people where they already work; it
needs no model, keys or billing inside Studio; each agent improves on its
own maker's schedule; and the command line is what a built-in helper will
need anyway, so nothing is thrown away.

## 24. Choosing the AI of each step

- **In the code** (the truth): each AI function names its model,
  `@ai(lm="claude-haiku-4.5")`, or takes the one `configure` sets.
- **On each step's card** the dashboard shows its model, and a little house
  for a model on this computer or a cloud with the company's name. Choosing
  another one writes `lm=` into the code, as a change the person sees.
- **Trying before choosing**, with no change to the code:
  `functai studio run club.py --lm kind=gemini-2.5-flash` (command line) or
  "Try with…" (dashboard). The call log already records the model of every
  call, and a model change is not a new version (`calls.md`, *Versions*),
  so the same rows answered by two models compare directly: right, cost,
  time, side by side.
- **Which models:** whatever lm15 reaches with the person's keys and
  sign-ins, local models included. Studio never adds a key of its own.
- **In the mockup:** click a step, pick another AI under "Its AI": it is
  tried on the judged examples (right, cost, whether anything left the
  computer) before "Use it for *kind*" writes it into the code.
- **The helper's AI:** the person's coding agent, on whatever model they
  chose for it. Later, the built-in helper uses the person's configured
  model, and may use a different one per job (writing code, explaining).

## 25. The command line: `functai studio`

One namespace, so an agent learns one manual (`functai studio --help`) and
nothing collides with today's `functai verify`. Built for agents first:

- never asks a question (an agent can't answer one): what it needs is a
  flag, and when it is missing it stops and says which;
- text for reading by default, `--json` for code, the same facts in both;
- long values shortened with their length (`"Hello and thank you so…" · 84
  words`), whole with `--full`, so a 5,000-word answer never floods an
  agent's context;
- **cost before running**: every command that calls a model says calls,
  money and time first; over `--budget` (default from the project, else
  $0.50) it stops with exit 3 and says the flag to add;
- exit codes: 0 done, 1 a requirement failed, 2 an error, 3 over budget;
- examples, steps and rows named the same way everywhere: `#5`,
  `#5/answer`, `kind`.

| command | does | built on |
|---|---|---|
| `functai studio doctor` | which languages, models and keys are here; where runs are written | lm15, config |
| `functai studio map club.py[:club_inbox]` | the steps and how they connect, from the code (no running); problems it could not follow | `functai.check`, `graph.md` (§21.1) |
| `functai studio run club.py --on examples/club_inbox.csv [--lm step=model] [--limit N] [--estimate]` | runs examples through the pipeline; resumes, uses the cache; prints right / wrong / failed / not judged per step, cost, time | `map`, the call log, the reply cache, issue #5 |
| `functai studio show club.py [#5] [--step kind] [--wrong] [--failed] [--full]` | what each step said, row by row, with people's judgements | `calls`, `rated` |
| `functai studio open [club.py]` | starts the dashboard on this project and prints its address (in Chattering: the program's page) | §27 |
| `functai studio rate #2/kind wrong --answer question --by maxime` | records a judgement **the person gave** (in the dashboard, or said in the agent's chat) | `functai.rate`; see rule below |
| `functai studio judge club.py --step reply --rule "one sentence, under 20 words"` | an AI judge's guesses, as judge calls, never ratings; with how often it agrees with people's judgements | judges, R3 |
| `functai studio why club.py [#2]` | for a wrong example: gives each step its right answer in turn, reruns what comes after, says which step's fix makes the end right | given answers (§21.3) |
| `functai studio blame club.py` | the same over every wrong example: which step to fix first | `why` |
| `functai studio compare club.py --before HEAD` (or `--before old.py`, `--lm kind=…`) | the current code against an earlier version, paired, on the **held-back** judged examples, with the range, in words first | `evaluate`, `compare`, `scores.md` |
| `functai studio check club.py` | the pipeline's requirements, each PASS, FAIL or NOT ENOUGH CHECKED; writes `REPORT.md` and the one-file page; exit 1 on a FAIL | §26 |
| `functai studio page club.py` | the one-file page (map, table, story) to hand to anyone | §27 |
| `functai studio propose club.py --patch fix.patch --note "…"` | leaves a change for the person to decide in the dashboard, measured, instead of editing the file | §17.5 |

**Two rules the command line carries, because an agent will be tempted:**

1. **Only people say what is right.** `rate` records what a person said,
   with their name; the agent's own opinion goes through `judge`. A
   relayed rating names the agent that relayed it (contract addition,
   §26). The skill says it in plain words; the dashboard is where people
   judge, so relaying is the exception.
2. **Held-back examples.** The first time examples are judged, the command
   line keeps a random part aside (`.functai/studio/split.json`). `show`
   hides their right answers unless `--held-back` is given (and then says
   so in every later `compare`); `compare` measures only on them. This
   stops an agent from fitting the examples by accident. It cannot stop
   one that reads the files on purpose: the files are plain, by design.

## 26. Requirements, and what the contract gains

**Requirements**, like `workshop`'s `d.require`: written in the code, next
to the program, because they are part of what it is for:

```python
@module(require=["right >= 85%", "kind: right >= 90%", "cost per call <= 0.1c", "failed <= 5%"])
def club_inbox(message: str) -> str: ...
```

A rate requirement has **three answers**, never two: **PASS** when the
whole 95% range is above the bar, **FAIL** when the whole range is below,
**NOT ENOUGH CHECKED** otherwise, with how many more judged examples would
likely settle it. `check` never says a pipeline is good on a hunch, and
the skill tells agents: no claim of "it works" without a PASS; otherwise
say "not checked". (Requirement syntax: a design question for its own
note; the three answers are the decision.)

**The contract**, all four languages in one commit:

1. `graph.md`: the program's map (§21.1).
2. The plain interface of each version in the log (§21.2).
3. Given answers (§21.3).
4. **A language's runner** (`runner.md`), new: the command line and the
   dashboard are written once and drive any language through it, the way
   an editor drives any language through one protocol. A runner is a
   process speaking JSON lines: `map` a program, `run` rows (with model
   overrides and given answers), `describe` its interface, stream events
   as `streaming.md` already defines them. Python's runs in its own
   process; TypeScript's, R's and Julia's are a small command in each
   package. Without it, Studio would be four programs, or Python-only.
5. **`via` on a rating**: the program or agent that recorded a person's
   judgement for them (`"claude-code"`, `"functai-studio"`). Optional;
   absent when the person rated directly.
6. **Requirements** on a program, as data (the strings above, and their
   results in the report), so that every language checks them the same
   way.

## 27. The person's side: the dashboard and the page

- **`functai studio open`** starts a local server over the project folder
  and prints the address (like Jupyter, marimo or TensorBoard: one command,
  one browser tab, no account). The agent tells the person: "I ran 9
  messages. Open http://localhost:7781 to check the answers; it takes two
  minutes." In Chattering, the agent shows it in the artifact panel, or
  the program's page is the dashboard.
- **Live with the agent:** the dashboard watches the files, so when the
  agent edits `club.py` or runs examples, the Map and Table change in
  front of the person, and every change and run says who made it.
- **The one-file page** (`page`, and written by `check`): the map, the
  table, the Story and the requirements in one HTML file with the examples
  inside. Opens anywhere, needs nothing, works in ten years: the
  equivalent of the `.step` file anyone's CAD program opens.
- **Files in the project** (all plain, all optional to keep):

  | path | what |
  |---|---|
  | `club.py` | the program (the truth) |
  | `examples/club_inbox.csv` | examples, people can open them in a spreadsheet |
  | `.functai/calls/` | this project's runs and ratings (the call log; Studio also reads the person's usual log folder) |
  | `.functai/studio/` | the held-back split, proposals, the last report |
  | `REPORT.md`, `club_inbox.html` | what `check` wrote |

## 28. The agent's method (the skill)

FunctAI already ships a skill for coding agents (`python/functai/skill/`,
installed by `functai skill --install`). It gains a section, in the
`workshop` way:

1. **Write the job as requirements first** (what right means, how often,
   at what cost), asking the person only what cannot be assumed.
2. **Say the plan before building**: the steps, in plain words, named
   after the person's request, and ask. Then write the pipeline as code,
   the smallest that could work, and check it with `map`.
3. **Write examples without their answers** (never invent right
   answers): typical, hard and odd ones, plus the person's real data when
   there is some.
4. **`run --estimate`, then `run`.** Read `show` for the failed and odd
   rows.
5. **Hand over:** `open`, and ask the person to judge ("two minutes, 20
   answers"). Do not judge for them.
6. **`blame`, fix the step at fault, `compare`** on held-back examples.
   Say the result in words, with its uncertainty.
7. **`check`**; report PASS / FAIL / NOT ENOUGH CHECKED for each
   requirement, and hand over the page. Never say "it works" without a
   PASS.

A short example of what the person sees in their agent:

```
you    Make me something that sorts my robotics club's Discord messages and answers questions from our FAQ.
agent  I wrote club.py (3 steps) and 9 example messages. Before running: 21 calls, about 0.06¢.
       Ran them: 1 failed (a photo; kind reads text only). Open http://localhost:7781 and mark the
       answers right or wrong (about 2 minutes); I can't know what's right for your club.
you    done
agent  4 right, 4 wrong. `blame`: kind causes 2 of the 4 (it calls "can I bring my cousin?" spam).
       I changed its instruction. On 5 judged messages I hadn't looked at: 3 right before, 5 after.
       Better, but 5 is too few to be sure. Requirement "right ≥ 85%": NOT ENOUGH CHECKED
       (about 20 more judged messages would settle it). The #5 fee question fails because your
       FAQ has no fees: that's a missing fact, not an AI problem.
```

## 29. The order now (replaces §13)

1. **The contract**, all four languages, one commit: §26 items 1 to 6.
2. **The command line and the skill**, Python first in time but driving
   every language through its runner; tested by giving real coding agents
   (Claude Code, Pi, Codex) real tasks with and without it, graded like
   `evals/skill/`.
3. **The dashboard**, read and judge first: Map, Table, Code, Story; each
   step's AI; live while the agent works; then the one-file page.
4. **The real test** (§10.3), with the person working through their own
   coding agent.
5. **Later:** the dashboard's own ask box and built-in helper, using the
   same commands; the household and classroom layer in Chattering.

## 30. Trade-offs of this round

- **Agent first leaves out people with no coding agent**, which includes
  most 14-year-olds today. They get the dashboard and starters until the
  built-in helper comes. This is the biggest cost of the new order, and it
  is why the helper stays planned rather than dropped.
- **Coding agents vary.** The experience is only as good as the agent's
  skill-following; Studio controls the commands and the skill, not the
  agent. Measured in step 2 of §29.
- **A runner protocol is one more thing every language must do**, in one
  commit with the rest. It is what keeps Studio one program instead of
  four.
- **Held-back examples guard against accidents, not cheating.**
- **A third answer for requirements** ("not enough checked") will feel
  frustrating early; it is the honest answer with few judged examples.
- **Choosing a step's model from the dashboard writes to the code**: the
  person sees a change to their file for what feels like a setting. The
  alternative (a settings file beside the code) would make the code no
  longer say how it runs.

## 31. Questions for Maxime

1. **Where the command line and dashboard live:** inside the `functai`
   Python package (`uvx functai studio …` works anywhere Python or uv
   is; the dashboard ships as built files inside it), or a separate
   package, perhaps in TypeScript, closer to Chattering? I lean Python:
   `functai.check` and `evaluate` are there, `uvx` makes it one command
   for any agent, and the runner protocol keeps other languages first
   class.
2. **`via` on ratings** (§26, item 5): agree?
3. **This project's runs in `.functai/calls/`** by default when run
   through `functai studio`, so a project carries its own evidence (and is
   git-ignored by default)?
4. **Requirements in the code** (`@module(require=[…])`), with the three
   answers: agree with the idea, before the syntax is designed?

## After a simulated usability run (2026-11-02)

Maxime ran the second mockup through a usability tool with three
**simulated** visitors (a 34-year-old first-time visitor, a 22-year-old
student, a 68-year-old retiree) and shared its five insights. Simulated
visitors catch obvious problems; they are not evidence of delight (§10.3).
Two findings were plain bugs in the mockup; most of the rest were right.

| finding | done in the mockup | what it means for the real Studio |
|---|---|---|
| The first screen asks "What do you want to make?" before saying what this is | a line under the heading (what it builds, that you own the code, that it shows how often it's right); "What you get" (steps → ✓✗ → "7 of 9 right"); starters labelled *For your club / school / games / art* under "Things people make"; "See a finished example" as a button; a trust line (free and open source, no account, undo, cost before every run) | the dashboard's empty state, and the agent's first words: say what will be made before making it |
| **It built the club example whatever you asked, and never said so** (bug) | requests it can't build are answered on the start page: "This mockup can only build one example…", with the plan a real helper would propose; nothing opens behind it. The mockup banner is now a visible strip | **a plan before building**, always: the agent (or helper) states the steps in plain words and asks; the pipeline is named after the request. Added to the skill's method (§28, step 2) |
| Price is hidden; students can't compare | the trust line says who is paid (the AI company you connect) and how much (usually under a cent per run, shown first); "Spent: 0.09¢" in the header, explained on hover | spending per session, in the header, from the call log's tokens and the price table (§9, item 4) |
| Developer words everywhere; the plain view is the fourth tab | "Step by step" (was Story) is the second tab; "Club inbox · Version 2" instead of `club_inbox v2`; "AI steps" instead of "calls"; a change is a plain sentence with "See the exact change" folded away; "Measured" instead of "FunctAI measured" | the same words in the dashboard and in the agent's messages; code shows only in the Code tab and in folded "exact change" boxes |
| **"You said…" about answers the person never gave** (bug: the mockup pre-fills an answer key) | every pre-filled judgement says "Marked wrong by the example answer key (filled in for this mockup)"; the helper says "the answer key says"; only the person's own clicks say "by you" | rule 7 of §17.4 (who did what), applied to every sentence: never attribute a judgement to someone who did not make it |
| Back left the site; no way home; the data flow to OpenAI unexplained | Back returns to the start page and "Continue: Club inbox (version 2, saved)" brings the work back; a "⌂ Start" button; each step that sends data says to whom, and that what the company keeps depends on its rules and the person's account | the dashboard has an address per view (Back works) and saves nothing of its own (work is files); a privacy line on every cloud step and before a file is first sent |

**Not done, on purpose:**

- **Made-up social proof** ("12,000 student clubs built helpers this
  month"): there are no such users. Real quotes only once real people
  have used it.
- **"We don't keep your data, and it isn't used for training"**: Studio
  keeps nothing remotely, but what OpenAI or another company keeps is
  theirs to say. The line names the company and points to its rules
  instead of promising for it.
- **"First drafts usually get about half right"**: no evidence for that
  number. It says instead that a first try is a starting point, and
  that 9 examples are too few to know.
- **Asking "Do you want to see the code?" at the start**: one more
  question before anything happens. Instead, code appears only where
  asked for (the Code tab, "See the exact change"), so non-coders never
  meet it unasked, and learners are always one click from it (R6).
- **"Leadership" and funnel framing**: Studio is a local, open tool, not
  a sign-up funnel. The clarity points still hold.

**Added to the real test (§10.3):** a five-second test on the start page
("what does this make?"), and recovery tasks for careful users (Back,
leaving and coming back, undoing a change).

## After a second simulated run (2026-11-02)

The same platform, five **simulated** visitors on the second mockup (a
coffee-roasting business owner, a law-firm marketing director, a freelance
designer). Several points were already fixed by the third mockup (Back,
"calls", the plain view's place). The rest made the fourth mockup:

| finding | done in mockup 4 | what it means for the real Studio |
|---|---|---|
| A price per test doesn't tell a business what a month will cost | **Costs** (header button and a suggestion): spent this session; cost per message, step by step; "if you run it on [300] messages a month: about 13.5¢", with what the estimate rests on; **your limit** ("ask me before any run above $1") | the estimate comes from the call log's tokens and a dated price table (§9, item 4); the limit is the command line's `--budget` (§25), and the dashboard's |
| Starters are all hobbies and school; a business owner can't see herself | *For work: Sort wholesale inquiries, Reply to café reviews*, beside school, club, games, art (wide walls, R7). The mockup still builds only the club example and says so, with the plan it would make for the work request | starters (with recorded runs, §8 level 0) must cover work as well as play |
| The prompt box is one line; a chip's long text is cut off | a box that grows with the text; Enter makes, Shift+Enter is a new line | same |
| The plain view is right for decision-makers but can't be shared; evaluation is phrased for engineers | the tab is now **Summary**: a one-page card (what it does, how often it's right, where it goes wrong, what it costs, where messages go, what was tested), **Download this summary** (one HTML file, real in the mockup: opens anywhere, prints to PDF), **Copy link**; then "Follow one message" step by step. Results read "Tested on 5 checked examples it hadn't seen: 3 right before the change, 5 right after" | `functai studio page` (§25, §27) writes exactly this file; the card's facts come from the log, the graph and the requirements |
| "Too few checked to know" reads as "it doesn't work" | everywhere it appears it is a next step: **check 10 more for a reliable score**, which opens the checking flow (one message at a time, keys 1 and 2, dots filling green and red, then the score with its range: "10 of 19 right, likely 32–73%") | the random check of §7.7 and Chattering 74, with ratings marked as one `sample` |
| Back leaves the app; no address per view or version | every view, version and example has an address (`#/club-inbox/v2/table?example=3`); Back and Forward move between views; work is saved in the browser and comes back after a reload or from a link; a link to an older version says which version you're now seeing | the dashboard's routes; work lives in files, so a link opens the same project on the same machine, and a hosted Studio (Chattering) makes links shareable with other people |
| The floating suggestions and box covered rows and code | the ask bar is part of the layout, below the content: nothing scrolls under it (tested: the last table row ends above it) | same |
| ✓ and ✗ didn't show on the tester's computer | every symbol (right, wrong, cloud, this computer, picture, link) is a drawn icon, not a font character | same |

**Not done, on purpose:**

- **"Plans & limits" with tiers**: there are none. Costs says so: Studio
  is free and open source; the person pays the AI company directly.
- **"Made by businesses like yours"**: no such users yet (as before: no
  invented social proof).
- **A "how we built this" or careers page**: this is a design mockup, not
  a company site; the design note is the "how".
- **Hiding the suggestions while scrolling**: unnecessary once they no
  longer cover anything.

The mockup's costs are now consistent with each other and roughly real
(about 0.015¢ per step for a small model on a ~300-word request, so about
0.045¢ per message and 0.32¢ for the first 9), labelled as approximate.

## References

- [1] https://doi.org/10.1145/3491102.3517582 — Wu et al. 2022, AI Chains, CHI
- [2] https://doi.org/10.1145/3491101.3519729 — Wu et al. 2022, PromptChainer, CHI EA
- [3] https://doi.org/10.1145/3544548.3581388 — Zamfirescu-Pereira et al. 2023, Why Johnny Can't Prompt, CHI
- [4] https://doi.org/10.1145/3654777.3676450 — Shankar et al. 2024, Who Validates the Validators?, UIST
- [5] https://doi.org/10.1145/3613904.3642216 — Kim et al. 2024, EvalLM, CHI
- [6] https://doi.org/10.1145/3613904.3642139 — Gero et al. 2024, Sensemaking of LLM outputs at scale, CHI
- [7] https://doi.org/10.1145/3613904.3642016 — Arawjo et al. 2024, ChainForge, CHI
- [8] https://doi.org/10.1145/3586183.3606833 — Kim et al. 2023, Cells, Generators, and Lenses, UIST
- [9] https://doi.org/10.1006/jvlc.1996.0009 — Green & Petre 1996, Cognitive Dimensions, JVLC
- [10] https://doi.org/10.1109/2.366157 — Burnett et al. 1995, Scaling up visual programming languages, Computer
- [11] https://doi.org/10.1145/3089799 — Weintrop & Wilensky 2017, Block-based vs text-based in high school, TOCE
- [12] https://doi.org/10.1145/2787622.2787752 — Weintrop 2015, Text, blocks and hybrid tools, ICER
- [13] https://doi.org/10.1184/r1/6621917 — Resnick et al., Design principles for tools to support creative thinking (2005 workshop report)
- [14] https://doi.org/10.1145/3173574.3173935 — Dasgupta & Hill 2018, How "wide walls" can increase engagement, CHI
- [15] https://doi.org/10.1145/1868358.1868363 — Maloney et al. 2010, The Scratch programming language and environment, TOCE
- [16] https://doi.org/10.1109/hcc.2002.1046334 — Blackwell 2002, Attention investment, HCC
- [17] https://doi.org/10.1145/3544548.3580817 — Liu et al. 2023, "What it wants me to say", CHI
- [18] https://doi.org/10.1177/0002764212469359 — Hill & Monroy-Hernández 2012, The remixing dilemma
- [19] https://doi.org/10.1145/3173574.3173606 — Rule et al. 2018, Exploration and explanation in computational notebooks, CHI
- [20] https://doi.org/10.1145/3290605.3300500 — Head et al. 2019, Managing messes in computational notebooks, CHI
- [21] https://doi.org/10.1145/3392826 — Zhang et al. 2020, How do data science workers collaborate?, CSCW
- [22] https://doi.org/10.1145/3351095.3375624 — Bhatt et al. 2020, Explainable machine learning in deployment, FAT*
- [23] https://doi.org/10.1145/3706598.3713581 — Epperson et al. 2025, Interactive debugging and steering of multi-agent AI systems, CHI
- [24] https://doi.org/10.1109/tvcg.2024.3394053 — Lu et al. 2024, AgentLens, TVCG
- [25] https://openalex.org/W6979253602 — Zhang et al. 2025, Which agent causes task failures and when?
- [26] https://doi.org/10.1109/mic.2012.12 — van der Aalst & Dustdar 2012, Process mining put into context
- [27] https://doi.org/10.1609/aimag.v35i4.2513 — Amershi et al. 2014, Power to the people, AI Magazine
- [28] https://doi.org/10.1145/3334480.3382839 — Carney et al. 2020, Teachable Machine, CHI EA
- [29] https://doi.org/10.1145/3459990.3463394 — Dwivedi 2021, Introducing children to machine learning through machine teaching, IDC
- [30] https://doi.org/10.1145/3290605.3300233 — Amershi et al. 2019, Guidelines for human-AI interaction, CHI
- [31] https://doi.org/10.1145/3313831.3376727 — Long & Magerko 2020, What is AI literacy?, CHI
- [32] https://doi.org/10.1145/2617588 — Cherry & Latulipe 2014, Creativity Support Index, TOCHI
- [33] https://doi.org/10.1145/3706598.3714051 — Reza et al. 2025, PromptHive, CHI
- [34] https://doi.org/10.1145/3706598.3713905 — Zhou et al. 2025, InstructPipe, CHI
- [35] https://doi.org/10.1145/157203.157207 — Mack & Nielsen 1993, Usability inspection methods
- [36] https://doi.org/10.1145/3586030 — Barke, James & Polikarpova 2023, Grounded Copilot, PACMPL
- [37] https://doi.org/10.48550/arxiv.2410.21819 — Wataoka et al. 2024, Self-preference bias in LLM-as-a-judge
- [38] https://doi.org/10.18653/v1/2024.acl-long.826 — Xu et al. 2024, Pride and prejudice: LLM amplifies self-bias in self-refinement, ACL
- [39] https://doi.org/10.1145/3449287 — Buçinca, Malaya & Gajos 2021, To trust or to think, PACM HCI
- [40] https://doi.org/10.1145/3613904.3642495 — Ferdowsi et al. 2024, Validating AI-generated code with live programming, CHI
- [41] https://doi.org/10.1145/302979.303030 — Horvitz 1999, Principles of mixed-initiative user interfaces, CHI
- [42] https://doi.org/10.1145/960201.957206 — Shneiderman 2002, Promoting universal usability with multi-layer interface design
- [43] https://doi.org/10.1145/3617367 — Prather et al. 2023, "It's weird that it knows what I want", TOCHI
