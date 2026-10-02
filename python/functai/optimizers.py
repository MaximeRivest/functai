"""Optimization.

An optimizer tunes what an AI function sends besides the inputs: its
instruction and its demos (worked examples, as lmcc turns). It never edits
code, types or the layout. It works on one AI function or on a ``@module``
(a Python function that calls several), and returns the new state of each
AI function; ``fn.opt(...)`` returns a copy with it, and the function itself is
unchanged.

- ``LabeledFewShot``: labeled examples as demos
- ``BootstrapFewShot``: run the program on examples, keep the runs the metric
  accepts as demos (their full turns: reasoning, tool calls and all)
- ``BootstrapFewShotWithRandomSearch``: many bootstrapped demo sets, keep the best
- ``InstructionSearch``: proposed instructions × demo sets, searched on minibatches
  (MIPRO-style, with random/greedy search instead of a Bayesian optimizer)
- ``GEPA``: the instruction rewritten from the function's mistakes, read with
  feedback in words, over a Pareto pool (design/04-gepa.md)
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import random
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from . import calllog
from .config import forced
from .core import P, R, FunctAIFunc, ProgramState
from .evaluation import (Metric, _Target, evaluate, expected_columns, expected_metrics, parallel,
                         resolve_metrics, rows_of, run_row, with_expected)

States = Dict[FunctAIFunc, ProgramState]
Row = Dict[str, Any]


def _metric(metric: Any, target: _Target, rows: Sequence[Row], *, required: bool) -> Optional[Metric]:
    """The one metric an optimizer scores with. None (no metric given, no
    labels): every run is accepted, when ``required`` is False."""
    if isinstance(metric, (list, tuple, Mapping)):
        raise TypeError("an optimizer scores with one metric; pass a function or a dpyr expression")
    resolved = resolve_metrics(metric, target, rows)
    if resolved:
        return resolved[0]
    if required:
        raise ValueError("this optimizer compares candidates, so it needs a metric: pass metric=..., or give "
                         f"the data a column named {target.output_names[0]!r} to match exactly")
    return None


def _passes(score: Optional[float], threshold: Optional[float]) -> bool:
    if score is None:
        return False
    return score >= threshold if threshold is not None else bool(score)


def _mean_score(program: Any, rows: Sequence[Row], metric: Metric, num_threads: int,
                states: Optional[States]) -> float:
    ev = evaluate(program, rows, {metric.name: metric.expr if metric.expr is not None else metric.fn},
                  num_threads=num_threads, states=states)
    return ev.score or 0.0


# ------------------------------------------------------------------ optimizers


class Optimizer:
    """The base class of optimizers: subclass it to write your own.

    An optimizer has one method, ``compile(program, *, trainset, valset=None)``:
    given an AI function or a ``@module`` and rows with known answers (a list
    of dicts), it
    returns the new state of each AI function it improves, as
    ``{fn: ProgramState(instructions=..., demos=...)}``. It never changes the
    functions itself: ``fn.opt(rows, optimizer=MyOptimizer())`` builds the
    improved copy from what it returns. ``metric`` (``metric(row,
    prediction)``) is set from ``fn.opt(metric=...)`` when the optimizer has
    none.

    ```{.python .no-run}
    class FirstRows(functai.Optimizer):          # the first three rows become worked examples
        def compile(self, program, *, trainset, valset=None):
            demos = tuple({"inputs": {"message": r["message"]}, "outputs": {"result": r["team"]}}
                          for r in trainset[:3])
            return {program: functai.ProgramState(demos=demos)}

    better = team.opt(rows, optimizer=FirstRows())
    ```
    """

    metric: Optional[Callable] = None

    def compile(self, program: Any, *, trainset: Sequence[Any], valset: Optional[Sequence[Any]] = None) -> States:
        raise NotImplementedError


def _with_context(row: Row) -> bool:
    """A row that was answered after earlier turns (``rated`` on a
    conversation): it measures, but is never a worked example (a worked
    example is one turn placed before the question)."""
    return bool(row.get("earlier")) or bool(row.get("helpers"))


def _labeled_demo(fn: FunctAIFunc, target: _Target, row: Row) -> Optional[Dict[str, Any]]:
    if _with_context(row):
        return None
    outs = set(fn._spec().outputs) | {"reasoning"}
    labels = {k: v for k, v in row.items() if k in outs}
    if not labels:
        return None
    return {"inputs": target.inputs_of(row), "outputs": labels}


class LabeledFewShot(Optimizer):
    """Rows with known answers become worked examples, sent before every call.

    The cheapest optimizer: no model is called to optimize. Use it as
    ``fn.opt(rows, optimizer=LabeledFewShot(k=8))``, or by name:
    ``functai.labeled_few_shot(fn, rows, k=8)``. One AI function only (a
    ``@module`` needs ``BootstrapFewShot``, which knows which step each
    example belongs to).

    Parameters
    ----------
    k : int
        At most this many examples (default 16).
    sample : bool
        True (default): a random sample of the rows; False: the first ``k``.
    seed : int
        The sample's seed, so the same rows give the same examples.
    """

    def __init__(self, k: int = 16, *, sample: bool = True, seed: int = 0):
        self.k, self.sample, self.seed = k, sample, seed

    def compile(self, program, *, trainset, valset=None) -> States:
        target = _Target(program)
        if not target.single:
            raise TypeError("LabeledFewShot needs labels per AI function; for a @module use BootstrapFewShot")
        fn = target.predictors[0]
        examples = rows_of(trainset)
        chosen = random.Random(self.seed).sample(examples, min(self.k, len(examples))) if self.sample \
            else examples[: self.k]
        demos = tuple(d for d in (_labeled_demo(fn, target, ex) for ex in chosen) if d)
        return {fn: dataclasses.replace(fn.state(), demos=demos)}


class BootstrapFewShot(Optimizer):
    """Run the program (or a ``teacher``: a stronger model name, or an AI function)
    on training examples; the runs ``metric`` accepts become demos, whole turns
    included. Labeled examples fill the rest, up to ``max_labeled_demos``."""

    def __init__(self, metric: Optional[Callable] = None, *, metric_threshold: Optional[float] = None,
                 max_bootstrapped_demos: int = 4, max_labeled_demos: int = 16, max_rounds: int = 1,
                 max_errors: int = 10, teacher: Any = None, teacher_settings: Optional[Dict[str, Any]] = None,
                 num_threads: int = 1, seed: int = 0):
        self.metric = metric
        self.metric_threshold = metric_threshold
        self.max_bootstrapped_demos = max_bootstrapped_demos
        self.max_labeled_demos = max_labeled_demos
        self.max_rounds = max(1, max_rounds)
        self.max_errors = max_errors
        self.teacher = teacher
        self.teacher_settings = dict(teacher_settings or {})
        self.num_threads = num_threads
        self.seed = seed

    def _teacher_overrides(self, round_idx: int) -> Dict[str, Any]:
        o = dict(self.teacher_settings)
        if isinstance(self.teacher, str):
            o["lm"] = self.teacher
        if round_idx > 0:                       # later rounds: fresh samples, not cached replies
            o.update(cache_replies=False, temperature=1.0)
        return o

    def _run_one(self, target: _Target, row: Row, round_idx: int,
                  metric: Optional[Metric]) -> Tuple[bool, list, Optional[str]]:
        if isinstance(self.teacher, FunctAIFunc) and target.single:
            teacher = _Target(self.teacher)
            run = run_row(teacher, row)
            traced = [(target.predictors[0], {"inputs": target.inputs_of(row), "outputs": dict(run.pred)})] \
                if run.pred is not None else []
        else:
            with forced(**self._teacher_overrides(round_idx)):
                run = run_row(target, row)
            traced = [(fn, p.turn) for fn, p in run.trace]
        if run.error:
            return False, [], run.error
        try:
            ok = metric is None or _passes(metric.score(target, row, run), self.metric_threshold)
        except Exception as exc:  # noqa: BLE001 — counted against max_errors
            return False, [], f"metric {metric.name}: {type(exc).__name__}: {exc}"
        return ok, traced, None

    def bootstrap(self, target: _Target, examples: Sequence[Row],
                  metric: Optional[Metric]) -> Tuple[Dict[FunctAIFunc, list], set]:
        boot: Dict[FunctAIFunc, list] = {fn: [] for fn in target.predictors}
        used: set = set()
        errors: List[str] = []
        if self.max_bootstrapped_demos <= 0:
            return boot, used
        for round_idx in range(self.max_rounds):
            pending = [i for i in range(len(examples)) if i not in used]
            batch = max(1, self.num_threads)
            for start in range(0, len(pending), batch):
                if all(len(v) >= self.max_bootstrapped_demos for v in boot.values()):
                    return boot, used
                chunk = pending[start:start + batch]
                rows = parallel(lambda i: self._run_one(target, examples[i], round_idx, metric), chunk,
                                self.num_threads)
                for i, (ok, traced, err) in zip(chunk, rows):
                    if err:
                        errors.append(err)
                        if len(errors) > self.max_errors:
                            raise RuntimeError(f"bootstrapping: {len(errors)} runs failed; last: {err}")
                        continue
                    if not ok:
                        continue
                    used.add(i)
                    if _with_context(examples[i]):
                        continue                  # asked with its earlier turns: measured, never an example
                    for fn, turn in traced:
                        if fn in boot and len(boot[fn]) < self.max_bootstrapped_demos and turn is not None:
                            boot[fn].append(turn)
        return boot, used

    def compile(self, program, *, trainset, valset=None) -> States:
        target = _Target(program)
        examples = rows_of(trainset)
        metric = _metric(self.metric, target, examples, required=False)
        boot, used = self.bootstrap(target, examples, metric)
        states: States = {}
        rng = random.Random(self.seed)
        for fn in target.predictors:
            demos = list(boot[fn])
            room = self.max_labeled_demos - len(demos)
            if target.single and room > 0:
                rest = [ex for i, ex in enumerate(examples) if i not in used]
                for ex in rng.sample(rest, min(room, len(rest))):
                    d = _labeled_demo(fn, target, ex)
                    if d:
                        demos.append(d)
            states[fn] = dataclasses.replace(fn.state(), demos=tuple(demos))
        return states


class BootstrapFewShotWithRandomSearch(Optimizer):
    """Try several sets of demos and keep the one that scores best on the validation rows.

    Several candidate demo sets (none, labeled only, bootstrapped, bootstrapped
    from shuffled examples), each scored on ``valset`` (default: the trainset);
    the best one wins. ``candidates`` holds every candidate afterwards, as rows
    (``{"candidate", "demos", "score"}``; ``dpyr.read(opt.candidates)`` makes
    them a table)."""

    def __init__(self, metric: Optional[Callable] = None, *, metric_threshold: Optional[float] = None,
                 max_bootstrapped_demos: int = 4, max_labeled_demos: int = 16, max_rounds: int = 1,
                 num_candidate_programs: int = 8, num_threads: int = 1, max_errors: int = 10,
                 teacher: Any = None, seed: int = 0, stop_at_score: Optional[float] = None):
        self.metric = metric
        self.metric_threshold = metric_threshold
        self.max_bootstrapped_demos = max_bootstrapped_demos
        self.max_labeled_demos = max_labeled_demos
        self.max_rounds = max_rounds
        self.num_candidate_programs = num_candidate_programs
        self.num_threads = num_threads
        self.max_errors = max_errors
        self.teacher = teacher
        self.seed = seed
        self.stop_at_score = stop_at_score
        self.candidates: List[Dict[str, Any]] = []

    def compile(self, program, *, trainset, valset=None) -> States:
        target = _Target(program)
        examples = rows_of(trainset)
        val = rows_of(valset) if valset is not None else examples
        metric = _metric(self.metric, target, examples, required=True)
        base: States = {fn: fn.state() for fn in target.predictors}
        best: Tuple[float, States] = (-1.0, base)
        self.candidates = []
        for seed in range(-3, self.num_candidate_programs):
            if seed == -3:
                label, states = "zero-shot", {fn: dataclasses.replace(st, demos=()) for fn, st in base.items()}
            elif seed == -2:
                if not target.single:
                    continue
                label = "labeled"
                states = LabeledFewShot(k=self.max_labeled_demos, seed=self.seed).compile(program, trainset=examples)
            else:
                shuffled = list(examples)
                size = self.max_bootstrapped_demos
                if seed >= 0:
                    rng = random.Random(self.seed + seed)
                    rng.shuffle(shuffled)
                    size = rng.randint(1, max(1, self.max_bootstrapped_demos))
                label = "bootstrapped" if seed == -1 else f"bootstrapped (seed {seed}, {size} demos)"
                states = BootstrapFewShot(self._metric_arg(metric), metric_threshold=self.metric_threshold,
                                          max_bootstrapped_demos=size, max_labeled_demos=self.max_labeled_demos,
                                          max_rounds=self.max_rounds, max_errors=self.max_errors,
                                          teacher=self.teacher, num_threads=self.num_threads,
                                          seed=self.seed + seed).compile(program, trainset=shuffled)
            score = _mean_score(program, val, metric, self.num_threads, states)
            self.candidates.append({"candidate": label, "demos": sum(len(st.demos) for st in states.values()),
                                    "score": score})
            if score > best[0]:
                best = (score, states)
            if self.stop_at_score is not None and score >= self.stop_at_score:
                break
        return best[1]

    @staticmethod
    def _metric_arg(metric: Metric) -> Any:
        return metric.expr if metric.expr is not None else metric.fn


_TIPS = [
    "",
    "Be concise and direct.",
    "Be precise: the task is high-stakes and mistakes are costly.",
    "Describe the expected output format exactly.",
    "Spell out the steps to follow before answering.",
    "Name the common mistakes on this task and how to avoid them.",
    "Give the model a helpful persona suited to the task.",
    "Consider edge cases and unusual inputs.",
]


class InstructionSearch(Optimizer):
    """Search instructions written by a model, with demo sets, and keep the best.

    Instruction candidates (the current one plus proposals written by
    ``prompt_lm`` from the code, the signature and a few examples) × demo sets
    (bootstrapped, unless both demo limits are 0), searched over ``num_trials``
    minibatch evaluations; the top combinations are then scored on the whole
    ``valset`` and the best wins. ``trials`` holds every trial afterwards, as
    rows (``dpyr.read(opt.trials)`` makes them a table).

    MIPRO-style; the search is random with greedy refinement, not Bayesian."""

    def __init__(self, metric: Optional[Callable] = None, *, num_candidates: int = 6, num_trials: int = 12,
                 minibatch_size: int = 20, max_bootstrapped_demos: int = 4, max_labeled_demos: int = 4,
                 prompt_lm: Any = None, num_threads: int = 1, seed: int = 0, full_eval_top: int = 3,
                 metric_threshold: Optional[float] = None, max_errors: int = 10, teacher: Any = None):
        self.metric = metric
        self.num_candidates = max(1, num_candidates)
        self.num_trials = max(1, num_trials)
        self.minibatch_size = minibatch_size
        self.max_bootstrapped_demos = max_bootstrapped_demos
        self.max_labeled_demos = max_labeled_demos
        self.prompt_lm = prompt_lm
        self.num_threads = num_threads
        self.seed = seed
        self.full_eval_top = max(1, full_eval_top)
        self.metric_threshold = metric_threshold
        self.max_errors = max_errors
        self.teacher = teacher
        self.trials: List[Dict[str, Any]] = []

    def _instructions(self, fn: FunctAIFunc, examples: Sequence[Row], input_names: Sequence[str],
                      rng: random.Random) -> List[Optional[str]]:
        from . import meta
        base = fn.state().instructions
        out: List[Optional[str]] = [base]
        proposals: List[str] = []
        sample = list(examples)
        for i in range(self.num_candidates - 1):
            rng.shuffle(sample)
            text = meta.propose_instruction(fn, examples=meta.examples_text(sample, input_names),
                                            previous=list(proposals),
                                            tip=_TIPS[i % len(_TIPS)], lm=self.prompt_lm,
                                            base_instruction=fn.instructions)
            if text and text not in proposals:
                proposals.append(text)
                out.append(text)
        return out

    def _demo_sets(self, program, target: _Target, examples: Sequence[Row],
                   metric: Metric) -> List[Dict[FunctAIFunc, tuple]]:
        base = {fn: tuple(fn.state().demos) for fn in target.predictors}
        if self.max_bootstrapped_demos <= 0 and self.max_labeled_demos <= 0:
            return [base]
        sets = [base]
        for k in range(self.num_candidates - 1):
            shuffled = list(examples)
            random.Random(self.seed + k).shuffle(shuffled)
            states = BootstrapFewShot(BootstrapFewShotWithRandomSearch._metric_arg(metric),
                                      metric_threshold=self.metric_threshold,
                                      max_bootstrapped_demos=self.max_bootstrapped_demos,
                                      max_labeled_demos=self.max_labeled_demos, max_errors=self.max_errors,
                                      teacher=self.teacher, num_threads=self.num_threads,
                                      seed=self.seed + k).compile(program, trainset=shuffled)
            sets.append({fn: st.demos for fn, st in states.items()})
        return sets

    def compile(self, program, *, trainset, valset=None) -> States:
        target = _Target(program)
        rng = random.Random(self.seed)
        examples = rows_of(trainset)
        val = rows_of(valset) if valset is not None else examples
        metric = _metric(self.metric, target, examples, required=True)
        fns = target.predictors
        instr = {fn: self._instructions(fn, examples, target.input_names, rng) for fn in fns}
        demo_sets = self._demo_sets(program, target, examples, metric)

        def states_of(combo: Tuple[int, ...]) -> States:
            k = len(fns)
            return {fn: ProgramState(instructions=instr[fn][combo[j]], demos=tuple(demo_sets[combo[k]][fn]))
                    for j, fn in enumerate(fns)}

        sizes = [len(instr[fn]) for fn in fns] + [len(demo_sets)]
        scores: Dict[Tuple[int, ...], List[float]] = {}
        self.trials = []
        for t in range(self.num_trials):
            if t == 0:
                combo = tuple(0 for _ in sizes)
            elif scores and t >= self.num_trials // 2 and rng.random() < 0.6:
                best = max(scores, key=lambda c: sum(scores[c]) / len(scores[c]))
                j = rng.randrange(len(sizes))
                combo = tuple(rng.randrange(sizes[i]) if i == j else best[i] for i in range(len(sizes)))
            else:
                combo = tuple(rng.randrange(n) for n in sizes)
            batch = val if len(val) <= self.minibatch_size else rng.sample(val, self.minibatch_size)
            score = _mean_score(program, batch, metric, self.num_threads, states_of(combo))
            scores.setdefault(combo, []).append(score)
            self.trials.append({"trial": t, "combo": combo, "minibatch_score": score})
        ranked = sorted(scores, key=lambda c: sum(scores[c]) / len(scores[c]), reverse=True)
        finalists = ranked[: self.full_eval_top]
        if len(val) <= self.minibatch_size:
            best_combo = finalists[0]
        else:
            full = {c: _mean_score(program, val, metric, self.num_threads, states_of(c)) for c in finalists}
            best_combo = max(full, key=full.get)
            for trial in self.trials:
                if trial["combo"] in full:
                    trial["full_score"] = full[trial["combo"]]
        return states_of(best_combo)


# ------------------------------------------------------------------ GEPA


def _answer_text(v: Any) -> str:
    if isinstance(v, str):
        return v
    try:
        return json.dumps(v, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(v)


def default_feedback(outputs: Sequence[str]) -> Callable[[Row, Any, Optional[str]], str]:
    """Words for one row: "right", "wrong: the right answer is ...", or the call's error."""
    from .evaluation import _norm

    def feedback(row: Row, pred: Any, error: Optional[str]) -> str:
        if pred is None:
            return f"the call failed: {error}"
        wrong = [k for k in outputs if k in row and _norm(row[k]) != _norm(pred.get(k))]
        if not wrong:
            return "right"
        return "wrong: " + "; ".join(f"the right {'answer' if k == 'result' else k} is {_answer_text(row[k])}"
                                     for k in wrong)
    return feedback


def shape_words(shape: Mapping[str, Any]) -> str:
    """A JSON Schema shape in words, for the reflection model (the same words in every language)."""
    options = shape.get("anyOf") or shape.get("oneOf")
    if options:
        kept = [o for o in options if o.get("type") != "null"]
        words = " or ".join(shape_words(o) for o in kept) or "nothing"
        return words + (", or nothing" if len(kept) < len(options) else "")
    if "enum" in shape:
        return "one of " + ", ".join(_answer_text(v) for v in shape["enum"])
    kind = shape.get("type")
    if kind == "array":
        return "a list of " + shape_words(shape.get("items") or {})
    if kind == "object":
        props = shape.get("properties") or {}
        return "a record of " + ", ".join(f"{k} ({shape_words(v)})" for k, v in props.items()) if props else "an object"
    return {"string": "text", "integer": "a whole number", "number": "a number", "boolean": "true or false"}.get(kind, "a value")


def fields_text(signature: Any) -> str:
    """What a function takes and returns, one field a line, with its type and words."""
    def line(f: Any) -> str:
        desc = getattr(f, "desc", None)
        return f"- {f.name}: {shape_words(f.shape)}" + (f". {desc}" if desc else "")
    ins = [line(f) for f in signature.inputs if f.purpose == "plain"]
    outs = [line(f) for f in signature.outputs if f.purpose == "plain"]
    return "\n".join(["Inputs:", *ins, "Outputs:", *outs])


def copies_an_input(instruction: str, rows: Sequence[Row], input_names: Sequence[str], at_least: int = 30) -> bool:
    """Does the instruction quote an input of ``at_least`` characters, verbatim
    (case and spacing ignored)? Such a proposal memorises cases (design/04-gepa.md)."""
    text = " ".join(instruction.split()).casefold()
    for row in rows:
        for k in input_names:
            v = row.get(k)
            if isinstance(v, str):
                v = " ".join(v.split()).casefold()
                if len(v) >= at_least and v in text:
                    return True
    return False


@dataclasses.dataclass
class _Candidate:
    instruction: str
    parents: Tuple[int, ...]
    kind: str                                   # "written", "reflect", "combine"
    scores: List[float] = dataclasses.field(default_factory=list)      # on the selection rows
    tried: List[str] = dataclasses.field(default_factory=list)         # children that did not beat it

    @property
    def mean(self) -> float:
        return sum(self.scores) / len(self.scores) if self.scores else 0.0


class GEPA(Optimizer):
    """Rewrite the instruction from the function's mistakes: GEPA (Agrawal et
    al., 2025), with functai's changes (design/04-gepa.md).

    Half the rows (or ``valset``) select, and are never shown; the other half
    give feedback. A pool of instructions is scored row by row on the selection
    rows; a parent is picked from its Pareto frontier, run on ``minibatch``
    feedback rows, and a ``teacher`` model reads its answers with feedback in
    words and writes a new instruction. A child that does better on the
    minibatch is scored on the selection rows and joins the pool. Every fourth
    step, two frontier candidates that win different rows are combined. The
    reflection sees what was tried and failed; a proposal that copies an input
    is dropped; ties go to the shorter instruction; a row is run once per
    instruction.

    ``budget`` counts calls of the function; ``trials`` holds every candidate
    afterwards (its selection score is optimistic: it was chosen on those rows).
    ``feedback(row, prediction, error) -> str`` gives the words (default: right,
    or the right answer, or the error)."""

    def __init__(self, metric: Optional[Callable] = None, *, budget: int = 300, minibatch: int = 4,
                 teacher: Any = None, feedback: Optional[Callable] = None, num_threads: int = 8, seed: int = 0):
        self.metric = metric
        self.budget = int(budget)
        self.minibatch = max(1, int(minibatch))
        self.teacher = teacher
        self.feedback = feedback
        self.num_threads = num_threads
        self.seed = seed
        self.trials: List[Dict[str, Any]] = []
        self.calls = 0
        self.reflections = 0

    # -- running

    def _scores(self, fn: FunctAIFunc, metric: Metric, rows: Sequence[Row], cand: _Candidate,
                memo: Dict[Tuple[str, int], Tuple[float, Any, Optional[str]]], keys: Sequence[int],
                feedback: Callable) -> List[Tuple[float, Any, Optional[str]]]:
        """Score, prediction and error of each row for this instruction; each (instruction, row) runs once."""
        todo = [i for i in keys if (cand.instruction, i) not in memo]
        if todo:
            state = ProgramState(instructions=cand.instruction, demos=fn.state().demos)
            ev = evaluate(fn, [rows[i] for i in todo], {metric.name: metric.expr if metric.expr is not None else metric.fn},
                          num_threads=self.num_threads, states={fn: state})
            self.calls += len(todo)
            scores, errors = ev.scores(), dict(ev.errors)
            for j, i in enumerate(todo):
                memo[(cand.instruction, i)] = (float(scores[j]), ev.predictions[j], errors.get(j))
        return [memo[(cand.instruction, i)] for i in keys]

    @staticmethod
    def _frontier(pool: List[_Candidate]) -> Dict[int, int]:
        """Candidates on the Pareto frontier, with how many selection rows each is best on."""
        n = len(pool[0].scores)
        wins: Dict[int, int] = {}
        for r in range(n):
            best = max(c.scores[r] for c in pool)
            for k, c in enumerate(pool):
                if c.scores[r] == best:
                    wins[k] = wins.get(k, 0) + 1

        def dominated(a: int) -> bool:
            return any(b != a and all(pool[b].scores[r] >= pool[a].scores[r] for r in range(n))
                       and any(pool[b].scores[r] > pool[a].scores[r] for r in range(n)) for b in wins)
        return {k: w for k, w in wins.items() if not dominated(k)}

    def _cases(self, rows: Sequence[Row], keys: Sequence[int], results, input_names: Sequence[str],
               outputs: Sequence[str], feedback: Callable) -> str:
        lines: List[str] = []
        for n, (i, (score, pred, error)) in enumerate(zip(keys, results), 1):
            lines.append(f"Case {n}")
            lines += [f"  {k}: {_answer_text(rows[i][k])}" for k in input_names if k in rows[i]]
            if pred is None:
                lines.append("  answer given: (none)")
            else:
                lines += [f"  answer given{'' if k == 'result' else ' ' + k}: {_answer_text(pred.get(k))}" for k in outputs]
            lines.append(f"  score: {score:g}")
            lines.append(f"  feedback: {feedback(rows[i], pred, error)}")
        return "\n".join(lines)

    def compile(self, program, *, trainset, valset=None) -> States:
        from . import meta
        target = _Target(program)
        if not target.single:
            raise TypeError("GEPA improves one AI function's instruction; a @module's feedback is about its "
                            "answer, and which function to blame is not decided yet. Optimize its AI functions one "
                            "by one, or use BootstrapFewShot for the module.")
        fn = target.predictors[0]
        rng = random.Random(self.seed)
        examples = rows_of(trainset)
        metric = _metric(self.metric, target, examples, required=True)
        if valset is not None:
            feed_rows, select_rows = examples, rows_of(valset)
        else:
            if len(examples) < 2:
                raise ValueError("GEPA needs at least 2 rows: some to learn from, some to choose with")
            shuffled = list(examples)
            rng.shuffle(shuffled)
            half = len(shuffled) // 2
            select_rows, feed_rows = shuffled[:half], shuffled[half:]
        outputs = target.output_names
        feedback = self.feedback or default_feedback(outputs)
        fields = fields_text(fn._spec(instructions=None).signature)
        teacher = self.teacher if self.teacher is not None else fn._effective().get("lm")
        self.calls, self.reflections, self.trials = 0, 0, []
        memo_feed: Dict[Tuple[str, int], Any] = {}
        memo_select: Dict[Tuple[str, int], Any] = {}
        all_select = list(range(len(select_rows)))

        def score_select(c: _Candidate) -> None:
            c.scores = [s for s, _p, _e in self._scores(fn, metric, select_rows, c, memo_select, all_select, feedback)]

        def record(c: _Candidate, k: Optional[int], before: Optional[float], after: Optional[float], note: str) -> None:
            self.trials.append({"candidate": k, "kind": c.kind, "parents": list(c.parents), "minibatch_parent": before,
                                "minibatch": after, "score": c.mean if k is not None else None,
                                "length": len(c.instruction), "note": note, "calls": self.calls,
                                "instruction": c.instruction})

        pool = [_Candidate(fn.instructions, (), "written")]
        score_select(pool[0])
        record(pool[0], 0, None, None, "the written instruction")
        order: List[int] = []

        def next_batch() -> List[int]:
            nonlocal order
            batch: List[int] = []
            while len(batch) < min(self.minibatch, len(feed_rows)):
                if not order:
                    order = list(range(len(feed_rows)))
                    rng.shuffle(order)
                i = order.pop()
                if i not in batch:
                    batch.append(i)
            return batch

        def mean_on(c: _Candidate, batch: List[int]) -> float:
            res = self._scores(fn, metric, feed_rows, c, memo_feed, batch, feedback)
            return sum(s for s, _p, _e in res)

        def admit(child: _Candidate, before: float, after: float) -> None:
            if self.calls + len(select_rows) > self.budget:
                record(child, None, before, after, "better on the minibatch; no budget left to score it")
                return
            score_select(child)
            pool.append(child)
            record(child, len(pool) - 1, before, after, "joined the pool")

        step = 0
        while self.calls + 2 * self.minibatch <= self.budget and step < self.budget:   # steps: a bound when rows are cached
            step += 1
            front = self._frontier(pool)
            if all(all((pool[k].instruction, i) in memo_feed and memo_feed[(pool[k].instruction, i)][0] >= 1
                       for i in range(len(feed_rows))) for k in front):
                record(pool[0], None, None, None, "right on every feedback row: no mistake left to learn from")
                break
            pair = self._pair(pool, front) if step % 4 == 0 else None
            if pair is not None:
                a, b = pair
                text = meta.combine(fields=fields, first=pool[a].instruction, second=pool[b].instruction, lm=teacher)
                self.reflections += 1
                child = _Candidate(text, (a, b), "combine")
                if not text or any(text == c.instruction for c in pool):
                    record(child, None, None, None, "no new instruction")
                    continue
                batch = next_batch()
                before = max(mean_on(pool[a], batch), mean_on(pool[b], batch))
                after = mean_on(child, batch)
                if after >= before:
                    admit(child, before, after)
                else:
                    record(child, None, before, after, "worse on the minibatch than its better parent")
                continue
            ks, ws = zip(*sorted(front.items()))
            k = rng.choices(ks, weights=ws)[0]
            parent = pool[k]
            batch = next_batch()
            results = self._scores(fn, metric, feed_rows, parent, memo_feed, batch, feedback)
            before = sum(s for s, _p, _e in results)
            if all(s >= 1 for s, _p, _e in results):
                continue                                   # nothing to learn from these rows
            text = meta.reflect(fields=fields, instruction=parent.instruction,
                                cases=self._cases(feed_rows, batch, results, target.input_names, outputs, feedback),
                                tried=parent.tried[-3:], lm=teacher)
            self.reflections += 1
            child = _Candidate(text, (k,), "reflect")
            if not text or any(text == c.instruction for c in pool):
                record(child, None, before, None, "no new instruction")
                continue
            if copies_an_input(text, feed_rows, target.input_names):
                parent.tried.append(text + "\n(dropped: it copied an input instead of stating a rule)")
                record(child, None, before, None, "copied an input: dropped")
                continue
            after = mean_on(child, batch)
            if after > before:
                admit(child, before, after)
            else:
                parent.tried.append(text)
                record(child, None, before, after, "not better on the minibatch")
        best = max(range(len(pool)), key=lambda i: (pool[i].mean, -len(pool[i].instruction), -i))
        for t in self.trials:
            t["chosen"] = t["candidate"] == best
        if best == 0:
            return {fn: fn.state()}
        return {fn: dataclasses.replace(fn.state(), instructions=pool[best].instruction)}

    @staticmethod
    def _pair(pool: List[_Candidate], front: Dict[int, int]) -> Optional[Tuple[int, int]]:
        """Two frontier candidates that each win rows the other loses, the most such rows."""
        best, pair = 0, None
        ks = sorted(front)
        for x in range(len(ks)):
            for y in range(x + 1, len(ks)):
                a, b = pool[ks[x]], pool[ks[y]]
                a_wins = sum(sa > sb for sa, sb in zip(a.scores, b.scores))
                b_wins = sum(sb > sa for sa, sb in zip(a.scores, b.scores))
                if a_wins and b_wins and min(a_wins, b_wins) > best:
                    best, pair = min(a_wins, b_wins), (ks[x], ks[y])
        return pair


# ------------------------------------------------------------------ .opt()


def _reject_foreign(opt: Any) -> None:
    mod = (opt.__module__ if isinstance(opt, type) else type(opt).__module__) or ""
    if mod.startswith("dspy"):
        raise TypeError("DSPy optimizers no longer run in functai 1.0. Use functai.BootstrapFewShot, "
                        "BootstrapFewShotWithRandomSearch, LabeledFewShot, or InstructionSearch "
                        "(instruction proposals + demos, like MIPROv2).")


def _instantiate(opt: Any, metric: Optional[Callable], opts: Dict[str, Any]) -> Optimizer:
    _reject_foreign(opt)
    if isinstance(opt, type):
        params = inspect.signature(opt.__init__).parameters
        kwargs = dict(opts)
        if metric is not None and "metric" in params:
            kwargs["metric"] = metric
        unknown = set(kwargs) - set(params)
        if unknown and not any(p.kind is p.VAR_KEYWORD for p in params.values()):
            raise TypeError(f"{opt.__name__} does not take {sorted(unknown)}")
        return opt(**kwargs)
    if opts:
        raise TypeError(f"optimizer options {sorted(opts)} need an optimizer class, not an instance")
    if metric is not None and getattr(opt, "metric", None) is None:
        opt.metric = metric
    return opt


@calllog.tagged("optimization")                # logged calls say they were part of an optimization
def optimize(program: Any, *, trainset: Optional[Sequence[Any]] = None, optimizer: Any = None,
             metric: Optional[Callable] = None, valset: Optional[Sequence[Any]] = None,
             call_defaults: Optional[Dict[str, Any]] = None, expected: Any = None,
             **opts) -> Tuple[States, Dict[FunctAIFunc, Dict[str, Any]]]:
    """What ``fn.opt(...)`` and ``module.opt(...)`` do: build the trainset (and
    synthesize examples with a teacher when ``n_synth`` > 0), run the optimizer.
    Returns the new state of each AI function and a record of the run, per
    function; nothing is changed (the callers make improved copies)."""
    from . import meta
    from .module import FunctAIModule
    teacher = opts.pop("teacher", None)
    teacher_lm = opts.pop("teacher_lm", None)
    n_synth = int(opts.pop("n_synth", 0) or 0)
    opts.pop("autogen", None)
    target = _Target(program, call_defaults=call_defaults)
    if isinstance(program, FunctAIModule) and call_defaults:
        program._opt_call_defaults = dict(call_defaults)
    examples = rows_of(trainset) if trainset is not None else []
    settings = target.predictors[0]._effective()
    teacher = teacher if teacher is not None else settings.get("teacher")
    teacher_lm = teacher_lm if teacher_lm is not None else settings.get("teacher_lm")
    if n_synth > 0:
        if not target.single:
            raise TypeError("n_synth synthesizes examples for one AI function; pass a trainset to a @module")
        fn = target.predictors[0]
        lm = teacher_lm or (teacher if isinstance(teacher, str) else None) or \
            (teacher._effective().get("lm") if isinstance(teacher, FunctAIFunc) else None)
        if lm is None:
            raise ValueError("n_synth needs a teacher: pass teacher_lm='...' (or teacher=)")
        labeler = teacher if isinstance(teacher, FunctAIFunc) else None
        for item in meta.synthesize(fn, n_synth, lm=lm, labeler=labeler):
            examples.append({**item["inputs"], **item["outputs"]})
    if not examples:
        raise ValueError("optimization needs examples: pass trainset=[...] (or n_synth with a teacher)")
    choice = optimizer if optimizer is not None else (settings.get("optimizer") or BootstrapFewShot)
    if isinstance(choice, type):
        params = inspect.signature(choice.__init__).parameters
        if "teacher" in params and "teacher" not in opts and (teacher_lm or teacher) is not None:
            opts["teacher"] = teacher_lm or teacher
    opt = _instantiate(choice, metric, opts)
    target.check(examples)
    in_context = sum(1 for ex in examples if _with_context(ex))
    if in_context:
        import warnings
        warnings.warn(f"[functai] {target.name}: {in_context} of {len(examples)} rows were answered after earlier "
                      f"turns (a conversation): they are asked again with those turns and scored, but never used "
                      f"as worked examples", stacklevel=3)
    val = rows_of(valset) if valset is not None else None
    mapping = expected_columns(target, expected, examples)
    if mapping:                                 # the right answers under the outputs' names
        examples = with_expected(examples, mapping, target)
        val = with_expected(val, mapping, target) if val is not None else None
        if metric is None:
            opt.metric = expected_metrics(mapping, target)[0].fn
    states = opt.compile(program, trainset=examples, valset=val)
    import time
    logs = {fn: {
        "ts": time.time(), "optimizer": type(opt).__name__,
        "metric": getattr(getattr(opt, "metric", None), "__name__", None),
        "n_examples": len(examples), "synthesized": n_synth > 0,
        "examples": list(examples), "n_demos": len(state.demos),
        "instructions_changed": state.instructions != fn.state().instructions,
        "trials": list(getattr(opt, "trials", None) or []),
    } for fn, state in states.items()}
    return states, logs


# ------------------------------------------------------------------ by name


def labeled_few_shot(fn: "FunctAIFunc[P, R]", data: Any, *, k: int = 16, expected: Any = None,
                     sample: bool = True, seed: int = 0) -> "FunctAIFunc[P, R]":
    '''An improved copy: up to ``k`` rows with known answers become worked examples.

    Examples
    --------
    ```python
    from typing import Literal

    @ai
    def team(message: str) -> Literal["shipping", "billing", "product", "account"]:
        """Which team should answer this customer message?"""
        ...

    taught = functai.labeled_few_shot(team, functai.datasets.tickets(), k=3, expected="category")
    taught.state()
    ```

    No model is called: the rows are the examples.
    '''
    return fn.opt(data, optimizer=LabeledFewShot(k, sample=sample, seed=seed), expected=expected)


def bootstrap_few_shot(fn: "FunctAIFunc[P, R]", data: Any, *, teacher: Any = None, expected: Any = None,
                       metric: Optional[Callable] = None, max_bootstrapped: int = 4, max_labeled: int = 16,
                       num_threads: int = 8, seed: int = 0) -> "FunctAIFunc[P, R]":
    """An improved copy: the function (or a stronger ``teacher`` model) runs on
    rows with known answers, and the runs that were right become worked
    examples, whole (reasoning and tool calls included); labeled rows fill the rest.

    Examples
    --------
    ```python
    # not run: the teacher answers every row (Make it better runs one)
    taught = functai.bootstrap_few_shot(team, train, teacher="gpt-6-sol", expected="category")
    ```
    """
    opt = BootstrapFewShot(metric, max_bootstrapped_demos=max_bootstrapped, max_labeled_demos=max_labeled,
                           teacher=teacher, num_threads=num_threads, seed=seed)
    return fn.opt(data, optimizer=opt, expected=expected)


def gepa(fn: "FunctAIFunc[P, R]", data: Any, *, teacher: Any = None, expected: Any = None,
         selection: Any = None, budget: int = 300, minibatch: int = 4, metric: Optional[Callable] = None,
         feedback: Optional[Callable] = None, num_threads: int = 8, seed: int = 0) -> "FunctAIFunc[P, R]":
    """An improved copy whose instruction a ``teacher`` model rewrote from the
    function's mistakes (``GEPA``; design/04-gepa.md).

    Half the rows (or ``selection``) choose and are never shown to the teacher;
    the other half give feedback. ``better.trials`` is the search. Its scores on
    the choosing rows flatter the one chosen: measure it on rows it never saw.
    The optimizer class ``GEPA`` takes the same options, for
    ``fn.opt(rows, optimizer=GEPA(...))`` and for ``@module``s' AI functions one at a time.

    Examples
    --------
    ```python
    # not run: a search asks the teacher many times (Make it better runs one)
    better = functai.gepa(team.using(lm="gpt-5.4-nano"), train, teacher="gpt-6-sol", expected="category")
    better.instructions
    functai.evaluate(better, test, expected="category")
    ```
    """
    opt = GEPA(metric, budget=budget, minibatch=minibatch, teacher=teacher, feedback=feedback,
               num_threads=num_threads, seed=seed)
    return fn.opt(data, optimizer=opt, expected=expected, valset=selection)


__all__ = ["Optimizer", "LabeledFewShot", "BootstrapFewShot", "BootstrapFewShotWithRandomSearch",
           "InstructionSearch", "GEPA", "optimize", "labeled_few_shot", "bootstrap_few_shot", "gepa"]
