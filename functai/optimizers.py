"""Evaluation and optimization.

An optimizer tunes what an AI function sends besides the inputs: its
instruction and its demos (worked examples, as lmcc turns). It never edits
code, types or the layout. It works on one AI function or on a ``@module``
(a Python function that calls several), and returns the new state of each
AI function; ``fn.opt(...)`` applies it in place and ``fn.undo_opt()`` reverts.

- ``LabeledFewShot``: labeled examples as demos
- ``BootstrapFewShot``: run the program on examples, keep the runs the metric
  accepts as demos (their full turns: reasoning, tool calls and all)
- ``BootstrapFewShotWithRandomSearch``: many bootstrapped demo sets, keep the best
- ``InstructionSearch``: proposed instructions × demo sets, searched on minibatches
  (MIPRO-style, with random/greedy search instead of a Bayesian optimizer)
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import contextvars
import dataclasses
import inspect
import random
import traceback
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .config import forced
from .core import _STATE_OVERRIDE, _TRACE, FunctAIFunc, ProgramState
from .data import Example, Prediction, as_example

States = Dict[FunctAIFunc, ProgramState]


# ------------------------------------------------------------------ metrics


def _norm(v: Any) -> Any:
    if isinstance(v, str):
        return " ".join(v.split()).casefold()
    if hasattr(v, "value") and type(v).__module__ != "builtins" and hasattr(type(v), "__members__"):
        return _norm(v.value)                                   # Enum members compare by value
    return v


def exact_match(example: Example, prediction: Any, trace: Any = None) -> float:
    """1.0 when every labeled output equals the prediction's (strings compared
    ignoring case and repeated whitespace), else 0.0."""
    keys = [k for k in example.labels() if k in prediction]
    if not keys:
        return 0.0
    return float(all(_norm(example[k]) == _norm(prediction[k]) for k in keys))


def _arity(metric: Callable) -> int:
    try:
        params = [p for p in inspect.signature(metric).parameters.values()
                  if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.VAR_POSITIONAL)]
    except (TypeError, ValueError):
        return 3
    if any(p.kind is p.VAR_POSITIONAL for p in params):
        return 3
    return len(params)


def score_of(metric: Callable, example: Example, prediction: Any, trace: Any = None) -> float:
    """``metric(example, prediction[, trace])`` as a float (bools count 0/1)."""
    if isinstance(metric, FunctAIFunc) or _arity(metric) < 3:
        value = metric(example, prediction)
    else:
        value = metric(example, prediction, trace)
    if isinstance(value, Prediction):
        value = next(iter(value.values()), 0.0)
    return float(value)


def _passes(score: float, threshold: Optional[float]) -> bool:
    return score >= threshold if threshold is not None else bool(score)


# ------------------------------------------------------------------ programs


class _Target:
    """What an optimizer runs: one AI function, or a module calling several."""

    def __init__(self, program: Any, *, call_defaults: Optional[Dict[str, Any]] = None):
        from .module import FunctAIModule
        self.program = program
        self.call_defaults = dict(call_defaults or {})
        if isinstance(program, FunctAIFunc):
            self.predictors: List[FunctAIFunc] = [program]
            self.input_names = list(program._sig.parameters)
            self.single = True
        elif isinstance(program, FunctAIModule):
            self.predictors = program.ai_functions()
            self.input_names = list(inspect.signature(program._fn).parameters)
            self.single = False
            self.call_defaults = {**program._opt_call_defaults, **self.call_defaults}
            if not self.predictors:
                raise ValueError(f"@module {program.__name__} calls no @ai function this optimizer could tune")
        else:
            raise TypeError(f"optimizers tune an @ai function or a @module, not {type(program).__name__}")

    def inputs_of(self, example: Example) -> Dict[str, Any]:
        keys = example.input_keys if example.input_keys is not None else [k for k in example if k in self.input_names]
        return {k: example[k] for k in keys}

    def run(self, example: Example) -> Prediction:
        inputs = self.inputs_of(example)
        if self.single:
            return self.program(**inputs, all=True)
        return Prediction({"result": self.program(**{**self.call_defaults, **inputs})})


@contextlib.contextmanager
def with_states(states: Optional[States]):
    """Run with candidate states in place of the functions' own (this context only)."""
    if not states:
        yield
        return
    token = _STATE_OVERRIDE.set({**_STATE_OVERRIDE.get(), **{id(fn): st for fn, st in states.items()}})
    try:
        yield
    finally:
        _STATE_OVERRIDE.reset(token)


@contextlib.contextmanager
def tracing():
    trace: List[Tuple[FunctAIFunc, Prediction]] = []
    token = _TRACE.set(trace)
    try:
        yield trace
    finally:
        _TRACE.reset(token)


def _parallel(fn: Callable, items: Sequence[Any], num_threads: int) -> List[Any]:
    """``fn`` over items, in order; each task runs in a copy of the caller's
    context, so ``with configure(...)`` and candidate states reach the threads."""
    if num_threads <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as pool:
        futures = [pool.submit(contextvars.copy_context().run, fn, x) for x in items]
        return [f.result() for f in futures]


# ------------------------------------------------------------------ evaluation


@dataclasses.dataclass
class EvaluationResult:
    """``score`` is the mean metric in percent (as DSPy reports it); ``results``
    holds ``(example, prediction, score)`` per example (prediction None on error)."""
    score: float
    results: List[Tuple[Example, Optional[Prediction], float]]
    errors: List[Tuple[Example, str]] = dataclasses.field(default_factory=list)

    @property
    def mean(self) -> float:
        return self.score / 100.0

    def __float__(self) -> float:
        return self.score

    def __repr__(self) -> str:
        return (f"EvaluationResult(score={self.score:.2f}, n={len(self.results)}"
                + (f", errors={len(self.errors)}" if self.errors else "") + ")")


def evaluate(program: Any, devset: Iterable[Any], metric: Callable, *, num_threads: int = 1,
             display_progress: bool = False, max_errors: Optional[int] = None, provide_traceback: bool = False,
             call_defaults: Optional[Dict[str, Any]] = None, states: Optional[States] = None) -> EvaluationResult:
    """Run ``program`` on every example and score it with ``metric``. An example
    whose run raises scores 0 (more than ``max_errors`` of them re-raises)."""
    target = _Target(program, call_defaults=call_defaults)
    examples = [as_example(x, target.input_names) for x in devset]

    def one(ex: Example):
        try:
            with with_states(states):
                pred = target.run(ex)
            return ex, pred, score_of(metric, ex, pred), None
        except Exception as exc:  # noqa: BLE001 — counted, reported
            detail = traceback.format_exc() if provide_traceback else f"{type(exc).__name__}: {exc}"
            return ex, None, 0.0, detail

    rows = _parallel(one, examples, num_threads)
    errors = [(ex, err) for ex, _p, _s, err in rows if err is not None]
    if max_errors is not None and len(errors) > max_errors:
        raise RuntimeError(f"evaluation: {len(errors)} examples failed; first: {errors[0][1]}")
    results = [(ex, pred, s) for ex, pred, s, _e in rows]
    score = 100.0 * sum(s for _e, _p, s in results) / len(results) if results else 0.0
    if display_progress:
        print(f"[functai] evaluated {len(results)} examples: {score:.1f}%"
              + (f" ({len(errors)} errors)" if errors else ""))
    return EvaluationResult(round(score, 2), results, errors)


class Evaluate:
    """DSPy-style evaluator: ``Evaluate(devset=..., metric=..., num_threads=4)(program)``."""

    def __init__(self, *, devset: Iterable[Any], metric: Callable, num_threads: int = 1,
                 display_progress: bool = False, max_errors: Optional[int] = None, provide_traceback: bool = False,
                 **_ignored):
        self.devset = list(devset)
        self.metric = metric
        self.num_threads = num_threads or 1
        self.display_progress = display_progress
        self.max_errors = max_errors
        self.provide_traceback = provide_traceback

    def __call__(self, program: Any, metric: Optional[Callable] = None, **kw) -> EvaluationResult:
        return evaluate(program, self.devset, metric or self.metric, num_threads=self.num_threads,
                        display_progress=self.display_progress, max_errors=self.max_errors,
                        provide_traceback=self.provide_traceback, **kw)


# ------------------------------------------------------------------ optimizers


class Optimizer:
    """Base class. ``compile(program, trainset=..., valset=...)`` returns the new
    state of each AI function; it never changes the functions itself."""

    metric: Optional[Callable] = None

    def compile(self, program: Any, *, trainset: Sequence[Any], valset: Optional[Sequence[Any]] = None) -> States:
        raise NotImplementedError


def _labeled_demo(fn: FunctAIFunc, target: _Target, ex: Example) -> Optional[Dict[str, Any]]:
    outs = set(fn._spec().outputs) | {"reasoning"}
    labels = {k: v for k, v in ex.labels().items() if k in outs}
    if not labels:
        return None
    return {"inputs": target.inputs_of(ex), "outputs": labels}


class LabeledFewShot(Optimizer):
    """Up to ``k`` labeled examples become demos (a random sample, or the first ``k``)."""

    def __init__(self, k: int = 16, *, sample: bool = True, seed: int = 0):
        self.k, self.sample, self.seed = k, sample, seed

    def compile(self, program, *, trainset, valset=None) -> States:
        target = _Target(program)
        if not target.single:
            raise TypeError("LabeledFewShot needs labels per AI function; for a @module use BootstrapFewShot")
        fn = target.predictors[0]
        examples = [as_example(x, target.input_names) for x in trainset]
        chosen = random.Random(self.seed).sample(examples, min(self.k, len(examples))) if self.sample \
            else examples[: self.k]
        demos = tuple(d for d in (_labeled_demo(fn, target, ex) for ex in chosen) if d)
        return {fn: dataclasses.replace(fn.state(), demos=demos)}


def _default_metric(examples: Sequence[Example], target: _Target) -> Optional[Callable]:
    """Exact match when the examples are labeled with an output, else None (accept every run)."""
    outs = set()
    for fn in target.predictors:
        outs |= set(fn._spec().outputs)
    outs.add("result")
    if any(k in outs for ex in examples for k in ex.labels()):
        return exact_match
    return None


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

    def _run_one(self, target: _Target, ex: Example, round_idx: int, metric) -> Tuple[bool, list, Optional[str]]:
        try:
            if isinstance(self.teacher, FunctAIFunc) and target.single:
                pred = self.teacher(**target.inputs_of(ex), all=True)
                trace = [(target.predictors[0], pred)]
                own = {"inputs": target.inputs_of(ex), "outputs": dict(pred)}
                ok = metric is None or _passes(score_of(metric, ex, pred, trace), self.metric_threshold)
                return ok, [(target.predictors[0], own)], None
            with forced(**self._teacher_overrides(round_idx)), tracing() as trace:
                pred = target.run(ex)
            ok = metric is None or _passes(score_of(metric, ex, pred, trace), self.metric_threshold)
            return ok, [(fn, p.turn) for fn, p in trace], None
        except Exception as exc:  # noqa: BLE001 — counted against max_errors
            return False, [], f"{type(exc).__name__}: {exc}"

    def bootstrap(self, target: _Target, examples: Sequence[Example], metric) -> Tuple[Dict[FunctAIFunc, list], set]:
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
                rows = _parallel(lambda i: self._run_one(target, examples[i], round_idx, metric), chunk,
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
                    for fn, turn in traced:
                        if fn in boot and len(boot[fn]) < self.max_bootstrapped_demos and turn is not None:
                            boot[fn].append(turn)
        return boot, used

    def compile(self, program, *, trainset, valset=None) -> States:
        target = _Target(program)
        examples = [as_example(x, target.input_names) for x in trainset]
        metric = self.metric if self.metric is not None else _default_metric(examples, target)
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
    """Several candidate demo sets (none, labeled only, bootstrapped, bootstrapped
    from shuffled examples), each scored on ``valset`` (default: the trainset);
    the best one wins. ``candidates`` holds every candidate's score afterwards."""

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
        self.candidates: List[Tuple[float, str]] = []

    def compile(self, program, *, trainset, valset=None) -> States:
        target = _Target(program)
        examples = [as_example(x, target.input_names) for x in trainset]
        val = [as_example(x, target.input_names) for x in (valset or trainset)]
        metric = self.metric if self.metric is not None else (_default_metric(examples, target) or exact_match)
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
                states = BootstrapFewShot(metric, metric_threshold=self.metric_threshold,
                                          max_bootstrapped_demos=size, max_labeled_demos=self.max_labeled_demos,
                                          max_rounds=self.max_rounds, max_errors=self.max_errors,
                                          teacher=self.teacher, num_threads=self.num_threads,
                                          seed=self.seed + seed).compile(program, trainset=shuffled)
            score = evaluate(program, val, metric, num_threads=self.num_threads, states=states).score
            self.candidates.append((score, label))
            if score > best[0]:
                best = (score, states)
            if self.stop_at_score is not None and score >= self.stop_at_score:
                break
        return best[1]


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
    """Instruction candidates (the current one plus proposals written by
    ``prompt_lm`` from the code, the signature and a few examples) × demo sets
    (bootstrapped, unless both demo limits are 0), searched over ``num_trials``
    minibatch evaluations; the top combinations are then scored on the whole
    ``valset`` and the best wins. ``trials`` holds every trial afterwards.

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

    def _instructions(self, fn: FunctAIFunc, examples: Sequence[Example], rng: random.Random) -> List[Optional[str]]:
        from . import meta
        base = fn.state().instructions
        out: List[Optional[str]] = [base]
        proposals: List[str] = []
        sample = list(examples)
        for i in range(self.num_candidates - 1):
            rng.shuffle(sample)
            text = meta.propose_instruction(fn, examples=meta.examples_text(sample), previous=list(proposals),
                                            tip=_TIPS[i % len(_TIPS)], lm=self.prompt_lm,
                                            base_instruction=fn.instructions)
            if text and text not in proposals:
                proposals.append(text)
                out.append(text)
        return out

    def _demo_sets(self, program, target: _Target, examples: Sequence[Example], metric) -> List[Dict[FunctAIFunc, tuple]]:
        base = {fn: tuple(fn.state().demos) for fn in target.predictors}
        if self.max_bootstrapped_demos <= 0 and self.max_labeled_demos <= 0:
            return [base]
        sets = [base]
        for k in range(self.num_candidates - 1):
            shuffled = list(examples)
            random.Random(self.seed + k).shuffle(shuffled)
            states = BootstrapFewShot(metric, metric_threshold=self.metric_threshold,
                                      max_bootstrapped_demos=self.max_bootstrapped_demos,
                                      max_labeled_demos=self.max_labeled_demos, max_errors=self.max_errors,
                                      teacher=self.teacher, num_threads=self.num_threads,
                                      seed=self.seed + k).compile(program, trainset=shuffled)
            sets.append({fn: st.demos for fn, st in states.items()})
        return sets

    def compile(self, program, *, trainset, valset=None) -> States:
        target = _Target(program)
        rng = random.Random(self.seed)
        examples = [as_example(x, target.input_names) for x in trainset]
        val = [as_example(x, target.input_names) for x in (valset or trainset)]
        metric = self.metric if self.metric is not None else (_default_metric(examples, target) or exact_match)
        fns = target.predictors
        instr = {fn: self._instructions(fn, examples, rng) for fn in fns}
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
            score = evaluate(program, batch, metric, num_threads=self.num_threads, states=states_of(combo)).score
            scores.setdefault(combo, []).append(score)
            self.trials.append({"trial": t, "combo": combo, "minibatch_score": score})
        ranked = sorted(scores, key=lambda c: sum(scores[c]) / len(scores[c]), reverse=True)
        finalists = ranked[: self.full_eval_top]
        if len(val) <= self.minibatch_size:
            best_combo = finalists[0]
        else:
            full = {c: evaluate(program, val, metric, num_threads=self.num_threads, states=states_of(c)).score
                    for c in finalists}
            best_combo = max(full, key=full.get)
            for trial in self.trials:
                if trial["combo"] in full:
                    trial["full_score"] = full[trial["combo"]]
        return states_of(best_combo)


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


def optimize(program: Any, *, trainset: Optional[Sequence[Any]] = None, optimizer: Any = None,
             metric: Optional[Callable] = None, valset: Optional[Sequence[Any]] = None,
             call_defaults: Optional[Dict[str, Any]] = None, **opts) -> States:
    """What ``fn.opt(...)`` and ``module.opt(...)`` do: build the trainset (and
    synthesize examples with a teacher when ``n_synth`` > 0), run the optimizer,
    apply the new states (each function keeps the old one for ``undo_opt``)."""
    from . import meta
    from .module import FunctAIModule
    teacher = opts.pop("teacher", None)
    teacher_lm = opts.pop("teacher_lm", None)
    n_synth = int(opts.pop("n_synth", 0) or 0)
    opts.pop("autogen", None)
    target = _Target(program, call_defaults=call_defaults)
    if isinstance(program, FunctAIModule) and call_defaults:
        program._opt_call_defaults = dict(call_defaults)
    examples = [as_example(x, target.input_names) for x in (trainset or [])]
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
            examples.append(Example({**item["inputs"], **item["outputs"]}).with_inputs(*item["inputs"]))
    if not examples:
        raise ValueError("optimization needs examples: pass trainset=[...] (or n_synth with a teacher)")
    choice = optimizer if optimizer is not None else (settings.get("optimizer") or BootstrapFewShot)
    if isinstance(choice, type):
        params = inspect.signature(choice.__init__).parameters
        if "teacher" in params and "teacher" not in opts and (teacher_lm or teacher) is not None:
            opts["teacher"] = teacher_lm or teacher
    opt = _instantiate(choice, metric, opts)
    val = [as_example(x, target.input_names) for x in valset] if valset else None
    states = opt.compile(program, trainset=examples, valset=val)
    import time
    for fn, state in states.items():
        fn._apply_state(state, log={
            "ts": time.time(), "optimizer": type(opt).__name__,
            "metric": getattr(getattr(opt, "metric", None), "__name__", None),
            "n_examples": len(examples), "synthesized": n_synth > 0,
            "examples": list(examples), "n_demos": len(state.demos),
            "instructions_changed": state.instructions != (fn._opt_stack[-1].instructions if fn._opt_stack else None),
        })
    return states


__all__ = ["Optimizer", "LabeledFewShot", "BootstrapFewShot", "BootstrapFewShotWithRandomSearch",
           "InstructionSearch", "Evaluate", "EvaluationResult", "evaluate", "exact_match", "optimize"]
