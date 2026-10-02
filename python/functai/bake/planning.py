"""The plan: everything a bake will do, decided before anything is spent.

    plan = functai.bake.plan(summarize, rows)      # or summarize.bake(rows, plan_only=True)
    print(plan)                                    # rows, tokens, student, where, time, price
    plan.using(where="tinker")                     # the same bake, one setting changed
    baked = plan.run()                             # do it

Where it trains (``where="auto"``, the default):

1. the user's preference: ``where=`` (a place or a list in order of
   preference), else ``functai.configure(bake_where=...)``;
2. here, when a GPU here can train the student (the CPU only for small ones);
3. a service that is set up (its key or login present, the account allowed,
   the student offered); with several, the cheapest estimate, the faster one
   on a tie;
4. none: it stops, and the plan says what each place is missing.

Which student (``student=None``): from a short tested list, the largest
that trains where it is going within a day; on a service, the largest one
that service offers.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .examples import BakeError
from .functions import Entry
from .sources import Item

DAY = 86400.0
TRAINING_OPTIONS = ("lora", "lora_rank", "lr", "epochs", "batch", "quantize", "packing", "devices", "device", "liger",
                    "max_new_tokens", "seed", "report_to", "volume")


def _say(log: Any) -> Callable[[str], None]:
    import sys
    if log is None or log is False:
        return lambda s: None
    if log is True:
        return lambda s: print(f"[bake] {s}", file=sys.stderr, flush=True)
    return log


def _count(n: float) -> str:
    """1,234 / 12.3k / 4.5M / 1.2B."""
    n = float(n or 0)
    for div, unit in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if n >= div:
            return f"{n / div:,.1f}{unit}"
    return f"{n:,.0f}"


def _money(x: Optional[float]) -> str:
    if x is None:
        return "price unknown"
    return f"${x:,.2f}" if x >= 0.995 or x == 0 else f"${x:.2f}"


# ------------------------------------------------------------------ sources


@dataclasses.dataclass
class Source:
    """What the examples come from, normalized."""
    kind: str                                    # "functions" | "program" | "examples"
    pairs: List[Tuple[Any, List[Dict[str, Any]]]] = dataclasses.field(default_factory=list)
    program: Any = None
    rows: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    examples: Any = None


def normalize(what: Any, data: Any) -> Source:
    from ..core import FunctAIFunc
    from ..evaluation import rows_of
    from ..module import FunctAIModule
    from .dataset import Examples
    if isinstance(what, Examples):
        if data is not None:
            raise BakeError("bake(examples) takes no data: the examples are the data")
        return Source("examples", examples=what)
    if isinstance(what, (str, Path)) and data is None:
        return Source("examples", examples=Examples.load(what))
    if isinstance(what, Mapping):
        if data is not None:
            raise BakeError("bake({function: rows, ...}) takes no separate data")
        pairs = []
        for fn, rows in what.items():
            if not isinstance(fn, FunctAIFunc):
                raise BakeError(f"bake({{function: rows}}): {fn!r} is not an AI function")
            pairs.append((fn, rows_of(rows)))
        return Source("functions", pairs=pairs)
    if isinstance(what, FunctAIModule):
        return Source("program", program=what, rows=rows_of(data))
    if isinstance(what, FunctAIFunc):
        return Source("functions", pairs=[(what, rows_of(data))])
    raise BakeError(f"bake() takes an AI function, {{function: rows}}, a program (@module), or examples; not "
                    f"{type(what).__name__}")


def _hash_rows(h, rows: Sequence[Mapping[str, Any]]) -> None:
    import lmcc
    from ..calllog import canonical
    for r in rows:
        try:
            plain = lmcc.turn.to_json(dict(r))
        except Exception:  # noqa: BLE001 — a value with no JSON form: its text
            plain = repr(sorted(r.items(), key=lambda kv: kv[0]))
        h.update(canonical(plain).encode() if not isinstance(plain, str) else plain.encode())
        h.update(b"\n")


# ------------------------------------------------------------------ the plan


class Plan:
    """A bake, decided (see the module docstring). ``run()`` does it."""

    def __init__(self, what: Any, data: Any, opts: Dict[str, Any]):
        self._what, self._data, self.opts = what, data, dict(opts)
        self.say = _say(opts.get("log", True))
        self.source = normalize(what, data)
        self.entries: Dict[str, Entry] = {}
        self.items: List[Item] = []
        self.test: Dict[str, List[Dict[str, Any]]] = {}
        self.notes: List[str] = []
        self.program_info: Dict[str, Any] = {}
        self._build()

    # ---- building

    def _entry(self, fn, rows: Sequence[Mapping[str, Any]]) -> Entry:
        o = self.opts
        names = [n for n, _req in fn._named_inputs()]
        fixed = {k: v for k, v in (o.get("fixed") or {}).items() if k in names}
        derived = {k: v for k, v in (o.get("derived") or {}).items() if k in names}
        layout = o.get("layout")
        if isinstance(layout, Mapping) and not ("template" in layout or "messages" in layout or "name" in layout):
            layout = layout.get(fn, layout.get(fn.__name__))
        return Entry.build(fn, layout=layout, reasoning=bool(o.get("reasoning")), fixed=fixed, derived=derived,
                           rows=rows)

    def _build(self) -> None:
        from . import _split
        from .sources import from_program, from_rows
        o = self.opts
        src = self.source
        if src.kind == "examples":
            ex = src.examples
            if not ex.tokenized:
                raise BakeError("these examples carry no tokens; make them with student= "
                                "(functai.bake.examples(..., student=...)) to train on them")
            if o.get("student") and o["student"] != ex.student:
                raise BakeError(f"the examples were tokenized for {ex.student}, not {o['student']}")
            self.examples = ex
            return
        self.examples = None
        pairs = list(src.pairs)
        if src.kind == "program":
            if o.get("plan_only"):
                raise BakeError("a program's examples come from running it (with the teacher), which spends; "
                                "plan_only is not possible for a program. Make the examples first: "
                                "ex = functai.bake.examples(program, rows, teacher=...), then bake(ex, plan_only=True)")
            self.say(f"running the program on {len(src.rows):,} rows to collect its AI calls")
            calls, self.program_info = from_program(src.program, src.rows, teacher=o.get("teacher"),
                                                    functions=o.get("functions"),
                                                    num_threads=o.get("num_threads", 8), log=self.say)
            pairs = []
            for _key, (fn, got) in calls.items():
                rows = []
                for inputs, outputs, rid in got:
                    rows.append({**inputs, **{k: v for k, v in outputs.items()}, "row_id": rid})
                pairs.append((fn, rows))
        labels = o.get("labels", "auto")
        if labels not in ("auto", "data", "teacher"):
            raise BakeError(f"labels is 'auto', 'data' or 'teacher', not {labels!r}")
        for fn, rows in pairs:
            if fn.__name__ in self.entries:
                raise BakeError(f"two functions named {fn.__name__}: a student answers each function by name")
            entry = self._entry(fn, rows)
            self.entries[fn.__name__] = entry
            items = from_rows(entry, rows, tags=o.get("tags"), weights=o.get("weights"), labels=labels)
            if src.kind == "program":
                for it in items:
                    it.source = "program"
            test_rows = o.get("test")
            if isinstance(test_rows, Mapping):
                test_rows = test_rows.get(fn, test_rows.get(fn.__name__))
            if test_rows is not None:
                from ..evaluation import rows_of
                self.test[fn.__name__] = rows_of(test_rows)
            elif o.get("report", True) is not False:
                labeled = [i for i, it in enumerate(items) if it.outputs is not None]
                if len(labeled) >= 60:
                    _rest, held = _split(labeled, float(o.get("holdout", 0.05)), 20, 200, int(o.get("seed") or 0))
                    held_set = set(held)
                    self.test[fn.__name__] = [rows[i] for i in held]
                    items = [it for i, it in enumerate(items) if i not in held_set]
            self.items += items
        known = {n for fn, _rows in pairs for n, _req in fn._named_inputs()}
        for kind in ("fixed", "derived"):
            missing = sorted(set(o.get(kind) or {}) - known)
            if missing:
                raise BakeError(f"{kind}={missing}: no function here has an input by that name "
                                f"(their inputs: {sorted(known)})")
        if labels == "data":
            dropped = sum(it.outputs is None for it in self.items)
            if dropped:
                self.notes.append(f"{dropped:,} rows without answers left out (labels='data')")
            self.items = [it for it in self.items if it.outputs is not None]
        if len([it for it in self.items]) < 10:
            raise BakeError(f"only {len(self.items)} training rows; a generative student needs at least 10 "
                            f"(hundreds to learn well)")
        self.to_label = [it for it in self.items if it.outputs is None]

    # ---- the decisions

    def decide(self) -> "Plan":
        from .hardware import kernels
        from .students import DEFAULT_STUDENTS, info
        from .trainers import PLACES, Job, trainer
        from ..config import effective
        o = self.opts
        local = bool(o.get("local_files_only"))
        where = o.get("where", "auto")
        if where in (None, "auto"):
            where = effective().get("bake_where") or "auto"
        prefs = list(where) if isinstance(where, (list, tuple)) else [where]
        candidates = [o["student"]] if o.get("student") else list(DEFAULT_STUDENTS)
        self.stats_by_student: Dict[str, Dict[str, Any]] = {}
        estimates: Dict[Tuple[str, str], Any] = {}
        infos = {}

        def stats_for(name: str) -> Dict[str, Any]:
            key = self._tokenizer_key(name)
            if key not in self.stats_by_student:
                self.stats_by_student[key] = self._stats(name)
            return self.stats_by_student[key]

        def est(place: str, name: str):
            if (place, name) not in estimates:
                if name not in infos:
                    infos[name] = info(name, local_files_only=local)
                st = infos[name]
                if not st.chat_template:
                    raise BakeError(f"{name} has no chat template: a generative student is a chat (instruct) model")
                s = stats_for(name)
                opts = {k: o.get(k) for k in TRAINING_OPTIONS}
                job = Job(student=st, stats=s, train_rows=s["train_rows"], options=opts)
                estimates[(place, name)] = trainer(place).estimate(job)
            return estimates[(place, name)]

        chosen = None
        reason = ""
        if prefs != ["auto"]:
            for place in prefs:
                if place not in PLACES and not hasattr(place, "estimate"):
                    raise BakeError(f"where= is 'auto', 'here', 'tinker', 'prime', 'export', a list of them, or "
                                    f"a Trainer; not {place!r}")
                pname = place if isinstance(place, str) else getattr(place, "name", "custom")
                for name in candidates:
                    e = est(place, name)
                    if e.ok:
                        chosen = (place, name)
                        reason = "as asked" if len(prefs) == 1 else f"the first place in where={prefs} that can train it"
                        break
                if chosen:
                    break
            if chosen is None:
                lines = []
                for place in prefs:
                    pname = place if isinstance(place, str) else getattr(place, "name", "custom")
                    probs = {p for name in candidates for p in est(place, name).problems}
                    lines.append(f"  {pname}: " + "; ".join(sorted(probs)))
                raise BakeError("cannot train where asked:\n" + "\n".join(lines))
        else:
            here = [(name, est("here", name)) for name in candidates]
            fits = [(n, e) for n, e in here if e.ok]
            within = [(n, e) for n, e in fits if e.seconds is None or e.seconds <= DAY]
            if within:
                chosen, reason = ("here", within[0][0]), "free: this machine can train it"
            elif fits:
                chosen = ("here", fits[-1][0])
                reason = "free: this machine can train it (slowly; the plan shows the cloud's price)"
            else:
                services = [p for p in ("tinker", "prime") if trainer(p).set_up()]
                pick = None
                for name in candidates:
                    ok = [(p, est(p, name)) for p in services if est(p, name).ok]
                    if ok:
                        ok.sort(key=lambda pe: (pe[1].dollars is None, pe[1].dollars or 0,
                                                pe[1].seconds if pe[1].seconds is not None else math.inf))
                        pick = (ok[0][0], name)
                        if len(ok) > 1:
                            reason = (f"the cheapest of the services set up ({', '.join(p for p, _ in ok)}); "
                                      f"configure(bake_where=...) to prefer one")
                        else:
                            reason = f"this machine cannot train it, and {ok[0][0]} is set up"
                        break
                if pick is None:
                    lines = ["  here: " + "; ".join(sorted({p for _n, e in here for p in e.problems}))]
                    for p in ("tinker", "prime"):
                        probs = {x for name in candidates for x in est(p, name).problems}
                        lines.append(f"  {p}: " + ("; ".join(sorted(probs)) if probs else "can train it"))
                    raise BakeError("nowhere to train this bake:\n" + "\n".join(lines) +
                                    "\nSet one up, or pass where= and student= (where='export' writes the "
                                    "examples for another tool)")
                chosen = pick
        self.where, self.student = chosen
        self.reason = reason
        self.student_info = infos[self.student]
        self.estimate = est(*chosen)
        self.stats = stats_for(self.student)
        self.alternatives = {}
        for p in ("here", "tinker", "prime"):
            if p == self.where:
                continue
            try:
                self.alternatives[p] = est(p, self.student)
            except Exception as exc:  # noqa: BLE001 — an alternative that cannot even estimate
                from .trainers import Estimate
                self.alternatives[p] = Estimate(False, [str(exc)], summary="unavailable")
        self.kernels = kernels()
        self.teacher_cost = None
        if self.to_label:
            from .sources import teacher_estimate
            self.teacher_cost = teacher_estimate(self.to_label, o.get("teacher"),
                                                 answer_tokens=self.stats.get("answer_known_mean"))
        self.fingerprint = self._fingerprint()
        return self

    def _tokenizer_key(self, name: str) -> str:
        # the Qwen3.5 students share one tokenizer and template: measure once
        return "qwen3.5" if name in ("Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B") \
            else name

    def _stats(self, student: str) -> Dict[str, Any]:
        """Token counts for the data under ``student``'s template: prompts for
        every row, answers for rows that have them (the others are estimated
        from those, or counted as unknown)."""
        from .dataset import validation_count
        if self.examples is not None:
            ex = self.examples
            tr = ex.split("train")
            s = tr.stats()
            s["train_rows"] = len(tr)
            s["val_tokens"] = ex.split("validation").stats().get("tokens", 0)
            s["answer_known_mean"] = None
            s["unknown_answers"] = 0
            return s
        from .template import describe, load_tokenizer, prompt_ids
        tok = load_tokenizer(student, local_files_only=bool(self.opts.get("local_files_only")))
        tpl = describe(tok)
        sample = self.items if len(self.items) <= 4000 else random.Random(0).sample(self.items, 4000)
        scale = len(self.items) / len(sample)
        raw_prompts, answers, weights = [], [], []
        for it in sample:
            msgs, answer = it.entry.messages(it.inputs, it.outputs)
            raw_prompts.append(len(prompt_ids(tok, msgs, tpl.kwargs)))
            weights.append(it.weight)
            if answer is not None:
                answers.append((len(tok(answer + tpl.end, add_special_tokens=False)["input_ids"]), it.weight))
        known = [a for a, _w in answers]
        mean_answer = statistics.mean(known) if known else None
        unknown_w = sum(it.weight for it in sample if it.outputs is None)
        unknown = sum(1 for it in sample if it.outputs is None)
        a_tokens = sum(a * w for a, w in answers) + unknown_w * (mean_answer or 0)
        p_tokens = sum(p * w for p, w in zip(raw_prompts, weights))
        p_sorted = sorted(raw_prompts)
        a_sorted = sorted(known)

        def pct(xs, q):
            return int(xs[min(len(xs) - 1, max(0, math.ceil(q * len(xs)) - 1))]) if xs else 0
        longest_answer = max(known) if known else 0
        n_val = validation_count(len(self.items), float(self.opts.get("validation", 0.02)))
        total = (p_tokens + a_tokens) * scale
        return {
            "rows": len(self.items), "train_rows": len(self.items) - n_val, "validation_rows": n_val,
            "tokens": int(total * (len(self.items) - n_val) / max(1, len(self.items))),
            "val_tokens": int(total * n_val / max(1, len(self.items))),
            "answer_tokens": int(a_tokens * scale),
            "prompt": {"median": int(statistics.median(raw_prompts)), "p90": pct(p_sorted, 0.9),
                       "p99": pct(p_sorted, 0.99), "max": int(max(raw_prompts))},
            "answer": {"median": int(statistics.median(known)) if known else None, "p90": pct(a_sorted, 0.9),
                       "p99": pct(a_sorted, 0.99), "max": longest_answer} if known else {"max": 0},
            "longest": int(max(raw_prompts) + (longest_answer or 0)) if known else int(max(raw_prompts) * 1.5),
            "answer_known_mean": mean_answer, "unknown_answers": int(unknown * scale), "sampled": scale > 1,
        }

    def _fingerprint(self) -> str:
        """What makes this bake this bake: the functions as the student reads
        them, the data, the teacher, the student, where, and the settings."""
        h = hashlib.sha256()
        h.update(json.dumps({"student": self.student, "where": self.where,
                             "opts": {k: self.opts.get(k) for k in TRAINING_OPTIONS + ("labels", "validation",
                                                                                        "holdout", "tags", "weights")},
                             "teacher": str(self.opts.get("teacher")),
                             "functions": {n: e.to_meta() for n, e in sorted(self.entries.items())}},
                            sort_keys=True, default=str).encode())
        if self.examples is not None:
            for r in self.examples.rows:
                h.update(bytes(memoryview(r["input_ids"]).cast("B")) if hasattr(r["input_ids"], "typecode")
                         else json.dumps(list(r["input_ids"])).encode())
        else:
            for it in self.items:
                _hash_rows(h, [{"i": it.inputs, "o": it.outputs, "w": it.weight, "t": it.tag}])
        return h.hexdigest()[:12]

    @property
    def name(self) -> str:
        if self.opts.get("name"):
            return self.opts["name"]
        names = sorted(self.entries) if self.entries else sorted((self.examples.entries or {}).keys())
        return "+".join(names) if len(names) <= 3 else f"{names[0]}+{len(names) - 1}"

    @property
    def folder(self) -> Path:
        from .running import runs_home
        if self.opts.get("run_folder"):
            return Path(self.opts["run_folder"]).expanduser().resolve()
        short = Path(self.student).name.lower()
        return runs_home() / f"{self.name}-{short}-{self.where if isinstance(self.where, str) else 'custom'}-" \
                             f"{self.fingerprint}"

    @property
    def output(self) -> Path:
        if self.opts.get("path"):
            return Path(self.opts["path"]).expanduser().resolve()
        return self.folder / ("export" if self.where == "export" else "baked")

    # ---- showing

    def __repr__(self) -> str:
        e = self.estimate
        s = self.stats
        st = self.student_info
        r = e.settings
        how = "LoRA r{} ".format(r.get("lora_rank")) if r.get("lora") else "all weights "
        prec = r.get("precision")
        head = (f"bake {self.name} → {self.student} ({_count(st.parameters)} parameters, sft, {how.strip()}"
                + (f", {prec}" if prec and prec != "service" else "") + (", 4-bit base" if r.get("quantize") else "")
                + ")")
        lines = [head]
        tr, va = s.get("train_rows", 0), s.get("validation_rows", s.get("rows", 0) - s.get("train_rows", 0))
        label = ""
        if self.to_label:
            c = self.teacher_cost or {}
            price = _money(c.get("dollars")) if c.get("output_known") else (
                _money(c.get("dollars")) + " for the prompts, plus the answers" if c.get("dollars") is not None
                else "price unknown")
            label = f" · {len(self.to_label):,} answered by {c.get('teacher', 'the teacher')} ≈ {price}"
        elif self.examples is None:
            label = " · answers from the data"
        tests = sum(len(v) for v in self.test.values())
        lines.append(f"  rows       {tr:,} train, {va:,} validation" + (f", {tests:,} test" if tests else "") + label)
        if len(self.entries) > 1:
            lines.append("  functions  " + ", ".join(f"{n}" for n in self.entries))
        left = sorted({n for e_ in self.entries.values() for n in e_.left_out})
        if left:
            lines.append(f"  left out   {', '.join(left)} (fixed or derived: not in the student's prompt)")
        p, a = s.get("prompt", {}), s.get("answer", {})
        tok = f"  tokens     {_count(s.get('tokens', 0))} per pass (prompts {p.get('median', 0):,} median / " \
              f"{p.get('p99', 0):,} p99"
        if a.get("median") is not None:
            tok += f"; answers {a['median']:,} / {a.get('p99', 0):,}"
        if s.get("unknown_answers"):
            tok += f"; {s['unknown_answers']:,} answers not written yet, counted at the mean"
        lines.append(tok + ")" + (" (from a sample)" if s.get("sampled") else ""))
        ctx = st.context
        lines.append(f"  longest    {s.get('longest', 0):,} tokens"
                     + (f" · fits the student's context ({ctx:,})" if ctx and s.get("longest", 0) <= ctx else
                        f" · LONGER than the student's context ({ctx:,})" if ctx else ""))
        lines.append(f"  where      {self.where}: {e.summary}" + (f"   ({self.reason})" if self.reason else ""))
        for pname, alt in self.alternatives.items():
            what = alt.summary if alt.ok else "; ".join(alt.problems[:2]) or alt.summary
            lines.append(f"             {pname}: {what}")
        r_ = e.settings
        if r_.get("learning_rate"):
            replies = (f"replies up to {r_.get('max_new_tokens'):,} tokens" if (s.get("answer") or {}).get("max")
                       else "reply length from the teacher's answers")
            lines.append(f"  training   {r_.get('epochs')} pass(es), about {r_.get('steps'):,} steps, "
                         f"lr {r_['learning_rate']:.1e} (warmup, constant, decay), {replies}")
        if self.where == "here":
            k = self.kernels
            marks = [f"flash attention {'✓' if k['flash_attn'] else '✗'}"]
            if st.hybrid:
                marks.append(f"flash-linear-attention {'✓' if k['flash_linear_attention'] else '✗'}")
                marks.append(f"causal-conv1d {'✓' if k['causal_conv1d'] else '✗'}")
            lines.append("  kernels    " + "  ".join(marks))
        lines.append(f"  run        {self.folder}" + ("  (exists: resumes)" if (self.folder / "plan.json").exists()
                                                       else ""))
        for n in self.notes + list(e.notes):
            lines.append(f"  note: {n}")
        return "\n".join(lines)

    __str__ = __repr__

    def using(self, **changes) -> "Plan":
        """The same bake with some settings changed (decided again)."""
        return Plan(self._what, self._data, {**self.opts, **changes}).decide()

    def to_dict(self) -> Dict[str, Any]:
        """What plan.json holds (what the run's process reads)."""
        from .template import describe, load_tokenizer
        tpl = self.examples.template if self.examples is not None else describe(
            load_tokenizer(self.student, local_files_only=bool(self.opts.get("local_files_only")))).to_dict()
        functions = [e.to_meta() for e in self.entries.values()] if self.entries else \
            [v if isinstance(v, dict) else v.to_meta() for v in self.examples.entries.values()]
        return {"name": self.name, "student": self.student, "where": self.where if isinstance(self.where, str)
                else getattr(self.where, "name", "custom"), "fingerprint": self.fingerprint,
                "settings": {**self.estimate.settings, **{k: self.opts[k] for k in ("volume",) if self.opts.get(k)}},
                "functions": functions, "template": tpl, "student_info": self.student_info.to_dict(),
                "output": str(self.output), "local_files_only": bool(self.opts.get("local_files_only")),
                "merge": self.opts.get("merge", True), "estimate": self.estimate.to_dict(),
                "inline": bool(self.opts.get("inline"))}

    # ---- doing

    def run(self, *, wait: Optional[bool] = None):
        """Answer the rows that need the teacher, write the examples, and train.
        Returns the baked model (``wait=False``: the Run, at once)."""
        from .dataset import make
        from .running import Run
        o = self.opts
        wait = o.get("wait", True) if wait is None else wait
        folder = self.folder
        existing = folder / "plan.json"
        if existing.exists():
            run = Run(folder)
            if run.state == "done" and Path(run.plan["output"]).joinpath("baked.json").exists():
                self.say(f"already baked: {run.plan['output']}")
                return self._finish(run) if wait else run
            self.say(f"resuming the run in {folder}")
            run.start(process=not o.get("inline"))
            return self._finish(run) if wait else run
        folder.mkdir(parents=True, exist_ok=True)
        if self.examples is not None:
            ex = self.examples
        else:
            if self.to_label:
                from .sources import label
                c = self.teacher_cost or {}
                self.say(f"asking {c.get('teacher', 'the teacher')} for {len(self.to_label):,} answers "
                         f"(≈ {_money(c.get('dollars'))})")
                self.labeling = label(self.items, o.get("teacher"), num_threads=o.get("num_threads", 16),
                                      log=self.say, keep=folder / "teacher.jsonl")
            else:
                self.labeling = {}
            from .template import describe, load_tokenizer
            tok = load_tokenizer(self.student, local_files_only=bool(o.get("local_files_only")))
            self.say("writing the training conversations")
            ex = make(self.items, tokenizer=tok, template=describe(tok), student=self.student,
                      validation=float(o.get("validation", 0.02)), seed=int(o.get("seed") or 0),
                      info={"labeling": self.labeling, "program": self.program_info}, log=self.say)
            self._refit(ex)
        ex.save(folder / "examples.parquet")
        plan = self.to_dict()
        plan["info"] = ex.info
        run = Run.create(folder, plan)
        self.say(f"training on {self.where}: {self.estimate.summary}")
        run.start(process=not o.get("inline"))
        if not wait:
            return run
        return self._finish(run)

    def _refit(self, ex) -> None:
        """Decide the settings again with the real answers' lengths (the
        teacher's answers were estimated); the place stays."""
        from .trainers import Job, trainer
        s = ex.split("train").stats()
        s["train_rows"] = len(ex.split("train"))
        s["val_tokens"] = ex.split("validation").stats().get("tokens", 0)
        self.stats = {**self.stats, **s}
        job = Job(student=self.student_info, stats=s, train_rows=s["train_rows"],
                  options={k: self.opts.get(k) for k in TRAINING_OPTIONS})
        e = trainer(self.where).estimate(job)
        if not e.ok:
            raise BakeError(f"with the teacher's answers, {self.where} can no longer train it: "
                            + "; ".join(e.problems))
        self.estimate = e

    def _finish(self, run):
        from .judging import judge
        baked = run.wait(show=self.say if self.opts.get("log", True) is not False else False)
        if baked is None:           # exported
            self.say(f"examples and recipes written to {run.plan['output']}")
            return run
        if self.opts.get("report", True) is not False and self.test and not baked.meta.get("report"):
            fns = {e.name: e.fn for e in self.entries.values() if e.fn is not None}
            if fns:
                judge(baked, {fns[n]: rows for n, rows in self.test.items() if n in fns},
                      metric=self.opts.get("metric"), teacher=self.opts.get("teacher"),
                      compare_teacher=bool(self.opts.get("compare_teacher")),
                      num_threads=self.opts.get("num_threads", 16), log=self.say, save=True,
                      labeling=getattr(self, "labeling", None) or (run.plan.get("info") or {}).get("labeling"))
        self.say(f"saved to {baked.path}")
        return baked


def plan(what: Any, data: Any = None, **opts) -> Plan:
    """Decide a generative bake without running anything (see the module docstring)."""
    return Plan(what, data, opts).decide()


__all__ = ["Plan", "plan", "normalize", "TRAINING_OPTIONS"]
