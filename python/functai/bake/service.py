"""Training a student of an AI function on Tinker (design 21, contract/training.md):
the plan and its price before anything is spent, a run folder that resumes,
five methods, checks by the function's own criteria while it trains, and a
model that answers calls on Tinker exactly as it was trained.

    s1 = functai.bake.bake(scanpath, rows, where="tinker", student="Qwen/Qwen3.5-4B",
                           criteria=[match], held_out_by="screen")
    s2 = functai.bake.bake(scanpath, rows, where="tinker", method="rl", start=s1, reward=match)
    scanpath.using(lm=s2)(screen=..., target=...)

Methods: ``sft`` (imitate the rows' answers, or a teacher's), ``rl`` (a reward: a
criterion, a function of one answer, or a group reward such as
``rewards.energy``), ``prefer`` (pairs of a better and a worse answer: DPO),
``distill`` (on-policy: a teacher scores every token the student writes),
``decision`` (one forward pass gives a probability for each allowed answer).

Uses Tinker's official SDK (``functai[tinker]``; no PyTorch); a student's
tokens come from its published files (``students.py``).
"""

from __future__ import annotations

import concurrent.futures as cf
import dataclasses
import datetime as dt
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import lm15

from . import training as T
from .examples import BakeError, row_inputs
from .functions import Entry, value_hash
from .tokens import Picture, Student, chunks_length, effort_of, student_conversation

METHODS = ("sft", "rl", "prefer", "distill", "decision")
TINKER_MODELS = "https://tinker-docs.thinkingmachines.ai/tinker/models.json"


def _sdk():
    try:
        import tinker
    except ImportError as err:
        raise ImportError('training on Tinker needs its SDK: pip install "functai[tinker]"') from err
    return tinker


def _ready(x: Any) -> Any:
    """A client from what the SDK returns (a client, or one of its older-style futures of one)."""
    tinker = _sdk()
    if isinstance(x, (tinker.BackcompatTrainingClientFuture, tinker.BackcompatSamplingClientFuture)):
        return x.result()
    return x


# ------------------------------------------------------------------ rewards


@dataclasses.dataclass(frozen=True)
class GroupReward:
    """A reward on a whole group of answers to one case: ``fn(answers, references, inputs) -> {"rewards", "score"}``
    (an unreadable answer is None)."""
    name: str
    fn: Callable[[Sequence[Any], Sequence[Any], Mapping[str, Any]], Dict[str, Any]]
    higher_is_better: bool = True


class rewards:
    """Group rewards for ``method="rl"``."""

    @staticmethod
    def energy(distance: Callable[[Any, Any], float], name: str = "energy") -> GroupReward:
        """The energy score (contract/training.md §8): a group of answers rewarded for
        being spread the way the references (the case's right answers) are spread.
        ``distance(a, b)`` is the task's distance between two answers, in [0, 1]."""
        def fn(answers, references, _inputs):
            if not references:
                raise BakeError("the energy score needs each case's references (rows with a known answer)")
            readable = [a is not None for a in answers]
            d_ref = [[distance(a, y) for y in references] if a is not None else [] for a in answers]
            d_pair = [[0.0 if i == j or a is None or b is None else distance(a, b) for j, b in enumerate(answers)]
                      for i, a in enumerate(answers)]
            return T.energy(readable, d_ref, d_pair)
        return GroupReward(name, fn, higher_is_better=False)


# ------------------------------------------------------------------ prices


def tinker_prices() -> Optional[Dict[str, Dict[str, Any]]]:
    """Tinker's price table (its models.json, kept a day): ``{tinker id: {prefill, cached, sample, train, context}}``."""
    from .prices import _fetch

    def dollars(v):
        if isinstance(v, str):
            try:
                return float(v.replace("$", "").replace(",", ""))
            except ValueError:
                return float("nan")
        return float(v) if isinstance(v, (int, float)) else float("nan")
    data = _fetch(TINKER_MODELS, "tinker-models.json")
    if not isinstance(data, list):
        return None
    out = {}
    for m in data:
        ctx = str(m.get("context") or "")
        mult = 1024 if ctx.upper().endswith("K") else 1
        try:
            context = int(float(ctx.rstrip("Kk")) * mult) if ctx else None
        except ValueError:
            context = None
        out[m["tinker_id"]] = {"prefill": dollars(m.get("prefill")), "cached": dollars(m.get("cached_prefill", m.get("prefill"))),
                               "sample": dollars(m.get("sample")), "train": dollars(m.get("train")), "context": context}
    return out


# ------------------------------------------------------------------ cases


@dataclasses.dataclass
class Case:
    """A row, read: its inputs (bound as a call binds them), its answer when known, its case key and side."""
    row: Dict[str, Any]
    inputs: Dict[str, Any]
    outputs: Optional[Dict[str, Any]]
    key: str
    side: str
    weight: float = 1.0


def _side(row: Mapping[str, Any], fn, share: float, held_out_by: Optional[str]) -> str:
    from ..calllog import canonical
    from ..measures import split_of
    if held_out_by:
        if held_out_by not in row:
            raise BakeError(f"held_out_by: a row has no {held_out_by}")
        import lmcc
        return split_of({"conversation": f"{held_out_by}:{canonical(lmcc.turn.to_json(row[held_out_by]))}"}, [],
                        share=share)
    return split_of(row, fn, share=share)


def _cases(fn, entry: Entry, rows: Sequence[Mapping[str, Any]], o: Mapping[str, Any], side: Optional[str] = None
           ) -> List[Case]:
    from ..measures import case_key
    out = []
    for i, row in enumerate(rows):
        row = dict(row)
        try:
            inputs = entry.row_inputs(row) if entry.fixed_values else row_inputs(fn, row)
        except Exception as err:  # noqa: BLE001
            raise BakeError(f"row {i + 1}: {err}") from err
        outs = {k: row[k] for k in entry.outputs if row.get(k) is not None}
        weight = float(row[o["weight"]]) if o.get("weight") else 1.0
        out.append(Case(row, inputs, outs or None, case_key(fn, row),
                        side or _side(row, fn, float(o.get("held_out") if o.get("held_out") is not None else 0.1),
                                      o.get("held_out_by")), weight))
    return out


def conversation(entry: Entry, inputs: Mapping[str, Any], outputs: Optional[Mapping[str, Any]]
                 ) -> Tuple[List[Dict[str, Any]], List[Picture], Optional[str]]:
    """One case as the student reads it: its conversation, pictures, and reply (with outputs)."""
    from .. import engine
    spec = entry.student_spec(entry.spec)
    values = engine.prepare_inputs(spec, {k: v for k, v in inputs.items() if k not in entry.left_out})
    plan = entry.plan
    messages, pictures = student_conversation(plan.render(plan.turn(values)).request("student"))
    if outputs is None:
        return messages, pictures, None
    example = plan.example(values, {k: outputs[k] for k in entry.outputs if k in outputs})
    both, _ = student_conversation(plan.render(plan.turn(values), turns=[example]).request("student"))
    return messages, pictures, [m for m in both if m["role"] == "assistant"][-1]["content"]


def read_outputs(entry: Entry, text: str, cut: bool) -> Optional[Dict[str, Any]]:
    """A reply's outputs as the function's layout reads it, or None when it cannot be read."""
    if cut:
        return None
    import lmcc_lm15
    response = lm15.Response(id=None, model="student", message=lm15.Message.assistant([lm15.TextPart(text)]),
                             finish_reason="stop", usage=lm15.Usage(input_tokens=0, output_tokens=0, total_tokens=0))
    try:
        return dict(lmcc_lm15.read(entry.plan, response).values)
    except Exception:  # noqa: BLE001 — a reply that does not read
        return None


# ------------------------------------------------------------------ the student on Tinker, called


def _model_input(chunks: Sequence[Dict[str, Any]], pictures: Sequence[Picture]):
    types = _sdk().types
    out = []
    for c in chunks:
        if "text" in c:
            out.append(types.EncodedTextChunk(tokens=list(c["text"])))
        else:
            p = pictures[c["picture"]]
            out.append(types.ImageChunk(data=p.data, format=p.format, expected_tokens=c["tokens"]))
    return types.ModelInput(chunks=out)


def _datum(d: Dict[str, Any], pictures: Sequence[Picture]):
    import numpy as np
    types = _sdk().types
    inputs = {}
    for k, v in d["loss_fn_inputs"].items():
        arr = np.asarray(v, dtype=np.int64 if k == "target_tokens" else np.float32)
        inputs[k] = types.TensorData.from_numpy(arr)
    return types.Datum(model_input=_model_input(d["model_input"], pictures), loss_fn_inputs=inputs)


def _seed(*parts: Any) -> int:
    return int(hashlib.sha256(":".join(str(p) for p in parts).encode()).hexdigest()[:8], 16)


class TinkerStudent:
    """Calls of a baked student on Tinker: the request rendered with the student's
    files, sampled, its reply read (training.md §5). A decision model answers the
    likeliest choice (``probabilities`` gives each one's)."""
    name = "tinker"
    fidelity = "tokens"

    def __init__(self, student: Student, sampler: Any, *, temperature: float = 0.0, max_tokens: int = 1024,
                 decision: Optional[Dict[str, Any]] = None, seed: Optional[int] = None, model: str = "student"):
        self.student = student
        self.sampler = sampler
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.decision = decision
        self.seed = seed
        self.model = model

    def _sample(self, prompt, pictures, **kw):
        types = _sdk().types
        return self.sampler.sample(prompt=_model_input(prompt, pictures), num_samples=kw.pop("num_samples", 1),
                                   sampling_params=types.SamplingParams(**kw)).result()

    def probabilities(self, request: Any) -> List[Tuple[Any, float]]:
        """A decision model's probability for each choice (one sampling request, nothing generated)."""
        import numpy as np
        if not self.decision:
            raise BakeError("this model answers by writing a reply; probabilities are a decision model's")
        types = _sdk().types
        messages, pictures = student_conversation(request)
        prefix, tokens = choice_point(self.student, messages, pictures, self.decision["replies"])
        n = chunks_length(prefix)
        rows = [list(tokens) if i == n - 1 else [-1] * len(tokens) for i in range(n)]
        r = self.sampler.sample(prompt=_model_input(prefix + [{"text": [tokens[0]]}], pictures), num_samples=1,
                                sampling_params=types.SamplingParams(max_tokens=1),
                                target_prompt_logprobs=types.TensorData.from_numpy(np.asarray(rows, dtype=np.int64))
                                ).result()
        lp = r.target_prompt_logprobs.tolist()[n - 1]
        return list(zip(self.decision["choices"], T.softmax(lp)))

    def complete(self, request: Any) -> Any:
        if not isinstance(request, dict):
            request = lm15.serde.request_to_dict(request)
        if self.decision:
            probs = self.probabilities(request)
            best = max(range(len(probs)), key=lambda i: probs[i][1])
            text, n_out, stopped, n_in = self.decision["replies"][best], 0, True, 0
        else:
            messages, pictures = student_conversation(request)
            prompt = self.student.render(messages, pictures).prompt
            cfg = request.get("config") or {}
            temperature = cfg.get("temperature")
            temperature = self.temperature if temperature is None else temperature
            kw = {"max_tokens": int(cfg.get("max_tokens") or self.max_tokens), "temperature": float(temperature),
                  "stop": [self.student.stop_token]}
            if self.seed is not None:
                kw["seed"] = _seed(self.seed, temperature, json.dumps(request, sort_keys=True, default=str))
            seq = self._sample(prompt, pictures, **kw).sequences[0]
            tokens = list(seq.tokens)
            text, n_out, stopped, n_in = self.student.read_reply(tokens), len(tokens), seq.stop_reason == "stop", \
                chunks_length(prompt)
        return lm15.Response(id=None, model=self.model, message=lm15.Message.assistant([lm15.TextPart(text)]),
                             finish_reason="stop" if stopped else "length",
                             usage=lm15.Usage(input_tokens=n_in, output_tokens=n_out, total_tokens=n_in + n_out))

    def close(self) -> None:
        pass


def _service(o: Mapping[str, Any], meta: Optional[Dict[str, str]] = None):
    tinker = _sdk()
    t = o.get("tinker") or {}
    if isinstance(t, tinker.ServiceClient):
        return t
    return tinker.ServiceClient(api_key=t.get("api_key"), base_url=t.get("base_url"), user_metadata=meta or {})


def runner_for(baked, *, tinker: Optional[Mapping[str, Any]] = None, student: Optional[Student] = None,
               files: Any = None) -> TinkerStudent:
    """The runner of a model trained here on Tinker: refused when the student's published files changed since
    (calls would not send the tokens it learned)."""
    meta = baked.meta
    tpl = meta["template"]
    if student is None or student.id != meta["student"]:
        student = Student.load(meta["student"], effort=float((tpl.get("kwargs") or {}).get("reasoning_effort", 0.0)),
                               folder=_folder_for(files, meta["student"]))
    if student.template_sha256 != tpl["sha256"] or (tpl.get("tokenizer_sha256")
                                                    and student.tokenizer_sha256 != tpl["tokenizer_sha256"]):
        raise BakeError(f"{meta['student']}'s published files changed since this model was trained (its template or "
                        f"tokenizer): calls would not send the tokens it learned")
    sc = _service({"tinker": tinker}, {"functai": "call", "baked": str(meta["name"])})
    sampler = _ready(sc.create_sampling_client(model_path=meta["weights"]["uri"]))
    return TinkerStudent(student, sampler, temperature=float((meta.get("sampling") or {}).get("temperature", 0.0)),
                         max_tokens=int((meta.get("generation") or {}).get("max_new_tokens") or 1024),
                         decision=(meta.get("training") or {}).get("decision"), model=meta["name"])


def _folder_for(files: Any, student: str) -> Optional[str]:
    if files is None:
        return None
    return files(student) if callable(files) else files


# ------------------------------------------------------------------ decisions


def decision_choices(fn) -> List[Any]:
    """The choices of a function whose answer is a few allowed values (Literal, Enum, bool)."""
    spec = fn._spec()
    out = [f for f in spec.signature.outputs if f.name == spec.main][0]
    shape = out.shape
    if isinstance(shape.get("enum"), list):
        return list(shape["enum"])
    if shape.get("type") == "boolean":
        return [True, False]
    raise BakeError(f'method "decision" is for a function whose answer is one of a few values (a Literal, an Enum, '
                    f'a bool); {fn.__name__}\'s {out.name} is not')


def choice_replies(entry: Entry, answer: str, choices: Sequence[Any], inputs: Mapping[str, Any]) -> List[str]:
    return [conversation(entry, inputs, {answer: ch})[2] for ch in choices]


def choice_point(student: Student, messages, pictures, replies: Sequence[str]):
    """Where the choices' replies first differ after a prompt, and each choice's token there."""
    rendered = [student.render(messages, pictures, r) for r in replies]
    flats = [T.positions(r.full) for r in rendered]
    at = rendered[0].reply_start
    while all(at < len(f) and f[at] == flats[0][at] for f in flats):
        at += 1
    tokens = [f[at] if at < len(f) else None for f in flats]
    if any(t is None for t in tokens) or len(set(tokens)) != len(tokens):
        raise BakeError("the answers' replies are not told apart by one token where they first differ; a decision "
                        "model needs that")
    prefix, pos = [], 0
    for c in rendered[0].full:
        n = len(c["text"]) if "text" in c else c["tokens"]
        if pos + n <= at:
            prefix.append(c)
        elif "text" in c and pos < at:
            prefix.append({"text": c["text"][:at - pos]})
        pos += n
        if pos >= at:
            break
    return prefix, tokens


# ------------------------------------------------------------------ the plan


def _say(log: Any) -> Callable[[str], None]:
    if log is None or log is False:
        return lambda s: None
    if log is True:
        return lambda s: print(f"[bake] {s}", file=sys.stderr, flush=True)
    return log


def _defaults(method: str, n: int, tokens: float, o: Mapping[str, Any]) -> Dict[str, Any]:
    g = o.get
    lr = g("lr") or {"sft": 1e-4, "rl": 4e-5, "prefer": 1e-5, "distill": 1e-4, "decision": 1e-4}[method]
    if g("batch"):
        batch = int(g("batch"))
    elif method in ("sft", "decision"):
        batch = max(1, min(n, max(8, min(32, round(n / 100)))))
    else:
        batch = min(16 if method == "prefer" else 32, max(1, n))
    epochs = int(g("epochs") or ((1 if tokens > 20e6 else 3 if n < 1000 else 2) if method == "sft" else
                                 2 if method == "rl" else 1))
    per = max(1, n // batch)
    steps = per * epochs
    return {"lr": float(lr), "batch": batch, "epochs": epochs, "steps": steps, "steps_per_epoch": per,
            "rank": int(g("lora_rank") or g("rank") or 32), "group": int(g("group") or 4),
            "loss": g("loss") or "importance_sampling", "normalize": True if g("normalize") is None else bool(g("normalize")),
            "kl": float(g("kl") or 0.0), "temperature": float(1.0 if g("temperature") is None else g("temperature")),
            "beta": float(g("beta") or 0.1), "seed": int(g("seed") or 0),
            "eval_every": int(g("eval_every") or max(1, round(steps / 10))),
            "temperatures": [float(t) for t in (g("temperatures") or (0.0, 0.7, 1.0))],
            "schedule": {"warmup": 0.03, "decay": 0.2} if method == "sft" else None,
            "keep_state": True if g("keep_state") is None else bool(g("keep_state"))}


def _unique(cases: Sequence[Case]) -> List[Case]:
    seen, out = set(), []
    for c in cases:
        if c.key not in seen:
            seen.add(c.key)
            out.append(c)
    return out


class TinkerPlan:
    """A bake on Tinker, decided and priced before anything is spent:
    ``print(plan)`` says what it trains on, how, what it costs and what is sent;
    ``plan.run()`` does it."""

    def __init__(self, fn, rows: Sequence[Mapping[str, Any]], o: Dict[str, Any]):
        from ..core import FunctAIFunc
        if not isinstance(fn, FunctAIFunc):
            raise BakeError("bake on Tinker takes one AI function (a module's functions: bake each)")
        self.fn, self.o = fn, o
        self.say = _say(o.get("log", True))
        self.method = o.get("method") or "sft"
        if self.method not in METHODS:
            raise BakeError(f"method is one of {', '.join(METHODS)}, not {self.method!r}")
        start = o.get("start")
        start_meta = start.meta if start is not None and hasattr(start, "meta") else None
        sid = o.get("student") or (start_meta or {}).get("student")
        if not sid:
            raise BakeError('bake on Tinker: name the student (student="Qwen/Qwen3.5-4B"), or start from an earlier bake')
        self.entry = Entry.build(fn, layout=o.get("layout"), reasoning=bool(o.get("reasoning")),
                                 fixed=o.get("fixed") or {}, derived=o.get("derived") or {}, rows=list(rows))
        if start_meta is not None:
            was = [d for d in start_meta["functions"] if d["name"] == fn.__name__]
            if not was or was[0]["fingerprint"] != self.entry.fingerprint:
                raise BakeError(f"start: that model was not trained on {fn.__name__} as it is now")
        start_student = getattr(getattr(start, "_runner", None), "student", None)
        if start_student is not None and start_student.id == sid and o.get("effort") is None:
            self.student = start_student
        else:
            self.student = Student.load(sid, effort=o.get("effort"), folder=_folder_for(o.get("student_files"), sid),
                                        local_files_only=bool(o.get("local_files_only")))
        self.notes: List[str] = []
        if o.get("test") is not None:
            from ..evaluation import rows_of
            self.train = [dataclasses.replace(c, side="train") for c in _cases(fn, self.entry, rows, o)]
            self.held = _cases(fn, self.entry, rows_of(o["test"]), o, side="held_out")
        else:
            every = _cases(fn, self.entry, rows, o)
            self.train = [c for c in every if c.side == "train"]
            self.held = [c for c in every if c.side == "held_out"]
        if self.method == "sft" and not o.get("teacher"):
            missing = sum(c.outputs is None for c in self.train)
            if missing:
                self.notes.append(f"{missing} rows without an answer are left out (teacher= answers them)")
            self.train = [c for c in self.train if c.outputs is not None]
        if self.method == "decision":
            decision_choices(fn)
        if not self.train and self.method != "prefer":
            raise BakeError("no rows to train on")
        self.rendered: Dict[str, Dict[str, Any]] = {}
        longest = reply_max = 0
        train_tokens = prompt_tokens = 0.0
        for c in self.train + self.held:
            msgs, pics, reply = conversation(self.entry, c.inputs, c.outputs)
            r = self.student.render(msgs, pics, reply)
            full = r.full or r.prompt
            self.rendered[self._rkey(c)] = {"chunks": full, "reply_start": r.reply_start, "pictures": pics,
                                            "prompt": r.prompt}
            longest = max(longest, chunks_length(full))
            if r.full:
                reply_max = max(reply_max, chunks_length(r.full) - r.reply_start)
            if c.side == "train":
                train_tokens += chunks_length(full) * c.weight
                prompt_tokens += chunks_length(r.prompt)
        n = len(o.get("pairs") or []) if self.method == "prefer" else \
            len(_unique(self.train)) if self.method in ("rl", "distill") else len(self.train)
        s = _defaults(self.method, n, train_tokens, o)
        s["max_tokens"] = int(o.get("max_tokens") or (max(64, math.ceil(reply_max * 1.25)) if reply_max else 1024))
        s["check_cases"] = min(len(self.held), int(o.get("check_cases") or 100))
        s["effort"] = effort_of(o.get("effort"))
        self.settings = s
        if not self.held:
            self.notes.append("no held-out cases: nothing to choose a checkpoint by; the last one is kept, at temperature 0")
        prices = tinker_prices()
        price = (prices or {}).get(sid)
        if prices and price is None:
            raise BakeError(f"Tinker does not train {sid} (it trains "
                            f"{', '.join(k for k in prices if ':' not in k)[:300]}…)")
        if price and price.get("context") and longest > price["context"]:
            raise BakeError(f"the longest conversation ({longest} tokens) is longer than Tinker's context for {sid} "
                            f"({price['context']})")
        answer = reply_max / 2 if reply_max else s["max_tokens"] / 2
        mean_prompt = prompt_tokens / len(self.train) if self.train else 0
        checks = math.ceil(s["steps"] / s["eval_every"]) + 1
        check_n = s["check_cases"] * len(s["temperatures"])
        training = sampling = 0.0
        if self.method == "sft":
            training = train_tokens * s["epochs"]
        elif self.method == "decision":
            training = train_tokens * s["epochs"] * 2
        elif self.method == "prefer":
            training = len(o.get("pairs") or []) * 2 * (mean_prompt + answer) * (s["epochs"] * 2 + 1)
        else:
            training = s["steps"] * s["batch"] * s["group"] * (mean_prompt + answer)
            if price:
                sampling = s["steps"] * s["batch"] * (mean_prompt * price["prefill"] + s["group"] * answer *
                                                      price["sample"]) / 1e6
                if self.method == "distill" or s["kl"] > 0:
                    sampling += s["steps"] * s["batch"] * s["group"] * (mean_prompt + answer) * price["prefill"] / 1e6
        if price:
            self.dollars = {"training": training * price["train"] / 1e6, "sampling": sampling,
                            "checks": checks * check_n * (mean_prompt * price["cached"] + answer * price["sample"]) / 1e6}
            self.dollars["total"] = sum(self.dollars.values())
        else:
            self.dollars = {"training": None, "sampling": None, "checks": None, "total": None}
            self.notes.append("Tinker's price table could not be read: no price")
        if not reply_max and self.method != "sft":
            self.notes.append("reply lengths are guessed (no known answers); the first check measures them")
        self.tokens = {"train_per_pass": train_tokens, "longest": longest, "reply_max": reply_max}
        reward = o.get("reward")
        fp = {"student": sid, "method": self.method, "settings": s, "entry": self.entry.to_meta(),
              "start": str((start_meta or {}).get("weights", {}).get("state_uri")) if start_meta else
              (start if isinstance(start, str) else None),
              "cases": [[c.key, c.side, value_hash(c.outputs) if c.outputs else None] for c in self.train + self.held],
              "pairs": len(o.get("pairs") or []), "reward": getattr(reward, "name", getattr(reward, "__name__", None))}
        self.fingerprint = hashlib.sha256(json.dumps(fp, sort_keys=True, default=str).encode()).hexdigest()[:12]
        self.name = o.get("name") or fn.__name__
        from .running import runs_home
        short = sid.split(":")[0].split("/")[-1].lower()
        self.folder = Path(o.get("run_folder") or runs_home() / f"{self.name}-{short}-tinker-{self.fingerprint}"
                           ).expanduser().resolve()
        self.output = Path(o.get("path") or self.folder / "baked").expanduser().resolve()

    @staticmethod
    def _rkey(c: Case) -> str:
        return c.key + (":" + value_hash(c.outputs) if c.outputs else "")

    def __repr__(self) -> str:
        s = self.settings

        def money(x):
            return "price unknown" if x is None else f"${x:,.2f}"
        effort = f", effort {self.student.kwargs['reasoning_effort']}" if "reasoning_effort" in self.student.kwargs else ""
        pics = any(r["pictures"] for r in self.rendered.values())
        lines = [f"bake {self.name} → {self.student.id} on Tinker ({self.method}, LoRA r{s['rank']}{effort})",
                 f"  cases      {len(self.train):,} train · {len(self.held):,} held out"
                 + (f" by {self.o['held_out_by']!r}" if self.o.get("held_out_by") else ""),
                 f"  tokens     {int(self.tokens['train_per_pass']):,} trained per pass · replies up to {s['max_tokens']} tokens",
                 f"  training   {s['epochs']} pass(es), {s['steps']} steps of {s['batch']}, lr {s['lr']:.1e}"
                 + (f", {s['group']} answers per case, {s['loss']}" if self.method == "rl" else ""),
                 f"  checks     every {s['eval_every']} steps, {s['check_cases']} cases at temperatures "
                 f"{', '.join(str(t) for t in s['temperatures'])}",
                 f"  money      ≈ {money(self.dollars['total'])} (training {money(self.dollars['training'])} · sampling "
                 f"{money(self.dollars['sampling'])} · checks {money(self.dollars['checks'])})",
                 f"  your data  {len(self.train) + len(self.held)} cases{' and their pictures' if pics else ''} are sent "
                 f"to Thinking Machines",
                 f"  run        {self.folder}"]
        if self.student.source == "tml-renderers":
            lines.append(f"  note: {self.student.id}'s files are not published: its tokens come from tml-renderers "
                         f"with Inkling-Small's template (the same format)")
        lines += [f"  note: {n}" for n in self.notes]
        return "\n".join(lines)

    __str__ = __repr__

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "student": self.student.id, "method": self.method, "where": "tinker",
                "fingerprint": self.fingerprint, "settings": self.settings, "tokens": self.tokens,
                "estimate": self.dollars, "functions": [self.entry.to_meta()], "template": self.student.describe(),
                "output": str(self.output), "notes": self.notes}

    def run(self, *, wait: Optional[bool] = None):
        """Train, check, keep the best, write the model; the Baked model (``wait=False``: a thread you can ``join``)."""
        wait = self.o.get("wait", True) if wait is None else wait
        r = _Run(self)
        if wait:
            return r.go()
        import threading
        t = threading.Thread(target=r.go, daemon=True, name=f"bake-{self.name}")
        t.start()
        return t


# ------------------------------------------------------------------ the run


class _Run:
    def __init__(self, plan: TinkerPlan):
        self.p = plan
        self.say = plan.say
        self.started = time.time()

    # ---- the folder

    def file(self, name: str) -> Path:
        return self.p.folder / name

    def save(self) -> None:
        tmp = self.file("run.json.tmp")
        tmp.write_text(json.dumps(self.st, indent=1, default=str) + "\n")
        tmp.replace(self.file("run.json"))

    def metric(self, **rec: Any) -> None:
        rec["seconds"] = round(time.time() - self.started, 1)
        with self.file("metrics.jsonl").open("a") as f:
            f.write(json.dumps(rec, default=float) + "\n")

    # ---- doing

    def go(self):
        p = self.p
        p.folder.mkdir(parents=True, exist_ok=True)
        done = p.output / "baked.json"
        if self.file("run.json").exists():
            self.st = json.loads(self.file("run.json").read_text())
            if self.st["state"] == "done" and done.exists():
                self.say(f"already baked: {p.output}")
                return self._load()
        else:
            self.st = {"state": "running", "step": 0, "states": [], "best": None, "checks": [],
                       "started": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
            self.file("plan.json").write_text(json.dumps(p.to_dict(), indent=1, default=str) + "\n")
        self.sc = _service(p.o, {"functai": "bake", "bake": p.name, "fingerprint": p.fingerprint})
        tinker = _sdk()
        t = p.o.get("tinker") or {}
        self.rest = tinker.RestClient(api_key=t.get("api_key") if isinstance(t, Mapping) else None,
                                      base_url=t.get("base_url") if isinstance(t, Mapping) else None)
        self.pool = cf.ThreadPoolExecutor(16)
        try:
            self.open()
            getattr(self, {"sft": "sft", "rl": "rollouts", "distill": "rollouts", "prefer": "prefer",
                           "decision": "decision"}[p.method])()
            baked = self.finish()
            self.st["state"] = "done"
            self.save()
            return baked
        except BaseException as err:
            self.st["state"] = "failed"
            self.st["error"] = f"{type(err).__name__}: {err}"
            self.save()
            raise
        finally:
            self.pool.shutdown(wait=False)

    def open(self) -> None:
        p = self.p
        start = p.o.get("start")
        if self.st["states"]:
            last = self.st["states"][-1]
            self.say(f"resuming from step {last['step']}")
            self.tc = _ready(self.sc.create_training_client_from_state_with_optimizer(last["path"]))
            self.st["step"] = last["step"]
        elif start is not None:
            path = start if isinstance(start, str) else (start.meta["weights"].get("state_uri") or "")
            if not path:
                raise BakeError("start: that model kept no training state (bake it with keep_state) to continue from")
            self.tc = _ready(self.sc.create_training_client_from_state(path))
        else:
            self.tc = _ready(self.sc.create_lora_training_client(base_model=p.student.id, rank=p.settings["rank"],
                                                                 seed=p.settings["seed"],
                                                                 user_metadata={"functai": "bake", "bake": p.name}))
        self.st["model_id"] = getattr(self.tc, "model_id", None)
        self.save()

    def lr_at(self, step: int) -> float:
        from .recipe import wsd
        s = self.p.settings
        sched = s["schedule"]
        return s["lr"] * wsd(step, s["steps"], sched["warmup"], sched["decay"]) if sched else s["lr"]

    def order(self, n: int, epoch: int) -> List[int]:
        idx = list(range(n))
        random.Random(self.p.settings["seed"] * 1_000_003 + epoch + 1).shuffle(idx)
        return idx

    def rendered(self, c: Case) -> Dict[str, Any]:
        r = self.p.rendered.get(self.p._rkey(c))
        if r is None:
            msgs, pics, reply = conversation(self.p.entry, c.inputs, c.outputs)
            x = self.p.student.render(msgs, pics, reply)
            r = {"chunks": x.full or x.prompt, "reply_start": x.reply_start, "pictures": pics, "prompt": x.prompt}
        return r

    def seed_for(self, step: int, key: str) -> int:
        return _seed(self.p.settings["seed"], step, key)

    def step(self, data: List[Any], loss: str, lr: float):
        """A step's two requests, sent together; both settled before either's failure is reported."""
        tinker = _sdk()
        fb = self.tc.forward_backward(data, loss)
        op = self.tc.optim_step(tinker.types.AdamParams(learning_rate=lr))
        errors = []
        out = None
        for i, f in enumerate((fb, op)):
            try:
                r = f.result()
                if i == 0:
                    out = r
            except Exception as err:  # noqa: BLE001
                errors.append(err)
        if errors:
            raise errors[0]
        return out

    def forward(self, data: List[Any]):
        return self.tc.forward(data, "cross_entropy").result()

    def after_step(self, step: int) -> None:
        s = self.p.settings
        self.st["step"] = step
        if step % s["eval_every"] == 0 or step == s["steps"]:
            self.check(step)
        every = max(1, round(s["steps"] / 4))
        if step % every == 0 and step < s["steps"]:
            path = self.tc.save_state(f"step-{step:06d}", ttl_seconds=86_400).result().path
            previous = self.st["states"][-1] if self.st["states"] else None
            self.st["states"].append({"step": step, "path": path})
            if previous:
                self._delete(previous["path"])
        self.save()

    def _delete(self, path: Optional[str]) -> None:
        if not path:
            return
        try:
            self.rest.delete_checkpoint_from_tinker_path(path).result()
        except Exception:  # noqa: BLE001 — already gone, or expired
            pass

    # ---- checks

    def sampler(self):
        return _ready(self.tc.save_weights_and_get_sampling_client())

    def check(self, step: int) -> None:
        p = self.p
        if not p.held or any(c["step"] == step for c in self.st["checks"]):
            return
        sampler = self.sampler()
        cases = p.held[:p.settings["check_cases"]]
        scores: Dict[str, Optional[float]] = {}
        agreement = None
        reward = p.o.get("reward")
        if p.method == "rl" and not p.o.get("criteria"):
            # rl is checked as it trains: the mean reward over a group of answers per case (one answer per case is a
            # coin flip at temperature 1), a group reward by its own score
            up = not isinstance(reward, GroupReward) or reward.higher_is_better
            for t in p.settings["temperatures"]:
                got = self.groups(sampler, cases, t, p.settings["group"], step)
                scores[str(t)] = sum(g["score"] if up else -g["score"] for g in got) / len(got) if got else None
                if t == p.settings["temperatures"][-1]:
                    agreement = self.agreement([g["answers"] for g in got])
        else:
            for t in p.settings["temperatures"]:
                scores[str(t)] = self.score(sampler, cases, t, step)
            if p.method == "rl":
                got = self.groups(sampler, cases[:20], p.settings["temperature"], p.settings["group"], step)
                agreement = self.agreement([g["answers"] for g in got])
        self.st["checks"].append({"step": step, "scores": scores, "agreement": agreement})
        self.metric(step=step, check=scores, **({"agreement": agreement} if agreement is not None else {}))
        self.say(f"check at step {step}: " + ", ".join(f"T{t} {'–' if v is None else f'{v:.3f}'}"
                                                         for t, v in scores.items())
                 + (f"; answers alike {agreement:.2f}" if agreement is not None else ""))
        best = T.choose(self.st["checks"])
        old = self.st["best"]
        if best and best["step"] == step and (not old or old["step"] != step):
            name = f"best-step-{step:06d}"
            sampler_path = self.tc.save_weights_for_sampler(name).result().path
            state = self.tc.save_state(f"{name}-state").result().path if p.settings["keep_state"] else None
            self.st["best"] = {**best, "sampler": sampler_path, "state": state}
            if old:
                self._delete(old.get("sampler"))
                self._delete(old.get("state"))
        elif best and old:
            self.st["best"] = {**best, "sampler": old.get("sampler"), "state": old.get("state")}
        self.save()

    @staticmethod
    def agreement(groups: Sequence[Sequence[Any]]) -> Optional[float]:
        per = []
        for g in groups:
            xs = [json.dumps(x, sort_keys=True, default=str) for x in g if x is not None]
            pairs = [(i, j) for i in range(len(xs)) for j in range(i + 1, len(xs))]
            if pairs:
                per.append(sum(xs[i] == xs[j] for i, j in pairs) / len(pairs))
        return sum(per) / len(per) if per else None

    def model_on(self, sampler, temperature: float, step: int):
        """The function on given weights: a model whose calls sample them."""
        meta = self.meta(None)
        runner = TinkerStudent(self.p.student, sampler, temperature=temperature, max_tokens=self.p.settings["max_tokens"],
                               decision=meta["training"].get("decision"), seed=self.seed_for(step, "check"),
                               model=self.p.name)
        return _transient(self.p.entry, meta, runner, self.p.folder)

    def decider(self) -> str:
        p = self.p
        crit = (p.o.get("criteria") or [None])[0]
        if crit is not None:
            return crit.name
        reward = p.o.get("reward")
        if p.method == "rl" and reward is not None:
            return self.reward_name()
        metric = p.o.get("metric")
        return getattr(metric, "__name__", "metric") if metric is not None else "exact_match"

    def reward_name(self) -> str:
        r = self.p.o["reward"]
        return getattr(r, "name", None) or getattr(r, "__name__", "reward")

    def score(self, sampler, cases: Sequence[Case], temperature: float, step: int) -> Optional[float]:
        from ..evaluation import evaluate
        from ..measures import Criterion
        p = self.p
        model = self.model_on(sampler, 0.0, step)
        # never the reply cache: a request a check repeats must be answered by these weights, not an earlier check's
        fn = p.fn.using(lm=model, temperature=temperature, cache_replies=False)
        rows = [c.row for c in cases]
        crit = (p.o.get("criteria") or [None])[0]
        reward = p.o.get("reward")
        if crit is None and p.method == "rl" and isinstance(reward, Criterion):
            crit = reward
        if crit is not None:
            ev = evaluate(fn, rows, criteria=[crit], num_threads=16)
            name = crit.name
        elif p.o.get("metric") is not None:
            ev = evaluate(fn, rows, p.o["metric"], num_threads=16)
            name = ev.metrics[0]
        else:
            ev = evaluate(fn, rows, num_threads=16)
            name = ev.metrics[0]
        per = ev.by_case(name)          # the deciding metric by its name: rows with answers add exact match too
        return sum(per.values()) / len(per) if per else None

    def groups(self, sampler, cases: Sequence[Case], temperature: float, group: int, step: int) -> List[Dict[str, Any]]:
        p = self.p
        answer = p.fn._spec().main
        refs: Dict[str, List[Any]] = {}
        import lmcc
        for c in p.train + p.held:
            if c.outputs and answer in c.outputs:
                refs.setdefault(c.key, []).append(lmcc.turn.to_json(c.outputs[answer]))
        types = _sdk().types

        def one(c: Case):
            r = self.rendered(c)
            res = sampler.sample(prompt=_model_input(r["prompt"], r["pictures"]), num_samples=group,
                                 sampling_params=types.SamplingParams(max_tokens=p.settings["max_tokens"],
                                                                      temperature=temperature,
                                                                      stop=[p.student.stop_token],
                                                                      seed=self.seed_for(step, c.key))).result()
            seqs = [{"tokens": list(q.tokens), "logprobs": list(q.logprobs) if q.logprobs is not None
                     else [0.0] * len(q.tokens), "cut": q.stop_reason != "stop"} for q in res.sequences]
            answers = [read_outputs(p.entry, p.student.read_reply(q["tokens"]), q["cut"]) for q in seqs]
            got = self.reward(c, answers, refs.get(c.key, []))
            return {"case": c, "answers": answers, "seqs": seqs, "rewards": got["rewards"], "score": got["score"],
                    "prompt": r["prompt"], "pictures": r["pictures"]}
        return list(self.pool.map(one, cases))

    def reward(self, c: Case, answers: Sequence[Optional[Dict[str, Any]]], references: Sequence[Any]) -> Dict[str, Any]:
        """Rewards of a group of answers (training.md §7): unreadable answers get the worst value."""
        import lmcc
        from ..measures import Criterion, ask_judge, criterion_of, judged_value, number_of
        p = self.p
        reward = p.o.get("reward")
        answer = p.fn._spec().main
        if reward is None:
            raise BakeError('method "rl" needs a reward: a Criterion with a judge, a function, or a group reward '
                            '(functai.bake.rewards.energy)')
        if isinstance(reward, GroupReward):
            return reward.fn([a.get(answer) if a else None for a in answers], list(references), c.inputs)
        inputs = {k: lmcc.turn.to_json(v) for k, v in c.inputs.items()}
        expected = lmcc.turn.to_json(c.outputs[answer]) if c.outputs and answer in c.outputs else None
        values = []
        for i, a in enumerate(answers):
            if a is None:
                values.append(0.0)
                continue
            args = {**inputs, "inputs": inputs, "outputs": a, "answer": a.get(answer), "expected": expected,
                    "references": list(references), "answers": [x.get(answer) if x else None for x in answers],
                    "index": i}
            if isinstance(reward, Criterion):
                got, _ = ask_judge(reward.judge, args)
                data = criterion_of(reward)
                v = number_of({**data, "value": judged_value(data, got)})
                values.append(0.0 if v is None else float(v))
            else:
                values.append(float(ask_judge(reward, args)[0]))
        return {"rewards": values, "score": sum(values) / max(1, len(values))}

    # ---- the methods

    def sft(self) -> None:
        p = self.p
        s = p.settings
        cases = p.train
        if p.o.get("teacher") and any(c.outputs is None for c in cases):
            cases = self.labelled(cases)
        if self.st["step"] == 0:
            self.check(0)
        for step in range(self.st["step"], s["steps"]):
            epoch, k = divmod(step, s["steps_per_epoch"])
            idx = self.order(len(cases), epoch)[k * s["batch"]:(k + 1) * s["batch"]]
            ds = [T.sft_datum(self.rendered(cases[i])["chunks"], self.rendered(cases[i])["reply_start"],
                              cases[i].weight) for i in idx]
            data = [_datum(d, self.rendered(cases[i])["pictures"]) for d, i in zip(ds, idx)]
            out = self.step(data, "cross_entropy", self.lr_at(step))
            num = den = 0.0
            for o_, d in zip(out.loss_fn_outputs, ds):
                for lp, w in zip(o_["logprobs"].tolist(), d["loss_fn_inputs"]["weights"]):
                    num -= lp * w
                    den += w
            self.metric(step=step + 1, loss=num / den if den else None, lr=self.lr_at(step), examples=len(data))
            self.after_step(step + 1)

    def labelled(self, cases: Sequence[Case]) -> List[Case]:
        from ..core import FunctAIFunc
        p = self.p
        t = p.o["teacher"]
        fn = t if isinstance(t, FunctAIFunc) else p.fn.using(lm=t)
        need = [c for c in cases if c.outputs is None]
        self.say(f"asking the teacher for {len(need)} answers")

        def one(c):
            try:
                return dict(fn.predict(**c.inputs))
            except Exception:  # noqa: BLE001 — a teacher that cannot answer a row: the row is left out
                return None
        got = dict(zip([id(c) for c in need], self.pool.map(one, need)))
        out = []
        for c in cases:
            if c.outputs is not None:
                out.append(c)
            elif got.get(id(c)):
                out.append(dataclasses.replace(c, outputs={k: got[id(c)][k] for k in p.entry.outputs if k in got[id(c)]}))
        return out

    def rollouts(self) -> None:
        """``rl`` and ``distill``: sample, reward (or score each token against the teacher), train."""
        p = self.p
        s = p.settings
        teacher = self.teacher_sampler() if p.method == "distill" else None
        reference = self.sampler() if s["kl"] > 0 else None
        if self.st["step"] == 0:
            self.check(0)
        cases = _unique(p.train)
        types = _sdk().types
        for step in range(self.st["step"], s["steps"]):
            epoch, k = divmod(step, s["steps_per_epoch"])
            batch = [cases[i] for i in self.order(len(cases), epoch)[k * s["batch"]:(k + 1) * s["batch"]]]
            sampler = self.sampler()
            data, skipped, rsum, rn, tokens_out = [], 0, 0.0, 0, 0
            if p.method == "rl":
                got = self.groups(sampler, batch, s["temperature"], s["group"], step)
                kept = []
                for g in got:
                    rsum += sum(g["rewards"])
                    rn += len(g["rewards"])
                    a = T.advantages(g["rewards"], s["normalize"])
                    if a is None:
                        skipped += 1
                    else:
                        kept.append((g, a))
                per_token = None
                if reference is not None and kept:
                    def deltas(ga):
                        g, _a = ga
                        start = chunks_length(g["prompt"])
                        out = []
                        for q in g["seqs"]:
                            lp = reference.compute_logprobs(_model_input(g["prompt"] + [{"text": q["tokens"]}],
                                                                         g["pictures"])).result()
                            out.append([q["logprobs"][j] - (lp[start + j] or 0.0) for j in range(len(q["tokens"]))])
                        return out
                    d = list(self.pool.map(deltas, kept))
                    flat_a = [[a[i]] * len(q["tokens"]) for g, a in kept for i, q in enumerate(g["seqs"])]
                    adj = T.kl_adjusted(flat_a, [x for row in d for x in row], s["kl"])
                    per_token, at = [], 0
                    for g, _a in kept:
                        per_token.append(adj[at:at + len(g["seqs"])])
                        at += len(g["seqs"])
                for gi, (g, a) in enumerate(kept):
                    for i, q in enumerate(g["seqs"]):
                        tokens_out += len(q["tokens"])
                        adv = per_token[gi][i] if per_token is not None else a[i]
                        data.append(_datum(T.rl_datum(g["prompt"], q["tokens"], q["logprobs"], adv), g["pictures"]))
            else:
                def one(c: Case, sampler=sampler, step=step):
                    r = self.rendered(c)
                    res = sampler.sample(prompt=_model_input(r["prompt"], r["pictures"]), num_samples=s["group"],
                                         sampling_params=types.SamplingParams(max_tokens=s["max_tokens"],
                                                                              temperature=s["temperature"],
                                                                              stop=[p.student.stop_token],
                                                                              seed=self.seed_for(step, c.key))).result()
                    tp = self.teacher_prompt(c, r)
                    t_start = chunks_length(tp)
                    out = []
                    for q in res.sequences:
                        toks = list(q.tokens)
                        own = list(q.logprobs) if q.logprobs is not None else [0.0] * len(toks)
                        lp = teacher.compute_logprobs(_model_input(tp + [{"text": toks}], r["pictures"])).result()
                        adv = [-(own[j] - (lp[t_start + j] or 0.0)) for j in range(len(toks))]   # reverse KL, per token
                        out.append((toks, own, adv))
                    return r, out
                for r, outs in self.pool.map(one, batch):
                    for toks, own, adv in outs:
                        rsum += sum(adv) / max(1, len(adv))
                        rn += 1
                        tokens_out += len(toks)
                        data.append(_datum(T.rl_datum(r["prompt"], toks, own, adv), r["pictures"]))
            if data:
                self.step(data, s["loss"], s["lr"])
            self.metric(step=step + 1, reward=rsum / rn if rn else None, groups_skipped=skipped, answers=len(data),
                        tokens_out=tokens_out)
            self.after_step(step + 1)

    def teacher_sampler(self):
        p = self.p
        t = p.o.get("teacher")
        if t is None:
            raise BakeError('method "distill" needs a teacher: a Tinker model id, or a baked model on Tinker')
        if hasattr(t, "meta"):
            if (t.meta.get("template") or {}).get("tokenizer_sha256") != p.student.tokenizer_sha256:
                raise BakeError("distill: the teacher's tokenizer is not the student's (on-policy distillation compares "
                                "token by token); bake with method='sft' and teacher= to learn its answers instead")
            return _ready(self.sc.create_sampling_client(model_path=t.meta["weights"]["uri"]))
        if isinstance(t, str):
            ts = Student.load(t, folder=_folder_for(p.o.get("student_files"), t))
            if ts.tokenizer_sha256 != p.student.tokenizer_sha256:
                raise BakeError(f"distill: {t}'s tokenizer is not {p.student.id}'s (on-policy distillation compares token "
                                f"by token); bake with method='sft' and teacher={t!r} to learn its answers instead")
            return _ready(self.sc.create_sampling_client(base_model=t))
        raise BakeError("distill: the teacher is a Tinker model id or a baked model on Tinker (an AI function teaches "
                        "through method='sft')")

    def teacher_prompt(self, c: Case, r: Mapping[str, Any]):
        """The teacher's prompt: the student's, or with the inputs it leaves out shown (context distillation)."""
        p = self.p
        if not p.entry.left_out:
            return r["prompt"]
        full = Entry.build(p.fn, layout=p.o.get("layout"), reasoning=bool(p.o.get("reasoning")))
        msgs, pics, _ = conversation(full, c.inputs, None)
        return p.student.render(msgs, pics).prompt

    def prefer(self) -> None:
        p = self.p
        s = p.settings
        from ..evaluation import rows_of  # noqa: F401
        pairs = []
        for pair in p.o.get("pairs") or []:
            inputs = row_inputs(p.fn, dict(pair["inputs"]))
            ma, pa, ra = conversation(p.entry, inputs, dict(pair["chosen"]))
            mb, pb, rb = conversation(p.entry, inputs, dict(pair["rejected"]))
            pairs.append({"chosen": p.student.render(ma, pa, ra), "rejected": p.student.render(mb, pb, rb), "pictures": pa})
        if not pairs:
            raise BakeError('method "prefer" needs pairs: [{"inputs": ..., "chosen": ..., "rejected": ...}]')

        def d(r):
            return T.sft_datum(r.full, r.reply_start, 1.0)

        def total(lp, datum):
            return sum(x for x, w in zip(lp, datum["loss_fn_inputs"]["weights"]) if w)
        self.say(f"the starting weights' log-probabilities of {len(pairs)} pairs")
        ref = self.forward([_datum(d(x[k]), x["pictures"]) for x in pairs for k in ("chosen", "rejected")])
        rho = [{"c": total(ref.loss_fn_outputs[2 * i]["logprobs"].tolist(), d(x["chosen"])),
                "r": total(ref.loss_fn_outputs[2 * i + 1]["logprobs"].tolist(), d(x["rejected"]))}
               for i, x in enumerate(pairs)]
        if self.st["step"] == 0:
            self.check(0)
        for step in range(self.st["step"], s["steps"]):
            epoch, k = divmod(step, s["steps_per_epoch"])
            idx = self.order(len(pairs), epoch)[k * s["batch"]:(k + 1) * s["batch"]]
            now = self.forward([_datum(d(pairs[i][k_]), pairs[i]["pictures"]) for i in idx for k_ in ("chosen", "rejected")])
            terms = [{"pi_c": total(now.loss_fn_outputs[2 * j]["logprobs"].tolist(), d(pairs[i]["chosen"])),
                      "rho_c": rho[i]["c"],
                      "pi_r": total(now.loss_fn_outputs[2 * j + 1]["logprobs"].tolist(), d(pairs[i]["rejected"])),
                      "rho_r": rho[i]["r"]} for j, i in enumerate(idx)]
            w = T.dpo(terms, s["beta"])
            data = []
            for j, i in enumerate(idx):
                for kk, wk in (("chosen", "chosen_weight"), ("rejected", "rejected_weight")):
                    r = pairs[i][kk]
                    data.append(_datum(T.weighted_datum(r.full, r.reply_start, lambda _k, v=w[j][wk]: v), pairs[i]["pictures"]))
            self.step(data, "cross_entropy", s["lr"])
            self.metric(step=step + 1, loss=sum(x["loss"] for x in w) / len(w),
                        accuracy=sum(x["z"] > 0 for x in w) / len(w))
            self.after_step(step + 1)

    def decision(self) -> None:
        p = self.p
        s = p.settings
        answer = p.fn._spec().main
        choices = decision_choices(p.fn)
        points = []
        for c in p.train:
            msgs, pics, _ = conversation(p.entry, c.inputs, None)
            prefix, tokens = choice_point(p.student, msgs, pics, choice_replies(p.entry, answer, choices, c.inputs))
            points.append((prefix, tokens, pics))

        def target_of(c: Case) -> List[float]:
            t = c.row.get("target")
            if isinstance(t, Mapping):
                return [float(t.get(str(ch), t.get(ch, 0.0)) or 0.0) for ch in choices]
            y = (c.outputs or {}).get(answer)
            return [1.0 if value_hash(ch) == value_hash(y) else 0.0 for ch in choices]

        def make(i, weights):
            prefix, tokens, pics = points[i]
            return _datum(T.decision_datum(prefix + [{"text": [tokens[0]]}], tokens, weights), pics)
        if self.st["step"] == 0:
            self.check(0)
        for step in range(self.st["step"], s["steps"]):
            epoch, k = divmod(step, s["steps_per_epoch"])
            idx = self.order(len(p.train), epoch)[k * s["batch"]:(k + 1) * s["batch"]]
            now = self.forward([make(i, [0.0] * len(points[i][1])) for i in idx])
            data, loss = [], 0.0
            for j, i in enumerate(idx):
                rows = now.loss_fn_outputs[j]["logprobs"].tolist()
                t = target_of(p.train[i])
                hard = sum(1 for x in t if x == 1.0) == 1 and all(x in (0.0, 1.0) for x in t)
                dw = T.decision_weights(rows[-1], t.index(1.0) if hard else None, None if hard else t)
                loss += dw["loss"]
                data.append(make(i, dw["weights"]))
            self.step(data, "cross_entropy", s["lr"])
            self.metric(step=step + 1, loss=loss / len(idx))
            self.after_step(step + 1)

    # ---- the model

    def meta(self, weights: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        p = self.p
        st = getattr(self, "st", {}) or {}
        decision = None
        if p.method == "decision":
            choices = decision_choices(p.fn)
            decision = {"choices": choices, "replies": choice_replies(p.entry, p.fn._spec().main, choices,
                                                                      p.train[0].inputs) if p.train else []}
        best = st.get("best")
        training = {"method": p.method, "where": "tinker", "steps": st.get("step", 0), "checks": st.get("checks", []),
                    "best": best, "criteria": {self.decider(): best["score"]} if best else {},
                    "trained_on": [self.reward_name()] if p.method == "rl" and p.o.get("reward") is not None else [],
                    "spent": {"estimated": p.dollars.get("total")}, "run_folder": str(p.folder)}
        if decision:
            training["decision"] = decision
        from .baked import FORMAT
        return {"functai_baked": FORMAT, "kind": "generative", "name": p.name, "student": p.student.id,
                "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                "functions": [p.entry.to_meta()],
                "weights": weights or {"form": "remote", "service": "tinker", "uri": "", "base": p.student.id},
                "template": p.student.describe(),
                "generation": {"max_new_tokens": p.settings["max_tokens"],
                               "max_model_len": p.tokens["longest"] + p.settings["max_tokens"]},
                "sampling": {"temperature": best["temperature"] if best else 0.0}, "training": training, "hashes": {}}

    def finish(self):
        p = self.p
        best = self.st.get("best") or {}
        state = best.get("state")
        if not state and p.settings["keep_state"] and not best.get("sampler"):
            state = self.tc.save_state("final").result().path
        for old in self.st["states"]:
            self._delete(old["path"])
        self.st["states"] = [{"step": self.st["step"], "path": state}] if state else []
        sampler = best.get("sampler") or self.tc.save_weights_for_sampler("final").result().path
        weights = {"form": "remote", "service": "tinker", "uri": sampler, "sampler_uri": sampler, "base": p.student.id,
                   "lora_rank": p.settings["rank"]}
        if state:
            weights["state_uri"] = state
        meta = self.meta(weights)
        try:
            tinker = _sdk()
            t = p.o.get("tinker") or {}
            admin = tinker.AdminClient(api_key=t.get("api_key"), base_url=t.get("base_url"))
            start = dt.datetime.fromisoformat(self.st["started"]).replace(minute=0, second=0, microsecond=0)
            end = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
            usage = admin.get_billing_usage(start, end).result()
            billed = sum(e.estimated_cost_usd or 0.0 for e in usage.data)
            meta["training"]["spent"] = {"estimated": p.dollars.get("total"), "billed": billed,
                                         "billed_through": str(usage.cost_data_through)}
        except Exception:  # noqa: BLE001 — billing records come later; the estimate stands
            pass
        p.output.mkdir(parents=True, exist_ok=True)
        from .baked import write_meta
        write_meta(p.output, meta)
        self.say(f"saved to {p.output}" + (f" (best check: step {best['step']} at temperature {best['temperature']}, "
                                           f"{self.decider()} {best['score']:.3f})" if best else ""))
        return self._load()

    def _load(self):
        from .baked import Baked
        b = Baked(self.p.output)
        b._runner = runner_for(b, tinker=self.p.o.get("tinker"), student=self.p.student,
                               files=self.p.o.get("student_files"))
        return b


def _transient(entry: Entry, meta: Dict[str, Any], runner: TinkerStudent, folder: Path):
    """The function on weights being trained (a check): a Baked model made from data, not read from a folder."""
    import threading
    from .baked import Baked
    b = Baked.__new__(Baked)
    b.path, b.meta, b.kind = folder, meta, "generative"
    b.entries = {entry.name: entry}
    b.fields, b._device, b._model, b._tokenizer, b._batcher, b._report = [], None, None, None, None, None
    b._lock = threading.Lock()
    b._runner = runner
    return b


def bake(fn, rows: Any, **o: Any):
    """Train a student of ``fn`` on Tinker (see the module's docs); returns the
    model (``plan_only=True``: the plan, nothing spent)."""
    from ..evaluation import rows_of
    plan = TinkerPlan(fn, rows_of(rows) if rows is not None else [], o)
    plan.say(str(plan))
    if o.get("plan_only"):
        return plan
    return plan.run()


__all__ = ["bake", "TinkerPlan", "TinkerStudent", "GroupReward", "rewards", "tinker_prices", "runner_for", "METHODS"]
