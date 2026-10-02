"""Bake an AI function into weights you own.

    baked = summarize.bake(rows)                 # decided from the data, the function and the machine
    fast = summarize.using(lm=baked)             # the same function, on the weights
    print(summarize.bake(rows, plan_only=True))  # what it would do, how long, what it costs: nothing spent

Two kinds of model:

- a **head** (``method="head"``) for functions whose every output has a fixed
  set of answers (Literal, Enum, bool): a small encoder reads the input and
  gives a probability for every answer. Trains in seconds, even on a CPU.
- a **generative student** (``method="sft"``) for everything else: a small chat
  model trained on the exact requests the function's layout writes, the loss
  on the reply only. It trains here (TRL, PEFT, every free GPU), on Tinker,
  on Prime Intellect, or anywhere else (``where="export"``).

``method="auto"`` (the default) picks the head when it can answer.

The layers, each usable alone:

    functai.bake.examples(fn, rows, student=...)   the training conversations (a table every trainer reads)
    functai.bake.plan(fn, rows)                    what a bake would do, decided, nothing spent
    functai.bake.bake(fn, rows, wait=False)        a Run: a folder and a process you can leave
    functai.bake.runs() / run(folder)              every run here; reattach
    functai.bake.judge(baked, fn, rows, metric=)   evaluate the student with any metric or AI judge
    functai.bake.adopt(folder, fn, examples=)      a model trained elsewhere, checked and used
    baked.on("vllm" | "tinker" | url)              where the weights answer

Training here needs ``pip install "functai[bake]"``; on Tinker, ``"functai[tinker]"``.
"""

from __future__ import annotations

import random
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .baked import Baked, is_baked, load
from .examples import BakeError, HeadField, answer_key, distribution, field_value, one_hot  # noqa: F401
from .report import BakeReport, FieldScores

__all__ = ["bake", "examples", "plan", "adopt", "judge", "runs", "run", "Baked", "load", "BakeError", "BakeReport",
           "is_baked", "Examples", "Plan", "Run"]


def __getattr__(name: str):
    # the classes load on first use: importing functai never imports PyTorch or the trainers
    if name == "Plan":
        from .planning import Plan
        return Plan
    if name == "Run":
        from .running import Run
        return Run
    if name == "Examples":
        from .dataset import Examples
        return Examples
    raise AttributeError(name)


def plan(what: Any, data: Any = None, **options):
    """What a generative bake would do, decided, with nothing spent.

    The same as ``bake(what, data, method="sft", plan_only=True, **options)``
    (and as ``fn.bake(rows, plan_only=True)`` when the function needs a
    generative student). Printed, the plan says the rows and how many a
    teacher must answer (and what that costs), the tokens per pass, the
    student, where it would train with the time or the price of each place
    set up, the training settings, missing speed-up kernels, and the run's
    folder.

    Parameters
    ----------
    what, data
        As for ``bake``: an AI function and its rows, ``{fn: rows}``, a
        ``@module`` and its inputs, or ``Examples``.
    **options
        Anything ``bake`` takes (``student=``, ``where=``, ``teacher=``,
        ``fixed=``, training settings...).

    Returns
    -------
    Plan
        ``print(plan)`` to read it, ``plan.using(where="tinker")`` for the
        same bake with one setting changed, ``plan.run()`` to do it.

    Examples
    --------
    ```python
    # not run: a plan reads the rows and the machine (the guide Bake it into a small model shows one)
    plan = functai.bake.plan(summarize, rows)
    print(plan)
    baked = plan.using(where="tinker").run()
    ```
    """
    return bake(what, data, method="sft", plan_only=True, **options)


def runs(home: Any = None):
    """Every bake run on this machine, newest first.

    A run is a folder (and, while it trains, a process of its own) under
    ``~/.cache/functai/bakes`` (``$XDG_CACHE_HOME/functai/bakes``), so runs
    started from a notebook that has since closed are listed too.

    Parameters
    ----------
    home : str or path, optional
        Another folder of runs.

    Returns
    -------
    list of Run
        Each with ``state`` (``running``, ``stopped``, ``done``,
        ``failed``...), ``metrics()``, ``wait()``, ``stop()``, ``resume()``.
    """
    from .running import runs as _runs
    return _runs(home)


def run(folder: Any):
    """A bake run, from its folder: reattach to it after a restart.

    Parameters
    ----------
    folder : str or path
        The run's folder (``run.folder``; listed by ``functai.bake.runs()``).

    Returns
    -------
    Run

    Examples
    --------
    ```python
    # not run: it needs a run on this machine
    run = functai.bake.run("~/.cache/functai/bakes/summarize-qwen3.5-2b-here-7f3a2c91d0e4")
    run.metrics()            # the loss curve so far
    baked = run.wait()       # the model, when it is done
    ```
    """
    from .running import Run
    return Run(folder)


def _say(log: Optional[Callable[[str], None]]) -> Callable[[str], None]:
    if log is None or log is False:
        return lambda s: None
    if log is True:
        return lambda s: print(f"[bake] {s}", file=sys.stderr, flush=True)
    return log


def _gold(row: Dict[str, Any], fields: Sequence[HeadField]) -> Optional[List[List[float]]]:
    """The row's own labels as targets (a ``<field>__probs`` column is a soft
    label), or None when a field has none."""
    out = []
    for f in fields:
        probs = row.get(f"{f.name}__probs")
        if isinstance(probs, dict):
            out.append(distribution(f, probs))
            continue
        value = field_value(row, f.name)
        if value is None:
            return None
        try:
            out.append(one_hot(f, value))
        except KeyError:
            raise BakeError(f"a row's {f.name!r} is {value!r}, which is not one of the answers "
                            f"{list(f.keys)[:8]}{'...' if len(f.keys) > 8 else ''}") from None
    return out


def _gold_values(row: Dict[str, Any], outputs: Sequence[str]) -> Optional[Dict[str, Any]]:
    """A row's own values for every output, or None when one is missing."""
    if any(row.get(n) is None for n in outputs):
        return None
    return {n: row[n] for n in outputs}


def _split(items: List[Any], share: float, lo: int, hi: int, seed: int) -> Tuple[List[Any], List[Any]]:
    items = list(items)
    random.Random(seed).shuffle(items)
    n = min(hi, max(lo, round(share * len(items)))) if items else 0
    n = min(n, len(items) // 2)
    return items[n:], items[:n]


def _teacher_labels(fn, teacher: Any, rows: List[Dict[str, Any]], fields: Sequence[HeadField], *,
                    num_threads: int, prices: Optional[Dict[str, Any]], say) -> Tuple[List[Optional[List[List[float]]]], Dict[str, Any]]:
    """Run the teacher on each row. Its probabilities when it gives them (soft
    targets), else its answer (hard). Returns targets (None for rows that failed)
    and what labeling cost."""
    from ..core import FunctAIFunc
    from ..evaluation import parallel
    from .examples import row_inputs
    tfn = teacher if isinstance(teacher, FunctAIFunc) else fn.using(lm=teacher)
    name = getattr(teacher, "__name__", None) if isinstance(teacher, FunctAIFunc) else getattr(teacher, "name", teacher)
    soft = [0]
    tokens = {"input_tokens": 0, "output_tokens": 0}
    failures: List[str] = []
    done = [0]
    lock = threading.Lock()        # the counters are shared by the labeling threads
    t0 = time.time()

    def one(row: Dict[str, Any]) -> Optional[List[List[float]]]:
        try:
            pred = tfn._invoke((), row_inputs(fn, row), full=True)
        except Exception as exc:  # noqa: BLE001 — a failed row is dropped and counted
            failures.append(f"{type(exc).__name__}: {exc}")
            return None
        usage = pred.usage
        with lock:
            for k in tokens:
                tokens[k] += usage.get(k, 0)
        out = []
        for f in fields:
            dist = (pred.probabilities or {}).get(f.name)
            try:
                if dist:
                    out.append(distribution(f, dist))
                    with lock:
                        soft[0] += 1
                else:
                    out.append(one_hot(f, field_value(pred, f.name)))
            except (KeyError, ValueError):
                failures.append(f"answer {field_value(pred, f.name)!r} is not one of the answers for {f.name}")
                return None
        with lock:
            done[0] += 1
            n = done[0]
        if n % 500 == 0:
            say(f"teacher: {n:,}/{len(rows):,} rows")
        return out

    targets = parallel(one, rows, max(1, num_threads))
    seconds = time.time() - t0
    ok = sum(t is not None for t in targets)
    info: Dict[str, Any] = {"teacher": name, "rows": ok, "failed": len(rows) - ok, "seconds": round(seconds, 1),
                            "soft": soft[0] >= ok * len(fields) and ok > 0,
                            "tokens_per_row": (tokens["input_tokens"] + tokens["output_tokens"]) / max(1, ok),
                            "seconds_per_row": seconds / max(1, ok), "concurrency": num_threads}
    price = (prices or {}).get("teacher")
    if price:
        info["dollars"] = (tokens["input_tokens"] * price[0] + tokens["output_tokens"] * price[1]) / 1e6
        info["dollars_per_row"] = info["dollars"] / max(1, ok)
    if failures:
        info["first_failure"] = failures[0]
        say(f"teacher: {len(failures)} rows failed; first: {failures[0]}")
    return targets, info


def bake_head(fn, data: Any, *, student: str = "jhu-clsp/ettin-encoder-17m",
              teacher: Any = None, labels: str = "auto", test: Any = None, holdout: float = 0.1,
              validation: float = 0.1, epochs: Optional[int] = None, lr: Optional[float] = None,
              batch_size: int = 32, max_length: Optional[int] = None, device: Optional[str] = None, seed: int = 0,
              num_threads: int = 16, path: "str | Path | None" = None, compare_teacher: bool = False,
              prices: Optional[Dict[str, Any]] = None, local_files_only: bool = False,
              log: Any = True) -> Baked:
    """A head model for ``fn`` (``bake(..., method="head")``); returns the baked model.

    - ``data``: rows (a list of dicts, or any table ``dpyr.read`` takes). Input
      columns are named like the parameters; a column named like an output is a
      label (a ``<output>__probs`` column of ``{answer: p}`` is a soft label).
    - ``teacher``: a model name or AI function that labels rows without labels
      (all rows with ``labels="teacher"``). Its probabilities are used when it
      gives them (Jev does), else its answers.
    - ``labels``: ``"auto"`` (the data's labels, the teacher's for the rest),
      ``"data"`` (only rows with labels), ``"teacher"`` (the teacher's for every
      training row; the data's labels are then used only to test).
    - ``test``: rows with labels to measure on; else ``holdout`` of the labeled
      rows is set aside (at least 50, at most 2,000).
    - ``validation``: share of the training rows kept to stop training at the
      best pass and to fit the temperature (at least 32, at most 1,000).
    - ``student``: a Hugging Face model id or local path. Ettin-17M trained in
      46 s to 91.5% on banking77 with human labels (2026-09-25).
    - ``epochs``, ``lr``, ``batch_size``, ``max_length``, ``seed``: training
      settings (defaults as measured best per model size).
    - ``device``: default the GPU with the most free memory if it has room,
      else the CPU; memory held by other programs is never taken.
    - ``path``: where the model is written (default under ~/.cache/functai/baked).
    - ``compare_teacher``: also run the teacher on the test rows, to report its
      accuracy next to the student's (the "teacher-limited" check).
    - ``prices``: ``{"teacher": (dollars per M input tokens, per M output tokens),
      "gpu_per_hour": dollars}`` to report money and break-even.
    """
    say = _say(log)
    if labels not in ("auto", "data", "teacher"):
        raise BakeError(f"labels is 'auto', 'data' or 'teacher', not {labels!r}")
    if fn._tools:
        raise BakeError(f"{fn.__name__} uses tools; a head model answers from the input alone. Bake a copy "
                        f"without tools, or a generative student (method='sft')")
    from . import heads
    from .examples import head_fields, head_layout, head_signature, input_plan, render_input, row_inputs
    from ..evaluation import rows_of

    spec = fn._variant_spec(reasoning=False, tools=False)
    fields = head_fields(spec)
    signature = head_signature(spec, fields)
    layout = head_layout(signature)
    plan = input_plan(signature, layout)
    rows = rows_of(data)
    test_rows = rows_of(test) if test is not None else None
    notes: List[str] = []

    # ---- labels
    gold = [_gold(r, fields) for r in rows]
    labeled = [i for i, g in enumerate(gold) if g is not None]
    if test_rows is None:
        pool, held = _split(labeled, holdout, 50, 2000, seed)
        test_idx = set(held)
        test_rows = [rows[i] for i in held]
        train_idx = [i for i in range(len(rows)) if i not in test_idx]
    else:
        train_idx = list(range(len(rows)))
    test_gold = [_gold(r, fields) for r in test_rows]
    if test_rows and any(g is None for g in test_gold):
        raise BakeError("every test row needs its label(s) for " + ", ".join(f.name for f in fields))
    truth = "labeled" if test_rows else "teacher"

    need_teacher = labels == "teacher" or (labels == "auto" and any(gold[i] is None for i in train_idx))
    if need_teacher and teacher is None:
        missing = sum(gold[i] is None for i in train_idx)
        raise BakeError(f"{missing:,} training rows have no label for {[f.name for f in fields]} and no teacher "
                        f"was given: pass teacher='jev-latest' (or another model), or labels='data'")
    labeling: Dict[str, Any] = {}
    targets: Dict[int, List[List[float]]] = {}
    if labels == "data":
        dropped = [i for i in train_idx if gold[i] is None]
        if dropped:
            notes.append(f"{len(dropped):,} rows without labels were left out (labels='data')")
        train_idx = [i for i in train_idx if gold[i] is not None]
        targets = {i: gold[i] for i in train_idx}
        source = "the data's labels"
    else:
        to_label = [i for i in train_idx if labels == "teacher" or gold[i] is None]
        targets = {i: gold[i] for i in train_idx if i not in set(to_label)}
        if to_label:
            say(f"labeling {len(to_label):,} rows with {getattr(teacher, '__name__', teacher)}")
            got, labeling = _teacher_labels(fn, teacher, [rows[i] for i in to_label], fields,
                                            num_threads=num_threads, prices=prices, say=say)
            for i, t in zip(to_label, got):
                if t is not None:
                    targets[i] = t
            train_idx = [i for i in train_idx if i in targets]
        kind = "teacher (soft)" if labeling.get("soft") else "teacher (hard)"
        n_teacher = sum(1 for i in to_label if i in targets)
        source = "the data's labels" if not n_teacher else kind if n_teacher == len(train_idx) \
            else f"the data's labels and {kind}"
    if not test_rows:                 # no human labels anywhere: hold out teacher-labeled rows instead
        rest, held = _split(train_idx, holdout, 50, 2000, seed)
        train_idx = rest
        test_rows = [rows[i] for i in held]
        test_gold = [targets[i] for i in held]
        notes.append("no human-labeled rows to test on: the numbers below are agreement with the teacher, "
                     "not accuracy; the student can only be as good as its teacher")
    if len(train_idx) < 10:
        raise BakeError(f"only {len(train_idx)} labeled training rows; bake needs at least 10 (hundreds to learn well)")

    # ---- validation split, texts, tokens
    train_idx, val_idx = _split(train_idx, validation, 32, 1000, seed + 1) if len(train_idx) >= 100 \
        else (train_idx, [])
    if not val_idx:
        notes.append("fewer than 100 training rows: no validation split, so no early stopping and no "
                     "temperature fitting")

    def texts_of(idx_or_rows) -> List[str]:
        return [render_input(plan, spec, row_inputs(fn, r)) for r in idx_or_rows]

    say("writing the inputs")
    train_texts = texts_of([rows[i] for i in train_idx])
    val_texts = texts_of([rows[i] for i in val_idx])
    test_texts = texts_of(test_rows)

    def field_targets(ts: List[List[List[float]]]) -> List[List[List[float]]]:
        return [[t[f] for t in ts] for f in range(len(fields))]

    train_t = field_targets([targets[i] for i in train_idx])
    val_t = field_targets([targets[i] for i in val_idx])
    say(f"loading {student}")
    model, tokenizer, architecture = heads.build(student, fields, local_files_only=local_files_only)
    n_params = heads.count_parameters(model)
    config = model.model.config if hasattr(model, "model") else model.backbone.config
    decoder = heads.is_decoder(config)
    limit = min(x for x in (getattr(tokenizer, "model_max_length", 512) or 512,
                            getattr(config, "max_position_embeddings", 512) or 512, 512) if x)
    raw_lengths = [len(x) for x in tokenizer(train_texts)["input_ids"]] if train_texts else []
    max_len = max_length or heads.choose_max_length(raw_lengths, limit)
    train_ids, cut_train = heads.encode(tokenizer, train_texts, max_len)
    val_ids, _ = heads.encode(tokenizer, val_texts, max_len)
    test_ids, cut_test = heads.encode(tokenizer, test_texts, max_len)
    if cut_train / max(1, len(train_ids)) > 0.01:
        notes.append(f"{cut_train:,} training inputs were longer than {max_len} tokens and were cut")

    # ---- train
    dev = heads.choose_device(device, heads.training_memory_gb(n_params))
    if dev == "cpu" and device is None:
        say("no GPU with enough free memory: training on the CPU")
    cfg = heads.default_config(n_params, len(train_idx), decoder, epochs=epochs, lr=lr, batch_size=batch_size,
                               seed=seed)
    say(f"training {n_params / 1e6:.1f}M parameters on {dev}: {len(train_idx):,} rows, up to {cfg.epochs} passes, "
        f"lr {cfg.lr:g}")
    result = heads.train(model, tokenizer, train_ids, train_t, val_ids, val_t, cfg, dev, log=say)

    # ---- calibrate: against human labels on the validation rows when they have them
    val_gold = [gold[i] if i < len(gold) else None for i in val_idx]
    calib_t = field_targets(val_gold) if val_idx and all(g is not None for g in val_gold) else val_t
    val_logits = heads.logits(model, tokenizer, val_ids, dev) if val_idx else [[] for _ in fields]
    temps = heads.fit_temperatures(val_logits, calib_t) if val_idx else [1.0] * len(fields)

    # ---- test
    from . import metrics
    test_logits = heads.logits(model, tokenizer, test_ids, dev)
    test_labels = [[metrics.argmax(g[f]) for g in test_gold] for f in range(len(fields))]
    field_scores: List[FieldScores] = []
    probs_all = []
    for f, fld in enumerate(fields):
        raw = [metrics.softmax(z) for z in test_logits[f]]
        cal = [metrics.softmax(z, temps[f]) for z in test_logits[f]]
        probs_all.append(cal)
        y = test_labels[f]
        right = sum(metrics.argmax(p) == t for p, t in zip(cal, y))
        wrong_by = {}
        for p, t in zip(cal, y):
            if metrics.argmax(p) != t:
                wrong_by[fld.keys[t]] = wrong_by.get(fld.keys[t], 0) + 1
        counts = {}
        for t in y:
            counts[fld.keys[t]] = counts.get(fld.keys[t], 0) + 1
        worst = sorted(({"answer": k, "accuracy": 1 - wrong_by.get(k, 0) / n, "rows": n}
                        for k, n in counts.items() if n >= 5), key=lambda d: d["accuracy"])[:5]
        field_scores.append(FieldScores(
            name=fld.name, accuracy=metrics.accuracy(cal, y), interval=list(metrics.wilson(right, len(y))),
            top3=metrics.top_k(cal, y, 3) if len(fld.keys) >= 5 else None, ece=metrics.ece(cal, y),
            ece_raw=metrics.ece(raw, y), nll=metrics.nll(cal, y), temperature=temps[f], worst=worst))
        unseen = [fld.keys[k] for k in range(len(fld.keys)) if not any(row[k] > 0 for row in train_t[f])]
        if unseen:
            notes.append(f"{fld.name}: {len(unseen)} answers never appear in the training labels "
                         f"({', '.join(unseen[:5])}{'...' if len(unseen) > 5 else ''}); the model cannot learn them")
    conf = [min(max(probs_all[f][r]) for f in range(len(fields))) for r in range(len(test_rows))]
    correct = [all(metrics.argmax(probs_all[f][r]) == test_labels[f][r] for f in range(len(fields)))
               for r in range(len(test_rows))]

    # ---- the teacher on the test rows (the ceiling of teacher labels)
    if teacher is not None and compare_teacher and truth == "labeled":
        say(f"running the teacher on the {len(test_rows):,} test rows")
        t_targets, t_info = _teacher_labels(fn, teacher, test_rows, fields, num_threads=num_threads, prices=prices,
                                            say=say)
        for f, fs in enumerate(field_scores):
            pairs = [(metrics.argmax(t[f]), y, metrics.argmax(probs_all[f][r]))
                     for r, (t, y) in enumerate(zip(t_targets, test_labels[f])) if t is not None]
            if pairs:
                fs.teacher_accuracy = sum(a == y for a, y, _s in pairs) / len(pairs)
                fs.agreement = sum(a == s for a, _y, s in pairs) / len(pairs)
        if not labeling:
            labeling = {**t_info, "rows": 0, "test_only": True}
        labeling.setdefault("seconds_per_row", t_info.get("seconds_per_row"))
        labeling.setdefault("dollars_per_row", t_info.get("dollars_per_row"))

    # ---- speed on the device it trained on (tokenizing included)
    import statistics as st
    sample = (test_texts + val_texts)[:2048] or train_texts[:2048]
    t0 = time.time()
    ids, _ = heads.encode(tokenizer, sample, max_len)
    heads.logits(model, tokenizer, ids, dev)
    batch_rps = len(sample) / max(1e-9, time.time() - t0)
    lat = []
    for text in sample[:50]:
        t1 = time.perf_counter()
        one_ids, _ = heads.encode(tokenizer, [text], max_len)
        heads.logits(model, tokenizer, one_ids, dev, batch_size=1)
        lat.append((time.perf_counter() - t1) * 1000)
    speed = {"device": dev, "rows_per_second": batch_rps, "latency_ms": st.median(lat) if lat else float("nan")}

    # ---- break-even against the teacher
    breakeven: Dict[str, Any] = {}
    t_row = labeling.get("seconds_per_row")
    s_row = 1 / batch_rps if batch_rps else None
    prep = result["seconds"] + (labeling.get("seconds", 0) if not labeling.get("test_only") else 0)
    if t_row and s_row and t_row > s_row:
        breakeven["rows_time"] = prep / (t_row - s_row)
    d_row = labeling.get("dollars_per_row")
    gpu = (prices or {}).get("gpu_per_hour")
    if d_row and gpu is not None:
        student_d = gpu / 3600 * s_row
        fixed = labeling.get("dollars", 0) if not labeling.get("test_only") else 0
        fixed += gpu / 3600 * result["seconds"]
        if d_row > student_d:
            breakeven["rows_money"] = fixed / (d_row - student_d)

    # ---- the ceiling of teacher labels
    for fs in field_scores:
        if source.startswith("teacher") and fs.teacher_accuracy is not None and \
                fs.accuracy <= fs.teacher_accuracy + 0.01:
            where = "about as accurate as" if fs.accuracy >= fs.interval[0] and fs.teacher_accuracy <= fs.interval[1] \
                else "below"
            notes.append(f"{fs.name}: the student ({fs.accuracy:.1%}) is {where} its teacher "
                         f"({fs.teacher_accuracy:.1%}); trained on teacher labels, it can at best match it. Human "
                         f"labels lifted the same kind of student from 77% to 91.5% on banking77: label more rows by "
                         f"hand, or use a stronger teacher")
    if truth == "labeled" and len(test_rows) < 200:
        lo, hi = field_scores[0].interval
        notes.append(f"only {len(test_rows)} test rows: accuracy is known to ±{(hi - lo) / 2:.0%}")

    report = BakeReport(
        function=fn.__name__, student=student, parameters=n_params, device=dev, truth=truth, label_source=source,
        teacher=None if teacher is None else str(getattr(teacher, "__name__", getattr(teacher, "name", teacher))),
        rows={"train": len(train_idx), "validation": len(val_idx), "test": len(test_rows)},
        training={**{k: v for k, v in result.items() if k != "history"}, "history": result["history"],
                  "passes_run": len(result["history"]), "lr": cfg.lr, "head_lr": cfg.head_lr,
                  "batch_size": cfg.batch_size, "seed": seed},
        fields=field_scores, coverage=metrics.coverage(conf, correct), confidence=conf, correct=correct,
        speed=speed, labeling=labeling, breakeven=breakeven, notes=notes, max_length=max_len,
        truncated=cut_train + cut_test)

    # ---- write it
    from .baked import default_home, write_meta
    from .examples import fingerprint as data_fingerprint
    from lmcc import signature_fingerprint, signature_to_dict
    name = f"{fn.__name__}-{Path(student).name}-{time.strftime('%Y%m%d-%H%M%S')}"
    out = Path(path).expanduser().resolve() if path else default_home() / name
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} exists and is not empty")
    out.mkdir(parents=True, exist_ok=True)
    model.to("cpu")
    model.save(str(out))
    tokenizer.save_pretrained(str(out / "tokenizer"))
    from .examples import CAPABILITIES as HEAD_CAPABILITIES
    write_meta(out, {
        "name": fn.__name__, "kind": "head", "student": student,
        "functions": [{"name": fn.__name__, "fingerprint": signature_fingerprint(signature),
                       "signature": signature_to_dict(signature), "layout": layout.dump(),
                       "outputs": [f.name for f in fields], "reasoning": False, "fixed": {}, "derived": {},
                       "capabilities": dict(HEAD_CAPABILITIES)}],
        "head": {"architecture": architecture, "decoder": decoder, "fields": [f.to_dict() for f in fields],
                 "max_length": max_len, "temperatures": temps},
        "weights": {"form": "merged", "base": student, "path": "model"},
        "parameters": n_params, "data": data_fingerprint(train_texts, train_t), "report": report.to_dict(),
    })
    say(f"saved to {out}")
    baked = Baked(out, device=dev)
    baked._model, baked._tokenizer = model.to(dev).eval(), tokenizer
    baked._report = report
    return baked


# ------------------------------------------------------------------ the one entry point


_SFT_ONLY = ("fixed", "derived", "layout", "reasoning", "tags", "weights", "functions", "where")


def _all_finite(fn) -> bool:
    from .examples import head_fields
    if fn._tools:
        return False
    try:
        head_fields(fn._variant_spec(reasoning=False, tools=False))
        return True
    except BakeError:
        return False


def bake(what: Any, data: Any = None, *, method: str = "auto", student: Optional[str] = None, teacher: Any = None,
         labels: str = "auto", where: Any = "auto", test: Any = None, metric: Any = None, report: bool = True,
         compare_teacher: bool = False, wait: bool = True, plan_only: bool = False,
         path: "str | Path | None" = None, run_folder: "str | Path | None" = None, fixed: Optional[Dict[str, Any]] = None,
         derived: Optional[Dict[str, str]] = None, layout: Any = None, reasoning: bool = False,
         tags: Optional[str] = None, weights: Optional[str] = None, functions: Optional[Sequence[Any]] = None,
         name: Optional[str] = None, validation: Optional[float] = None, holdout: Optional[float] = None,
         num_threads: int = 16, local_files_only: bool = False, log: Any = True, seed: int = 0, **training):
    """Train weights that answer an AI function (or several); returns the baked model.

    ``what`` and ``data``:

    - ``fn, rows``: one function; rows are dicts (or any table ``dpyr.read``
      takes) with the inputs under the parameters' names and, when known, the
      answers under the outputs' names (``result`` for the return value).
    - ``{fn: rows, fn2: rows2}``: one student for several functions, each
      called through its own layout.
    - ``program, rows``: a ``@module``; it runs on each row (with ``teacher``
      answering every AI call inside it), and every call becomes an example
      of its function (``functions=`` keeps some).
    - ``examples``: made by ``functai.bake.examples`` (or a file of them).

    The main choices (all decided from the data when left out; ``plan_only=True``
    prints the plan and spends nothing):

    - ``method``: ``"head"``, ``"sft"``, or ``"auto"`` (a head when every
      output is finite and nothing asks for a generative student).
    - ``student``: a Hugging Face model id or folder.
    - ``teacher``: answers the rows without answers: a model name, an AI
      function, or ``{fn: teacher}``; default each function's own model.
      ``labels="teacher"`` asks it for every row; ``"data"`` uses only rows
      with answers.
    - ``where``: ``"here"``, ``"tinker"``, ``"prime"``, ``"export"``, a list in
      order of preference, or a ``Trainer``; ``"auto"``: here when this machine
      can train it, else the cheapest service set up
      (``configure(bake_where=...)`` sets a preference once).
    - ``fixed={"input": value}``: an input with one value in every row, left
      out of the student's prompt; a call with another value is refused.
      ``derived={"input": "other input"}``: an input decided by another, left
      out too.
    - ``test``: rows to judge on (else a share of the rows with answers is set
      aside); ``metric``: how to score them (anything ``evaluate`` takes; an
      AI judge for open text); ``report=False`` skips judging.
    - ``wait=False``: return the ``Run`` at once (a folder and a process that
      outlive this one); running the same bake again resumes it.
    - training settings: ``lora`` (True/False), ``lora_rank``, ``lr``,
      ``epochs``, ``batch`` (examples per step), ``quantize="4bit"``,
      ``packing``, ``devices``, ``liger``, ``max_new_tokens``, ``merge``
      (merge the adapter into the weights; default True), ``report_to``
      (["wandb"], ...).
    - for a head: ``epochs``, ``lr``, ``batch_size``, ``max_length``,
      ``device``, ``prices``, ``holdout``, ``validation`` as before.
    """
    from ..core import FunctAIFunc
    single = isinstance(what, FunctAIFunc)
    sft_asked = any(v not in (None, False, "auto") for v in (fixed, derived, layout, tags, weights, functions)) or \
        reasoning or (where not in ("auto", None, "here")) or plan_only or not wait
    if method == "auto":
        method = "head" if single and _all_finite(what) and not sft_asked and not training.get("lora") else "sft"
    if method == "head":
        if not single:
            raise BakeError("a head model answers one function: bake(fn, rows, method='head')")
        extra = {k: v for k, v in {"fixed": fixed, "derived": derived, "layout": layout, "tags": tags,
                                   "weights": weights, "functions": functions}.items() if v}
        if extra or where not in ("auto", None, "here") or plan_only or not wait:
            raise BakeError(f"a head model trains here, in seconds; {sorted(extra) or ['where/plan_only/wait']} "
                            f"apply to generative students (method='sft')")
        head_opts = {k: training.pop(k) for k in ("epochs", "lr", "batch_size", "max_length", "device", "prices")
                     if k in training}
        if training:
            raise TypeError(f"bake(method='head') does not take {sorted(training)}")
        return bake_head(what, data, student=student or "jhu-clsp/ettin-encoder-17m", teacher=teacher, labels=labels,
                         test=test, holdout=0.1 if holdout is None else holdout,
                         validation=0.1 if validation is None else validation, seed=seed, num_threads=num_threads,
                         path=path, compare_teacher=compare_teacher, local_files_only=local_files_only, log=log,
                         **head_opts)
    if method != "sft":
        raise BakeError(f"method is 'auto', 'head' or 'sft', not {method!r}")
    from .planning import TRAINING_OPTIONS, Plan
    unknown = set(training) - set(TRAINING_OPTIONS) - {"merge", "inline"}
    if unknown:
        raise TypeError(f"bake() does not take {sorted(unknown)}; training settings: "
                        f"{sorted(set(TRAINING_OPTIONS) | {'merge'})}")
    opts = dict(student=student, teacher=teacher, labels=labels, where=where, test=test, metric=metric, report=report,
                compare_teacher=compare_teacher, wait=wait, plan_only=plan_only, path=path, run_folder=run_folder,
                fixed=fixed, derived=derived, layout=layout, reasoning=reasoning, tags=tags, weights=weights,
                functions=functions, name=name, num_threads=num_threads, local_files_only=local_files_only, log=log,
                seed=seed, validation=0.02 if validation is None else validation,
                holdout=0.05 if holdout is None else holdout, **training)
    p = Plan(what, data, opts).decide()
    if plan_only:
        return p
    _say(log)(str(p))
    return p.run()


def examples(what: Any, data: Any = None, *, student: Optional[str] = None, teacher: Any = None,
             labels: str = "auto", fixed: Optional[Dict[str, Any]] = None, derived: Optional[Dict[str, str]] = None,
             layout: Any = None, reasoning: bool = False, tags: Optional[str] = None, weights: Optional[str] = None,
             functions: Optional[Sequence[Any]] = None, validation: float = 0.02, seed: int = 0,
             num_threads: int = 16, local_files_only: bool = False, log: Any = True):
    """The training conversations for an AI function (or several, or a
    program), as functai will call the student: a table every trainer reads
    (see ``functai.bake.dataset``). With ``student=``, each row also carries the
    exact tokens under that student's chat template (``input_ids``) and where
    the reply starts. Rows without answers are answered by ``teacher`` (default
    each function's own model; the cost is said before it is spent)."""
    from .dataset import make
    from .planning import Plan, _money
    from .sources import label, teacher_estimate
    p = Plan(what, data, dict(teacher=teacher, labels=labels, fixed=fixed, derived=derived, layout=layout,
                              reasoning=reasoning, tags=tags, weights=weights, functions=functions, report=False,
                              num_threads=num_threads, local_files_only=local_files_only, log=log, seed=seed))
    say = _say(log)
    if p.examples is not None:
        return p.examples
    info: Dict[str, Any] = {"program": p.program_info}
    if p.to_label:
        c = teacher_estimate(p.to_label, teacher)
        say(f"asking {c['teacher']} for {len(p.to_label):,} answers (prompts ≈ {_money(c['dollars'])}, plus the "
            f"answers)")
        info["labeling"] = label(p.items, teacher, num_threads=num_threads, log=say)
    tok = tpl = None
    if student:
        from .template import describe, load_tokenizer
        tok = load_tokenizer(student, local_files_only=local_files_only)
        tpl = describe(tok)
    return make(p.items, tokenizer=tok, template=tpl, student=student, validation=validation, seed=seed, info=info,
                log=say)


def judge(baked, fn: Any, rows: Any = None, **options):
    """Measure a generative student on test rows: ``functai.evaluate`` on the
    function running on the baked model, plus what only a student has.

    ``bake`` judges the student when it ends (``report=True``); ``judge``
    does it again later, on other rows or with another metric. A head
    model's report is made when it is baked (``baked.report``).

    Parameters
    ----------
    baked : Baked
        The student.
    fn : AI function, or dict
        The function and ``rows``, or ``{function: rows}`` for a student of
        several functions.
    rows : list of dict, or a table
        Test rows with the right answers (never rows it trained on).
    metric : optional
        Anything ``evaluate`` takes: a function, a dpyr expression, an AI
        judge, a dict per output. Without one, a function whose outputs are
        all finite (``Literal``, ``Enum``, ``bool``) is scored by exact match;
        any other is not scored (exact match on open text measures nothing):
        the report then gives readability and samples to read.
    compare_teacher : bool
        Also run the teacher on the same rows, to compare (it costs a
        teacher pass). Default False.
    by : str, optional
        A column of the rows (a tag, a source) to score each of its values
        apart.
    samples : int
        How many answers the report shows to read (default 5).
    save : bool
        Write the report into the model's ``baked.json``.
    num_threads : int
        Rows judged at once (default 16).

    Returns
    -------
    report
        Printed: per function, the score with its 95% range (per group with
        ``by``), readability (the share of replies its layout reads back into
        the output types), speed, the cost of a call next to the teacher's,
        and samples.

    Examples
    --------
    ```python
    # not run: it needs a baked model
    report = functai.bake.judge(baked, summarize, test_rows, metric=faithful)
    print(report)
    ```
    """
    from .judging import judge as _judge
    return _judge(baked, fn, rows, **options)


def adopt(folder: "str | Path", fn: Any, *, examples: Any = None, student: Optional[str] = None,
          path: "str | Path | None" = None, layout: Any = None, fixed: Optional[Dict[str, Any]] = None,
          derived: Optional[Dict[str, str]] = None, reasoning: bool = False, name: Optional[str] = None,
          max_new_tokens: Optional[int] = None) -> Baked:
    """A model trained elsewhere (TRL, Axolotl, Unsloth, by hand), as a baked
    model functai calls through the function's layout.

    ``folder``: a Hugging Face model folder, or a LoRA adapter folder (its base
    named in ``adapter_config.json``, or ``student=``). ``fn``: the AI function
    (or a list of them) it answers. ``examples``: the examples it was trained
    on (``functai.bake.examples(..., student=...)``, or their file): their
    layouts are used, and their tokens are checked against the folder's chat
    template, so a model trained on other tokens is refused. Without examples,
    the function's own layout is used and nothing can be checked."""
    import json
    import os
    import shutil
    from .baked import default_home, write_meta
    from .dataset import Examples
    from .functions import Entry
    from .template import describe, example_ids, load_tokenizer
    src = Path(folder).expanduser().resolve()
    if not src.is_dir():
        raise BakeError(f"{src} is not a folder")
    fns = list(fn) if isinstance(fn, (list, tuple)) else [fn]
    adapter = (src / "adapter_config.json").exists()
    base = student
    if adapter and base is None:
        base = json.loads((src / "adapter_config.json").read_text()).get("base_model_name_or_path")
    if adapter and not base:
        raise BakeError("the adapter does not name its base model: pass student=")
    ex = Examples.load(examples) if isinstance(examples, (str, Path)) else examples
    entries: Dict[str, Entry] = {}
    for f in fns:
        if ex is not None and f.__name__ in (ex.entries or {}):
            e = ex.entries[f.__name__]
            entry = e if isinstance(e, Entry) else Entry.from_meta(e)
            entry.check(f._variant_spec(reasoning=entry.reasoning, tools=False))
        else:
            names = [n for n, _r in f._named_inputs()]
            entry = Entry.build(f, layout=layout, reasoning=reasoning,
                                fixed={k: v for k, v in (fixed or {}).items() if k in names},
                                derived={k: v for k, v in (derived or {}).items() if k in names})
        entries[f.__name__] = entry
    tok_src = src if (src / "tokenizer_config.json").exists() else (base or src)
    tok = load_tokenizer(str(tok_src))
    tpl = describe(tok)
    notes = []
    if ex is not None and ex.tokenized:
        sample = [r for r in ex.rows if r["function"] in entries][:50]
        bad = 0
        for r in sample:
            msgs = r["messages"]
            ids, start = example_ids(tok, msgs[:-1], msgs[-1]["content"], tpl.kwargs, tpl)
            if list(ids) != list(r["input_ids"]) or start != r["answer_start"]:
                bad += 1
        if bad:
            raise BakeError(f"{bad} of {len(sample)} examples tokenize differently with this folder's chat template "
                            f"than they were trained on: the model would be called with other tokens than it "
                            f"learned. Adopt it with the tokenizer it was trained with")
    else:
        notes.append("adopted without its training examples: the tokens it learned could not be checked")
    name = name or "+".join(sorted(entries))
    out = Path(path).expanduser().resolve() if path else default_home() / f"{name}-{src.name}-adopted"
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} exists and is not empty")
    out.mkdir(parents=True, exist_ok=True)

    def link(a, b):
        try:
            os.link(a, b)
        except OSError:
            shutil.copy2(a, b)
    if adapter:
        shutil.copytree(src, out / "adapter", copy_function=link)
        weights = {"form": "lora", "base": base, "adapter": "adapter", "from": str(src)}
    else:
        shutil.copytree(src, out / "model", copy_function=link)
        weights = {"form": "merged", "base": base or str(src), "path": "model", "from": str(src)}
    tok.save_pretrained(str(out / "tokenizer"))
    s = ex.stats() if ex is not None and ex.tokenized else {}
    new = max_new_tokens or (int(s["answer"]["max"] * 1.25) if s else 2048)
    from .students import info as student_info
    try:
        ctx = student_info(base or str(src)).context
    except Exception:  # noqa: BLE001
        ctx = None
    model_len = int(min(ctx or 10 ** 9, (s.get("longest", 0) + new) if s else (ctx or 32768)))
    write_meta(out, {"kind": "generative", "name": name, "student": base or str(src),
                     "functions": [e.to_meta() for e in entries.values()], "weights": weights,
                     "template": tpl.to_dict(), "generation": {"max_new_tokens": new, "max_model_len": model_len},
                     "run": {"where": "adopted", "from": str(src)}, "notes": notes})
    return Baked(out)
