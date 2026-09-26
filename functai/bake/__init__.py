"""Bake an AI function into weights you own.

    baked = classify.bake(rows, student="jhu-clsp/ettin-encoder-17m")   # human labels in the rows
    baked = classify.bake(rows, teacher="jev-latest")                    # a teacher labels them
    print(baked.report)
    fast = classify.using(lm=baked)                                      # the same function, on the weights
    safe = classify.using(lm=baked, escalate_to="claude-opus-5.5",       # unsure rows go to a big model
                          escalate_below=baked.report.threshold(0.95)["threshold"])

``method="head"`` (the default) trains a classifier for functions whose outputs
have a fixed set of answers (Literal, Enum, bool): the input goes in, a
probability for every answer comes out; no prompt, no generation.
``method="sft"`` trains a small chat model to write the function's answers in a
fixed layout, for open outputs (see ``functai.bake.sft``).

Needs PyTorch and transformers: ``pip install "functai[bake]"``.
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

__all__ = ["bake", "Baked", "load", "BakeError", "BakeReport", "is_baked"]


def _say(log: Optional[Callable[[str], None]]) -> Callable[[str], None]:
    if log is None:
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
            pred = tfn(**row_inputs(fn, row), all=True)
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


def bake(fn, data: Any, *, student: str = "jhu-clsp/ettin-encoder-17m", method: str = "head",
         teacher: Any = None, labels: str = "auto", test: Any = None, holdout: float = 0.1,
         validation: float = 0.1, epochs: Optional[int] = None, lr: Optional[float] = None,
         batch_size: int = 32, max_length: Optional[int] = None, device: Optional[str] = None, seed: int = 0,
         num_threads: int = 16, path: "str | Path | None" = None, compare_teacher: bool = True,
         prices: Optional[Dict[str, Any]] = None, local_files_only: bool = False,
         log: Any = True, **method_options) -> Baked:
    """Train weights that answer ``fn``; returns the baked model (see ``Baked``).

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
    if method == "sft":
        from . import sft
        return sft.bake(fn, data, student=student, teacher=teacher, labels=labels, test=test, holdout=holdout,
                        validation=validation, epochs=epochs, lr=lr, batch_size=batch_size, device=device,
                        seed=seed, num_threads=num_threads, path=path, local_files_only=local_files_only,
                        log=say, **method_options)
    if method != "head":
        raise BakeError(f"method is 'head' or 'sft', not {method!r}")
    if method_options:
        raise TypeError(f"bake(method='head') does not take {sorted(method_options)}")
    if labels not in ("auto", "data", "teacher"):
        raise BakeError(f"labels is 'auto', 'data' or 'teacher', not {labels!r}")
    if fn._tools:
        raise BakeError(f"{fn.__name__} uses tools; a head model answers from the input alone. Bake a copy "
                        f"without tools, or a generative student (method='sft')")
    from . import heads
    from .examples import head_fields, head_layout, head_signature, input_plan, render_input, row_inputs
    from ..evaluation import rows_of
    from ..signature import build_spec

    s = fn._effective()
    spec = build_spec(fn._fn, instructions=fn._current_state().instructions,
                      include_fn_name=bool(s.get("include_fn_name_in_instructions")), reasoning=False, tools=False)
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
    write_meta(out, {
        "name": fn.__name__, "kind": "head", "student": student, "architecture": architecture, "decoder": decoder,
        "fields": [f.to_dict() for f in fields], "layout": layout.dump(), "signature": signature_to_dict(signature),
        "fingerprint": signature_fingerprint(signature), "max_length": max_len, "temperatures": temps,
        "parameters": n_params, "data": data_fingerprint(train_texts, train_t), "report": report.to_dict(),
    })
    say(f"saved to {out}")
    baked = Baked(out, device=dev)
    baked._model, baked._tokenizer = model.to(dev).eval(), tokenizer
    baked._report = report
    return baked
