"""A verifiers taskset for any @ai function and a table of rows: ``functai-rows``.

    vf-eval functai-rows \\
        --env.taskset.program mytask:classify --env.taskset.data rows.jsonl \\
        --env.agent.harness.id functai-verifiers --env.agent.harness.program mytask:classify

One task per row. The row's columns named like the function's parameters are
the inputs; its columns named like the function's outputs are the expected
answers, and the reward ``agrees`` is the share of them the reply got right
(compared as functai's exact_match compares: text ignoring case and spaces).
A thinking teacher's reasoning is dropped before the trace is recorded, so
rollouts train a student to answer (``--env.taskset.task.teacher-reasoning keep``
to keep it). The data is any file functai.evaluation reads (JSON lines, JSON,
CSV, parquet) or a Hugging Face ``dataset:name:split``.
"""

from __future__ import annotations

import importlib
import json
import random
from collections.abc import Iterator
from pathlib import Path

import verifiers.v1 as vf

from functai.core import FunctAIFunc
from functai_verifiers import AnswerOnlyTask, AnswerOnlyTaskConfig, OneTurnEnv, RowTaskData, turns

__all__ = ["RowsEnv", "RowsTaskset"]


class RowsTaskConfig(AnswerOnlyTaskConfig):
    program: str = ""
    """Set by the taskset: the function the rows are for."""


class RowsConfig(vf.TasksetConfig):
    program: str = ""
    """The @ai function: ``pkg.module:function``."""
    data: str = ""
    """The rows: a file (``.jsonl``, ``.json``, ``.csv``, ``.parquet``) or ``dataset:<name>:<split>``."""
    limit: int | None = None
    shuffle_seed: int | None = 0
    """Rows are shuffled with this seed so any first N are a fair sample; None keeps the order."""
    task: RowsTaskConfig = RowsTaskConfig()


def _program(path: str) -> FunctAIFunc:
    module, _, name = path.rpartition(":")
    fn = getattr(importlib.import_module(module), name)
    if not isinstance(fn, FunctAIFunc):
        raise TypeError(f"{path!r} is not an @ai function")
    return fn


def _rows(data: str) -> list[dict]:
    if data.startswith("dataset:"):
        from datasets import load_dataset
        _, name, split = data.split(":", 2)
        return [dict(r) for r in load_dataset(name, split=split)]
    p = Path(data)
    if p.suffix == ".jsonl":
        return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    if p.suffix == ".json":
        return list(json.loads(p.read_text()))
    from functai.evaluation import rows_of
    return rows_of(str(p))


def _norm(v):
    from functai.evaluation import _norm as norm
    return norm(v)


class RowsTask(AnswerOnlyTask[RowsTaskConfig]):
    @vf.reward(weight=1.0)
    async def agrees(self, trace: vf.Trace) -> float:
        """Share of the row's expected outputs the reply got right; 0 when unreadable."""
        recorded = turns(trace)
        expected = self.data.info.get("expected", {})
        if not recorded or "refusal" in recorded[-1] or not expected:
            return 0.0
        got = recorded[-1]["outputs"]
        return sum(_norm(got.get(k)) == _norm(v) for k, v in expected.items()) / len(expected)


class RowsEnv(OneTurnEnv):
    pass


def row_tasks(config: RowsConfig) -> Iterator[RowsTask]:
    """The tasks of a rows taskset config (shared by generated environment packages)."""
    if not config.program or not config.data:
        raise ValueError("functai-rows needs --env.taskset.program pkg.module:function and --env.taskset.data")
    fn = _program(config.program)
    params = [n for n in fn._sig.parameters]
    outputs = [f.name for f in fn.signature.outputs if f.purpose == "plain"]
    rows = _rows(config.data)
    if config.shuffle_seed is not None:
        random.Random(config.shuffle_seed).shuffle(rows)
    for i, row in enumerate(rows):
        if config.limit is not None and i >= config.limit:
            return
        missing = [p for p in params if p not in row and fn._sig.parameters[p].default is fn._sig.parameters[p].empty]
        if missing:
            raise ValueError(f"row {i} lacks the input(s) {missing}")
        info = {"inputs": {p: row[p] for p in params if p in row},
                "expected": {o: row[o] for o in outputs if row.get(o) is not None}}
        yield RowsTask(RowTaskData(idx=i, prompt=None, info=info), config.task)


class RowsTaskset(vf.Taskset[RowsTask, RowsConfig]):
    def load(self) -> Iterator[RowsTask]:
        return row_tasks(self.config)
