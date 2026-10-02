"""The training conversations: what functai owns, in a form every trainer reads.

    ex = functai.bake.examples(summarize, rows, student="Qwen/Qwen3.5-4B")
    ex.save("train.parquet")      # or .jsonl / .jsonl.gz; ex.to_hf() for a datasets.Dataset
    ex.table                      # a dpyr table, to filter and count

One row per example:

    function        the AI function it trains
    messages        the chat as the layout writes it, the reply last
                    (TRL, Axolotl, Unsloth and prime-rl read this as is)
    prompt,         the same chat split at the reply (written on save; TRL's
    completion      prompt/completion form)
    input_ids       the exact tokens under the student's chat template (with
                    ``student=``): what training sees and what a call sends
    answer_start    where the reply starts in input_ids: the loss is on the
                    tokens from there to the end
    prompt_tokens,  token counts (with ``student=``)
    answer_tokens
    tag, weight     a label you gave the row, and its weight
    row_id, split   its source row, and "train" or "validation"
    source          "data", "teacher" or "program"
"""

from __future__ import annotations

import gzip
import json
import math
import statistics
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence

from .examples import BakeError

COLUMNS = ("function", "messages", "input_ids", "answer_start", "prompt_tokens", "answer_tokens", "tag", "weight",
           "row_id", "split", "source")


def _ids_column(pa, seqs):
    """Token ids as an Arrow list<int32> column, without a Python int per token."""
    import numpy as np
    lens = np.fromiter((len(s) for s in seqs), dtype=np.int64, count=len(seqs))
    offsets = np.zeros(len(seqs) + 1, dtype=np.int64)
    np.cumsum(lens, out=offsets[1:])
    values = np.concatenate([np.frombuffer(s, dtype=np.int32) if hasattr(s, "typecode") else
                             np.asarray(s, dtype=np.int32) for s in seqs]) if seqs else np.zeros(0, np.int32)
    if offsets[-1] < 2 ** 31:
        return pa.ListArray.from_arrays(pa.array(offsets.astype(np.int32)), pa.array(values))
    return pa.LargeListArray.from_arrays(pa.array(offsets), pa.array(values))


def _pct(xs: Sequence[int], q: float) -> int:
    if not xs:
        return 0
    s = sorted(xs)
    return s[min(len(s) - 1, int(math.ceil(q * len(s))) - 1)]


class Examples:
    """Training conversations (see the module docstring)."""

    def __init__(self, rows: List[Dict[str, Any]], *, student: Optional[str] = None,
                 template: Optional[Dict[str, Any]] = None, entries: Optional[Dict[str, Any]] = None,
                 info: Optional[Dict[str, Any]] = None):
        self.rows = rows
        self.student = student
        self.template = template            # Template.to_dict() of the student the tokens were made with
        self.entries = entries or {}        # function name → Entry (in memory) or its meta (read from a file)
        self.info = dict(info or {})        # how they were made: labeling, program runs

    # ---- reading

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return iter(self.rows)

    def __getitem__(self, key):
        if isinstance(key, str):
            return [r.get(key) for r in self.rows]
        if isinstance(key, slice):
            return self._with(self.rows[key])
        return self.rows[key]

    @property
    def tokenized(self) -> bool:
        return bool(self.rows) and all(r.get("input_ids") is not None for r in self.rows)

    @property
    def functions(self) -> List[str]:
        return sorted({r["function"] for r in self.rows})

    def split(self, name: str) -> "Examples":
        return self._with([r for r in self.rows if r.get("split", "train") == name])

    def filter(self, keep: Callable[[Dict[str, Any]], bool]) -> "Examples":
        """The examples ``keep(row)`` is true for: ``ex.filter(lambda r: r["prompt_tokens"] < 16_000)``."""
        return self._with([r for r in self.rows if keep(r)])

    def _with(self, rows: List[Dict[str, Any]]) -> "Examples":
        return Examples(rows, student=self.student, template=self.template, entries=self.entries, info=self.info)

    def stats(self) -> Dict[str, Any]:
        """Counts and token lengths (lengths need tokens: ``student=``)."""
        out: Dict[str, Any] = {"rows": len(self.rows), "functions": {}, "splits": {}}
        for r in self.rows:
            out["functions"][r["function"]] = out["functions"].get(r["function"], 0) + 1
            sp = r.get("split", "train")
            out["splits"][sp] = out["splits"].get(sp, 0) + 1
        if self.tokenized:
            p = [r["prompt_tokens"] for r in self.rows]
            a = [r["answer_tokens"] for r in self.rows]
            t = [x + y for x, y in zip(p, a)]
            w = [float(r.get("weight", 1.0)) for r in self.rows]
            out.update({
                "tokens": int(sum(x * wi for x, wi in zip(t, w))),
                "answer_tokens": int(sum(x * wi for x, wi in zip(a, w))),
                "prompt": {"median": int(statistics.median(p)), "p90": _pct(p, 0.9), "p99": _pct(p, 0.99),
                           "max": max(p)},
                "answer": {"median": int(statistics.median(a)), "p90": _pct(a, 0.9), "p99": _pct(a, 0.99),
                           "max": max(a)},
                "longest": max(t),
            })
        return out

    def __repr__(self) -> str:
        s = self.stats()
        fns = ", ".join(f"{k} {v:,}" for k, v in s["functions"].items())
        line = f"<Examples: {s['rows']:,} conversations ({fns})"
        if self.tokenized:
            line += (f"; {s['tokens']:,} tokens for {self.student}; prompts {s['prompt']['median']:,} median / "
                     f"{s['prompt']['p99']:,} p99, answers {s['answer']['median']:,} / {s['answer']['p99']:,}")
        return line + ">"

    # ---- tables and files

    def records(self, columns: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
        """Plain rows for files and tables: ``prompt`` and ``completion`` added,
        token ids as lists."""
        out = []
        for r in self.rows:
            d = {k: r.get(k) for k in COLUMNS if k in r}
            if d.get("input_ids") is not None:
                d["input_ids"] = [int(x) for x in d["input_ids"]]
            d["prompt"] = r["messages"][:-1]
            d["completion"] = r["messages"][-1:]
            out.append({k: v for k, v in d.items() if columns is None or k in columns})
        return out

    @property
    def table(self):
        """The examples as a dpyr table (``pip install "functai[data]"``)."""
        from ..evaluation import _dpyr
        return _dpyr().from_dict({k: [r.get(k) for r in self.records()] for k in self.records()[0]}) \
            if self.rows else _dpyr().from_dict({})

    def to_hf(self, columns: Optional[Sequence[str]] = None):
        """A Hugging Face ``datasets.Dataset``."""
        try:
            import datasets
        except ImportError as err:
            raise ImportError('pip install datasets (or "functai[bake]")') from err
        return datasets.Dataset.from_list(self.records(columns))

    def _meta(self) -> Dict[str, Any]:
        ents = {k: (v.to_meta() if hasattr(v, "to_meta") else v) for k, v in self.entries.items()}
        return {"functai_examples": 1, "student": self.student, "template": self.template, "functions": ents,
                "info": self.info}

    def save(self, path: "str | Path", *, columns: Optional[Sequence[str]] = None) -> Path:
        """Write the examples: ``.parquet``, ``.jsonl`` or ``.jsonl.gz``. What
        made them (the functions' layouts, the student's template) goes in the
        Parquet file's metadata, or a ``.meta.json`` file beside a JSONL one."""
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = json.dumps(self._meta(), ensure_ascii=False, default=str)
        name = path.name
        if name.endswith(".parquet"):
            try:
                import pyarrow as pa
                import pyarrow.parquet as pq
            except ImportError as err:
                raise ImportError("writing Parquet needs pyarrow: pip install pyarrow, or save as .jsonl") from err
            ids = None
            if self.tokenized and (columns is None or "input_ids" in columns):
                ids = _ids_column(pa, [r["input_ids"] for r in self.rows])
            recs = self.records([c for c in (columns or (*COLUMNS, "prompt", "completion")) if c != "input_ids"])
            table = pa.Table.from_pylist(recs)
            if ids is not None:
                table = table.append_column("input_ids", ids)
            table = table.replace_schema_metadata({**(table.schema.metadata or {}), b"functai": meta.encode()})
            pq.write_table(table, path, compression="zstd")
        elif name.endswith(".jsonl") or name.endswith(".jsonl.gz"):
            recs = self.records(columns)
            opener = gzip.open if name.endswith(".gz") else open
            with opener(path, "wt", encoding="utf-8") as f:
                for r in recs:
                    f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
            Path(str(path).removesuffix(".gz").removesuffix(".jsonl") + ".meta.json").write_text(meta)
        else:
            raise BakeError(f"examples are saved as .parquet, .jsonl or .jsonl.gz, not {path.suffix!r}")
        return path

    @classmethod
    def load(cls, path: "str | Path") -> "Examples":
        """Examples written by ``save`` (or by a bake run)."""
        path = Path(path).expanduser()
        name = path.name
        meta: Dict[str, Any] = {}
        if name.endswith(".parquet"):
            import pyarrow.parquet as pq
            table = pq.read_table(path)
            raw = (table.schema.metadata or {}).get(b"functai")
            meta = json.loads(raw) if raw else {}
            recs = table.to_pylist()
        else:
            opener = gzip.open if name.endswith(".gz") else open
            with opener(path, "rt", encoding="utf-8") as f:
                recs = [json.loads(line) for line in f if line.strip()]
            side = Path(str(path).removesuffix(".gz").removesuffix(".jsonl") + ".meta.json")
            if side.exists():
                meta = json.loads(side.read_text())
        return cls.from_records(recs, meta)

    @classmethod
    def from_records(cls, recs: Sequence[Dict[str, Any]], meta: Optional[Dict[str, Any]] = None) -> "Examples":
        meta = meta or {}
        rows = []
        for r in recs:
            if "messages" not in r or "function" not in r:
                raise BakeError("examples need the columns 'function' and 'messages' (from functai.bake.examples)")
            rows.append({k: r.get(k) for k in COLUMNS if k in r})
        return cls(rows, student=meta.get("student"), template=meta.get("template"),
                   entries=meta.get("functions") or {}, info=meta.get("info") or {})


def validation_count(n: int, share: float = 0.02) -> int:
    """How many of ``n`` training rows to keep for validation: ``share`` of
    them, at least 16 and at most 500; none below 100 rows (too few to spare)."""
    if n < 100 or share <= 0:
        return 0
    return min(500, max(16, round(share * n)))


def make(items: Sequence[Any], *, tokenizer: Any = None, template: Any = None, student: Optional[str] = None,
         validation: float = 0.02, seed: int = 0, info: Optional[Dict[str, Any]] = None,
         log: Callable[[str], None] = lambda s: None) -> Examples:
    """Examples from items with outputs: the messages each function's entry
    writes, and with a tokenizer, the student's exact tokens."""
    import array
    import random
    from .template import example_ids
    kept = [it for it in items if it.outputs is not None and it.weight > 0]
    if not kept:
        raise BakeError("no example has an answer to learn")
    order = list(range(len(kept)))
    random.Random(seed).shuffle(order)
    n_val = validation_count(len(kept), validation)
    val = set(order[:n_val])
    rows: List[Dict[str, Any]] = []
    kwargs = template.kwargs if template is not None else {}
    every = max(1000, len(kept) // 10)
    for k, it in enumerate(kept):
        msgs, answer = it.entry.messages(it.inputs, it.outputs)
        row: Dict[str, Any] = {"function": it.entry.name, "messages": msgs + [{"role": "assistant", "content": answer}],
                               "tag": it.tag, "weight": float(it.weight), "row_id": it.row_id,
                               "split": "validation" if k in val else "train", "source": it.source}
        if tokenizer is not None:
            ids, start = example_ids(tokenizer, msgs, answer, kwargs, template)
            row["input_ids"] = array.array("i", ids)
            row["answer_start"] = start
            row["prompt_tokens"] = start
            row["answer_tokens"] = len(ids) - start
        rows.append(row)
        if tokenizer is not None and (k + 1) % every == 0:
            log(f"tokenized {k + 1:,}/{len(kept):,} conversations")
    entries = {}
    for it in kept:
        entries.setdefault(it.entry.name, it.entry)
    return Examples(rows, student=student, template=template.to_dict() if template is not None else None,
                    entries=entries, info=info)


__all__ = ["Examples", "COLUMNS", "make", "validation_count"]
