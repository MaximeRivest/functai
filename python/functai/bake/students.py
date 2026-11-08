"""What bake knows about a student model before loading it: its size, shape,
context and chat template, read from its config files (a few kilobytes from
the Hugging Face Hub, or a local folder), so a plan can be made without
PyTorch and without downloading weights.

The default students are a short tested list. Any Hugging Face causal model
with a chat template works by name.
"""

from __future__ import annotations

import dataclasses
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

from .examples import BakeError

# The students bake picks from when none is named, largest first. 4B is the
# default: strong enough to learn most single tasks from a teacher, still cheap
# to run; a smaller one when the place it trains cannot train 4B in a working
# day. Larger students (9B and up) only when named.
DEFAULT_STUDENTS = ("Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-0.8B")


@dataclasses.dataclass(frozen=True)
class StudentInfo:
    """A student's shape, as its config files say."""
    name: str
    parameters: int                  # every weight (vision towers included when the checkpoint has them)
    hidden: int
    layers: int
    intermediate: int
    heads: int
    kv_heads: int
    head_dim: int
    vocab: int
    context: Optional[int]           # the longest sequence it was built for
    layer_types: tuple               # per layer: "full_attention", "linear_attention", ...
    tied_embeddings: bool
    chat_template: bool
    family: str                      # "qwen", "llama", ... (the learning-rate rule's family)

    @property
    def hybrid(self) -> bool:
        """It has layers that are not attention (linear attention, state space):
        their state crosses the boundary between packed examples."""
        return any(t != "full_attention" for t in self.layer_types)

    @property
    def full_attention_layers(self) -> int:
        return sum(t == "full_attention" for t in self.layer_types) if self.layer_types else self.layers

    @property
    def embedding_parameters(self) -> int:
        return self.vocab * self.hidden * (1 if self.tied_embeddings else 2)

    def lora_parameters(self, rank: int) -> int:
        """Parameters of a LoRA adapter of ``rank`` on every linear layer of every block."""
        d, f = self.hidden, self.intermediate
        attn = self.heads * self.head_dim
        kv = self.kv_heads * self.head_dim
        per_block = rank * ((d + attn) + 2 * (d + kv) + (attn + d) + 2 * (d + f) + (f + d))
        return per_block * self.layers

    def to_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        d["layer_types"] = list(self.layer_types)
        return d


def _read(name: str, filename: str, local_files_only: bool) -> Optional[Dict[str, Any]]:
    path = Path(name).expanduser()
    if path.is_dir():
        f = path / filename
        return json.loads(f.read_text()) if f.exists() else None
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError
    except ImportError as err:
        raise ImportError('bake reads student models from the Hugging Face Hub: pip install "functai[bake]" '
                          '(or "functai[tinker]")') from err
    try:
        return json.loads(Path(hf_hub_download(name, filename, local_files_only=local_files_only)).read_text())
    except EntryNotFoundError:
        return None
    except Exception as exc:  # noqa: BLE001 — not found, offline, gated: said once, plainly
        if filename == "config.json":
            raise BakeError(f"cannot read the student {name!r} from the Hugging Face Hub ({type(exc).__name__}: "
                            f"{exc}). Check the name, your connection, or HF_TOKEN for gated models") from None
        return None


def _has_template(name: str, local_files_only: bool) -> bool:
    path = Path(name).expanduser()
    if path.is_dir():
        if (path / "chat_template.jinja").exists() or (path / "chat_template.json").exists():
            return True
        cfg = path / "tokenizer_config.json"
        return cfg.exists() and bool(json.loads(cfg.read_text()).get("chat_template"))
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(name, "chat_template.jinja", local_files_only=local_files_only)
        return True
    except Exception:  # noqa: BLE001 — older repos keep it in tokenizer_config.json
        cfg = _read(name, "tokenizer_config.json", local_files_only)
        return bool(cfg and cfg.get("chat_template"))


def _family(name: str, model_type: str) -> str:
    low = f"{name} {model_type}".lower()
    for fam in ("qwen", "llama", "gemma", "mistral", "phi", "smollm", "olmo", "granite", "gpt_oss", "deepseek"):
        if fam in low:
            return fam
    return model_type or "unknown"


@lru_cache(maxsize=64)
def _info(name: str, local_files_only: bool) -> StudentInfo:
    cfg = _read(name, "config.json", local_files_only)
    if cfg is None:
        raise BakeError(f"{name!r} has no config.json: it is not a Hugging Face model")
    text = cfg.get("text_config") or cfg              # vision-language checkpoints nest the language model
    hidden = int(text.get("hidden_size") or text.get("d_model") or 0)
    layers = int(text.get("num_hidden_layers") or text.get("n_layer") or 0)
    heads = int(text.get("num_attention_heads") or 0)
    if not hidden or not layers or not heads:
        raise BakeError(f"{name!r} does not look like a language model (no hidden size, layers or heads in its config)")
    head_dim = int(text.get("head_dim") or hidden // heads)
    vocab = int(text.get("vocab_size") or cfg.get("vocab_size") or 0)
    inter = int(text.get("intermediate_size") or 4 * hidden)
    types = tuple(text.get("layer_types") or ())
    if not types and text.get("model_type", "").startswith(("mamba", "falcon_h", "jamba", "zamba")):
        types = ("state_space",) * layers
    tied = bool(text.get("tie_word_embeddings", cfg.get("tie_word_embeddings", False)))
    index = _read(name, "model.safetensors.index.json", local_files_only)
    size = (index or {}).get("metadata", {}).get("total_size")
    if size:
        dtype = str(text.get("dtype") or text.get("torch_dtype") or cfg.get("torch_dtype") or "bfloat16")
        params = int(size / (4 if "32" in dtype else 2))
    else:     # one-file checkpoints: count from the shape
        kv = int(text.get("num_key_value_heads") or heads)
        per = hidden * heads * head_dim * 2 + hidden * kv * head_dim * 2 + 3 * hidden * inter
        params = layers * per + vocab * hidden * (1 if tied else 2)
    context = text.get("max_position_embeddings") or cfg.get("max_position_embeddings")
    return StudentInfo(name=name, parameters=params, hidden=hidden, layers=layers, intermediate=inter, heads=heads,
                       kv_heads=int(text.get("num_key_value_heads") or heads), head_dim=head_dim, vocab=vocab,
                       context=int(context) if context else None, layer_types=types, tied_embeddings=tied,
                       chat_template=_has_template(name, local_files_only),
                       family=_family(name, str(text.get("model_type") or cfg.get("model_type") or "")))


def info(name: str, *, local_files_only: bool = False) -> StudentInfo:
    """A student's shape from its config files (cached for the session)."""
    local_files_only = local_files_only or os.environ.get("HF_HUB_OFFLINE") == "1"
    try:
        return _info(name, local_files_only)
    except BakeError:
        if not local_files_only:
            raise
        raise BakeError(f"{name!r} is not in the local Hugging Face cache (local_files_only=True)") from None


def learning_rate(student: StudentInfo, *, lora: bool) -> float:
    """The learning rate bake starts from: Thinking Machines' fitted rule
    (tinker-cookbook ``hyperparam_utils.get_lr``, from "LoRA Without Regret",
    2025): 5e-5 for full fine-tuning, ten times that for LoRA (alpha 32), scaled
    by (2000 / hidden size) to a power fitted per family (Qwen 0.0775, Llama
    0.781; others take Qwen's, the flatter one)."""
    exponent = 0.781 if student.family == "llama" else 0.0775
    lr = 5e-5 * (10.0 if lora else 1.0)
    return lr * (2000 / student.hidden) ** exponent


def lora_rank(answer_tokens: int) -> int:
    """LoRA's rank by how much there is to learn. "LoRA Without Regret" found
    rank 32 matches full fine-tuning on small and medium supervised sets and
    that capacity runs out only on large ones; bake doubles it past 20M and
    again past 100M answer tokens."""
    return 32 if answer_tokens < 20e6 else 64 if answer_tokens < 100e6 else 128


__all__ = ["StudentInfo", "info", "learning_rate", "lora_rank", "DEFAULT_STUDENTS"]
