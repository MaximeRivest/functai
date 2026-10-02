"""From chat messages to a student's tokens, with its own chat template.

The tokens are made once, here, and travel: trainers that take tokens (TRL
with ``input_ids``, Tinker) train on exactly these, and runners that take
tokens (in-process, vLLM's completions, Tinker's sampler) are called with the
same function. ``Template`` records how (the template's hash, its keyword
arguments, and the token that ends a reply) so a model trained elsewhere can
be checked against it (``adopt``).
"""

from __future__ import annotations

import dataclasses
import hashlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .examples import BakeError


def template_kwargs(tokenizer) -> Dict[str, Any]:
    """Qwen3-style templates think by default; a student answers directly
    (the layout's reasoning output, when trained, is its thinking)."""
    return {"enable_thinking": False} if "enable_thinking" in (tokenizer.chat_template or "") else {}


@dataclasses.dataclass
class Template:
    """How a student's chat template turns messages into tokens."""
    sha256: str
    kwargs: Dict[str, Any]
    end: str                          # what the template writes after a reply ("<|im_end|>\n")
    stop_token_ids: List[int]         # tokens that end a reply when generating

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Template":
        return cls(d["sha256"], dict(d.get("kwargs") or {}), d.get("end", ""), list(d.get("stop_token_ids") or []))


def load_tokenizer(name: str, *, local_files_only: bool = False):
    try:
        from transformers import AutoTokenizer
        from transformers.utils import logging as hf_logging
    except ImportError as err:
        raise ImportError('bake writes a student\'s tokens with its tokenizer: pip install "functai[bake]" '
                          '(or "functai[tinker]")') from err
    before = hf_logging.get_verbosity()
    hf_logging.set_verbosity_error()
    try:
        tok = AutoTokenizer.from_pretrained(name, local_files_only=local_files_only)
    finally:
        hf_logging.set_verbosity(before)
    if not tok.chat_template:
        raise BakeError(f"{name} has no chat template; a generative student is a chat (instruct) model")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def describe(tokenizer) -> Template:
    kwargs = template_kwargs(tokenizer)
    probe = [{"role": "user", "content": "x"}]
    marker = "\u2063END\u2063"
    f = tokenizer.apply_chat_template(probe + [{"role": "assistant", "content": marker}], tokenize=False, **kwargs)
    end = f.split(marker, 1)[1] if marker in f else (tokenizer.eos_token or "")
    stops: List[int] = []
    if end.strip():
        first = tokenizer(end, add_special_tokens=False)["input_ids"][:1]
        stops += first
    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id not in stops:
        stops.append(tokenizer.eos_token_id)
    return Template(sha256="sha256:" + hashlib.sha256((tokenizer.chat_template or "").encode()).hexdigest(),
                    kwargs=kwargs, end=end, stop_token_ids=stops)


def prompt_ids(tokenizer, messages: List[Dict[str, str]], kwargs: Dict[str, Any]) -> List[int]:
    return list(tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                              return_dict=False, **kwargs))


def example_ids(tokenizer, messages: List[Dict[str, str]], answer: str, kwargs: Dict[str, Any],
                template: Optional[Template] = None) -> Tuple[List[int], int]:
    """Token ids of prompt + reply (+ what the template writes after it), and
    where the reply starts: the loss is on the reply only. The prompt's ids are
    exactly ``prompt_ids``, what a call sends."""
    prompt = prompt_ids(tokenizer, messages, kwargs)
    full = list(tokenizer.apply_chat_template(messages + [{"role": "assistant", "content": answer}],
                                              tokenize=True, return_dict=False, **kwargs))
    if full[:len(prompt)] != prompt:
        # a template that writes the last reply differently from a generation
        # prompt: keep the call's prompt, and append the reply and the end
        end = template.end if template is not None else (tokenizer.eos_token or "")
        full = prompt + tokenizer(answer + end, add_special_tokens=False)["input_ids"]
    return full, len(prompt)


def lengths(tokenizer, texts: Sequence[str]) -> List[int]:
    return [len(x) for x in tokenizer(list(texts), add_special_tokens=False)["input_ids"]]


__all__ = ["Template", "template_kwargs", "load_tokenizer", "describe", "prompt_ids", "example_ids"]
