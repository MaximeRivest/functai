"""A student a service trains, from its own published files (contract/training.md):
how a conversation, pictures included, becomes the exact tokens it is trained on
and called with, where its reply starts and ends, and how its generated tokens
read back as a reply.

    st = Student.load("Qwen/Qwen3.5-4B")                   # tokenizer.json, the chat template, picture settings
    r = st.render(messages, pictures, reply)               # r.prompt, r.full, r.reply_start
    st.read_reply(generated_tokens)                        # the reply text the function's layout reads

Needs ``tokenizers``, ``jinja2`` and ``transformers`` (no PyTorch): its chat
templates render as Hugging Face renders them. Inkling 2 Small, whose files are
not published yet, renders through ``tml-renderers`` (``functai[inkling2]``)
with Inkling-Small's template, the same format.
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import json
import math
import re
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .examples import BakeError

_CONTRACT = None


def students_table() -> Dict[str, Any]:
    """contract/students.json, as this package carries it."""
    global _CONTRACT
    if _CONTRACT is None:
        _CONTRACT = json.loads((Path(__file__).parent / "students.json").read_text())
    return _CONTRACT


# ------------------------------------------------------------------ pictures


@dataclasses.dataclass(frozen=True)
class PictureRule:
    """A family's picture rule with its numbers (contract/training.md §4)."""
    rule: str                      # "grid" | "qwen"
    placeholder: str
    patch: int
    merge: Optional[int] = None
    min_pixels: Optional[int] = None
    max_pixels: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in dataclasses.asdict(self).items() if v is not None}


@dataclasses.dataclass(frozen=True)
class Picture:
    """A picture as a service takes it: its bytes, size, format and hash."""
    data: bytes
    width: int
    height: int
    format: str                    # "png" | "jpeg"
    sha256: str                    # "sha256:<hex>"


def picture_size(b: bytes) -> Optional[Dict[str, Any]]:
    """A picture's width, height and format from its bytes (PNG's IHDR, a JPEG's first frame header), or None."""
    if len(b) >= 24 and b[:8] == b"\x89PNG\r\n\x1a\n":
        w, h = struct.unpack(">II", b[16:24])
        return {"width": w, "height": h, "format": "png"}
    if len(b) >= 4 and b[:2] == b"\xff\xd8":
        i = 2
        while i + 4 <= len(b):
            marker = b[i + 1]
            length = struct.unpack(">H", b[i + 2:i + 4])[0]
            if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                if i + 9 > len(b):
                    return None
                h, w = struct.unpack(">HH", b[i + 5:i + 9])
                return {"width": w, "height": h, "format": "jpeg"}
            i += 2 + length
    return None


def grid_tokens(w: int, h: int, patch: int) -> int:
    """Inkling's rule: ceil(w/P)·ceil(h/P), no resizing."""
    return math.ceil(w / patch) * math.ceil(h / patch)


def qwen_tokens(w: int, h: int, patch: int, merge: int, min_pixels: int, max_pixels: int) -> Optional[int]:
    """Qwen's rule (smart resize into the pixel range, multiples of patch·merge, round half to even); None past an aspect ratio of 200."""
    f = patch * merge
    if max(w, h) / min(w, h) > 200:
        return None
    hb = round(h / f) * f                  # Python's round: half to even, as the rule says
    wb = round(w / f) * f
    if hb * wb > max_pixels:
        b = math.sqrt(h * w / max_pixels)
        hb = max(f, math.floor(h / b / f) * f)
        wb = max(f, math.floor(w / b / f) * f)
    elif hb * wb < min_pixels:
        b = math.sqrt(min_pixels / (h * w))
        hb = math.ceil(h * b / f) * f
        wb = math.ceil(w * b / f) * f
    return (hb // f) * (wb // f)


def picture_tokens(rule: PictureRule, w: int, h: int) -> int:
    if rule.rule == "grid":
        return grid_tokens(w, h, rule.patch)
    n = qwen_tokens(w, h, rule.patch, rule.merge, rule.min_pixels, rule.max_pixels)
    if n is None:
        raise BakeError(f"a {w}×{h} picture is too long and thin for this student (its sides' ratio is over 200)")
    return n


def picture(data: bytes) -> Picture:
    """A picture from its bytes; refused when it is not a PNG or a JPEG (``training-picture-format``)."""
    size = picture_size(data)
    if size is None:
        raise BakeError("a picture a service trains on is a PNG or a JPEG; convert it (another format, or bytes that "
                        "are not a picture)")
    return Picture(data=bytes(data), sha256="sha256:" + hashlib.sha256(data).hexdigest(), **size)


# ------------------------------------------------------------------ conversations


def student_conversation(request: Any) -> Tuple[List[Dict[str, Any]], List[Picture]]:
    """An lm15 request (object or canonical JSON) as a student's conversation
    (contract/training.md §2): the system text first, a developer message as a
    system one; a message with only text has its text as ``content``, one with
    pictures a list (adjacent text joined, each picture ``{"type": "image"}``);
    the pictures, numbered in order, beside."""
    if not isinstance(request, dict):
        import lm15
        request = lm15.serde.request_to_dict(request)
    messages: List[Dict[str, Any]] = []
    pictures: List[Picture] = []
    system = request.get("system")
    if system:
        messages.append({"role": "system", "content": system if isinstance(system, str) else
                         "".join(p.get("text", "") for p in system)})
    for m in request.get("messages", []):
        role = m["role"]
        if role not in ("user", "assistant", "system", "developer"):
            raise BakeError(f"a student reads chats; the request has a {role!r} message (tool calls are not trained yet)")
        parts: List[Dict[str, Any]] = []
        for p in m.get("parts", []):
            if p.get("type") == "text":
                if parts and parts[-1]["type"] == "text":
                    parts[-1]["text"] += p["text"]
                else:
                    parts.append({"type": "text", "text": p["text"]})
            elif p.get("type") == "image":
                data = p.get("data")
                if not isinstance(data, str):
                    raise BakeError("a picture given by its address (url, file id) cannot be trained on: give its bytes")
                pictures.append(picture(base64.b64decode(data)))
                parts.append({"type": "image"})
            else:
                raise BakeError(f"a student reads text and pictures; the request carries a {p.get('type')} part")
        r = "system" if role == "developer" else role
        if all(x["type"] == "text" for x in parts):
            messages.append({"role": r, "content": "".join(x["text"] for x in parts)})
        else:
            messages.append({"role": r, "content": parts})
    return messages, pictures


# ------------------------------------------------------------------ the student


def family_of(student: str) -> Optional[Dict[str, Any]]:
    repo = student.split(":")[0]
    for f in students_table()["families"]:
        if any(repo.startswith(p) for p in f["prefixes"]):
            return f
    return None


def effort_of(effort: Union[float, str, None]) -> float:
    """An effort as a number: a number in [0, 0.99], or one of Inkling's preset names."""
    if effort is None:
        return 0.0
    if isinstance(effort, (int, float)) and not isinstance(effort, bool):
        if not 0 <= effort <= 0.99:
            raise ValueError(f"an effort is a number from 0 to 0.99, not {effort}")
        return float(effort)
    table = students_table()["efforts"]
    if effort not in table or effort == "about":
        raise ValueError(f"an effort is a number from 0 to 0.99 or one of "
                         f"{', '.join(k for k in table if k != 'about')}, not {effort!r}")
    return float(table[effort])


def template_kwargs(template: str, effort: float = 0.0) -> Dict[str, Any]:
    """The template's keyword arguments (§1): thinking off, the effort when it takes one."""
    out: Dict[str, Any] = {}
    if "enable_thinking" in template:
        out["enable_thinking"] = False
    if "reasoning_effort" in template:
        out["reasoning_effort"] = effort
    return out


def _dig(o: Any, path: str) -> Any:
    for k in path.split("."):
        o = o.get(k) if isinstance(o, dict) else None
    return o


def picture_rule_of(family: Dict[str, Any], config: Dict[str, Any]) -> PictureRule:
    p = family["picture"]

    def num(key: str) -> int:
        v = _dig(config, p[key])
        if not isinstance(v, (int, float)):
            raise BakeError(f"the student's {p['file']} has no {p[key]}")
        return int(v)
    if p["rule"] == "grid":
        return PictureRule("grid", p["placeholder"], num("patch"))
    return PictureRule("qwen", p["placeholder"], num("patch"), num("merge"), num("min_pixels"), num("max_pixels"))


class _HFTokens:
    """tokenizer.json through the ``tokenizers`` library."""

    def __init__(self, text: str):
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_str(text)
        data = json.loads(text)
        self.specials = {t["id"] for t in data.get("added_tokens", []) if t.get("special")}

    def encode(self, text: str) -> List[int]:
        return self.tok.encode(text, add_special_tokens=False).ids

    def decode(self, ids: Sequence[int]) -> str:
        return self.tok.decode(list(ids), skip_special_tokens=False) if ids else ""

    def token_id(self, text: str) -> Optional[int]:
        return self.tok.token_to_id(text)


class _TMLTokens:
    """Inkling 2 Small's h200k tokenizer, from tml-renderers (until its files are published)."""

    SPECIAL = re.compile(r"<\|([A-Za-z0-9_]+)\|>")

    def __init__(self):
        try:
            import torch  # noqa: F401 — tml-renderers looks for PyTorch's library in memory when it loads
            from tml_renderers import tokenizers
        except ImportError as err:
            raise BakeError('Inkling 2 Small\'s files are not published yet; its tokens come from tml-renderers until '
                            'then: pip install "functai[inkling2]" (it needs PyTorch; the CPU build is enough)') from err
        self.tok = tokenizers.H200kBaseChatTokenizer()
        self.names = set(self.tok.special_tokens())
        self.specials = {self.tok.encode_special(n) for n in self.names}

    def encode(self, text: str) -> List[int]:
        out: List[int] = []
        at = 0
        for m in self.SPECIAL.finditer(text):
            if m.group(1) not in self.names:
                continue
            if m.start() > at:
                out += self.tok.encode_ordinary(text[at:m.start()])
            out.append(self.tok.encode_special(m.group(1)))
            at = m.end()
        if at < len(text):
            out += self.tok.encode_ordinary(text[at:])
        return out

    def decode(self, ids: Sequence[int]) -> str:
        return self.tok.decode(list(ids)) if ids else ""

    def token_id(self, text: str) -> Optional[int]:
        m = self.SPECIAL.fullmatch(text)
        if m and m.group(1) in self.names:
            return self.tok.encode_special(m.group(1))
        ids = self.tok.encode_ordinary(text)
        return ids[0] if len(ids) == 1 else None


@dataclasses.dataclass
class Rendered:
    """A conversation as a student's tokens (contract/training.md §3)."""
    prompt: List[Dict[str, Any]]
    full: Optional[List[Dict[str, Any]]]
    reply_start: int
    prefix_fallback: bool = False


def chunks_length(chunks: Sequence[Dict[str, Any]]) -> int:
    """A chunk list's length: a picture counts its positions."""
    return sum(len(c["text"]) if "text" in c else c["tokens"] for c in chunks)


def _render(template: str, messages: Sequence[Dict[str, Any]], generation: bool, kwargs: Dict[str, Any]) -> str:
    from transformers.utils.chat_template_utils import render_jinja_template
    try:
        rendered, _ = render_jinja_template(conversations=[list(messages)], chat_template=template,
                                            add_generation_prompt=generation, **kwargs)
    except Exception as err:  # noqa: BLE001 — the template's raise_exception, or a template error
        raise BakeError(f"the student's chat template refused the conversation: {err}") from err
    return rendered[0]


class Student:
    """A student: its tokenizer, template and picture rule, as the contract says to use them."""

    def __init__(self, student: str, tokens: Any, template: str, *, picture_rule: Optional[PictureRule] = None,
                 tokenizer_config: Optional[Dict[str, Any]] = None, effort: float = 0.0, source: str = "files",
                 tokenizer_sha256: str = ""):
        self.id = student
        self.repo = student.split(":")[0]
        self.template = template
        self.kwargs = template_kwargs(template, effort)
        self.picture = picture_rule
        self.source = source
        self._tok = tokens
        self._extra: Dict[str, Any] = {}
        for k in ("bos_token", "eos_token", "unk_token", "pad_token"):
            v = (tokenizer_config or {}).get(k)
            if isinstance(v, str):
                self._extra[k] = v
            elif isinstance(v, dict) and isinstance(v.get("content"), str):
                self._extra[k] = v["content"]
        self.template_sha256 = "sha256:" + hashlib.sha256(template.encode()).hexdigest()
        self.tokenizer_sha256 = tokenizer_sha256
        self.lead, self.end, self.stop_token, self.end_head = self._ends()

    # ---- making one

    @classmethod
    def from_files(cls, student: str, tokenizer: str, template: str, *, picture_rule: Optional[PictureRule] = None,
                   tokenizer_config: Optional[Dict[str, Any]] = None, effort: Union[float, str, None] = None) -> "Student":
        """A student from its files given as text (tests, files kept elsewhere)."""
        return cls(student, _HFTokens(tokenizer), template, picture_rule=picture_rule, tokenizer_config=tokenizer_config,
                   effort=effort_of(effort), tokenizer_sha256="sha256:" + hashlib.sha256(tokenizer.encode()).hexdigest())

    @classmethod
    def load(cls, student: str, *, effort: Union[float, str, None] = None, folder: "str | Path | None" = None,
             local_files_only: bool = False) -> "Student":
        """A student by the service's id (``Qwen/Qwen3.5-4B``,
        ``thinkingmachines/Inkling-2-Small:peft:262144``), from the Hugging Face
        repository of its name up to the first ``:`` (or ``folder``)."""
        repo = student.split(":")[0]
        family = family_of(student)

        def get(name: str) -> Optional[str]:
            if folder is not None:
                p = Path(folder) / name
                return p.read_text() if p.exists() else None
            from huggingface_hub import hf_hub_download
            try:
                return Path(hf_hub_download(repo, name, local_files_only=local_files_only)).read_text()
            except Exception:  # noqa: BLE001 — absent, gated, offline: None, said by the caller
                return None
        tok_text = get("tokenizer.json")
        if tok_text is None:
            unpublished = (family or {}).get("unpublished", {}).get(repo)
            if unpublished and folder is None:
                return cls._tml(student, family, effort_of(effort), local_files_only)
            raise BakeError(f"{repo} has no published tokenizer.json (or it cannot be read here); a student's tokens "
                            f"are made from its published files")
        cfg_text = get("tokenizer_config.json")
        cfg = json.loads(cfg_text) if cfg_text else {}
        template = get("chat_template.jinja") or (cfg.get("chat_template") if isinstance(cfg.get("chat_template"), str)
                                                  else None)
        if not template:
            raise BakeError(f"{repo} has no chat template: a student is a chat (instruct) model")
        rule = None
        if family:
            text = get(family["picture"]["file"])
            if text:
                rule = picture_rule_of(family, json.loads(text))
        return cls(student, _HFTokens(tok_text), template, picture_rule=rule, tokenizer_config=cfg,
                   effort=effort_of(effort), tokenizer_sha256="sha256:" + hashlib.sha256(tok_text.encode()).hexdigest())

    @classmethod
    def _tml(cls, student: str, family: Dict[str, Any], effort: float, local_files_only: bool) -> "Student":
        """Inkling 2 Small before its files: h200k tokens, Inkling-Small's template and picture settings."""
        from huggingface_hub import hf_hub_download
        base = "thinkingmachines/Inkling-Small"
        try:
            template = Path(hf_hub_download(base, "chat_template.jinja", local_files_only=local_files_only)).read_text()
            cfg = json.loads(Path(hf_hub_download(base, family["picture"]["file"],
                                                  local_files_only=local_files_only)).read_text())
        except Exception as err:  # noqa: BLE001
            raise BakeError(f"{student} renders with {base}'s template until its own files are out, and that cannot "
                            f"be read here ({type(err).__name__})") from err
        import importlib.metadata as md
        tokens = _TMLTokens()
        version = md.version("tml-renderers")
        return cls(student, tokens, template, picture_rule=picture_rule_of(family, cfg), effort=effort,
                   source="tml-renderers", tokenizer_sha256=f"tml-renderers:{version}:h200k")

    # ---- tokens

    def encode(self, text: str) -> List[int]:
        return self._tok.encode(text)

    def decode(self, ids: Sequence[int]) -> str:
        return self._tok.decode(ids)

    def token_id(self, text: str) -> Optional[int]:
        return self._tok.token_id(text)

    def _render(self, messages: Sequence[Dict[str, Any]], generation: bool) -> str:
        return _render(self.template, messages, generation, {**self._extra, **self.kwargs})

    def _ends(self) -> Tuple[str, str, int, str]:
        s = "\u2063R\u2063"
        probe = [{"role": "user", "content": "x"}]
        p = self._render(probe, True)
        f = self._render(probe + [{"role": "assistant", "content": s}], False)
        at = f.index(s)
        lead = f[len(p):at] if f.startswith(p) and at >= len(p) else ""
        end = f[at + len(s):]
        stop = None
        for t in self.encode(end):
            if t in self._tok.specials:
                stop = t
        if stop is None and self._extra.get("eos_token"):
            stop = self.token_id(self._extra["eos_token"])
        if stop is None:
            raise BakeError(f"{self.id}'s template writes nothing that ends a reply, and its tokenizer has no "
                            f"end-of-sequence token")
        stop_text = self.decode([stop])
        head = end[:end.index(stop_text)] if stop_text in end else ""
        return lead, end, stop, head

    def _chunks(self, ids: Sequence[int], pictures: Sequence[Picture]) -> Tuple[List[Dict[str, Any]], int]:
        ph = self.token_id(self.picture.placeholder) if self.picture else None
        out: List[Dict[str, Any]] = []
        cur: List[int] = []
        k = 0
        for t in ids:
            if ph is not None and t == ph:
                if k >= len(pictures):
                    raise BakeError(f"the template wrote more pictures than the conversation has ({len(pictures)})")
                if cur:
                    out.append({"text": cur})
                pic = pictures[k]
                out.append({"picture": k, "tokens": picture_tokens(self.picture, pic.width, pic.height)})
                cur = []
                k += 1
            else:
                cur.append(t)
        if cur:
            out.append({"text": cur})
        return out, k

    def render(self, messages: Sequence[Dict[str, Any]], pictures: Sequence[Picture] = (),
               reply: Optional[str] = None) -> Rendered:
        """A conversation's chunks (contract/training.md §3): the prompt, and with
        ``reply`` the whole conversation and where the reply starts."""
        if pictures and not self.picture:
            raise BakeError(f"{self.id} has no picture rule (contract/students.json): it is not trained on pictures here")
        prompt_ids = self.encode(self._render(messages, True))
        prompt, used = self._chunks(prompt_ids, pictures)
        if used != len(pictures):
            raise BakeError(f"the conversation has {len(pictures)} pictures and the template shows {used}")
        start = chunks_length(prompt)
        if reply is None:
            return Rendered(prompt, None, start)
        full_ids = self.encode(self._render(list(messages) + [{"role": "assistant", "content": reply}], False))
        fallback = False
        if full_ids[:len(prompt_ids)] != prompt_ids:
            full_ids = prompt_ids + self.encode(reply + self.end)
            fallback = True
        full, used = self._chunks(full_ids, pictures)
        if used != len(pictures):
            raise BakeError(f"the conversation has {len(pictures)} pictures and the template shows {used}")
        return Rendered(prompt, full, start, fallback)

    def read_reply(self, generated: Sequence[int]) -> str:
        """The reply text in generated tokens (contract/training.md §5)."""
        ids = list(generated)
        if ids and ids[-1] == self.stop_token:
            ids = ids[:-1]
        text = self.decode(ids)
        if self.lead and self.lead in text:
            text = text[text.rindex(self.lead) + len(self.lead):]
        if self.end_head and self.end_head in text:
            text = text[:text.index(self.end_head)]
        return text

    def describe(self) -> Dict[str, Any]:
        """What ``baked.json`` keeps of the template (contract/training.md §11)."""
        return {"sha256": self.template_sha256, "kwargs": dict(self.kwargs), "end": self.end,
                "stop_token_ids": [self.stop_token], "source": self.source, "repo": self.repo,
                "tokenizer_sha256": self.tokenizer_sha256, "lead": self.lead,
                "picture": self.picture.to_dict() if self.picture else None}


__all__ = ["Student", "Picture", "PictureRule", "Rendered", "picture", "picture_size", "grid_tokens", "qwen_tokens",
           "picture_tokens", "student_conversation", "chunks_length", "family_of", "effort_of", "template_kwargs",
           "students_table"]
