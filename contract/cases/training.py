"""The cases in training/: tokens from a student's files, pictures, reading a
reply, training examples, advantages, the energy score, losses as weights,
and choosing what to keep, from ../training.md.

Turning a template into text is Jinja's rule as Hugging Face applies it
(``ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)``,
``raise_exception``), and text into tokens is the ``tokenizers`` library's
(both borrowed, as baked.py borrows lmcc); everything else is written here
from the rules. Never imports FunctAI.

The tokenizer (``toy-tokenizer.json``) is a small byte-level BPE written
here, with the special tokens of two toy templates shaped like Inkling's
and Qwen's (``inkling-like.jinja``, ``qwen-like.jinja``). A case names
them; a harness loads them as it would a student's published files.
"""

import base64
import json
import math
import struct
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "training"

SPECIALS = ["<|endoftext|>", "<|message_user|>", "<|message_model|>", "<|message_system|>", "<|content_text|>",
            "<|content_image|>", "<|content_thinking|>", "<|end_message|>", "<|content_model_end_sampling|>",
            "<|pic|>", "<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|image_pad|>", "<|vision_end|>"]

INKLING_LIKE = r"""{%- set role_token = {"user": "<|message_user|>", "assistant": "<|message_model|>", "system": "<|message_system|>"} -%}
{%- macro effort() -%}
{%- set eff = reasoning_effort if reasoning_effort is defined and reasoning_effort is not none else 0.9 -%}
{{- "<|message_system|><|content_text|>Thinking effort level: " -}}{%- if eff == 0.0 -%}0{%- else -%}{{ eff }}{%- endif -%}{{- "<|end_message|>" -}}
{%- endmacro -%}
{%- set state = namespace(done=false) -%}
{%- for message in messages -%}
{%- if not state.done and message.role != "system" -%}{{- effort() -}}{%- set state.done = true -%}{%- endif -%}
{%- set rtok = role_token[message.role] -%}
{%- if message.role == "assistant" and message.reasoning_content is defined and message.reasoning_content -%}
{{- "<|message_model|><|content_thinking|>" ~ message.reasoning_content ~ "<|end_message|>" -}}
{%- endif -%}
{%- if message.content is string -%}
{{- rtok ~ "<|content_text|>" ~ message.content ~ "<|end_message|>" -}}
{%- else -%}
{%- for part in message.content -%}
{%- if part.type == "text" -%}{{- rtok ~ "<|content_text|>" ~ part.text ~ "<|end_message|>" -}}
{%- elif part.type == "image" -%}{{- rtok ~ "<|content_image|><|pic|><|end_message|>" -}}
{%- else -%}{{- raise_exception("Unsupported content part type: " ~ part.type) -}}{%- endif -%}
{%- endfor -%}
{%- endif -%}
{%- if message.role == "assistant" -%}{{- "<|content_model_end_sampling|>" -}}{%- endif -%}
{%- endfor -%}
{%- if not state.done -%}{{- effort() -}}{%- endif -%}
{%- if add_generation_prompt -%}{{- "<|message_model|>" -}}{%- endif -%}"""

QWEN_LIKE = r"""{%- for message in messages -%}
{{- "<|im_start|>" ~ message.role ~ "\n" -}}
{%- if message.role == "assistant" and enable_thinking is defined and enable_thinking is false -%}{{- "<think>\n\n</think>\n\n" -}}{%- endif -%}
{%- if message.content is string -%}{{- message.content -}}
{%- else -%}{%- for part in message.content -%}
{%- if part.type == "image" -%}{{- "<|vision_start|><|image_pad|><|vision_end|>" -}}{%- else -%}{{- part.text -}}{%- endif -%}
{%- endfor -%}{%- endif -%}
{{- "<|im_end|>\n" -}}
{%- endfor -%}
{%- if add_generation_prompt -%}
{{- "<|im_start|>assistant\n" -}}
{%- if enable_thinking is defined and enable_thinking is false -%}{{- "<think>\n\n</think>\n\n" -}}{%- endif -%}
{%- endif -%}"""

# A template that writes an earlier reply differently from how the model writes it (it drops the
# reply's thinking block on the last turn): the prompt is not where the full conversation begins.
NOT_A_PREFIX = r"""{%- for message in messages -%}
{{- "<|im_start|>" ~ message.role ~ "\n" ~ message.content ~ "<|im_end|>\n" -}}
{%- endfor -%}
{%- if add_generation_prompt -%}{{- "<|im_start|>assistant\n<think>\n" -}}{%- endif -%}"""

TEMPLATES = {"inkling-like.jinja": INKLING_LIKE, "qwen-like.jinja": QWEN_LIKE, "not-a-prefix.jinja": NOT_A_PREFIX}


# ------------------------------------------------------------------ the borrowed rules


def toy_tokenizer() -> str:
    """A byte-level BPE: GPT-2's pre-tokenizer, 256 byte symbols, a few merges, and the special tokens."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from tokenizers.pre_tokenizers import ByteLevel
    alphabet = ByteLevel.alphabet()
    vocab = {ch: i for i, ch in enumerate(sorted(alphabet))}
    merges = []

    def merge(a, b):
        merges.append((a, b))
        vocab.setdefault(a + b, len(vocab))
    for a, b in [("Ġ", "t"), ("h", "e"), ("Ġt", "he"), ("i", "n"), ("e", "r"), ("o", "n"), ("<", "/"),
                 ("Ċ", "<"), ("Ċ", "Ċ"), ("r", "e"), ("s", "u"), ("l", "t"), ("su", "lt"), ("re", "sult"),
                 ("w", "h"), ("wh", "y"), ("Ġ", "b"), ("Ġb", "u"), ("t", "t"), ("Ġbu", "tt"), ("Ġbutt", "on"),
                 ("<", "s"), ("<s", "c"), ("Ġ", "["), ("[", "1"), ("Ġ", "2"), ("2", "]")]:
        merge(a, b)
    tok = Tokenizer(models.BPE(vocab=vocab, merges=merges))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()
    tok.add_special_tokens(SPECIALS)
    return tok.to_str(pretty=True)


def tokenizer():
    from tokenizers import Tokenizer
    return Tokenizer.from_str(toy_tokenizer())


def render(template: str, messages, add_generation_prompt: bool, kwargs: dict) -> str:
    from jinja2.exceptions import TemplateError
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(message):
        raise TemplateError(message)
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
    env.globals["raise_exception"] = raise_exception
    return env.from_string(template).render(messages=messages, add_generation_prompt=add_generation_prompt, **kwargs)


def encode(text: str):
    return tokenizer().encode(text, add_special_tokens=False).ids


def token_id(text: str) -> int:
    return tokenizer().token_to_id(text)


def decode(ids):
    return tokenizer().decode(ids, skip_special_tokens=False)


# ------------------------------------------------------------------ pictures


def png(w: int, h: int) -> bytes:
    """A valid PNG of w x h grey pixels."""
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    raw = b"".join(b"\x00" + b"\x80" * w for _ in range(h))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def jpeg_header(w: int, h: int, sof: int = 0xC0) -> bytes:
    """The start of a JPEG: SOI, an APP0 segment, a DQT-like segment, then the start of frame."""
    app0 = b"\xff\xe0" + struct.pack(">H", 16) + b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    dqt = b"\xff\xdb" + struct.pack(">H", 4) + b"\x00\x00"
    frame = bytes([0xFF, sof]) + struct.pack(">HBHHB", 11, 8, h, w, 1) + b"\x01\x11\x00"
    return b"\xff\xd8" + app0 + dqt + frame + b"\xff\xd9"


def picture_size(data: bytes):
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return list(struct.unpack(">II", data[16:24])) + ["png"]
    if data[:2] == b"\xff\xd8":
        i = 2
        while i + 4 <= len(data):
            marker = data[i + 1]
            length = struct.unpack(">H", data[i + 2:i + 4])[0]
            if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                h, w = struct.unpack(">HH", data[i + 5:i + 9])
                return [w, h, "jpeg"]
            i += 2 + length
    return None


def grid_tokens(w, h, patch):
    return math.ceil(w / patch) * math.ceil(h / patch)


def qwen_tokens(w, h, patch, merge, min_pixels, max_pixels):
    f = patch * merge
    if max(w, h) / min(w, h) > 200:
        return None
    hb = round(h / f) * f                       # Python's round: half to even, as the rule says
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


# ------------------------------------------------------------------ the rules of training.md


def kwargs_for(template: str, effort: float = 0.0) -> dict:
    out = {}
    if "enable_thinking" in template:
        out["enable_thinking"] = False
    if "reasoning_effort" in template:
        out["reasoning_effort"] = effort
    return out


def merged(chunks):
    out = []
    for c in chunks:
        if "text" in c and out and "text" in out[-1]:
            out[-1] = {"text": out[-1]["text"] + c["text"]}
        elif "text" in c and not c["text"]:
            continue
        else:
            out.append(c)
    return out


def chunked(ids, placeholder, counts, start=0):
    """ids → chunks: each placeholder token, in order, the next picture."""
    chunks, k = [], start
    cur = []
    for t in ids:
        if t == placeholder:
            if k >= len(counts):
                return None, k
            chunks.append({"text": cur})
            chunks.append({"picture": k, "tokens": counts[k]})
            cur, k = [], k + 1
        else:
            cur.append(t)
    chunks.append({"text": cur})
    return merged(chunks), k


def length(chunks):
    return sum(len(c["text"]) if "text" in c else c["tokens"] for c in chunks)


def ends(template: str, kwargs: dict):
    s = "\u2063R\u2063"
    probe = [{"role": "user", "content": "x"}]
    p = render(template, probe, True, kwargs)
    f = render(template, probe + [{"role": "assistant", "content": s}], False, kwargs)
    at = f.index(s)
    lead = f[len(p):at] if f.startswith(p) else ""
    end = f[at + len(s):]
    tok = tokenizer()
    end_ids = tok.encode(end, add_special_tokens=False).ids
    specials = set(SPECIALS)
    stop = None
    for t in end_ids:
        if tok.id_to_token(t) in specials:
            stop = t
    if stop is None:
        stop = token_id("<|endoftext|>")
    stop_text = tok.id_to_token(stop)
    head = end[:end.index(stop_text)] if stop_text in end else ""
    return {"lead": lead, "end": end, "stop_token": stop, "end_head": head}


def tokens(template, messages, reply, counts, placeholder_text, effort=0.0):
    kw = kwargs_for(template, effort)
    e = ends(template, kw)
    ph = token_id(placeholder_text)
    prompt_ids = encode(render(template, messages, True, kw))
    out = {"kwargs": kw, **e}
    prompt, used = chunked(prompt_ids, ph, counts)
    if prompt is None or used != len(counts):
        return {"refuses": "training-pictures"}
    out["prompt"] = prompt
    if reply is not None:
        full_ids = encode(render(template, messages + [{"role": "assistant", "content": reply}], False, kw))
        if full_ids[:len(prompt_ids)] != prompt_ids:
            full_ids = prompt_ids + encode(reply + e["end"])
            out["prefix_fallback"] = True
        full, used = chunked(full_ids, ph, counts)
        if full is None or used != len(counts):
            return {"refuses": "training-pictures"}
        out["full"] = full
        out["reply_start"] = length(prompt)
    return out


def read_reply(generated, lead, end_head, stop):
    ids = list(generated)
    if ids and ids[-1] == stop:
        ids = ids[:-1]
    text = decode(ids)
    if lead and lead in text:
        text = text[text.rindex(lead) + len(lead):]
    if end_head and end_head in text:
        text = text[:text.index(end_head)]
    return text


def flat(chunks):
    """Positions: token ids, None where a picture is."""
    out = []
    for c in chunks:
        out += c["text"] if "text" in c else [None] * c["tokens"]
    return out


def shift(chunks):
    """(model_input, targets): the chunks without their last token; the tokens from the second on (0 at pictures)."""
    chunks = [dict(c) for c in chunks]
    while chunks and "picture" in chunks[-1]:
        chunks.pop()
    last = chunks[-1]
    inp = chunks[:-1] + ([{"text": last["text"][:-1]}] if len(last["text"]) > 1 else [])
    targets = [0 if t is None else t for t in flat(chunks)[1:]]
    return inp, targets


def sft_datum(chunks, reply_start, weight=1.0):
    inp, targets = shift(chunks)
    pos = flat(chunks)
    w = [weight if (i + 1) >= reply_start and pos[i + 1] is not None else 0.0 for i in range(len(targets))]
    return {"model_input": inp, "loss_fn_inputs": {"target_tokens": targets, "weights": w}}


def rl_datum(prompt, sampled, logprobs, advantage):
    chunks = merged([dict(c) for c in prompt] + [{"text": list(sampled)}])
    inp, targets = shift(chunks)
    start = length(prompt)
    lp = [logprobs[i + 1 - start] if i + 1 >= start else 0.0 for i in range(len(targets))]
    adv = [advantage if i + 1 >= start else 0.0 for i in range(len(targets))]
    return {"model_input": inp, "loss_fn_inputs": {"target_tokens": targets, "logprobs": lp, "advantages": adv}}


def advantages(rewards, normalize):
    if max(rewards) - min(rewards) < 1e-9:
        return None
    mu = sum(rewards) / len(rewards)
    a = [r - mu for r in rewards]
    if normalize:
        sd = math.sqrt(sum((r - mu) ** 2 for r in rewards) / len(rewards))
        a = [x / (sd + 1e-6) for x in a]
    return a


def kl_adjusted(adv_tokens, deltas, beta):
    """adv_tokens, deltas: per answer, per sampled token. The mean δ over every token of the step."""
    all_d = [d for row in deltas for d in row]
    mean = sum(all_d) / len(all_d)
    return [[a + beta * (mean - d) for a, d in zip(arow, drow)] for arow, drow in zip(adv_tokens, deltas)]


def energy(answers_ok, d_ref, d_pair):
    """answers_ok: readable flags; d_ref[i][k]: d(answer i, reference k); d_pair[i][j]: d(answer i, answer j)."""
    idx = [i for i, ok in enumerate(answers_ok) if ok]
    rewards = [-1.0] * len(answers_ok)
    if not idx:
        return {"rewards": rewards, "score": 1.0}
    a = {i: sum(d_ref[i]) / len(d_ref[i]) for i in idx}
    if len(idx) == 1:
        rewards[idx[0]] = -a[idx[0]]
        return {"rewards": rewards, "score": a[idx[0]]}
    for i in idx:
        rewards[i] = -a[i] + sum(d_pair[i][j] for j in idx if j != i) / (len(idx) - 1)
    pairs = [d_pair[i][j] for i in idx for j in idx if i != j]
    score = sum(a.values()) / len(idx) - 0.5 * sum(pairs) / len(pairs)
    return {"rewards": rewards, "score": score}


def sigmoid(x):
    return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))


def dpo(pairs, beta):
    """pairs: [{pi_c, rho_c, pi_r, rho_r}] → per pair: z, loss, chosen weight, rejected weight."""
    out = []
    for p in pairs:
        z = beta * ((p["pi_c"] - p["rho_c"]) - (p["pi_r"] - p["rho_r"]))
        w = beta * sigmoid(-z) / len(pairs)
        loss = math.log1p(math.exp(-z)) if z > -30 else -z
        out.append({"z": z, "loss": loss, "chosen_weight": w, "rejected_weight": -w})
    return out


def softmax(xs):
    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    s = sum(e)
    return [x / s for x in e]


def decision_weights(logprobs, answer=None, target=None):
    p = softmax(logprobs)
    if answer is not None:
        return {"p": p, "loss": -math.log(p[answer]), "weights": [(1.0 if k == answer else 0.0) - p[k]
                                                                  for k in range(len(p))]}
    s = sum(pj * (pj - tj) for pj, tj in zip(p, target))
    return {"p": p, "loss": sum((pk - tk) ** 2 for pk, tk in zip(p, target)),
            "weights": [-2 * pk * ((pk - tk) - s) for pk, tk in zip(p, target)]}


def choose(checks):
    """checks: [{"step", "scores": {temperature: mean}}] → the chosen (step, temperature)."""
    best = None
    for c in checks:
        for t in sorted(c["scores"], key=float):
            v = c["scores"][t]
            if best is None or v > best[0]:
                best = (v, c["step"], float(t))
    return {"step": best[1], "temperature": best[2], "score": best[0]}


# ------------------------------------------------------------------ the cases


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


SCREEN = [{"role": "system", "content": "Function: scan\n\nWhere the eyes stop."},
          {"role": "user", "content": [{"type": "text", "text": "<screen>\n"}, {"type": "image"},
                                       {"type": "text", "text": "\n</screen>\n<target>\nthe button\n</target>\n"}]}]
TEXT = [{"role": "system", "content": "Function: capital"},
        {"role": "user", "content": "<country>\nPeru — Où?\n</country>\n"}]
TWO = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "then"}, {"type": "image"}]}]
REPLY = "<why>\nthe button\n</why>\n<result>\n[[1, 2]]\n</result>"


def cases() -> dict:
    out = {}

    def tok_case(name, description, template, messages, reply, sizes, placeholder, rule, effort=0.0):
        counts = [grid_tokens(w, h, 40) if rule == "grid" else qwen_tokens(w, h, 16, 2, 1024, 16384)
                  for w, h in sizes]
        out[name] = {"description": description, "tokenizer": "toy-tokenizer.json", "template": template,
                     "effort": effort,
                     "picture": ({"rule": "grid", "patch": 40, "placeholder": placeholder} if rule == "grid" else
                                 {"rule": "qwen", "patch": 16, "merge": 2, "min_pixels": 1024, "max_pixels": 16384,
                                  "placeholder": placeholder}),
                     "messages": messages, "reply": reply, "pictures": [b64(png(w, h)) for w, h in sizes],
                     "expect": tokens(TEMPLATES[template], messages, reply, counts, placeholder, effort)}

    tok_case("tokens-01-text-qwen-like", "A text conversation under a Qwen-shaped template: thinking switched off by "
             "the template's own keyword, the prompt a prefix of the full conversation, the reply starting where "
             "the prompt ends; the stop token is <|im_end|> and nothing comes before a reply.",
             "qwen-like.jinja", TEXT, "<result>\nLima\n</result>", [], "<|image_pad|>", "qwen")
    tok_case("tokens-02-picture-inkling-like", "A picture between text parts under an Inkling-shaped template: the "
             "placeholder token becomes the picture's chunk of ceil(w/40)·ceil(h/40) positions; the effort is "
             "written as 0; the model writes <|content_text|> before its reply and stops on the turn's end.",
             "inkling-like.jinja", SCREEN, REPLY, [(81, 40)], "<|pic|>", "grid")
    tok_case("tokens-03-two-pictures-qwen-rule", "Two pictures, numbered in order, each sized by the qwen rule.",
             "qwen-like.jinja", TWO, "ok", [(100, 40), (64, 64)], "<|image_pad|>", "qwen")
    tok_case("tokens-04-effort-written", "An effort other than 0 is written as the template writes it.",
             "inkling-like.jinja", TEXT, "<result>\nLima\n</result>", [], "<|pic|>", "grid", effort=0.2)
    tok_case("tokens-05-not-a-prefix", "A template whose generation prompt is not where the full conversation "
             "begins: the full tokens are the prompt's, then the reply and the end.",
             "not-a-prefix.jinja", TEXT, "<result>\nLima\n</result>", [], "<|image_pad|>", "qwen")
    tok_case("tokens-06-too-few-pictures", "A conversation with more pictures than placeholders is refused.",
             "qwen-like.jinja", TEXT, "x", [(40, 40)], "<|image_pad|>", "qwen")

    sizes = [(png(81, 40), "png"), (png(1, 1), "png"), (jpeg_header(1280, 800), "jpeg"),
             (jpeg_header(390, 844, 0xC2), "jpeg"), (b"GIF89a\x10\x00\x10\x00", None), (b"\x89PNG", None)]
    out["pictures-01-sizes"] = {
        "description": "A picture's width, height and format from its bytes (PNG IHDR; a JPEG's first frame "
                       "header, progressive ones too); anything else is not a picture a service takes (null).",
        "pictures": [{"data": b64(d), "expect": (dict(zip(("width", "height", "format"), picture_size(d)))
                                                  if picture_size(d) else None)} for d, _f in sizes]}
    grid = [(1, 1), (39, 39), (40, 40), (41, 41), (80, 80), (81, 40), (390, 844), (1280, 800), (1281, 800),
            (4000, 3000), (8000, 8000)]
    out["pictures-02-grid"] = {"description": "The grid rule (Inkling, 40 px): ceil(w/40)·ceil(h/40), no resizing.",
                               "patch": 40, "sizes": [{"width": w, "height": h, "expect": grid_tokens(w, h, 40)}
                                                      for w, h in grid]}
    qsizes = [(1040, 528), (1072, 560), (1056, 544), (1280, 800), (390, 844), (4000, 3000), (20000, 300),
              (48, 48), (10, 10), (3000, 14), (5000, 20)]
    out["pictures-03-qwen"] = {
        "description": "The qwen rule with Qwen3.5's numbers (patch 16, merge 2, 65,536 to 16,777,216 pixels): "
                       "sizes rounded to multiples of 32 half to even (1040 × 528 is 32.5 × 16.5 blocks: 32 × 16, "
                       "where rounding half up would give 33 × 17), scaled into the pixel range (small pictures up "
                       "to 65,536 pixels); refused (null) past an aspect ratio of 200.",
        "patch": 16, "merge": 2, "min_pixels": 65536, "max_pixels": 16777216,
        "sizes": [{"width": w, "height": h, "expect": qwen_tokens(w, h, 16, 2, 65536, 16777216)} for w, h in qsizes]}

    ink = ends(INKLING_LIKE, {"reasoning_effort": 0.0})
    qw = ends(QWEN_LIKE, {"enable_thinking": False})
    gens = []
    for template_name, e, ids in [
        ("inkling-like.jinja", ink, encode("<|content_text|><result>\nLima\n</result><|end_message|>") + [ink["stop_token"]]),
        ("inkling-like.jinja", ink, encode("<|content_thinking|>Peru's capital.<|end_message|><|message_model|>"
                                           "<|content_text|><result>\nLima\n</result><|end_message|>") + [ink["stop_token"]]),
        ("inkling-like.jinja", ink, encode("<|content_text|><result>\nLi")),
        ("qwen-like.jinja", qw, encode("<result>\nLima\n</result>") + [qw["stop_token"]]),
    ]:
        gens.append({"template": template_name, "generated": ids,
                     "expect": read_reply(ids, e["lead"], e["end_head"], e["stop_token"])})
    out["reply-01-reading"] = {
        "description": "Reading what a student generated: the stop token left out, special tokens kept; after the "
                       "lead's last occurrence (so a thinking part before it is dropped); cut at the end's head. A "
                       "reply cut by the length limit is read as far as it goes.",
        "tokenizer": "toy-tokenizer.json", "effort": 0.0, "generated": gens}

    t = tokens(INKLING_LIKE, SCREEN, REPLY, [grid_tokens(81, 40, 40)], "<|pic|>")
    text_case = tokens(QWEN_LIKE, TEXT, "<result>\nLima\n</result>", [], "<|image_pad|>")
    out["datums-01-sft"] = {
        "description": "A supervised example: the input is the conversation without its last token, the targets "
                       "the tokens from the second on, the weights the example's weight on the reply only.",
        "chunks": text_case["full"], "reply_start": text_case["reply_start"], "weight": 2.0,
        "expect": sft_datum(text_case["full"], text_case["reply_start"], 2.0)}
    out["datums-02-sft-picture"] = {
        "description": "A picture takes its positions in the input; its targets are 0 and their weights 0.",
        "chunks": t["full"], "reply_start": t["reply_start"], "weight": 1.0,
        "expect": sft_datum(t["full"], t["reply_start"], 1.0)}
    sampled = encode("<|content_text|>[[1, 2]]<|end_message|>") + [ink["stop_token"]]
    lps = [-(0.1 * (k + 1)) for k in range(len(sampled))]
    out["datums-03-rl"] = {
        "description": "A sampled answer: the prompt's chunks then the sampled tokens; the sampler's "
                       "log-probabilities and the answer's advantage on the sampled positions only; no weights.",
        "prompt": t["prompt"], "sampled": sampled, "logprobs": lps, "advantage": 0.75,
        "expect": rl_datum(t["prompt"], sampled, lps, 0.75)}

    out["advantages-01-groups"] = {
        "description": "Advantages: each reward minus its group's mean (divided by the group's population standard "
                       "deviation + 1e-6 with normalize); a group whose rewards are all equal is left out (null).",
        "groups": [{"rewards": r, "normalize": n, "expect": advantages(r, n)}
                   for r, n in [([0.2, 0.4, 0.9, 0.5], False), ([0.2, 0.4, 0.9, 0.5], True), ([1, 0, 0, 0], True),
                                ([0.3, 0.3, 0.3], False), ([0.5, 0.5 + 1e-12], False), ([0, 1], False)]]}
    adv = [[0.5, 0.5, 0.5], [-0.5, -0.5]]
    deltas = [[0.1, -0.2, 0.05], [0.3, 0.0]]
    out["advantages-02-kl"] = {
        "description": "KL to the starting weights: each sampled token's advantage plus β·(mean δ − δ), the mean "
                       "over every sampled token of the step.",
        "advantages": adv, "deltas": deltas, "beta": 0.05, "expect": kl_adjusted(adv, deltas, 0.05)}

    d_ref = [[0.2, 0.6], [0.4, 0.4], [0.9, 0.1], [0.5, 0.5]]
    d_pair = [[0, 0.3, 0.8, 0.2], [0.3, 0, 0.5, 0.6], [0.8, 0.5, 0, 0.7], [0.2, 0.6, 0.7, 0]]
    out["energy-01"] = {
        "description": "The energy score of a group against its references: each answer's reward is minus its mean "
                       "distance to the references plus its mean distance to the other answers; an unreadable "
                       "answer gets −1 and is left out of the others' spread; one readable answer, its distance only.",
        "groups": [{"readable": ok, "d_ref": d_ref, "d_pair": d_pair, "expect": energy(ok, d_ref, d_pair)}
                   for ok in ([True] * 4, [True, False, True, True], [False, False, True, False],
                              [False] * 4)]}
    pairs = [{"pi_c": -12.0, "rho_c": -12.5, "pi_r": -9.0, "rho_r": -8.0},
             {"pi_c": -30.0, "rho_c": -30.0, "pi_r": -31.0, "rho_r": -31.0},
             {"pi_c": -1.0, "rho_c": -400.0, "pi_r": -400.0, "rho_r": -1.0}]
    out["weights-01-dpo"] = {
        "description": "DPO as weights: z = β((π_c − ρ_c) − (π_r − ρ_r)); the chosen reply's trained tokens get "
                       "β·σ(−z)/P, the rejected one's −β·σ(−z)/P; the loss is log(1 + e^(−z)).",
        "beta": 0.1, "pairs": pairs, "expect": dpo(pairs, 0.1)}
    out["weights-02-decisions"] = {
        "description": "Decisions as weights over the choices' log-probabilities: a known answer (cross-entropy over "
                       "the choices) or a target distribution (the Brier score).",
        "decisions": [{"logprobs": lp, "answer": a, "target": tg, "expect": decision_weights(lp, a, tg)}
                      for lp, a, tg in [([-1.2, -0.4, -2.0], 1, None), ([-1.2, -0.4, -2.0], None, [0.5, 0.5, 0.0]),
                                        ([-0.1, -3.0], None, [1.0, 0.0]), ([-5.0, -5.0], 0, None)]]}
    checks = [{"step": 0, "scores": {"0.0": 0.15, "0.7": 0.12, "1.0": 0.11}},
              {"step": 40, "scores": {"0.0": 0.08, "0.7": 0.28, "1.0": 0.26}},
              {"step": 80, "scores": {"0.0": 0.09, "0.7": 0.28, "1.0": 0.27}}]
    out["choosing-01"] = {
        "description": "The best check and temperature: the highest held-out score; the earlier check, then the "
                       "lower temperature, on a tie.", "checks": checks, "expect": choose(checks)}
    return out


def write() -> int:
    OUT.mkdir(exist_ok=True)
    for old in OUT.glob("*.json"):
        if old.name != "toy-tokenizer.json":
            old.unlink()
    (OUT / "toy-tokenizer.json").write_text(toy_tokenizer() + "\n")
    for name, text in TEMPLATES.items():
        (OUT / name).write_text(text)            # exactly the template: no newline added after it
    cs = cases()
    for name, case in cs.items():
        (OUT / f"{name}.json").write_text(json.dumps(case, indent=1, ensure_ascii=False) + "\n")
    return len(cs)


if __name__ == "__main__":
    print(write(), "cases")
