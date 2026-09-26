"""Head models: a pretrained text model reads the input, one answer layer per
finite output gives a probability for every allowed answer. No prompt, no
generation: one forward pass per row.

- One output: a standard Hugging Face ``AutoModelForSequenceClassification``
  (servable as is by transformers, vLLM ``--convert classify``, TEI, ONNX).
- Several outputs: one shared backbone, one linear layer per output
  (mean pooling for encoders, last token for decoders).

Training follows what measured best in the banking77 runs (2026-09-24/25):
the whole model trained, AdamW, linear warmup then cosine decay, batch 32,
cross-entropy against the target distribution (soft targets keep a teacher's
uncertainty), bf16 autocast on GPU, and a temperature fitted afterwards on
held-out rows so the probabilities mean what they say.
"""

from __future__ import annotations

import copy
import dataclasses
import math
import os
import random
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .examples import HeadField

_HEAD_PREFIXES = ("classifier", "score", "pre_classifier", "heads", "head.")


def _nixos_triton() -> None:
    """Triton finds the CUDA driver with /sbin/ldconfig, which NixOS does not have;
    it reads TRITON_LIBCUDA_PATH first (ModernBERT/Ettin's GPU kernels, vLLM)."""
    if "TRITON_LIBCUDA_PATH" not in os.environ and not os.path.exists("/sbin/ldconfig") and \
            os.path.exists("/run/opengl-driver/lib/libcuda.so"):
        os.environ["TRITON_LIBCUDA_PATH"] = "/run/opengl-driver/lib"


def _torch():
    _nixos_triton()
    try:
        import torch
        import transformers  # noqa: F401 (checked here; used by the callers)
    except ImportError as err:
        raise ImportError('training and running baked models needs PyTorch and transformers: '
                          'pip install "functai[bake]"') from err
    return torch


# ------------------------------------------------------------------ devices


def choose_device(device: Optional[str], need_gb: float) -> str:
    """The device to use: as asked, else the CUDA GPU with the most free memory
    when it has ``need_gb`` free, else the CPU. Never takes memory other
    programs hold (GPUs shared with running services stay usable)."""
    torch = _torch()
    if device:
        return device
    if torch.cuda.is_available():
        best, free_best = None, 0.0
        for i in range(torch.cuda.device_count()):
            free, _total = torch.cuda.mem_get_info(i)
            if free / 2 ** 30 > free_best:
                best, free_best = i, free / 2 ** 30
        if best is not None and free_best >= need_gb:
            return f"cuda:{best}"
    return "cpu"


def training_memory_gb(n_params: int) -> float:
    """fp32 weights, gradients and AdamW's two moments (16 bytes a parameter), plus
    room for activations at batch 32."""
    return n_params * 16 / 2 ** 30 * 1.3 + 0.6


# ------------------------------------------------------------------ the model


def is_decoder(config: Any) -> bool:
    archs = getattr(config, "architectures", None) or []
    return any(a.endswith("ForCausalLM") for a in archs) or bool(getattr(config, "is_decoder", False))


def build(student: str, fields: Sequence[HeadField], *, local_files_only: bool = False):
    """(model, tokenizer, architecture) for a student and the fields it will answer."""
    torch = _torch()
    from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification, AutoTokenizer
    from transformers.utils import logging as hf_logging
    before = hf_logging.get_verbosity()
    hf_logging.set_verbosity_error()      # "classifier newly initialized" is the point, not news
    hf_logging.disable_progress_bar()
    try:
        return _build(torch, student, fields, local_files_only, AutoConfig, AutoModel,
                      AutoModelForSequenceClassification, AutoTokenizer)
    finally:
        hf_logging.set_verbosity(before)
        hf_logging.enable_progress_bar()


def _build(torch, student, fields, local_files_only, AutoConfig, AutoModel, AutoModelForSequenceClassification,
           AutoTokenizer):
    tokenizer = AutoTokenizer.from_pretrained(student, local_files_only=local_files_only)
    config = AutoConfig.from_pretrained(student, local_files_only=local_files_only)
    decoder = is_decoder(config)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    if decoder:
        tokenizer.padding_side = "right"
    if len(fields) == 1:
        f = fields[0]
        model = AutoModelForSequenceClassification.from_pretrained(
            student, num_labels=len(f.keys), id2label=dict(enumerate(f.keys)),
            label2id={k: i for i, k in enumerate(f.keys)}, local_files_only=local_files_only,
            pad_token_id=tokenizer.pad_token_id, dtype=torch.float32)
        if getattr(model.config, "pad_token_id", None) is None:
            model.config.pad_token_id = tokenizer.pad_token_id
        return SingleHead(model), tokenizer, "hf-sequence-classification"
    backbone = AutoModel.from_pretrained(student, local_files_only=local_files_only, dtype=torch.float32)
    return MultiHead(backbone, [len(f.keys) for f in fields], decoder=decoder), tokenizer, "multi-head"


def _module_base():
    return _torch().nn.Module


class SingleHead(_module_base()):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return [self.model(input_ids=input_ids, attention_mask=attention_mask).logits]

    def save(self, path: str) -> None:
        self.model.save_pretrained(os.path.join(path, "model"), safe_serialization=True)


class MultiHead(_module_base()):
    def __init__(self, backbone, sizes: Sequence[int], *, decoder: bool, dropout: float = 0.1):
        torch = _torch()
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        hidden = backbone.config.hidden_size
        self.dropout = torch.nn.Dropout(dropout)
        self.heads = torch.nn.ModuleList([torch.nn.Linear(hidden, n) for n in sizes])

    def forward(self, input_ids, attention_mask):
        states = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if self.decoder:           # the last real token has read the whole input
            last = attention_mask.sum(dim=1) - 1
            pooled = states[_torch().arange(states.shape[0], device=states.device), last]
        else:                      # mean over real tokens
            mask = attention_mask.unsqueeze(-1).to(states.dtype)
            pooled = (states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        pooled = self.dropout(pooled)
        return [h(pooled) for h in self.heads]

    def save(self, path: str) -> None:
        from safetensors.torch import save_file
        self.backbone.save_pretrained(os.path.join(path, "model"), safe_serialization=True)
        save_file({k: v.contiguous() for k, v in self.heads.state_dict().items()},
                  os.path.join(path, "heads.safetensors"))


def load(path: str, architecture: str, sizes: Sequence[int], *, decoder: bool, device: str):
    from transformers.utils import logging as hf_logging
    before = hf_logging.get_verbosity()
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
    try:
        return _load(path, architecture, sizes, decoder=decoder, device=device)
    finally:
        hf_logging.set_verbosity(before)
        hf_logging.enable_progress_bar()


def _load(path: str, architecture: str, sizes: Sequence[int], *, decoder: bool, device: str):
    torch = _torch()
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(path, "tokenizer"))
    if architecture == "hf-sequence-classification":
        model = SingleHead(AutoModelForSequenceClassification.from_pretrained(os.path.join(path, "model"),
                                                                               dtype=torch.float32))
    else:
        from safetensors.torch import load_file
        backbone = AutoModel.from_pretrained(os.path.join(path, "model"), dtype=torch.float32)
        model = MultiHead(backbone, sizes, decoder=decoder)
        model.heads.load_state_dict(load_file(os.path.join(path, "heads.safetensors")))
    return model.to(device).eval(), tokenizer


# ------------------------------------------------------------------ data


def encode(tokenizer, texts: Sequence[str], max_length: int) -> Tuple[List[List[int]], int]:
    """Token ids per text (truncated to ``max_length``), and how many were cut."""
    enc = tokenizer(list(texts), truncation=False, add_special_tokens=True)["input_ids"]
    closing = {t for t in (tokenizer.sep_token_id, tokenizer.eos_token_id) if t is not None}
    cut = 0
    out = []
    for ids in enc:
        if len(ids) > max_length:
            cut += 1
            # keep a closing special token ([SEP], </s>): the model was pretrained to see it
            ids = ids[:max_length - 1] + ids[-1:] if ids[-1] in closing else ids[:max_length]
        out.append(ids)
    return out, cut


def choose_max_length(lengths: Sequence[int], limit: int) -> int:
    """Long enough for 99% of the training rows (rounded up to a multiple of 8), at most ``limit``."""
    if not lengths:
        return min(limit, 128)
    s = sorted(lengths)
    p99 = s[min(len(s) - 1, int(math.ceil(0.99 * len(s))) - 1)]
    return int(min(limit, max(16, math.ceil(p99 / 8) * 8)))


def _batch(ids: Sequence[List[int]], pad: int, device: str, pad_left: bool = False):
    torch = _torch()
    width = max(len(x) for x in ids)
    input_ids = torch.full((len(ids), width), pad, dtype=torch.long)
    mask = torch.zeros((len(ids), width), dtype=torch.long)
    for i, x in enumerate(ids):
        if pad_left:
            input_ids[i, width - len(x):] = torch.tensor(x)
            mask[i, width - len(x):] = 1
        else:
            input_ids[i, :len(x)] = torch.tensor(x)
            mask[i, :len(x)] = 1
    return input_ids.to(device), mask.to(device)


# ------------------------------------------------------------------ training


@dataclasses.dataclass
class TrainConfig:
    epochs: int
    lr: float
    head_lr: float
    batch_size: int = 32
    warmup: float = 0.05
    patience: int = 2
    seed: int = 0
    weight_decay: float = 0.01


def default_config(n_params: int, n_rows: int, decoder: bool, *, epochs: Optional[int] = None,
                   lr: Optional[float] = None, batch_size: int = 32, seed: int = 0) -> TrainConfig:
    """The settings that measured best on banking77 for each size of model, and a
    pass count that keeps the number of steps near that of 6 passes over 9.5k rows
    when there are fewer rows (at most 60 passes). Early stopping ends sooner."""
    if lr is None:
        lr = 5e-4 if n_params < 10e6 else 1e-4 if n_params < 100e6 else 5e-5 if n_params < 400e6 else 3e-5
    if epochs is None:
        epochs = min(60, max(6, round(6 * 9493 / max(1, n_rows))))
    head_lr = lr * 10 if decoder else lr
    return TrainConfig(epochs=epochs, lr=lr, head_lr=head_lr, batch_size=batch_size, seed=seed)


def _seed(seed: int) -> None:
    torch = _torch()
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loss(logits: List[Any], targets: List[Any]):
    torch = _torch()
    total = 0
    for z, q in zip(logits, targets):
        total = total + torch.nn.functional.cross_entropy(z.float(), q)   # soft or one-hot targets
    return total


def train(model, tokenizer, train_ids: Sequence[List[int]], train_targets: List[List[List[float]]],
          val_ids: Sequence[List[int]], val_targets: List[List[List[float]]], cfg: TrainConfig, device: str,
          log: Callable[[str], None] = lambda s: None) -> Dict[str, Any]:
    """Train in place; keep the weights of the best validation epoch. Returns what happened."""
    torch = _torch()
    _seed(cfg.seed)
    model.to(device).train()
    head, body = [], []
    for n, p in model.named_parameters():
        name = n.split("model.", 1)[-1] if n.startswith("model.") else n
        (head if name.startswith(_HEAD_PREFIXES) else body).append(p)
    optim = torch.optim.AdamW([{"params": body, "lr": cfg.lr}, {"params": head, "lr": cfg.head_lr}],
                              weight_decay=cfg.weight_decay)
    steps_per_epoch = math.ceil(len(train_ids) / cfg.batch_size)
    total = steps_per_epoch * cfg.epochs
    warm = max(1, int(cfg.warmup * total))

    def schedule(step: int) -> float:
        if step < warm:
            return (step + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))

    sched = torch.optim.lr_scheduler.LambdaLR(optim, schedule)
    use_amp = device.startswith("cuda") and torch.cuda.is_bf16_supported()
    pad = tokenizer.pad_token_id
    targets_t = [torch.tensor(f, dtype=torch.float32) for f in train_targets]
    rng = random.Random(cfg.seed)
    best = (float("inf"), None, -1)
    history = []
    bad = 0
    t0 = time.time()
    step = 0
    for epoch in range(cfg.epochs):
        order = list(range(len(train_ids)))
        rng.shuffle(order)
        model.train()
        running = 0.0
        for b in range(0, len(order), cfg.batch_size):
            idx = order[b:b + cfg.batch_size]
            ids, mask = _batch([train_ids[i] for i in idx], pad, device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = model(ids, mask)
            loss = _loss(logits, [t[idx].to(device) for t in targets_t])
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            step += 1
            running += loss.item()
        val_loss = evaluate_loss(model, tokenizer, val_ids, val_targets, device) if val_ids else running
        history.append({"epoch": epoch + 1, "train_loss": running / steps_per_epoch, "val_loss": val_loss,
                        "seconds": round(time.time() - t0, 1)})
        log(f"epoch {epoch + 1}/{cfg.epochs}: train loss {running / steps_per_epoch:.4f}, "
            f"validation loss {val_loss:.4f}")
        if val_loss < best[0] - 1e-4:
            best = (val_loss, copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()}), epoch + 1)
            bad = 0
        else:
            bad += 1
            if bad >= cfg.patience:
                log(f"stopped early: no better validation loss for {cfg.patience} passes")
                break
    if best[1] is not None:
        model.load_state_dict(best[1])
    model.eval()
    return {"history": history, "best_epoch": best[2], "best_val_loss": best[0], "steps": step,
            "seconds": round(time.time() - t0, 1), "mixed_precision": "bf16" if use_amp else "fp32"}


def logits(model, tokenizer, ids: Sequence[List[int]], device: str, batch_size: int = 256) -> List[List[List[float]]]:
    """Raw scores per field and row (fp32), batched by length for speed."""
    torch = _torch()
    order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
    out: List[Optional[List[List[float]]]] = [None] * len(ids)
    pad = tokenizer.pad_token_id
    model.eval()
    with torch.inference_mode():
        for b in range(0, len(order), batch_size):
            idx = order[b:b + batch_size]
            tid, mask = _batch([ids[i] for i in idx], pad, device)
            zs = [z.float().cpu() for z in model(tid, mask)]
            for j, i in enumerate(idx):
                out[i] = [z[j].tolist() for z in zs]
    n_fields = len(out[0]) if out else 0
    return [[row[f] for row in out] for f in range(n_fields)]   # type: ignore[index]


def evaluate_loss(model, tokenizer, ids, targets, device) -> float:
    torch = _torch()
    zs = logits(model, tokenizer, ids, device)
    total = 0.0
    for z, q in zip(zs, targets):
        total += float(torch.nn.functional.cross_entropy(torch.tensor(z), torch.tensor(q)))
    return total


def fit_temperatures(field_logits: List[List[List[float]]], field_targets: List[List[List[float]]]) -> List[float]:
    """One temperature per field, minimizing cross-entropy on held-out rows (LBFGS on log T)."""
    torch = _torch()
    temps = []
    for z, q in zip(field_logits, field_targets):
        if not z:
            temps.append(1.0)
            continue
        zt, qt = torch.tensor(z), torch.tensor(q)
        log_t = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(zt / log_t.exp(), qt)
            loss.backward()
            return loss
        opt.step(closure)
        temps.append(float(log_t.detach().exp().clamp(0.05, 20.0)))
    return temps


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())
