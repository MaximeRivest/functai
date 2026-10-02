"""The machine bake might train on: its accelerators, which fast kernels it
has, and estimates of memory and time for a training run.

Estimates are deliberately simple (and labeled "≈" wherever shown). They
decide whether a plan is worth starting here; the run itself measures, and
its first step is its most memory-hungry one, so a wrong estimate fails in
seconds, not hours (see ``trainers.here``).
"""

from __future__ import annotations

import dataclasses
import importlib.util
import os
import platform
from functools import lru_cache
from typing import Any, Dict, List, Optional

from .students import StudentInfo

GB = 2 ** 30


def nixos_triton() -> None:
    """Triton finds the CUDA driver with /sbin/ldconfig, which NixOS does not
    have; it reads TRITON_LIBCUDA_PATH first."""
    if "TRITON_LIBCUDA_PATH" not in os.environ and not os.path.exists("/sbin/ldconfig") and \
            os.path.exists("/run/opengl-driver/lib/libcuda.so"):
        os.environ["TRITON_LIBCUDA_PATH"] = "/run/opengl-driver/lib"


# Dense tensor throughput for training (bf16 or fp16 with fp32 accumulation),
# in TFLOPS, by the words in the device name. GeForce cards accumulate in fp32
# at half their advertised fp16 rate. Unknown devices fall back by vendor.
_PEAK_TFLOPS = [
    ("B200", 2250), ("H200", 989), ("H100 PCIe", 756), ("H100", 989), ("GH200", 989), ("H800", 989),
    ("A100", 312), ("A800", 312), ("L40S", 362), ("L40", 181), ("L4", 121), ("A10G", 70), ("A10", 125),
    ("A30", 165), ("A40", 150), ("A6000", 155), ("RTX 6000 Ada", 364), ("RTX 6000", 130), ("A5000", 111),
    ("A4000", 77), ("V100", 125), ("T4", 65), ("P100", 19),
    ("5090", 210), ("5080", 113), ("5070", 62), ("4090", 165), ("4080", 97), ("4070", 58), ("4060", 30),
    ("3090", 71), ("3080", 60), ("3070", 41), ("3060", 25), ("2080", 40), ("2070", 30), ("2060", 26),
]
_GEFORCE_HALF_RATE = ("RTX 50", "RTX 40", "RTX 30", "RTX 20", "GeForce")


def _peak_tflops(name: str) -> float:
    for key, tf in _PEAK_TFLOPS:
        if key in name:
            if any(k in name for k in _GEFORCE_HALF_RATE) and "RTX 6000" not in name and "A6000" not in name:
                return tf / 2
            return tf
    return 30.0


@dataclasses.dataclass(frozen=True)
class Device:
    """One accelerator (or the CPU)."""
    kind: str                     # "cuda", "mps", "cpu"
    index: Optional[int]
    name: str
    total_gb: float
    free_gb: float
    bf16: bool
    capability: Optional[tuple]   # CUDA compute capability
    tflops: float

    @property
    def id(self) -> str:
        return f"{self.kind}:{self.index}" if self.index is not None else self.kind

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def devices() -> List[Device]:
    """Accelerators here, freest first; the CPU last. Memory other programs hold
    is not counted as free (a GPU shared with a running service stays usable)."""
    out: List[Device] = []
    try:
        import torch
    except ImportError:
        return [Device("cpu", None, platform.processor() or "CPU", 0.0, 0.0, False, None, 0.5)]
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            try:
                free, total = torch.cuda.mem_get_info(i)
            except Exception:  # noqa: BLE001 — a device in a bad state is skipped
                continue
            p = torch.cuda.get_device_properties(i)
            cap = (p.major, p.minor)
            out.append(Device("cuda", i, p.name, total / GB, free / GB, cap >= (8, 0), cap, _peak_tflops(p.name)))
        out.sort(key=lambda d: -d.free_gb)
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        try:
            total = torch.mps.recommended_max_memory() / GB
        except Exception:  # noqa: BLE001
            total = 8.0
        out.append(Device("mps", None, "Apple silicon", total, total * 0.8, True, None, 10.0))
    out.append(Device("cpu", None, platform.processor() or "CPU", 0.0, 0.0, False, None, 0.5))
    return out


# ------------------------------------------------------------------ kernels


def _importable(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


@lru_cache(maxsize=1)
def kernels() -> Dict[str, bool]:
    """Which optional kernels are importable here (not loaded)."""
    hub = False
    if _importable("kernels"):
        try:
            from transformers.utils.import_utils import is_kernels_available
            hub = bool(is_kernels_available())
        except Exception:  # noqa: BLE001 — an older transformers without hub kernels
            hub = False
    return {
        "flash_attn": _importable("flash_attn") or hub,     # a flash-attention kernel for packed examples
        "flash_attn_hub": hub,
        "flash_attn_package": _importable("flash_attn"),
        "flash_linear_attention": _importable("fla"),
        "causal_conv1d": _importable("causal_conv1d"),
        "liger": _importable("liger_kernel"),
        "bitsandbytes": _importable("bitsandbytes"),
    }


def attention_implementation(device: Device) -> Optional[str]:
    """The flash-attention implementation to train packed examples with, or None
    when there is none here (the run then pads instead of packing)."""
    k = kernels()
    if device.kind != "cuda" or not device.capability or device.capability < (8, 0):
        return None
    if k["flash_attn_package"]:
        return "flash_attention_2"
    if k["flash_attn_hub"]:
        return "kernels-community/flash-attn2"
    return None


# ------------------------------------------------------------------ estimates


def weight_bytes(student: StudentInfo, quantize: Optional[str]) -> float:
    """The base weights in memory: 16-bit, or 4-bit (NF4 with double
    quantization, about 0.53 bytes a weight) with the embeddings kept 16-bit."""
    if quantize == "4bit":
        emb = student.embedding_parameters
        return emb * 2 + (student.parameters - emb) * 0.53
    return student.parameters * 2


def memory_per_token(student: StudentInfo) -> float:
    """Training memory a token of a micro-batch takes, with gradient
    checkpointing: each block's input kept for the backward pass, one block's
    activations recomputed at a time, and the answer's vocabulary scores in
    chunks (TRL's ``chunked_nll``). A margin of 1.5 for what the formula misses."""
    d, f, L = student.hidden, student.intermediate, student.layers
    kept = L * d * 2
    one_block = (16 * d + 6 * f + 4 * student.heads * student.head_dim) * 2
    return (kept + one_block) * 1.5


def fixed_training_bytes(student: StudentInfo, *, quantize: Optional[str], lora_rank: Optional[int]) -> float:
    """Memory that does not grow with the batch: weights, the trained
    parameters with their gradients and AdamW state, and the framework's own."""
    w = weight_bytes(student, quantize)
    if lora_rank:
        trained = student.lora_parameters(lora_rank) * 16          # fp32 weight, grad, two moments
    else:
        trained = student.parameters * 14                            # grad (2) + fp32 master and moments (12)
    return w + trained + 1.2 * GB


def micro_batch_tokens(student: StudentInfo, device: Device, *, quantize: Optional[str], lora_rank: Optional[int],
                       longest: int, cap: int = 32768) -> int:
    """The most tokens one micro-batch can hold on ``device`` (0: not even the
    longest example fits). At least the longest example, at most ``cap``."""
    free = device.free_gb * GB * 0.92
    room = free - fixed_training_bytes(student, quantize=quantize, lora_rank=lora_rank)
    if room <= 0:
        return 0
    tokens = int(room / memory_per_token(student))
    if tokens < longest:
        return 0
    return max(longest, min(cap, tokens))


def flops_per_token(student: StudentInfo, *, lora: bool, context: float) -> float:
    """Training FLOPs a token costs: forward (2N), backward through the
    activations (2N; and 2N more for weight gradients without LoRA), the forward
    again for gradient checkpointing (2N), and attention over ``context`` tokens
    on the attention layers."""
    n = student.parameters
    dense = (6 if lora else 8) * n
    attn = 4 * 4 * student.full_attention_layers * context * student.heads * student.head_dim
    return dense + attn


def training_seconds(student: StudentInfo, tokens: float, *, devices_: List[Device], lora: bool, context: float,
                     padding: float = 1.0, efficiency: float = 0.3) -> float:
    """Seconds to train on ``tokens`` across ``devices_`` (data parallel), at
    ``efficiency`` of their peak (0.3: measured LoRA runs on consumer and
    datacenter GPUs land between 0.25 and 0.45). ``padding``: tokens computed
    per real token."""
    flops = flops_per_token(student, lora=lora, context=context) * tokens * padding
    peak = sum(d.tflops for d in devices_) * 1e12
    scale = 0.9 if len(devices_) > 1 else 1.0                   # gradient exchange between GPUs
    return flops / (peak * efficiency * scale)


__all__ = ["Device", "devices", "kernels", "attention_implementation", "micro_batch_tokens", "training_seconds",
           "weight_bytes", "nixos_triton"]
