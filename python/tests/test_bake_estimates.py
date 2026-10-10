"""What bake estimates before a run (memory, time) and what runs measure to
correct it. Offline, without PyTorch: shapes are written here, as a model's
config files give them."""

import types

import pytest

from functai.bake import hardware
from functai.bake.hardware import GB, Device
from functai.bake.students import StudentInfo

# Qwen3.5-4B as its config.json says (24 of its 32 layers are linear attention)
QWEN35_4B = StudentInfo(
    name="Qwen/Qwen3.5-4B", parameters=4_659_868_928, hidden=2560, layers=32, intermediate=9216, heads=16,
    kv_heads=4, head_dim=256, vocab=248_320, context=262_144,
    layer_types=("linear_attention",) * 3 + ("full_attention",) + ("linear_attention",) * 3 + ("full_attention",)
    + ("linear_attention",) * 3 + ("full_attention",) + ("linear_attention",) * 3 + ("full_attention",)
    + ("linear_attention",) * 3 + ("full_attention",) + ("linear_attention",) * 3 + ("full_attention",)
    + ("linear_attention",) * 3 + ("full_attention",) + ("linear_attention",) * 3 + ("full_attention",),
    tied_embeddings=True, chat_template=True, family="qwen")
MiB = 2 ** 20


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))      # no measures from this machine's runs
    return tmp_path


def gpu(name, total=95.0, free=94.0):
    return Device("cuda", 0, name, total, free, True, (12, 0), hardware._peak_tflops(name))


def test_the_speed_table_knows_the_cards_people_rent():
    # an RTX PRO 6000 matched nothing and was assumed at 30 TFLOPS: 3.7 days planned, 9.3 hours taken
    assert hardware._peak_tflops("NVIDIA RTX PRO 6000 Blackwell Server Edition") == 500
    assert hardware._peak_tflops("NVIDIA B300 SXM6 AC") == 2250
    assert hardware._peak_tflops("NVIDIA GeForce RTX 3090") == 71 / 2          # GeForce: half rate in fp32
    assert hardware._peak_tflops("NVIDIA RTX A6000") == 155
    assert hardware.known_speed("NVIDIA H100 80GB HBM3") and not hardware.known_speed("Some New Card")


def test_a_hybrid_students_memory_matches_what_a_real_run_needed():
    # 2026-10-09, an RTX PRO 6000 (95 GB): 16-bit weights ran out of memory on the longest example
    # (87,119 tokens); bake then took 4-bit weights, which fit. The plan had said 16-bit fit (58.9 / 95 GB).
    card = gpu("NVIDIA RTX PRO 6000 Blackwell Server Edition", total=94.97, free=94.5)
    longest = 87_119
    assert hardware.micro_batch_tokens(QWEN35_4B, card, quantize=None, lora_rank=64, longest=longest) == 0
    assert hardware.micro_batch_tokens(QWEN35_4B, card, quantize="4bit", lora_rank=64, longest=longest) >= longest
    assert 0.85 <= hardware.memory_per_token(QWEN35_4B) / MiB <= 1.05


def test_a_run_s_measures_replace_the_formula(home):
    formula = hardware.memory_per_token(QWEN35_4B)
    hardware.remember(QWEN35_4B.name, None, bytes_per_token=500_000)
    assert hardware.memory_per_token(QWEN35_4B) == pytest.approx(550_000)    # with a margin for PyTorch's cache
    assert hardware.memory_per_token(QWEN35_4B, "4bit") == formula           # measured for 16-bit only
    card = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
    hardware.remember(QWEN35_4B.name, None, card, tokens_per_second=2000.0, context=14_000)
    secs = hardware.training_seconds(QWEN35_4B, 2000.0 * 3600, devices_=[gpu(card)], lora=True, context=12_000)
    assert secs == pytest.approx(3600)
    # examples of another length train at another speed: the formula again
    far = hardware.training_seconds(QWEN35_4B, 2000.0 * 3600, devices_=[gpu(card)], lora=True, context=1_000)
    assert far != pytest.approx(3600)


def test_four_bit_weights_take_longer():
    d = [gpu("NVIDIA H100 80GB HBM3")]
    plain = hardware.training_seconds(QWEN35_4B, 1e8, devices_=d, lora=True, context=10_000)
    four = hardware.training_seconds(QWEN35_4B, 1e8, devices_=d, lora=True, context=10_000, quantize="4bit")
    assert four == pytest.approx(plain / 0.75)


def test_a_long_reply_has_time_to_finish():
    from functai.bake.runners import Endpoint
    e = Endpoint(types.SimpleNamespace(name="m"), "http://127.0.0.1:1/v1", timeout=600)
    assert e._wait(1000) == 600
    assert e._wait(66_858) > 6000          # was cut at 10 minutes: about 60 tokens a second gets 36,000


def test_serving_on_a_full_gpu_says_so(tmp_path, monkeypatch):
    from functai.bake import BakeError, runners
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "model.safetensors").write_bytes(b"\0" * (3 * MiB))
    baked = types.SimpleNamespace(name="m", path=tmp_path, meta={"weights": {"form": "merged", "path": "model"}})
    monkeypatch.setattr(hardware, "devices", lambda: [Device("cuda", 0, "Card", 0.024, 0.0005, True, (8, 6), 70)])
    with pytest.raises(BakeError, match="GB free"):
        runners.start_vllm(baked)
