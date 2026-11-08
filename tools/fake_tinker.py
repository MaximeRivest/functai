"""A stand-in for Tinker (Thinking Machines' training service), for offline tests.

It speaks Tinker's HTTP protocol as the official Python SDK (tinker 0.39)
speaks it: JSON for most requests, protobuf for ``forward_backward``'s
request and for the results of training and sampling (when asked with
``Accept: application/x-protobuf``), work returned as futures collected with
``retrieve_future`` (HTTP 408 or ``{"type": "try_again"}`` while not ready).
Tests run the official SDK against it, which checks this file against the
real protocol; FunctAI's TypeScript client is then tested against the same
server.

The "model" is a lookup table: the next token's logits depend on a hash of
everything before it (``table[hash(prefix)][token]``, 0 when absent, over the
base model's vocabulary). Untrained, every next token is equally likely, so it
samples noise; training moves the table the way gradients move a network:

- ``cross_entropy``: ``-sum(w * log p(target))`` (one target per position, or
  K targets per position with K weights);
- ``importance_sampling``: ``-sum(p/q * A)``, ``q`` the sampler's
  probabilities, ``A`` the advantages; ``ppo`` clips the ratio as Tinker
  documents (``clip_low_threshold``, ``clip_high_threshold``, default 0.8 and
  1.2); ``cispo``, ``dro``, ``gspo`` and ``stable_reinforce`` are treated as
  ``importance_sampling`` here;
- ``optim_step`` applies the summed gradient with plain SGD at
  ``learning_rate * lr_scale`` (``lr_scale`` 20,000 by default, so that a
  LoRA learning rate of 1e-4 moves a logit by 2 per unit of gradient).

Pictures (``ImageChunk``) take ``expected_tokens`` positions; for a base model
with a ``patch`` size, ``expected_tokens`` must equal
``ceil(width/patch) * ceil(height/patch)`` (read from the PNG or JPEG header),
else the request fails with HTTP 400, as Tinker does.

Run it alone (prints its address; ``TINKER_BASE_URL`` for the SDK):

    python tools/fake_tinker.py [--port 0]

Control (tests): ``POST /_fake/config`` sets options (``try_again``: how many
times each future answers "try again" first; ``billing_blocked``: how many
requests answer 402; ``fail``: ``{path: n}``, the next n requests to a path
answer 400; ``lr_scale``; ``storage_limit_gb``; ``models``: more base
models); ``GET /_fake/state`` returns what was asked of it.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import random
import re
import struct
import sys
import threading
import time
import urllib.parse
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ------------------------------------------------------------------ the catalogue

GB = 1024 ** 3

# base model -> vocabulary, prices ($ per million tokens), context, picture patch, checkpoint sizes
MODELS: Dict[str, Dict[str, Any]] = {
    "thinkingmachines/Inkling-2-Small": {"vocab": 200_064, "prefill": 0.30, "cached": 0.03, "sample": 1.00,
                                         "train": 1.50, "context": 65_536, "patch": 40,
                                         "state_gb": 50.0, "sampler_gb": 17.0},
    "thinkingmachines/Inkling-Small": {"vocab": 200_064, "prefill": 0.58, "cached": 0.116, "sample": 1.44,
                                       "train": 1.73, "context": 65_536, "patch": 40,
                                       "state_gb": 30.0, "sampler_gb": 10.0},
    "Qwen/Qwen3.5-4B": {"vocab": 248_320, "prefill": 0.33, "cached": 0.066, "sample": 1.005, "train": 0.737,
                        "context": 65_536, "patch": None, "state_gb": 0.4, "sampler_gb": 0.13},
    "Qwen/Qwen3-8B": {"vocab": 151_936, "prefill": 0.195, "cached": 0.039, "sample": 0.60, "train": 0.44,
                      "context": 32_768, "patch": None, "state_gb": 0.5, "sampler_gb": 0.17},
}


def picture_size(data: bytes) -> Tuple[int, int]:
    """(width, height) from a PNG's IHDR or a JPEG's first SOF segment."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return struct.unpack(">II", data[16:24])
    if data[:2] == b"\xff\xd8":
        i = 2
        while i + 4 <= len(data):
            if data[i] != 0xFF:
                i += 1
                continue
            marker = data[i + 1]
            if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
                i += 2
                continue
            length = struct.unpack(">H", data[i + 2:i + 4])[0]
            if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                h, w = struct.unpack(">HH", data[i + 5:i + 9])
                return w, h
            i += 2 + length
    raise ValueError("not a PNG or JPEG picture")


# ------------------------------------------------------------------ the table model


class Table:
    """next-token logits by prefix hash: {prefix: {token: logit}}."""

    def __init__(self, vocab: int, logits: Optional[Dict[str, Dict[int, float]]] = None):
        self.vocab = vocab
        self.logits: Dict[str, Dict[int, float]] = logits if logits is not None else {}

    def copy(self) -> "Table":
        return Table(self.vocab, {h: dict(row) for h, row in self.logits.items()})

    def _z(self, row: Dict[int, float], temp: float = 1.0) -> float:
        return (self.vocab - len(row)) + sum(math.exp(v / temp) for v in row.values())

    def logprob(self, h: str, token: int) -> float:
        row = self.logits.get(h, {})
        return row.get(token, 0.0) - math.log(self._z(row))

    def probs(self, h: str) -> Tuple[Dict[int, float], float]:
        """(listed tokens' probabilities, probability of each unlisted token)."""
        row = self.logits.get(h, {})
        z = self._z(row)
        return {t: math.exp(v) / z for t, v in row.items()}, 1.0 / z

    def sample(self, h: str, temp: float, rng: random.Random) -> int:
        row = self.logits.get(h, {})
        if temp <= 0:
            best = max(row.items(), key=lambda kv: (kv[1], -kv[0]), default=None)
            if best is not None and best[1] > 0:
                return best[0]
            t = 0
            while t in row:
                t += 1
            return t
        z = self._z(row, temp)
        x = rng.random() * z
        for t, v in sorted(row.items()):
            x -= math.exp(v / temp)
            if x < 0:
                return t
        while True:                                  # an unlisted token, uniformly
            t = rng.randrange(self.vocab)
            if t not in row:
                return t


def _h(prev: str, item: Any) -> str:
    return hashlib.blake2b(f"{prev}|{item}".encode(), digest_size=10).hexdigest()


def prefix_hashes(chunks: Sequence[Dict[str, Any]], extra: Sequence[int] = ()) -> Tuple[List[Optional[int]], List[str]]:
    """The tokens of a model input (None at picture positions) and, for each
    position i, the hash of everything before it (len + 1 hashes)."""
    tokens: List[Optional[int]] = []
    hashes = [_h("", "start")]
    for c in chunks:
        if c["type"] == "encoded_text":
            for t in c["tokens"]:
                tokens.append(int(t))
                hashes.append(_h(hashes[-1], int(t)))
        elif c["type"] == "image":
            digest = hashlib.sha256(c["data"]).hexdigest()[:16]
            for k in range(int(c["expected_tokens"])):
                tokens.append(None)
                hashes.append(_h(hashes[-1], f"img:{digest}:{k}"))
        else:
            raise ValueError(f"unsupported chunk {c['type']}")
    for t in extra:
        tokens.append(int(t))
        hashes.append(_h(hashes[-1], int(t)))
    return tokens, hashes


# ------------------------------------------------------------------ the service


class Fail(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status, self.detail = status, detail


class FakeTinker:
    def __init__(self, *, api_key: Optional[str] = None, seed: int = 0):
        self.api_key = api_key
        self.lock = threading.RLock()
        self.rng = random.Random(seed)
        self.models = {k: dict(v) for k, v in MODELS.items()}
        self.options: Dict[str, Any] = {"try_again": 0, "billing_blocked": 0, "lr_scale": 20_000.0,
                                        "storage_limit_gb": 100.0, "paid_storage": False, "fail": {}}
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.runs: Dict[str, Dict[str, Any]] = {}            # model_id -> {base, rank, table, grad, seq, meta}
        self.samplers: Dict[str, Dict[str, Any]] = {}        # sampling_session_id -> {table, base}
        self.checkpoints: Dict[str, Dict[str, Any]] = {}     # tinker path -> {table, type, run, ...}
        self.futures: Dict[str, Dict[str, Any]] = {}         # request_id -> {result, proto, polls}
        self.log: List[Dict[str, Any]] = []
        self.usage: List[Dict[str, Any]] = []                # billing events
        self.prompt_cache: set = set()

    # ---- helpers

    def model(self, base: str) -> Dict[str, Any]:
        if base not in self.models:
            raise Fail(400, f"Unsupported base model {base!r}")
        return self.models[base]

    def _future(self, result: Any, kind: str, **extra: Any) -> Dict[str, Any]:
        rid = f"req-{uuid.uuid4().hex[:12]}"
        self.futures[rid] = {"result": result, "kind": kind, "polls": 0}
        return {"request_id": rid, **extra}

    def _check_seq(self, run: Dict[str, Any], seq_id: Optional[int], what: str) -> None:
        if seq_id is None:
            return
        if seq_id <= run["seq"]:
            raise Fail(400, f"{what}: seq_id {seq_id} is not after {run['seq']} (requests must be sent in order)")
        run["seq"] = seq_id

    def _bill(self, kind: str, tokens: int, base: str, session: Optional[str], **extra: Any) -> None:
        m = self.models[base]
        price = {"training": m["train"], "sampling_prefill": m["cached"] if extra.get("cached") else m["prefill"],
                 "sampling_sample": m["sample"]}[kind]
        self.usage.append({"type": kind, "token_count": tokens, "base_model": base, "session_id": session,
                           "estimated_cost_usd": tokens * price / 1e6, "rate": price, **extra})

    def _run(self, model_id: str) -> Dict[str, Any]:
        run = self.runs.get(model_id)
        if run is None:
            raise Fail(404, f"Unknown model_id {model_id!r}")
        return run

    def _storage_gb(self) -> float:
        return sum(c["size_bytes"] for c in self.checkpoints.values()) / GB

    def _save(self, run_id: str, kind: str, name: Optional[str], table: Table, extra: Dict[str, Any]) -> str:
        run = self.runs[run_id]
        m = self.models[run["base"]]
        size = int((m["state_gb"] if kind == "weights" else m["sampler_gb"]) * GB)
        if not self.options["paid_storage"] and self._storage_gb() + size / GB > self.options["storage_limit_gb"]:
            raise Fail(400, "Checkpoint storage limit reached: training is halted. Delete checkpoints or enable "
                            "paid storage")
        name = name or f"{len([c for c in self.checkpoints.values() if c['run'] == run_id]):06d}"
        path = f"tinker://{run_id}/{kind}/{name}"
        if path in self.checkpoints and not extra.get("overwrite"):
            raise Fail(409, f"Checkpoint {path} exists")
        ttl = extra.get("ttl_seconds")
        self.checkpoints[path] = {"table": table.copy(), "type": "training" if kind == "weights" else "sampler",
                                  "run": run_id, "id": f"{kind}/{name}", "base": run["base"], "rank": run["rank"],
                                  "time": time.time(), "size_bytes": size,
                                  "expires_at": time.time() + ttl if ttl else None,
                                  "user_metadata": extra.get("user_metadata"),
                                  "grad_state": dict(run.get("opt") or {})}
        return path

    # ---- requests

    def handle(self, method: str, path: str, query: Dict[str, str], body: Any, headers: Dict[str, str]
               ) -> Tuple[int, Any, str]:
        """(status, payload, content type: "json" or "proto")."""
        with self.lock:
            self.log.append({"method": method, "path": path, "time": time.time(),
                             "body": body if isinstance(body, dict) else None})
            if not path.startswith("/_fake/") and path != "/api/v1/healthz":
                key = headers.get("x-api-key")
                if self.api_key and key != self.api_key:
                    raise Fail(401, "Invalid API key")
                if self.options["billing_blocked"] > 0 and path not in (
                        "/api/v1/client/config", "/api/v1/client/dynamic_config", "/api/v1/telemetry",
                        "/api/v1/get_server_capabilities", "/api/v1/create_session", "/api/v1/session_heartbeat"):
                    self.options["billing_blocked"] -= 1
                    raise Fail(402, "Access is blocked due to billing status. Please add payment")
            fail = self.options.get("fail") or {}
            if fail.get(path, 0) > 0:                       # a test asked this path to fail (a bad request)
                fail[path] -= 1
                raise Fail(400, f"failed on purpose ({path})")
            route = self.route(method, path)
            if route is None:
                raise Fail(404, f"Not found: {method} {path}")
            fn, params = route
            return fn(query=query, body=body, headers=headers, **params)

    def route(self, method: str, path: str):
        table = [
            ("GET", r"/api/v1/healthz", self.healthz),
            ("POST", r"/api/v1/client/config", self.client_config),
            ("POST", r"/api/v1/client/dynamic_config", self.client_dynamic_config),
            ("POST", r"/api/v1/telemetry", self.telemetry),
            ("GET", r"/api/v1/get_server_capabilities", self.capabilities),
            ("POST", r"/api/v1/create_session", self.create_session),
            ("POST", r"/api/v1/session_heartbeat", self.heartbeat),
            ("POST", r"/api/v1/create_model", self.create_model),
            ("POST", r"/api/v1/get_info", self.get_info),
            ("POST", r"/api/v1/weights_info", self.weights_info),
            ("POST", r"/api/v1/load_weights", self.load_weights),
            ("POST", r"/api/v1/forward_backward", self.forward_backward),
            ("POST", r"/api/v1/optim_step", self.optim_step),
            ("POST", r"/api/v1/save_weights", self.save_weights),
            ("POST", r"/api/v1/save_weights_for_sampler", self.save_weights_for_sampler),
            ("POST", r"/api/v1/create_sampling_session", self.create_sampling_session),
            ("POST", r"/api/v1/asample", self.asample),
            ("GET", r"/api/v1/samplers/(?P<sampler_id>[^/]+)", self.get_sampler),
            ("POST", r"/api/v1/retrieve_future", self.retrieve_future),
            ("GET", r"/api/v1/checkpoints", self.list_user_checkpoints),
            ("GET", r"/api/v1/training_runs/(?P<run_id>[^/]+)/checkpoints", self.list_checkpoints),
            ("DELETE", r"/api/v1/training_runs/(?P<run_id>[^/]+)/checkpoints/(?P<cid>.+)", self.delete_checkpoint),
            ("PUT", r"/api/v1/training_runs/(?P<run_id>[^/]+)/checkpoints/(?P<cid>.+)/ttl", self.set_ttl),
            ("GET", r"/api/v1/billing/usage/events", self.billing_events),
            ("GET", r"/api/v1/billing/usage/checkpoints/current", self.storage_usage),
            ("POST", r"/_fake/config", self.fake_config),
            ("GET", r"/_fake/state", self.fake_state),
        ]
        # the TTL route is more specific than delete's: try routes ending in /ttl first
        table.sort(key=lambda r: 0 if r[1].endswith("/ttl") else 1)
        for m, pattern, fn in table:
            if m == method:
                g = re.fullmatch(pattern, path)
                if g:
                    return fn, g.groupdict()
        return None

    # ---- service

    def healthz(self, **_):
        return 200, {"status": "ok"}, "json"

    def client_config(self, **_):
        return 200, {"pjwt_auth_enabled": False, "credential_default_source": "api_key",
                     "proto_compress_fwdbwd": True, "parallel_fwdbwd_chunks": False,
                     "sample_use_retrieve_futures": False, "sample_join_sampling_session": False,
                     "create_model_via_load_weights": False, "use_pyqwest_transport": False}, "json"

    def client_dynamic_config(self, **_):
        return 200, {}, "json"

    def telemetry(self, **_):
        return 200, {"status": "accepted"}, "json"

    def capabilities(self, **_):
        return 200, {"supported_models": [{"model_name": k} for k in self.models]}, "json"

    def create_session(self, body, **_):
        sid = f"session-{uuid.uuid4().hex[:10]}"
        self.sessions[sid] = {"heartbeats": 0, "user_metadata": body.get("user_metadata") or {},
                              "tags": body.get("tags") or [], "created": time.time()}
        return 200, {"type": "create_session", "session_id": sid}, "json"

    def heartbeat(self, body, **_):
        s = self.sessions.get(body.get("session_id"))
        if s is None:
            raise Fail(404, "Invalid session")
        s["heartbeats"] += 1
        return 200, {"type": "session_heartbeat"}, "json"

    def _session(self, body: Dict[str, Any]) -> str:
        sid = body.get("session_id")
        if sid not in self.sessions:
            raise Fail(404, "Invalid session")
        return sid

    def create_model(self, body, **_):
        sid = self._session(body)
        m = self.model(body["base_model"])
        rank = (body.get("lora_config") or {}).get("rank")
        model_id = f"{sid}:train:{int(body.get('model_seq_id', 0))}"      # as Tinker derives it
        self.runs[model_id] = {"base": body["base_model"], "rank": rank, "table": Table(m["vocab"]), "grad": {},
                               "seq": 0, "session": sid, "meta": body.get("user_metadata"), "opt": {"steps": 0}}
        return 200, self._future({"type": "create_model", "model_id": model_id}, "json", model_id=model_id), "json"

    def get_info(self, body, **_):
        run = self._run(body["model_id"])
        return 200, {"type": "get_info", "model_id": body["model_id"], "is_lora": run["rank"] is not None,
                     "lora_rank": run["rank"], "model_name": run["base"],
                     "model_data": {"model_name": run["base"], "tokenizer_id": run["base"]}}, "json"

    def weights_info(self, body, **_):
        ck = self.checkpoints.get(body.get("tinker_path"))
        if ck is None:
            raise Fail(404, "Checkpoint not found")
        return 200, {"base_model": ck["base"], "is_lora": ck["rank"] is not None, "lora_rank": ck["rank"]}, "json"

    def load_weights(self, body, **_):
        ck = self.checkpoints.get(body["path"])
        if ck is None or ck["type"] != "training":
            raise Fail(404, f"Training checkpoint {body['path']!r} not found")
        model_id = body.get("model_id")
        if model_id:
            run = self._run(model_id)
            self._check_seq(run, body.get("seq_id"), "load_weights")
        else:
            sid = self._session(body)
            model_id = f"{sid}:train:{int(body.get('model_seq_id', 0))}"
            self.runs[model_id] = {"base": ck["base"], "rank": ck["rank"], "grad": {}, "seq": 0, "session": sid,
                                   "meta": body.get("user_metadata"), "opt": {"steps": 0}}
            run = self.runs[model_id]
        run["table"] = ck["table"].copy()
        run["opt"] = dict(ck["grad_state"]) if body.get("optimizer") else {"steps": 0}
        run["loaded_from"] = body["path"]
        return 200, self._future({"type": "load_weights", "path": body["path"], "model_id": model_id}, "json"), \
            "json"

    # ---- training

    def _decode_fwdbwd(self, raw: bytes, headers: Dict[str, str]) -> Dict[str, Any]:
        from tinker.proto import tinker_public_pb2 as pb
        if headers.get("content-encoding") == "zstd":
            import zstandard
            raw = zstandard.ZstdDecompressor().decompress(raw, max_output_size=1 << 31)
        msg = pb.ForwardBackwardRequest()
        msg.ParseFromString(raw)
        data = []
        for d in msg.data:
            chunks = []
            for c in d.model_input:
                kind = c.WhichOneof("chunk")
                if kind == "encoded_text":
                    chunks.append({"type": "encoded_text", "tokens": list(struct.unpack(
                        f"<{len(c.encoded_text.tokens) // 4}i", c.encoded_text.tokens))})
                elif kind == "image":
                    chunks.append({"type": "image", "data": bytes(c.image.data), "format": c.image.format,
                                   "expected_tokens": c.image.expected_tokens if c.image.HasField(
                                       "expected_tokens") else None})
                else:
                    raise Fail(400, f"unsupported chunk {kind}")
            inputs = {}
            for name, t in d.loss_fn_inputs.items():
                if t.WhichOneof("encoding") != "dense":
                    raise Fail(400, "sparse tensors are not supported by the fake")
                fmt = {1: "f", 2: "q"}[t.dtype]
                size = {1: 4, 2: 8}[t.dtype]
                vals = list(struct.unpack(f"<{len(t.dense) // size}{fmt}", t.dense))
                shape = list(t.shape) or [len(vals)]
                inputs[name] = {"data": vals, "shape": shape}
            data.append({"chunks": chunks, "inputs": inputs})
        config = {k: v for k, v in msg.loss_fn_config.items()}
        for k, v in msg.loss_fn_config_v2.items():
            config[k] = v.number if v.WhichOneof("value") == "number" else v.text
        return {"model_id": msg.model_id, "seq_id": msg.seq_id, "loss_fn": msg.loss_fn, "data": data,
                "forward_only": msg.forward_only, "config": config}

    def _from_json_fwdbwd(self, body: Dict[str, Any], forward_only: bool) -> Dict[str, Any]:
        inp = body["forward_backward_input"]
        data = []
        for d in inp["data"]:
            chunks = [self._json_chunk(c) for c in d["model_input"]["chunks"]]
            inputs = {}
            for name, t in d["loss_fn_inputs"].items():
                vals = t["data"]
                shape = t.get("shape") or [len(vals)]
                inputs[name] = {"data": [x for row in vals for x in row] if vals and isinstance(vals[0], list)
                                else list(vals), "shape": shape}
            data.append({"chunks": chunks, "inputs": inputs})
        return {"model_id": body["model_id"], "seq_id": body.get("seq_id"), "loss_fn": inp["loss_fn"],
                "data": data, "forward_only": forward_only, "config": inp.get("loss_fn_config") or {}}

    @staticmethod
    def _json_chunk(c: Dict[str, Any]) -> Dict[str, Any]:
        if c.get("type", "encoded_text") == "encoded_text":
            return {"type": "encoded_text", "tokens": list(c["tokens"])}
        if c["type"] == "image":
            return {"type": "image", "data": base64.b64decode(c["data"]), "format": c["format"],
                    "expected_tokens": c.get("expected_tokens")}
        raise Fail(400, f"unsupported chunk {c.get('type')}")

    def _check_pictures(self, base: str, chunks: Sequence[Dict[str, Any]]) -> None:
        patch = self.models[base].get("patch")
        for c in chunks:
            if c["type"] != "image":
                continue
            if c["format"] not in ("png", "jpeg"):
                raise Fail(400, f"Unsupported image format {c['format']!r}")
            if c.get("expected_tokens") is None:
                raise Fail(400, "ImageChunk.expected_tokens is required")
            if patch:
                w, h = picture_size(c["data"])
                want = math.ceil(w / patch) * math.ceil(h / patch)
                if c["expected_tokens"] != want:
                    raise Fail(400, f"Image token count mismatch: expected_tokens={c['expected_tokens']} but the "
                                    f"{w}x{h} image has {want} tokens")

    def forward_backward(self, body, headers, **_):
        raw = body if isinstance(body, (bytes, bytearray)) else None
        if raw is not None:
            req = self._decode_fwdbwd(bytes(raw), headers)
        else:
            req = self._from_json_fwdbwd(body, bool(body.get("forward_only")))
        run = self._run(req["model_id"])
        self._check_seq(run, req["seq_id"], "forward_backward")
        loss_fn = req["loss_fn"]
        if loss_fn not in ("cross_entropy", "importance_sampling", "ppo", "cispo", "dro", "gspo", "stable_reinforce"):
            raise Fail(400, f"Unknown loss function {loss_fn!r}")
        table: Table = run["table"]
        outputs, total, tokens = [], 0.0, 0
        for d in req["data"]:
            self._check_pictures(run["base"], d["chunks"])
            toks, hashes = prefix_hashes(d["chunks"])
            n = len(toks)
            tokens += n
            inp = d["inputs"]
            targets = inp["target_tokens"]
            k = targets["shape"][1] if len(targets["shape"]) == 2 else 1
            if targets["shape"][0] != n:
                raise Fail(400, f"target_tokens has {targets['shape'][0]} positions; the input has {n}")
            if loss_fn == "cross_entropy":
                allowed = {"target_tokens", "weights"}
            else:
                allowed = {"target_tokens", "logprobs", "advantages", "clip_low_threshold", "clip_high_threshold",
                           "sequence_ratio_mask"}
            extra = set(inp) - allowed
            if extra:
                raise Fail(400, f"loss_fn_inputs {sorted(extra)} are not accepted by {loss_fn}")
            lp_out = []
            for i in range(n):
                ctx = hashes[i + 1]          # position i predicts token i+1: its context is tokens[0..i]
                for j in range(k):
                    y = int(targets["data"][i * k + j])
                    if y < 0:                           # a padded target: no prediction, no loss
                        lp_out.append(float("nan"))
                        continue
                    lp = table.logprob(ctx, y)
                    lp_out.append(lp)
                    if loss_fn == "cross_entropy":
                        w = float(inp["weights"]["data"][i * k + j])
                        if w == 0.0:
                            continue
                        total += -w * lp
                        if not req["forward_only"]:
                            self._grad(run, ctx, y, coef=w)
                    else:
                        if k != 1:
                            raise Fail(400, f"{loss_fn} takes one target per position")
                        a = float(inp["advantages"]["data"][i])
                        if a == 0.0:
                            continue
                        q = float(inp["logprobs"]["data"][i])
                        ratio = math.exp(lp - q)
                        coef = ratio * a
                        if loss_fn == "ppo":
                            lo = float(req["config"].get("clip_low_threshold", 0.8))
                            hi = float(req["config"].get("clip_high_threshold", 1.2))
                            if (a > 0 and ratio > hi) or (a < 0 and ratio < lo):
                                coef = 0.0
                            total += -min(ratio * a, max(min(ratio, hi), lo) * a)
                        else:
                            total += -ratio * a
                        if not req["forward_only"] and coef:
                            self._grad(run, ctx, y, coef=coef)
            outputs.append({"logprobs": lp_out, "shape": [n, k] if k > 1 else [n]})
        if not req["forward_only"]:
            self._bill("training", tokens, run["base"], run["session"])
        else:
            self._bill("training", tokens, run["base"], run["session"])     # forward passes bill as training
        result = {"loss_fn_output_type": "CrossEntropyLossReturn" if loss_fn == "cross_entropy" else
                  "ImportanceSamplingLossReturn", "outputs": outputs, "metrics": {"loss:sum": total}}
        return 200, self._future(result, "fwdbwd"), "json"

    def _grad(self, run: Dict[str, Any], ctx: str, y: int, coef: float) -> None:
        """Accumulate d(-coef * log p(y | ctx))/d logits: coef * (p - onehot(y))."""
        listed, other = run["table"].probs(ctx)
        g = run["grad"].setdefault(ctx, {})
        for t, p in listed.items():
            g[t] = g.get(t, 0.0) + coef * p
        if y in listed:
            g[y] -= coef
        else:                        # the target was not in the table yet: its probability is the unlisted one's
            g[y] = g.get(y, 0.0) + coef * (other - 1.0)

    def optim_step(self, body, **_):
        run = self._run(body["model_id"])
        self._check_seq(run, body.get("seq_id"), "optim_step")
        params = body.get("adam_params") or body.get("optim_params") or {}
        lr = float(params.get("learning_rate", 1e-4)) * float(self.options["lr_scale"])
        table: Table = run["table"]
        for ctx, g in run["grad"].items():
            row = table.logits.setdefault(ctx, {})
            for t, v in g.items():
                row[t] = row.get(t, 0.0) - lr * v
        run["grad"] = {}
        run["opt"]["steps"] = run["opt"].get("steps", 0) + 1
        return 200, self._future({"type": "optim_step", "metrics": {"steps": run["opt"]["steps"]}}, "json"), "json"

    # ---- weights

    def save_weights(self, body, **_):
        run = self._run(body["model_id"])
        self._check_seq(run, body.get("seq_id"), "save_weights")
        path = self._save(body["model_id"], "weights", body.get("path"), run["table"], body)
        return 200, self._future({"type": "save_weights", "path": path}, "json"), "json"

    def save_weights_for_sampler(self, body, **_):
        run = self._run(body["model_id"])
        self._check_seq(run, body.get("seq_id"), "save_weights_for_sampler")
        if body.get("path"):
            path = self._save(body["model_id"], "sampler_weights", body["path"], run["table"], body)
            return 200, self._future({"type": "save_weights_for_sampler", "path": path}, "json"), "json"
        sid = f"sampler-{uuid.uuid4().hex[:10]}"
        self.samplers[sid] = {"table": run["table"].copy(), "base": run["base"], "session": run["session"]}
        return 200, self._future({"type": "save_weights_for_sampler", "sampling_session_id": sid}, "json"), "json"

    def create_sampling_session(self, body, **_):
        sid = self._session(body)
        if body.get("model_path"):
            ck = self.checkpoints.get(body["model_path"])
            if ck is None or ck["type"] != "sampler":
                raise Fail(404, f"Sampler checkpoint {body['model_path']!r} not found")
            table, base = ck["table"].copy(), ck["base"]
            ssid = f"sampler-{uuid.uuid4().hex[:10]}"
            self.samplers[ssid] = {"table": table, "base": base, "session": sid, "path": body["model_path"]}
            return 200, {"type": "create_sampling_session", "sampling_session_id": ssid}, "json"
        else:
            base = body.get("base_model")
            table = Table(self.model(base)["vocab"])
        ssid = f"sampler-{uuid.uuid4().hex[:10]}"
        self.samplers[ssid] = {"table": table, "base": base, "session": sid}
        return 200, {"type": "create_sampling_session", "sampling_session_id": ssid}, "json"

    def get_sampler(self, sampler_id, **_):
        s = self.samplers.get(sampler_id)
        if s is None:
            raise Fail(404, "Unknown sampling session")
        return 200, {"sampler_id": sampler_id, "base_model": s["base"], "model_path": s.get("path")}, "json"

    # ---- sampling

    def asample(self, body, **_):
        if body.get("sampling_session_id"):
            s = self.samplers.get(body["sampling_session_id"])
            if s is None:
                raise Fail(404, "Unknown sampling session")
        elif body.get("model_path"):
            ck = self.checkpoints.get(body["model_path"])
            if ck is None:
                raise Fail(404, "Sampler checkpoint not found")
            s = {"table": ck["table"], "base": ck["base"], "session": None}
        else:
            s = {"table": Table(self.model(body["base_model"])["vocab"]), "base": body["base_model"], "session": None}
        table: Table = s["table"]
        chunks = [self._json_chunk(c) for c in body["prompt"]["chunks"]]
        self._check_pictures(s["base"], chunks)
        p = body.get("sampling_params") or {}
        stop = p.get("stop")
        stop_tokens = {int(x) for x in stop} if isinstance(stop, list) and stop and isinstance(stop[0], int) else set()
        max_tokens = p.get("max_tokens") or 256
        temp = float(p.get("temperature", 1.0))
        rng = random.Random(p["seed"]) if p.get("seed") is not None else self.rng
        toks, hashes = prefix_hashes(chunks)
        key = hashes[-1]
        cached = key in self.prompt_cache
        self.prompt_cache.add(key)
        n = int(body.get("num_samples", 1))
        self._bill("sampling_prefill", len(toks), s["base"], s["session"], cached=cached)
        seqs = []
        for _ in range(n):
            ctx, out, lps, reason = key, [], [], "length"
            for _i in range(max_tokens):
                t = table.sample(ctx, temp, rng)
                out.append(t)
                lps.append(table.logprob(ctx, t))
                ctx = _h(ctx, t)
                if t in stop_tokens:
                    reason = "stop"
                    break
            seqs.append({"tokens": out, "logprobs": lps, "stop_reason": reason})
            self._bill("sampling_sample", len(out), s["base"], s["session"])
        result: Dict[str, Any] = {"sequences": seqs}
        if body.get("prompt_logprobs"):
            result["prompt_logprobs"] = [float("nan")] + [
                table.logprob(hashes[i], toks[i]) if toks[i] is not None else float("nan") for i in range(1, len(toks))]
        if body.get("target_prompt_logprobs"):
            t = body["target_prompt_logprobs"]
            shape = t.get("shape") or [len(t["data"])]
            k = shape[1] if len(shape) == 2 else 1
            if t.get("sparse_crow_indices") is not None:          # CSR: the entries left out are -1
                crow, cols = t["sparse_crow_indices"], t["sparse_col_indices"]
                flat = [-1] * (shape[0] * k)
                for i in range(shape[0]):
                    for at in range(crow[i], crow[i + 1]):
                        flat[i * k + cols[at]] = t["data"][at]
            else:
                flat = [x for row in t["data"] for x in row] if t["data"] and isinstance(t["data"][0], list) \
                    else list(t["data"])
            if shape[0] != len(toks) - 1:
                raise Fail(400, f"target_prompt_logprobs has {shape[0]} rows; the prompt needs len(prompt) - 1 = "
                                f"{len(toks) - 1}")
            vals = []
            for i in range(shape[0]):
                for j in range(k):
                    y = int(flat[i * k + j])            # -1: a padded entry (Tinker's recipes pad with -1)
                    vals.append(table.logprob(hashes[i + 1], y) if y >= 0 else float("nan"))
            result["target_prompt_logprobs"] = {"data": vals, "shape": [shape[0], k] if k > 1 else [shape[0]]}
        ids = [f"seq-{uuid.uuid4().hex[:12]}" for _ in seqs]
        for sid_, q in zip(ids, seqs):
            q["sequence_id"] = sid_
        return 200, self._future(result, "sample", sample_sequence_ids=ids), "json"

    # ---- futures

    def retrieve_future(self, body, headers, **_):
        f = self.futures.get(body["request_id"])
        if f is None:
            raise Fail(410, "Promise expired")
        if f["polls"] < int(self.options["try_again"]):
            f["polls"] += 1
            if f["polls"] % 2:
                return 408, {"type": "try_again", "request_id": body["request_id"], "queue_state": "active"}, "json"
            return 200, {"type": "try_again", "request_id": body["request_id"], "queue_state": "active"}, "json"
        proto = "application/x-protobuf" in headers.get("accept", "")
        r = f["result"]
        if f["kind"] == "fwdbwd":
            return (200, self._fwdbwd_proto(r), "proto") if proto else (200, self._fwdbwd_json(r), "json")
        if f["kind"] == "sample":
            return (200, self._sample_proto(r), "proto") if proto else (200, self._sample_json(r), "json")
        return 200, r, "json"

    @staticmethod
    def _fwdbwd_json(r):
        return {"loss_fn_output_type": r["loss_fn_output_type"], "metrics": r["metrics"],
                "loss_fn_outputs": [{"logprobs": {"data": [None if x != x else x for x in o["logprobs"]],
                                                  "dtype": "float32", "shape": o["shape"]}}
                                    for o in r["outputs"]]}

    @staticmethod
    def _fwdbwd_proto(r) -> bytes:
        from tinker.proto import tinker_public_pb2 as pb
        msg = pb.ForwardBackwardOutput()
        msg.loss_fn_output_type = r["loss_fn_output_type"]
        for k, v in r["metrics"].items():
            msg.metrics[k] = v
        rec = msg.loss_fn_outputs.add()
        rec.type_tag = r["loss_fn_output_type"]
        rec.num_datums = len(r["outputs"])
        k = r["outputs"][0]["shape"][1] if r["outputs"] and len(r["outputs"][0]["shape"]) == 2 else None
        bt = rec.fields["logprobs"]
        bt.dtype = pb.DTYPE_FLOAT32
        if k:
            bt.trailing_shape.extend([k])
        data, offsets = b"", [0]
        for o in r["outputs"]:
            data += struct.pack(f"<{len(o['logprobs'])}f", *o["logprobs"])
            offsets.append(len(data))
        bt.data = data
        bt.offsets = struct.pack(f"<{len(offsets)}q", *offsets)
        return msg.SerializeToString()

    @staticmethod
    def _sample_json(r):
        out = {"type": "sample", "sequences": [{"tokens": s["tokens"], "logprobs": s["logprobs"],
                                                "stop_reason": s["stop_reason"], "sequence_id": s.get("sequence_id")}
                                               for s in r["sequences"]]}
        if "prompt_logprobs" in r:
            out["prompt_logprobs"] = [None if x != x else x for x in r["prompt_logprobs"]]
        if "target_prompt_logprobs" in r:
            t = r["target_prompt_logprobs"]
            out["target_prompt_logprobs"] = {"data": [None if x != x else x for x in t["data"]], "dtype": "float32",
                                             "shape": t["shape"]}
        return out

    @staticmethod
    def _sample_proto(r) -> bytes:
        from tinker.proto import tinker_public_pb2 as pb
        msg = pb.SampleResponse()
        for s in r["sequences"]:
            q = msg.sequences.add()
            q.stop_reason = pb.STOP_REASON_STOP if s["stop_reason"] == "stop" else pb.STOP_REASON_LENGTH
            q.tokens = struct.pack(f"<{len(s['tokens'])}i", *s["tokens"])
            q.logprobs = struct.pack(f"<{len(s['logprobs'])}f", *s["logprobs"])
        if "prompt_logprobs" in r:
            msg.prompt_logprobs = struct.pack(f"<{len(r['prompt_logprobs'])}f", *r["prompt_logprobs"])
        if "target_prompt_logprobs" in r:
            t = r["target_prompt_logprobs"]
            msg.target_prompt_logprobs.dtype = pb.DTYPE_FLOAT32
            msg.target_prompt_logprobs.shape.extend(t["shape"])
            msg.target_prompt_logprobs.dense = struct.pack(f"<{len(t['data'])}f", *t["data"])
        return msg.SerializeToString()

    # ---- checkpoints, billing

    def _ck_json(self, path: str, c: Dict[str, Any]) -> Dict[str, Any]:
        import datetime as dt
        iso = lambda s: dt.datetime.fromtimestamp(s, dt.timezone.utc).isoformat()  # noqa: E731
        return {"checkpoint_id": c["id"], "checkpoint_type": c["type"], "time": iso(c["time"]), "tinker_path": path,
                "size_bytes": c["size_bytes"], "public": False, "user_metadata": c["user_metadata"],
                "expires_at": iso(c["expires_at"]) if c["expires_at"] else None, "base_model": c["base"]}

    def _expire(self) -> None:
        now = time.time()
        for p in [p for p, c in self.checkpoints.items() if c["expires_at"] and c["expires_at"] < now]:
            del self.checkpoints[p]

    def list_user_checkpoints(self, query, **_):
        self._expire()
        items = sorted(self.checkpoints.items(), key=lambda kv: -kv[1]["time"])
        off, lim = int(query.get("offset", 0)), int(query.get("limit", 100))
        page = items[off:off + lim]
        return 200, {"checkpoints": [self._ck_json(p, c) for p, c in page],
                     "cursor": {"offset": off, "limit": lim, "total_count": len(items)}}, "json"

    def list_checkpoints(self, run_id, **_):
        self._expire()
        return 200, {"checkpoints": [self._ck_json(p, c) for p, c in self.checkpoints.items() if c["run"] == run_id]}, \
            "json"

    def delete_checkpoint(self, run_id, cid, **_):
        path = f"tinker://{run_id}/{cid}"
        if path not in self.checkpoints:
            raise Fail(404, "Checkpoint not found")
        del self.checkpoints[path]
        return 200, None, "json"

    def set_ttl(self, run_id, cid, body, **_):
        path = f"tinker://{run_id}/{cid}"
        if path not in self.checkpoints:
            raise Fail(404, "Checkpoint not found")
        ttl = body.get("ttl_seconds")
        self.checkpoints[path]["expires_at"] = time.time() + ttl if ttl else None
        return 200, None, "json"

    def billing_events(self, query, **_):
        import datetime as dt
        now = dt.datetime.now(dt.timezone.utc).replace(minute=0, second=0, microsecond=0)
        data = [{"bucket_start": now.isoformat(), "bucket_end": (now + dt.timedelta(hours=1)).isoformat(),
                 "base_model": u["base_model"], "session_id": u["session_id"],
                 "estimated_cost_usd": u["estimated_cost_usd"],
                 "effective_rate_usd_per_million_tokens": u["rate"],
                 "event_info": {"type": u["type"], "token_count": u["token_count"],
                                **({"cached": bool(u.get("cached"))} if u["type"] == "sampling_prefill" else {})}}
                for u in self.usage]
        sessions = {sid: {"user_metadata": s["user_metadata"]} for sid, s in self.sessions.items()}
        return 200, {"data": data, "sessions": sessions, "cost_data_through": now.isoformat()}, "json"

    def storage_usage(self, **_):
        self._expire()
        size = sum(c["size_bytes"] for c in self.checkpoints.values())
        paid = self.options["paid_storage"]
        return 200, {"effective_rate_usd_per_gigabyte_month": 0.10 if paid else 0.0,
                     "data": [{"checkpoint_count": len(self.checkpoints), "size_bytes": size,
                               "size_gigabytes": size / GB,
                               "estimated_monthly_cost_usd": round(size / GB * 0.10, 2) if paid else 0.0}]}, "json"

    # ---- control

    def fake_config(self, body, **_):
        models = body.pop("models", None)
        if models:
            self.models.update(models)
        self.options.update(body)
        return 200, {"options": self.options}, "json"

    def fake_state(self, **_):
        return 200, {"sessions": {k: {kk: vv for kk, vv in v.items()} for k, v in self.sessions.items()},
                     "runs": {k: {"base": v["base"], "rank": v["rank"], "seq": v["seq"], "meta": v["meta"],
                                  "steps": v["opt"].get("steps", 0), "loaded_from": v.get("loaded_from")}
                              for k, v in self.runs.items()},
                     "checkpoints": {p: self._ck_json(p, c) for p, c in self.checkpoints.items()},
                     "paths": [e["path"] for e in self.log],
                     "usage": self.usage, "options": self.options}, "json"


# ------------------------------------------------------------------ HTTP


def _json_default(x):
    if isinstance(x, float) and x != x:
        return None
    raise TypeError(type(x).__name__)


class _Handler(BaseHTTPRequestHandler):
    server_version = "fake-tinker/1"
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):          # quiet
        pass

    def _go(self, method: str) -> None:
        fake: FakeTinker = self.server.fake          # type: ignore[attr-defined]
        url = urllib.parse.urlsplit(self.path)
        path = url.path
        prefix = getattr(self.server, "prefix", "")
        if prefix and path.startswith(prefix):
            path = path[len(prefix):]
        query = {k: v[-1] for k, v in urllib.parse.parse_qs(url.query).items()}
        length = int(self.headers.get("content-length") or 0)
        raw = self.rfile.read(length) if length else b""
        headers = {k.lower(): v for k, v in self.headers.items()}
        ctype = headers.get("content-type", "")
        try:
            if "protobuf" in ctype:
                body: Any = raw
            elif raw:
                body = json.loads(raw)
            else:
                body = {}
            status, payload, kind = fake.handle(method, path, query, body, headers)
        except Fail as e:
            status, payload, kind = e.status, {"detail": e.detail}, "json"
        except Exception as e:  # noqa: BLE001 — a bug in a test or in the fake: said, not hidden
            import traceback
            traceback.print_exc()
            status, payload, kind = 500, {"detail": f"{type(e).__name__}: {e}"}, "json"
        if kind == "proto":
            out, ctype_out = payload, "application/x-protobuf"
        else:
            out, ctype_out = (b"null" if payload is None else json.dumps(payload, default=_json_default,
                                                                        allow_nan=False).encode()), \
                "application/json"
        self.send_response(status)
        self.send_header("content-type", ctype_out)
        self.send_header("content-length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def do_GET(self):
        self._go("GET")

    def do_POST(self):
        self._go("POST")

    def do_PUT(self):
        self._go("PUT")

    def do_DELETE(self):
        self._go("DELETE")


def serve(*, port: int = 0, api_key: Optional[str] = None, prefix: str = "", seed: int = 0
          ) -> Tuple[ThreadingHTTPServer, str, FakeTinker]:
    """Start the fake in a thread: (server, base url, the fake's state)."""
    fake = FakeTinker(api_key=api_key, seed=seed)
    server = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    server.fake = fake               # type: ignore[attr-defined]
    server.prefix = prefix           # type: ignore[attr-defined]
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}{prefix}", fake


def main(argv: Optional[List[str]] = None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--api-key", default=None)
    a = ap.parse_args(argv)
    server, url, _fake = serve(port=a.port, api_key=a.api_key)
    print(url, flush=True)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main(sys.argv[1:])
