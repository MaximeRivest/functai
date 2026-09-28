"""Save a functai program with everything it depends on; load it; prove it loads.

    report = functai.check(pipeline)          # the graph and the problems
    functai.save(pipeline, "pipeline/")       # refuses while check finds errors
    functai.verify("pipeline/", trust=True)   # a fresh environment, from the saved requirements only
    pipeline = functai.load("pipeline/", trust=True)

A saved program is a folder you can read, diff and put in git:

    functai.json          the contract: entry, every AI function's settings,
                          instruction, demos, signature and fingerprints;
                          versions; file hashes
    code/<module>.py      the code the program reaches, one file per original
                          module (a notebook or script is ``main.py``): the
                          functions and classes verbatim, the constants by
                          value, the imports it needs
    files/                the data files read with ``functai.file(...)``
    requirements.txt      the packages the code reaches, pinned
    requirements.lock     those and everything they pull in, as installed here

Loading runs the saved code, so it needs ``trust=True``. The hashes catch
accidental edits and corruption; they are not a signature against someone
who edits both the code and ``functai.json``.
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import datetime as _dt
import hashlib
import importlib
import itertools
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import types
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import lm15
import lmcc

from . import calllog

FORMAT = 1
MANIFEST = "functai.json"

# The facts lmcc binds a layout against when fingerprinting: fixed, so the
# fingerprint depends on the program, not on which model it will meet.
PROBE_CAPABILITIES = {"instruct": True, "native_structured_output": True, "native_function_calling": True,
                      "stop_sequences": True, "native_reasoning": False, "assistant_prefill": False}

# Settings that shape the prompt are part of the program: their effective value
# (from configure(...) too) is saved. Model and sampling settings are the
# deployment's: only the function's own are saved, so configure() still reaches
# the loaded program.
LAYOUT_SETTINGS = ("adapter", "module", "include_fn_name_in_instructions")
NOT_SAVED = ("client", "api_key", "auth", "optimizer", "teacher", "router",
             "log_calls", "caller",      # where calls are logged and who calls: the deployment's
             "observers", "journal")     # who receives a call's events: the host's


# ------------------------------------------------------------------ data files


class _FileMap:
    def __init__(self, base: Path, mapping: Dict[str, str]):
        self.base, self.mapping = base, mapping

    def resolve(self, given: str) -> Path:
        saved = self.mapping.get(given)
        if saved is None:
            raise FileNotFoundError(f"{given!r} was not saved with this program; functai saves the files named by "
                                    f"functai.file('...') with a literal path")
        return self.base / saved


_file_maps: Dict[str, _FileMap] = {}


def _files_for(module_file: str) -> Optional[_FileMap]:
    """The data files of the saved program ``module_file`` belongs to (generated code calls this)."""
    root = Path(module_file).resolve().parent.parent
    key = str(root)
    if key not in _file_maps:
        try:
            manifest = json.loads((root / MANIFEST).read_text())
        except (OSError, ValueError):
            return None
        _file_maps[key] = _FileMap(root / "files", manifest.get("data_files", {}))
    return _file_maps[key]


def file(path: "str | os.PathLike[str]") -> Path:
    """A data file the program reads: ``open(functai.file("data/stopwords.txt"))``.

    A relative path is relative to the file of the code that calls this (the
    current directory in a notebook). ``functai.save`` copies every file named
    this way with a literal path into the saved program, and the loaded program
    reads its own copy."""
    caller = sys._getframe(1).f_globals
    saved = caller.get("__functai_files__")
    if isinstance(saved, _FileMap):
        return saved.resolve(os.fspath(path))
    return _resolve_here(os.fspath(path), caller)


def _resolve_here(given: str, module_globals: Dict[str, Any]) -> Path:
    p = Path(given).expanduser()
    if p.is_absolute():
        return p
    base = module_globals.get("__file__")
    return (Path(base).resolve().parent / p) if base else (Path.cwd() / p)


# ------------------------------------------------------------------ writing code


def _flat(module: str) -> str:
    return "main" if module == "__main__" else module.replace(".", "__")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _position(node) -> int:
    fn = getattr(node.obj, "_fn", node.obj)
    code = getattr(fn, "__code__", None)
    if code is not None:
        return code.co_firstlineno
    return getattr(fn, "__firstlineno__", 10 ** 9)


def _generate(report, module: str) -> str:
    """The source of one saved module."""
    bindings = report.bindings.get(module, {})
    nodes = [n for n in report.nodes.values() if n.module == module]
    names_of_defs = {n.name for n in nodes}
    lines = [f'"""Saved by functai from {module!r}. Generated: edit the original code, then save again."""', ""]
    if any(n.future_annotations for n in nodes):
        lines += ["from __future__ import annotations", ""]
    lines += ["import functai.saved as __functai_saved__", ""]
    imports = sorted({b.stmt for b in bindings.values() if b.kind == "import"})
    local = []
    for name, b in sorted(bindings.items()):
        if b.kind == "local":
            local.append(f"from .{_flat(b.module)} import {b.attr}" + ("" if b.attr == name else f" as {name}"))
        elif b.kind == "localmod":
            local.append(f"from . import {_flat(b.module)} as {name}")
    lines += imports + ([""] if imports else []) + sorted(local) + ([""] if local else [])
    lines += ["__functai_files__ = __functai_saved__._files_for(__file__)", ""]

    # definitions and values, in an order where everything a def evaluates exists
    items: Dict[str, Tuple[int, str, set]] = {}
    for n in nodes:
        deps = {x for x in n.__dict__.get("_def_deps", set())}
        items[n.name] = (_position(n), n.source.rstrip() + "\n", deps)
    wrappers: List[str] = []
    aliases: List[str] = []
    for name, b in bindings.items():
        if b.kind != "value":
            continue
        if b.expr in names_of_defs and b.needs == (b.expr,):
            aliases.append(f"{name} = {b.expr}")
            continue
        items[name] = (-1, f"{name} = {b.expr}\n", set(b.needs))
    for n in nodes:
        n_deps = set()
        tree_names = getattr(n, "_deftime", set())
        for dep in tree_names:
            if dep in items and dep != n.name:
                n_deps.add(dep)
        pos, src, _ = items[n.name]
        items[n.name] = (pos, src, n_deps)
    order: List[str] = []
    placed: set = set()
    pending = sorted(items, key=lambda k: (items[k][0], k))
    while pending:
        progressed = False
        for k in list(pending):
            if not ({d for d in items[k][2] if d in items} - placed - {k}):
                order.append(k)
                placed.add(k)
                pending.remove(k)
                progressed = True
                break
        if not progressed:                      # a cycle: keep the original order
            order += pending
            break
    for k in order:
        lines += [items[k][1], ""]
    for n in nodes:
        if n.kind == "ai":
            wrappers.append(f"{n.name} = __functai_saved__._rebuild_ai({n.name}, {n.key!r})")
        elif n.kind == "module":
            wrappers.append(f"{n.name} = __functai_saved__._rebuild_module({n.name}, {n.key!r})")
    lines += ["", *wrappers] + ([""] if aliases else []) + aliases
    return "\n".join(lines).rstrip() + "\n"


# ------------------------------------------------------------------ the manifest


def _is_baked(obj: Any) -> bool:
    return getattr(type(obj), "__functai_baked__", False) is True


def _settings_json(fn, where: str, node=None, models: Optional[Dict[str, str]] = None
                   ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """(settings, lm15 config) as JSON, for one AI function. A baked model is
    ``{"baked": <name in models/>}``; an AI function to escalate to, ``{"node": key}``."""
    from .config import CONFIG_FIELDS, effective
    own = dict(fn._settings)
    eff = effective(own)
    out: Dict[str, Any] = {}
    config: Dict[str, Any] = {}
    for k, v in own.items():
        if k in NOT_SAVED or v is None:
            continue
        if k in CONFIG_FIELDS:
            config[k] = v
            continue
        if _is_baked(v):
            out[k] = {"baked": (models or {})[str(v.path)]}
            continue
        if k == "escalate_to" and not isinstance(v, str):
            out[k] = {"node": getattr(node, "escalate", None)}
            continue
        if k == "lm" and not isinstance(v, str):
            from .models import model_string
            v = getattr(getattr(v, "selection", None), "routed", None) or model_string(v)
        if k == "adapter" and isinstance(v, lmcc.Adapter):
            v = v.dump()
        out[k] = v
    for k in LAYOUT_SETTINGS:
        if k not in out and eff.get(k) is not None:
            v = eff[k]
            out[k] = v.dump() if isinstance(v, lmcc.Adapter) else v
    teacher = own.get("teacher")
    if isinstance(teacher, str):
        out["teacher"] = teacher
    config_json = lm15.serde.config_to_dict(lm15.Config(**config)) if config else {}
    json.dumps(out)
    return out, config_json


def _probe_inputs(fn, spec, examples: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Inputs to render when fingerprinting: one made from the input types, the
    first demos' inputs, and the examples given to save."""
    from .graph import state_to_json
    to_json = lmcc.turn.to_json
    probes = [_sample_inputs(spec)]
    for demo in state_to_json(fn._state)["demos"][:3]:
        if isinstance(demo, dict) and isinstance(demo.get("inputs"), dict):
            probes.append(demo["inputs"])
    for ex in examples:
        probes.append(to_json(ex, where="example"))
    unique: List[Dict[str, Any]] = []
    for p in probes:
        if p not in unique:
            unique.append(p)
    return unique


def _sample(shape: Dict[str, Any]) -> Any:
    if "enum" in shape:
        return shape["enum"][0]
    if "anyOf" in shape:
        options = [s for s in shape["anyOf"] if s.get("type") != "null"]
        return _sample(options[0]) if options else None
    t = shape.get("type")
    return {"string": "example text", "integer": 3, "number": 2.5, "boolean": True, "array": [],
            "object": {}, "null": None}.get(t, "example text")


def _sample_inputs(spec) -> Dict[str, Any]:
    return {f.name: _sample(f.shape) for f in spec.signature.inputs if f.purpose == "plain"}


def probe_plan(fn, spec, settings: Dict[str, Any]) -> Tuple[Any, List[Any]]:
    """The plan and worked examples an AI function renders probes with: its
    layout under fixed capabilities (a baked model's own layout and facts), and
    no conversation memory."""
    from . import adapters
    baked = settings.get("lm") if _is_baked(settings.get("lm")) else None
    if baked is not None:      # the layout and facts the weights were trained with
        plan = adapters.bind(adapters.Layout(adapter=baked.layout), spec.signature, baked.capabilities,
                             baked.provider)
    else:
        plan = adapters.bind(fn._layout(settings), spec.signature, PROBE_CAPABILITIES, "probe")
    return plan, fn._past(plan, spec, {**settings, "stateful": False})


def probe_request(fn, spec, plan, past, inputs: Dict[str, Any]) -> Dict[str, Any]:
    """The request an AI function renders for ``inputs`` (lmcc.Refusal when it cannot)."""
    from . import engine
    values = engine.prepare_inputs(spec, inputs)
    if spec.tools:
        values["tools"] = list(fn._tool_specs)
    return plan.render(plan.turn(values), turns=past).request("probe")


def request_fingerprint(request: Dict[str, Any]) -> str:
    return "sha256:" + _sha256(calllog.canonical(request).encode())


def _fingerprints(fn, probes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """What the program sends, as hashes: the signature, and the exact request
    for each probe (instruction, layout, demos, tools) under fixed capabilities."""
    spec = fn._spec()
    settings = fn._effective()
    baked = settings.get("lm") if _is_baked(settings.get("lm")) else None
    plan, past = probe_plan(fn, spec, settings)
    renders = []
    requests = []
    for inputs in probes:
        try:
            request = probe_request(fn, spec, plan, past, inputs)
            renders.append(request_fingerprint(request))
            requests.append(request)
        except lmcc.Refusal as exc:
            renders.append(f"refused:{exc.code}")
    out = {"signature": lmcc.signature_fingerprint(spec.signature), "requests": renders}
    if baked is not None and requests:
        out["answers"] = _model_answers(baked, requests)
    return out


def _model_answers(baked: Any, requests: List[Dict[str, Any]]) -> List[Any]:
    """What the weights answer to the probe requests, computed on the CPU in fp32
    (the same everywhere, up to float rounding): each field's probabilities for a
    head, the greedy reply for a generative student."""
    cpu = type(baked)(baked.path, device="cpu", check=False)
    requests_ = [lm15.serde.request_from_dict(r) for r in requests]
    if cpu.kind == "head":
        from .bake.examples import request_text
        dists = cpu.probabilities([request_text(r) for r in requests_])
        return [{f: {k: round(p, 6) for k, p in d.items()} for f, d in dist.items()} for dist in dists]
    return [cpu.complete(r).text for r in requests_]


def _same_answers(was: List[Any], now: List[Any], tolerance: float = 2e-3) -> Optional[str]:
    for i, (a, b) in enumerate(zip(was, now)):
        if isinstance(a, dict):
            for field, dist in a.items():
                other = b.get(field, {})
                if max(dist, key=dist.get) != max(other, key=other.get):
                    return f"probe {i}: the weights answer {max(other, key=other.get)!r}, not {max(dist, key=dist.get)!r}"
                gap = max(abs(dist[k] - other.get(k, 0.0)) for k in dist)
                if gap > tolerance:
                    return f"probe {i}: the weights' probabilities moved by {gap:.4f} (more than {tolerance})"
        elif a != b:
            return f"probe {i}: the weights reply {b!r}, not {a!r}"
    return None


def _baked_models(report) -> Dict[str, Any]:
    """Every baked model the program uses: its folder → (name in models/, the model)."""
    out: Dict[str, Any] = {}
    taken: set = set()
    for n in report.nodes.values():
        for b in n.baked.values():
            key = str(b.path)
            if key in out:
                continue
            name, i = b.name, 1
            while name in taken:
                i += 1
                name = f"{b.name}-{i}"
            taken.add(name)
            out[key] = (name, b)
    return out


def _manifest(report, examples, allowed, models: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    from .graph import state_to_json
    nodes: Dict[str, Any] = {}
    for key, n in report.nodes.items():
        entry: Dict[str, Any] = {"kind": n.kind, "module": n.module, "name": n.name}
        if n.kind in ("ai", "module"):
            entry["interface"] = n.obj.interface
        if n.kind == "ai":
            fn = n.obj
            settings, config = _settings_json(fn, key, n, models)
            probes = _probe_inputs(fn, fn._spec(), examples if key == report.entry else ())
            entry["ai"] = {
                "settings": settings,
                "config": config,
                "template": [dict(m) for m in fn._template] if fn._template is not None else None,
                "tools": n.tools,
                "teacher": n.teacher,
                "state": state_to_json(fn._state),
                "requires": list(n.requires),
                "signature": lmcc.signature_to_dict(fn._spec().signature),
                "probes": probes,
                "fingerprints": _fingerprints(fn, probes),
                # null: the model writes the whole body, so another language can run this
                # function from this entry alone (contract/saved.md)
                "body": None if calllog._model_body(fn.__wrapped__) else {"code": calllog.code_hash(fn.__wrapped__)},
                "version": calllog.ai_version(fn),
            }
        elif n.kind == "module":
            m = n.obj
            entry["module_program"] = {"call_defaults": lmcc.turn.to_json(dict(n.obj._opt_call_defaults)),
                                       "requires": list(n.requires)}
            if m._declared:
                entry["module_program"]["declared"] = True      # rebuilt from the interface, not derived again
            own = {k: v for k, v in m._settings.items() if k not in NOT_SAVED}
            if own:
                entry["module_program"]["settings"] = own
        nodes[key] = entry
    return {
        "functai_saved": FORMAT,
        "language": "python",
        "entry": report.entry,
        "created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "python": ".".join(map(str, sys.version_info[:3])),
        "modules": {m: f"code/{_flat(m)}.py" for m in report.bindings},
        "nodes": nodes,
        "requirements": [r.spec for r in report.requirements.values()] + list(report.declared),
        "allowed": [dataclasses.asdict(p) for p in allowed],
        "warnings": [dataclasses.asdict(p) for p in report.warnings],
    }


# ------------------------------------------------------------------ save


def _data_files(report) -> Dict[str, Tuple[str, Path]]:
    """given → (saved relative path, source path) for every functai.file(...) literal."""
    out: Dict[str, Tuple[str, Path]] = {}
    for n in report.nodes.values():
        mod = sys.modules.get(n.module)
        g = vars(mod) if mod is not None else {}
        if n.module == "__main__" and not g.get("__file__"):
            g = {}
        for given in n.files:
            src = _resolve_here(given, g)
            rel = Path(given)
            saved = (Path("abs") / src.name) if rel.is_absolute() else Path(*[p for p in rel.parts if p != ".."])
            if given in out and out[given][1] != src:
                raise ValueError(f"functai.file({given!r}) names two different files from two modules; "
                                 f"use distinct paths")
            out[given] = (saved.as_posix(), src)
    return out


def _global_settings() -> Dict[str, Any]:
    """The settings configure(...) set when saving (what a replay needs), as JSON;
    connections and secrets are never saved."""
    from . import config
    out: Dict[str, Any] = {}
    config_fields: Dict[str, Any] = {}
    for k, v in {**config._GLOBAL, **config._SCOPED.get()}.items():
        if k in NOT_SAVED or v is None or v == config.DEFAULTS.get(k, object()):
            continue
        if k in config.CONFIG_FIELDS:
            config_fields[k] = v
        elif k == "lm" and not isinstance(v, str):
            from .models import model_string
            out[k] = getattr(getattr(v, "selection", None), "routed", None) or model_string(v)
        elif k == "adapter" and isinstance(v, lmcc.Adapter):
            out[k] = v.dump()
        else:
            try:
                json.dumps(v)
            except TypeError:
                continue
            out[k] = v
    if config_fields:
        out["__config__"] = lm15.serde.config_to_dict(lm15.Config(**config_fields))
    return out


def _settings_from_json(data: Dict[str, Any]) -> Dict[str, Any]:
    from .config import CONFIG_FIELDS
    out = {k: v for k, v in data.items() if k != "__config__"}
    if data.get("__config__"):
        config, default = lm15.serde.config_from_dict(data["__config__"]), lm15.Config()
        for f in dataclasses.fields(config):
            if f.name in CONFIG_FIELDS and getattr(config, f.name) != getattr(default, f.name):
                out[f.name] = getattr(config, f.name)
    return out


def _output_json(value: Any) -> Any:
    try:
        return {"json": lmcc.turn.to_json(value)}
    except lmcc.Refusal:
        return {"repr": repr(value)}


def _record(program: Any, inputs_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Run the program on each input, for real, and keep every model exchange."""
    from . import engine
    recorded = []
    routes: Dict[str, Any] = {}
    for inputs in inputs_list:
        rec: Dict[str, Any] = {"exchanges": [], "routes": routes}
        token = engine.RECORDING.set(rec)
        try:
            output = program(**inputs)
        finally:
            engine.RECORDING.reset(token)
        recorded.append({"inputs": lmcc.turn.to_json(inputs, where="run inputs"), "output": _output_json(output),
                         "exchanges": rec["exchanges"]})
    return {"settings": _global_settings(), "routes": routes, "recordings": recorded}


def save(program: Any, path: "str | os.PathLike[str]", *, include: Iterable[str] = (),
         requires: Iterable[str] = (), allow: Iterable[str] = (), examples: Iterable[Dict[str, Any]] = (),
         record: Iterable[Dict[str, Any]] = (), overwrite: bool = False, weights: str = "copy"):
    '''Save a program to a folder, with everything it depends on.

    The folder holds the code the program reaches, each AI function's
    settings, instruction and demos, data files read with ``functai.file``,
    and pinned requirements. It is written whole or not at all. Keys and
    connections are never saved.

    Parameters
    ----------
    program : AI function, module, or function
        The program's entry point.
    path : str or path
        The folder to write.
    include, requires : list of str
        As for ``check``.
    allow : list of str
        Problems to accept on purpose, like ``["hidden-state"]`` to save a
        global's current value.
    record : list of dict
        Inputs to run the program on now, for real (this calls the model).
        The replies are recorded, and ``verify`` replays them to prove the
        saved program gives the same results, without calling a model.
    examples : list of dict
        Inputs to fingerprint the rendered requests of, besides the demos.
    overwrite : bool
        Replace an existing folder.
    weights : str
        Baked weights: ``"copy"`` them into the folder (default) or
        ``"reference"`` them.

    Returns
    -------
    Report
        The ``check`` report of what was saved.

    Raises
    ------
    Refused
        While ``check`` finds errors, with the report.

    See Also
    --------
    check : what would be saved.
    verify : prove the folder runs in a fresh environment.
    load : read it back.

    Examples
    --------
    ```python
    import tempfile, os

    @ai
    def capital(country: str) -> str:
        """The country's capital city."""
        ...

    folder = os.path.join(tempfile.mkdtemp(), "capital")
    save(capital, folder, record=[{"country": "Kenya"}])
    sorted(os.listdir(folder))
    ```
    '''
    from .graph import Refused, check, lock
    report = check(program, include=include, requires=requires)
    allow = set(allow)
    blocking = [p for p in report.errors if p.code not in allow]
    if blocking:
        raise Refused(report)
    allowed = [p for p in report.errors if p.code in allow]
    target = Path(path).expanduser().resolve()
    if target.exists() and not overwrite:
        raise FileExistsError(f"{target} exists; save(..., overwrite=True) replaces it")
    examples = list(examples)
    files = _data_files(report)
    for given, (_saved, src) in files.items():
        if not src.exists():
            raise FileNotFoundError(f"functai.file({given!r}): {src} does not exist")
    recorded = _record(program, list(record)) if record else None
    if weights not in ("copy", "reference"):
        raise ValueError(f"weights is 'copy' or 'reference', not {weights!r}")
    baked = _baked_models(report)
    manifest = _manifest(report, examples, allowed, {k: name for k, (name, _b) in baked.items()})
    manifest["models"] = {}
    manifest["data_files"] = {given: saved for given, (saved, _src) in files.items()}
    tmp = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        hashes: Dict[str, str] = {}
        (tmp / "code").mkdir()
        for module in report.bindings:
            rel = manifest["modules"][module]
            text = _generate(report, module)
            (tmp / rel).write_text(text)
            hashes[rel] = _sha256(text.encode())
        for n in report.nodes.values():            # modules that only define nodes (no bindings)
            rel = f"code/{_flat(n.module)}.py"
            if n.module not in manifest["modules"]:
                manifest["modules"][n.module] = rel
                text = _generate(report, n.module)
                (tmp / rel).write_text(text)
                hashes[rel] = _sha256(text.encode())
        for _key, (name, b) in baked.items():
            if weights == "copy":
                shutil.copytree(b.path, tmp / "models" / name)
                for f in sorted((tmp / "models" / name).rglob("*")):
                    if f.is_file():
                        hashes[f.relative_to(tmp).as_posix()] = _sha256(f.read_bytes())
                manifest["models"][name] = {"folder": f"models/{name}", "student": b.student, "kind": b.kind}
            else:
                manifest["models"][name] = {"path": str(b.path), "student": b.student, "kind": b.kind,
                                            "baked_json": _sha256((b.path / "baked.json").read_bytes()),
                                            "hashes": b.meta.get("hashes", {})}
        for given, (saved, src) in files.items():
            dest = tmp / "files" / saved
            dest.parent.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                shutil.copytree(src, dest)
                for f in sorted(dest.rglob("*")):
                    if f.is_file():
                        hashes[f.relative_to(tmp).as_posix()] = _sha256(f.read_bytes())
            else:
                shutil.copyfile(src, dest)
                hashes[f"files/{saved}"] = _sha256(dest.read_bytes())
        if recorded is not None:
            text = json.dumps(recorded, indent=1, ensure_ascii=False) + "\n"
            (tmp / "recordings.json").write_text(text)
            hashes["recordings.json"] = _sha256(text.encode())
        direct = "\n".join(manifest["requirements"]) + "\n"
        locked = "\n".join(r.spec for r in lock(report.requirements.values())) + "\n"
        (tmp / "requirements.txt").write_text(direct)
        (tmp / "requirements.lock").write_text(locked)
        hashes["requirements.txt"] = _sha256(direct.encode())
        hashes["requirements.lock"] = _sha256(locked.encode())
        manifest["hashes"] = dict(sorted(hashes.items()))
        (tmp / MANIFEST).write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
        if target.exists():
            old = target.with_name(f".{target.name}.old-{os.getpid()}")
            os.replace(target, old)
            os.replace(tmp, target)
            shutil.rmtree(old, ignore_errors=True)
        else:
            os.replace(tmp, target)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return report


# ------------------------------------------------------------------ load


class LoadRefused(Exception):
    """The saved program cannot be loaded as saved; ``.problems`` says why, and
    ``.code`` the contract's word for it when there is one (``saved-code``,
    ``saved-differs``, ``interface-malformed``...: contract/saved.md)."""

    def __init__(self, message: str, problems: List[str], code: Optional[str] = None):
        super().__init__(message + "".join(f"\n  - {p}" for p in problems))
        self.problems = problems
        self.code = code


def _read_manifest(root: Path) -> Dict[str, Any]:
    try:
        manifest = json.loads((root / MANIFEST).read_text())
    except FileNotFoundError:
        raise LoadRefused(f"{root} is not a saved functai program", [f"{MANIFEST} is missing"]) from None
    if manifest.get("functai_saved") != FORMAT:
        raise LoadRefused(f"{root}: unknown saved format", [f"functai_saved={manifest.get('functai_saved')!r}"])
    return manifest


def _check_hashes(root: Path, manifest: Dict[str, Any]) -> List[str]:
    problems = []
    for rel, digest in manifest.get("hashes", {}).items():
        p = root / rel
        if not p.exists():
            problems.append(f"{rel} is missing")
        elif _sha256(p.read_bytes()) != digest:
            problems.append(f"{rel} was changed since it was saved")
    for p in (root / "code").glob("*.py"):
        if p.relative_to(root).as_posix() not in manifest.get("hashes", {}):
            problems.append(f"{p.relative_to(root)} was added after saving")
    return problems


def _check_environment(manifest: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """(missing, drifted) requirements, from requirements.txt's pins."""
    import importlib.metadata as md
    import re
    missing, drifted = [], []
    for spec in manifest.get("requirements", []):
        m = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:==\s*([^\s;]+))?", spec)
        if not m:
            continue
        try:
            have = md.version(m.group(1))
        except md.PackageNotFoundError:
            missing.append(spec)
            continue
        if m.group(2) and have != m.group(2):
            drifted.append(f"{m.group(1)} {have} here, {m.group(2)} when saved")
    return missing, drifted


_counter = itertools.count(1)
_import_lock = threading.Lock()
_programs: Dict[str, Dict[str, Any]] = {}          # package name → manifest
_roots: Dict[str, str] = {}                         # package name → the saved folder
_ids: Dict[str, str] = {}                           # package name → "sha256:" of its functai.json


def origin(module: Optional[str]) -> Tuple[str, Optional[str]]:
    """Where code comes from: for a module of a loaded saved program, the
    module it was saved from and the saved folder's id (``sha256:`` of its
    ``functai.json``); for any other module, itself (code with no module,
    run with exec: ``__main__``) and None."""
    module = module or "__main__"
    package, _, flat = module.rpartition(".")
    manifest = _programs.get(package)
    if manifest is None:
        return module, None
    original = next((m for m in manifest.get("modules", {}) if _flat(m) == flat), module)
    return original, _ids.get(package)


def _manifest_of(fn: Any) -> Tuple[Dict[str, Any], str]:
    package = fn.__module__.rsplit(".", 1)[0]
    return _programs[package], package


def _node_object(package: str, manifest: Dict[str, Any], key: str) -> Any:
    module, _, name = key.rpartition(":")
    mod = importlib.import_module(f"{package}.{_flat(module)}")
    return getattr(mod, name)


def _rebuild_ai(fn: Any, key: str):
    """Generated code calls this: the AI function saved under ``key``, rebuilt from ``fn``."""
    from .config import CONFIG_FIELDS
    from .core import FunctAIFunc, ProgramState
    manifest, package = _manifest_of(fn)
    data = manifest["nodes"][key]["ai"]
    settings = dict(data["settings"])
    for k, v in list(settings.items()):
        if isinstance(v, dict) and "baked" in v:
            settings[k] = _baked_model(package, manifest, v["baked"])
        elif isinstance(v, dict) and "node" in v:
            settings.pop(k)                  # an AI function: set after every module is loaded
    if data.get("config"):
        config = lm15.serde.config_from_dict(data["config"])
        default = lm15.Config()
        for f in dataclasses.fields(config):
            v = getattr(config, f.name)
            if f.name in CONFIG_FIELDS and v != getattr(default, f.name):
                settings[f.name] = v
    tools = []
    for t in data.get("tools", []):
        if isinstance(t, str):
            tools.append(_node_object(package, manifest, t))
        elif "import" in t:
            mod, _, attr = t["import"].partition(":")
            tools.append(getattr(importlib.import_module(mod), attr))
        else:
            from lmcc_std.tools import Tool
            tools.append(Tool(**t["tool"]))
    program = FunctAIFunc(fn, tools=tools or None, template=data.get("template"), requires=data.get("requires"),
                          **settings)
    program.load_state(ProgramState.from_dict(data["state"]))
    return program


_models_loaded: Dict[Tuple[str, str], Any] = {}


def _baked_model(package: str, manifest: Dict[str, Any], name: str):
    """A saved program's baked model, loaded once (its files checked)."""
    from .bake.baked import Baked
    key = (package, name)
    if key not in _models_loaded:
        info = manifest["models"][name]
        if "folder" in info:
            path = Path(_roots[package]) / info["folder"]
        else:
            path = Path(info["path"])
            if not (path / "baked.json").exists():
                raise LoadRefused(f"the baked model {name!r} is referenced, not copied", [f"{path} does not exist"])
            if _sha256((path / "baked.json").read_bytes()) != info["baked_json"]:
                raise LoadRefused(f"the baked model {name!r} changed since saving", [f"{path}/baked.json"])
        _models_loaded[key] = Baked(path)
    return _models_loaded[key]


def _rebuild_module(fn: Any, key: str):
    from .module import FunctAIModule
    manifest, _package = _manifest_of(fn)
    node = manifest["nodes"][key]
    data = node.get("module_program", {})
    declared = node.get("interface") if data.get("declared") else None
    m = FunctAIModule(fn, requires=data.get("requires", ()), interface=declared, **(data.get("settings") or {}))
    m._opt_call_defaults = dict(data.get("call_defaults") or {})
    return m


def _verify_loaded(package: str, manifest: Dict[str, Any]) -> List[str]:
    """Fingerprints recomputed here against the saved ones."""
    from .interface import signature
    problems = []
    for key, node in manifest["nodes"].items():
        if node["kind"] in ("ai", "module") and isinstance(node.get("interface"), dict):
            obj = _node_object(package, manifest, key)
            if signature(obj.interface) != signature(node["interface"]):
                problems.append(f"{key}: it takes or gives other data than its saved interface says")
        if node["kind"] != "ai":
            continue
        fn = _node_object(package, manifest, key)
        data = node["ai"]
        now = _fingerprints(fn, data["probes"])
        was = data["fingerprints"]
        if now["signature"] != was["signature"]:
            problems.append(f"{key}: its signature differs from the saved one")
        for i, (a, b) in enumerate(zip(now["requests"], was["requests"])):
            if a != b:
                problems.append(f"{key}: probe {i} renders a different request than when saved "
                                f"(the prompt changed: {b} → {a})")
        if "answers" in was:
            moved = _same_answers(was["answers"], now.get("answers", []))
            if moved:
                problems.append(f"{key}: its baked model does not answer as it did when saved: {moved}")
    return problems


def load(path: "str | os.PathLike[str]", *, trust: bool = False, check_env: str = "refuse",
         node: Optional[str] = None):
    '''Load a saved program, ready to call.

    Checks before running anything: the files' hashes (catching accidental
    edits) and the packages this environment has. After loading, checks that
    every AI function renders the same requests as when it was saved.

    A folder written in another language (TypeScript, R, Julia) holds no
    Python code: its AI functions load from ``functai.json`` alone, as data
    (``from_manifest``), and need no ``trust``.

    Parameters
    ----------
    path : str or path
        The saved folder.
    trust : bool
        Must be True: loading runs the saved code. The hashes catch
        accidents, not someone who edits both the code and ``functai.json``.
    node : str, optional
        For a folder of another language: the AI function to load, by key
        (``"module:name"``); the entry by default.
    check_env : str
        ``"refuse"`` (default) refuses on version or request differences;
        ``"warn"`` loads anyway, with warnings. Missing packages always
        refuse.

    Returns
    -------
    AI function or module
        The program, as it was saved.

    Raises
    ------
    LoadRefused
        When a check fails, saying which and why.

    See Also
    --------
    save : write the folder.
    verify : prove it runs in a fresh environment.

    Examples
    --------
    ```python
    import tempfile, os

    @ai
    def capital(country: str) -> str:
        """The country's capital city."""
        ...

    folder = os.path.join(tempfile.mkdtemp(), "capital")
    save(capital, folder)
    loaded = load(folder, trust=True)
    loaded("Peru")
    ```
    '''
    root = Path(path).expanduser().resolve()
    manifest = _read_manifest(root)
    if manifest.get("language", "python") != "python":
        return from_manifest(manifest, node=node, saved_id="sha256:" + _sha256((root / MANIFEST).read_bytes()))
    problems = _check_hashes(root, manifest)
    if problems:
        raise LoadRefused(f"{root} does not match what was saved", problems)
    missing, drifted = _check_environment(manifest)
    if missing:
        raise LoadRefused(f"{root} needs packages this environment does not have (install with "
                          f"`pip install -r {root / 'requirements.lock'}`)", missing)
    if not trust:
        code = "\n".join(f"  {root / rel}" for rel in manifest["modules"].values())
        raise PermissionError(f"loading {root} runs its saved Python code:\n{code}\n"
                              f"read it, then load(..., trust=True)")
    if drifted:
        msg = f"{root}: different package versions than when saved: " + "; ".join(drifted)
        warnings.warn(msg, stacklevel=2)
    digest = _sha256(_canonical(manifest.get("hashes", {})).encode())[:10]
    package = f"_functai_saved_{digest}_{next(_counter)}"
    pkg = types.ModuleType(package)
    pkg.__path__ = [str(root / "code")]
    pkg.__package__ = package
    pkg.__file__ = str(root / "code" / "__init__.py")
    _programs[package] = manifest
    _roots[package] = str(root)
    _ids[package] = "sha256:" + _sha256((root / MANIFEST).read_bytes())
    with _import_lock, _no_bytecode():
        sys.modules[package] = pkg
        entry_module, _, entry_name = manifest["entry"].rpartition(":")
        try:
            obj = getattr(importlib.import_module(f"{package}.{_flat(entry_module)}"), entry_name)
            for key, node in manifest["nodes"].items():
                if node["kind"] == "ai" and node["ai"].get("teacher"):
                    _node_object(package, manifest, key)._settings["teacher"] = \
                        _node_object(package, manifest, node["ai"]["teacher"])
                esc = node.get("ai", {}).get("settings", {}).get("escalate_to")
                if isinstance(esc, dict) and esc.get("node"):
                    _node_object(package, manifest, key)._settings["escalate_to"] = \
                        _node_object(package, manifest, esc["node"])
        except BaseException:
            for name in [m for m in sys.modules if m == package or m.startswith(package + ".")]:
                del sys.modules[name]
            raise
    differences = _verify_loaded(package, manifest)
    if differences:
        if check_env == "refuse":
            raise LoadRefused(f"{root} does not behave here as it did when saved "
                              f"(load(..., check_env='warn') to use it anyway)", differences)
        warnings.warn("; ".join(differences), stacklevel=2)
    return obj


@contextlib.contextmanager
def _no_bytecode():
    before = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        yield
    finally:
        sys.dont_write_bytecode = before


# ------------------------------------------------------------------ verify


@dataclasses.dataclass
class Verification:
    """What ``verify`` found.

    What ``verify`` found. ``ok`` means: a fresh environment built from the
    saved requirements alone loaded the program, every AI function rendered
    exactly the requests it rendered when saved, and each recording
    (``save(record=...)``) produced the same result against its recorded replies."""
    ok: bool
    fresh: bool
    problems: List[str]
    log: str = ""

    def __repr__(self) -> str:
        head = ("verified" if self.ok else "NOT verified") + (" in a fresh environment" if self.fresh
                                                                 else " in this environment")
        return head + "".join(f"\n  - {p}" for p in self.problems)

    def __bool__(self) -> bool:
        return self.ok


class _Replay:
    """A client that answers each request with the reply recorded for it."""

    def __init__(self, recorded: Dict[str, Any]):
        self.routes = recorded["routes"]
        self.finals = {v[2]: v for v in self.routes.values()}
        self.replies: Dict[str, List[Dict[str, Any]]] = {}
        self.requests: Dict[str, Dict[str, Any]] = {}
        for run in recorded["recordings"]:
            for ex in run["exchanges"]:
                key = _canonical(ex["request"])
                self.replies.setdefault(key, []).append(ex["response"])
                self.requests[key] = ex["request"]

    def resolve(self, model: str):
        from .models import _Route
        if model in self.finals:
            v = self.finals[model]
        elif model in self.routes and self.routes[model][2] != model:
            raise lm15.UnknownModelError(f"replay: {model!r} was sent as {self.routes[model][2]!r}", model=model)
        elif model in self.routes:
            v = self.routes[model]
        else:
            raise lm15.UnknownModelError(f"replay: the program did not use {model!r} when saved", model=model)
        return _Route(v[0], v[1])

    def complete(self, request: Any) -> Any:
        d = lm15.serde.request_to_dict(request)
        key = _canonical(d)
        queue = self.replies.get(key)
        if not queue:
            raise _ReplayMiss(_closest(d, self.requests.values()))
        reply = queue.pop(0) if len(queue) > 1 else queue[0]
        return lm15.serde.response_from_dict(reply)


class _ReplayMiss(RuntimeError):
    pass


def _closest(request: Dict[str, Any], recorded: Iterable[Dict[str, Any]]) -> str:
    import difflib
    text = json.dumps(request, indent=1, sort_keys=True, ensure_ascii=False).splitlines()
    best, best_ratio = None, -1.0
    for r in recorded:
        other = json.dumps(r, indent=1, sort_keys=True, ensure_ascii=False).splitlines()
        ratio = difflib.SequenceMatcher(None, text, other).ratio()
        if ratio > best_ratio:
            best, best_ratio = other, ratio
    if best is None:
        return "the program sent a request, but none was recorded"
    diff = [ln for ln in difflib.unified_diff(best, text, "recorded", "sent now", n=1, lineterm="")][:30]
    return "the program sent a request it did not send when saved:\n" + "\n".join(diff)


def _replay(root: Path, entry: Any) -> List[str]:
    from .config import configure
    path = root / "recordings.json"
    if not path.exists():
        return []
    recorded = json.loads(path.read_text())
    replay = _Replay(recorded)
    problems = []
    settings = _settings_from_json(recorded.get("settings", {}))
    with configure(**{**settings, "client": replay, "cache_replies": False, "api_retries": 0}):
        for i, run in enumerate(recorded["recordings"]):
            try:
                output = entry(**run["inputs"])
            except _ReplayMiss as exc:
                problems.append(f"recording {i}: {exc}")
                continue
            except Exception as exc:  # noqa: BLE001 — the finding
                hint = f" (declare {exc.name!r} with requires=)" if isinstance(exc, ModuleNotFoundError) else ""
                problems.append(f"recording {i}: the program raised {type(exc).__name__}: {exc}{hint}")
                continue
            now = _output_json(output)
            if _canonical(now) != _canonical(run["output"]):
                problems.append(f"recording {i}: returned {now} but returned {run['output']} when saved")
    return problems


def verify_here(path: "str | os.PathLike[str]") -> Verification:
    try:
        entry = load(path, trust=True, check_env="refuse")
    except LoadRefused as exc:
        return Verification(False, False, list(exc.problems))
    except Exception as exc:  # noqa: BLE001 — anything that stops loading is the finding
        hint = ""
        if isinstance(exc, ModuleNotFoundError):
            hint = f" (the saved requirements do not provide {exc.name!r}: declare it with requires=)"
        return Verification(False, False, [f"loading failed: {type(exc).__name__}: {exc}{hint}"])
    problems = _replay(Path(path).expanduser().resolve(), entry)
    return Verification(not problems, False, problems)


def verify(path: "str | os.PathLike[str]", *, trust: bool = False, fresh: bool = True,
           python: Optional[str] = None, timeout: float = 900) -> Verification:
    '''Prove a saved program runs somewhere else, without calling a model.

    Builds a new environment with ``uv`` from the folder's lock file alone,
    loads the program there from an empty directory (so nothing from your
    project can leak in), then checks that every AI function renders
    byte-identical requests, and that each recording (``save(record=...)``)
    replays to the same result.

    Parameters
    ----------
    path : str or path
        The saved folder.
    trust : bool
        Must be True: verifying runs the saved code.
    fresh : bool
        Build a new environment (default). ``False`` checks in this one:
        quicker, and blind to packages installed here but not declared.
    python : str, optional
        The Python version or interpreter for the new environment.
    timeout : float
        Seconds before giving up.

    Returns
    -------
    Verification
        Displays what was checked; ``.ok`` is True when everything matched.

    See Also
    --------
    save : write the folder.
    load : use it.

    Examples
    --------
    ```python
    import tempfile, os

    @ai
    def capital(country: str) -> str:
        """The country's capital city."""
        ...

    folder = os.path.join(tempfile.mkdtemp(), "capital")
    save(capital, folder, record=[{"country": "Kenya"}])
    verify(folder, trust=True, fresh=False)
    ```
    '''
    root = Path(path).expanduser().resolve()
    if not trust:
        raise PermissionError(f"verify loads {root}'s saved code; read it, then verify(..., trust=True)")
    if not fresh:
        return verify_here(root)
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("verify builds a fresh environment with uv, which is not installed here; install uv, "
                           "or verify(..., fresh=False) to check in this environment (weaker)")
    manifest = _read_manifest(root)
    version = python or ".".join(manifest["python"].split(".")[:2])
    with tempfile.TemporaryDirectory(prefix="functai-verify-") as tmp:
        env_dir = Path(tmp) / "venv"
        log: List[str] = []

        def run(cmd: List[str]) -> subprocess.CompletedProcess:
            done = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True, timeout=timeout,
                                  env={k: v for k, v in os.environ.items() if not k.startswith(("PYTHON", "VIRTUAL_ENV"))})
            log.append("$ " + " ".join(cmd) + "\n" + done.stdout + done.stderr)
            return done

        # Packages installed from a folder are rebuilt from it: uv's cache does not
        # notice source edits there, and verify must test the code as it is now.
        local = [line.split(" @ ")[0].strip() for line in (root / "requirements.lock").read_text().splitlines()
                 if " @ file://" in line]
        refresh = [arg for name in local for arg in ("--refresh-package", name)]
        steps = [[uv, "venv", "-q", "--python", version, str(env_dir)],
                 # --no-sources: a package built from a folder must not pull its own development
                 # sources (tool.uv.sources); the lock alone decides what is installed.
                 [uv, "pip", "install", "-q", "--no-sources", *refresh, "--python", str(env_dir / "bin" / "python"),
                  "-r", str(root / "requirements.lock")]]
        for cmd in steps:
            if run(cmd).returncode != 0:
                return Verification(False, True, [f"building the environment failed: {log[-1].strip()[-800:]}"],
                                    "\n".join(log))
        done = run([str(env_dir / "bin" / "python"), "-I", "-m", "functai", "verify", str(root)])
        try:
            result = json.loads(done.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            return Verification(False, True, [f"the check did not run: {(done.stderr or done.stdout)[-800:]}"],
                                "\n".join(log))
        return Verification(result["ok"], True, result["problems"], "\n".join(log))


def main(argv: List[str]) -> int:
    """``python -m functai verify <folder>``: load a saved program here and compare
    its fingerprints; prints one JSON line (what ``verify`` reads in a fresh
    environment)."""
    if len(argv) == 2 and argv[0] == "verify":
        v = verify_here(argv[1])
        print(json.dumps({"ok": v.ok, "problems": v.problems}))
        return 0 if v.ok else 1
    print("usage: python -m functai verify <saved program folder>", file=sys.stderr)
    return 2


# ------------------------------------------------------------------ another language's AI functions, from data


def _manifest_data(source: Any) -> Tuple[Dict[str, Any], Optional[str]]:
    """(the manifest, the saved id) of a folder, a functai.json, or a manifest already read."""
    if isinstance(source, dict):
        return source, None
    p = Path(os.fspath(source)).expanduser()
    file = p / MANIFEST if p.is_dir() else p
    try:
        raw = file.read_bytes()
    except OSError as exc:
        raise LoadRefused(f"{p} is not a saved functai program", [str(exc)], "saved-malformed") from None
    try:
        manifest = json.loads(raw)
    except ValueError as exc:
        raise LoadRefused(f"{file} is not JSON", [str(exc)], "saved-malformed") from None
    return manifest, "sha256:" + _sha256(raw)


def _refuse(code: str, message: str) -> LoadRefused:
    return LoadRefused(f"[{code}] {message}", [], code)


def _check_form(manifest: Any) -> Dict[str, Any]:
    """The checks before any node is read (saved.md, step 1): a format this
    reader knows, then the manifest's form where a loader reads it."""
    if not isinstance(manifest, dict):
        raise _refuse("saved-malformed", "functai.json is not a JSON object")
    if manifest.get("functai_saved") != FORMAT:
        raise _refuse("saved-format", f"functai.json is format {manifest.get('functai_saved')!r}; this reader "
                                      f"reads format {FORMAT}")
    nodes = manifest.get("nodes")
    if not isinstance(manifest.get("entry"), str) or not isinstance(nodes, dict):
        raise _refuse("saved-malformed", "functai.json needs an entry and nodes")
    if manifest["entry"] not in nodes:
        raise _refuse("saved-malformed", f"its entry {manifest['entry']!r} is not one of its nodes")
    for key, n in nodes.items():
        if not isinstance(n, dict) or n.get("kind") not in ("ai", "module", "function", "class") \
                or not isinstance(n.get("name"), str) or not isinstance(n.get("module"), str):
            raise _refuse("saved-malformed", f"node {key!r} is not a node (kind, module and name)")
        if "interface" in n and not _interface_form(n["interface"]):
            raise _refuse("saved-malformed", f"node {key!r}: its interface is not of the interface's form")
        if n["kind"] == "ai":
            ai = n.get("ai")
            if not isinstance(ai, dict) or not isinstance(ai.get("signature"), dict) \
                    or not isinstance(ai.get("settings", {}), dict):
                raise _refuse("saved-malformed", f"node {key!r} has no \"ai\" entry with its signature")
    return manifest


def _interface_form(iface: Any) -> bool:
    """What interface.schema.json checks (the rest is programs.md's, interface-malformed)."""
    from .interface import NAME
    if not isinstance(iface, dict) or set(iface) - {"description", "inputs", "outputs"}:
        return False
    if not isinstance(iface.get("description"), str) or not isinstance(iface.get("inputs"), list) \
            or not isinstance(iface.get("outputs"), list) or not iface["outputs"]:
        return False
    for direction, keys in (("inputs", {"name", "shape", "desc", "type", "opaque", "optional"}),
                            ("outputs", {"name", "shape", "desc", "type", "opaque"})):
        for f in iface[direction]:
            if not isinstance(f, dict) or set(f) - keys or not isinstance(f.get("name"), str) \
                    or not NAME.fullmatch(f["name"]) or not isinstance(f.get("shape"), dict):
                return False
            if any(k in f and f[k] is not True for k in ("opaque", "optional")):
                return False
            if f.get("opaque") and f["shape"]:
                return False
            if any(k in f and not isinstance(f[k], str) for k in ("desc", "type")):
                return False
    return True


def _plain_fields(node: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [f for f in node["ai"]["signature"].get("fields") or [] if (f.get("purpose") or "plain") == "plain"]


def _checked_interface(key: str, node: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """A node's interface, refused as programs.md refuses it, and for an AI
    node, refused ``saved-differs`` when it does not describe the data its
    signature takes and gives (its words may differ)."""
    from . import interface as _interface
    iface = node.get("interface")
    if iface is None:
        return None
    try:
        _interface.check(iface, ai=node["kind"] == "ai", program=key)
    except _interface.InterfaceError as exc:
        raise LoadRefused(str(exc), [], "interface-malformed") from None
    if node["kind"] == "ai":
        fields = _plain_fields(node)
        theirs = {"inputs": [f for f in fields if f.get("direction") == "input"],
                  "outputs": [f for f in fields if f.get("direction") == "output"]}
        if _interface.signature(iface) != _interface.signature(theirs):
            raise _refuse("saved-differs", f"{key}: its interface says it takes or gives other data than its "
                                           f"signature does")
    return iface


def describe(source: Any, node: Optional[str] = None) -> Dict[str, Any]:
    """What a saved program takes and gives, without loading it or running
    anything: the node's interface (contract/saved.md, *Describing without
    loading*), whatever language wrote the folder.

    ``source``: a saved folder, its ``functai.json``, or the manifest as a
    dict; ``node``: a program's key (``"module:name"``), the entry by default.
    Raises ``LoadRefused`` (``.code``: ``saved-format``, ``saved-malformed``,
    ``saved-not-ai``, ``interface-malformed``, ``saved-differs``,
    ``saved-no-interface``). A node written before interfaces were saved: an
    AI function is described by its signature, with its instruction (the
    prompt) as the description; a module is not known."""
    manifest, _id = _manifest_data(source)
    m = _check_form(manifest)
    key = node or m["entry"]
    n = m["nodes"].get(key)
    if n is None:
        raise _refuse("saved-malformed", f"functai.json has no node {key!r}")
    if n["kind"] not in ("ai", "module"):
        raise _refuse("saved-not-ai", f"{key} is plain code ({n['kind']}), not a program")
    iface = _checked_interface(key, n)
    if iface is not None:
        return copy.deepcopy(iface)
    if n["kind"] == "module":
        raise _refuse("saved-no-interface", f"{key} was saved before modules' interfaces were: what it takes and "
                                            f"gives is not known")
    sig = n["ai"]["signature"]

    def field(f: Dict[str, Any]) -> Dict[str, Any]:
        out = {"name": f["name"], "shape": f["shape"]}
        if f.get("desc"):
            out["desc"] = f["desc"]
        if isinstance(f.get("type"), str):
            out["type"] = f["type"]
        return out
    plain = _plain_fields(n)
    return {"description": sig.get("instructions") or "",
            "inputs": [field(f) for f in plain if f.get("direction") == "input"],
            "outputs": [field(f) for f in plain if f.get("direction") == "output"]}


def from_manifest(source: Any, *, node: Optional[str] = None, saved_id: Optional[str] = None):
    """An AI function from a saved manifest's data alone, whatever language
    wrote it (contract/saved.md, *Loading an AI function in another
    language*): no code runs.

    Refuses (``LoadRefused``, with ``.code``) what cannot be run from data: a
    module (``saved-not-ai``), code of its own beside the model
    (``saved-code``), tools (``saved-tools``), a baked model or another
    program as a setting (``saved-model``); and a function that would not
    send what was saved (``saved-differs``) or whose interface is refused.
    Its optional inputs, and the defaults they are sent with, come from the
    node's interface. Values come back as JSON (dicts, lists), not the saving
    language's types."""
    manifest, file_id = _manifest_data(source)
    m = _check_form(manifest)
    language = m.get("language", "python")
    key = node or m["entry"]
    n = m["nodes"].get(key)
    if n is None:
        raise _refuse("saved-malformed", f"functai.json has no node {key!r}")
    if n["kind"] != "ai":
        what = "a module" if n["kind"] == "module" else f"a {n['kind']}"
        raise _refuse("saved-not-ai", f"{key} is {what}: code in {language}, which is run only where it can be; its "
                                      f"AI functions load by key")
    data = n["ai"]
    if "body" not in data or data["body"] is not None:
        raise _refuse("saved-code", f"{key} runs code of its own beside the model (written in {language}); only "
                                    f"{language} can run it")
    if data.get("tools"):
        raise _refuse("saved-tools", f"{key} has tools ({data['tools']}): a tool is code")
    for k, v in (data.get("settings") or {}).items():
        if isinstance(v, dict) and ("baked" in v or "node" in v):
            raise _refuse("saved-model", f"{key}: its setting {k} is {v}, which this loader cannot reach")
    fn = _LoadedAI(key, n, saved_id or file_id)
    iface = _checked_interface(key, n)
    fn._saved_interface = iface
    requests = (data.get("fingerprints") or {}).get("requests") or []
    spec, settings = fn._spec(), fn._effective()
    plan, past = probe_plan(fn, spec, settings)
    for i, probe in enumerate(data.get("probes") or []):
        try:
            got = request_fingerprint(probe_request(fn, spec, plan, past, probe))
        except lmcc.Refusal as exc:
            got = f"refused:{exc.code}"
        if i < len(requests) and got != requests[i]:
            raise _refuse("saved-differs", f"{key}: for probe {i} it would send {got}, but {requests[i]} was saved")
    if isinstance(data.get("version"), str) and data["version"] != fn.version:
        raise _refuse("saved-differs", f"{key}: its version here is {fn.version}, but {data['version']} was saved")
    return fn


def _loaded_function(name: str, inputs: List[Dict[str, Any]]):
    """A Python function whose parameters are the inputs (an optional one with
    its default), for binding a call's arguments; the model writes its body."""
    import keyword
    params = []
    defaults: Dict[str, Any] = {}
    for f in inputs:
        pname = f["name"]
        if keyword.iskeyword(pname):
            raise _refuse("saved-malformed", f"input {pname!r} is a Python keyword")
        if f.get("optional"):
            defaults[pname] = copy.deepcopy((f.get("shape") or {}).get("default"))
            params.append(f"{pname}=__defaults__[{pname!r}]")
        else:
            params.append(pname)
    ident = name if name.isidentifier() and not keyword.iskeyword(name) else "loaded"
    source = f"def {ident}({', '.join(params)}):\n    ...\n"
    namespace: Dict[str, Any] = {"__defaults__": defaults}
    exec(compile(source, f"<functai loaded {name}>", "exec"), namespace)
    return namespace[ident]


def _loaded_class():
    from .core import FunctAIFunc

    class LoadedAIFunc(FunctAIFunc):
        """An AI function built from a saved manifest's data: its signature is
        the saved one, its body the model's."""

        _loaded = True

        def __init__(self, key: str, node: Dict[str, Any], saved_id: Optional[str]):
            from .config import CONFIG_FIELDS
            data = node["ai"]
            self._saved_core = lmcc.signature_from_dict(data["signature"])
            self._saved_node = node
            self._saved_interface = None
            self._saved_id = saved_id
            iface = node.get("interface")
            plain = [f for f in self._saved_core.fields if (f.purpose or "plain") == "plain"]
            ins = [{"name": f.name, "shape": f.shape} for f in plain if f.direction == "input"]
            if isinstance(iface, dict):
                optional = {f["name"]: f for f in iface.get("inputs") or [] if isinstance(f, dict)
                            and f.get("optional")}
                ins = [optional.get(f["name"], f) for f in ins]
            fn = _loaded_function(node["name"], ins)
            fn.__module__ = node["module"]
            fn.__qualname__ = fn.__name__ = node["name"]
            settings: Dict[str, Any] = {}
            for k, v in (data.get("settings") or {}).items():
                if v is None or k in NOT_SAVED:
                    continue
                if k == "lm" and not isinstance(v, str):
                    continue
                settings[k] = v
            if data.get("config"):
                config, default = lm15.serde.config_from_dict(data["config"]), lm15.Config()
                for f in dataclasses.fields(config):
                    v = getattr(config, f.name)
                    if f.name in CONFIG_FIELDS and v != getattr(default, f.name):
                        settings[f.name] = v
            if any(f.purpose == "reasoning" for f in self._saved_core.fields):
                settings["module"] = "cot"
            super().__init__(fn, template=data.get("template"), **settings)
            from .core import ProgramState
            self.load_state(ProgramState.from_dict(data.get("state") or {}))

        def _check_definition(self) -> None:
            self._check_log_content()      # its interface is checked by from_manifest (saved.md, step 6)

        def _spec(self, instructions: Optional[str] = None):
            from .signature import Spec
            if instructions is None:
                instructions = self._current_state().instructions
            key = ("loaded", instructions)
            spec = self._spec_cache.get(key)
            if spec is None:
                core = self._saved_core
                if instructions is not None:
                    core = lmcc.SignatureCore(instructions.strip(), list(core.fields))
                outputs = [f.name for f in core.outputs if (f.purpose or "plain") == "plain"]
                spec = Spec(signature=core, main=outputs[-1], outputs=tuple(outputs),
                            annotations={f.name: None for f in core.outputs},
                            params=tuple(f.name for f in core.inputs if (f.purpose or "plain") == "plain"),
                            reasoning=any(f.purpose == "reasoning" for f in core.outputs), tools=False)
                self._spec_cache[key] = spec
            return spec

        @property
        def interface(self) -> Dict[str, Any]:
            if self._saved_interface is not None:
                return copy.deepcopy(self._saved_interface)
            iface = super().interface
            iface["description"] = self._saved_core.instructions
            return iface

    return LoadedAIFunc


_LOADED_CLASS: List[Any] = []


def _LoadedAI(key: str, node: Dict[str, Any], saved_id: Optional[str]):
    if not _LOADED_CLASS:
        _LOADED_CLASS.append(_loaded_class())
    return _LOADED_CLASS[0](key, node, saved_id)


__all__ = ["save", "load", "verify", "file", "describe", "from_manifest", "LoadRefused", "Verification"]
