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
NOT_SAVED = ("client", "api_key", "auth", "optimizer", "teacher", "router")


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


def _settings_json(fn, where: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """(settings, lm15 config) as JSON, for one AI function."""
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


def _fingerprints(fn, probes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """What the program sends, as hashes: the signature, and the exact request
    for each probe (instruction, layout, demos, tools) under fixed capabilities."""
    from . import adapters, engine
    spec = fn._spec()
    settings = fn._effective()
    plan = adapters.bind(fn._layout(settings), spec.signature, PROBE_CAPABILITIES, "probe")
    past = fn._past(plan, spec, {**settings, "stateful": False})
    renders = []
    for inputs in probes:
        values = engine.prepare_inputs(spec, inputs)
        if spec.tools:
            values["tools"] = list(fn._tool_specs)
        try:
            request = plan.render(plan.turn(values), turns=past).request("probe")
            renders.append("sha256:" + _sha256(_canonical(request).encode()))
        except lmcc.Refusal as exc:
            renders.append(f"refused:{exc.code}")
    return {"signature": lmcc.signature_fingerprint(spec.signature), "requests": renders}


def _manifest(report, examples, allowed) -> Dict[str, Any]:
    from .graph import state_to_json
    nodes: Dict[str, Any] = {}
    for key, n in report.nodes.items():
        entry: Dict[str, Any] = {"kind": n.kind, "module": n.module, "name": n.name}
        if n.kind == "ai":
            fn = n.obj
            settings, config = _settings_json(fn, key)
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
            }
        elif n.kind == "module":
            entry["module_program"] = {"call_defaults": lmcc.turn.to_json(dict(n.obj._opt_call_defaults)),
                                       "requires": list(n.requires)}
        nodes[key] = entry
    return {
        "functai_saved": FORMAT,
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
         record: Iterable[Dict[str, Any]] = (), overwrite: bool = False):
    """Save ``program`` (an @ai function, an @module, or a Python function that
    calls them) with its code, data files and pinned requirements, to a folder.

    Refuses (``functai.graph.Refused``, with the report) while ``check`` finds
    errors; ``allow=["hidden-state", ...]`` records a deliberate exception.
    ``include`` / ``requires``: as for ``check``. ``examples``: inputs of the
    program (an AI function) to fingerprint, besides its demos.

    ``record``: inputs to run the program on now, for real (this calls the
    model); every model reply is recorded, and ``verify`` replays the recordings
    to check that the saved program, tools and helpers included, produces the
    same results in a fresh environment, without calling a model. Writes a new folder whole or
    not at all. Returns the report."""
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
    manifest = _manifest(report, examples, allowed)
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
    """The saved program cannot be loaded as saved; ``.problems`` says why."""

    def __init__(self, message: str, problems: List[str]):
        super().__init__(message + "".join(f"\n  - {p}" for p in problems))
        self.problems = problems


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


def _rebuild_module(fn: Any, key: str):
    from .module import FunctAIModule
    manifest, _package = _manifest_of(fn)
    data = manifest["nodes"][key].get("module_program", {})
    m = FunctAIModule(fn, requires=data.get("requires", ()))
    m._opt_call_defaults = dict(data.get("call_defaults") or {})
    return m


def _verify_loaded(package: str, manifest: Dict[str, Any]) -> List[str]:
    """Fingerprints recomputed here against the saved ones."""
    problems = []
    for key, node in manifest["nodes"].items():
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
    return problems


def load(path: "str | os.PathLike[str]", *, trust: bool = False, check_env: str = "refuse"):
    """The saved program at ``path``, ready to call.

    Checks first, runs nothing: the files' hashes, the requirements this
    environment has, and (after loading) that every AI function renders the same
    requests as when it was saved. ``trust=True`` is required because loading
    runs the saved code. ``check_env="warn"`` loads despite version or request
    differences, with warnings; missing packages always refuse."""
    root = Path(path).expanduser().resolve()
    manifest = _read_manifest(root)
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
    with _import_lock, _no_bytecode():
        sys.modules[package] = pkg
        entry_module, _, entry_name = manifest["entry"].rpartition(":")
        try:
            obj = getattr(importlib.import_module(f"{package}.{_flat(entry_module)}"), entry_name)
            for key, node in manifest["nodes"].items():
                if node["kind"] == "ai" and node["ai"].get("teacher"):
                    _node_object(package, manifest, key)._settings["teacher"] = \
                        _node_object(package, manifest, node["ai"]["teacher"])
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
    """What ``verify`` found. ``ok`` means: a fresh environment built from the
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
    """Prove the saved program is self-contained: build a new environment with
    ``uv`` from ``requirements.lock`` alone, load the program there (in isolated
    mode, from an empty directory, so nothing from your project leaks in), and
    compare every AI function's rendered requests with the saved fingerprints.
    No model is called. ``fresh=False`` checks in this environment instead
    (weaker: packages installed here but not declared go unnoticed)."""
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
                 [uv, "pip", "install", "-q", *refresh, "--python", str(env_dir / "bin" / "python"), "-r",
                  str(root / "requirements.lock")]]
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


__all__ = ["save", "load", "verify", "file", "LoadRefused", "Verification"]
