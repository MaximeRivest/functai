"""Files as inputs: an `Attachments <https://github.com/maximerivest/attachments>`_
result as an AI function's input type (design/13-files-and-attachments.md).

    from attachments import Artifacts, att

    @ai
    def summarize(report: Artifacts) -> str:
        "The report in three sentences."

    summarize(att("report.pdf[images: true]"))    # its text and its page images, page by page

``att()`` turns a PDF, a Word or Excel file, slides, a web page, a folder, a
repository, a picture into ``Artifacts``: text, images and metadata. Here it
becomes an lmcc type binding (lmcc 0.8.6, D-61), made once, the first time
it is needed:

- **What the model sees**: ``Artifacts.parts()``, Attachments' own
  provider-neutral parts, written at the input's place in the prompt: each
  page's text followed by that page's images, as image parts the model sees
  (not base64 in the text). File names are shown, as ``att()`` prints them
  (Attachments' default): a folder's files are told apart by them. To keep a
  name from reaching the model (a classifier whose files are named after their
  answers), give the picture as an ``lm15`` ``ImagePart`` instead: it has none.
- **Its JSON form**, for the call log, the reply cache's key, replays,
  conversations and served calls: Attachments' wire form (``to_wire``,
  ``from_wire``: images as base64), so a value logged in one process is the
  same ``Artifacts`` when it is replayed in another.
- **A file that failed** (``att()`` never raises: a missing file is an
  artifact with an error and no text) refuses the call before any request,
  naming the source and why, instead of reaching the model as an empty
  document.

Binding is lazy: importing functai never imports attachments. The binding is
made when ``attachments`` is imported (before or after functai), and when a
function that reads ``Artifacts`` is planned, saved programs included.
"""
from __future__ import annotations

import importlib.abc
import importlib.util
import sys
import threading
from typing import Any, Dict, List, Optional

#: The type name lmcc gives the input (a field's ``type``): what a loaded or
#: served program is recognised by.
TYPE_NAME = "Artifacts"

#: The input's shape: a list of Attachments artifacts in their wire form
#: (Attachments' ``spec/artifact.schema.json`` describes each; it is checked
#: by ``Artifacts.from_wire``, which knows it, rather than copied here, where
#: it would fall behind Attachments' additive changes).
SHAPE: Dict[str, Any] = {
    "type": "array",
    "items": {"type": "object"},
    "description": "Attachments artifacts in their wire form (attachments' spec/artifact.schema.json): "
                   "each {text, images: [{name, mimetype, bytes_b64, page?}], audio, video, meta}",
}

_lock = threading.Lock()
_installed: set = set()


class FileError(ValueError):
    """An ``Artifacts`` input holds a file that could not be read."""


def _artifacts(value: Any) -> Any:
    from attachments import Artifacts
    if isinstance(value, Artifacts):
        return value
    if isinstance(value, list):                 # its wire form: a replay, a served call, another language
        return Artifacts.from_wire(value)
    raise TypeError(f"an Artifacts input takes att(...)'s result or its wire form, not {type(value).__name__}")


def describe_errors(errors: List[Dict[str, Any]]) -> str:
    """``report.pdf: unpack-error — …; scan.png: parse-error — …`` (the first five)."""
    shown = [f"{e.get('source')}: {e.get('code')} — {e.get('message')}" for e in errors[:5]]
    more = f"; and {len(errors) - 5} more" if len(errors) > 5 else ""
    return "; ".join(shown) + more


def check_inputs(inputs: Dict[str, Any], *, program: str = "") -> None:
    """Refuse an ``Artifacts`` input holding a file that could not be read
    (``interface-input``, naming the input), before any request."""
    if "attachments" not in sys.modules:
        return
    from attachments import Artifacts
    for name, value in inputs.items():
        if isinstance(value, Artifacts) and value.errors:
            from .errors import InterfaceError
            raise InterfaceError("interface-input", name,
                                 f"{program + ': ' if program else ''}input {name!r}: {len(value.errors)} of its files "
                                 f"could not be read, so the model would get an empty or partial document: "
                                 f"{describe_errors(value.errors)}. Do what each says "
                                 f"(att(...).errors lists them), or leave that file out")


def _write(value: Any) -> List[Dict[str, Any]]:
    """The parts the model is shown: Attachments' own, page by page."""
    artifacts = _artifacts(value)
    if artifacts.errors:
        raise FileError(f"{len(artifacts.errors)} of the files could not be read, so the model would get an empty "
                        f"or partial document: {describe_errors(artifacts.errors)}. Do what each says "
                        f"(att(...).errors lists them), or leave that file out")
    parts = artifacts.parts()
    return parts or [{"type": "text", "text": "(no content)"}]


def install(registry: Any = None) -> bool:
    """Bind ``attachments.Artifacts`` in lmcc's registry (the default one, or
    ``registry``). Returns False when attachments is not installed. Binding
    again is harmless."""
    try:
        from attachments import Artifacts
    except ImportError:
        return False
    import lmcc
    key = id(registry)
    with _lock:
        if key in _installed:
            return True
        reg = registry if registry is not None else lmcc.default_registry
        reg.format(Artifacts, shape=SHAPE, write=_write, writes="parts", direction="in",
                   to_json=Artifacts.to_wire, from_json=Artifacts.from_wire)
        _installed.add(key)
    return True


def ensure_for(type_names: Any) -> None:
    """Make the binding when a signature reads ``Artifacts`` (a loaded or served
    program, defined where attachments was never imported)."""
    if TYPE_NAME in type_names and not _installed:
        if not install():
            raise ImportError("this function reads files (an Artifacts input) and attachments is not installed: "
                              "pip install 'functai[files]'")


# ------------------------------------------------------------------ binding when attachments is imported


class _Watch(importlib.abc.MetaPathFinder):
    """Binds Artifacts once the ``attachments`` package has been imported, by
    whoever imports it, without importing it otherwise."""

    def find_spec(self, name: str, path: Any = None, target: Any = None) -> Optional[importlib.machinery.ModuleSpec]:
        if name != "attachments":
            return None
        sys.meta_path.remove(self)                   # find the real spec, once
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        loader = spec.loader
        run = loader.exec_module

        def exec_module(module: Any) -> None:
            run(module)
            _unwatch()
            install()

        loader.exec_module = exec_module             # type: ignore[method-assign]
        return spec


_watch: Optional[_Watch] = None


def _unwatch() -> None:
    global _watch
    if _watch is not None and _watch in sys.meta_path:
        sys.meta_path.remove(_watch)
    _watch = None


def watch() -> None:
    """Called once by ``import functai``: bind now if attachments is already
    imported, else when it is."""
    global _watch
    if "attachments" in sys.modules:
        install()
        return
    if _watch is None:
        _watch = _Watch()
        sys.meta_path.insert(0, _watch)


__all__ = ["install", "ensure_for", "watch", "FileError", "SHAPE", "TYPE_NAME"]
