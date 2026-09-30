"""The reply cache: a model's reply to a request, reused for the same request
(contract/replies.md).

    functai.configure(cache_replies="disk")     # kept across runs, in one SQLite file
    functai.configure(cache_replies=True)       # in this process's memory only
    fn.using(replicate=2)                       # a third, independent answer to the same request

A reply is kept only once it was read (lmcc read it, and its values fit
their types), so an interrupted run leaves nothing half-written. One flight
per key: while one caller asks the model, another caller of the same
request waits for its reply, in this process (a lock per key) and across
processes (a claim with a lease, in the same SQLite file).

A call whose ``log_content`` drops any field is never written to disk: a
disk cache holds whole requests and replies, which is what ``log_content``
forbids keeping.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import socket
import sqlite3
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional

FORMAT = 1
_LEASE = 120.0                       # seconds a claim holds a key against other processes
_OWNER = f"{socket.gethostname()}:{os.getpid()}:{secrets.token_hex(3)}"


def _owner() -> str:
    global _OWNER
    pid = os.getpid()
    if f":{pid}:" not in _OWNER:                     # a forked child is another owner
        _OWNER = f"{socket.gethostname()}:{pid}:{secrets.token_hex(3)}"
    return _OWNER


def key(request: Any, replicate: int = 0) -> str:
    """The cache key of a request: ``sha256:`` of the canonical JSON of
    ``{"functai_reply": 1, "request": <lm15 canonical request>, "replicate": n}``."""
    from lm15.serde import request_to_dict
    from .calllog import canonical
    doc = {"functai_reply": FORMAT, "request": request_to_dict(request), "replicate": int(replicate or 0)}
    return "sha256:" + hashlib.sha256(canonical(doc).encode("utf-8")).hexdigest()


def _dump(response: Any) -> str:
    from lm15.serde import response_to_dict
    from .calllog import canonical
    return canonical(response_to_dict(response))


def _load(text: str) -> Any:
    from lm15.serde import response_from_dict
    return response_from_dict(json.loads(text))


# ------------------------------------------------------------------ stores


class _Keys:
    """One flight per key in this process: a condition, and who holds each key."""

    def __init__(self) -> None:
        self.cond = threading.Condition()
        self.held: Dict[str, int] = {}               # key → thread id holding it

    def acquire(self, k: str, cancelled: Any = None) -> None:
        me = threading.get_ident()
        with self.cond:
            while k in self.held and self.held[k] != me:
                if cancelled is not None:
                    cancelled()
                self.cond.wait(0.1)
            self.held[k] = me

    def release(self, k: str) -> None:
        with self.cond:
            if self.held.get(k) == threading.get_ident():
                del self.held[k]
            self.cond.notify_all()


class MemoryReplies:
    """Replies kept in this process's memory, at most ``capacity`` (the least
    recently used go first)."""

    durable = False

    def __init__(self, capacity: int = 20_000):
        self.capacity = capacity
        self._data: "OrderedDict[str, Any]" = OrderedDict()
        self._lock = threading.Lock()
        self._keys = _Keys()

    def get(self, k: str) -> Any:
        with self._lock:
            hit = self._data.get(k)
            if hit is not None:
                self._data.move_to_end(k)
            return hit

    def put(self, k: str, response: Any) -> None:
        with self._lock:
            self._data[k] = response
            self._data.move_to_end(k)
            while len(self._data) > self.capacity:
                self._data.popitem(last=False)

    def discard(self, k: str) -> None:
        with self._lock:
            self._data.pop(k, None)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        return len(self._data)

    # one flight per key
    def claim(self, k: str, cancelled: Any = None) -> Any:
        self._keys.acquire(k, cancelled)
        return self.get(k)

    def unclaim(self, k: str) -> None:
        self._keys.release(k)

    def __repr__(self) -> str:
        return f"<MemoryReplies: {len(self._data)} replies>"


def default_path() -> Path:
    """Where ``cache_replies="disk"`` keeps replies: the user's cache folder."""
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    elif os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return base / "functai" / "replies.sqlite"


_SCHEMA = """
CREATE TABLE IF NOT EXISTS replies (
    key TEXT PRIMARY KEY,
    format INTEGER NOT NULL,
    created TEXT NOT NULL,
    model TEXT,
    response TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS claims (
    key TEXT PRIMARY KEY,
    owner TEXT NOT NULL,
    until REAL NOT NULL
);
"""


class DiskReplies:
    """Replies kept in one SQLite file (contract/replies.md), shared by every
    process and thread that opens it: writes are transactions, so nothing is
    half-written; a claim with a lease makes one flight per key across
    processes. The file and its folder are readable by their owner only."""

    durable = True

    def __init__(self, path: "str | os.PathLike[str] | None" = None, *, lease: float = _LEASE):
        p = Path(os.fspath(path)).expanduser() if path is not None else default_path()
        if p.suffix not in (".sqlite", ".sqlite3", ".db"):
            p = p / "replies.sqlite"
        self.path = p.absolute()
        self.lease = float(lease)
        self._local = threading.local()
        self._keys = _Keys()
        self._pid = os.getpid()
        os.makedirs(self.path.parent, mode=0o700, exist_ok=True)
        new = not self.path.exists()
        with self._db() as db:
            db.executescript(_SCHEMA)
        if new:
            try:
                os.chmod(self.path, 0o600)
            except OSError:
                pass

    def _db(self) -> sqlite3.Connection:
        db = getattr(self._local, "db", None)
        if db is None or self._pid != os.getpid():
            self._pid = os.getpid()
            db = sqlite3.connect(self.path, timeout=30.0, isolation_level=None, check_same_thread=False)
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA synchronous=NORMAL")
            self._local.db = db
        return db

    def get(self, k: str) -> Any:
        row = self._db().execute("SELECT response, format FROM replies WHERE key = ?", (k,)).fetchone()
        if row is None or row[1] != FORMAT:
            return None
        try:
            return _load(row[0])
        except Exception:  # noqa: BLE001 — a row this reader cannot read is no reply
            return None

    def put(self, k: str, response: Any) -> None:
        from .calllog import _iso
        db = self._db()
        db.execute("BEGIN IMMEDIATE")
        try:
            db.execute("INSERT OR REPLACE INTO replies (key, format, created, model, response) VALUES (?, ?, ?, ?, ?)",
                       (k, FORMAT, _iso(time.time()), getattr(response, "model", None), _dump(response)))
            db.execute("DELETE FROM claims WHERE key = ? AND owner = ?", (k, _owner()))
            db.execute("COMMIT")
        except BaseException:
            db.execute("ROLLBACK")
            raise

    def discard(self, k: str) -> None:
        self._db().execute("DELETE FROM replies WHERE key = ?", (k,))

    def clear(self) -> None:
        db = self._db()
        db.execute("DELETE FROM replies")
        db.execute("DELETE FROM claims")

    def __len__(self) -> int:
        return int(self._db().execute("SELECT COUNT(*) FROM replies").fetchone()[0])

    def claim(self, k: str, cancelled: Any = None) -> Any:
        """The reply to ``k`` if one is kept; else take the key (one flight):
        wait while another thread or process holds it, then take it."""
        self._keys.acquire(k, cancelled)
        pause = 0.05
        try:
            while True:
                hit = self.get(k)
                if hit is not None:
                    return hit
                db = self._db()
                now = time.time()
                db.execute("BEGIN IMMEDIATE")
                try:
                    row = db.execute("SELECT owner, until FROM claims WHERE key = ?", (k,)).fetchone()
                    mine = row is None or row[0] == _owner() or row[1] < now
                    if mine:
                        db.execute("INSERT OR REPLACE INTO claims (key, owner, until) VALUES (?, ?, ?)",
                                   (k, _owner(), now + self.lease))
                    db.execute("COMMIT")
                except BaseException:
                    db.execute("ROLLBACK")
                    raise
                if mine:
                    return None
                if cancelled is not None:
                    cancelled()
                time.sleep(pause)
                pause = min(pause * 2, 0.5)
        except BaseException:
            self._keys.release(k)
            raise

    def unclaim(self, k: str) -> None:
        try:
            self._db().execute("DELETE FROM claims WHERE key = ? AND owner = ?", (k, _owner()))
        finally:
            self._keys.release(k)

    def __repr__(self) -> str:
        return f"<DiskReplies {self.path}>"


MEMORY = MemoryReplies()
_disks: Dict[str, DiskReplies] = {}
_disks_lock = threading.Lock()


def store_for(setting: Any) -> Any:
    """The store a ``cache_replies`` setting names, or None when off."""
    if setting is None or setting is False:
        return None
    if setting is True:
        return MEMORY
    if isinstance(setting, str) and setting.strip().lower() == "memory":
        return MEMORY
    if isinstance(setting, (str, os.PathLike)):
        path = default_path() if isinstance(setting, str) and setting.strip().lower() == "disk" else setting
        where = str(Path(os.fspath(path)).expanduser().absolute())
        with _disks_lock:
            store = _disks.get(where)
            if store is None:
                store = _disks[where] = DiskReplies(where)
            return store
    if callable(getattr(setting, "get", None)) and callable(getattr(setting, "put", None)):
        return setting
    raise TypeError("cache_replies is False, True (memory), 'disk', a folder or .sqlite path, or an object with "
                    f"get(key) and put(key, reply); not {setting!r}")


def check_setting(setting: Any) -> None:
    if setting is None or isinstance(setting, bool):
        return
    if isinstance(setting, str) and not setting.strip():
        raise ValueError("cache_replies is False, True, 'disk' or a path, not an empty text")
    if isinstance(setting, (str, os.PathLike)):
        return
    if not (callable(getattr(setting, "get", None)) and callable(getattr(setting, "put", None))):
        raise TypeError("cache_replies is False, True (memory), 'disk', a folder or .sqlite path, or an object with "
                        f"get(key) and put(key, reply); not {type(setting).__name__}")


class Flight:
    """One request's turn at the cache: ``reply`` (a kept reply, or None: this
    caller asks the model), then ``keep(response)`` once the reply was read,
    or ``drop()`` when it could not be (a kept reply that no longer reads is
    forgotten). ``end()`` always."""

    __slots__ = ("store", "key", "reply", "_ended")

    def __init__(self, store: Any, k: str, reply: Any):
        self.store, self.key, self.reply, self._ended = store, k, reply, False

    def keep(self, response: Any) -> None:
        if self.reply is None:
            try:
                self.store.put(self.key, response)
            except Exception as exc:  # noqa: BLE001 — a cache that cannot write never fails a call
                from .calllog import _warn_once
                _warn_once(("replies", type(exc).__name__), f"a reply could not be kept in the cache "
                                                            f"({type(exc).__name__}: {exc})")

    def drop(self) -> None:
        discard = getattr(self.store, "discard", None)
        if self.reply is not None and callable(discard):
            discard(self.key)

    def end(self) -> None:
        if not self._ended:
            self._ended = True
            unclaim = getattr(self.store, "unclaim", None)
            if callable(unclaim):
                unclaim(self.key)


def begin(settings: Dict[str, Any], request: Any, cancelled: Any = None) -> Optional[Flight]:
    """The cache's turn for this request under these settings, or None when
    no cache applies. A disk (or any durable) cache is skipped for a call
    whose log_content drops a field: it would keep what may not be kept."""
    try:
        store = store_for(settings.get("cache_replies"))
    except (TypeError, ValueError, OSError) as exc:
        from .calllog import _warn_once
        _warn_once(("replies-store", type(exc).__name__), f"the reply cache cannot be opened ({exc}); replies are "
                                                          f"not cached")
        return None
    if store is None:
        return None
    if getattr(store, "durable", True) and not _content_whole():
        store = MEMORY
    k = key(request, settings.get("replicate") or 0)
    claim = getattr(store, "claim", None)
    reply = claim(k, cancelled) if callable(claim) else store.get(k)
    return Flight(store, k, reply)


def _content_whole() -> bool:
    """Whether the call in progress keeps every field (no log_content layer
    drops one); True outside a call (nothing is known to be dropped)."""
    from . import calllog
    call = calllog.current()
    if call is None:
        return True
    try:
        return not calllog.dropped_now(call.program)
    except Exception:  # noqa: BLE001 — unsure: not written
        return False


def clear(which: Any = None) -> None:
    """Forget cached replies: the memory cache (default), or the store a
    ``cache_replies`` value names (``"disk"``, a path)."""
    if which is None or which is True:
        MEMORY.clear()
        return
    store = store_for(which)
    if store is not None and callable(getattr(store, "clear", None)):
        store.clear()


__all__ = ["key", "MemoryReplies", "DiskReplies", "store_for", "begin", "clear", "default_path", "FORMAT"]
