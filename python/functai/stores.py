"""Where conversations are kept (contract/conversations.md, *Stores*).

A store keeps each conversation as an ordered list of records (JSON
objects) and never changes one. It has two methods every store must have:

- ``append(conversation, records, *, expect=None) -> int``: add the
  records at the end, all or none, numbering them (``seq``: 1 for a
  conversation's first record); with ``expect``, only if the conversation
  holds exactly that many records now, else ``ConversationError``
  (``store-conflict``). Returns how many it holds after.
- ``read(conversation, after=0) -> list``: the records after position
  ``after``, in order.

and may have:

- ``events``: a store of call tree logs (contract/streaming.md, *The rules a
  store keeps*), so another process can watch a turn while it is written;
- ``wait(conversation, after, timeout)``: block until a record after
  ``after`` exists (or the time is up), instead of being asked again;
- ``durability``: ``"memory"`` (this process), ``"disk"`` (written and
  flushed to disk before ``append`` returns), or the store's own word;
- ``persistent``: whether it keeps records beyond this process (a store
  that does refuses a program whose ``log_content`` drops a field).

Two stores are here: ``MemoryConversations`` (the default: this process's
memory) and ``FolderStore`` (files in a folder, locked across processes).
"""

from __future__ import annotations

import copy
import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence

from .errors import ConversationError, EventRefused

FORMAT = 1
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,199}")


def check_id(conversation: Any) -> str:
    """A conversation's id: 1 to 200 ASCII letters, digits, ``.``, ``_`` or
    ``-``, not starting with a punctuation mark (so it is a file name on every
    system, and never a path)."""
    if not isinstance(conversation, str) or not _ID.fullmatch(conversation):
        raise ConversationError("conversation-id", f"a conversation's id is 1 to 200 letters, digits, '.', '_' or "
                                                   f"'-', starting with a letter or digit; not {conversation!r}")
    return conversation


# ------------------------------------------------------------------ memory


class MemoryConversations:
    """Conversations kept in this process's memory (lost when it ends): the
    store a conversation uses when none is named. One per process by
    default, so opening the same id again opens the same conversation."""

    durability = "memory"
    persistent = False

    def __init__(self) -> None:
        from .eventlog import MemoryStore
        self._records: Dict[str, List[Dict[str, Any]]] = {}
        self._cond = threading.Condition()
        self.events = MemoryStore()

    def append(self, conversation: str, records: Sequence[Mapping[str, Any]], *, expect: Optional[int] = None) -> int:
        check_id(conversation)
        with self._cond:
            log = self._records.setdefault(conversation, [])
            if expect is not None and expect != len(log):
                raise ConversationError("store-conflict", f"conversation {conversation} holds {len(log)} records, "
                                                          f"not {expect}")
            for r in records:
                rec = copy.deepcopy(dict(r))
                rec["seq"] = len(log) + 1
                log.append(rec)
            self._cond.notify_all()
            return len(log)

    def read(self, conversation: str, after: int = 0) -> List[Dict[str, Any]]:
        with self._cond:
            return copy.deepcopy(self._records.get(conversation, [])[int(after or 0):])

    def wait(self, conversation: str, after: int, timeout: float) -> None:
        with self._cond:
            self._cond.wait_for(lambda: len(self._records.get(conversation, [])) > after, timeout)

    def conversations(self) -> List[str]:
        with self._cond:
            return [c for c, log in self._records.items() if log]

    def __repr__(self) -> str:
        return f"<MemoryConversations: {len(self._records)} conversations>"


MEMORY = MemoryConversations()


# ------------------------------------------------------------------ files


class _Lock:
    """An exclusive lock on a file, across processes (``flock``; Windows:
    ``msvcrt.locking``) and across this process's threads."""

    _threads: Dict[str, threading.Lock] = {}
    _guard = threading.Lock()

    def __init__(self, path: Path):
        self.path = path
        with _Lock._guard:
            self.thread = _Lock._threads.setdefault(str(path), threading.Lock())
        self.fd: Optional[int] = None

    def __enter__(self) -> "_Lock":
        self.thread.acquire()
        try:
            self.fd = os.open(self.path, os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0), 0o600)
            if sys.platform == "win32":
                import msvcrt
                while True:
                    try:
                        msvcrt.locking(self.fd, msvcrt.LK_LOCK, 1)
                        break
                    except OSError:
                        time.sleep(0.01)
            else:
                import fcntl
                fcntl.flock(self.fd, fcntl.LOCK_EX)
        except BaseException:
            if self.fd is not None:
                os.close(self.fd)
                self.fd = None
            self.thread.release()
            raise
        return self

    def __exit__(self, *exc: Any) -> None:
        try:
            if self.fd is not None:
                if sys.platform == "win32":
                    import msvcrt
                    try:
                        os.lseek(self.fd, 0, 0)
                        msvcrt.locking(self.fd, msvcrt.LK_UNLCK, 1)
                    except OSError:
                        pass
                os.close(self.fd)                 # closing releases flock
        finally:
            self.fd = None
            self.thread.release()


class _Lines:
    """A JSONL file read incrementally: what was parsed is kept, and only
    what was appended since is read again."""

    def __init__(self, path: Path):
        self.path = path
        self.size = 0
        self.items: List[Dict[str, Any]] = []
        self.lock = threading.Lock()

    def load(self) -> List[Dict[str, Any]]:
        with self.lock:
            try:
                size = self.path.stat().st_size
            except FileNotFoundError:
                self.size, self.items = 0, []
                return self.items
            if size < self.size:                  # replaced: read it all again
                self.size, self.items = 0, []
            if size > self.size:
                with open(self.path, "rb") as f:
                    f.seek(self.size)
                    data = f.read(size - self.size)
                end = data.rfind(b"\n") + 1       # a line still being written is read next time
                for raw in data[:end].split(b"\n"):
                    if raw.strip():
                        try:
                            self.items.append(json.loads(raw))
                        except ValueError:
                            self.items.append({"unreadable": True})
                self.size += end
            return self.items


def _write(path: Path, lines: Sequence[bytes], durable: bool) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_CLOEXEC", 0), 0o600)
    try:
        view = memoryview(b"".join(lines))
        while view:
            view = view[os.write(fd, view):]
        if durable:
            os.fsync(fd)
    finally:
        os.close(fd)


def _dump(record: Mapping[str, Any]) -> bytes:
    return (json.dumps(record, ensure_ascii=False, separators=(",", ":"), default=str) + "\n").encode("utf-8")


class FolderEvents:
    """Call tree logs kept in files, by the rules every store keeps
    (contract/streaming.md): ``<folder>/<tree>.jsonl`` holds the kept events,
    ``<tree>.writer`` the last writer number a claim gave. Each claim and
    each append is one step, under the log's lock."""

    batches = True
    writer = None

    def __init__(self, folder: "str | os.PathLike[str]", *, durable: bool = False):
        self.folder = Path(folder)
        os.makedirs(self.folder, mode=0o700, exist_ok=True)
        self.durable = durable
        self._files: Dict[str, _Lines] = {}
        self._guard = threading.Lock()

    def _lines(self, tree: str) -> _Lines:
        with self._guard:
            f = self._files.get(tree)
            if f is None:
                f = self._files[tree] = _Lines(self.folder / f"{_safe(tree)}.jsonl")
            return f

    def _writer_of(self, tree: str) -> int:
        try:
            return int((self.folder / f"{_safe(tree)}.writer").read_text().strip() or 1)
        except (FileNotFoundError, ValueError):
            return 1

    def extend(self, events: Sequence[Mapping[str, Any]]) -> str:
        from .eventlog import apply_append, position
        events = list(events)
        if not events:
            return "duplicate"
        tree = events[0].get("tree") if isinstance(events[0], Mapping) else None
        if not isinstance(tree, str):
            raise EventRefused("event-malformed", "not an event of format 2 of this log")
        with _Lock(self.folder / f"{_safe(tree)}.lock"):
            kept = self._lines(tree).load()
            logs = {tree: list(kept)}
            writers = {tree: self._writer_of(tree)}
            answers, new = [], []
            for e in events:
                before = len(logs[tree])
                try:
                    answers.append(apply_append(logs, writers, e, tree))
                except EventRefused as exc:
                    named = position(e) if isinstance(e, Mapping) and isinstance(e.get("writer"), int) \
                        and isinstance(e.get("seq"), int) else None
                    raise EventRefused(exc.code, str(exc), event=named) from None
                if len(logs[tree]) > before:
                    new.append(_dump(logs[tree][-1]))
            if new:
                _write(self.folder / f"{_safe(tree)}.jsonl", new, self.durable)
            return "duplicate" if all(a == "duplicate" for a in answers) else "kept"

    def append(self, event: Mapping[str, Any]) -> str:
        return self.extend([event])

    def claim(self, tree: str) -> Dict[str, Any]:
        from .eventlog import _finished, position
        with _Lock(self.folder / f"{_safe(tree)}.lock"):
            log = self._lines(tree).load()
            if not log:
                raise EventRefused("event-unknown", f"no log {tree}")
            if _finished(log, tree):
                raise EventRefused("event-after-end", "the log is finished")
            writer = self._writer_of(tree) + 1
            path = self.folder / f"{_safe(tree)}.writer"
            tmp = path.with_suffix(".writer.tmp")
            tmp.write_text(str(writer))
            os.replace(tmp, path)
            return {"writer": writer, "after": position(log[-1])}

    def read(self, tree: str, after: Optional[Mapping[str, Any]] = None) -> List[Dict[str, Any]]:
        from .eventlog import read_after
        return read_after([e for e in self._lines(tree).load() if "unreadable" not in e], after)

    def finished(self, tree: str) -> bool:
        from .eventlog import _finished
        return _finished(self._lines(tree).load(), tree)

    def __repr__(self) -> str:
        return f"<FolderEvents {self.folder}>"


def _safe(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,199}", name or ""):
        raise EventRefused("event-malformed", f"{name!r} is not a log's id")
    return name


def default_folder() -> Path:
    """Where ``store=True`` keeps conversations: ``$XDG_DATA_HOME/functai/conversations``."""
    from .calllog import default_folder as calls
    return calls().parent / "conversations"


class FolderStore:
    """Conversations kept in a folder, shared by every process that opens it:

        <folder>/conversations/<id>.jsonl   the records, one per line
        <folder>/conversations/<id>.lock    taken while appending
        <folder>/trees/<tree>.jsonl         each turn's call tree log (kept form)

    Each append is one step across processes (a lock on the conversation),
    written and flushed to disk before it returns (``durability`` ``"disk"``).
    Files are readable by their owner only."""

    durability = "disk"
    persistent = True

    def __init__(self, folder: "str | os.PathLike[str]"):
        self.folder = Path(os.fspath(folder)).expanduser().absolute()
        self._conversations = self.folder / "conversations"
        os.makedirs(self._conversations, mode=0o700, exist_ok=True)
        self.events = FolderEvents(self.folder / "trees")
        self._files: Dict[str, _Lines] = {}
        self._guard = threading.Lock()

    def _lines(self, conversation: str) -> _Lines:
        with self._guard:
            f = self._files.get(conversation)
            if f is None:
                f = self._files[conversation] = _Lines(self._conversations / f"{conversation}.jsonl")
            return f

    def append(self, conversation: str, records: Sequence[Mapping[str, Any]], *, expect: Optional[int] = None) -> int:
        check_id(conversation)
        with _Lock(self._conversations / f"{conversation}.lock"):
            held = self._lines(conversation).load()
            n = len(held)
            if expect is not None and expect != n:
                raise ConversationError("store-conflict", f"conversation {conversation} holds {n} records, "
                                                          f"not {expect}")
            lines = []
            for r in records:
                n += 1
                lines.append(_dump({**dict(r), "seq": n}))
            if lines:
                _write(self._conversations / f"{conversation}.jsonl", lines, True)
            return n

    def read(self, conversation: str, after: int = 0) -> List[Dict[str, Any]]:
        check_id(conversation)
        return copy.deepcopy(self._lines(conversation).load()[int(after or 0):])

    def wait(self, conversation: str, after: int, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        pause = 0.02
        while time.monotonic() < deadline:
            if len(self._lines(conversation).load()) > after:
                return
            time.sleep(min(pause, max(0.0, deadline - time.monotonic())))
            pause = min(pause * 2, 0.2)

    def conversations(self) -> List[str]:
        return sorted(p.stem for p in self._conversations.glob("*.jsonl"))

    def __repr__(self) -> str:
        return f"<FolderStore {self.folder}>"


_folders: Dict[str, FolderStore] = {}
_folders_lock = threading.Lock()


def store_for(store: Any) -> Any:
    """The store a ``store=`` value names: None (this process's memory), True
    (the default folder), a folder, or an object with ``append`` and
    ``read``."""
    if store is None or store is False:
        return MEMORY
    if store is True:
        store = default_folder()
    if isinstance(store, (str, os.PathLike)):
        where = str(Path(os.fspath(store)).expanduser().absolute())
        with _folders_lock:
            found = _folders.get(where)
            if found is None:
                found = _folders[where] = FolderStore(where)
            return found
    if callable(getattr(store, "append", None)) and callable(getattr(store, "read", None)):
        return store
    raise TypeError("store is None (memory), True (the default folder), a folder, or an object with "
                    f"append(conversation, records, expect=) and read(conversation, after=); not {store!r}")


def is_persistent(store: Any) -> bool:
    """Whether a store keeps records beyond this process (unknown: yes)."""
    flag = getattr(store, "persistent", None)
    return True if flag is None else bool(flag)


def wait(store: Any, conversation: str, after: int, timeout: float) -> None:
    """Wait for a record after ``after`` (at most ``timeout`` seconds)."""
    w = getattr(store, "wait", None)
    if callable(w):
        w(conversation, after, timeout)
    else:
        time.sleep(min(timeout, 0.1))


def events_of(store: Any) -> Any:
    """A store's call tree logs, or None."""
    ev = getattr(store, "events", None)
    return ev if ev is not None and callable(getattr(ev, "read", None)) else None


def follow(store: Any, tree: str, after: Optional[Mapping[str, Any]] = None, *, poll: float = 0.05,
           stop: Any = None, timeout: Optional[float] = None) -> Iterator[Dict[str, Any]]:
    """The kept events of a tree's log from a store, after ``after``: those
    kept so far, then each as it is kept, until the log's last event (or
    ``stop()`` says so, or ``timeout`` seconds pass without one)."""
    from .eventlog import position
    events = events_of(store)
    if events is None:
        return
    last = dict(after) if after is not None else None
    quiet = time.monotonic()
    while True:
        got = events.read(tree, last)
        for e in got:
            last = position(e)
            quiet = time.monotonic()
            yield e
            if e.get("call") == tree and e.get("kind") in ("done", "failed"):
                return
        if stop is not None and stop():
            return
        if timeout is not None and time.monotonic() - quiet > timeout:
            return
        time.sleep(poll)


__all__ = ["MemoryConversations", "FolderStore", "FolderEvents", "store_for", "check_id", "default_folder", "MEMORY"]
