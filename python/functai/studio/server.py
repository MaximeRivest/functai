"""The dashboard's server: Python's own HTTP server, the page in ``ui/``, and a small
JSON API over the same functions the command line uses.

It listens on this computer only (127.0.0.1) unless told otherwise; on any other
address it prints a link with a key, and every request must carry that key. Every
request that changes something must carry the header ``X-Studio: 1``, which a page
from another site cannot add, so other sites cannot act through your browser.
Long work (running examples, what-ifs, comparisons) runs as jobs, one at a time.
"""

from __future__ import annotations

import hashlib
import http.server
import ipaddress
import json
import mimetypes
import os
import secrets
import shutil
import socket
import sys
import threading
import time
import traceback
import urllib.parse
import uuid
import webbrowser
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .project import Project, StudioError, find_project

UI = Path(__file__).parent / "ui"


class Jobs:
    """Long work, one at a time, with progress."""

    def __init__(self) -> None:
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
        self.run_lock = threading.Lock()

    def start(self, kind: str, label: str, fn: Callable[[Callable[[int, int], None]], Any]) -> Dict[str, Any]:
        job = {"id": uuid.uuid4().hex[:10], "kind": kind, "label": label, "state": "queued", "done": 0, "total": 0,
               "result": None, "error": None, "started": time.time(), "ended": None}
        with self.lock:
            self.jobs[job["id"]] = job
            for old in sorted(self.jobs.values(), key=lambda j: j["started"])[:-30]:
                self.jobs.pop(old["id"], None)

        def progress(done: int, total: int) -> None:
            job["done"], job["total"] = done, total

        def work() -> None:
            with self.run_lock:
                job["state"] = "running"
                try:
                    job["result"] = fn(progress)
                    job["state"] = "done"
                except StudioError as exc:
                    job["state"], job["error"] = "failed", str(exc)
                except Exception as exc:  # noqa: BLE001 — shown to the person, and logged here
                    traceback.print_exc()
                    job["state"], job["error"] = "failed", f"{type(exc).__name__}: {exc}"
                job["ended"] = time.time()
        threading.Thread(target=work, daemon=True).start()
        return job

    def list(self) -> List[Dict[str, Any]]:
        with self.lock:
            return sorted(self.jobs.values(), key=lambda j: -j["started"])[:12]

    def busy(self) -> bool:
        return any(j["state"] in ("queued", "running") for j in self.jobs.values())


class Studio:
    """The server's state: the folder, the project in it (if any), the jobs."""

    def __init__(self, project: Optional[Project], folder: Path, key: Optional[str]):
        self.project = project
        self.folder = folder
        self.key = key
        self.jobs = Jobs()
        self.last_ok: Optional[Dict[str, Any]] = None
        self.cache: Dict[str, Any] = {}
        from .connect import Logins
        self.logins = Logins()
        self.home = folder                 # where Studio was started: the pipelines it lists

    def open(self, project: Optional[Project]) -> None:
        self.project, self.last_ok = project, None
        if project is not None:
            os.chdir(project.root)
            project.ensure_dirs()
            try:
                project.load()
            except StudioError:
                pass                       # shown on the page, with the code to fix it

    def projects(self) -> List[Dict[str, Any]]:
        """The pipelines in the home folder and its subfolders (one level), without running them."""
        import ast as _ast

        from .project import _candidate_files
        out = []
        for folder in [self.home] + sorted(d for d in self.home.iterdir() if d.is_dir() and not d.name.startswith(".")):
            for f in _candidate_files(folder):
                about = ""
                try:
                    doc = _ast.get_docstring(_ast.parse(f.read_text(encoding="utf-8"))) or ""
                    about = doc.strip().splitlines()[0] if doc.strip() else ""
                except (SyntaxError, OSError, UnicodeDecodeError):
                    pass
                ex = folder / "examples"
                n = sum(max(0, len(x.read_text(encoding="utf-8").splitlines()) - (1 if x.suffix == ".csv" else 0))
                        for x in ex.glob("*.*") if x.suffix in (".csv", ".jsonl")) if ex.is_dir() else 0
                out.append({"path": str(f.relative_to(self.home)), "name": f.stem.replace("_", " ").capitalize(),
                            "about": about, "examples": n, "modified": f.stat().st_mtime})
        return sorted(out, key=lambda x: -x["modified"])

    def accounts(self) -> Dict[str, Any]:
        from . import connect
        conns = connect.connections()
        ready = {c["provider"]: c for c in conns if c["status"] == "ready"}
        cur = self.logins.current
        return {"connections": conns, "can_add": connect.can_add(), "catalogue": connect.catalogue(ready),
                "default_model": connect.default_model(self.project.settings().get("lm") if self.project else None),
                "login": cur.view() if cur else None}

    # ---- what changed

    def etag(self) -> str:
        h = hashlib.sha256()
        p = self.project
        try:
            from .. import accounts as _acc
            cred = _acc.store_path(_acc.auth_for(True))
            h.update(f"{cred.stat().st_mtime_ns if cred and cred.exists() else '-'}".encode())
        except Exception:  # noqa: BLE001
            pass
        cur = self.logins.current
        if cur is not None:
            h.update(f"{cur.id}:{cur.state}:{len(cur.notices)}:{bool(cur.prompt)}".encode())
        if p is None:
            for d in [self.home] + [x for x in self.home.iterdir() if x.is_dir() and not x.name.startswith(".")]:
                h.update(f"{d}:{d.stat().st_mtime_ns}".encode())
            h.update(str([(j["id"], j["state"]) for j in self.jobs.list()]).encode())
            return h.hexdigest()[:16]
        beside = sorted(x for x in p.root.iterdir() if x.is_file() and not x.name.startswith("."))
        for path in (*beside, p.examples_path(), p.settings_path, p.studio_dir / "blame.json",
                     p.studio_dir / "whatif-judgements.jsonl", p.studio_dir / "edits" / "history.jsonl"):
            try:
                h.update(f"{path}:{path.stat().st_mtime_ns}:{path.stat().st_size}".encode())
            except OSError:
                h.update(f"{path}:-".encode())
        if p.calls_dir.exists():
            for day in sorted(p.calls_dir.iterdir()):
                if day.is_dir():
                    for f in day.iterdir():
                        st = f.stat()
                        h.update(f"{f.name}:{st.st_mtime_ns}:{st.st_size}".encode())
        h.update(str([(j["id"], j["state"]) for j in self.jobs.list()]).encode())
        return h.hexdigest()[:16]

    def state(self) -> Dict[str, Any]:
        from .data import snapshot
        p = self.project
        jobs = self.jobs.list()
        if p is None:
            return {"project": None, "folder": str(self.home), "projects": self.projects(), "jobs": jobs,
                    "accounts": self.accounts(), "local": None}
        try:
            snap = snapshot(p)
            for name in ("blame.json",):
                try:
                    snap["blame"] = json.loads((p.studio_dir / name).read_text())
                except (OSError, ValueError):
                    snap["blame"] = None
            from . import running
            from .page import facts, score_words
            costs = running.step_costs(p)
            stale = [e for e in p.examples() if e["key"] not in {r["key"] for r in snap["rows"] if r.get("current")}]
            snap["estimates"] = {
                "why": {str(r["n"]): running.estimate_why(p, r, costs)["cents"]
                        for r in snap["rows"] if r.get("ran") and r["state"] == "wrong"},
                "blame": running.estimate_blame(p, snap)["cents"],
                "per_example": running._sum(costs, list(costs))["cents"],
                "not_run": running.estimate(p, stale)["cents"] if stale else 0.0,
                "examples": __import__("functai.studio.helper", fromlist=["x"]).estimate_cents(p, kind="examples"),
            }
            snap["facts"] = facts(snap, blame=snap["blame"], monthly=p.settings().get("monthly"))
            from . import edit
            snap["code"]["sha"] = edit.sha(snap["code"]["text"])
            snap["wiring"] = edit.wiring(p)
            snap["edits"] = edit.history(p)[-12:]
            snap["files"] = {name: (p.root / rel).read_text(encoding="utf-8") if (p.root / rel).is_file() else ""
                             for name, rel in snap["wiring"]["files"].items()}
            snap["score_words"] = score_words(snap["scores"]["end"])
            snap["jobs"] = jobs
            snap["project"] = True
            snap["accounts"] = self.accounts()
            snap["home"] = str(self.home)
            snap["fill_in"] = [name for name, text in snap["files"].items() if text.startswith("(Write ")]
            self.last_ok = snap
            return snap
        except StudioError as exc:
            out = dict(self.last_ok or {"project": True})
            out["error"] = str(exc)
            out["jobs"] = jobs
            out["accounts"] = self.accounts()
            try:
                from .edit import sha
                text = p.file.read_text(encoding="utf-8")
                out["code"] = {"text": text, "changed": [], "tracked": False, "sha": sha(text)}
            except OSError:
                pass
            return out


def _json(handler: http.server.BaseHTTPRequestHandler, status: int, data: Any) -> None:
    body = json.dumps(data, ensure_ascii=False, default=str).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


def make_handler(studio: Studio):
    class Handler(http.server.BaseHTTPRequestHandler):
        server_version = "FunctAIStudio"

        def log_message(self, fmt: str, *args: Any) -> None:  # quiet: the terminal is the agent's
            if os.environ.get("FUNCTAI_STUDIO_LOG"):
                super().log_message(fmt, *args)

        # ---- the key, when not on this computer only

        def _allowed(self, query: Dict[str, List[str]]) -> bool:
            if not studio.key:
                return True
            cookie = self.headers.get("Cookie") or ""
            if f"studio_key={studio.key}" in cookie:
                return True
            return secrets.compare_digest((query.get("key") or [""])[0], studio.key)

        def do_GET(self) -> None:  # noqa: N802
            url = urllib.parse.urlparse(self.path)
            query = urllib.parse.parse_qs(url.query)
            if not self._allowed(query):
                return _json(self, 403, {"error": "this Studio needs its key: use the link it printed"})
            if url.path in ("/", "/index.html"):
                return self._file(UI / "index.html", set_key=bool(studio.key))
            if url.path.startswith("/ui/"):
                target = (UI / url.path[4:]).resolve()
                if UI.resolve() in target.parents and target.is_file():
                    return self._file(target)
                return _json(self, 404, {"error": "no such file"})
            if url.path == "/api/state":
                since = (query.get("since") or [""])[0]
                tag = studio.etag()
                if since and since == tag:
                    return _json(self, 200, {"etag": tag, "unchanged": True, "jobs": studio.jobs.list()})
                st = studio.state()
                st["etag"] = studio.etag()        # reading may have written settings the first time
                return _json(self, 200, st)
            if url.path == "/api/page":
                return self._page()
            if url.path == "/api/accounts/login":
                cur = studio.logins.current
                return _json(self, 200, cur.view() if cur else None)
            if url.path == "/api/jobs":
                return _json(self, 200, studio.jobs.list())
            if url.path == "/api/call":
                return self._call((query.get("id") or [""])[0])
            if url.path == "/api/example":
                from .data import example_detail
                p = studio.project
                if p is None:
                    return _json(self, 404, {"error": "no program yet"})
                try:
                    return _json(self, 200, example_detail(p, (query.get("n") or [""])[0]))
                except StudioError as exc:
                    return _json(self, 404, {"error": str(exc)})
            if url.path == "/api/version":
                return self._version((query.get("hash") or [""])[0])
            return _json(self, 404, {"error": "no such address"})

        def do_POST(self) -> None:  # noqa: N802
            url = urllib.parse.urlparse(self.path)
            if not self._allowed(urllib.parse.parse_qs(url.query)):
                return _json(self, 403, {"error": "this Studio needs its key"})
            if self.headers.get("X-Studio") != "1":
                return _json(self, 403, {"error": "requests that change things must come from the dashboard"})
            try:
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}") if length else {}
            except ValueError:
                return _json(self, 400, {"error": "not JSON"})
            try:
                local = _loopback(self.client_address[0])
                return _json(self, 200, post(studio, url.path, body, local=local))
            except NeedsYes as exc:
                return _json(self, 409, {"error": str(exc), "estimate": exc.estimate})
            except StudioError as exc:
                return _json(self, 400, {"error": str(exc)})
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                return _json(self, 500, {"error": f"{type(exc).__name__}: {exc}"})

        def _file(self, path: Path, *, set_key: bool = False) -> None:
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", (mimetypes.guess_type(path.name)[0] or "application/octet-stream")
                             + ("; charset=utf-8" if path.suffix in (".html", ".js", ".css") else ""))
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-cache")
            if set_key:
                self.send_header("Set-Cookie", f"studio_key={studio.key}; Path=/; HttpOnly; SameSite=Strict")
            self.end_headers()
            self.wfile.write(data)

        def _call(self, call_id: str) -> None:
            """One call's whole record (what was sent, what came back), for "the exact request"."""
            from .data import read_log
            p = studio.project
            if p is None:
                return _json(self, 404, {"error": "no program yet"})
            calls, _ = read_log(p)
            rec = next((c for c in calls if c.get("id") == call_id), None)
            if rec is None:
                return _json(self, 404, {"error": "no such call in this project"})
            return _json(self, 200, rec)

        def _version(self, version: str) -> None:
            """A kept version's code, and how the file now differs from it."""
            import difflib

            from .data import version_file
            p = studio.project
            if p is None or not version.startswith("sha256:"):
                return _json(self, 404, {"error": "no such version"})
            path = version_file(p, version)
            if not path.exists():
                return _json(self, 404, {"error": "that version's code was not kept (it ran before Studio kept "
                                                  "versions); git may still have it"})
            old = path.read_text(encoding="utf-8")
            now = p.file.read_text(encoding="utf-8")
            diff = list(difflib.unified_diff(old.splitlines(), now.splitlines(), "that version", "now", lineterm="", n=2))
            return _json(self, 200, {"code": old, "diff": diff[2:]})

        def _page(self) -> None:
            from .data import snapshot
            from .page import summary_html
            p = studio.project
            if p is None:
                return _json(self, 404, {"error": "no program yet"})
            blame = None
            try:
                blame = json.loads((p.studio_dir / "blame.json").read_text())
            except (OSError, ValueError):
                pass
            monthly = p.settings().get("monthly")
            html = summary_html(snapshot(p), blame=blame, monthly=monthly).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Disposition", f'attachment; filename="{p.name}-summary.html"')
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)

    return Handler


class NeedsYes(StudioError):
    def __init__(self, message: str, estimate: Dict[str, Any]):
        super().__init__(message)
        self.estimate = estimate


SECRET_ONLY_HERE = ("a key or a sign-in code is accepted only from this computer, or through an encrypted "
                    "link to it (https, such as tailscale serve): this page came over plain http from the "
                    "network, where anyone could read it")


def post(studio: Studio, path: str, body: Dict[str, Any], *, local: bool = True) -> Any:
    """Every change the dashboard can ask for."""
    import functai

    from . import connect, helper, newpipe, running
    from .edit import set_model
    p = studio.project
    # ---- AI accounts (no project needed)
    if path == "/api/accounts/key":
        if not local:
            raise StudioError(SECRET_ONLY_HERE)
        return connect.add_key(body.get("provider", ""), body.get("key", ""))
    if path == "/api/accounts/login":
        return studio.logins.start(body["provider"], body.get("method")).view()
    if path == "/api/accounts/login/answer":
        if not local:
            raise StudioError(SECRET_ONLY_HERE)
        studio.logins.get(body["id"]).answer(str(body.get("value", "")))
        return {"ok": True}
    if path == "/api/accounts/login/cancel":
        studio.logins.get(body["id"]).cancel()
        return {"ok": True}
    if path == "/api/accounts/signout":
        return connect.sign_out(body["provider"])
    # ---- pipelines: the ready example, a new one, opening and closing
    if path == "/api/example":
        from .cli import EXAMPLES
        dest, i = studio.home / "club-inbox", 2
        while dest.exists():
            dest, i = studio.home / f"club-inbox-{i}", i + 1
        shutil.copytree(EXAMPLES / "club", dest, ignore=shutil.ignore_patterns("__pycache__", ".functai"))
        studio.open(find_project(None, folder=dest))
        return {"made": str(dest)}
    if path == "/api/projects/open":
        target = (studio.home / body["path"]).resolve()
        if studio.home.resolve() not in [target, *target.parents]:
            raise StudioError("that is outside the folder Studio was started in")
        studio.open(find_project(str(target)))
        return {"ok": True}
    if path == "/api/projects/close":
        studio.open(None)
        return {"ok": True}
    if path == "/api/new":
        made = newpipe.write(studio.home, body.get("plan") or newpipe.blank())
        studio.open(find_project(str(made)))
        return {"made": str(made.relative_to(studio.home))}
    if path == "/api/draft":
        if studio.jobs.busy():
            raise StudioError("something is already running: wait for it to finish")
        req = body.get("request", "")
        return studio.jobs.start("draft", "Drafting a plan", lambda prog: newpipe.check(helper.draft_plan(req)))
    if path == "/api/draft/estimate":
        return {"cents": helper.estimate_cents(p, kind="plan"), "model": connect.default_model(None)}
    if p is None:
        raise StudioError("no program open: choose one, or make one")
    person = p.person()
    if path == "/api/rate":
        verdict = body.get("verdict")
        if verdict not in ("right", "wrong", None):
            raise StudioError("verdict is right, wrong or null (withdraw)")
        kw: Dict[str, Any] = {"by": person, "folder": p.calls_dir}
        if body.get("answer") not in (None, ""):
            kw["answer"] = body["answer"]
        if body.get("note"):
            kw["note"] = body["note"]
        if body.get("sample"):
            kw["sample"] = str(body["sample"])
        functai.rate(body["call"], verdict, **kw)
        return {"ok": True, "by": person}
    if path == "/api/settings":
        allowed = {k: body[k] for k in ("monthly", "cap_dollars", "lm", "by", "budget_dollars") if k in body}
        return p.update_settings(**allowed)
    if path == "/api/use-model":
        return set_model(p, body["step"], body["model"], base=body.get("base"))
    if path == "/api/edit":
        from . import edit
        op, base = body.get("op"), body.get("base")
        if studio.jobs.busy():
            raise StudioError("something is running on this code: wait for it to finish, then change it")
        if op == "model":
            return edit.set_model(p, body["step"], body["model"], base=base)
        if op == "models":
            return edit.set_all_models(p, list(p.steps()), body["model"], base=base)
        if op == "instruction":
            return edit.set_instruction(p, body["step"], body["text"], base=base)
        if op == "answer":
            return edit.set_answer(p, body["step"], body["kind"], body.get("choices"), base=base)
        if op == "input":
            return edit.set_input(p, body["step"], body["param"], body["source"], line=body.get("line"), base=base)
        if op == "value":
            return edit.set_value(p, int(body["line"]), int(body["col"]), body["value"], base=base)
        if op == "file":
            return edit.set_file(p, body["name"], body["text"])
        if op == "add":
            return edit.add_step(p, body["after"], body["name"], body["instruction"], answer=body.get("answer", "same"),
                                 choices=body.get("choices"), extra=body.get("extra"), base=base)
        if op == "remove":
            return edit.remove_step(p, body["step"], base=base)
        if op == "source":
            return edit.set_source(p, body["text"], base=base)
        raise StudioError(f"no such change: {op}")
    if path == "/api/undo":
        from . import edit
        if studio.jobs.busy():
            raise StudioError("something is running on this code: wait for it to finish")
        return edit.undo(p)
    if path == "/api/examples":
        from . import edit
        op = body.get("op")
        if op == "add":
            return edit.add_examples(p, [body["values"]])
        if op == "update":
            return edit.update_example(p, int(body["n"]), body["values"])
        if op == "delete":
            return edit.delete_example(p, int(body["n"]))
        if op == "paste":
            return edit.paste_examples(p, body.get("text", ""))
        raise StudioError(f"no such change to the examples: {op}")
    if path == "/api/examples/write":
        if studio.jobs.busy():
            raise StudioError("something is already running: wait for it to finish")
        count = int(body.get("count") or 10)

        def write_some(prog: Callable[[int, int], None]) -> Any:
            from . import edit
            rows = helper.draft_examples(p, count)
            if not rows:
                raise StudioError("the helper wrote no usable example: try again, or write a few yourself")
            return edit.add_examples(p, rows, what=f"added {len(rows)} examples written by the AI helper")
        return studio.jobs.start("examples", f"Writing {count} examples", write_some)
    if path == "/api/default-model":
        return p.update_settings(lm=body.get("model") or None)
    if path == "/api/restore":
        if studio.jobs.busy():
            raise StudioError("something is running on this code: wait for it to finish")
        from .. import calllog
        from .data import keep_version_code, version_file
        target = version_file(p, str(body.get("hash", "")))
        if not str(body.get("hash", "")).startswith("sha256:") or not target.exists():
            raise StudioError("that version's code was not kept, so Studio cannot go back to it")
        keep_version_code(p, calllog.program_info(p.load())["version"])   # the current code stays restorable
        tmp = p.file.with_suffix(p.file.suffix + ".studio-tmp")
        tmp.write_text(target.read_text(encoding="utf-8"), encoding="utf-8")
        tmp.replace(p.file)
        return {"ok": True, "file": p.file.name}
    if path == "/api/whatif-judge":
        running.judge_whatif(p, body["inputs"], body["end"], body["verdict"], by=person)
        return {"ok": True}
    from .data import snapshot

    def guard(est: Dict[str, Any], what: str) -> None:
        """Every action that asks a model says its cost first and waits for a yes above the limit."""
        cap = float(p.settings().get("cap_dollars") or 0)
        if est["cents"] > cap * 100 and not body.get("yes"):
            raise NeedsYes(f"{what} would cost about {running.fmt_cents(est['cents'])}, more than your limit "
                           f"(${cap:.2f})", est)

    if studio.jobs.busy() and path != "/api/estimate":
        raise StudioError("something is already running: wait for it to finish")
    if path == "/api/run":
        examples = p.examples()
        if body.get("only"):
            want = {str(x) for x in body["only"]}
            examples = [e for e in examples if str(e["n"]) in want]
        if not examples:
            raise StudioError(f"no examples to run: write {p.examples_path().relative_to(p.root)}")
        ov = body.get("lm") or {}
        guard(running.estimate(p, examples, ov), "This run")
        label = (f"Trying {', '.join(f'{k} on {v}' for k, v in ov.items())}" if ov else "Running") + \
            f" {len(examples)} example{'s' if len(examples) != 1 else ''}"
        return studio.jobs.start("run", label, lambda prog: running.run_examples(
            p, examples, overrides=ov, origin="dashboard", progress=prog))
    if path == "/api/estimate":
        snap = snapshot(p)
        if body.get("action") == "compare":
            return running.estimate_compare(p, snap, before=body.get("before"), overrides=body.get("lm") or {},
                                            use_all=bool(body.get("all")))
        if body.get("action") == "blame":
            return running.estimate_blame(p, snap)
        row = next((r for r in snap["rows"] if str(r["n"]) == str(body.get("n"))), None)
        if body.get("action") == "why" and row is not None:
            return running.estimate_why(p, row)
        raise StudioError("estimate what? why (with n), blame, or compare")
    if path == "/api/why":
        snap = snapshot(p)
        row = next((r for r in snap["rows"] if str(r["n"]) == str(body["n"])), None)
        if row is None:
            raise StudioError(f"no example #{body['n']}")
        guard(running.estimate_why(p, row), "Finding the step at fault")
        return studio.jobs.start("why", f"What if, on #{body['n']}", lambda prog: running.why(p, body["n"]))
    if path == "/api/blame":
        guard(running.estimate_blame(p, snapshot(p)), "Finding the step to fix first")
        def do(prog: Callable[[int, int], None]) -> Any:
            res = running.blame(p, progress=prog)
            p.ensure_dirs()
            (p.studio_dir / "blame.json").write_text(json.dumps(res, default=str, indent=1))
            return res
        return studio.jobs.start("blame", "Finding the step to fix first", do)
    if path == "/api/compare":
        ov = body.get("lm") or {}
        guard(running.estimate_compare(p, snapshot(p), before=body.get("before"), overrides=ov,
                                       use_all=bool(body.get("all"))), "This comparison")
        return studio.jobs.start("compare", "Comparing", lambda prog: running.compare(
            p, before=body.get("before"), overrides=ov, use_all=bool(body.get("all"))))
    raise StudioError(f"no such action: {path}")


def _free_port(host: str, port: int) -> int:
    for p in range(port, port + 40):
        with socket.socket(socket.AF_INET6 if ":" in host else socket.AF_INET) as s:
            # as the server will bind it: a port its last run just closed is free to use again
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((host, p))
                return p
            except OSError:
                continue
    raise StudioError(f"no free port from {port} on {host}")


def _addresses(host: str) -> List[str]:
    """This computer's other addresses a link can use (when listening on all of them)."""
    if host not in ("0.0.0.0", "::", ""):
        return []
    found: List[str] = []
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            addr = info[4][0]
            if not addr.startswith("127.") and addr not in found:
                found.append(addr)
    except OSError:
        pass
    try:                                  # the address used to reach other machines (no packet is sent)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("100.100.100.100", 53))
            addr = s.getsockname()[0]
            if not addr.startswith("127.") and addr not in found:
                found.append(addr)
    except OSError:
        pass
    return found


def _loopback(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host in ("localhost",)


def serve(project: Optional[Project], *, host: str = "127.0.0.1", port: int = 7781, folder: Optional[Path] = None,
          open_browser: bool = True, ready: Optional[Callable[[str], None]] = None) -> None:
    folder = Path(folder or os.getcwd()).resolve()
    if project is not None:
        os.chdir(project.root)
        project.ensure_dirs()
    # a link that stays the same across restarts: FUNCTAI_STUDIO_KEY (at least 16 characters)
    fixed = os.environ.get("FUNCTAI_STUDIO_KEY", "")
    key = None if _loopback(host) else (fixed if len(fixed) >= 16 else secrets.token_urlsafe(18))
    studio = Studio(project, folder, key)
    port = _free_port(host, port)
    httpd = http.server.ThreadingHTTPServer((host, port), make_handler(studio))
    shown = "localhost" if _loopback(host) else (socket.gethostname() if host in ("0.0.0.0", "::", "") else host)
    url = f"http://{shown}:{port}/" + (f"?key={key}" if key else "")
    what = f"{project.name} ({project.file.name})" if project else str(folder)
    print(f"FunctAI Studio: {what}\n  {url}\n  (Ctrl+C stops it)", file=sys.stderr, flush=True)
    if not key:
        print("  Only this computer can open it. From another one: functai studio open --host 0.0.0.0 (it then "
              "prints links with a key).", file=sys.stderr, flush=True)
    else:
        for other in _addresses(host):
            print(f"  http://{other}:{port}/?key={key}", file=sys.stderr, flush=True)
        print(f"  Plain http only: a name ending in .ts.net opens only over https, which Tailscale can give it:\n"
              f"    tailscale serve --bg --https={port + 10000} http://127.0.0.1:{port}\n"
              f"  then https://<this computer>.<your tailnet>.ts.net:{port + 10000}/?key={key}",
              file=sys.stderr, flush=True)
    if ready:
        ready(url)
    if open_browser and (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
                         or sys.platform in ("darwin", "win32")):
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
