"""Git checkpoints around every tool that changes things (a port of
Chattering's `checkpoints`): a snapshot of the working tree before and after,
kept as entries of the turn, so any tool's changes can be reviewed or undone.

A snapshot is the tree of every file git would track (untracked ones
included, ignored ones not), written through a temporary index: the
repository's own index, branch and working tree are never touched."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import functai


def snapshot(repo: Path) -> str:
    """The id of a git tree holding the working tree as it is now."""
    with tempfile.TemporaryDirectory() as tmp:
        env = {**os.environ, "GIT_INDEX_FILE": os.path.join(tmp, "index")}

        def git(*args: str) -> str:
            return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True,
                                  env=env).stdout.strip()

        git("add", "-A", ".")
        return git("write-tree")


def plugin(repo: "str | Path") -> functai.Plugin:
    root = Path(repo)
    p = functai.Plugin("checkpoints", version="1.0.0")
    before: dict = {}

    @p.tool_call
    def take_before(tool: Any) -> None:
        if tool.effects != "reads":
            before[(tool.path, tool.invocation)] = snapshot(root)

    @p.tool_result
    def take_after(result: Any) -> None:
        start = before.pop((result.path, result.invocation), None)
        if start is not None:
            entry = {"tool": result.name, "invocation": result.invocation, "before": start, "after": snapshot(root)}
            try:
                result.remember("checkpoint", entry)
            except ValueError:
                pass                                    # outside a conversation: nowhere to keep it

    return p


def undo(repo: "str | Path", checkpoint: dict) -> None:
    """Put back what that tool changed: files it created are removed, files it
    changed or deleted are restored as they were before it ran."""
    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout
    for line in git("diff", "--name-status", "--no-renames", checkpoint["before"], checkpoint["after"]).splitlines():
        status, _, name = line.partition("\t")
        if status == "A":
            (Path(repo) / name).unlink(missing_ok=True)
        else:
            git("restore", f"--source={checkpoint['before']}", "--worktree", "--", name)   # the index untouched
