"""A type checker sees the API as documented: tests/typing/api.py under basedpyright."""

import shutil
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).parent


def test_the_api_type_checks_as_documented():
    exe = shutil.which("basedpyright") or str(Path(__import__("sys").executable).parent / "basedpyright")
    if not Path(exe).exists():
        pytest.skip("basedpyright is not installed (uv sync --all-groups)")
    out = subprocess.run([exe, "--level", "error", "-p", str(HERE / "typing" / "pyrightconfig.json"),
                          str(HERE / "typing" / "api.py")], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stdout + out.stderr
