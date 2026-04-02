#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project-local Python enforcement for UnitPort.

Runtime entrypoints must execute under the repository-local ``.venv311``
interpreter only. If launched from any other interpreter and the project venv
exists, the process is transparently re-executed under that interpreter.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

_REEXEC_ENV_KEY = "UNITPORT_VENV_REEXEC"


def expected_project_python(project_root: Path) -> Path:
    """Return the required interpreter path for this repository."""
    root = Path(project_root).resolve()
    if platform.system() == "Windows":
        return root / ".venv311" / "Scripts" / "python.exe"
    return root / ".venv311" / "bin" / "python"


def ensure_project_venv(project_root: Path, script_path: Path | None = None) -> None:
    """
    Enforce execution under the repository-local ``.venv311`` interpreter.

    Behavior:
    - If already running under the expected interpreter: no-op.
    - If not, and the expected interpreter exists: re-exec the current script
      under the project interpreter and terminate this process with the child
      exit code.
    - If not, and the expected interpreter does not exist: fail with a clear
      error message.
    """
    expected = expected_project_python(project_root).resolve()
    actual = Path(sys.executable).resolve()
    if actual == expected:
        return

    if not expected.exists():
        raise RuntimeError(
            "UnitPort must be launched with the project-local .venv311 interpreter,\n"
            "but that interpreter does not exist.\n"
            f"Expected: {expected}\n"
            f"Actual:   {actual}"
        )

    if os.environ.get(_REEXEC_ENV_KEY) == "1":
        raise RuntimeError(
            "UnitPort attempted to re-execute under the project-local .venv311 interpreter,\n"
            "but the process is still running under a different interpreter.\n"
            f"Expected: {expected}\n"
            f"Actual:   {actual}"
        )

    if script_path is None:
        script = Path(sys.argv[0] or "").resolve()
    else:
        script = Path(script_path).resolve()

    if not script.exists():
        raise RuntimeError(
            "UnitPort must be launched with the project-local .venv311 interpreter.\n"
            "Automatic re-execution failed because the current script path could not be resolved.\n"
            f"Expected interpreter: {expected}\n"
            f"Actual interpreter:   {actual}\n"
            f"Script path:          {script}"
        )

    env = os.environ.copy()
    env[_REEXEC_ENV_KEY] = "1"
    cmd = [str(expected), str(script), *sys.argv[1:]]
    raise SystemExit(subprocess.call(cmd, env=env))
