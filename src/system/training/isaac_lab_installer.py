"""IsaacLabInstallThread — QThread worker for Isaac Lab + Isaac Sim installation.

Spawns ``deploy/install_isaac.py`` as a subprocess and parses its structured
stdout markers to report progress back to the UI via Qt signals.

Usage::

    thread = IsaacLabInstallThread("/path/to/install/dir")
    thread.progress.connect(my_label.setText)
    thread.step_changed.connect(on_step)    # (current, total, description)
    thread.finished_ok.connect(on_done)     # receives Isaac Lab root path str
    thread.failed.connect(on_error)         # receives error message
    thread.start()

    # To cancel:
    thread.cancel()
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QThread, Signal

log = logging.getLogger(__name__)

# Locate the installer script within the engines package.
_REPO_ROOT = Path(__file__).resolve().parents[3]  # src/system/training/ → repo root
_INSTALLER_SCRIPT = (
    _REPO_ROOT / "src" / "system" / "engines" / "isaac" / "install_script.py"
)

# Structured output markers emitted by install_isaac.py
_STEP_RE = re.compile(r"^\[STEP (\d+)/(\d+)\]\s*(.+)")
_PROGRESS_RE = re.compile(r"^\[PROGRESS\]\s*(\d+)")
_OK_RE = re.compile(r"^\[OK\]\s*(.*)")
_WARN_RE = re.compile(r"^\[WARN\]\s*(.*)")
_ERROR_RE = re.compile(r"^\[ERROR\]\s*(.*)")
_ROOT_RE = re.compile(r"^\[ISAAC_LAB_ROOT\]\s*(.*)")
_DONE_RE = re.compile(r"^\[DONE\]")


class IsaacLabInstallThread(QThread):
    """Worker thread that drives Isaac Lab installation.

    Signals
    -------
    progress(str)
        Human-readable status text (step description, OK/WARN messages).
    step_changed(int, int, str)
        Emitted when a new installation step begins: (current, total, description).
    finished_ok(str)
        Emitted on success with the absolute path to the IsaacLab root directory.
    failed(str)
        Emitted on error or cancellation with the error message.
    log_line(str)
        Every raw output line from the subprocess (for detailed log views).
    """

    progress = Signal(str)
    step_changed = Signal(int, int, str)
    finished_ok = Signal(str)
    failed = Signal(str)
    log_line = Signal(str)

    def __init__(
        self,
        install_dir: str,
        python_hint: str = "",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._install_dir = install_dir
        self._python_hint = python_hint
        self._cancelled = False
        self._isaac_lab_root: Optional[str] = None
        self._process: Optional[subprocess.Popen] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def cancel(self) -> None:
        """Request cancellation."""
        self._cancelled = True
        proc = self._process
        if proc is not None:
            try:
                proc.terminate()
            except Exception:
                pass

    @property
    def isaac_lab_root(self) -> Optional[str]:
        """Isaac Lab root directory (set on successful completion)."""
        return self._isaac_lab_root

    # ------------------------------------------------------------------
    # QThread entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        if not _INSTALLER_SCRIPT.is_file():
            self.failed.emit(
                f"Installer script not found: {_INSTALLER_SCRIPT}\n"
                "Ensure deploy/install_isaac.py exists in the project root."
            )
            return

        # Use UnitPort's own venv Python to run the installer script.
        # The installer script itself creates a *separate* venv for Isaac.
        python_exe = sys.executable
        cmd = [
            python_exe, str(_INSTALLER_SCRIPT),
            "--install-dir", self._install_dir,
        ]
        if self._python_hint:
            cmd.extend(["--python", self._python_hint])

        log.info("[isaac-installer] Launching: %s", " ".join(cmd))
        self.progress.emit("Starting Isaac Lab installation…")

        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
                cwd=str(_REPO_ROOT),
            )
            assert self._process.stdout is not None

            for raw_line in self._process.stdout:
                if self._cancelled:
                    break
                line = raw_line.rstrip()
                if not line:
                    continue

                self.log_line.emit(line)
                self._parse_line(line)

            self._process.wait()
            exit_code = self._process.returncode

            if self._cancelled:
                self.failed.emit("Installation cancelled by user.")
                return

            if exit_code != 0:
                self.failed.emit(
                    f"Installation failed (exit code {exit_code}). "
                    "Check the log output for details."
                )
                return

            if not self._isaac_lab_root:
                self.failed.emit(
                    "Installation completed but Isaac Lab root was not reported."
                )
                return

            # Register the path in the unified engine registry
            self._register_path(self._isaac_lab_root)
            self.finished_ok.emit(self._isaac_lab_root)

        except Exception as exc:
            self.failed.emit(f"Installation error: {exc}")
        finally:
            self._process = None

    # ------------------------------------------------------------------
    # Line parsing
    # ------------------------------------------------------------------

    def _parse_line(self, line: str) -> None:
        m = _STEP_RE.match(line)
        if m:
            cur, total, desc = int(m.group(1)), int(m.group(2)), m.group(3)
            self.step_changed.emit(cur, total, desc)
            self.progress.emit(f"[{cur}/{total}] {desc}")
            return

        m = _PROGRESS_RE.match(line)
        if m:
            return  # percentage — UI can use step_changed for progress bar

        m = _OK_RE.match(line)
        if m:
            self.progress.emit(f"✓ {m.group(1)}")
            return

        m = _WARN_RE.match(line)
        if m:
            self.progress.emit(f"⚠ {m.group(1)}")
            return

        m = _ERROR_RE.match(line)
        if m:
            self.progress.emit(f"✗ {m.group(1)}")
            return

        m = _ROOT_RE.match(line)
        if m:
            self._isaac_lab_root = m.group(1).strip()
            return

        if _DONE_RE.match(line):
            self.progress.emit("Installation complete!")
            return

    # ------------------------------------------------------------------
    # Path registration
    # ------------------------------------------------------------------

    @staticmethod
    def _register_path(isaac_lab_root: str) -> None:
        """Register Isaac Lab local path in the unified engine registry."""
        from src.system.engines.registry import get_engine_registry
        get_engine_registry().register_isaac_local(isaac_lab_root)
        log.info("[isaac-installer] Registered Isaac Lab path: %s", isaac_lab_root)
