# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""SB3 subprocess orchestrator — Popen + line protocol + cancel.

REWRITE-WITH-DEMO-REF from DEMO ``training_process.py`` (multiprocessing.Queue
version) — Stage 10 D3 decision flips it to ``subprocess.Popen`` + a
stdout line protocol so:

  * UnitPort's main process stays decoupled from the SB3 / mujoco / torch
    import graph. Spawning that under ``multiprocessing`` re-imports the
    parent's PyQt6 + SDK + register state, which is slow and fragile;
    a fresh Python process loads only what the SB3 entry script needs.
  * The cancel path is identical to :class:`IsaacLabBackend` — Windows
    ``taskkill /F /T /PID`` / POSIX ``killpg(SIGTERM)`` — so RELEASE has
    one playbook for all training backends.
  * Logs flow through the same MSG_LOG / MSG_PROGRESS / MSG_FINISHED
    protocol the UI already consumes for Isaac Lab.

Spec injection: the child reads a single JSON document from stdin then
shuts the input pipe, so the parent doesn't need to keep stdin open.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from threading import Event
from typing import Any, Callable, Dict, List, Optional

from application.training._sdk.logging_bridge import (
    train_log_error,
    train_log_info,
    train_log_warning,
)


# ---------------------------------------------------------------------------
# Message types — same as IsaacLabBackend so the UI can consume both
# ---------------------------------------------------------------------------

MSG_LOG = "log"
MSG_PROGRESS = "progress"
MSG_METRICS = "metrics"
MSG_FINISHED = "finished"
MSG_ERROR = "error"
MSG_CANCELLED = "cancelled"


# ---------------------------------------------------------------------------
# Line protocol — what the entry script emits on stdout
# ---------------------------------------------------------------------------

_PROGRESS_RE = re.compile(
    r"\[UnitPort\]\[PROGRESS\]\s+"
    r"step=(?P<step>\d+)\s+total=(?P<total>\d+)"
    r"(?:\s+unit=(?P<unit>\w+))?"
    r"\s+reward=(?P<reward>[-+]?[\d.]+|nan)"
)
_METRICS_RE = re.compile(r"\[UnitPort\]\[METRICS\]\s+(?P<json>\{.*\})")
_FINISHED_RE = re.compile(r"\[UnitPort\]\[FINISHED\]\s+(?P<json>\{.*\})")
_ERROR_RE = re.compile(r"\[UnitPort\]\[ERROR\]\s+(?P<msg>.+)$")
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class SB3SubprocessBackend:
    """Drive an SB3 training run in a child Python process.

    Lifecycle:
        ``run_blocking()`` (synchronous, Task thread):
            1. spawn child via ``subprocess.Popen`` with stdin pipe
            2. write spec JSON + close stdin
            3. tail stdout line-by-line, emitting messages via
               ``on_message(msg)``
            4. ``proc.wait()``; emit final MSG_FINISHED / MSG_ERROR
        ``cancel()`` (any thread):
            * Set the cancel event
            * Kill the process tree (Windows taskkill / POSIX killpg)

    Args:
        spec: dict (will be JSON-serialized to stdin). Schema:
            {"spec": <TrainingSpec.to_dict() output>,
             "run_id": str,
             "total_timesteps": int|None,
             "export_bundle": bool}
        run_id: unique identifier for the run (used in log prefix).
        python: Python interpreter to run the entry script under.
            Defaults to ``sys.executable`` (the same venv as Studio).
        cwd: working directory for the child. Defaults to
            ``RELEASE/src/`` so ``import application`` resolves.
        env: optional environment overrides.
        timeout_sec: max wall-clock time for the run; child is force-
            killed past this. ``None`` (default) disables.
    """

    def __init__(
        self,
        spec: Dict[str, Any],
        run_id: str,
        *,
        python: Optional[str] = None,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        timeout_sec: Optional[float] = None,
    ) -> None:
        self.spec = dict(spec or {})
        self.run_id = str(run_id)
        self.python = python or sys.executable
        # Default cwd: RELEASE/src so child can `import application`.
        if cwd is None:
            cwd = str(Path(__file__).resolve().parents[3])
        self.cwd = cwd
        self.env = env
        self.timeout_sec = timeout_sec

        self.on_message: Optional[Callable[[Dict[str, Any]], None]] = None
        self._process: Optional[subprocess.Popen] = None
        self._cancelled = Event()
        self._exit_code: Optional[int] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_blocking(self) -> None:
        """Synchronous run — spawn + tail + wait. Caller's thread blocks."""
        self._cancelled.clear()
        self._exit_code = None
        try:
            self._spawn()
            self._send_stdin()
            self._tail_stdout()
        finally:
            self._wait_with_timeout()

    def cancel(self) -> None:
        """Kill the process tree and mark the run cancelled."""
        self._cancelled.set()
        proc = self._process
        if proc is None or proc.poll() is not None:
            return
        pid = proc.pid
        train_log_info(self.run_id, f"Terminating SB3 child (pid={pid})")
        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(pid)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    check=False,
                )
            else:
                import signal
                try:
                    pgid = os.getpgid(pid)
                except ProcessLookupError:
                    pgid = pid
                try:
                    os.killpg(pgid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(pgid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
        except Exception as exc:  # noqa: BLE001
            train_log_error(self.run_id, f"Failed to kill SB3 child: {exc}")
            try:
                proc.kill()
            except Exception:
                pass

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    @property
    def exit_code(self) -> Optional[int]:
        return self._exit_code

    def build_command(self) -> List[str]:
        """Construct the child command line.

        Uses ``-u`` (unbuffered stdout) so progress lines reach the parent
        without buffering. ``-m`` so the entry module resolves through
        Python's import machinery.
        """
        return [
            self.python,
            "-u",
            "-m",
            "application.training.launcher.sb3_entry",
        ]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _emit(self, msg: Dict[str, Any]) -> None:
        if self.on_message is None:
            return
        try:
            self.on_message(msg)
        except Exception as exc:  # noqa: BLE001
            train_log_error(self.run_id, f"on_message callback raised: {exc}")

    def _spawn(self) -> None:
        cmd = self.build_command()
        train_log_info(self.run_id, f"Spawning SB3 subprocess: {' '.join(cmd)}")
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        # Make sure RELEASE/src is on the path even when cwd differs.
        path_extra = str(Path(__file__).resolve().parents[3])
        if env.get("PYTHONPATH"):
            env["PYTHONPATH"] = path_extra + os.pathsep + env["PYTHONPATH"]
        else:
            env["PYTHONPATH"] = path_extra
        if self.env:
            env.update(self.env)

        kwargs: Dict[str, Any] = {
            "cwd": self.cwd,
            "env": env,
            "stdin": subprocess.PIPE,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,  # merge so order is preserved
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
            "bufsize": 1,
        }
        if os.name == "nt":
            # Detach console + new process group so taskkill /T can sweep
            # the whole subtree without touching the parent.
            kwargs["creationflags"] = (
                getattr(subprocess, "CREATE_NO_WINDOW", 0)
                | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            )
        else:
            # New session so killpg targets only the child tree.
            kwargs["start_new_session"] = True
        self._process = subprocess.Popen(cmd, **kwargs)

    def _send_stdin(self) -> None:
        proc = self._process
        if proc is None or proc.stdin is None:
            return
        try:
            payload = json.dumps(self.spec, ensure_ascii=False)
            proc.stdin.write(payload)
            proc.stdin.write("\n")
            proc.stdin.flush()
        finally:
            try:
                proc.stdin.close()
            except Exception:
                pass

    def _tail_stdout(self) -> None:
        proc = self._process
        if proc is None or proc.stdout is None:
            return
        for raw in proc.stdout:
            if self._cancelled.is_set():
                break
            line = (raw or "").rstrip("\r\n")
            if not line:
                continue
            self._process_line(line)

    def _process_line(self, line: str) -> None:
        clean = _ANSI_RE.sub("", line)
        # Preserve the raw line in the LOG stream so downstream UI sees
        # the verbatim stdout — exactly the IsaacLab pattern.
        self._emit({"type": MSG_LOG, "data": clean})

        m = _PROGRESS_RE.search(clean)
        if m:
            try:
                step = int(m.group("step"))
                total = int(m.group("total"))
                rew = m.group("reward")
                rew_val = float("nan") if rew == "nan" else float(rew)
            except (TypeError, ValueError):
                return
            unit = (m.group("unit") or "step")
            # 7-tuple: (count, total, reward, best_reward, ep_len, unit). The
            # 6th slot historically carried an unused "training" phase string;
            # it now carries the progress unit ("iter"/"step") so the UI label
            # matches the budget. (sb3_task tolerates the 6- or 7-tuple.)
            self._emit({
                "type": MSG_PROGRESS,
                "data": (step, total, rew_val, rew_val, 0.0, unit),
            })
            return

        m = _METRICS_RE.search(clean)
        if m:
            try:
                payload = json.loads(m.group("json"))
            except json.JSONDecodeError:
                return
            self._emit({"type": MSG_METRICS, "data": payload})
            return

        m = _FINISHED_RE.search(clean)
        if m:
            try:
                payload = json.loads(m.group("json"))
            except json.JSONDecodeError:
                payload = {"raw": m.group("json")}
            self._emit({"type": MSG_FINISHED, "data": payload})
            return

        m = _ERROR_RE.search(clean)
        if m:
            self._emit({"type": MSG_ERROR, "data": m.group("msg")})
            return

    def _wait_with_timeout(self) -> None:
        proc = self._process
        if proc is None:
            return
        try:
            self._exit_code = proc.wait(timeout=self.timeout_sec)
        except subprocess.TimeoutExpired:
            train_log_warning(
                self.run_id,
                f"SB3 child exceeded timeout ({self.timeout_sec}s) — killing",
            )
            self.cancel()
            try:
                self._exit_code = proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._exit_code = -1

        if self._cancelled.is_set():
            self._emit({"type": MSG_CANCELLED, "data": self.run_id})
        elif self._exit_code != 0:
            self._emit({
                "type": MSG_ERROR,
                "data": f"SB3 child exited non-zero ({self._exit_code})",
            })


__all__ = [
    "MSG_LOG",
    "MSG_PROGRESS",
    "MSG_METRICS",
    "MSG_FINISHED",
    "MSG_ERROR",
    "MSG_CANCELLED",
    "SB3SubprocessBackend",
]
