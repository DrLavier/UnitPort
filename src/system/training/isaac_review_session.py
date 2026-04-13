"""UnitPort-side manager for the persistent Isaac Review Session.

Per ``knowledge_base/IsaacSim_design.yaml`` §2 and §4.

Responsibilities
----------------
1. Spawn the ``il_review_session.py`` subprocess with the right Isaac
   Lab launcher (mirrors how IsaacLabBackend launches the train /
   play subprocesses today).
2. Maintain a write pipe to the subprocess's stdin for IPC commands.
3. Run a daemon thread that reads the subprocess's stdout, parses
   structured JSON messages, and updates session state / fires Qt
   signals.
4. Detect subprocess crashes / EOF and transition to an error state
   so the UI can show "Retry Viewer".
5. Provide a Pythonic API to the rest of UnitPort:
       start(env_cfg_file, num_envs=1)
       stop()
       load_checkpoint(path, auto_mode=True)
       set_command(vx, vy, wz)
       set_control_mode(mode)         # AUTO | MANUAL
       reset() / pause() / resume() / snapshot(label)
       is_ready() / state() / current_checkpoint()
6. Auto-cleanup at UnitPort exit via ``atexit``.

The class is implemented as a process-wide singleton because Isaac Sim
licenses one Kit instance per machine and we don't want two viewers
fighting for the GPU.  Use :func:`get_review_session` to obtain it.

Threading model
---------------
- Main thread (Qt event loop): start/stop/load_checkpoint/set_command
  are called from here.  All public methods are non-blocking — they
  push commands into the stdin pipe and return immediately.
- Stdout reader thread: a daemon ``threading.Thread`` that drains the
  subprocess's stdout line by line, filters by the
  ``[UNITPORT_REVIEW_MSG]`` sentinel, and updates session state.
  Communicates state changes to the main thread via Qt signals.

State machine
-------------
``stopped`` → ``starting`` → ``ready`` → (``stopped``)
                ↘                ↗
                  ``error`` ─────
"""

from __future__ import annotations

import atexit
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

try:
    from PySide6.QtCore import QObject, QTimer, Signal
except Exception:  # pragma: no cover — non-Qt unit tests
    QObject = object  # type: ignore[assignment, misc]
    QTimer = None  # type: ignore[assignment, misc]

    class _StubSignal:  # type: ignore[no-redef]
        def __init__(self, *_a, **_kw): pass
        def connect(self, *_a, **_kw): pass
        def emit(self, *_a, **_kw): pass

    Signal = _StubSignal  # type: ignore[assignment, misc]


_MSG_PREFIX = "[UNITPORT_REVIEW_MSG]"


# ---------------------------------------------------------------------------
# Session states
# ---------------------------------------------------------------------------

STATE_STOPPED  = "stopped"
STATE_STARTING = "starting"
STATE_READY    = "ready"
STATE_ERROR    = "error"


# ---------------------------------------------------------------------------
# IsaacReviewSession singleton
# ---------------------------------------------------------------------------

class IsaacReviewSession(QObject):
    """Long-lived manager for the il_review_session.py subprocess."""

    # Qt signals — emitted on the main thread thanks to QueuedConnection
    state_changed   = Signal(str)            # one of STATE_*
    log_line        = Signal(str)            # raw stdout line (sans prefix)
    checkpoint_loaded = Signal(str, bool)    # path, ok
    stage_changed   = Signal(int, str, float, float, float)  # idx, name, vx, vy, wz
    telemetry       = Signal(dict)           # {fps, step_count, ...}
    error_occurred  = Signal(str, str)       # where, detail

    _instance: Optional["IsaacReviewSession"] = None
    _instance_lock = threading.Lock()

    def __init__(self):
        super().__init__()
        self._proc: Optional[subprocess.Popen] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._reader_stop = threading.Event()
        self._lock = threading.RLock()
        self._state: str = STATE_STOPPED
        self._last_error: str = ""
        self._current_checkpoint: str = ""
        self._control_mode: str = "AUTO"
        self._ready_info: Dict[str, Any] = {}
        # Buffer recent stdout lines for diagnostic display
        self._log_buffer: List[str] = []
        self._log_buffer_max = 200
        # T6: CommandBus poll timer + last sent values for edge detection.
        # Created lazily on first start() so headless unit tests don't
        # need a Qt event loop.
        self._cmd_poll_timer = None  # QTimer
        self._last_sent_command: Optional[tuple] = None  # (vx, vy, vyaw)
        # T8: edge-detection cache for gamepad button events
        self._last_button_state: Dict[str, bool] = {}
        # T9: button → action binding map (loaded lazily, refreshed
        # each time start() is called so user.ini edits take effect
        # without restarting UnitPort).
        self._button_bindings_cache: Optional[Dict[str, str]] = None
        # Queued load_checkpoint — set when load_checkpoint() is called
        # before the session reaches READY state.  Fired automatically
        # when state transitions to READY.  This is what powers the
        # "auto-open viewer on Export Review" UX: clicking Review
        # before Open Viewer transparently kicks off start() AND queues
        # the checkpoint to load as soon as Kit finishes booting.
        self._pending_load: Optional[tuple] = None  # (path, auto_mode)
        atexit.register(self._atexit_cleanup)

    # ------------------------------------------------------------------
    # Singleton accessor
    # ------------------------------------------------------------------

    @classmethod
    def instance(cls) -> "IsaacReviewSession":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    # ------------------------------------------------------------------
    # State accessors
    # ------------------------------------------------------------------

    def state(self) -> str:
        with self._lock:
            return self._state

    def is_ready(self) -> bool:
        return self.state() == STATE_READY

    def is_alive(self) -> bool:
        with self._lock:
            return self._proc is not None and self._proc.poll() is None

    def current_checkpoint(self) -> str:
        with self._lock:
            return self._current_checkpoint

    def control_mode(self) -> str:
        with self._lock:
            return self._control_mode

    def last_error(self) -> str:
        with self._lock:
            return self._last_error

    def ready_info(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._ready_info)

    def recent_logs(self, n: int = 20) -> List[str]:
        with self._lock:
            return list(self._log_buffer[-n:])

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, env_cfg_file: str, num_envs: int = 1,
              isaac_lab_launcher: Optional[str] = None,
              isaac_lab_python: Optional[str] = None,
              extra_env: Optional[Dict[str, str]] = None) -> bool:
        """Spawn the il_review_session subprocess.

        Parameters
        ----------
        env_cfg_file
            Absolute path to the UnitPort-compiled @configclass file.
            Determines obs/action layout and env contents.  Required.
        num_envs
            Number of envs in the viewer.  Default 1 (single robot).
        isaac_lab_launcher / isaac_lab_python
            Paths to ``isaaclab.bat`` (Windows) or the Isaac venv python.
            If both None, auto-detected via
            :func:`isaac_lab_backend.detect_isaac_lab`.
        extra_env
            Optional environment variable overrides.

        Returns
        -------
        True if the subprocess was launched (state → starting),
        False otherwise.
        """
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                # Already running — refuse to start a second one
                return False
            if not env_cfg_file or not os.path.isfile(env_cfg_file):
                self._set_error("start", f"compiled config not found: {env_cfg_file}")
                return False

            cmd = self._build_command(
                env_cfg_file=env_cfg_file,
                num_envs=num_envs,
                isaac_lab_launcher=isaac_lab_launcher,
                isaac_lab_python=isaac_lab_python,
            )
            if cmd is None:
                self._set_error("start", "could not build review session command — Isaac Lab not detected")
                return False

            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            env["OMNI_KIT_ACCEPT_EULA"] = "YES"
            env["PYTHONIOENCODING"] = "utf-8"
            if extra_env:
                env.update({str(k): str(v) for k, v in extra_env.items()})

            popen_kwargs = dict(
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,           # line-buffered for live stdout
                env=env,
            )
            if os.name == "nt":
                popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                popen_kwargs["start_new_session"] = True

            try:
                self._proc = subprocess.Popen(cmd, **popen_kwargs)
            except Exception as exc:
                self._set_error("start", f"subprocess.Popen failed: {exc}")
                return False

            self._set_state(STATE_STARTING)
            self._current_checkpoint = ""
            self._ready_info = {}
            self._button_bindings_cache = self._load_button_bindings()
            self._reader_stop.clear()
            self._reader_thread = threading.Thread(
                target=self._stdout_reader_loop,
                daemon=True,
                name="IsaacReviewSessionReader",
            )
            self._reader_thread.start()
            return True

    def stop(self, timeout_s: float = 10.0) -> None:
        """Send shutdown via IPC, wait, then escalate to SIGTERM/SIGKILL."""
        with self._lock:
            proc = self._proc
        if proc is None:
            return

        # 1. Polite shutdown via IPC
        self._send_raw({"op": "shutdown"})
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            pass

        # 2. SIGTERM
        if proc.poll() is None:
            try:
                if os.name == "nt":
                    proc.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    proc.terminate()
            except Exception:
                pass
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                pass

        # 3. SIGKILL
        if proc.poll() is None:
            try:
                proc.kill()
            except Exception:
                pass
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                pass

        with self._lock:
            self._reader_stop.set()
            self._proc = None
            self._set_state(STATE_STOPPED)

    # ------------------------------------------------------------------
    # IPC commands (non-blocking)
    # ------------------------------------------------------------------

    def load_checkpoint(self, path: str, auto_mode: bool = True) -> bool:
        """Load a policy checkpoint into the live actor.

        Behaviour:
          - If state == READY → send IPC immediately, return True
          - If state in (STARTING, STOPPED) → queue the request, return
            True. Will fire automatically when state transitions to
            READY (within ~30-60 s for a cold start).
          - If state == ERROR → return False (caller should restart
            the session first).
        """
        if self._state == STATE_ERROR:
            return False
        # Queue the load if subprocess isn't ready yet — this powers
        # the "click Review without first clicking Open Viewer" UX.
        if not self.is_alive() or self._state != STATE_READY:
            with self._lock:
                self._pending_load = (str(path), bool(auto_mode))
            return True
        return self._send_raw({
            "op": "load_checkpoint",
            "path": str(path),
            "auto_mode": bool(auto_mode),
        })

    def queue_load_checkpoint(self, path: str, auto_mode: bool = True) -> None:
        """Always-queue variant — useful when you've just called start()
        and want to make sure the load fires after READY even if a
        previous _pending_load existed."""
        with self._lock:
            self._pending_load = (str(path), bool(auto_mode))

    def set_command(self, vx: float, vy: float, wz: float) -> bool:
        if not self.is_alive() or self.control_mode() != "MANUAL":
            return False
        return self._send_raw({"op": "command", "vx": float(vx),
                               "vy": float(vy), "wz": float(wz)})

    def set_control_mode(self, mode: str) -> bool:
        mode = str(mode).upper()
        if mode not in ("AUTO", "MANUAL"):
            return False
        if not self.is_alive():
            return False
        with self._lock:
            self._control_mode = mode
        return self._send_raw({"op": "control_mode", "mode": mode})

    def reset(self) -> bool:
        return self._send_raw({"op": "reset"})

    def pause(self) -> bool:
        return self._send_raw({"op": "pause"})

    def resume(self) -> bool:
        return self._send_raw({"op": "resume"})

    def snapshot(self, label: str = "") -> bool:
        return self._send_raw({"op": "snapshot", "label": str(label)})

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _send_raw(self, payload: dict) -> bool:
        with self._lock:
            proc = self._proc
        if proc is None or proc.stdin is None or proc.poll() is not None:
            return False
        try:
            line = json.dumps(payload, ensure_ascii=False) + "\n"
            proc.stdin.write(line)
            proc.stdin.flush()
            return True
        except Exception as exc:
            self._set_error("send_raw", str(exc))
            return False

    def _build_command(self, *, env_cfg_file: str, num_envs: int,
                       isaac_lab_launcher: Optional[str],
                       isaac_lab_python: Optional[str]) -> Optional[List[str]]:
        """Construct the full command line for the subprocess."""
        # Auto-detect Isaac Lab installation if paths weren't supplied.
        # NOTE: ``detect_isaac_lab()`` returns a plain ``Dict[str, str]``
        # with keys ``path`` / ``python`` / ``launcher`` — NOT an object
        # with attributes.  Use dict access throughout.
        if not isaac_lab_launcher and not isaac_lab_python:
            try:
                from src.system.training.isaac_lab_backend import detect_isaac_lab
                detection = detect_isaac_lab()
                if detection is None:
                    return None
                isaac_lab_launcher = detection.get("launcher", "") or None
                isaac_lab_python = detection.get("python", "") or None
            except Exception:
                return None

        script = str(Path(__file__).resolve().parent / "il_review_session.py")
        args: List[str] = [
            "--task", "UnitPort-Review-v0",
            "--unitport_config", str(env_cfg_file),
            "--num_envs", str(int(num_envs)),
        ]

        if isaac_lab_launcher and Path(isaac_lab_launcher).exists():
            if os.name == "nt" and isaac_lab_launcher.lower().endswith(".bat"):
                # Same Windows wrapper trick as IsaacLabConfig._build_launcher_cmd
                wrapper = self._write_win_wrapper(
                    isaac_lab_launcher, script, args
                )
                return ["cmd", "/c", wrapper]
            return [isaac_lab_launcher, "-p", script, *args]

        if isaac_lab_python and Path(isaac_lab_python).exists():
            return [isaac_lab_python, script, *args]

        return None

    def _write_win_wrapper(self, isaaclab_bat: str, script: str,
                           args: List[str]) -> str:
        """Mirror IsaacLabConfig._write_win_wrapper for review subprocess."""
        import tempfile
        bat_dir = tempfile.gettempdir()
        bat_path = os.path.join(bat_dir, f"unitport_review_{os.getpid()}.bat")
        # Build a wrapper that resolves Python from isaaclab.bat env, then
        # invokes it with quoted paths so spaces in the install path don't
        # break the call.
        argstr = " ".join(f'"{a}"' if " " in a else a for a in args)
        wrapper = (
            f'@echo off\r\n'
            f'call "{isaaclab_bat}" -p "{script}" {argstr}\r\n'
        )
        with open(bat_path, "w", encoding="utf-8") as fh:
            fh.write(wrapper)
        return bat_path

    def _stdout_reader_loop(self) -> None:
        """Daemon thread: parse subprocess stdout into structured events."""
        proc = self._proc
        if proc is None or proc.stdout is None:
            return

        try:
            for line in proc.stdout:
                if self._reader_stop.is_set():
                    break
                line = line.rstrip("\n")
                if not line:
                    continue

                # Buffer raw line for diagnostic display
                with self._lock:
                    self._log_buffer.append(line)
                    if len(self._log_buffer) > self._log_buffer_max:
                        self._log_buffer = self._log_buffer[-self._log_buffer_max:]

                try:
                    self.log_line.emit(line)
                except Exception:
                    pass

                # Structured messages start with our sentinel
                if not line.startswith(_MSG_PREFIX):
                    continue
                payload_str = line[len(_MSG_PREFIX):]
                try:
                    payload = json.loads(payload_str)
                except json.JSONDecodeError:
                    continue
                self._dispatch_message(payload)
        except Exception:
            pass

        # Subprocess died — flip to error state if we weren't already stopping
        with self._lock:
            if self._state != STATE_STOPPED and not self._reader_stop.is_set():
                self._set_error("subprocess", "review session exited unexpectedly")

    def _dispatch_message(self, payload: dict) -> None:
        msg = str(payload.get("msg", "")).strip()
        if msg == "ready":
            with self._lock:
                self._ready_info = dict(payload)
                self._set_state(STATE_READY)
        elif msg == "loaded":
            ok = bool(payload.get("ok", False))
            path = str(payload.get("path", ""))
            if ok:
                with self._lock:
                    self._current_checkpoint = path
                    cm = str(payload.get("control_mode", self._control_mode)).upper()
                    if cm in ("AUTO", "MANUAL"):
                        self._control_mode = cm
            try:
                self.checkpoint_loaded.emit(path, ok)
            except Exception:
                pass
        elif msg == "stage_changed":
            try:
                self.stage_changed.emit(
                    int(payload.get("index", 0)),
                    str(payload.get("name", "")),
                    float(payload.get("vx", 0.0)),
                    float(payload.get("vy", 0.0)),
                    float(payload.get("wz", 0.0)),
                )
            except Exception:
                pass
        elif msg == "telemetry":
            try:
                self.telemetry.emit(dict(payload))
            except Exception:
                pass
        elif msg == "error":
            where = str(payload.get("where", ""))
            detail = str(payload.get("detail", ""))
            with self._lock:
                self._last_error = f"{where}: {detail}"
            try:
                self.error_occurred.emit(where, detail)
            except Exception:
                pass
        elif msg == "shutdown_complete":
            with self._lock:
                self._set_state(STATE_STOPPED)

    def _set_state(self, new_state: str) -> None:
        # Caller must hold _lock OR be the constructor.
        if self._state == new_state:
            return
        self._state = new_state
        # Start/stop the CommandBus poll timer based on lifecycle state
        if new_state == STATE_READY:
            self._start_command_bus_polling()
            # Fire any queued checkpoint load that arrived during boot.
            self._fire_pending_load_if_any()
        elif new_state in (STATE_STOPPED, STATE_ERROR):
            self._stop_command_bus_polling()
            # Drop the queued load — restart will need a new request
            with self._lock:
                self._pending_load = None
        try:
            self.state_changed.emit(new_state)
        except Exception:
            pass

    def _fire_pending_load_if_any(self) -> None:
        """If a load_checkpoint was queued during STARTING, fire it now."""
        with self._lock:
            pending = self._pending_load
            self._pending_load = None
        if pending is None:
            return
        path, auto_mode = pending
        self._send_raw({
            "op": "load_checkpoint",
            "path": str(path),
            "auto_mode": bool(auto_mode),
        })

    # ------------------------------------------------------------------
    # T6: CommandBus polling — forwards live gamepad/keyboard input
    # ------------------------------------------------------------------

    def _start_command_bus_polling(self) -> None:
        """Begin polling GlobalInputManager's CommandBus at ~50 Hz.

        Reads (vx, vy, vyaw) from the bus and forwards as IPC
        ``op: command`` messages — but only when control_mode is
        MANUAL.  Edge detection prevents flooding the subprocess with
        identical commands when the user isn't actively moving the
        sticks (gamepad publishes the smoothed value continuously).
        """
        if QTimer is None:
            return
        if self._cmd_poll_timer is not None:
            return
        try:
            self._cmd_poll_timer = QTimer(self)
            self._cmd_poll_timer.setInterval(20)  # 50 Hz
            self._cmd_poll_timer.timeout.connect(self._poll_command_bus)
            self._cmd_poll_timer.start()
        except Exception:
            self._cmd_poll_timer = None

    def _stop_command_bus_polling(self) -> None:
        if self._cmd_poll_timer is not None:
            try:
                self._cmd_poll_timer.stop()
                self._cmd_poll_timer.deleteLater()
            except Exception:
                pass
            self._cmd_poll_timer = None
        self._last_sent_command = None
        self._last_button_state.clear()

    def _poll_command_bus(self) -> None:
        """Timer callback — fired ~50 times per second on the Qt main thread.

        Behaviour:
          1. If session not ready or control_mode != MANUAL → no-op for axes,
             but still poll buttons (so AUTO-mode users can still hit
             "load_next" / "reset" via gamepad).
          2. Read (vx, vy, vyaw) from GlobalInputManager.bus, edge-detect,
             forward as ``op: command`` IPC if changed.
          3. Read button event keys (``__btn_*``) from the bus, fire IPC
             ops on press transitions (T8 integration).
        """
        try:
            from src.system.runtime.global_input_manager import (
                get_global_input_manager,
            )
        except Exception:
            return
        try:
            mgr = get_global_input_manager()
        except Exception:
            return

        # Forward velocity command (only in MANUAL mode)
        if self.control_mode() == "MANUAL":
            try:
                values = mgr.get_live_values()
                vx = float(values.get("vx", 0.0))
                vy = float(values.get("vy", 0.0))
                vyaw = float(values.get("vyaw", 0.0))
                triple = (round(vx, 4), round(vy, 4), round(vyaw, 4))
                if triple != self._last_sent_command:
                    self._last_sent_command = triple
                    self._send_raw({
                        "op": "command",
                        "vx": vx, "vy": vy, "wz": vyaw,
                    })
            except Exception:
                pass

        # Button events — see T8.  CommandBus key convention:
        #   __btn_<action>  = 1.0 while pressed, 0.0 when released.
        # We do edge detection here (0→1 = press, 1→0 = release) so
        # the IPC dispatch fires exactly once per press.
        self._poll_button_events(mgr)

    def _poll_button_events(self, mgr) -> None:
        """Read ``__btn_*`` keys from the CommandBus and dispatch IPC ops."""
        try:
            bus = getattr(mgr, "_bus", None)
            if bus is None:
                return
            with bus._lock:
                snapshot = {
                    k: v for k, v in bus._values.items() if k.startswith("__btn_")
                }
        except Exception:
            return

        for key, raw in snapshot.items():
            now_pressed = bool(raw and raw > 0.5)
            was_pressed = self._last_button_state.get(key, False)
            self._last_button_state[key] = now_pressed
            if now_pressed and not was_pressed:
                # Press transition — dispatch
                self._dispatch_button_action(key)

    # Default mapping from CommandBus button keys to IPC ops.  Phase B
    # ships these as the out-of-box defaults; users can override per-
    # button via user.ini under the [REVIEW_GAMEPAD_BINDINGS] section,
    # e.g.::
    #
    #     [REVIEW_GAMEPAD_BINDINGS]
    #     a = reset
    #     b = pause_resume
    #     x = load_next
    #     y = snapshot
    #     lb = load_next
    #     rb = load_next
    #     start = reset
    #
    # The full Controller Panel rebinding UI for these buttons is
    # tracked as Phase B-followup; the user.ini override path is the
    # supported v1 customization mechanism.
    _DEFAULT_BUTTON_BINDINGS: Dict[str, str] = {
        "__btn_a":     "reset",
        "__btn_b":     "pause_resume",
        "__btn_x":     "load_next",
        "__btn_y":     "snapshot",
    }

    def _load_button_bindings(self) -> Dict[str, str]:
        """Merge user.ini overrides into the default button binding map.

        Reads from ``[REVIEW_GAMEPAD_BINDINGS]`` in user.ini.  The
        section name is intentionally separate from the existing
        Controller Panel bindings ([CONTROLLER]) so editing one
        doesn't accidentally clobber the other.
        """
        bindings = dict(self._DEFAULT_BUTTON_BINDINGS)
        try:
            from src.system.core.config_manager import ConfigManager
            cfg = ConfigManager()
            user = cfg.user_config  # underlying configparser.ConfigParser
            section = "REVIEW_GAMEPAD_BINDINGS"
            if user.has_section(section):
                for key, action in user.items(section):
                    if not key or not action:
                        continue
                    bindings[f"__btn_{key.lower()}"] = str(action).strip().lower()
        except Exception:
            pass
        return bindings

    def _dispatch_button_action(self, btn_key: str) -> None:
        if not hasattr(self, "_button_bindings_cache") or self._button_bindings_cache is None:
            self._button_bindings_cache = self._load_button_bindings()
        action = self._button_bindings_cache.get(btn_key)
        if action is None:
            return
        try:
            if action == "reset":
                self.reset()
            elif action == "pause_resume":
                # Toggle: there's no introspection of pause state from
                # here, so we send pause+resume as separate ops and let
                # the subprocess handle the latch.  Simplest version =
                # just send pause; the user clicks again to resume via
                # the controller panel "Resume" action (future).
                self.pause()
            elif action == "snapshot":
                self.snapshot()
            elif action == "load_next":
                # The "next checkpoint" needs the bundle list — out of
                # scope for the singleton.  Emit a signal so the UI
                # layer (which knows the bundle list) can handle it.
                try:
                    self.error_occurred.emit(
                        "load_next",
                        "load_next gamepad action requires UI-side handling "
                        "(not implemented in v1)"
                    )
                except Exception:
                    pass
        except Exception:
            pass

    def _set_error(self, where: str, detail: str) -> None:
        with self._lock:
            self._last_error = f"{where}: {detail}"
            self._set_state(STATE_ERROR)
        try:
            self.error_occurred.emit(where, detail)
        except Exception:
            pass

    def _atexit_cleanup(self) -> None:
        """atexit hook — make sure the subprocess doesn't outlive UnitPort."""
        try:
            if self.is_alive():
                self.stop(timeout_s=3.0)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Module-level convenience accessor
# ---------------------------------------------------------------------------

def get_review_session() -> IsaacReviewSession:
    """Return the process-wide IsaacReviewSession singleton."""
    return IsaacReviewSession.instance()
