"""IsaacLabBackend — external process orchestrator for Isaac Lab training.

Isaac Lab runs in its own environment (conda/venv with Isaac Sim deps).
UnitPort communicates via filesystem (logs, exported artifacts) and subprocess.

This is a process orchestrator, NOT a Python API integration.

Signal interface mirrors training_process.py's message protocol so the UI
can display progress identically for SB3 and Isaac Lab backends.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.system.training.isaac_lab_config import IsaacLabConfig

log = logging.getLogger(__name__)

# Message types (same as training_process.py)
MSG_LOG = "log"
MSG_PROGRESS = "progress"
MSG_METRICS = "metrics"
MSG_FINISHED = "finished"
MSG_ERROR = "error"
MSG_CANCELLED = "cancelled"


# ---------------------------------------------------------------------------
# Installation detection
# ---------------------------------------------------------------------------

def _find_isaac_launcher(root: Path) -> Optional[str]:
    """Return the path to ``isaaclab.bat`` or ``isaaclab.sh`` if present."""
    for name in ("isaaclab.bat", "isaaclab.sh"):
        p = root / name
        if p.exists():
            return str(p)
    return None


def _validate_isaac_root(root: Path) -> Optional[Dict[str, str]]:
    """Validate *root* as an Isaac Lab installation directory.

    Returns ``{"path": ..., "python": ..., "launcher": ...}`` on
    success, or ``None``.

    *launcher* is the path to ``isaaclab.bat``/``isaaclab.sh`` which
    can be used with ``-p`` flag as a portable Python entry-point
    (handles conda, kit-python, and pip-installed isaacsim transparently).
    """
    launcher = _find_isaac_launcher(root)
    # Look for isaaclab.sh / isaaclab.bat
    if launcher:
        python = _find_isaac_python(root)
        return {"path": str(root), "python": python or "", "launcher": launcher}
    # Maybe the path points directly to a python interpreter
    if (root / "python").exists() or (root / "python.exe").exists():
        return {"path": str(root.parent), "python": str(root), "launcher": ""}
    # Minimal fallback: directory contains an ``isaaclab`` package
    if (root / "isaaclab").is_dir() or (root / "source").is_dir():
        python = _find_isaac_python(root)
        return {"path": str(root), "python": python or "", "launcher": _find_isaac_launcher(root) or ""}
    return None


def _read_isaac_local_root() -> Optional[str]:
    """Read the registered Isaac Lab root from the unified engine registry.

    Returns the root path string, or ``None`` if not registered.
    """
    from src.system.engines.registry import get_engine_registry
    return get_engine_registry().get_isaac_local_root()


def detect_isaac_lab() -> Optional[Dict[str, str]]:
    """Detect Isaac Lab installation.

    Checks in order:
      1. ``ISAAC_LAB_PATH`` environment variable
      2. Unified engine registry (``engine_registry.json``)

    Returns a dict with ``path`` and ``python`` keys, or None.
    """
    # 1. Environment variable (explicit override, highest priority)
    env_path = os.environ.get("ISAAC_LAB_PATH", "")
    if env_path:
        result = _validate_isaac_root(Path(env_path))
        if result is not None:
            return result

    # 2. Registered path from engine registry
    json_path = _read_isaac_local_root()
    if json_path:
        p = Path(json_path)
        if p.is_dir():
            result = _validate_isaac_root(p)
            if result is not None:
                return result

    return None


def _find_isaac_python(isaac_lab_root: Path) -> Optional[str]:
    """Find Isaac Lab's Python interpreter.

    Search order:
      1. Isaac Sim kit python (``_isaac_sim/python.bat`` etc.)
      2. Local venv inside Isaac Lab root
      3. Active conda env (``CONDA_PREFIX``)
      4. System Python on PATH (via ``shutil.which``)
    """
    candidates = [
        isaac_lab_root / "_isaac_sim" / "python.sh",
        isaac_lab_root / "_isaac_sim" / "python.bat",
        isaac_lab_root / ".venv" / "bin" / "python",
        isaac_lab_root / ".venv" / "Scripts" / "python.exe",
    ]
    # Conda environment (Isaac Lab's recommended setup)
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if conda_prefix:
        candidates.append(Path(conda_prefix) / "python.exe")
        candidates.append(Path(conda_prefix) / "bin" / "python")
    for c in candidates:
        if c.exists():
            return str(c)
    return None


# ---------------------------------------------------------------------------
# Progress parser — tails log output for reward curves
# ---------------------------------------------------------------------------

# RSL-RL outputs "Learning iteration 100/1500" as a centered header line.
# Reward is on a separate line: "Mean reward:  12.34"
_ITER_LINE_RE = re.compile(
    r"[Ll]earning\s+[Ii]teration\s+(\d+)\s*/\s*(\d+)"
)

# Launcher prints this line right after ``runner.learn()`` returns. The
# backend uses it as the authoritative "training loop completed" signal so
# the UI can flip to 100% even when the printed iteration counter only
# reaches max-1 (RSL-RL is 0-indexed) and so we don't have to wait for the
# child process to actually exit before declaring training done.
_TRAINING_DONE_SENTINEL = "[UnitPort] TRAINING_LOOP_DONE"

# Completion marker: "Saving model to ..." or model_XXX.pt
_SAVED_RE = re.compile(r"(?:[Ss]aving\s+model\s+to|model_\d+\.pt)\s*[:]?\s*(.+?)(?:\s|$)")


def parse_log_line(line: str) -> Optional[Dict[str, Any]]:
    """Parse a single log line into a message dict (or None)."""
    m = _SAVED_RE.search(line)
    if m:
        return {
            "type": MSG_LOG,
            "data": f"Model saved: {m.group(1).strip()}",
        }
    return None


# ---------------------------------------------------------------------------
# Multi-line RSL-RL metric block accumulator
# ---------------------------------------------------------------------------
# RSL-RL v5+ outputs per-iteration blocks like:
#
#   ################################################################################
#                        Learning iteration 100/1500
#
#                                 Total steps: 2457600
#                            Steps per second: 50000
#                             Collection time: 0.123s
#                               Learning time: 0.456s
#                           Mean policy loss: 0.0123
#                            Mean value loss: 0.0456
#                                 Mean reward: 12.34
#                        Mean episode length: 500.00
#                             Mean action std: 0.50
#   --------------------------------------------------------------------------------
#                            Iteration time: 0.58s
#                              Time elapsed: 00:01:23
#                                       ETA: 00:12:34
#
# Block start: line of "####"
# Iteration header: "Learning iteration N/M"
# Metrics: right-padded "Key:  value" lines
# Block end: line of "----"

# Matches "Learning iteration 100/1500"
_ITER_LINE_RE_ACC = re.compile(r"[Ll]earning\s+[Ii]teration\s+(\d+)\s*/\s*(\d+)")

# Matches right-padded "Key: value" lines like "Mean reward: 12.34"
_METRIC_LINE_RE = re.compile(r"^\s*([\w\s]+\w)\s*:\s*([-\d.eE+]+)")

# Block boundaries
_BLOCK_START_RE = re.compile(r"^#{10,}")
_BLOCK_END_RE = re.compile(r"^-{10,}")

# ---------------------------------------------------------------------------
# AMP single-line metric format
# ---------------------------------------------------------------------------
# Our vendored AMPOnPolicyRunner._log() emits one line per iteration in the
# format:
#   [AMP] it=  24/1500  rew=12.345  len=480.0  vf=7.677  surr=-0.005
#         amp=0.034  gp=0.077  pol=-0.89  exp=+0.87  t=2.16s
#
# Unlike rsl_rl's multi-line block (which MetricsAccumulator parses), this
# is a single self-contained line — easy to regex, but it carries different
# canonical fields (no entropy/kl, but extra amp_loss / grad_pen / policy_pred
# / expert_pred). The training panel knows how to display both.
#
# IMPORTANT: ``rew`` and ``len`` may be ``nan`` for the first few iterations
# while the reward deque is still empty. The parser converts those to ``None``
# in the emitted dict so the chart can carry-forward instead of zeroing out.
_AMP_LINE_RE = re.compile(
    r"\[AMP\]\s*"
    r"it=\s*(?P<it>\d+)\s*/\s*(?P<total>\d+)\s+"
    r"rew=\s*(?P<rew>nan|[-+]?\d*\.?\d+)\s+"
    r"style=\s*(?P<style>nan|[-+]?\d*\.?\d+)\s+"
    r"task=\s*(?P<task>nan|[-+]?\d*\.?\d+)\s+"
    r"len=\s*(?P<len>nan|[-+]?\d*\.?\d+)\s+"
    r"vf=\s*(?P<vf>[-+]?\d*\.?\d+)\s+"
    r"surr=\s*(?P<surr>[-+]?\d*\.?\d+)\s+"
    r"amp=\s*(?P<amp>[-+]?\d*\.?\d+)\s+"
    r"gp=\s*(?P<gp>[-+]?\d*\.?\d+)\s+"
    r"pol=\s*(?P<pol>[-+]?\d*\.?\d+)\s+"
    r"exp=\s*(?P<exp>[-+]?\d*\.?\d+)\s+"
    r"acc_p=\s*(?P<acc_p>[-+]?\d*\.?\d+)\s+"
    r"acc_e=\s*(?P<acc_e>[-+]?\d*\.?\d+)\s+"
    r"t=\s*(?P<dt>[-+]?\d*\.?\d+)s"
)

# Stage transition line from AMPOnPolicyRunner._learn_staged():
#   [UnitPort][STAGE] Stage 1/3: AMP On  (2000 iters)
_STAGE_LINE_RE = re.compile(
    r"\[UnitPort\]\[STAGE\]\s*Stage\s+(?P<idx>\d+)/(?P<total>\d+):\s*"
    r"(?P<name>.+?)\s+\((?P<iters>\d+)\s*iters\)"
)

# Stage complete line:
#   [UnitPort][STAGE] Stage 1 (AMP On) complete. best_reward=12.345  iters=2000
_STAGE_COMPLETE_RE = re.compile(
    r"\[UnitPort\]\[STAGE\]\s*Stage\s+(?P<idx>\d+)\s+\((?P<name>[^)]+)\)\s+complete\.\s+"
    r"best_reward=(?P<best>[-+]?\d*\.?\d+)"
)


def _to_float_or_none(s: str) -> Optional[float]:
    if s is None:
        return None
    s = s.strip().lower()
    if s == "nan":
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def parse_amp_line(line: str) -> Optional[Dict[str, Any]]:
    """Parse a vendored-AMP per-iteration log line into a metrics dict.

    Returns ``{"type": MSG_METRICS, "data": {...}}`` matching the same
    schema TrainingPanel.on_metrics consumes for vanilla PPO, plus the
    AMP-only extras (``amp_loss``, ``grad_pen``, ``policy_pred``,
    ``expert_pred``). Returns ``None`` if the line doesn't match.
    """
    m = _AMP_LINE_RE.search(line)
    if m is None:
        return None
    rew = _to_float_or_none(m.group("rew"))
    style_r = _to_float_or_none(m.group("style"))
    task_r = _to_float_or_none(m.group("task"))
    ep_len = _to_float_or_none(m.group("len"))
    data: Dict[str, Any] = {
        "iteration": int(m.group("it")),
        "total": int(m.group("total")),
        "reward_mean": rew if rew is not None else 0.0,
        "reward_min": rew if rew is not None else 0.0,
        "reward_max": rew if rew is not None else 0.0,
        "ep_len_mean": ep_len if ep_len is not None else 0.0,
        "policy_loss": float(m.group("surr")),
        "value_loss": float(m.group("vf")),
        # B4 (AMP fix plan): separate style / task reward breakdown so
        # the training panel can plot them independently and so users
        # can diagnose whether style is dominating or task signal is
        # actually flowing through the (1-l)*style + l*task mix.
        "style_reward": style_r if style_r is not None else 0.0,
        "task_reward": task_r if task_r is not None else 0.0,
        "style_reward_valid": style_r is not None,
        "task_reward_valid": task_r is not None,
        # AMP-specific extras (TrainingPanel handles these in on_metrics)
        "amp_loss": float(m.group("amp")),
        "grad_pen": float(m.group("gp")),
        "policy_pred": float(m.group("pol")),
        "expert_pred": float(m.group("exp")),
        "accuracy_policy": float(m.group("acc_p")),
        "accuracy_expert": float(m.group("acc_e")),
        "iteration_time": float(m.group("dt")),
        "algorithm": "AMP_PPO",
        # Sentinel — the panel uses this to know reward fields are stale
        # and shouldn't be plotted as a literal zero.
        "reward_valid": rew is not None,
    }
    return {"type": MSG_METRICS, "data": data}


# Map RSL-RL printed keys → our canonical keys
_KEY_MAP = {
    "mean reward": "reward_mean",
    "mean episode length": "ep_len_mean",
    "mean policy loss": "policy_loss",
    "mean value loss": "value_loss",
    "mean action std": "action_std",
    "mean extrinsic reward": "reward_extrinsic",
    "mean intrinsic reward": "reward_intrinsic",
    "total steps": "total_steps",
    "steps per second": "fps",
    "collection time": "collect_time",
    "learning time": "learn_time",
    "iteration time": "iteration_time",
    "learning rate": "lr",
}


class MetricsAccumulator:
    """Stateful accumulator that collects RSL-RL metric blocks across lines."""

    def __init__(self) -> None:
        self._in_block = False
        self._iteration: Optional[int] = None
        self._total: Optional[int] = None
        self._fields: Dict[str, float] = {}

    def feed(self, line: str) -> Optional[Dict[str, Any]]:
        stripped = line.strip()

        # Block start: "####..."
        if _BLOCK_START_RE.match(stripped):
            # Flush previous block if any
            result = self._flush()
            self._in_block = True
            self._fields = {}
            return result

        if not self._in_block:
            return None

        # Iteration header
        m = _ITER_LINE_RE_ACC.search(line)
        if m:
            self._iteration = int(m.group(1))
            self._total = int(m.group(2))
            return None

        # Block end: "----..."  → emit metrics
        if _BLOCK_END_RE.match(stripped):
            return self._flush()

        # Metric line: "Key: value"
        m = _METRIC_LINE_RE.match(line)
        if m:
            raw_key = m.group(1).strip().lower()
            try:
                value = float(m.group(2))
            except ValueError:
                return None
            canonical = _KEY_MAP.get(raw_key, raw_key.replace(" ", "_"))
            self._fields[canonical] = value

        return None

    def flush(self) -> Optional[Dict[str, Any]]:
        return self._flush()

    def _flush(self) -> Optional[Dict[str, Any]]:
        if self._iteration is None or not self._fields:
            self._in_block = False
            self._iteration = None
            self._fields = {}
            return None
        data = {
            "iteration": self._iteration,
            "total": self._total or 0,
            "reward_mean": self._fields.get("reward_mean", 0.0),
            "reward_min": self._fields.get("reward_min", 0.0),
            "reward_max": self._fields.get("reward_max", 0.0),
            "ep_len_mean": self._fields.get("ep_len_mean", 0.0),
            "policy_loss": self._fields.get("policy_loss", 0.0),
            "value_loss": self._fields.get("value_loss", 0.0),
            "entropy": self._fields.get("entropy", 0.0),
            "kl": self._fields.get("kl", 0.0),
            "lr": self._fields.get("lr", 0.0),
        }
        self._in_block = False
        self._iteration = None
        self._fields = {}
        return {"type": MSG_METRICS, "data": data}


# ---------------------------------------------------------------------------
# IsaacLabBackend
# ---------------------------------------------------------------------------

class IsaacLabBackend:
    """Orchestrates Isaac Lab training as an external subprocess.

    Usage::

        config = IsaacLabConfig(task_name="Isaac-Velocity-Flat-Go2-v0", ...)
        backend = IsaacLabBackend(config)
        backend.on_message = my_callback   # receives same dicts as training_process
        backend.start()
        ...
        backend.cancel()
    """

    def __init__(self, config: IsaacLabConfig) -> None:
        self.config = config
        self.on_message: Optional[Callable[[Dict[str, Any]], None]] = None
        self._process: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._cancelled = threading.Event()
        self._export_path: Optional[Path] = None
        # Most recent (step, total) parsed from the rsl_rl iteration header
        # so the TRAINING_LOOP_DONE sentinel can emit a 100% MSG_PROGRESS
        # without depending on the iteration line ever printing total/total.
        self._last_iter_total: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Launch training in a background thread."""
        if self._thread and self._thread.is_alive():
            log.warning("IsaacLabBackend already running")
            return
        self._cancelled.clear()
        self._thread = threading.Thread(
            target=self._run, name="isaac_lab_train", daemon=True,
        )
        self._thread.start()

    def cancel(self) -> None:
        """Request cancellation of the running training.

        Isaac Lab's launcher (``isaaclab.bat``/``isaaclab.sh``) spawns the
        actual Python interpreter, which in turn spawns Isaac Sim. Calling
        ``Popen.terminate()`` only kills the immediate child (the launcher
        script), leaving the grandchildren — including the multi-GB Isaac
        Sim process — alive in the background. We must kill the whole
        process tree.
        """
        self._cancelled.set()
        proc = self._process
        if not proc or proc.poll() is not None:
            return
        pid = proc.pid
        log.info("Terminating Isaac Lab process tree (pid=%d)", pid)
        try:
            if os.name == "nt":
                # taskkill /T walks the parent-child chain via the Win32
                # ToolHelp snapshot, /F sends WM_CLOSE then TerminateProcess.
                # This is the only reliable built-in way to nuke the whole
                # Isaac Sim subtree on Windows without pywin32 / Job Objects.
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
                # Escalate to SIGKILL if the group ignores SIGTERM.
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(pgid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
        except Exception:
            log.exception("Failed to kill Isaac Lab process tree")
            # Last-ditch fallback so we never leave the call site without
            # at least having asked the immediate child to die.
            try:
                proc.kill()
            except Exception:
                pass

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def export_path(self) -> Optional[Path]:
        return self._export_path

    # ------------------------------------------------------------------
    # Dry run (for testing without Isaac Lab installed)
    # ------------------------------------------------------------------

    def launch_review(self, checkpoint_dir: str) -> None:
        """Launch Isaac Sim viewport to replay the trained policy.

        Opens a visible Isaac Sim window (not headless) in a detached process.
        Does not block or capture output — the user closes the window manually.
        """
        cmd = self.config.build_review_command(checkpoint_dir)
        log.info("Launching review: %s", " ".join(cmd))
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        env.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
        subprocess.Popen(
            cmd,
            cwd=self.config.isaac_lab_path or None,
            env=env,
        )

    def dry_run(self) -> List[str]:
        """Return the command that would be executed (for debugging)."""
        return self.config.build_command()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _emit(self, msg: Dict[str, Any]) -> None:
        if self.on_message:
            try:
                self.on_message(msg)
            except Exception:
                log.exception("Error in on_message callback")

    def _tail_subprocess(self) -> Dict[str, Any]:
        """Drain the running subprocess's stdout, parse metrics + checkpoints
        and capture startup-error breadcrumbs. Blocks until the child exits.

        Returns a small dict so the outer ``_run`` can decide whether to
        retry (e.g. fall back to headless) or proceed to the export step.
        """
        best_reward = float("-inf")
        last_checkpoint_path: Optional[str] = None
        metrics_acc = MetricsAccumulator()
        saw_metrics_block = False
        startup_error_lines: list[str] = []
        _STARTUP_ERROR_HINTS = (
            "dependency solver failure",
            "application failed to start",
            "failed to load extension",
            "module not found",
            "cuda error",
            "failed to launch",
            "ImportError",
            "Traceback (most recent call last)",
        )

        for line in self._process.stdout:
            if self._cancelled.is_set():
                break
            line = line.rstrip()
            bw, cp = self._process_line(line, metrics_acc, best_reward)
            if bw != best_reward:
                saw_metrics_block = True
            best_reward = bw
            if cp:
                last_checkpoint_path = cp
            low = line.lower()
            if any(h.lower() in low for h in _STARTUP_ERROR_HINTS):
                startup_error_lines.append(line.strip())

        # Flush any pending metrics block
        metrics_msg = metrics_acc.flush()
        if metrics_msg:
            self._emit(metrics_msg)
            saw_metrics_block = True

        self._process.wait()
        retcode = self._process.returncode
        return {
            "retcode": retcode,
            "best_reward": best_reward,
            "last_checkpoint_path": last_checkpoint_path,
            "training_started": bool(last_checkpoint_path) or saw_metrics_block,
            "startup_error_lines": startup_error_lines,
        }

    def _run(self) -> None:
        """Thread target: launch subprocess, tail output, handle completion."""
        try:
            log_dir = Path(self.config.log_dir) if self.config.log_dir else None
            if log_dir:
                log_dir.mkdir(parents=True, exist_ok=True)

            env = dict(os.environ)
            env["PYTHONUNBUFFERED"] = "1"
            # Auto-accept Omniverse Kit EULA so Isaac Sim doesn't block
            # on an interactive license prompt inside the subprocess.
            env.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

            # We may launch the subprocess twice: once with the user's
            # configured ``headless`` preference, and (only if the first
            # attempt dies before the training loop ever started with the
            # specific Isaac Sim "dependency solver failure" symptom) a
            # second time forced to ``--headless``. Most users would rather
            # get a working training run than a viewport, so we recover
            # automatically and surface a warning instead of an error.
            attempt = 0
            max_attempts = 2
            outcome: Optional[Dict[str, Any]] = None
            while attempt < max_attempts:
                attempt += 1
                cmd = self.config.build_command()
                cmd_str = " ".join(cmd)
                log.info("Launching Isaac Lab (attempt %d): %s", attempt, cmd_str)
                self._emit({"type": MSG_LOG, "data": f"[UnitPort] Command: {cmd_str}"})
                self._emit({"type": MSG_LOG, "data": f"[UnitPort] CWD: {self.config.isaac_lab_path or '(none)'}"})
                self._emit({"type": MSG_LOG, "data": "[UnitPort] Starting subprocess..."})

                kwargs: dict = {
                    "stdout": subprocess.PIPE,
                    "stderr": subprocess.STDOUT,
                    "text": True,
                    "bufsize": 1,
                    "cwd": self.config.isaac_lab_path or None,
                    "env": env,
                    # Isaac Sim emits UTF-8 (incl. © and other non-ASCII glyphs).
                    # Without an explicit encoding, Python on Windows falls back
                    # to the locale codec (GBK on zh-CN systems) and the stdout
                    # iterator raises UnicodeDecodeError mid-stream. Force UTF-8
                    # and replace any stray bytes so tailing never crashes.
                    "encoding": "utf-8",
                    "errors": "replace",
                }
                # Make sure the child process itself also writes UTF-8 instead
                # of inheriting the parent's locale codepage, otherwise some
                # Kit/omni log lines get re-encoded to GBK before reaching us.
                env.setdefault("PYTHONIOENCODING", "utf-8")
                if os.name == "nt":
                    # CREATE_NEW_PROCESS_GROUP isolates the launcher and all
                    # of its descendants so taskkill /T can walk the tree
                    # without accidentally touching the UnitPort GUI process.
                    kwargs["creationflags"] = (
                        subprocess.CREATE_NO_WINDOW
                        | subprocess.CREATE_NEW_PROCESS_GROUP
                    )
                else:
                    # Put the child into its own session so we can later
                    # signal the entire process group via os.killpg.
                    kwargs["start_new_session"] = True

                self._process = subprocess.Popen(cmd, **kwargs)
                self._emit({"type": MSG_LOG, "data": f"[UnitPort] PID: {self._process.pid}"})

                outcome = self._tail_subprocess()

                if self._cancelled.is_set():
                    self._emit({"type": MSG_CANCELLED})
                    return

                # Auto-retry path: only viable on the first attempt, only
                # when the user wanted a viewport, only when Kit died at
                # bootstrap (no training started) with a 55 / dependency
                # solver signature.
                if (
                    attempt < max_attempts
                    and outcome["retcode"] in {55}
                    and not outcome["training_started"]
                    and not self.config.headless
                    and any(
                        "dependency solver failure" in (line or "").lower()
                        or "application failed to start" in (line or "").lower()
                        for line in outcome["startup_error_lines"]
                    )
                ):
                    self._emit({
                        "type": MSG_LOG,
                        "data": (
                            "[UnitPort] Isaac Sim viewport launch failed at "
                            "Kit bootstrap — retrying with --headless. "
                            "(Most likely cause: stale Kit extension cache or "
                            "an unsupported display configuration.)"
                        ),
                    })
                    self.config.headless = True
                    continue

                # Either succeeded, hit a non-recoverable error, or already
                # retried — fall through to the post-run handling below.
                break

            assert outcome is not None
            best_reward = outcome["best_reward"]
            last_checkpoint_path = outcome["last_checkpoint_path"]
            training_started = outcome["training_started"]
            startup_error_lines = outcome["startup_error_lines"]
            retcode = outcome["retcode"]

            self._emit({"type": MSG_LOG, "data": f"[UnitPort] Process exited with code {retcode}"})

            # Isaac Sim / Kit returns 55 in two very different situations:
            #   (a) Carbonite "user-requested shutdown" — the user closed
            #       the Sim viewport mid-training. We promote the last
            #       checkpoint to a finished run and fall through to export.
            #   (b) Kit's dependency solver / extension loader failed at
            #       startup, so the sim never came up and no training ever
            #       happened. In that case 55 is a *real* error and we
            #       surface a helpful message.
            _USER_CLOSE_RETCODES = {55}
            if retcode in _USER_CLOSE_RETCODES:
                if training_started:
                    self._emit({
                        "type": MSG_LOG,
                        "data": "[UnitPort] Isaac Sim window closed by user — "
                                "promoting last checkpoint to a finished run.",
                    })
                    # fall through to the export branch below.
                else:
                    detail = (
                        startup_error_lines[-1]
                        if startup_error_lines
                        else "Isaac Sim never reached the training loop "
                             "(Kit bootstrap failed)."
                    )
                    self._emit({
                        "type": MSG_ERROR,
                        "data": (
                            "Isaac Lab failed to start (exit code 55). "
                            f"{detail}\n\n"
                            "If this only happens with the viewport window "
                            "enabled, try running headless or check that the "
                            "GPU drivers / Isaac Sim asset cache are healthy."
                        ),
                    })
                    return
            elif retcode != 0:
                self._emit({
                    "type": MSG_ERROR,
                    "data": f"Isaac Lab exited with code {retcode}",
                })
                return

            # --- Export step ---
            export_dir = self._run_export(last_checkpoint_path)
            if export_dir:
                self._export_path = export_dir
                self._emit({
                    "type": MSG_FINISHED,
                    "data": {
                        "bundle_path": str(export_dir),
                        "artifact_path": str(export_dir),
                        "step": 0,
                        "reward_mean": best_reward,
                        "best_reward": best_reward,
                        "ep_len_mean": 0.0,
                    },
                })
            else:
                self._emit({
                    "type": MSG_FINISHED,
                    "data": {
                        "bundle_path": self.config.log_dir,
                        "artifact_path": self.config.log_dir,
                        "step": 0,
                        "reward_mean": best_reward,
                        "best_reward": best_reward,
                        "ep_len_mean": 0.0,
                    },
                })

        except FileNotFoundError as exc:
            self._emit({
                "type": MSG_ERROR,
                "data": f"Isaac Lab not found: {exc}. "
                        "Check ISAAC_LAB_PATH or isaac_lab_python config.",
            })
        except Exception as exc:
            log.exception("IsaacLabBackend error")
            self._emit({"type": MSG_ERROR, "data": str(exc)})

    _ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

    def _process_line(self, line: str, metrics_acc, best_reward: float):
        """Process a single log line: emit, parse progress/metrics, track checkpoint."""
        self._emit({"type": MSG_LOG, "data": line})
        checkpoint_path = None

        # Strip ANSI color codes before regex matching
        clean = self._ANSI_RE.sub("", line)

        # Detect iteration header → emit progress for the progress bar.
        # RSL-RL is 0-indexed: ``for it in range(max_iterations)`` prints
        # ``Learning iteration {it}/{max}``, so the LAST line of a finished
        # run is e.g. "1499/1500". The natural ``int(step/total*100)`` then
        # caps at 99 forever. Bump the *displayed* step to total whenever
        # we see the final iteration line so the bar reaches 100%; the
        # raw ``step`` is still recorded for downstream consumers below.
        m = _ITER_LINE_RE.search(clean)
        if m:
            step = int(m.group(1))
            total = int(m.group(2))
            self._last_iter_total = total
            display_step = total if step >= total - 1 else step
            self._emit({
                "type": MSG_PROGRESS,
                "data": (display_step, total, best_reward, best_reward, 0.0, "training"),
            })

        # Authoritative "training loop completed" sentinel from
        # il_train_launcher.py. Emit a forced 100% progress so the UI
        # reaches the goal even if no iteration line ever rendered with
        # display_step == total (e.g. an off-by-one in a future rsl_rl
        # version, or a run that printed in a slightly different format).
        if _TRAINING_DONE_SENTINEL in clean:
            total = self._last_iter_total or 1
            self._emit({
                "type": MSG_PROGRESS,
                "data": (total, total, best_reward, best_reward, 0.0, "training"),
            })

        parsed = parse_log_line(clean)
        if parsed:
            self._emit(parsed)

        # AMP-specific single-line metrics. The vendored AMPOnPolicyRunner
        # doesn't print rsl_rl's multi-line block (different log format
        # entirely), so MetricsAccumulator.feed() never matches and the
        # chart stays empty. Parse the single-line ``[AMP] it=…`` form
        # directly and emit MSG_METRICS + a synthetic MSG_PROGRESS.
        amp_parsed = parse_amp_line(clean)
        if amp_parsed:
            self._emit(amp_parsed)
            amp_data = amp_parsed.get("data", {})
            it = int(amp_data.get("iteration", 0))
            total = int(amp_data.get("total", 0))
            if total > 0:
                self._last_iter_total = total
                rew = float(amp_data.get("reward_mean", 0.0))
                if amp_data.get("reward_valid") and rew > best_reward:
                    best_reward = rew
                display_step = total if it >= total - 1 else it + 1
                self._emit({
                    "type": MSG_PROGRESS,
                    "data": (display_step, total, rew, best_reward, 0.0, "training"),
                })

        # ── Stage transition detection ──
        # Parse "[UnitPort][STAGE] Stage X/Y: Name (N iters)" lines and
        # emit a stage-specific progress message so the UI can show the
        # current stage in the progress strip.
        stage_m = _STAGE_LINE_RE.search(clean)
        if stage_m:
            self._current_stage_idx = int(stage_m.group("idx"))
            self._current_stage_total = int(stage_m.group("total"))
            self._current_stage_name = stage_m.group("name").strip()
            self._emit({
                "type": MSG_LOG,
                "data": (
                    f"[STAGE] ▶ Stage {self._current_stage_idx}/"
                    f"{self._current_stage_total}: "
                    f"{self._current_stage_name}"
                ),
            })

        # Inject stage info into AMP metrics if we're in staged mode
        if amp_parsed and hasattr(self, "_current_stage_idx"):
            amp_data = amp_parsed.get("data", {})
            amp_data["stage_idx"] = getattr(self, "_current_stage_idx", -1)
            amp_data["stage_total"] = getattr(self, "_current_stage_total", 0)
            amp_data["stage_name"] = getattr(self, "_current_stage_name", "")

        metrics_msg = metrics_acc.feed(clean)
        if metrics_msg:
            self._emit(metrics_msg)
            # Update best_reward from metrics
            reward = metrics_msg.get("data", {}).get("reward_mean", best_reward)
            if reward > best_reward:
                best_reward = reward

        m = _SAVED_RE.search(clean)
        if m:
            checkpoint_path = m.group(1).strip()

        return best_reward, checkpoint_path

    def _tail_log_file(self, metrics_acc, best_reward: float):
        """Tail the log file written by the separate console window (Windows)."""
        import time as _time

        log_path = getattr(self, "_log_file", None)
        last_checkpoint_path = None
        file_pos = 0

        while self._process.poll() is None:
            if self._cancelled.is_set():
                break
            _time.sleep(0.3)
            if log_path and log_path.exists():
                try:
                    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                        f.seek(file_pos)
                        new_data = f.read()
                        file_pos = f.tell()
                    for line in new_data.splitlines():
                        line = line.rstrip()
                        if not line:
                            continue
                        bw, cp = self._process_line(line, metrics_acc, best_reward)
                        best_reward = bw
                        if cp:
                            last_checkpoint_path = cp
                except Exception:
                    pass

        # Final read after process exits
        if log_path and log_path.exists():
            try:
                with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                    f.seek(file_pos)
                    for line in f.read().splitlines():
                        line = line.rstrip()
                        if not line:
                            continue
                        bw, cp = self._process_line(line, metrics_acc, best_reward)
                        best_reward = bw
                        if cp:
                            last_checkpoint_path = cp
            except Exception:
                pass

        return best_reward, last_checkpoint_path

    def _run_export(self, checkpoint_path: Optional[str]) -> Optional[Path]:
        """Run play.py export to get policy.onnx + env.yaml."""
        if not checkpoint_path:
            log.warning("No checkpoint path found; skipping export")
            return None

        try:
            cmd = self.config.build_export_command(checkpoint_path)
            self._emit({"type": MSG_LOG, "data": f"Export: {' '.join(cmd)}"})

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                cwd=self.config.isaac_lab_path or None,
            )

            if result.returncode != 0:
                self._emit({
                    "type": MSG_LOG,
                    "data": f"Export warning (rc={result.returncode}): {result.stderr[:500]}",
                })

            # Find exported/ directory near the checkpoint
            cp_path = Path(checkpoint_path)
            for candidate in [
                cp_path.parent / "exported",
                cp_path.parent.parent / "exported",
            ]:
                if candidate.is_dir():
                    return candidate

            log.warning("Could not find exported/ directory near %s", checkpoint_path)
            return cp_path.parent

        except Exception as exc:
            self._emit({
                "type": MSG_LOG,
                "data": f"Export failed: {exc}",
            })
            return None
