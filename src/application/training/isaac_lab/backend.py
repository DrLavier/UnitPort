# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""IsaacLabBackend — external subprocess orchestrator for Isaac Lab training.

Ported from DEMO ``src/system/training/isaac_lab_backend.py`` with these
RELEASE-specific changes:

* ``IsaacLabConfig`` import switches to ``application.training.isaac_lab.config``.
* Logging dispatches through ``train_log_*`` so messages get the ``[run_id]``
  prefix ``CmdLogWidget`` filters on.
* No post-training Isaac Sim ``play.py`` subprocess. The training launcher
  already writes ``params/env.yaml`` + ``params/agent.yaml`` + ``model_*.pt``
  to ``log_dir`` during the run, and ``bundle_finalizer`` converts the
  latest ``.pt`` to ONNX in-process via ``export_rsl_rl_actor_to_onnx``.
  The legacy ``_run_export`` spawn-a-second-Isaac-Sim path was redundant
  (30s+ Kit cold boot) and a documented source of RTX scenedb crashes;
  removed per CLAUDE.md §1.8 anti-fallback rule (the path was only ever
  "successful" via its own argparse-crash fallback handing run_dir to
  finalize — a textbook silent-fallback bug). Bundle export now happens
  via ``application.training.isaac_lab.bundle_finalizer.finalize_isaac_lab_bundle``
  on the run_dir directly.

The subprocess + signal interface (MSG_*, on_message callback shape) and the
log-parsing regex tables (``_ITER_LINE_RE`` / ``_AMP_LINE_RE`` /
``MetricsAccumulator`` / ``_KEY_MAP``) are byte-for-byte identical so the
training UI can consume RSL-RL output the same way for both backends.
"""

from __future__ import annotations

import os
import re
import subprocess
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from application.training._sdk.logging_bridge import (
    train_log_error,
    train_log_info,
    train_log_warning,
)

from .config import IsaacLabConfig


# Message types (same as DEMO training_process.py)
MSG_LOG = "log"
MSG_PROGRESS = "progress"
MSG_METRICS = "metrics"
MSG_FINISHED = "finished"
MSG_ERROR = "error"
MSG_CANCELLED = "cancelled"


# ---------------------------------------------------------------------------
# Log parser regexes (verbatim from DEMO)
# ---------------------------------------------------------------------------

_ITER_LINE_RE = re.compile(r"[Ll]earning\s+[Ii]teration\s+(\d+)\s*/\s*(\d+)")

_TRAINING_DONE_SENTINEL = "[UnitPort] TRAINING_LOOP_DONE"

_SAVED_RE = re.compile(
    r"(?:[Ss]aving\s+model\s+to|model_\d+\.pt)\s*[:]?\s*(.+?)(?:\s|$)"
)

_ITER_LINE_RE_ACC = re.compile(
    r"[Ll]earning\s+[Ii]teration\s+(\d+)\s*/\s*(\d+)"
)

_METRIC_LINE_RE = re.compile(r"^\s*([\w\s]+\w)\s*:\s*([-\d.eE+]+)")

_BLOCK_START_RE = re.compile(r"^#{10,}")
_BLOCK_END_RE = re.compile(r"^-{10,}")

_AMP_LINE_RE = re.compile(
    r"\[(?:AMP|PPO)\]\s*"
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

_STAGE_LINE_RE = re.compile(
    r"\[UnitPort\]\[STAGE\]\s*Stage\s+(?P<idx>\d+)/(?P<total>\d+):\s*"
    r"(?P<name>.+?)\s+\((?P<iters>\d+)\s*iters\)"
)

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


def parse_log_line(line: str) -> Optional[Dict[str, Any]]:
    """Parse a single log line into a message dict (or None)."""
    m = _SAVED_RE.search(line)
    if m:
        return {"type": MSG_LOG, "data": f"Model saved: {m.group(1).strip()}"}
    return None


def parse_amp_line(line: str) -> Optional[Dict[str, Any]]:
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
        "style_reward": style_r if style_r is not None else 0.0,
        "task_reward": task_r if task_r is not None else 0.0,
        "style_reward_valid": style_r is not None,
        "task_reward_valid": task_r is not None,
        "amp_loss": float(m.group("amp")),
        "grad_pen": float(m.group("gp")),
        "policy_pred": float(m.group("pol")),
        "expert_pred": float(m.group("exp")),
        "accuracy_policy": float(m.group("acc_p")),
        "accuracy_expert": float(m.group("acc_e")),
        "iteration_time": float(m.group("dt")),
        "algorithm": "AMP_PPO",
        "reward_valid": rew is not None,
    }
    return {"type": MSG_METRICS, "data": data}


_KEY_MAP = {
    "mean reward": "reward_mean",
    "mean episode length": "ep_len_mean",
    "mean policy loss": "policy_loss",
    "mean surrogate loss": "policy_loss",       # rsl_rl variant
    "mean value loss": "value_loss",
    "mean entropy loss": "entropy",             # rsl_rl variant
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

        if _BLOCK_START_RE.match(stripped):
            result = self._flush()
            self._in_block = True
            self._fields = {}
            return result

        if not self._in_block:
            return None

        m = _ITER_LINE_RE_ACC.search(line)
        if m:
            self._iteration = int(m.group(1))
            self._total = int(m.group(2))
            return None

        if _BLOCK_END_RE.match(stripped):
            return self._flush()

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
        # Pass through ONLY the fields rsl_rl actually emitted this block.
        # The previous implementation injected 0.0 placeholders for absent
        # keys (policy_loss / value_loss / entropy / kl / lr); downstream
        # the chart treated them as legitimate series and rendered flat
        # zero lines. ``data.update(...)`` keeps the metrics dict honest:
        # only what was parsed shows up. Callers that read with .get(k, 0.0)
        # still get a safe default.
        data: Dict[str, Any] = {
            "iteration": self._iteration,
            "total": self._total or 0,
        }
        data.update(self._fields)
        self._in_block = False
        self._iteration = None
        self._fields = {}
        return {"type": MSG_METRICS, "data": data}


# ---------------------------------------------------------------------------
# IsaacLabBackend
# ---------------------------------------------------------------------------

class IsaacLabBackend:
    """Orchestrates Isaac Lab training as an external subprocess."""

    def __init__(self, config: IsaacLabConfig, run_id: str = "isaac") -> None:
        self.config = config
        self.run_id = run_id or config.run_id or "isaac"
        self.on_message: Optional[Callable[[Dict[str, Any]], None]] = None
        self._process: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._cancelled = threading.Event()
        self._last_iter_total: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Launch training in a background thread (fire-and-forget).

        The TrainingTask wrapper drives ``run_blocking()`` directly in
        its own QThread; ``start()`` is kept for direct subprocess use.
        """
        if self._thread and self._thread.is_alive():
            train_log_warning(self.run_id, "IsaacLabBackend already running")
            return
        self._cancelled.clear()
        self._thread = threading.Thread(
            target=self._run, name="isaac_lab_train", daemon=True,
        )
        self._thread.start()

    def run_blocking(self) -> None:
        """Synchronous variant: drives the subprocess on the calling thread.
        Used by IsaacLabTrainingTask.run() so cancellation can be polled
        through the SDK Task contract instead of a daemon thread.
        """
        self._cancelled.clear()
        self._run()

    def cancel(self) -> None:
        """Kill the entire process tree (Isaac Sim spawns grandchildren)."""
        self._cancelled.set()
        proc = self._process
        if not proc or proc.poll() is not None:
            return
        pid = proc.pid
        train_log_info(self.run_id, f"Terminating Isaac Lab process tree (pid={pid})")
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
        except Exception as exc:
            train_log_error(self.run_id, f"Failed to kill Isaac Lab process tree: {exc}")
            try:
                proc.kill()
            except Exception:
                pass

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

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
            except Exception as exc:
                train_log_error(self.run_id, f"Error in on_message callback: {exc}")

    _ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

    def _process_line(self, line: str, metrics_acc, best_reward: float):
        self._emit({"type": MSG_LOG, "data": line})
        checkpoint_path = None
        clean = self._ANSI_RE.sub("", line)

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

        if _TRAINING_DONE_SENTINEL in clean:
            total = self._last_iter_total or 1
            self._emit({
                "type": MSG_PROGRESS,
                "data": (total, total, best_reward, best_reward, 0.0, "training"),
            })

        parsed = parse_log_line(clean)
        if parsed:
            self._emit(parsed)

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

        stage_m = _STAGE_LINE_RE.search(clean)
        if stage_m:
            self._current_stage_idx = int(stage_m.group("idx"))
            self._current_stage_total = int(stage_m.group("total"))
            self._current_stage_name = stage_m.group("name").strip()
            self._emit({
                "type": MSG_LOG,
                "data": (
                    f"[STAGE] ▶ Stage {self._current_stage_idx}/"
                    f"{self._current_stage_total}: {self._current_stage_name}"
                ),
            })

        if amp_parsed and hasattr(self, "_current_stage_idx"):
            amp_data = amp_parsed.get("data", {})
            amp_data["stage_idx"] = getattr(self, "_current_stage_idx", -1)
            amp_data["stage_total"] = getattr(self, "_current_stage_total", 0)
            amp_data["stage_name"] = getattr(self, "_current_stage_name", "")

        metrics_msg = metrics_acc.feed(clean)
        if metrics_msg:
            self._emit(metrics_msg)
            reward = metrics_msg.get("data", {}).get("reward_mean", best_reward)
            if reward > best_reward:
                best_reward = reward

        m = _SAVED_RE.search(clean)
        if m:
            checkpoint_path = m.group(1).strip()

        return best_reward, checkpoint_path

    def _tail_subprocess(self) -> Dict[str, Any]:
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
        """Launch subprocess, tail output, handle completion."""
        try:
            log_dir = Path(self.config.log_dir) if self.config.log_dir else None
            if log_dir:
                log_dir.mkdir(parents=True, exist_ok=True)

            env = dict(os.environ)
            env["PYTHONUNBUFFERED"] = "1"
            env.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
            env.setdefault("PYTHONIOENCODING", "utf-8")

            # NOTE: BAR1 aperture preflight runs on the UI thread before this
            # task is submitted (MainWindow._on_start_training → Bar1RiskDialog),
            # so the user can interactively choose Continue / Abort. It is
            # deliberately NOT repeated here — a worker-thread block after the
            # user already chose "Continue" would contradict that choice. The
            # pure logic lives in bar1_preflight.assess_bar1_risk.

            attempt = 0
            max_attempts = 2
            outcome: Optional[Dict[str, Any]] = None
            while attempt < max_attempts:
                attempt += 1
                cmd = self.config.build_command()
                cmd_str = " ".join(cmd)
                train_log_info(self.run_id, f"Launching Isaac Lab (attempt {attempt}): {cmd_str}")
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
                    "encoding": "utf-8",
                    "errors": "replace",
                }
                if os.name == "nt":
                    kwargs["creationflags"] = (
                        subprocess.CREATE_NO_WINDOW
                        | subprocess.CREATE_NEW_PROCESS_GROUP
                    )
                else:
                    kwargs["start_new_session"] = True

                self._process = subprocess.Popen(cmd, **kwargs)
                self._emit({"type": MSG_LOG, "data": f"[UnitPort] PID: {self._process.pid}"})

                outcome = self._tail_subprocess()

                if self._cancelled.is_set():
                    self._emit({"type": MSG_CANCELLED})
                    return

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
                            "Kit bootstrap — retrying with --headless."
                        ),
                    })
                    self.config.headless = True
                    continue

                break

            assert outcome is not None
            best_reward = outcome["best_reward"]
            last_checkpoint_path = outcome["last_checkpoint_path"]
            training_started = outcome["training_started"]
            startup_error_lines = outcome["startup_error_lines"]
            retcode = outcome["retcode"]

            self._emit({"type": MSG_LOG, "data": f"[UnitPort] Process exited with code {retcode}"})

            _USER_CLOSE_RETCODES = {55}
            if retcode in _USER_CLOSE_RETCODES:
                if training_started:
                    self._emit({
                        "type": MSG_LOG,
                        "data": "[UnitPort] Isaac Sim window closed by user — "
                                "promoting last checkpoint to a finished run.",
                    })
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
                            f"{detail}"
                        ),
                    })
                    return
            elif retcode != 0:
                self._emit({
                    "type": MSG_ERROR,
                    "data": f"Isaac Lab exited with code {retcode}",
                })
                return

            # Stdout regex (``_SAVED_RE``) often misses the checkpoint
            # path because RSL-RL versions vary in what they print on
            # save (some emit nothing). Disk-scan ``log_dir`` for the
            # highest-iter ``model_*.pt`` as a deterministic alternative
            # — RSL-RL's canonical filename, layout-stable across
            # versions. This is for the post-training log breadcrumb
            # only; bundle_finalizer scans the same run_dir itself, so
            # finalize does not depend on this scan succeeding.
            # WHY KEPT (Rule 1.c — on-disk ground-truth read vs unreliable
            # stdout text parse): the disk holds the actual saved
            # checkpoints; stdout parse is an opportunistic fast path.
            if not last_checkpoint_path and self.config.log_dir:
                log_dir_path = Path(self.config.log_dir)

                def _iter_num(p: Path) -> int:
                    try:
                        return int(p.stem.split("_", 1)[1])
                    except (IndexError, ValueError):
                        return -1

                candidates = [
                    p for p in log_dir_path.rglob("model_*.pt")
                    if _iter_num(p) >= 0
                ]
                if candidates:
                    candidates.sort(key=_iter_num, reverse=True)
                    last_checkpoint_path = str(candidates[0])
                    self._emit({
                        "type": MSG_LOG,
                        "data": (
                            f"[UnitPort] last checkpoint resolved via "
                            f"disk-scan: {candidates[0].name} "
                            f"(iter {_iter_num(candidates[0])})"
                        ),
                    })

            # Training writes ``params/env.yaml`` + ``params/agent.yaml``
            # + ``model_*.pt`` to ``self.config.log_dir`` during the run
            # (see il_train_launcher.py:1486-1487 + RSL-RL save_interval).
            # ``bundle_finalizer`` consumes that layout directly and
            # converts the latest ``.pt`` to ONNX in-process via
            # ``export_rsl_rl_actor_to_onnx`` (pure torch, no Isaac Sim
            # subprocess). The old spawn-a-second-Isaac-Sim ``_run_export``
            # path was pure overhead (30s+ Kit cold boot) and the source
            # of RTX scenedb crashes on certain Kit versions — see
            # CLAUDE.md §1.9 anti-fallback / portable-artifact rules.
            exported_dir = self.config.log_dir

            self._emit({
                "type": MSG_FINISHED,
                "data": {
                    "bundle_path": str(exported_dir),
                    "artifact_path": str(exported_dir),
                    "exported_dir": str(exported_dir),
                    "last_checkpoint_path": last_checkpoint_path or "",
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
            train_log_error(self.run_id, f"IsaacLabBackend error: {exc}")
            self._emit({"type": MSG_ERROR, "data": str(exc)})
