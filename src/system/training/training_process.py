#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Isolated training subprocess for UnitPort Training Ground.

Runs the full SB3 training + export pipeline in a completely separate Python
interpreter so the UI process is never impacted by GIL contention or heavy
compute.

Message protocol (dicts sent through multiprocessing.Queue):
    {"type": "log",       "data": str}
    {"type": "progress",  "data": (step, total, reward_mean, best_reward, ep_len_mean, status)}
    {"type": "eval",      "data": (mean_reward, std_reward, success_rate, passed)}
    {"type": "finished",  "data": {"bundle_path": str, "artifact_path": str,
                                   "step": int, "reward_mean": float,
                                   "best_reward": float, "ep_len_mean": float}}
    {"type": "error",     "data": str}
    {"type": "cancelled"}
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    import multiprocessing as mp


# ---------------------------------------------------------------------------
# Message type constants
# ---------------------------------------------------------------------------

MSG_LOG       = "log"
MSG_PROGRESS  = "progress"
MSG_METRICS   = "metrics"
MSG_EVAL      = "eval"
MSG_FINISHED  = "finished"
MSG_ERROR     = "error"
MSG_CANCELLED = "cancelled"


# ---------------------------------------------------------------------------
# Helper: set process priority (best-effort, non-fatal)
# ---------------------------------------------------------------------------

def _elevate_process_priority() -> None:
    """Raise the current process to ABOVE_NORMAL priority (Windows) or nice -5 (POSIX)."""
    try:
        if os.name == "nt":
            import ctypes
            ABOVE_NORMAL = 0x00008000
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            ctypes.windll.kernel32.SetPriorityClass(handle, ABOVE_NORMAL)
        else:
            os.nice(-5)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Main worker — runs inside the subprocess
# ---------------------------------------------------------------------------

def run_training_in_process(
    spec,
    queue: "mp.Queue",
    cancel_event: "mp.Event",
    extra_sys_path: Optional[List[str]] = None,
) -> None:
    """
    Entry point for the training subprocess.

    Parameters
    ----------
    spec:
        Fully compiled ``TrainingJobSpec``.
    queue:
        Multiprocessing Queue used to send messages back to the parent thread.
    cancel_event:
        Multiprocessing Event; when set the training loop should stop cleanly.
    extra_sys_path:
        Additional sys.path entries from the parent (needed so the child
        process can import project modules after spawn).
    """
    # ── Restore import paths ──────────────────────────────────────────────
    if extra_sys_path:
        for p in reversed(extra_sys_path):
            if p and p not in sys.path:
                sys.path.insert(0, p)

    # ── Process priority ─────────────────────────────────────────────────
    _elevate_process_priority()

    # ── CUDA / PyTorch optimisations ─────────────────────────────────────
    try:
        import torch
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True          # auto-tune conv kernels
            torch.backends.cudnn.allow_tf32 = True
        # Keep PyTorch CPU intra-op threads low — the training subprocess runs
        # mostly CUDA ops; the CPU cores are better used by MuJoCo workers.
        # 2 threads covers CPU-side preprocessing without competing with envs.
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    # ── Throttled helpers ────────────────────────────────────────────────
    _last_progress_ts = [0.0]
    _PROGRESS_INTERVAL = 0.25   # max 4 UI updates / second

    def _put(msg: dict) -> None:
        try:
            queue.put_nowait(msg)
        except Exception:
            pass

    def _log(text: str) -> None:
        _put({"type": MSG_LOG, "data": text})

    def _progress(step, total, reward_mean, best_reward, ep_len_mean, status):
        now = time.monotonic()
        if now - _last_progress_ts[0] < _PROGRESS_INTERVAL:
            return
        _last_progress_ts[0] = now
        _put({"type": MSG_PROGRESS,
              "data": (step, total,
                       float(reward_mean), float(best_reward),
                       float(ep_len_mean), str(status))})

    def _metrics(metrics_dict):
        _put({"type": MSG_METRICS, "data": metrics_dict})

    def _cancel() -> bool:
        return cancel_event.is_set()

    # ── Run training ─────────────────────────────────────────────────────
    try:
        from src.system.training.sb3_trainer import SB3Trainer

        # Vis-check is not supported in subprocess mode (requires a display
        # attached to this process, and the window would be invisible on most
        # setups).  Log a one-time notice if vis checks were configured.
        if getattr(spec, "vis_check_config", None) is not None:
            _log(
                "[training_process] Note: vis-check milestones are skipped in "
                "subprocess training mode.  Use the Review button after training."
            )

        # ── Hardware auto-tune ───────────────────────────────────────────
        # Detect CPU/GPU and scale n_envs / batch_size for max throughput.
        # Only overrides params that are still at their UI defaults.
        try:
            from src.system.training.hardware_tuner import auto_tune_for_hardware
            auto_tune_for_hardware(spec, log_fn=_log)
        except Exception as _ht_exc:
            _log(f"[auto-tune] Skipped (error: {_ht_exc})")

        trainer = SB3Trainer(
            spec,
            progress_fn=_progress,
            log_fn=_log,
            cancel_fn=_cancel,
            vis_check_fn=None,          # see note above
            metrics_fn=_metrics,
        )
        model = trainer.train()

        # ── Cancellation guard ──────────────────────────────────────────
        if _cancel():
            _put({"type": MSG_CANCELLED})
            return

        # ── Optional post-training evaluation ──────────────────────────
        if getattr(spec, "eval_config", None) is not None:
            try:
                eval_result = trainer.evaluate(model)
                _put({
                    "type": MSG_EVAL,
                    "data": (
                        float(eval_result.get("mean_reward", 0.0)),
                        float(eval_result.get("std_reward", 0.0)),
                        float(eval_result.get("success_rate", 0.0)),
                        bool(eval_result.get("passed", False)),
                    ),
                })
            except Exception as exc:
                _log(f"[eval] Evaluation failed (non-fatal): {exc}")

        # ── Export bundle ───────────────────────────────────────────────
        _log("[export] Exporting bundle…")
        from src.system.training.bundle_exporter import export_bundle
        from src.system.core.utils.path_helper import get_project_root

        # Reconstruct the active project on the worker side from the
        # context the parent injected into the spec. This lets the
        # bundle exporter resolve ``projects/<pid>/checkpoints/`` without
        # falling back to a global path.
        _proj_store = None
        _proj_id = str(getattr(spec, "_project_id", "") or "")
        _proj_root = getattr(spec, "_project_root", None)
        if _proj_id and _proj_root:
            try:
                from pathlib import Path as _P
                from src.system.core.project_store import ProjectStore as _PS
                from src.system.core import project_session as _ps_mod
                _proj_store = _PS(_P(_proj_root))
                _ps_mod.set_active(_proj_store, _proj_id)
            except Exception as _exc:
                _log(f"[export] Failed to reattach project session: {_exc}")
                _proj_store = None

        export_result = export_bundle(
            model,
            spec,
            run_id=getattr(spec, "_run_id", ""),
            output_root=get_project_root(),
            log_fn=_log,
            project_id=_proj_id or None,
            project_store=_proj_store,
        )

        algo = getattr(getattr(spec, "algorithm_config", None), "algorithm", "")
        if export_result.runtime_bundle_path:
            _log(f"[{algo}] Runtime bundle: {export_result.runtime_bundle_path}")
        if export_result.artifact_path:
            _log(f"[{algo}] Training artifact: {export_result.artifact_path}")

        _put({
            "type": MSG_FINISHED,
            "data": {
                "bundle_path":    str(export_result.primary_path),
                "artifact_path":  str(export_result.artifact_path or ""),
            },
        })

    except Exception as exc:
        _put({"type": MSG_ERROR, "data": f"{exc}\n{traceback.format_exc()}"})
