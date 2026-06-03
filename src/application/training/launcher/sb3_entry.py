# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""SB3 child-process entry script.

Spawned by :class:`SB3SubprocessBackend.run_blocking`. Reads a JSON
payload from stdin, runs :func:`train_sb3` (Stage 7) end-to-end, and
emits structured stdout lines that the parent parses via the
``[UnitPort][PROGRESS|METRICS|FINISHED|ERROR]`` line protocol.

Stdin payload schema:
    {
      "spec":            <TrainingSpec.to_dict()>,
      "run_id":          str,
      "total_timesteps": int | null,
      "export_bundle":   bool       # default true
    }

Stdout protocol:
    Plain lines               → MSG_LOG (parent forwards verbatim)
    [UnitPort][PROGRESS] step=X total=Y reward=R    → MSG_PROGRESS
    [UnitPort][METRICS] {"key": v, ...}              → MSG_METRICS
    [UnitPort][FINISHED] {...}                       → MSG_FINISHED
    [UnitPort][ERROR] msg                            → MSG_ERROR

Cancellation: parent sends taskkill / SIGTERM to the process tree —
SB3's ``model.learn`` is a tight loop that surrenders to OS signals
before the next gradient step. We do not need cooperative polling
here for Stage 10 v1; the entry script just exits cleanly on SIGTERM.

Spawn safety: ``if __name__ == "__main__"`` guard is mandatory — under
Windows ``spawn``, importing this module re-runs top-level code, and
without the guard the SB3 + mujoco import path triggers recursive
spawn.
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Stdout helpers — emit the parent's line protocol
# ---------------------------------------------------------------------------


def _emit_line(line: str) -> None:
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def emit_progress(step: int, total: int, reward: float, unit: str = "step") -> None:
    rew_s = "nan" if reward != reward else f"{reward:.6f}"
    # ``unit`` is "iter" for PPO (on-policy; the budget is max_iterations) or
    # "step" for SAC/TD3 (off-policy; no rollout iteration). The parent labels
    # the progress line with it so the displayed count matches the budget unit.
    _emit_line(
        f"[UnitPort][PROGRESS] step={step} total={total} unit={unit} reward={rew_s}"
    )


def emit_metrics(payload: Dict[str, Any]) -> None:
    _emit_line(f"[UnitPort][METRICS] {json.dumps(payload, ensure_ascii=False)}")


def emit_finished(payload: Dict[str, Any]) -> None:
    _emit_line(f"[UnitPort][FINISHED] {json.dumps(payload, ensure_ascii=False)}")


def emit_error(msg: str) -> None:
    # One line per logical error; stack traces are emitted separately as plain
    # log lines so the parent's MSG_LOG path captures them.
    _emit_line(f"[UnitPort][ERROR] {msg}")


# ---------------------------------------------------------------------------
# SB3 progress callback — bridges model.learn → emit_progress
# ---------------------------------------------------------------------------


def _make_progress_callback(total_timesteps: int):
    """Return a SB3 ``BaseCallback`` that emits MSG_PROGRESS each rollout."""
    from stable_baselines3.common.callbacks import BaseCallback

    class _Inner(BaseCallback):
        def __init__(inner_self) -> None:
            super().__init__(verbose=0)
            inner_self._last_emit = 0

        def _on_step(inner_self) -> bool:
            return True

        def _on_rollout_end(inner_self) -> None:
            # PPO is iteration-based (the canvas budget is max_iterations):
            # report iter = num_timesteps / (n_steps × n_envs), total =
            # max_iterations. SAC/TD3 (no .n_steps) stay step-based.
            num_ts = int(inner_self.num_timesteps)
            n_steps = getattr(inner_self.model, "n_steps", None)
            n_envs = int(getattr(inner_self.model, "n_envs", 1) or 1)
            if n_steps:
                rollout = max(1, int(n_steps) * n_envs)
                count = num_ts // rollout
                count_total = max(1, total_timesteps // rollout)
                unit = "iter"
            else:
                count = num_ts
                count_total = total_timesteps
                unit = "step"
            # Best-effort reward + mean episode length read from SB3's
            # ep_info_buffer. ``r`` and ``l`` are the per-episode return /
            # length that ``Monitor`` always logs. ep_len_mean is the canonical
            # progress signal for locomotion (Isaac Lab emits the same key),
            # so the chart series matches across backends.
            reward = float("nan")
            ep_len_mean: Optional[float] = None
            try:
                buf = list(inner_self.model.ep_info_buffer or [])
                if buf:
                    rewards = [float(d.get("r", 0.0)) for d in buf]
                    reward = sum(rewards) / max(1, len(rewards))
                    lengths = [
                        float(d["l"]) for d in buf
                        if isinstance(d, dict) and "l" in d
                    ]
                    if lengths:
                        ep_len_mean = sum(lengths) / len(lengths)
            except Exception:
                pass
            emit_progress(count, count_total, reward, unit)
            if ep_len_mean is not None:
                # The metrics sample MUST share the progress X unit: the chart
                # keys its X-axis on "step", and emitting a different scale here
                # (raw num_timesteps while progress reports iterations) puts two
                # wildly different X scales on the same axis → the curve appears
                # to jump back and forth (worsened by large n_envs, which grows
                # num_timesteps n_envs× faster per rollout). So "step" = the same
                # ``count`` (iter for PPO / env-steps for off-policy) emit_progress
                # used. Raw env-step count is kept under "total_steps" (a
                # non-series metadata key) for any consumer that wants it.
                emit_metrics({
                    "step": count,
                    "total_steps": num_ts,
                    "ep_len_mean": ep_len_mean,
                })

    return _Inner()


# ---------------------------------------------------------------------------
# Spec rebuilding — JSON dict → TrainingSpec
# ---------------------------------------------------------------------------


def _spec_from_payload(payload: Dict[str, Any]):
    """Reconstruct :class:`TrainingSpec` from the JSON payload's ``spec`` field.

    The payload may carry the spec dict in the shape produced by
    ``TrainingSpec.to_dict()``. We use ``TrainingSpec.from_dict`` (Stage 3)
    for a safe round-trip. Returns ``None`` on missing / malformed payload.
    """
    from application.training.training_spec import TrainingSpec

    raw = payload.get("spec")
    if not isinstance(raw, dict):
        return None
    return TrainingSpec.from_dict(raw)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> int:
    # 1. Read JSON spec from stdin
    raw = sys.stdin.read()
    if not raw.strip():
        emit_error("empty stdin payload")
        return 2
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        emit_error(f"invalid JSON on stdin: {exc}")
        return 2

    run_id = str(payload.get("run_id") or "sb3_run")
    total_timesteps = payload.get("total_timesteps")
    if total_timesteps is not None:
        try:
            total_timesteps = int(total_timesteps)
        except (TypeError, ValueError):
            total_timesteps = None
    do_export = bool(payload.get("export_bundle", True))

    _emit_line(f"[UnitPort] sb3_entry start run_id={run_id} total_timesteps={total_timesteps}")

    # 2. Shared subprocess bootstrap (RegistryHub + ProjectStore + AppSignals
    #    backend + project-scoped run_dir). The IL launcher uses the same
    #    helper — keeping a single implementation prevents partition drift
    #    between SB3 (`<project>/training/runs/sb3_mujoco/<run_id>/`) and
    #    Isaac Lab (`<project>/training/runs/isaac_lab/<run_id>/`).
    try:
        from application.training.launcher._entry_bootstrap import (
            bootstrap_subprocess,
        )
        boot = bootstrap_subprocess(
            payload, default_backend="sb3_mujoco", log_line=_emit_line,
        )
    except Exception as exc:  # noqa: BLE001
        emit_error(f"_entry_bootstrap failed: {exc}")
        return 3
    bound_project = boot.bound_project
    backend_id = boot.backend_id
    run_dir = boot.run_dir

    spec = _spec_from_payload(payload)
    if spec is None or not getattr(spec, "robot", None) or not spec.robot.sku:
        emit_error("spec.robot is missing or unresolved")
        return 4

    # 3. Run training
    try:
        from application.training.sb3_trainer import train_sb3

        steps = total_timesteps if total_timesteps is not None else int(
            spec.algorithm.total_timesteps
        )
        callback = _make_progress_callback(steps)
        result = train_sb3(
            spec, total_timesteps=steps, callback=callback, run_dir=run_dir,
        )
    except Exception as exc:  # noqa: BLE001
        for frame in traceback.format_exception(type(exc), exc, exc.__traceback__):
            for fl in str(frame).splitlines():
                _emit_line(fl)
        emit_error(f"train_sb3 raised: {exc}")
        return 5

    # 4. Bundle export (Stage 7's end-to-end wrapper)
    bundle_payload: Optional[Dict[str, Any]] = None
    if do_export:
        try:
            from application.training.bundle_exporter import BundleExporter

            out = BundleExporter.export_bundle(
                result.model, spec, run_id=run_id, backend_id=backend_id,
                vec_env=result.vec_env,
            )
            bundle_payload = {
                "bundle_path": str(out.bundle_path),
                "manifest_path": str(out.manifest_path),
                "policy_path": str(out.policy_path),
                "source_path": str(out.source_path),
            }
        except Exception as exc:  # noqa: BLE001
            emit_error(f"BundleExporter.export_bundle raised: {exc}")
            # don't return 6 yet — still surface what we have

    # 5. FINISHED message
    finished = {
        "run_id": run_id,
        "algorithm": result.algorithm,
        "obs_dim": result.obs_dim,
        "action_dim": result.action_dim,
        "total_timesteps": result.total_timesteps,
        "bundle": bundle_payload,
    }
    emit_finished(finished)

    try:
        result.vec_env.close()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
