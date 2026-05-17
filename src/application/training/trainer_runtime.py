"""application.training.trainer_runtime — Trainer-node submit helper.

Stage 4 lifted ``ILPPOTrainerNode.execute`` and ``AMPTrainerNode.execute``
out of the node files into one helper so both nodes route through the
:mod:`application.training.backend` selector. Both nodes are fundamentally
"submit a training run" sinks — only the default ``training_mode`` differs.

Stage 4 status:
    * Backend dispatch via ``select_backend(...)``: works for the Isaac
      Lab path (Stage 1 adapter, real PPO + AMP_PPO via rsl_rl).
    * SB3+MuJoCo path raises ``NotImplementedError`` (Stage 10 will land
      ``SB3MujocoBackendAdapter.build_task``); this is the deliberate
      gating signal Stage 1 reserved.
    * Spec compilation from canvas inputs is partial — we read from the
      ``inputs`` dict that the canvas engine supplies, not from a full
      ``WorkflowIR``. The full spec-compile path is reserved for the
      submit button (Stage 12 wires it on top of
      ``application.training.spec_compiler.compile_training_spec``).

Stage 12 (submit button): :func:`submit_canvas_training` is the play-button
entrypoint. It takes the IR-shape dict produced by
``CanvasPage.to_workflow_dict()``, runs it through
``canvas_to_ir → compile_training_spec → raise_if_errors → select_backend``,
and submits the resulting Task to the SDK ``TasksManager``. The canvas-bound
``WorkflowIR.backend`` is the sole source of truth for backend selection
(per the "canvas binds backend once" rule in CLAUDE.md §1).
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from unitport_sdk import get_tasks_manager, log_info, log_warning


def _make_run_id(node_id: str, *, schema_id: str = "") -> str:
    """Build the canonical ``{slug}__{utc}__{uuid6}`` run id used by every
    submit_* helper in this module."""
    slug = node_id or schema_id or "run"
    return "{}__{}__{}".format(
        slug,
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        uuid.uuid4().hex[:6],
    )


def submit_il_trainer(
    *,
    node_id: str,
    schema_id: str,
    params: Dict[str, Any],
    inputs: Dict[str, Any],
    default_mode: str = "PPO",
) -> Dict[str, Any]:
    """Disabled training entrypoint — kept for canvas IR shape only.

    Historically this was called from ``il_ppo_trainer.execute()`` /
    ``amp_trainer.execute()`` whenever the canvas IR was evaluated. It
    built a minimal hand-rolled spec **without** stashing the canvas
    dict, which downstream caused ``IsaacLabTrainingTask`` to fall back
    to Isaac Lab's stock task ``Isaac-Velocity-Flat-Unitree-Go2-v0`` —
    silently running a built-in env instead of the user's canvas. The
    user spent 2026-05-10 debugging that exact symptom.

    The legitimate IL training entry point is the top Play button only
    (``MainWindow._on_start_training`` → :func:`submit_canvas_training`),
    which carries the full canvas dict through ``spec.meta``. Mission
    Control's Start button forwards to the same Play button click.

    Calling this function now raises ``RuntimeError`` to surface the
    misuse. Pipe-shape inspection that doesn't require an actual run
    can use the node's static ``_OUTPUT_SHAPE`` metadata instead.
    """
    raise RuntimeError(
        f"[{schema_id}] submit_il_trainer is no longer a valid IL "
        f"training entry point — the canvas dict isn't reachable from "
        f"this code path, which used to silently fall back to Isaac "
        f"Lab's stock 'Isaac-Velocity-Flat-Unitree-Go2-v0' task. "
        f"Use the top Play button instead (it routes through "
        f"submit_canvas_training, which carries the full canvas)."
    )


def submit_sb3_trainer(
    *,
    node_id: str,
    schema_id: str,
    inputs: Dict[str, Any],
) -> Dict[str, Any]:
    """``train`` node entrypoint — reads ``algo_config`` input to pick the
    backend, then dispatches via :func:`select_backend`.

    SB3 path lives here (algorithm_config + env_assembler + train trio).

    Phase 3 boundary (rule §B6):  the ``train`` node is the **SB3 sink**
    — it must dispatch to ``BACKEND_SB3_MUJOCO`` only. ``BACKEND_AUTO``
    used to fall through ``select_backend`` and could resolve to Isaac
    Lab when both backends were registered, which would feed the SB3
    spec dict to ``IsaacLabBackendAdapter.build_task`` — guaranteed to
    misbehave because the two backends consume different spec shapes
    (SB3 reads ``algorithm`` + flat hyperparams; IL reads
    ``task.isaac_task_name`` + ``algorithm.il_ppo``). Lock the dispatch
    here. Users who want IL training drop an ``il_ppo_trainer`` node on
    the canvas instead.
    """
    from application.training.backend import (
        BACKEND_AUTO,
        BACKEND_SB3_MUJOCO,
        ensure_default_backends,
        select_backend,
    )

    if not isinstance(inputs, dict):
        inputs = {}

    algo = inputs.get("algo_config") or {}
    if not isinstance(algo, dict):
        algo = {}
    env_cfg = inputs.get("env_config") or {}
    if not isinstance(env_cfg, dict):
        env_cfg = {}
    eval_cfg = inputs.get("eval_config") or {}
    if not isinstance(eval_cfg, dict):
        eval_cfg = {}

    pref = str(algo.get("backend") or BACKEND_SB3_MUJOCO).strip()
    if pref == BACKEND_AUTO:
        pref = BACKEND_SB3_MUJOCO
    if pref != BACKEND_SB3_MUJOCO:
        # Reject explicit non-SB3 selection so the spec shape mismatch
        # surfaces here, not deep inside IsaacLabConfig.from_training_spec.
        raise ValueError(
            f"submit_sb3_trainer: backend={pref!r} is not allowed on the "
            f"'train' node — drop an il_ppo_trainer / amp_trainer node "
            f"instead. (Rule §B6.)"
        )
    ensure_default_backends()
    backend = select_backend(pref)

    spec: Dict[str, Any] = {
        "algorithm": str(algo.get("algorithm") or "ppo").lower(),
        "training_mode": "PPO",
        "total_timesteps": int(algo.get("total_timesteps", 1_000_000) or 1_000_000),
        "learning_rate": float(algo.get("learning_rate", 3e-4) or 3e-4),
        "batch_size": int(algo.get("batch_size", 256) or 256),
        "gamma": float(algo.get("gamma", 0.99) or 0.99),
        "seed": int(algo.get("seed", 42) or 42),
        "device": str(algo.get("device", "auto") or "auto"),
        "n_envs": int(env_cfg.get("n_envs", 8) or 8),
        "vec_type": str(env_cfg.get("vec_type", "subproc") or "subproc"),
        "obs_normalize": bool(env_cfg.get("obs_normalize", True)),
        "reward_normalize": bool(env_cfg.get("reward_normalize", False)),
        "eval_episodes": int(eval_cfg.get("eval_episodes", 0) or 0),
        "eval_interval": int(eval_cfg.get("eval_interval", 0) or 0),
    }

    run_id = _make_run_id(node_id, schema_id=schema_id)

    log_info(
        f"[{schema_id}] submit run_id={run_id} algo={spec['algorithm']} backend={backend.name}"
    )

    task = backend.build_task(spec, run_id=run_id)
    tid = get_tasks_manager().submit(task)

    train_pipe: Dict[str, Any] = {
        "task_id": tid,
        "run_id": run_id,
        "backend": backend.name,
        "algorithm": spec["algorithm"],
        "n_envs": spec["n_envs"],
        "total_timesteps": spec["total_timesteps"],
        "seed": spec["seed"],
    }
    if hasattr(task, "run_dir"):
        train_pipe["run_dir"] = str(task.run_dir)

    return {
        "train_pipe": train_pipe,
        "vis_check": {"task_id": tid, "status": "submitted"},
    }


def submit_canvas_training(
    canvas_dict: Dict[str, Any],
    *,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Play-button entrypoint: full canvas → IR → spec → backend → submit.

    Drives the canonical Stage 12 path documented in
    :mod:`application.training.spec_compiler`'s docstring::

        ir = canvas_to_ir(canvas_dict)
        spec, issues = compile_training_spec(ir)
        raise_if_errors(issues)
        backend = select_backend(ir.backend or "auto")
        task = backend.build_task(spec.to_dict(), run_id)

    Args:
        canvas_dict: IR-shape dict produced by ``CanvasPage.to_workflow_dict()``.
        run_id: optional explicit run id; auto-generated when omitted.

    Returns:
        ``{"task_id", "run_id", "backend", "algorithm", "issues"}``.
        ``issues`` is the full validator output (warnings + non-fatal info)
        — hard errors already raised by :func:`raise_if_errors`.

    Raises:
        ValueError:   spec validation reported hard errors.
        RuntimeError: no training backend available.
    """
    from application.compiler.lowering import canvas_to_ir
    from application.training.backend import (
        BACKEND_AUTO,
        ensure_default_backends,
        select_backend,
    )
    from application.training.spec_compiler import compile_training_spec
    from application.training.spec_validator import raise_if_errors

    ir = canvas_to_ir(canvas_dict if isinstance(canvas_dict, dict) else {})
    spec, issues = compile_training_spec(ir)
    raise_if_errors(issues)

    ensure_default_backends()
    pref = (getattr(ir, "backend", None) or BACKEND_AUTO).strip() or BACKEND_AUTO
    backend = select_backend(pref)

    rid = run_id or _make_run_id("canvas", schema_id="play")
    # Stash the source canvas dict on spec.meta so the IsaacLab backend can
    # rebuild the @configclass env_cfg in the run dir (consumed by
    # IsaacLabTrainingTask.__init__ → env_cfg_compiler.compile_env_cfg_to_file).
    # SB3 backend ignores this key. The canvas dict is JSON-clean (loaded
    # from canvas .json or produced by CanvasPage.to_workflow_dict), so it
    # survives ``TrainingSpec.to_dict()``'s asdict round-trip.
    try:
        if not isinstance(spec.meta, dict):
            spec.meta = {}
        if isinstance(canvas_dict, dict):
            spec.meta["__canvas_dict__"] = canvas_dict
    except Exception:
        pass
    task = backend.build_task(spec.to_dict(), run_id=rid)
    tid = get_tasks_manager().submit(task)

    log_info(
        f"[play] submit run_id={rid} backend={backend.name} "
        f"algo={spec.algorithm.algorithm} pref={pref}"
    )

    issues_out: list = []
    for issue in issues:
        to_dict = getattr(issue, "to_dict", None)
        if callable(to_dict):
            try:
                issues_out.append(to_dict())
                continue
            except Exception:
                pass
        issues_out.append(repr(issue))

    return {
        "task_id": tid,
        "run_id": rid,
        "backend": backend.name,
        "algorithm": spec.algorithm.algorithm,
        "issues": issues_out,
    }


__all__ = ["submit_il_trainer", "submit_sb3_trainer", "submit_canvas_training"]
