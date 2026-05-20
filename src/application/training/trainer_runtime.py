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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from unitport_sdk import get_tasks_manager, log_info, log_warning


@dataclass
class DeployCoverageReport:
    """Structured result of cross-format IR-role coverage analysis.

    Built by :func:`compute_deploy_coverage` from the registry's
    ``joints_per_format`` tables for a SKU. The UI submit hook reads
    this BEFORE calling :func:`submit_canvas_training` and surfaces a
    blocking modal to the user when ``has_gap`` is True, so the user
    can decide whether to spend compute on a training run whose
    resulting bundle won't deploy to one (or both) targets.

    A non-UI caller (CLI batch script, automated test) can ignore the
    report and proceed straight to ``submit_canvas_training``; the
    existing :func:`_pre_flight_warn_cross_format_coverage` still
    log_warning's the same content, so nothing is silently swallowed.
    """
    sku: str = ""
    robot_name: str = ""
    has_gap: bool = False
    # Roles declared by USD's joints_per_format but absent from MJCF's
    # (or vice versa). Excludes bucket roles (misc / sensor* / base).
    missing_in_mjcf: List[str] = field(default_factory=list)
    missing_in_usd: List[str] = field(default_factory=list)
    # Human-readable deploy-target consequence: e.g. ["MuJoCo unavailable"].
    affected_targets: List[str] = field(default_factory=list)
    # Whether each per-format table is present at all (vs declared but null).
    has_mjcf_table: bool = False
    has_usd_table: bool = False
    # Single-format setups are valid and ``has_gap`` is False there;
    # this lets the UI distinguish "no gap because alignment is perfect"
    # from "no gap because only one format is even populated".
    single_format_only: bool = False


def compute_deploy_coverage(sku: str) -> DeployCoverageReport:
    """Compare MJCF / USD IR-role sets in the registry for ``sku``.

    Single source of truth for the cross-format coverage check — used by
    both the UI submit-confirmation modal (:meth:`MainWindow._confirm_
    deploy_coverage`) and the legacy log-only :func:`_pre_flight_warn_
    cross_format_coverage` wrapper retained for non-UI callers. Bucket
    roles (``misc`` / ``sensor*`` / ``base``) are excluded from the
    comparison: they legitimately repeat or differ across formats
    (MJCF cosmetic shells, floating-base joint that USD collapses into
    the articulation root, etc.).
    """
    report = DeployCoverageReport(sku=sku)
    if not sku:
        return report

    from registers import robots as _r

    entry = _r.get_robot(sku) or {}
    report.robot_name = entry.get("name") or sku
    joints_pf = entry.get("joints_per_format") or {}

    def _ir_role_set(fmt: str) -> Optional[frozenset]:
        block = joints_pf.get(fmt)
        if not isinstance(block, dict) or not block:
            return None
        return frozenset(
            str((spec or {}).get("ir_role") or "").strip()
            for spec in block.values()
            if isinstance(spec, dict)
            and str((spec or {}).get("ir_role") or "").strip()
            and not str((spec or {}).get("ir_role") or "").strip().startswith("sensor")
            and str((spec or {}).get("ir_role") or "").strip() not in ("misc", "base")
        )

    mjcf_set = _ir_role_set("MJCF")
    usd_set = _ir_role_set("USD")
    report.has_mjcf_table = mjcf_set is not None
    report.has_usd_table = usd_set is not None

    if mjcf_set is None or usd_set is None:
        # Single-format setups are legitimate — no cross-format check
        # possible / needed. Bundle will deploy to whichever format is
        # populated; the other target is "unavailable" by definition.
        report.single_format_only = True
        return report
    if mjcf_set == usd_set:
        return report

    report.has_gap = True
    report.missing_in_mjcf = sorted(usd_set - mjcf_set)
    report.missing_in_usd = sorted(mjcf_set - usd_set)
    if report.missing_in_mjcf:
        report.affected_targets.append(
            "MuJoCo deploy (MJCF doesn't declare the missing roles)"
        )
    if report.missing_in_usd:
        report.affected_targets.append(
            "IsaacSim / cloud deploy (USD doesn't declare the missing roles)"
        )
    return report


def _pre_flight_dump_assets(sku: str, *, log_prefix: str = "[pre-train]") -> None:
    """Pre-flight safety net: dump declared-but-empty per-format tables before training.

    For the given robot ``sku``, walks ``assets.{MJCF,USD,USD_URL}``; for
    every format with a non-empty asset declaration whose
    ``joints_per_format[fmt]`` is null/empty, runs
    :meth:`RobotAssetService.dump_and_persist` synchronously. Skips
    formats with no declared asset entirely — the bundle exporter won't
    ship that format, so there's nothing to populate.

    Why this exists despite the boot self-check (CLAUDE.md §1.8 / Pass A):
    boot Pass A runs once per session; between boots a user can add a new
    robot, repoint an asset path, or import a custom variant — any of
    which leaves a declared-but-empty table that the boot pass already
    missed. This pre-flight is the per-training-run safety net. Common
    case (tables already populated) is a single registry lookup + dict
    test — no I/O, no subprocess.

    Loud failure: refuses to submit training when a required dump fails,
    rather than queueing a run that will later raise from
    ``spec_compiler`` / ``bundle_finalizer`` with a less specific error.
    """
    if not sku:
        return

    from registers import robots as _r
    from application.service.robot_assets import get_robot_asset_service

    entry = _r.get_robot(sku) or {}
    assets = entry.get("assets") or {}
    joints_pf = entry.get("joints_per_format") or {}

    # Per-format declaration tests:
    #   * MJCF — asset.path only (no Nucleus equivalent).
    #   * USD  — asset.path OR asset.USD_URL (Nucleus URL is a first-class
    #            asset declaration; the dump subprocess can read both).
    needs_dump: list[str] = []
    if assets.get("MJCF"):
        tbl = joints_pf.get("MJCF") or {}
        if not (isinstance(tbl, dict) and tbl):
            needs_dump.append("MJCF")
    if assets.get("USD") or assets.get("USD_URL"):
        tbl = joints_pf.get("USD") or {}
        if not (isinstance(tbl, dict) and tbl):
            needs_dump.append("USD")

    if not needs_dump:
        return

    svc = get_robot_asset_service()
    name = entry.get("name") or sku
    log_info(
        f"{log_prefix} sku={sku!r} ({name}) has declared-but-empty "
        f"format(s): {needs_dump}; auto-dumping before training"
    )
    for fmt in needs_dump:
        result = svc.dump_and_persist(sku, fmt)
        if not result.ok:
            raise RuntimeError(
                f"{log_prefix} pre-flight {fmt} dump failed for sku={sku!r} "
                f"({name}): kind={result.error_kind} msg={result.error}. "
                f"Training refused — fix the asset path / re-import the "
                f"asset and retry. CLAUDE.md §1.8: shipping a bundle "
                f"compiled against a non-existent per-format table would "
                f"surface later as 'IR role X has no physical name' at "
                f"export time."
            )
        log_info(
            f"{log_prefix} dumped {fmt} ({len(result.joints)} joints, "
            f"{len(result.bodies)} bodies, "
            f"{result.n_unresolved} unresolved IR-role assignment(s))"
        )


def _pre_flight_warn_cross_format_coverage(
    sku: str, *, log_prefix: str = "[pre-train]"
) -> None:
    """Pre-flight ADVISORY (non-blocking): log_warning wrapper around
    :func:`compute_deploy_coverage` for callers (CLI / batch) that don't
    surface a UI dialog. UI callers should read :class:`DeployCoverageReport`
    directly via ``compute_deploy_coverage`` and present a modal.
    """
    report = compute_deploy_coverage(sku)
    if not report.has_gap:
        return
    log_warning(
        f"{log_prefix} sku={report.sku!r} ({report.robot_name}): MJCF "
        f"and USD IR-role sets don't fully overlap. Training will proceed "
        f"(uses the active format only), but the exported bundle will "
        f"have reduced deploy target coverage. Affected: "
        f"{report.affected_targets}. "
        f"Roles in USD missing from MJCF: {report.missing_in_mjcf}. "
        f"Roles in MJCF missing from USD: {report.missing_in_usd}. "
        f"If you intended both deploy targets to work, repoint the "
        f"smaller-DOF asset to a matching variant (e.g. Unitree G1: "
        f"``menagerie/unitree_g1/scene_with_hands.xml`` instead of "
        f"``scene.xml``) and re-Dump."
    )


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

    # Pre-flight: declared assets with empty per-format tables → auto-dump.
    # MUST run BEFORE compile_training_spec because the compiler reads
    # ``joints_per_format[active_format]`` (see CLAUDE.md §1.8 fix that
    # made empty tables a loud raise instead of a silent MJCF fallback).
    # The dump synchronously updates the in-memory registry via
    # ``RobotAssetService.set_discovered_bodies`` so the immediately-
    # following compile sees the freshly-populated table.
    #
    # After dumps complete, advise on cross-format IR-role coverage gaps
    # (non-blocking — training only needs the active format complete,
    # cross-format mismatch only affects which deploy targets the
    # resulting bundle supports). The warning lets the user know up
    # front rather than discovering it at bundle finalize / MuJoCo load.
    if ir.robot_id:
        _pre_flight_dump_assets(ir.robot_id, log_prefix="[play]")
        _pre_flight_warn_cross_format_coverage(
            ir.robot_id, log_prefix="[play]"
        )

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


__all__ = [
    "submit_il_trainer",
    "submit_sb3_trainer",
    "submit_canvas_training",
    "compute_deploy_coverage",
    "DeployCoverageReport",
]
