# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

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

from unitport_sdk import (
    Task,
    get_tasks_manager,
    log_error,
    log_info,
    log_warning,
)


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
    from registers.robots import is_actuated_ir_role

    entry = _r.get_robot(sku) or {}
    report.robot_name = entry.get("name") or sku
    joints_pf = entry.get("joints_per_format") or {}

    # Only actuated motion joints count toward cross-format coverage. Bucket
    # roles (sensor/misc/base) and the parked '__out_of_scope__' sentinel
    # legitimately repeat or differ across formats (MJCF freejoint the USD
    # articulation collapses into its root, cosmetic shells, IMU mounts) — they
    # must NOT register as a deploy-coverage gap. Shared predicate (same one
    # the RobotSpecRef actuated-joint filter and JointIRResolver use).
    def _ir_role_set(fmt: str) -> Optional[frozenset]:
        block = joints_pf.get(fmt)
        if not isinstance(block, dict) or not block:
            return None
        return frozenset(
            role
            for spec in block.values()
            if isinstance(spec, dict)
            for role in (str((spec or {}).get("ir_role") or "").strip(),)
            if is_actuated_ir_role(role)
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


def sim2sim_residual_preflight(canvas_dict: Dict[str, Any]) -> Optional[Any]:
    """Auto cross-engine plant-residual self-check, run BEFORE submit consumes
    compute. Isaac Lab backend only — the PhysX-train → MuJoCo-deploy case is the
    only cross-engine one; an SB3 canvas trains and deploys in MuJoCo.

    Measure-if-stale against the plant fingerprint (sha256 of MJCF + USD): a
    cached clean / parameterizable verdict returns instantly; a stale or missing
    one runs the measurement (MuJoCo in-process + PhysX in the Isaac Lab venv),
    auto-feeding the sim2sim_ranges registry that the bundle DR-coverage
    provenance reads. Returns the :class:`Sim2SimVerdict`, or None when the check
    does not apply (non-Isaac backend / no SKU) or could not run (logged). The UI
    reads ``verdict.needs_ack`` to soft-block on a non-convergent dimension.
    """
    from application.training.backend import BACKEND_ISAAC_LAB

    backend = str(canvas_dict.get("backend") or "").strip()
    if backend != BACKEND_ISAAC_LAB:
        return None
    sku = str(canvas_dict.get("robot_id") or "").strip()
    if not sku:
        return None
    try:
        from pathlib import Path

        from unitport_sdk import Paths
        from application.training.validation.sim2sim_measurement.auto_selfcheck import (
            selfcheck,
        )

        run_dir = Path(Paths.RUNTIME_DIR) / "sim2sim_selfcheck" / sku
        return selfcheck(sku, run_dir=run_dir)
    except Exception as exc:  # noqa: BLE001
        # Advisory auto-optimization — a measurement glitch must not wedge the
        # Play button (mirrors the deploy-coverage compute degradation policy).
        # The realized==designed alignment gates at export stay the hard gate.
        log_warning(f"[pre-train] sim2sim residual self-check skipped: {exc!r}")
        return None


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


def build_canvas_training_task(
    canvas_dict: Dict[str, Any],
    *,
    run_id: Optional[str] = None,
    task: Optional["Task"] = None,
) -> tuple:
    """Heavy canvas → IR → spec → backend → build pipeline, WITHOUT submit.

    Runs the canonical Stage 12 path documented in
    :mod:`application.training.spec_compiler`'s docstring::

        ir = canvas_to_ir(canvas_dict)
        spec, issues = compile_training_spec(ir)
        raise_if_errors(issues)
        backend = select_backend(ir.backend or "auto")
        built_task = backend.build_task(spec.to_dict(), run_id)

    This is the expensive part — asset dumps (subprocess), spec compilation
    (MJCF read + mass-matrix solve + sim2sim calibration) and, for Isaac Lab,
    env_cfg compilation inside ``build_task`` — and is therefore designed to
    run OFF the Qt GUI thread (see :class:`CanvasTrainingPrepareTask`). It
    deliberately does NOT call ``TasksManager.submit``: ``submit`` spawns a
    ``QThread`` and must run on the GUI thread, so the caller submits the
    returned (unsubmitted) task itself.

    Args:
        canvas_dict: IR-shape dict produced by ``CanvasPage.to_workflow_dict()``.
        run_id: optional explicit run id; auto-generated when omitted.
        task: optional owning :class:`~unitport_sdk.Task`; when present, its
            ``log_info`` / ``set_progress`` surface live preparation progress
            in the cmd log, and ``check_cancelled`` is honored between stages
            so a Stop press during prep unwinds cleanly.

    Returns:
        ``(built_task, meta)`` where ``meta`` is
        ``{"run_id", "backend", "algorithm", "issues", "target"}``.

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
    from application.service.engines.service import get_engine_service
    from application.training.spec_compiler import compile_training_spec
    from application.training.spec_validator import raise_if_errors

    def _progress(ratio: float, text: str) -> None:
        if task is not None:
            task.check_cancelled()
            task.set_progress(ratio, text)
            task.log_info(text)

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
        _progress(0.1, "Resolving robot assets…")
        _pre_flight_dump_assets(ir.robot_id, log_prefix="[play]")
        _pre_flight_warn_cross_format_coverage(
            ir.robot_id, log_prefix="[play]"
        )

    _progress(0.3, "Compiling training spec…")
    spec, issues = compile_training_spec(ir)
    raise_if_errors(issues)

    ensure_default_backends()
    pref = (getattr(ir, "backend", None) or BACKEND_AUTO).strip() or BACKEND_AUTO

    rid = run_id or _make_run_id("canvas", schema_id="play")
    # Stash the source canvas dict on spec.meta so the IsaacLab backend can
    # rebuild the @configclass env_cfg in the run dir (consumed by
    # IsaacLabTrainingTask.__init__ → env_cfg_compiler.compile_env_cfg_to_file).
    # SB3 backend ignores this key. The canvas dict is JSON-clean (loaded
    # from canvas .json or produced by CanvasPage.to_workflow_dict), so it
    # survives ``TrainingSpec.to_dict()``'s asdict round-trip. MUST happen
    # before ``spec.to_dict()`` below (both local and cloud paths read it).
    try:
        if not isinstance(spec.meta, dict):
            spec.meta = {}
        if isinstance(canvas_dict, dict):
            spec.meta["__canvas_dict__"] = canvas_dict
    except Exception:
        pass

    # Train target: "local" (default) vs "cloud". This is ORTHOGONAL to the
    # engine preference — cloud answers "where it runs", the engine
    # (isaac_lab / sb3_mujoco) answers "what runs it". We never fold cloud into
    # select_backend (that would poison its local is_available() chain); the
    # fork lives here, the orchestration layer. A misconfigured cloud target
    # raises in build_task and surfaces in the Play button's dialog — there is
    # NO silent fallback to a local run (CLAUDE.md §8).
    target = get_engine_service().get_link_target()
    if target == "cloud":
        from application.training.remote_backend import (
            ensure_remote_backends,
            select_remote_backend,
        )

        ensure_remote_backends()
        engine_name = _resolve_cloud_engine_name(pref)
        backend = select_remote_backend(engine_name)
    else:
        backend = select_backend(pref)

    _progress(0.6, "Building training task (compiling env config)…")
    built = backend.build_task(spec.to_dict(), run_id=rid)
    _progress(0.95, "Ready to launch")

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

    meta = {
        "run_id": rid,
        "backend": backend.name,
        "algorithm": spec.algorithm.algorithm,
        "issues": issues_out,
        "target": target,
        "pref": pref,
    }
    return built, meta


def submit_canvas_training(
    canvas_dict: Dict[str, Any],
    *,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Synchronous canvas → IR → spec → backend → build → submit.

    Thin wrapper over :func:`build_canvas_training_task` that also submits the
    built task to the SDK ``TasksManager``. Retained for non-UI callers (CLI
    batch scripts, tests) that can afford to block. **The GUI Play button must
    NOT call this** — building is heavy and would freeze the UI thread; it goes
    through :class:`CanvasTrainingPrepareTask` instead so the build happens on a
    worker slot.

    Returns ``{"task_id", "run_id", "backend", "algorithm", "issues",
    "target"}`` — ``issues`` is the full validator output (warnings + non-fatal
    info); hard errors already raised by ``raise_if_errors``.

    Raises:
        ValueError:   spec validation reported hard errors.
        RuntimeError: no training backend available.
    """
    built, meta = build_canvas_training_task(canvas_dict, run_id=run_id)
    tid = get_tasks_manager().submit(built)
    log_info(
        f"[play] submit run_id={meta['run_id']} target={meta['target']} "
        f"backend={meta['backend']} algo={meta['algorithm']} "
        f"pref={meta.get('pref', '')}"
    )
    return {
        "task_id": tid,
        "run_id": meta["run_id"],
        "backend": meta["backend"],
        "algorithm": meta["algorithm"],
        "issues": meta["issues"],
        "target": meta["target"],
    }


class CanvasTrainingPrepareTask(Task):
    """Off-GUI-thread heavy preparation for the Play button.

    Clicking ▶ used to run the *entire* submit pipeline synchronously on the
    Qt GUI thread: the cross-engine sim2sim self-check (drives BOTH physics
    engines — MuJoCo in-process + PhysX in the Isaac Lab venv), asset dumps
    (subprocess), ``compile_training_spec`` (MJCF read + mass-matrix solve +
    calibration) and ``build_task`` (Isaac Lab env_cfg compilation). All of
    that blocked the event loop for seconds-to-minutes — the run only reached a
    worker thread at the final ``TasksManager.submit``. The user saw the whole
    window freeze.

    This Task moves that work onto a ``TasksManager`` slot. It returns the
    built — but **unsubmitted** — training task plus the sim2sim verdict; the
    GUI ``task_finished`` handler then (on the GUI thread, where it belongs)
    shows the sim2sim acknowledgement dialog if needed, calls
    ``TasksManager.submit`` (which spawns the training ``QThread``), and wires
    up progress tracking.

    Failures propagate as ``self.error`` (routed to the canvas/cloud error
    dialogs by the GUI); a Stop press during prep cancels the task and unwinds
    via ``TaskCancelledException`` (no error, silent).
    """

    def __init__(
        self,
        canvas_dict: Dict[str, Any],
        *,
        run_id: Optional[str] = None,
    ) -> None:
        super().__init__(name="train:prepare")
        self._canvas_dict = canvas_dict
        self._run_id = run_id

    def run(self) -> Dict[str, Any]:
        self.check_cancelled()
        # Cross-engine plant-residual self-check (Isaac Lab only; measure-if-
        # stale, cached by plant fingerprint). The heavy measurement happens
        # here on the worker; the GUI surfaces the soft-block acknowledgement
        # modal from the verdict after this task finishes.
        self.set_progress(0.02, "Cross-engine sim2sim self-check…")
        verdict = sim2sim_residual_preflight(self._canvas_dict)
        self.check_cancelled()

        built, meta = build_canvas_training_task(
            self._canvas_dict, run_id=self._run_id, task=self,
        )
        self.check_cancelled()
        return {"training_task": built, "verdict": verdict, **meta}


def _resolve_cloud_engine_name(pref: str) -> str:
    """Fold a backend preference to a concrete engine name for cloud submission.

    Unlike the local :func:`select_backend`, this does NOT probe local
    ``is_available()`` — the job runs on the remote host, which need not have
    the engine installed in this machine's venv. ``"auto"`` folds to Isaac Lab
    (mirrors the head of the local ``_AUTO_PREFERENCE``). Unknown preferences
    raise rather than silently defaulting (CLAUDE.md §8).
    """
    from application.training.backend import (
        BACKEND_AUTO,
        BACKEND_ISAAC_LAB,
        BACKEND_SB3_MUJOCO,
    )

    p = (pref or BACKEND_AUTO).strip().lower()
    if p in ("", BACKEND_AUTO):
        return BACKEND_ISAAC_LAB
    if p in (BACKEND_ISAAC_LAB, BACKEND_SB3_MUJOCO):
        return p
    raise ValueError(
        f"Unknown backend preference {pref!r} for cloud target; valid: "
        f"{BACKEND_AUTO!r}, {BACKEND_ISAAC_LAB!r}, {BACKEND_SB3_MUJOCO!r}."
    )


__all__ = [
    "submit_il_trainer",
    "submit_sb3_trainer",
    "submit_canvas_training",
    "build_canvas_training_task",
    "CanvasTrainingPrepareTask",
    "compute_deploy_coverage",
    "sim2sim_residual_preflight",
    "DeployCoverageReport",
]
