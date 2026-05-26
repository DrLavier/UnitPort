# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""application.training.spec_compiler — IR → TrainingSpec assembly.

Stage 3.B/C surface: takes a :class:`WorkflowIR` (the canvas IR contract from
:mod:`application.compiler.lowering`) and emits a populated
:class:`TrainingSpec` according to ``MIGRATION_MAP.md``'s 27-node 1:N table.

The function is **side-effect free**: it does NOT touch disk, does NOT
spawn subprocesses, and does NOT submit to TasksManager. It reads
``registers.robots`` to snapshot a :class:`RobotSpecRef` into the spec.

Stage 3 status:
    * **SB3 path (PPO/SAC/TD3)** — fully wired. covers algorithm_config,
      robot, actor_setting, joint_init, init_pose, base_asset,
      physics_config, play_ground_setting (via fallback when absent),
      task_config, rewards (single + multigated), terminations,
      obs_action_config, env_assembler, train, domain_rand, eval_config,
      vis_check, export.
    * **IL path (PPO + AMP_PPO)** — partial. il_ppo_trainer / amp_trainer
      hyperparams are read into ``spec.algorithm.il_ppo``;
      il_observation, il_policy_network, discriminator,
      training_motion, stage_switch all flow through. Validation is
      delegated to :mod:`spec_validator`.

Two-phase usage:

    >>> ir = canvas_to_ir(canvas_dict)              # Stage 3 (later)
    >>> spec, issues = compile_training_spec(ir)
    >>> raise_if_errors(issues)                     # raises on hard errors
    >>> backend = select_backend(spec.algorithm.backend)
    >>> task = backend.build_task(spec.to_dict(), run_id)
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from unitport_sdk import log_warning

from registers import backends as backends_registry

from application.training.training_spec import (
    AlgorithmConfig,
    AMPConfig,
    ActorConfig,
    ActuatorConfig,
    ActionScaleCurriculum,
    CheckpointConfig,
    CommandCurriculumConfig,
    ContactSensorConfig,
    DiscriminatorConfig,
    DomainRandConfig,
    DomainRandIlConfig,
    DomainRandSb3Config,
    DomainRandSchedule,
    EnvAssemblerConfig,
    EvalConfig,
    ExportConfig,
    GaitConfig,
    HeightScanConfig,
    IlPpoConfig,
    ImitationLearningConfig,
    InitPoseConfig,
    MotionConfig,
    MotionRefConfig,
    ObsActionContract,
    PhysicsConfig,
    PolicyNetConfig,
    RewardConfig,
    RobotSpecRef,
    RoughTerrainConfig,
    SceneConfig,
    StageScheduleConfig,
    TaskConfig,
    TerminationConfig,
    TrainingSpec,
    VisCheckConfig,
)
from application.training.spec_validator import (
    AlgorithmFamily,
    IssueCode,
    Severity,
    ValidationIssue,
    check_spec,
    check_topology,
    classify_family,
)

if TYPE_CHECKING:
    from application.compiler.lowering import IRNode, WorkflowIR


# ---------------------------------------------------------------------------
# Compile-time issue context
# ---------------------------------------------------------------------------
#
# spec_compiler is documented as synchronous + side-effect-free. We use a
# single module-level list to thread the active issues collector into helpers
# (``_as_json``, ``_populate_robot``) without retrofitting kwargs onto every
# call site. ``compile_training_spec`` sets the active list on entry and
# clears it on exit; helpers call ``_emit_issue`` to push validation findings
# that surface alongside the topology / check_spec results.

_active_issues: Optional[List[ValidationIssue]] = None


def _emit_issue(issue: ValidationIssue) -> None:
    """Push a finding into the active compile context, or log_warning when
    no compile is in flight (defensive — should not happen during normal
    operation)."""
    if _active_issues is not None:
        _active_issues.append(issue)
    else:
        log_warning(f"[spec_compiler] orphan issue (no active compile): {issue}")


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def compile_training_spec(
    ir: "WorkflowIR",
    *,
    skip_validation: bool = False,
) -> Tuple[TrainingSpec, List[ValidationIssue]]:
    """Assemble :class:`TrainingSpec` from a :class:`WorkflowIR`.

    Args:
        ir: workflow IR (output of ``canvas_to_ir``)
        skip_validation: if True, skip topology/spec checks (useful for
            partial-canvas previews).

    Returns:
        ``(spec, issues)`` — ``spec`` is always returned (even if errors
        present); ``issues`` lists every finding. Caller decides whether to
        raise via :func:`raise_if_errors`.
    """
    global _active_issues
    issues: List[ValidationIssue] = []
    if not skip_validation:
        issues.extend(check_topology(ir))

    by_id: Dict[str, "IRNode"] = {n.schema_id: n for n in ir.nodes}
    family = classify_family(ir)
    # Backend is bound at the canvas level, not per-node. Translate the
    # canvas engine_id (e.g. ``"sb3_mujoco"``) to the per-node backend kind
    # (``"sb3"`` / ``"isaac_lab"``) consumed by rewards / terminations /
    # domain_rand populates.
    canvas_backend_kind = backends_registry.node_backend_kind(getattr(ir, "backend", None))

    prev_ctx, _active_issues = _active_issues, issues
    try:
        spec = TrainingSpec()
        _populate_robot(spec, by_id, canvas_backend_kind)
        _populate_algorithm(spec, by_id, family)
        _populate_actor(spec, by_id)
        _populate_obs_action(spec, by_id, canvas_backend_kind)
        _populate_physics(spec, by_id)
        _populate_scene(spec, by_id)
        _populate_task(spec, by_id)
        _populate_rewards(spec, by_id, ir, canvas_backend_kind)
        _populate_terminations(spec, by_id, canvas_backend_kind)
        _populate_motion(spec, by_id, family)
        _populate_il(spec, by_id, family)
        _populate_domain_rand(spec, by_id, canvas_backend_kind)
        _populate_stage_schedule(spec, by_id)
        _populate_env(spec, by_id)
        _populate_eval(spec, by_id)
        _populate_vis(spec, by_id)
        _populate_export(spec, by_id)

        # Strict-canvas single source of truth: backend lives ONLY on the
        # canvas root. Node-level backend ParamSpecs have been removed.
        # Every spec.*.backend field is derived from ``ir.backend`` /
        # ``canvas_backend_kind`` (the per-node "kind" form, sb3 /
        # isaac_lab — what reward/termination/domain_rand registries key
        # off). If the canvas itself lacks an engine binding the topology
        # check has already flagged it; emit ERROR here too as a
        # belt-and-suspenders for backend-derived populators that ran.
        canvas_engine_id = (getattr(ir, "backend", None) or "").strip()
        canvas_backend_kind_resolved = (canvas_backend_kind or "").strip()
        if canvas_backend_kind_resolved:
            spec.algorithm.backend = canvas_backend_kind_resolved
        elif canvas_engine_id:
            spec.algorithm.backend = canvas_engine_id
        else:
            _emit_issue(ValidationIssue(
                code=IssueCode.BACKEND_REGISTRY_MISMATCH,
                severity=Severity.ERROR,
                node_id="",
                field="backend",
                message=(
                    "canvas has no engine binding (canvas.backend is "
                    "empty); cannot derive spec.algorithm.backend. Open "
                    "the canvas in Studio and re-pick a backend."
                ),
            ))
    finally:
        _active_issues = prev_ctx

    spec.meta.setdefault("algorithm_family", family.value)
    spec.meta.setdefault("source_node_count", len(ir.nodes))
    spec.meta.setdefault("source_edge_count", len(ir.edges))

    if not skip_validation:
        issues.extend(check_spec(spec))

    return spec, issues


# ---------------------------------------------------------------------------
# Helpers — IR access
# ---------------------------------------------------------------------------


def _p(node: Optional["IRNode"], key: str, default: Any = None) -> Any:
    """Read one IRParam value off a node, or return ``default``."""
    if node is None:
        return default
    p = node.params.get(key)
    if p is None:
        return default
    val = getattr(p, "value", default)
    return val if val is not None else default


def read_headless_and_num_envs(ir: "WorkflowIR") -> Tuple[bool, int]:
    """Read the ``(headless, num_envs)`` an Isaac Lab run will use straight
    from the canvas IR, via the *same* node/param reads the compiler uses
    (``il_ppo_trainer``/``amp_trainer``.``headless`` and
    ``env_assembler``.``n_envs``). The UI BAR1 preflight calls this so its
    risk verdict matches the config that will actually launch — rather than
    re-deriving param keys (which would silently drift from the compiler)."""
    by_id = {n.schema_id: n for n in ir.nodes}
    il_node = by_id.get("il_ppo_trainer") or by_id.get("amp_trainer")
    headless = _as_bool(_p(il_node, "headless"), True)
    num_envs = _as_int(_p(by_id.get("env_assembler"), "n_envs"), 8)
    return headless, num_envs


def _as_int(v: Any, default: int) -> int:
    """Coerce ``v`` to int. ``None`` / empty string → caller default (treated
    as "user did not set" — Node.validate() is responsible for required-param
    enforcement). Any non-empty value that fails int() conversion emits a
    ``Severity.ERROR INVALID_PARAM_TYPE`` issue and falls back to ``default``
    so the rest of the populate path can finish; the compile end-of-run
    ``raise_if_errors`` will surface the error to the user."""
    if v is None or v == "":
        return default
    try:
        return int(v)
    except (TypeError, ValueError):
        _emit_issue(ValidationIssue(
            code=IssueCode.INVALID_PARAM_TYPE,
            severity=Severity.ERROR,
            message=f"_as_int: cannot coerce {v!r} ({type(v).__name__}) to int",
        ))
        return default


def _as_float(v: Any, default: float) -> float:
    """See :func:`_as_int` for the strict-mode contract."""
    if v is None or v == "":
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        _emit_issue(ValidationIssue(
            code=IssueCode.INVALID_PARAM_TYPE,
            severity=Severity.ERROR,
            message=f"_as_float: cannot coerce {v!r} ({type(v).__name__}) to float",
        ))
        return default


def _as_bool(v: Any, default: bool) -> bool:
    """See :func:`_as_int` for the strict-mode contract."""
    if v is None or v == "":
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "1", "yes", "on"):
            return True
        if s in ("false", "0", "no", "off"):
            return False
        _emit_issue(ValidationIssue(
            code=IssueCode.INVALID_PARAM_TYPE,
            severity=Severity.ERROR,
            message=f"_as_bool: cannot coerce {v!r} to bool",
        ))
        return default
    if isinstance(v, (int, float)):
        return bool(v)
    _emit_issue(ValidationIssue(
        code=IssueCode.INVALID_PARAM_TYPE,
        severity=Severity.ERROR,
        message=f"_as_bool: cannot coerce {v!r} ({type(v).__name__}) to bool",
    ))
    return default


def _as_str(v: Any, default: str) -> str:
    """``None`` → default; anything else stringified."""
    return str(v) if v is not None else default


def _as_json(v: Any, default: Any) -> Any:
    """Parse a value that may be a JSON string (canvas saves JSON-typed
    params as strings) or already a dict/list.

    A JSON parse failure pushes a ``Severity.ERROR`` ``INVALID_JSON_PARAM``
    issue into the active compile context (see :func:`compile_training_spec`)
    so :func:`raise_if_errors` halts the play-button submission instead of
    letting a typo slip through as a silent zero-config training run.
    """
    if v is None or v == "":
        return default
    if isinstance(v, (dict, list)):
        return v
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception as exc:
            preview = v if len(v) <= 80 else (v[:77] + "...")
            _emit_issue(ValidationIssue(
                code=IssueCode.INVALID_JSON_PARAM,
                severity=Severity.ERROR,
                message=f"JSON parse failed: {exc}; raw={preview!r}",
            ))
            return default
    _emit_issue(ValidationIssue(
        code=IssueCode.INVALID_PARAM_TYPE,
        severity=Severity.ERROR,
        message=f"_as_json: expected JSON string/dict/list, got {type(v).__name__}",
    ))
    return default


def _as_pair(v: Any, default: Tuple[float, float]) -> Tuple[float, float]:
    """Coerce a 2-tuple of floats. Length != 2 or non-numeric → emit ERROR."""
    if v is None or v == "":
        return default
    parsed = _as_json(v, None)
    if parsed is None:
        return default
    if not isinstance(parsed, (list, tuple)) or len(parsed) < 2:
        _emit_issue(ValidationIssue(
            code=IssueCode.INVALID_PARAM_TYPE,
            severity=Severity.ERROR,
            message=f"_as_pair: expected list of length 2, got {parsed!r}",
        ))
        return default
    try:
        return (float(parsed[0]), float(parsed[1]))
    except (TypeError, ValueError):
        _emit_issue(ValidationIssue(
            code=IssueCode.INVALID_PARAM_TYPE,
            severity=Severity.ERROR,
            message=f"_as_pair: non-numeric element in {parsed!r}",
        ))
        return default


def _wired_input(ir: "WorkflowIR", target_node_id: str, target_port: str) -> bool:
    return any(
        e.target_node == target_node_id and e.target_port == target_port
        for e in ir.edges
    )


def _resolve_latest_checkpoint(run_id: str) -> str:
    """Return abs path to the highest-step ``.pt`` under
    ``<project>/training/runs/<*>/<run_id>/checkpoints/`` for the active
    project, or ``""`` when no such checkpoint exists.

    Walks via the ``TrainingAssetsCache`` singleton so the run-dir
    lookup hits the canvas-init cache instead of re-scanning. Per-step
    ``.pt`` enumeration is a direct ``Path.glob`` on the single run's
    ``checkpoints/`` subdir (cheap; not cached).

    Empty return surfaces an inconsistency upstream — the validator's
    ``_check_base_asset_resolvable`` is expected to have already rejected
    the canvas before we got here; this function never silently maps a
    missing run to a different one.
    """
    if not run_id:
        return ""
    from application.service.training_assets import get_training_assets
    asset = get_training_assets().find_run(run_id)
    if asset is None or not asset.path.is_dir():
        return ""
    cps = asset.path / "checkpoints"
    if not cps.is_dir():
        return ""

    def _step(name: str) -> int:
        from pathlib import Path as _P
        stem = _P(name).stem
        if "_" in stem:
            tail = stem.rsplit("_", 1)[-1]
            if tail.isdigit():
                return int(tail)
        return -1

    pts = sorted(cps.glob("*.pt"), key=lambda p: _step(p.name), reverse=True)
    return str(pts[0]) if pts else ""


# ---------------------------------------------------------------------------
# Per-section populators
# ---------------------------------------------------------------------------


def _resolve_active_format(
    rs: Any, active_override: str, canvas_backend_kind: str,
) -> str:
    """Mirror of RobotNode.execute()'s active_format derivation.

    Priority:
      1. ``active_override`` ∈ {mjcf, usd, urdf} — explicit user choice
      2. Backend-mandated format:
           - isaac_lab → USD (mandatory; raises if joints_per_format["USD"]
             is null — see below)
           - else      → first available among ("MJCF", "USD", "URDF")

    CLAUDE.md §1.8: previously this walked ("USD", "URDF", "MJCF") for
    Isaac Lab and silently picked the first format with a populated joint
    table. That allowed IL training to proceed against MJCF joints when
    joints_per_format["USD"] was null, while Isaac Lab actually spawned
    the USD asset and reported its own articulation joint order to the
    policy. The trained bundle then shipped MJCF-ordered joint names →
    deploy ObsBuilder slotted obs/action through the wrong joints → user
    saw "electric-shock twitching". Refuse this substitution: for IL,
    USD is the contract.
    """
    override = str(active_override or "").strip().lower()
    if override in ("mjcf", "usd", "urdf"):
        return override.upper()
    bk = str(canvas_backend_kind or "").strip().lower()
    available = set(rs.available_formats)
    if bk == "isaac_lab":
        # Hard requirement: USD must be populated for the IL pipeline.
        # The caller (``_populate_robot``) checks ``active_format in
        # rs.available_formats`` after this returns; returning "USD" here
        # lets that gate emit a clean ValidationIssue pointing at "Dump
        # USD" when the table is missing, instead of a stale fallback.
        return "USD"
    for fmt in ("MJCF", "USD", "URDF"):
        if fmt in available:
            return fmt
    return getattr(rs, "preferred_format", "MJCF")


def _populate_robot(
    spec: TrainingSpec,
    by_id: Dict[str, "IRNode"],
    canvas_backend_kind: Optional[str] = None,
) -> None:
    n = by_id.get("robot")
    if n is None:
        # Topology check (R1) already flags missing required nodes per family;
        # silent return here lets the topology issue carry the error.
        return
    asset_id = _as_str(_p(n, "asset_id"), "")
    active_override = _as_str(_p(n, "active_override", "auto"), "auto")
    if not asset_id:
        _emit_issue(ValidationIssue(
            code=IssueCode.UNRESOLVED_ROBOT_ASSET,
            severity=Severity.ERROR,
            node_id=n.id,
            field="asset_id",
            message="robot.asset_id is empty; pick a robot in the canvas Robot node",
        ))
        return
    try:
        from registers import robots as registers_robots
        rs = registers_robots.get_robot_spec(asset_id)
    except Exception as exc:  # noqa: BLE001
        _emit_issue(ValidationIssue(
            code=IssueCode.UNRESOLVED_ROBOT_ASSET,
            severity=Severity.ERROR,
            node_id=n.id,
            field="asset_id",
            message=(
                f"registers.robots.get_robot_spec({asset_id!r}) raised: {exc}; "
                "the canvas robot is not registered or its registration failed validation"
            ),
        ))
        return
    if rs is None:
        _emit_issue(ValidationIssue(
            code=IssueCode.UNRESOLVED_ROBOT_ASSET,
            severity=Severity.ERROR,
            node_id=n.id,
            field="asset_id",
            message=(
                f"unknown robot asset_id={asset_id!r}; not found in registers.robots "
                "(import the robot via Robot Assets sidebar first)"
            ),
        ))
        return

    # Per-format schema (Stage 2): pin the format the rest of the pipeline
    # will resolve against. Mirrors RobotNode.execute()'s logic so canvas
    # display (Robot Asset card, BodyMappingTable) and spec compile agree.
    active_format = _resolve_active_format(rs, active_override, canvas_backend_kind or "")
    if not active_format or active_format not in rs.available_formats:
        # CLAUDE.md §1.8: surface as a hard error rather than silently
        # producing a spec with no joints/bodies (or silently substituting
        # a different format — see _resolve_active_format docstring).
        bk = str(canvas_backend_kind or "").strip().lower()
        if bk == "isaac_lab" and active_format == "USD":
            # IL-specific message: the backend mandates USD, but the
            # robot's USD joint table is empty.
            msg = (
                f"robot {asset_id!r} has no USD joint table populated "
                f"(joints_per_format[\"USD\"] is null); Isaac Lab training "
                f"requires the USD articulation order. Open the Robot Asset "
                f"card and run \"Dump USD\" so the registry records the "
                f"joint names Isaac Lab will report at runtime. Available "
                f"formats for this robot: "
                f"{sorted(rs.available_formats) or '[]'}."
            )
        else:
            msg = (
                f"robot {asset_id!r} has no joints/bodies declared for "
                f"format {active_format!r} (available={rs.available_formats}); "
                f"open the Robot Asset card and run Dump MJCF / Dump USD "
                f"before training"
            )
        _emit_issue(ValidationIssue(
            code=IssueCode.UNRESOLVED_ROBOT_ASSET,
            severity=Severity.ERROR,
            node_id=n.id,
            field="asset_id",
            message=msg,
        ))
        return

    spec.robot = RobotSpecRef.from_registry(
        rs, active_format=active_format,
    )


def _populate_algorithm(
    spec: TrainingSpec,
    by_id: Dict[str, "IRNode"],
    family: AlgorithmFamily,
) -> None:
    a = AlgorithmConfig()
    cfg = by_id.get("algorithm_config")
    if cfg is not None:
        a.algorithm = _as_str(_p(cfg, "algorithm"), "PPO")
        # NOTE: ``algorithm_config.backend`` was removed — backend is a
        # canvas-level concept (single source of truth). ``a.backend`` is
        # populated unconditionally at the end of ``compile_training_spec``
        # from ``ir.backend`` / ``canvas_backend_kind``.
        a.total_timesteps = _as_int(_p(cfg, "total_timesteps"), 1_000_000)
        a.learning_rate = _as_float(_p(cfg, "learning_rate"), 3e-4)
        a.batch_size = _as_int(_p(cfg, "batch_size"), 256)
        a.gamma = _as_float(_p(cfg, "gamma"), 0.99)
        a.seed = _as_int(_p(cfg, "seed"), 42)
        a.device = _as_str(_p(cfg, "device"), "auto")
        a.hidden_dim_1 = _as_int(_p(cfg, "hidden_dim_1"), 256)
        a.hidden_dim_2 = _as_int(_p(cfg, "hidden_dim_2"), 256)
        a.activation = _as_str(_p(cfg, "activation"), "relu")
        a.ent_coef = _as_str(_p(cfg, "ent_coef"), "auto")
        a.checkpoint_interval = _as_int(_p(cfg, "checkpoint_interval"), 50_000)
        a.resume_mode = _as_str(_p(cfg, "resume_mode"), "scratch")
        a.policy_id_out = _as_str(_p(cfg, "policy_id_out"), "trained_policy")
        # PPO
        a.gae_lambda = _as_float(_p(cfg, "gae_lambda"), 0.95)
        a.n_steps = _as_int(_p(cfg, "n_steps"), 2048)
        a.n_epochs = _as_int(_p(cfg, "n_epochs"), 10)
        a.lr_schedule = _as_str(_p(cfg, "lr_schedule"), "cosine")
        # Off-policy
        a.gradient_steps = _as_int(_p(cfg, "gradient_steps"), 1)
        a.tau = _as_float(_p(cfg, "tau"), 5e-3)
        a.target_entropy = _as_str(_p(cfg, "target_entropy"), "auto")
        a.buffer_size_mode = _as_str(_p(cfg, "buffer_size_mode"), "auto")
        a.buffer_size = _as_int(_p(cfg, "buffer_size"), 1_000_000)
        a.learning_starts = _as_int(_p(cfg, "learning_starts"), 10_000)
        a.use_per = _as_bool(_p(cfg, "use_per"), False)
        a.per_alpha = _as_float(_p(cfg, "per_alpha"), 0.6)
        a.per_beta = _as_float(_p(cfg, "per_beta"), 0.4)
        # Collapse guard
        a.collapse_guard = _as_bool(_p(cfg, "collapse_guard"), True)
        a.collapse_threshold = _as_float(_p(cfg, "collapse_threshold"), 0.65)
        a.collapse_patience = _as_int(_p(cfg, "collapse_patience"), 5)
        a.collapse_lr_factor = _as_float(_p(cfg, "collapse_lr_factor"), 0.3)
        a.collapse_max_rollbacks = _as_int(_p(cfg, "collapse_max_rollbacks"), 0)

    # IL trainer (mode + RSL-RL hyperparams) — pick il_ppo_trainer or amp_trainer
    il_node = by_id.get("il_ppo_trainer") or by_id.get("amp_trainer")
    if il_node is not None:
        a.training_mode = _as_str(_p(il_node, "training_mode"), "PPO")
        a.seed = _as_int(_p(il_node, "seed", a.seed), a.seed)
        a.il_ppo = IlPpoConfig(
            max_iterations=_as_int(_p(il_node, "max_iterations"), 1500),
            num_steps_per_env=_as_int(_p(il_node, "num_steps_per_env"), 24),
            num_learning_epochs=_as_int(_p(il_node, "num_learning_epochs"), 5),
            num_minibatches=_as_int(_p(il_node, "num_minibatches"), 4),
            learning_rate=_as_float(_p(il_node, "learning_rate"), 1e-3),
            discount_factor=_as_float(_p(il_node, "discount_factor"), 0.99),
            gae_lambda=_as_float(_p(il_node, "gae_lambda"), 0.95),
            clip_param=_as_float(_p(il_node, "clip_param"), 0.2),
            entropy_coef=_as_float(_p(il_node, "entropy_coef"), 0.01),
            value_loss_coef=_as_float(_p(il_node, "value_loss_coef"), 1.0),
            max_grad_norm=_as_float(_p(il_node, "max_grad_norm"), 1.0),
            schedule=_as_str(_p(il_node, "schedule"), "adaptive"),
            desired_kl=_as_float(_p(il_node, "desired_kl"), 0.01),
            save_interval=_as_int(_p(il_node, "save_interval"), 100),
            headless=_as_bool(_p(il_node, "headless"), True),
            entropy_schedule_enabled=_as_bool(_p(il_node, "entropy_schedule_enabled"), False),
            entropy_schedule_start=_as_float(_p(il_node, "entropy_schedule_start"), 0.01),
            entropy_schedule_end=_as_float(_p(il_node, "entropy_schedule_end"), 0.003),
            entropy_schedule_ramp_iters=_as_int(_p(il_node, "entropy_schedule_ramp_iters"), 1500),
        )

    # Policy network (il_policy_network)
    pn = by_id.get("il_policy_network")
    if pn is not None:
        a.policy_net = PolicyNetConfig(
            actor_hidden_dims=_as_json(_p(pn, "actor_hidden_dims"), [128, 64, 32]),
            critic_hidden_dims=_as_json(_p(pn, "critic_hidden_dims"), [128, 64, 32]),
            activation=_as_str(_p(pn, "activation"), "elu"),
            init_noise_std=_as_float(_p(pn, "init_noise_std"), -1.0),
        )

    # Checkpoint (base_asset) — v2 schema:
    #   start_point ∈ {__new__, __latest__, __load__}
    #   checkpoint_id is "run:<abs .pt>" or "export:<abs .onnx>" when __load__
    #   last_run_id is the canvas's most recent run_id (trainer-written)
    # Translated here into the legacy CheckpointConfig shape so the trainer
    # / IsaacLab adapter consume an unchanged contract.
    ba = by_id.get("base_asset")
    if ba is not None:
        sp = _as_str(_p(ba, "start_point"), "__new__")
        lm = _as_str(_p(ba, "load_mode"), "scratch")
        if sp == "__new__":
            a.checkpoint = CheckpointConfig(
                start_point="__new__", asset_id="",
                checkpoint_file="", load_mode="scratch",
            )
        elif sp == "__latest__":
            rid = _as_str(_p(ba, "last_run_id"), "")
            pt_path = _resolve_latest_checkpoint(rid)
            # Validator's _check_base_asset_resolvable already rejected
            # empty/missing cases — if we still got here with no path,
            # surface the inconsistency rather than masking with scratch.
            a.checkpoint = CheckpointConfig(
                start_point=f"run:{pt_path}" if pt_path else "__latest__",
                asset_id="",
                checkpoint_file=str(pt_path) if pt_path else "",
                load_mode=lm or "resume",
            )
        elif sp == "__load__":
            cid = _as_str(_p(ba, "checkpoint_id"), "")
            if cid.startswith("run:"):
                path = cid[4:]
                default_lm = "resume"
            elif cid.startswith("export:"):
                path = cid[7:]
                default_lm = "warm_start_actor"
            else:
                path = ""
                default_lm = lm or "warm_start_actor"
            a.checkpoint = CheckpointConfig(
                start_point=cid or "__load__", asset_id="",
                checkpoint_file=path,
                load_mode=lm or default_lm,
            )
        else:
            # Unknown start_point token — preserve verbatim so downstream
            # surfaces the bad value; no silent rewrite to __new__.
            a.checkpoint = CheckpointConfig(
                start_point=sp, asset_id="", checkpoint_file="",
                load_mode=lm or "scratch",
            )

    # Family override (in case algorithm_config.algorithm disagrees with trainer)
    if family in (AlgorithmFamily.IL_PPO, AlgorithmFamily.IL_AMP_PPO):
        a.training_mode = "AMP_PPO" if family is AlgorithmFamily.IL_AMP_PPO else "PPO"
    spec.algorithm = a


def _compile_pd_param(
    spec: TrainingSpec,
    by_id: Dict[str, "IRNode"],
) -> Tuple[Optional[Any], Optional[float], Optional[float], Optional[bool], Optional[bool], Optional[bool]]:
    """Build the canonical PDParam from the ActuatorPDNode + family defaults.

    Returns ``(pd_param, effort_limit, velocity_limit, resolve_at_reset,
    calibration_blocking, skip_calibration)``; ``pd_param`` and the three
    toggles come from RobotNode, while ``effort_limit`` / ``velocity_limit``
    are read from the ActorSetting node (their single canvas source of
    truth — de-duplicated off RobotNode). Every field is ``None`` when no
    PD config is present (engine compilers then fall back to the legacy
    ``ActuatorConfig.stiffness/damping`` scalar path).

    Validation that fails fast here:
      * ``robot`` node missing — falls back silently (older canvases).
      * ``spec.robot.families`` is empty — emits ``UNKNOWN_PARAM_VALUE``.
      * ``pd_groups`` JSON references a group that doesn't exist for
        the robot's primary family — ``UNKNOWN_PARAM_VALUE``.
    """
    # PD config now lives on RobotNode (merged from the short-lived
    # ActuatorPDNode in May 2026). Falls back to None when the canvas
    # is older than the merge and has no PD knobs on the robot node.
    pd_node = by_id.get("robot")
    if pd_node is None:
        return None, None, None, None, None, None
    # When RobotNode predates the PD merge it won't carry any pd_*
    # params; treat that as "no PD configured" and let legacy scalars
    # take over via the ActorSetting.actuator path.
    if not any(_p(pd_node, k, None) is not None for k in (
        "pd_groups", "pd_param_mode",
    )):
        return None, None, None, None, None, None

    families = list(getattr(spec.robot, "families", []) or [])
    if not families:
        _emit_issue(ValidationIssue(
            code=IssueCode.UNKNOWN_PARAM_VALUE,
            severity=Severity.ERROR,
            node_id=pd_node.id,
            field="pd_param",
            message=(
                "actuator_pd is wired but the bound robot has no declared "
                "families; cannot resolve (omega_n, zeta) defaults. Fix "
                "by completing the Robot Asset card's family selection."
            ),
        ))
        return None, None, None, None, None, None

    primary_family = families[0]

    try:
        from application.physics.pd_param import PDParam
        from registers import pd_groups as _pd_groups
    except Exception as exc:  # noqa: BLE001
        _emit_issue(ValidationIssue(
            code=IssueCode.UNKNOWN_PARAM_VALUE,
            severity=Severity.ERROR,
            node_id=pd_node.id,
            field="pd_param",
            message=(
                f"application.physics / registers.pd_groups failed to "
                f"import: {exc}"
            ),
        ))
        return None, None, None, None, None, None

    try:
        base = PDParam.from_family_defaults(primary_family)
    except Exception as exc:  # noqa: BLE001
        _emit_issue(ValidationIssue(
            code=IssueCode.UNKNOWN_PARAM_VALUE,
            severity=Severity.ERROR,
            node_id=pd_node.id,
            field="pd_param",
            message=(
                f"family={primary_family!r} has no PD groups registered "
                f"in pd_groups_definitions.json: {exc}"
            ),
        ))
        return None, None, None, None, None, None

    overrides_raw = _as_json(_p(pd_node, "pd_groups", "{}"), {})
    if not isinstance(overrides_raw, dict):
        overrides_raw = {}
    # Drop unknown groups loudly — silently dropping would mask a typo
    # the user explicitly typed into the canvas (rule §1.8).
    known_groups = set(base.groups.keys())
    clean_overrides: Dict[str, Dict[str, float]] = {}
    for gid, vals in overrides_raw.items():
        if gid not in known_groups:
            _emit_issue(ValidationIssue(
                code=IssueCode.UNKNOWN_PARAM_VALUE,
                severity=Severity.ERROR,
                node_id=pd_node.id,
                field="pd_groups",
                message=(
                    f"pd_groups[{gid!r}] is not a group defined for "
                    f"family={primary_family!r} (known groups: "
                    f"{sorted(known_groups)}). Either correct the group "
                    f"name or add it to pd_groups_definitions.json."
                ),
            ))
            continue
        if isinstance(vals, dict):
            clean_overrides[gid] = vals

    try:
        pd_param = base.with_overrides(clean_overrides) if clean_overrides else base
    except Exception as exc:  # noqa: BLE001
        _emit_issue(ValidationIssue(
            code=IssueCode.PARAM_OUT_OF_RANGE,
            severity=Severity.ERROR,
            node_id=pd_node.id,
            field="pd_groups",
            message=f"pd_groups override rejected: {exc}",
        ))
        return None, None, None, None, None, None

    # effort_limit / velocity_limit live ONLY on the ActorSetting node
    # (§4 Actuator overrides). They were de-duplicated off RobotNode
    # (ex pd_effort_limit / pd_velocity_limit) so there is one canvas
    # source of truth and both engines agree. The PD-process toggles
    # below stay on RobotNode alongside the (omega_n, zeta) source.
    actor_node = by_id.get("actor_setting")
    return (
        pd_param,
        _as_float(_p(actor_node, "effort_limit", 30.0), 30.0) if actor_node is not None else None,
        _as_float(_p(actor_node, "velocity_limit", 30.0), 30.0) if actor_node is not None else None,
        _as_bool(_p(pd_node, "pd_resolve_at_reset", True), True),
        _as_bool(_p(pd_node, "pd_calibration_blocking", True), True),
        _as_bool(_p(pd_node, "pd_skip_calibration", False), False),
    )


def _populate_actor(spec: TrainingSpec, by_id: Dict[str, "IRNode"]) -> None:
    n = by_id.get("actor_setting")
    if n is None:
        return
    actor = ActorConfig(
        init_pos_x=_as_float(_p(n, "init_pos_x"), 0.0),
        init_pos_y=_as_float(_p(n, "init_pos_y"), 0.0),
        init_pos_z=_as_float(_p(n, "init_pos_z"), 0.4),
        contact_sensors=ContactSensorConfig(
            body_names=list(_as_json(_p(n, "contact_body_names"), [])),
            history_length=_as_int(_p(n, "contact_history_length"), 3),
            track_air_time=_as_bool(_p(n, "contact_track_air_time"), True),
        ),
        actuator=ActuatorConfig(
            stiffness=_as_float(_p(n, "stiffness"), 25.0),
            damping=_as_float(_p(n, "damping"), 0.5),
            effort_limit=_as_float(_p(n, "effort_limit"), 30.0),
            velocity_limit=_as_float(_p(n, "velocity_limit"), 30.0),
        ),
        action_joint_names_expr=list(_as_json(_p(n, "action_joint_names_expr"), [])),
        action_scale=_as_float(_p(n, "action_scale"), 0.25),
        action_use_default_offset=_as_bool(_p(n, "action_use_default_offset"), True),
        action_curriculum=ActionScaleCurriculum(
            enabled=_as_bool(_p(n, "action_scale_curriculum_enabled"), False),
            start_scale=_as_float(_p(n, "action_scale_curriculum_start"), 0.15),
            end_scale=_as_float(_p(n, "action_scale_curriculum_end"), 0.25),
            ramp_iters=_as_int(_p(n, "action_scale_curriculum_ramp_iters"), 300),
        ),
    )

    # joint_init: actor_setting.init_joint_angles is the single source of
    # truth (joint_init node + input port removed in node-library cleanup).
    actor.joint_init = dict(_as_json(_p(n, "init_joint_angles"), {}))

    # init_pose: migrated onto actor_setting (init_pose node deleted).
    # ``base_height`` and ``custom_qpos`` are intentionally not exposed —
    # base_height duplicates init_pos_z / the model's nominal base z;
    # custom_qpos was unused by any canvas. Re-add as actor_setting params
    # later if a concrete need surfaces.
    actor.init_pose = InitPoseConfig(
        mode=_as_str(_p(n, "init_pose_mode"), "default"),
        noise_scale=_as_float(_p(n, "init_pose_noise_scale"), 0.05),
        base_height=-1.0,
        keyframe_name=_as_str(_p(n, "init_pose_keyframe_name"), "home"),
        custom_qpos=[],
        rsi_prob=_as_float(_p(n, "init_pose_rsi_prob"), 1.0),
        rsi_sample_mode=_as_str(_p(n, "init_pose_rsi_sample_mode"), "frame_0"),
    )

    # Stage C: canonical PD parameterization from the ActuatorPDNode.
    # Falls back to None (legacy scalar path) when no actuator_pd node
    # is wired; engine compilers handle both forks.
    (
        pd_param, pd_eff, pd_vel,
        pd_resolve_at_reset, pd_cal_blocking, pd_cal_skip,
    ) = _compile_pd_param(spec, by_id)
    actor.pd_param = pd_param
    actor.effort_limit = pd_eff
    actor.velocity_limit = pd_vel
    actor.resolve_at_reset = pd_resolve_at_reset
    actor.calibration_blocking = pd_cal_blocking
    actor.skip_calibration = pd_cal_skip

    spec.actor = actor


def _populate_obs_action(
    spec: TrainingSpec,
    by_id: Dict[str, "IRNode"],
    canvas_backend_kind: str = "",
) -> None:
    oa = ObsActionContract()
    n = by_id.get("obs_action_config")
    if n is not None:
        oa.contract_preset = _as_str(_p(n, "contract_preset"), "custom")
        oa_components_raw = _p(n, "obs_components", "")
        if isinstance(oa_components_raw, str):
            oa.components = oa_components_raw.split() or oa.components
        elif isinstance(oa_components_raw, list):
            oa.components = list(oa_components_raw)
        oa.obs_clip_range = _as_float(_p(n, "obs_clip_range"), 100.0)
        oa.frame_stack = _as_int(_p(n, "frame_stack"), 1)
        oa.action_type = _as_str(_p(n, "action_type"), "joint_position")
        oa.action_scale = _as_float(_p(n, "action_scale"), 1.0)
        oa.action_clip = _as_float(_p(n, "action_clip"), 1.0)

    il = by_id.get("il_observation")
    if il is not None:
        oa.il_terms = dict(_as_json(_p(il, "obs_terms"), {}))
        oa.group_name = _as_str(_p(il, "group_name"), "policy")
        oa.enable_corruption = _as_bool(_p(il, "enable_corruption"), True)
        oa.corruption_noise_std = _as_float(_p(il, "corruption_noise_std"), 0.05)
        oa.corruption_curriculum_enabled = _as_bool(_p(il, "obs_noise_curriculum_enabled"), False)
        oa.corruption_curriculum_start = _as_float(_p(il, "obs_noise_curriculum_start"), 0.25)
        oa.corruption_curriculum_end = _as_float(_p(il, "obs_noise_curriculum_end"), 1.0)
        oa.corruption_curriculum_ramp_iters = _as_int(_p(il, "obs_noise_curriculum_ramp_iters"), 500)
        # Variant tags on obs terms (Stage 4 architecture, biped/humanoid
        # special-casing). Source ends up in
        # ``oa.inline_source_overrides``; env_cfg_compiler IL emit path
        # (_custom_observation_funcs) reads it.
        _collect_variant_sources(
            oa.il_terms,
            kind="observation",
            backend=canvas_backend_kind,
            overrides=oa.inline_source_overrides,
            il_params_overrides=oa.il_params_overrides,
            robot_sku=getattr(spec.robot, "sku", "") or None,
        )
    spec.obs_action = oa


def _populate_physics(spec: TrainingSpec, by_id: Dict[str, "IRNode"]) -> None:
    n = by_id.get("physics_config")
    if n is None:
        return
    spec.physics = PhysicsConfig(
        sim_dt=_as_float(_p(n, "sim_dt"), 5e-3),
        control_dt=_as_float(_p(n, "control_dt"), 0.02),
        episode_max_steps=_as_int(_p(n, "episode_max_steps"), 1000),
        action_type=_as_str(_p(n, "action_type"), "joint_position"),
    )


def _populate_scene(spec: TrainingSpec, by_id: Dict[str, "IRNode"]) -> None:
    n = by_id.get("play_ground_setting")
    if n is None:
        return
    spec.scene = SceneConfig(
        scene_id=_as_str(_p(n, "scene_id"), "flat_ground"),
        scene_type=_as_str(_p(n, "scene_type"), "flat"),
        gravity_z=_as_float(_p(n, "gravity_z"), -9.81),
        arena_extent_x=_as_float(_p(n, "arena_extent_x"), 10.0),
        arena_extent_y=_as_float(_p(n, "arena_extent_y"), 10.0),
        friction_static=_as_float(_p(n, "friction_static"), 1.0),
        friction_dynamic=_as_float(_p(n, "friction_dynamic"), 0.8),
        rough=RoughTerrainConfig(
            amplitude=_as_float(_p(n, "roughness_amplitude"), 0.08),
            slope_max_deg=_as_float(_p(n, "slope_max"), 20.0),
            curriculum_enabled=_as_bool(_p(n, "curriculum_enabled"), False),
            difficulty_levels=_as_int(_p(n, "difficulty_levels"), 10),
        ),
        height_scan=HeightScanConfig(
            enabled=_as_bool(_p(n, "height_scan_enabled"), False),
            resolution=_as_float(_p(n, "scan_resolution"), 0.1),
            size_x=_as_float(_p(n, "scan_size_x"), 1.6),
            size_y=_as_float(_p(n, "scan_size_y"), 1.0),
        ),
        sim_dt=_as_float(_p(n, "sim_dt"), 5e-3),
        gpu_max_rigid_contact_count=_as_int(_p(n, "gpu_max_rigid_contact_count"), 524288),
        gpu_max_rigid_patch_count=_as_int(_p(n, "gpu_max_rigid_patch_count"), 81920),
    )


def _populate_task(spec: TrainingSpec, by_id: Dict[str, "IRNode"]) -> None:
    n = by_id.get("task_config")
    if n is None:
        return
    spec.task = TaskConfig(
        task_type=_as_str(_p(n, "task_type"), "velocity_tracking"),
        command_mode=_as_str(_p(n, "command_mode"), "fixed"),
        target_lin_vel_x=_as_float(_p(n, "target_lin_vel_x"), 0.5),
        target_lin_vel_y=_as_float(_p(n, "target_lin_vel_y"), 0.0),
        target_ang_vel_z=_as_float(_p(n, "target_ang_vel_z"), 0.0),
        curriculum=_as_bool(_p(n, "curriculum"), False),
        success_threshold=_as_float(_p(n, "success_threshold"), 0.8),
        truncation_max_steps=_as_int(_p(n, "truncation_max_steps"), 0),
        curriculum_schedule=dict(_as_json(_p(n, "curriculum_schedule"), {})),
        resampling_time_range=_as_pair(_p(n, "resampling_time_range"), (0.0, 0.0)),
        isaac_task_name=_as_str(_p(n, "isaac_task_name"), ""),
    )


def _collect_variant_sources(
    terms: Dict[str, Any],
    *,
    kind: str,
    backend: str,
    overrides: Dict[str, str],
    il_params_overrides: Optional[Dict[str, str]] = None,
    robot_sku: Optional[str] = None,
) -> None:
    """Walk a ``{key: payload}`` term dict; resolve variant tags into source.

    Mutates ``overrides`` (Python source) and ``il_params_overrides``
    (``il_params`` template per variant) in place. ``payload`` may be a
    legacy scalar weight (always preset, no-op here) or a structured
    dict with an optional ``"variant"`` field per
    :func:`application.compiler.term_payload.parse_term_payload`.

    ``robot_sku`` is forwarded to :func:`resolver.resolve` so the
    family filter rejects variants whose ``families`` list doesn't
    intersect the bound robot's families. A rejection falls back to
    preset (logged warning, not an error) — the canvas keeps training.

    Both user variants (USER_CONFIG_DIR overlay) and system variants
    (SDK factory under ``src/scripts``) are accepted; the resolver
    handles user-first ordering internally.

    Conflict policy: if the same ``key`` appears twice with different
    variants (e.g. two items pin different variants of
    ``action_rate_penalty``), the *last* iteration wins and a warning
    is logged. The Isaac Lab env_cfg can only emit one function per
    key, so collisions need user-side resolution; the warning surfaces
    in the training log.
    """
    try:
        from application.compiler.term_payload import parse_term_payload
        from application.service.scripts import resolver
    except Exception as exc:                                      # noqa: BLE001
        log_warning(
            f"[spec_compiler] variant resolver unavailable: {exc!r}"
        )
        return
    for key, val in (terms or {}).items():
        _, variant, _applies = parse_term_payload(val)
        if not variant:
            continue
        resolved = resolver.resolve(
            kind, key, variant=variant, backend=backend,
            robot_sku=robot_sku,
        )
        if resolved is None or resolved.origin not in ("user_variant", "system_variant"):
            log_warning(
                f"[spec_compiler] variant {variant!r} for {kind}/{key} "
                f"not resolvable for backend={backend!r} "
                f"robot_sku={robot_sku!r} — falling back to preset "
                f"(family filter or missing source)"
            )
            continue
        if key in overrides and overrides[key] != resolved.source:
            log_warning(
                f"[spec_compiler] variant conflict for {kind}/{key}: "
                f"two terms request different variants of the same "
                f"function; last write wins"
            )
        overrides[key] = resolved.source
        if il_params_overrides is not None and resolved.il_params_override:
            il_params_overrides[key] = resolved.il_params_override


def _populate_rewards(
    spec: TrainingSpec,
    by_id: Dict[str, "IRNode"],
    ir: "WorkflowIR",
    canvas_backend_kind: str,
) -> None:
    """R3: rewards × per-item / multigated merge.

    Strategy:
      * v2 per-item path: for each ``rewards`` node, follow its
        ``reward_pipe`` outbound edges into training_motion nodes' dynamic
        ``reward_in__<item_id>`` ports. Each match contributes the rewards
        node's ``reward_terms`` to ``rc.terms_by_item[item_id]``. A given
        item is supplied by at most one rewards node (UI enforces
        ``multi=False`` on item input).
      * Legacy multigated path: if a ``multigated_reward`` exists, the old
        stage_0/stage_1 merge still runs to populate ``rc.terms`` /
        ``rc.stages`` — so canvases that haven't migrated to per-item
        wiring keep working as the fallback global reward set.
      * Lone-rewards fallback: if there's exactly one rewards node and no
        multigated, its ``reward_terms`` is copied into ``rc.terms`` as
        the global fallback (consumed for items lacking a per-item
        connection).

    ``backend`` no longer lives on the rewards node — it derives from the
    canvas-bound backend (``ir.backend`` → ``node_backend_kind``).
    """
    rewards_nodes = [n for n in ir.nodes if n.schema_id == "rewards"]
    training_motion_ids = {
        n.id for n in ir.nodes if n.schema_id == "training_motion"
    }
    multi = by_id.get("multigated_reward")
    rc = RewardConfig()
    rc.backend = canvas_backend_kind

    # v2: collect per-item reward terms by walking each rewards node's
    # reward_pipe outbound edges. Target ports follow the
    # ``reward_in__<item_id>`` convention (see nodes.training_motion.node).

    def _follow_reward_pipe(src_id: str, src_port: str):
        """Yield (item_id, target_tm_id) for reachable training_motion.reward_in__<id>."""
        for e in ir.edges:
            if e.source_node != src_id or e.source_port != src_port:
                continue
            tgt_id = e.target_node
            tport = e.target_port or ""
            if tgt_id in training_motion_ids and tport.startswith("reward_in__"):
                item_id = tport[len("reward_in__"):]
                if item_id:
                    yield item_id, tgt_id

    # ``reward_in__<item_id>`` is now multi=True (see nodes.training_motion):
    # an item may be fed by ≥2 rewards nodes. The spec_validator's
    # _check_reward_term_conflicts has already rejected any same-key /
    # different-value collision before we get here, so the union below is
    # value-stable regardless of iteration order.
    for rn in rewards_nodes:
        rn_terms = dict(_as_json(_p(rn, "reward_terms"), {}))
        if not rn_terms:
            continue
        # Resolve any variant tags carried in the term payloads
        # (Stage 4). Source is staged in ``rc.inline_source_overrides``;
        # the IL env_cfg compiler reads this map and emits the variant
        # body in place of the registry preset's ``il_inline`` block.
        # ``robot_sku`` activates the family filter in resolver.resolve()
        # so a quadruped canvas can't accidentally pin a biped variant.
        _collect_variant_sources(
            rn_terms,
            kind="reward",
            backend=canvas_backend_kind,
            overrides=rc.inline_source_overrides,
            il_params_overrides=rc.il_params_overrides,
            robot_sku=getattr(spec.robot, "sku", "") or None,
        )
        for item_id, tm_id in _follow_reward_pipe(rn.id, "reward_pipe"):
            existing = rc.terms_by_item.get(item_id, {})
            rc.terms_by_item[item_id] = {**existing, **rn_terms}

    if multi is not None and rewards_nodes:
        # Index reward nodes by IR-id.
        by_iid = {n.id: n for n in rewards_nodes}
        # Lookup which rewards node feeds each multigated stage port.
        stage_inputs: Dict[str, Optional["IRNode"]] = {"stage_0": None, "stage_1": None}
        for e in ir.edges:
            if e.target_node == multi.id and e.target_port in stage_inputs:
                stage_inputs[e.target_port] = by_iid.get(e.source_node)

        stages: List[Dict[str, Any]] = []
        for port_name in ("stage_0", "stage_1"):
            r = stage_inputs.get(port_name)
            if r is None:
                continue
            stage_terms = dict(_as_json(_p(r, "reward_terms"), {}))
            _collect_variant_sources(
                stage_terms,
                kind="reward",
                backend=canvas_backend_kind,
                overrides=rc.inline_source_overrides,
                il_params_overrides=rc.il_params_overrides,
                robot_sku=getattr(spec.robot, "sku", "") or None,
            )
            stages.append(stage_terms)
        rc.stages = stages
        # Active stage on training start is stage 0.
        rc.terms = dict(stages[0]) if stages else {}
        # Use either reward node's std / threshold (stage_0 wins).
        head = stage_inputs["stage_0"] or stage_inputs["stage_1"]
        if head is not None:
            rc.std = _as_float(_p(head, "std"), 0.25)
            rc.threshold = _as_float(_p(head, "threshold"), 0.5)
    elif rewards_nodes:
        n = rewards_nodes[0]
        lone_terms = dict(_as_json(_p(n, "reward_terms"), {}))
        _collect_variant_sources(
            lone_terms,
            kind="reward",
            backend=canvas_backend_kind,
            overrides=rc.inline_source_overrides,
            il_params_overrides=rc.il_params_overrides,
            robot_sku=getattr(spec.robot, "sku", "") or None,
        )
        rc.terms = lone_terms
        rc.std = _as_float(_p(n, "std"), 0.25)
        rc.threshold = _as_float(_p(n, "threshold"), 0.5)

    spec.rewards = rc


def _populate_terminations(
    spec: TrainingSpec,
    by_id: Dict[str, "IRNode"],
    canvas_backend_kind: str,
) -> None:
    n = by_id.get("terminations")
    if n is None:
        return
    conditions = dict(_as_json(_p(n, "termination_conditions"), {}))
    tc = TerminationConfig(
        backend=canvas_backend_kind,
        conditions=conditions,
        curriculum_enabled=_as_bool(_p(n, "termination_curriculum_enabled"), False),
        curriculum_start=_as_float(_p(n, "termination_curriculum_start"), 0.18),
        curriculum_end=_as_float(_p(n, "termination_curriculum_end"), 0.22),
        curriculum_ramp_iters=_as_int(_p(n, "termination_curriculum_ramp_iters"), 500),
    )
    _collect_variant_sources(
        conditions,
        kind="termination",
        backend=canvas_backend_kind,
        overrides=tc.inline_source_overrides,
        il_params_overrides=tc.il_params_overrides,
        robot_sku=getattr(spec.robot, "sku", "") or None,
    )
    spec.terminations = tc


def _resolve_training_items(
    user_items: Dict[str, Any],
    robot,
) -> Dict[str, Any]:
    """H4-A: bind ``training_items`` to the canonical ``registers.commands``
    catalog. Behaviour:

    * Empty user dict → seed from
      :func:`registers.commands.default_items_for_families` (the 11 builtin
      templates documented in ``commands_defaults.json``).
    * Non-empty user dict → keep every key, but enrich the value with
      registry metadata (command_template / speed_channel / zero_channels
      / gait_hint) via :func:`registers.commands.enrich_user_item`.

    The result is the per-item dict shape consumed by
    :class:`MotionConfig.training_items` and by the SB3 / Isaac Lab
    runtimes downstream — so the canvas no longer carries inline JSON
    that drifts away from the registry.
    """
    from registers import commands as commands_registry

    families = list(getattr(robot, "families", []) or [])

    if not user_items:
        return commands_registry.default_items_for_families(families)

    out: Dict[str, Any] = {}
    for item_id, raw in user_items.items():
        out[str(item_id)] = commands_registry.enrich_user_item(str(item_id), raw)
    return out


def _populate_motion(
    spec: TrainingSpec,
    by_id: Dict[str, "IRNode"],
    family: AlgorithmFamily,
) -> None:
    n = by_id.get("training_motion")
    if n is None:
        return
    items = dict(_as_json(_p(n, "training_items"), {}))
    items = _resolve_training_items(items, spec.robot)
    spec.motion = MotionConfig(
        training_items=items,
        mapping_mode=_as_str(_p(n, "mapping_mode"), "linear"),
        deadzone=_as_float(_p(n, "deadzone"), 0.10),
        curve_exponent=_as_float(_p(n, "curve_exponent"), 2.0),
        runtime_clip=_as_bool(_p(n, "runtime_clip"), True),
        gait=GaitConfig(
            enabled=_as_bool(_p(n, "gait_enabled"), False),
            frequency_range_hz=_as_pair(_p(n, "gait_frequency_range"), (1.5, 3.5)),
            body_height_range_m=_as_pair(_p(n, "body_height_range"), (0.28, 0.40)),
            step_height_range_m=_as_pair(_p(n, "step_height_range"), (0.03, 0.15)),
            presets=_as_str(_p(n, "gait_presets"), ""),
        ),
        resampling_time_range=_as_pair(_p(n, "resampling_time_range"), (4.0, 12.0)),
        zero_command_probability=_as_float(_p(n, "zero_command_probability"), 0.1),
        cmd_step_change_prob=_as_float(_p(n, "cmd_step_change_prob"), 0.01),
        command_curriculum=CommandCurriculumConfig(
            enabled=_as_bool(_p(n, "command_curriculum_enabled"), False),
            start=_as_float(_p(n, "command_curriculum_start"), 0.25),
            end=_as_float(_p(n, "command_curriculum_end"), 1.0),
            ramp_iters=_as_int(_p(n, "command_curriculum_ramp_iters"), 800),
        ),
    )

    # Reference clip slice goes to spec.il.motion_ref.
    # The on-disk path stored in ``training_items[*].clip`` can be either
    # an absolute path or a ``pack:<package>:<subdir>/<file>`` token that
    # ``resolve_pack_ref`` looks up across both motion archive roots —
    # project-shipped ``<PROJECT_ROOT>/custom_mods/archives/`` first,
    # then user-installed ``<USER_CONFIG>/scripts/training_motion/archives/``.
    # Downstream consumers (validator's _validate_motion_clips,
    # isaac_lab/config.py's amp_motion_files plumbing, the launcher
    # subprocess) all expect a real on-disk path — resolve the token
    # here so the spec carries absolute paths from compile-time onward.
    # ``resolve_pack_ref`` surfaces a clean error listing every path it
    # tried, so the user knows where to drop the missing archive file.
    clip_paths: Dict[str, str] = {}
    for item_id, payload in items.items():
        if not (isinstance(payload, dict) and isinstance(payload.get("clip"), str)):
            continue
        raw = payload["clip"]
        if not raw:
            continue
        if raw.startswith("pack:"):
            try:
                from scripts import resolve_pack_ref
                resolved = resolve_pack_ref(raw)
            except ValueError as exc:
                _emit_issue(ValidationIssue(
                    code=IssueCode.INVALID_MOTION_CLIP,
                    severity=Severity.ERROR,
                    node_id=n.id,
                    field=f"training_items.{item_id}.clip",
                    message=str(exc),
                ))
                continue
            except Exception as exc:  # noqa: BLE001
                _emit_issue(ValidationIssue(
                    code=IssueCode.INVALID_MOTION_CLIP,
                    severity=Severity.ERROR,
                    node_id=n.id,
                    field=f"training_items.{item_id}.clip",
                    message=f"resolve_pack_ref({raw!r}) failed: {exc!r}",
                ))
                continue
            clip_paths[item_id] = str(resolved)
        else:
            clip_paths[item_id] = raw
    spec.il.motion_ref = MotionRefConfig(
        consumer_mode=_as_str(_p(n, "consumer_mode"), "amp"),
        phase_mode=_as_str(_p(n, "phase_mode"), "loop"),
        motion_fps=_as_float(_p(n, "motion_fps"), 50.0),
        random_start_phase=_as_bool(_p(n, "random_start_phase"), True),
        clip_paths=clip_paths,
    )

    # S6: motion clips must validate against the canvas robot before submit —
    # mismatched dof / unmapped IR roles surface here, not in the subprocess.
    if (
        family is AlgorithmFamily.IL_AMP_PPO
        and clip_paths
        and spec.robot.sku  # robot resolved; UNRESOLVED_ROBOT_ASSET handles the inverse
    ):
        _validate_motion_clips(n.id, clip_paths, spec.robot)


def _validate_motion_clips(
    node_id: str,
    clip_paths: Dict[str, str],
    robot,
) -> None:
    """Load each clip and run :func:`validate_clip_against_robot`; emit ERROR
    issues for load failures and IR-role / dof mismatches (CLAUDE.md §1.2)."""
    from pathlib import Path
    from application.training.motion.loader import LoaderError, get_loader
    from application.training.motion.validator import (
        MotionValidationError,
        validate_clip_against_robot,
    )

    for item_id, clip_path in clip_paths.items():
        if not clip_path:
            continue
        suffix = Path(clip_path).suffix.lower()
        if suffix == ".npy":
            format_id = "unitport_npy"
        elif suffix in (".txt", ".json"):
            format_id = "amp_legged_gym"
        else:
            _emit_issue(ValidationIssue(
                code=IssueCode.INVALID_MOTION_CLIP,
                severity=Severity.ERROR,
                node_id=node_id,
                field=f"training_items.{item_id}.clip",
                message=(
                    f"clip {clip_path!r}: cannot infer format from suffix "
                    f"{suffix!r}; expected .npy / .txt / .json"
                ),
            ))
            continue

        try:
            clip = get_loader(format_id).load(clip_path)
        except (LoaderError, FileNotFoundError, OSError) as exc:
            _emit_issue(ValidationIssue(
                code=IssueCode.INVALID_MOTION_CLIP,
                severity=Severity.ERROR,
                node_id=node_id,
                field=f"training_items.{item_id}.clip",
                message=f"clip {clip_path!r}: load failed: {exc}",
            ))
            continue

        try:
            report = validate_clip_against_robot(clip, robot)
        except MotionValidationError as exc:
            _emit_issue(ValidationIssue(
                code=IssueCode.INVALID_MOTION_CLIP,
                severity=Severity.ERROR,
                node_id=node_id,
                field=f"training_items.{item_id}.clip",
                message=f"clip {clip_path!r}: {exc}",
            ))
            continue

        if not report.ok:
            _emit_issue(ValidationIssue(
                code=IssueCode.INVALID_MOTION_CLIP,
                severity=Severity.ERROR,
                node_id=node_id,
                field=f"training_items.{item_id}.clip",
                message=(
                    f"clip {clip_path!r}: {report.message} "
                    f"(unmapped_bodies={report.unmapped_bodies!r})"
                ),
            ))


def _populate_il(
    spec: TrainingSpec,
    by_id: Dict[str, "IRNode"],
    family: AlgorithmFamily,
) -> None:
    if family is not AlgorithmFamily.IL_AMP_PPO:
        return
    il = ImitationLearningConfig(motion_ref=spec.il.motion_ref)
    amp_node = by_id.get("amp_trainer") or by_id.get("il_ppo_trainer")
    disc = by_id.get("discriminator")
    # discriminator is the single source of truth for the 9 AMP hyperparams.
    # The trainer node still carries the same fields (deprecated, kept for
    # legacy canvases that predate the discriminator split); only read them
    # as a fallback when the canvas has no discriminator node.
    if amp_node is not None and disc is None:
        il.amp = AMPConfig(
            amp_reward_coef=_as_float(_p(amp_node, "amp_reward_coef"), 2.0),
            task_reward_lerp=_as_float(_p(amp_node, "task_reward_lerp"), 0.5),
            disc_grad_penalty=_as_float(_p(amp_node, "disc_grad_penalty"), 10.0),
            disc_label_smoothing=_as_float(_p(amp_node, "disc_label_smoothing"), 0.9),
            amp_replay_buffer_size=_as_int(_p(amp_node, "amp_replay_buffer_size"), 1_000_000),
            num_preload_transitions=_as_int(_p(amp_node, "num_preload_transitions"), 2_000_000),
            disc_lr=_as_float(_p(amp_node, "disc_lr"), 1e-4),
            lerp_schedule=_as_str(_p(amp_node, "lerp_schedule"), "none"),
            lerp_schedule_json=_as_str(_p(amp_node, "lerp_schedule_json"), ""),
        )
    if disc is not None:
        il.amp.disc = DiscriminatorConfig(
            modules=dict(_as_json(_p(disc, "disc_modules"), {})),
            hidden_dims=list(_as_json(_p(disc, "disc_hidden_dims"), [1024, 512])),
            amp_reward_coef=_as_float(_p(disc, "amp_reward_coef"), 2.0),
            task_reward_lerp=_as_float(_p(disc, "task_reward_lerp"), 0.5),
            lerp_schedule=_as_str(_p(disc, "lerp_schedule"), "none"),
            lerp_schedule_json=_as_str(_p(disc, "lerp_schedule_json"), ""),
            disc_lr=_as_float(_p(disc, "disc_lr"), 1e-4),
            disc_grad_penalty=_as_float(_p(disc, "disc_grad_penalty"), 10.0),
            disc_label_smoothing=_as_float(_p(disc, "disc_label_smoothing"), 0.9),
            amp_replay_buffer_size=_as_int(_p(disc, "amp_replay_buffer_size"), 1_000_000),
            num_preload_transitions=_as_int(_p(disc, "num_preload_transitions"), 200_000),
            amp_obs_fields=_as_str(_p(disc, "amp_obs_fields"), ""),
            auto_inject_ref_obs=_as_bool(_p(disc, "auto_inject_ref_obs"), True),
            disc_logit_clamp_max=_as_float(_p(disc, "disc_logit_clamp_max"), 4.0),
            reward_clamp_per_step=_as_float(_p(disc, "reward_clamp_per_step"), 50.0),
            policy_std_clamp_max=_as_float(_p(disc, "policy_std_clamp_max"), 1.5),
        )
    spec.il = il


def _populate_domain_rand(
    spec: TrainingSpec,
    by_id: Dict[str, "IRNode"],
    canvas_backend_kind: str,
) -> None:
    n = by_id.get("domain_rand")
    if n is None:
        # domain_rand is optional — align the empty config's backend to the
        # canvas binding so ``_check_backend_registries`` does not flag a
        # phantom mismatch when the node simply isn't wired.
        spec.domain_rand.backend = canvas_backend_kind
        return
    backend = canvas_backend_kind
    spec.domain_rand = DomainRandConfig(
        backend=backend,
        schedule=DomainRandSchedule(
            mode=_as_str(_p(n, "rand_schedule"), "none"),
            start_step=_as_int(_p(n, "rand_schedule_start_step"), 0),
            end_step=_as_int(_p(n, "rand_schedule_end_step"), 500_000),
        ),
        sb3=DomainRandSb3Config(
            enabled=_as_bool(_p(n, "enabled"), True),
            mass_range=_as_pair(_p(n, "mass_range"), (0.8, 1.2)),
            friction_range=_as_pair(_p(n, "friction_range"), (0.5, 1.5)),
            # Stage H — read the new fields. Legacy canvases with
            # motor_strength_range / joint_damping_range raise via the
            # R_DR1 check in spec_validator (fail-loud per §1.8).
            omega_n_log_uniform=_as_pair(_p(n, "omega_n_log_uniform"), (0.8, 1.25)),
            zeta_log_uniform=_as_pair(_p(n, "zeta_log_uniform"), (0.9, 1.11)),
            obs_noise_std=_as_float(_p(n, "obs_noise_std"), 0.01),
            push_robot=_as_bool(_p(n, "push_robot"), False),
            push_interval_steps=_as_int(_p(n, "push_interval_steps"), 200),
            push_force_range=_as_pair(_p(n, "push_force_range"), (50.0, 150.0)),
        ),
        il=DomainRandIlConfig(
            friction_enabled=_as_bool(_p(n, "enable_friction_rand"), True),
            friction_mode=_as_str(_p(n, "friction_mode"), "reset"),
            static_friction_range=_as_pair(_p(n, "static_friction_range"), (0.5, 1.25)),
            dynamic_friction_range=_as_pair(_p(n, "dynamic_friction_range"), (0.5, 1.25)),
            restitution_range=_as_pair(_p(n, "restitution_range"), (0.0, 0.4)),
            mass_enabled=_as_bool(_p(n, "enable_mass_rand"), True),
            mass_mode=_as_str(_p(n, "mass_mode"), "reset"),
            mass_target_body=_as_str(_p(n, "mass_target_body"), "base"),
            mass_offset_range=_as_pair(_p(n, "mass_offset_range"), (-0.5, 0.5)),
            push_enabled=_as_bool(_p(n, "enable_external_push"), True),
            push_mode=_as_str(_p(n, "push_mode"), "interval"),
            push_interval_range_s=_as_pair(_p(n, "push_interval_range_s"), (10.0, 15.0)),
            push_velocity_x_range=_as_pair(_p(n, "push_velocity_x_range"), (-0.5, 0.5)),
            push_velocity_y_range=_as_pair(_p(n, "push_velocity_y_range"), (-0.5, 0.5)),
            init_pose_enabled=_as_bool(_p(n, "enable_init_pose_rand"), True),
            init_pose_mode=_as_str(_p(n, "init_pose_mode"), "reset"),
            init_pos_x_range=_as_pair(_p(n, "init_pos_x_range"), (-0.5, 0.5)),
            init_pos_y_range=_as_pair(_p(n, "init_pos_y_range"), (-0.5, 0.5)),
            init_yaw_range=_as_pair(_p(n, "init_yaw_range"), (-3.14, 3.14)),
            joint_noise_enabled=_as_bool(_p(n, "enable_joint_noise"), True),
            joint_noise_mode=_as_str(_p(n, "joint_noise_mode"), "reset"),
            joint_pos_noise=_as_float(_p(n, "joint_pos_noise"), 0.25),
            joint_vel_noise=_as_float(_p(n, "joint_vel_noise"), 0.1),
        ),
    )


def _populate_stage_schedule(spec: TrainingSpec, by_id: Dict[str, "IRNode"]) -> None:
    sched = StageScheduleConfig()
    multi = by_id.get("multigated_reward")
    if multi is not None:
        sched.max_step_stage0 = _as_int(_p(multi, "max_step_stage0"), 0)
        sched.reward_threshold_ratio = _as_float(_p(multi, "reward_threshold_ratio"), 0.75)
        sched.min_ep_window = _as_int(_p(multi, "min_ep_window"), 10)
        sched.plateau_window = _as_int(_p(multi, "plateau_window"), 10)
        sched.plateau_eps = _as_float(_p(multi, "plateau_eps"), 5e-3)
        sched.blend_steps = _as_int(_p(multi, "blend_steps"), 3000)
        sched.stage_behavior = _as_str(_p(multi, "stage_behavior"), "replace")
        sched.ep_reward_window = _as_int(_p(multi, "ep_reward_window"), 20)
        sched.stage1_ratio = _as_float(_p(multi, "stage1_ratio"), 1.5)
    sw = by_id.get("stage_switch")
    if sw is not None:
        sched.stages = list(_as_json(_p(sw, "stages_config"), []))
        sched.checkpoint_strategy = _as_str(_p(sw, "checkpoint_strategy"), "both")
    spec.stage_schedule = sched


def _populate_env(spec: TrainingSpec, by_id: Dict[str, "IRNode"]) -> None:
    n = by_id.get("env_assembler")
    if n is None:
        return
    spec.env = EnvAssemblerConfig(
        n_envs=_as_int(_p(n, "n_envs"), 8),
        vec_type=_as_str(_p(n, "vec_type"), "subproc"),
        obs_normalize=_as_bool(_p(n, "obs_normalize"), True),
        reward_normalize=_as_bool(_p(n, "reward_normalize"), False),
        clip_obs=_as_float(_p(n, "clip_obs"), 10.0),
        clip_reward=_as_float(_p(n, "clip_reward"), 10.0),
        action_clip_range=_as_float(_p(n, "action_clip_range"), 1.0),
        enable_monitor=_as_bool(_p(n, "enable_monitor"), True),
        time_limit_override=_as_int(_p(n, "time_limit_override"), 0),
        eval_disable_rand=_as_bool(_p(n, "eval_disable_rand"), True),
    )


def _populate_eval(spec: TrainingSpec, by_id: Dict[str, "IRNode"]) -> None:
    n = by_id.get("eval_config")
    if n is None:
        return
    spec.eval = EvalConfig(
        eval_episodes=_as_int(_p(n, "eval_episodes"), 20),
        deterministic=_as_bool(_p(n, "deterministic"), True),
        success_threshold=_as_float(_p(n, "success_threshold"), 0.8),
        record_video=_as_bool(_p(n, "record_video"), False),
        eval_interval=_as_int(_p(n, "eval_interval"), 50_000),
        save_best_model=_as_bool(_p(n, "save_best_model"), True),
        video_dir=_as_str(_p(n, "video_dir"), ""),
    )


def _populate_vis(spec: TrainingSpec, by_id: Dict[str, "IRNode"]) -> None:
    n = by_id.get("vis_check")
    if n is None:
        return
    spec.vis = VisCheckConfig(
        trigger_mode=_as_str(_p(n, "trigger_mode"), "step_interval"),
        vis_start_step=_as_int(_p(n, "vis_start_step"), 50_000),
        vis_interval_steps=_as_int(_p(n, "vis_interval_steps"), 50_000),
        num_vis_checks=_as_int(_p(n, "num_vis_checks"), 5),
        vis_episodes=_as_int(_p(n, "vis_episodes"), 3),
        deterministic=_as_bool(_p(n, "deterministic"), True),
    )


def _populate_export(spec: TrainingSpec, by_id: Dict[str, "IRNode"]) -> None:
    n = by_id.get("export")
    if n is None:
        return
    spec.export = ExportConfig(
        bundle_name=_as_str(_p(n, "bundle_name"), "<NEW>"),
        version=_as_str(_p(n, "version"), "v1"),
        include_onnx=_as_bool(_p(n, "include_onnx"), True),
        include_torchscript=_as_bool(_p(n, "include_torchscript"), True),
        include_normalization=_as_bool(_p(n, "include_normalization"), True),
        bundle_targets=list(_as_json(_p(n, "bundle_targets"), ["runtime_bundle"])),
        overwrite=_as_bool(_p(n, "overwrite"), True),
        review_backend=_as_str(_p(n, "review_backend"), "mujoco"),
        review_scene_id=_as_str(_p(n, "review_scene_id"), "flat_ground"),
        auto_import=_as_bool(_p(n, "auto_import"), True),
    )


__all__ = ["compile_training_spec"]
