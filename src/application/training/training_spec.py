# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""application.training.training_spec — Canvas → backend training contract.

This module defines :class:`TrainingSpec`, the canonical typed payload that
``application.compiler.lowering.canvas_to_ir`` emits and Stage 7/8/9/10
backend launchers (Isaac Lab + SB3+MuJoCo) consume.

The 1:N field-level mapping from the 27 canvas nodes onto this dataclass
tree lives in :mod:`MIGRATION_MAP.md`. **Read that doc before editing
fields here** — every field has a manifest source.

Stage 3 scope (this module):
    * Define the dataclass tree.
    * Provide ``TrainingSpec.to_dict() / from_dict()`` for IPC + JSON
      persistence (the SB3 subprocess launcher in Stage 10 ships specs as
      JSON over stdin).
    * Provide ``TrainingSpec.minimal()`` as a default-construction helper
      for tests + the smoke validator.

Out of scope here:
    * Field-by-field defaults that mirror EVERY manifest knob — the
      manifest is the source of truth, and lowering.py is what reads it.
      We declare structure + sentinel defaults; lowering is responsible
      for populating non-default values.
    * Validation. ``spec_validator.py`` (Stage 3.B) takes a populated spec
      and walks the cross-cutting rules R1–R7 from MIGRATION_MAP.md.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

# ---------------------------------------------------------------------------
# Algorithm / trainer
# ---------------------------------------------------------------------------


@dataclass
class PolicyNetConfig:
    """``il_policy_network`` actor/critic architecture.

    缺口① — ``rnn_type`` selects a recurrent (GRU/LSTM) actor; ``"none"``
    keeps the feed-forward MLP (default, byte-identical to the pre-recurrent
    path). The recurrent fields are inert when ``rnn_type == "none"``.
    """

    actor_hidden_dims: List[int] = field(default_factory=lambda: [128, 64, 32])
    critic_hidden_dims: List[int] = field(default_factory=lambda: [128, 64, 32])
    activation: str = "elu"
    init_noise_std: float = 1.0  # direct action std (legged_gym: 1.0 = std 1.0); never exp()'d
    rnn_type: str = "none"        # "none" | "gru" | "lstm"
    rnn_hidden_size: int = 256
    rnn_num_layers: int = 1


@dataclass
class IlPpoConfig:
    """RSL-RL-style PPO hyperparams (used by il_ppo_trainer + amp_trainer).

    SB3 path's PPO/SAC/TD3 hyperparams live on ``AlgorithmConfig`` directly
    because their key surface is different (n_steps vs num_steps_per_env etc.).
    Lowering picks ONE source — never both — per ``MIGRATION_MAP.md` R7.
    """

    max_iterations: int = 1500
    num_steps_per_env: int = 24
    num_learning_epochs: int = 5
    num_minibatches: int = 4
    learning_rate: float = 1e-3
    discount_factor: float = 0.99
    gae_lambda: float = 0.95
    clip_param: float = 0.2
    entropy_coef: float = 0.01
    value_loss_coef: float = 1.0
    max_grad_norm: float = 1.0
    schedule: str = "adaptive"        # "adaptive" | "fixed"
    desired_kl: float = 0.01
    save_interval: int = 100
    headless: bool = True
    # Entropy schedule (always-visible)
    entropy_schedule_enabled: bool = False
    entropy_schedule_start: float = 0.01
    entropy_schedule_end: float = 0.003
    entropy_schedule_ramp_iters: int = 1500


@dataclass
class CheckpointConfig:
    """``base_asset`` start-point selector — drives resume / warm-start."""

    start_point: str = "__new__"        # "__new__" | "__latest_export__" | "asset:<id>" | "run:<path>"
    asset_id: str = ""
    checkpoint_file: str = ""
    load_mode: str = "scratch"          # "scratch" | "resume" | "warm_start_actor"


@dataclass
class AlgorithmConfig:
    """``algorithm_config`` + (optional) ``il_ppo_trainer`` mode passthrough.

    ``training_mode`` is sourced from the trainer node when present, else
    derived from ``algorithm`` (SB3 SAC/TD3 imply non-PPO; SB3 PPO ⊂
    training_mode='PPO'). Always one of {"PPO", "AMP_PPO", "SAC", "TD3"}.
    """

    algorithm: str = "PPO"               # "PPO" | "SAC" | "TD3"
    training_mode: str = "PPO"           # "PPO" | "AMP_PPO"
    backend: str = "auto"                # "auto" | "isaac_lab" | "sb3_mujoco"
    # Training-length budget (sb3 path; il path uses IlPpoConfig.* instead).
    # PPO is iteration-based (matches IsaacLab's mental model + decouples the
    # budget from n_steps / n_envs): the launcher derives the SB3
    # ``total_timesteps`` = max_iterations × n_steps × n_envs. SAC/TD3 are
    # off-policy and have no rollout "iteration", so they use total_timesteps
    # directly.
    max_iterations: int = 1500           # PPO budget (× n_steps × n_envs → total_timesteps)
    total_timesteps: int = 1_000_000     # SAC/TD3 budget (env steps)
    learning_rate: float = 3e-4
    batch_size: int = 256
    gamma: float = 0.99
    seed: int = 42
    device: str = "auto"
    hidden_dim_1: int = 256
    hidden_dim_2: int = 256
    activation: str = "relu"
    ent_coef: str = "auto"               # "auto" | float-as-string
    checkpoint_interval: int = 50_000
    resume_mode: str = "scratch"
    policy_id_out: str = "trained_policy"
    # PPO-only (sb3 path)
    gae_lambda: float = 0.95
    n_steps: int = 2048
    n_epochs: int = 10
    lr_schedule: str = "cosine"          # "cosine" | "linear" | "constant"
    # Off-policy (sb3 SAC/TD3 path)
    gradient_steps: int = 1
    tau: float = 5e-3
    target_entropy: str = "auto"
    buffer_size_mode: str = "auto"
    buffer_size: int = 1_000_000
    learning_starts: int = 10_000
    use_per: bool = False
    per_alpha: float = 0.6
    per_beta: float = 0.4
    # Collapse guard (always visible)
    collapse_guard: bool = True
    collapse_threshold: float = 0.65
    collapse_patience: int = 5
    collapse_lr_factor: float = 0.3
    collapse_max_rollbacks: int = 0
    # Source-specific subspecs
    il_ppo: Optional[IlPpoConfig] = None       # set when algorithm is RSL-RL-driven
    policy_net: PolicyNetConfig = field(default_factory=PolicyNetConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)


# ---------------------------------------------------------------------------
# Actor / robot context
# ---------------------------------------------------------------------------


@dataclass
class InitPoseConfig:
    """``init_pose`` — episode start pose mode + RSI."""

    mode: str = "default"                # default / reference_frame_0 / keyframe / custom
    noise_scale: float = 0.05
    base_height: float = -1.0            # -1 = auto from RobotSpec
    keyframe_name: str = "home"
    custom_qpos: List[float] = field(default_factory=list)
    rsi_prob: float = 1.0
    rsi_sample_mode: str = "frame_0"     # frame_0 / uniform_phase


@dataclass
class ActuatorConfig:
    """Actuator torque/velocity caps. PD gains are NOT here.

    The only PD source is :attr:`ActorConfig.pd_param` (a :class:`PDParam`
    built from family defaults + RobotNode (omega_n, zeta) overrides);
    per-joint kp/kd are mass-weighted from it by the engine solvers
    (CLAUDE.md §10). The scalar ``stiffness``/``damping`` fields were removed
    in 2026-05 — raw kp/kd no longer exist in the data model. ``effort_limit``
    / ``velocity_limit`` are sourced from the RobotNode and kept here in sync
    with :attr:`ActorConfig.effort_limit` / :attr:`ActorConfig.velocity_limit`.
    """

    effort_limit: float = 30.0
    velocity_limit: float = 30.0


@dataclass
class ContactSensorConfig:
    """``actor_setting`` §3 — contact sensor declarations."""

    body_names: List[str] = field(default_factory=list)
    history_length: int = 3
    track_air_time: bool = True


@dataclass
class ActionScaleCurriculum:
    """``actor_setting`` §6."""

    enabled: bool = False
    start_scale: float = 0.15
    end_scale: float = 0.25
    ramp_iters: int = 300


@dataclass
class ActorConfig:
    """Combined ``actor_setting + joint_init + init_pose`` per MIGRATION_MAP.

    Phase 5 — IR-only joint contract:
      * ``joint_init`` keys MUST be IR role names from
        ``spec.robot.joint_ir_roles`` (e.g. ``"hip_FL"``, ``"thigh_FL"``).
        Physical USD joint names (``"FL_hip_joint"``) and vendor
        abbreviations (``"fl_hx"``) are rejected by ``spec_validator.R8``.
      * ``action_joint_names_expr`` items MUST be either a regex pattern
        (``".*"``, ``"hip_.*"``) or an exact IR role name.  Literal physical
        names are rejected.
      * Translation IR → physical happens at the substrate-emit boundary
        (env_cfg_compiler / generic_mujoco_env / deploy joint_space) via
        :class:`application.training.joint_ir.JointIRResolver`.
    """

    init_pos_x: float = 0.0
    init_pos_y: float = 0.0
    init_pos_z: float = 0.4
    # joint_init port wins over actor_setting.init_joint_angles when wired.
    # Keys MUST be IR roles (Phase 5 §IR-only contract); see class docstring.
    joint_init: Dict[str, float] = field(default_factory=dict)
    contact_sensors: ContactSensorConfig = field(default_factory=ContactSensorConfig)
    actuator: ActuatorConfig = field(default_factory=ActuatorConfig)
    # Action space (§5 absorbed from IL Action Config).
    # Items MUST be regex OR exact IR role names (Phase 5).
    action_joint_names_expr: List[str] = field(default_factory=list)
    action_scale: float = 0.25
    action_use_default_offset: bool = True
    action_curriculum: ActionScaleCurriculum = field(default_factory=ActionScaleCurriculum)
    init_pose: InitPoseConfig = field(default_factory=InitPoseConfig)
    # Canonical PD parameterization (Stage C of sim2sim_mass-matrix-adaptive).
    # Populated by ``spec_compiler._compile_pd_param`` from the RobotNode's
    # (omega_n, zeta) + family defaults. ``None`` only when the bound robot has
    # no declared families (pre-PD canvases); the scalar stiffness/damping
    # fallback was REMOVED in 2026-05 (raw kp/kd no longer exist, §10), so
    # bundle export now raises on ``pd_param is None`` rather than fabricating
    # scalar gains. The SB3 training env keeps a 25/0.5 default ONLY for the
    # legacy no-pd_param case (with a WARN).
    #
    # The type is intentionally Any here (not PDParam) to keep
    # ``application.training.training_spec`` free of an ``application.physics``
    # import dependency; the compiler stamps the typed object in.
    pd_param: Optional[Any] = None
    # Per-joint scalar limits flowing from ActuatorPDNode (§2). Overrides
    # ActuatorConfig.effort_limit / velocity_limit when ``pd_param`` is
    # present; mirror of the legacy fields so engine compilers have a
    # single source of truth regardless of which canvas node provided
    # the value.
    effort_limit: Optional[float] = None
    velocity_limit: Optional[float] = None
    # Runtime behavior toggles from ActuatorPDNode §3-§4. Consumed by
    # generic_mujoco_env (resolve_at_reset → DR-aware re-solve) and
    # sim2sim_calibration (calibration_*). None = use the ActuatorPDNode
    # default; the env / finalizer guard against missing values.
    resolve_at_reset: Optional[bool] = None
    calibration_blocking: Optional[bool] = None
    skip_calibration: Optional[bool] = None


# ---------------------------------------------------------------------------
# Obs / action contract
# ---------------------------------------------------------------------------


@dataclass
class ObsActionContract:
    """``obs_action_config`` (SB3 keys) + ``il_observation`` (IL keys).

    The SB3 path reads ``components`` / ``frame_stack`` / ``action_*``;
    the IL path reads ``il_terms`` / ``corruption*``. Both are kept on the
    same dataclass so backends pick what they need.
    """

    # SB3-style
    contract_preset: str = "custom"
    components: List[str] = field(default_factory=lambda: [
        "joint_pos", "joint_vel", "imu", "command", "previous_action",
    ])
    obs_clip_range: float = 100.0
    frame_stack: int = 1
    action_type: str = "joint_position"  # joint_position | torque
    action_scale: float = 1.0
    action_clip: float = 1.0
    # IL-style (il_observation). il_terms = POLICY obs (the deployed obs). The
    # critic's privileged terms are training-only and never enter this contract.
    il_terms: Dict[str, Any] = field(default_factory=dict)
    enable_corruption: bool = True
    corruption_noise_std: float = 0.05
    corruption_curriculum_enabled: bool = False
    corruption_curriculum_start: float = 0.25
    corruption_curriculum_end: float = 1.0
    corruption_curriculum_ramp_iters: int = 500
    # Variant injection for observations (parallel to RewardConfig /
    # TerminationConfig). Populated by spec_compiler when a term in
    # ``il_terms`` carries a ``variant`` tag in its payload dict.
    # Consumed by env_cfg_compiler IL emit path
    # (``_custom_observation_funcs``).
    inline_source_overrides: Dict[str, str] = field(default_factory=dict)
    il_params_overrides: Dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Physics / scene / task
# ---------------------------------------------------------------------------


@dataclass
class PhysicsConfig:
    """``physics_config`` (sb3 path); IL path overlays sim_dt from PlayGround.

    sim_dt aligns with :class:`SceneConfig.sim_dt` (0.005) so an unconfigured
    canvas does not fire ``_check_sim_dt`` warnings every compile.
    """

    sim_dt: float = 5e-3
    control_dt: float = 0.02
    episode_max_steps: int = 1000
    action_type: str = "joint_position"


@dataclass
class HeightScanConfig:
    enabled: bool = False
    resolution: float = 0.1
    size_x: float = 1.6
    size_y: float = 1.0


@dataclass
class RoughTerrainConfig:
    # NOTE (2026-06): amplitude / slope_max_deg removed — the canvas sliders
    # they mirrored were never fed into the IsaacLab generator's
    # noise_range / slope_range, so they had zero effect. Removed rather
    # than left as dead fields. The generator is driven by `proportions` +
    # `difficulty_levels`.
    curriculum_enabled: bool = False
    difficulty_levels: int = 10
    # Initial terrain difficulty level (curriculum start row); must be
    # < difficulty_levels. Maps to TerrainImporterCfg.max_init_terrain_level.
    max_init_terrain_level: int = 5
    # Per-sub-terrain mix weights (IsaacLab normalises internally). Defaults
    # are byte-equal to ROUGH_TERRAINS_CFG so legacy canvases are unchanged.
    proportions: Dict[str, float] = field(default_factory=lambda: {
        "pyramid_stairs": 0.2,
        "pyramid_stairs_inv": 0.2,
        "boxes": 0.2,
        "random_rough": 0.2,
        "hf_pyramid_slope": 0.1,
        "hf_pyramid_slope_inv": 0.1,
    })


@dataclass
class CustomTerrainConfig:
    """User-imported heightfield terrain (``scene_type == 'custom'``).

    The single cross-engine source is a canonical heightfield ``.npz``
    (see :mod:`application.training.terrain`) referenced by
    ``source_path``; MuJoCo and IsaacLab each derive their terrain from
    the SAME array (the cross-engine consistency gate enforces parity).
    Robot-agnostic — terrain is geometry, so no SKU/family binding.

    Pure additive field on :class:`SceneConfig` (default ``enabled=False``)
    → forward-safe through the strict ``_spec_from_dict`` (PV②): old specs
    lacking it load on the default.
    """

    enabled: bool = False
    #: Canonical heightfield ``.npz`` — a USER_CONFIG_DIR asset at train
    #: time, the in-bundle path at deploy/review time (§9 self-contained).
    source_path: str = ""
    #: Loader ``format_id`` for ``source_path`` (canonical asset is npz).
    source_format: str = "heightfield_npz"
    #: IsaacLab int16 vertical discretisation (m); see
    #: ``terrain.isaaclab_lowering.DEFAULT_VERTICAL_SCALE``. Recorded so the
    #: deploy/review side rebuilds the grid identically to training.
    vertical_scale: float = 0.005
    #: Expected height-bytes sha256 (provenance / integrity, §8).
    sha256: str = ""


@dataclass
class SceneConfig:
    """``play_ground_setting`` — full §2 Scene block."""

    scene_id: str = "flat_ground"
    scene_type: str = "flat"             # flat | rough | custom
    gravity_z: float = -9.81
    arena_extent_x: float = 10.0
    arena_extent_y: float = 10.0
    friction_static: float = 1.0
    friction_dynamic: float = 0.8
    rough: RoughTerrainConfig = field(default_factory=RoughTerrainConfig)
    custom: CustomTerrainConfig = field(default_factory=CustomTerrainConfig)
    height_scan: HeightScanConfig = field(default_factory=HeightScanConfig)
    # IL path uses these as authoritative (R4)
    sim_dt: float = 5e-3
    gpu_max_rigid_contact_count: int = 524288
    gpu_max_rigid_patch_count: int = 81920


@dataclass
class TaskConfig:
    """``task_config`` — RL task semantics (velocity tracking + curriculum).

    Channel naming follows :data:`registers.commands.commands_defaults.json`:
    ``lin_vel_x / lin_vel_y / ang_vel_z`` are the canonical command channels
    shared by the env command vector, AMP tag router, Isaac Lab reward
    tracking, and the registry's command_template entries.
    """

    task_type: str = "velocity_tracking"
    command_mode: str = "fixed"          # fixed | random
    target_lin_vel_x: float = 0.5
    target_lin_vel_y: float = 0.0
    target_ang_vel_z: float = 0.0
    curriculum: bool = False
    success_threshold: float = 0.8
    truncation_max_steps: int = 0
    curriculum_schedule: Dict[str, Any] = field(default_factory=dict)
    resampling_time_range: Tuple[float, float] = (0.0, 0.0)
    # Isaac Lab gym registry id (e.g. ``Isaac-Velocity-Flat-Unitree-Go2-v0``)
    # — read by :class:`IsaacLabConfig` when the canvas backend is Isaac Lab.
    # Empty falls back to the adapter's hardcoded default.
    isaac_task_name: str = ""


# ---------------------------------------------------------------------------
# Rewards / terminations / stage schedule
# ---------------------------------------------------------------------------


@dataclass
class RewardConfig:
    """``rewards`` + ``multigated_reward`` merge (R3).

    v2 颗粒度：``terms_by_item`` 是 per-motion-item 的 reward_terms 映射，
    由 spec_compiler 沿每个 rewards 节点的 ``reward_pipe`` 出边反查
    target training_motion item id 后填入。``terms`` 仍保留作为"全局
    rewards"的兼容字段（multigated_reward / 旧 canvas / item 缺连线时
    的 fallback）。

    Variant injection (Stage 4): ``inline_source_overrides`` is the
    flat ``{key: python_source}`` map populated by spec_compiler from
    user-selected variants in the canvas reward_terms dict. The
    Isaac Lab env_cfg compiler reads this to emit the variant body in
    place of the registry's preset ``il_inline`` block.
    """

    backend: str = "sb3_mujoco"          # "sb3_mujoco" | "isaac_lab"
    terms: Dict[str, Any] = field(default_factory=dict)         # active stage / fallback
    terms_by_item: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    stages: List[Dict[str, Any]] = field(default_factory=list)  # [stage0, stage1, ...]
    std: float = 0.25
    threshold: float = 0.5
    # Variant source overrides — key → Python source. Populated by
    # spec_compiler when a reward term carries a non-preset ``variant``
    # tag in its payload dict. Consumed by env_cfg_compiler (IL) and
    # the SB3 env if a variant-aware reward registry is wired later.
    inline_source_overrides: Dict[str, str] = field(default_factory=dict)
    # Per-variant overrides of the preset's ``il_params`` template
    # (e.g. swap ``body_names={ir:feet}`` for ``body_names={ir:ankles}``
    # on a biped variant of ``feet_air_time``). Populated by
    # spec_compiler from ``VariantMeta.il_params_override`` when the
    # variant declares one; absent keys fall back to preset il_params.
    il_params_overrides: Dict[str, str] = field(default_factory=dict)


@dataclass
class TerminationConfig:
    """``terminations`` — backend-keyed; curriculum on base_height threshold."""

    backend: str = "sb3_mujoco"
    conditions: Dict[str, Any] = field(default_factory=dict)
    curriculum_enabled: bool = False
    curriculum_start: float = 0.18
    curriculum_end: float = 0.22
    curriculum_ramp_iters: int = 500
    # See :class:`RewardConfig.inline_source_overrides`.
    inline_source_overrides: Dict[str, str] = field(default_factory=dict)
    # See :class:`RewardConfig.il_params_overrides`.
    il_params_overrides: Dict[str, str] = field(default_factory=dict)


STAGE_SCHEDULE_SCHEMA_VERSION = 2

# Per-stage H0 defaults. The schema carries the full forward-compatible
# field set (role / obs_profile / init_from / distill_loss) so the H2
# teacher-student stage can land without a schema migration; H0 locks
# every non-default to a fail-loud raise via validate_stage_entry_h0.
_STAGE_H0_DEFAULTS: Dict[str, Any] = {
    "role": "trainable",
    "obs_profile": "default",
    "init_from": None,
    "distill_loss_enabled": False,
}


@dataclass
class DistillLossSpec:
    """Teacher-student distillation block on a stage entry.

    H0 locks every field to its default; any non-default raises with
    ``reserved for H2 teacher-student stage``.
    """

    enabled: bool = False
    teacher_stage: Optional[str] = None
    loss_type: Optional[str] = None


@dataclass
class StageEntry:
    """One entry in :class:`StageScheduleConfig.stages` (schema v2).

    Field set mirrors PRINCIPLES R3-C verbatim — declared up-front so the
    H2 teacher-student stage only requires unlocking the H0 raise
    branches in :func:`validate_stage_entry_h0`, not a schema migration.
    """

    name: str = ""
    role: str = "trainable"
    iterations: int = 0
    obs_profile: str = "default"
    init_from: Optional[str] = None
    distill_loss: DistillLossSpec = field(default_factory=DistillLossSpec)
    overrides: Dict[str, Any] = field(default_factory=dict)


def _stage_h2_raise(field_path: str, found: Any, stage_name: str) -> None:
    """Raise the standard three-part directive for an H0-locked field.

    Field path is the dotted location inside the stage entry
    (``role`` / ``obs_profile`` / ``init_from`` / ``distill_loss.enabled``).
    """
    raise ValueError(
        f"stage_schedule.stages[{stage_name!r}].{field_path}={found!r}: "
        f"reserved for H2 teacher-student stage. "
        f"H0 only accepts the defaults role='trainable', "
        f"obs_profile='default', init_from=None, distill_loss.enabled=False."
    )


def validate_stage_entry_h0(entry: Mapping[str, Any]) -> None:
    """Raise on any H0-locked stage field that carries a non-default value.

    Shared by every stage_schedule boundary (spec_compiler populate,
    spec_validator, isaac_lab serializer, il_train_launcher decoder,
    amp_on_policy_runner consumer) so that a hand-edited or
    out-of-process payload cannot bypass the populate-time check.
    """
    name = str(entry.get("name", ""))
    role = entry.get("role", _STAGE_H0_DEFAULTS["role"])
    if role != _STAGE_H0_DEFAULTS["role"]:
        _stage_h2_raise("role", role, name)
    obs_profile = entry.get("obs_profile", _STAGE_H0_DEFAULTS["obs_profile"])
    if obs_profile != _STAGE_H0_DEFAULTS["obs_profile"]:
        _stage_h2_raise("obs_profile", obs_profile, name)
    init_from = entry.get("init_from", _STAGE_H0_DEFAULTS["init_from"])
    if init_from is not _STAGE_H0_DEFAULTS["init_from"]:
        _stage_h2_raise("init_from", init_from, name)
    distill = entry.get("distill_loss", {})
    if distill is None:
        distill = {}
    if not isinstance(distill, Mapping):
        raise ValueError(
            f"stage_schedule.stages[{name!r}].distill_loss must be a dict, "
            f"got {type(distill).__name__}."
        )
    if bool(distill.get("enabled", _STAGE_H0_DEFAULTS["distill_loss_enabled"])):
        _stage_h2_raise("distill_loss.enabled", True, name)
    overrides = entry.get("overrides", {})
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, Mapping):
        raise ValueError(
            f"stage_schedule.stages[{name!r}].overrides must be a dict, "
            f"got {type(overrides).__name__}."
        )


def validate_stage_schedule_dict_h0(d: Mapping[str, Any]) -> None:
    """Raise if a serialized stage_schedule dict is malformed for H0.

    Verifies ``schema_version`` is present and equals the current
    :data:`STAGE_SCHEDULE_SCHEMA_VERSION` (no silent v1 fill — missing
    versions are fail-loud, not back-filled with defaults), then
    delegates per-entry checks to :func:`validate_stage_entry_h0`.
    """
    if not isinstance(d, Mapping):
        raise ValueError(
            f"stage_schedule payload must be a dict, got {type(d).__name__}."
        )
    if "schema_version" not in d:
        raise ValueError(
            "stage_schedule.schema_version missing: required field. "
            "Re-emit the schedule with "
            f"schema_version={STAGE_SCHEDULE_SCHEMA_VERSION}."
        )
    sv = d["schema_version"]
    if sv != STAGE_SCHEDULE_SCHEMA_VERSION:
        raise ValueError(
            f"stage_schedule.schema_version={sv!r}: expected "
            f"{STAGE_SCHEDULE_SCHEMA_VERSION}."
        )
    stages = d.get("stages", [])
    if stages is None:
        stages = []
    if not isinstance(stages, list):
        raise ValueError(
            f"stage_schedule.stages must be a list, got {type(stages).__name__}."
        )
    for entry in stages:
        if not isinstance(entry, Mapping):
            raise ValueError(
                f"stage_schedule.stages entry must be a dict, "
                f"got {type(entry).__name__}."
            )
        validate_stage_entry_h0(entry)


@dataclass
class StageScheduleConfig:
    """``stage_switch`` + ``multigated_reward`` curriculum gate (schema v2).

    ``schema_version`` is the forward-compatibility hook for H2
    teacher-student staging; absent on a deserialized payload is
    fail-loud at :func:`validate_stage_schedule_dict_h0`.
    """

    schema_version: int = STAGE_SCHEDULE_SCHEMA_VERSION
    stages: List[StageEntry] = field(default_factory=list)
    checkpoint_strategy: str = "both"    # both | best | last
    # multigated_reward fields:
    max_step_stage0: int = 0
    reward_threshold_ratio: float = 0.75
    min_ep_window: int = 10
    plateau_window: int = 10
    plateau_eps: float = 5e-3
    blend_steps: int = 3000
    stage_behavior: str = "replace"      # replace | accumulate
    ep_reward_window: int = 20
    stage1_ratio: float = 1.5

    def validate_h0(self) -> None:
        """Re-run the H0 dict-level guard on this dataclass.

        Defensive — covers the case where the dataclass was mutated in
        memory after :func:`spec_compiler._populate_stage_schedule`.
        """
        validate_stage_schedule_dict_h0(asdict(self))


# ---------------------------------------------------------------------------
# Motion / IL / AMP
# ---------------------------------------------------------------------------


@dataclass
class GaitConfig:
    """``training_motion`` §3 Walk-These-Ways.

    Range fields default to ``None`` ("not configured"): the numeric
    family defaults live ONLY in
    ``registers/data/gait_commands_catalog.json`` (``default_ranges``,
    read via ``gait_presets.default_ranges_for_family``).
    ``spec_compiler._populate_motion`` resolves canvas-or-registry
    values whenever gait is enabled; a disabled gait legitimately
    carries ``None`` (no downstream reader — the env_cfg_compiler reads
    the node params directly, and spec_validator F2 reads ``enabled``
    only). Kept as carried provenance in the spec record, never a
    duplicate default-value declaration.
    """

    enabled: bool = False
    frequency_range_hz: Optional[Tuple[float, float]] = None
    body_height_range_m: Optional[Tuple[float, float]] = None
    step_height_range_m: Optional[Tuple[float, float]] = None
    presets: str = ""


@dataclass
class CommandCurriculumConfig:
    enabled: bool = False
    start: float = 0.25
    end: float = 1.0
    ramp_iters: int = 800


# Sentinel id for the implicit default package that wraps all flat
# ``training_items`` when no explicit package layer is authored (Method A,
# Slice 1). Reserved — a user-authored package may not use this id.
DEFAULT_PACKAGE_ID = "__default__"


@dataclass
class TrainingPackage:
    """A training package — an authoring grouping + reward-composition unit that
    always maps to ONE policy network (Method A, Slice 1).

    A package groups training items (membership = ``training_items[id]['package_id']``)
    and owns an inline reward set plus one coarse ``package_weight`` global-rebalance
    lever (e.g. locomotion 0.7 vs manipulation 0.3). It is NOT a canvas node (it lives
    inside the single ``training_motion`` node), NOT a sub-policy, and NEVER owns a joint
    subset — all enabled packages collapse into one reward vector / one obs vector / one
    action space / one PPO update. Reward joint-paging is an orthogonal ``reward-over-joints``
    concern and is never conflated with package grouping.

    ``reward_terms`` reuses the paged payload shape a rewards node emits
    (``{page_id: {func: payload}}``); it is lowered into the shared reward seam at
    compile time so downstream readers stay package-blind. See
    ``knowledge_base/training_package_schema_design.md``.
    """

    package_id: str = ""
    name: str = ""                       # display only; i18n at the UI boundary, never a key
    enabled: bool = True
    package_weight: float = 1.0          # multiplies the term weights of this package's items
    reward_terms: Dict[str, Any] = field(default_factory=dict)  # paged {page_id: {func: payload}}
    # Skill trigger gate (Method A / skill_command_path_design.md Slice 3). When set to
    # a skill_id, this package's reward is gated on that skill's trigger command (a
    # post-pulse window with decay) — the package becomes a "skill package" (§1: a skill
    # is a package whose command interface is a trigger channel). "" = ungated (default).
    gated_by: str = ""


@dataclass
class MotionConfig:
    """``training_motion`` — command envelope only.

    Reference-clip slice goes to ``ImitationLearningConfig.motion_ref``.
    """

    training_items: Dict[str, Any] = field(default_factory=dict)
    # Package layer (Method A, Slice 1) — presence-gated additive. Empty ⇒
    # ``resolve_effective_packages()`` wraps all flat items into one implicit
    # default package at ``package_weight=1.0`` (byte-identical to pre-package
    # behavior). Membership is ``training_items[id]['package_id']``; each package
    # carries inline ``reward_terms`` + a coarse ``package_weight`` rebalance lever.
    # NOT a canvas node, NOT a sub-policy, NEVER owns a joint subset. See
    # ``knowledge_base/training_package_schema_design.md``.
    packages: Dict[str, TrainingPackage] = field(default_factory=dict)
    # Skill trigger channels (skill_command_path_design.md Slice 2/3) —
    # {skill_id: {name, kind:"trigger", latch:"pulse"|"hold", enabled}}. Each enabled
    # entry becomes a trigger command channel on the contract (Slice 2) and, on
    # IsaacLab, a command term + obs term + reward gate for the package whose
    # ``gated_by`` names it (Slice 3). Empty => no triggers (byte-identical).
    skill_items: Dict[str, Any] = field(default_factory=dict)
    mapping_mode: str = "linear"
    deadzone: float = 0.10
    curve_exponent: float = 2.0
    runtime_clip: bool = True
    gait: GaitConfig = field(default_factory=GaitConfig)
    resampling_time_range: Tuple[float, float] = (4.0, 12.0)
    # Superseded by the explicit 'stand' training item — no longer a node param.
    # Kept as an inert field (default 0.0 = off) so deploy/SB3 readers that still
    # getattr() it don't break; nothing in the canvas can set it non-zero now.
    zero_command_probability: float = 0.0
    cmd_step_change_prob: float = 0.01
    command_curriculum: CommandCurriculumConfig = field(default_factory=CommandCurriculumConfig)
    # Adaptive item sampling (Phase C, IsaacLab-only). Items start at equal
    # weight; when enabled the env-side command term re-biases sampling toward
    # under-performing items every ``adaptive_update_interval`` iterations,
    # bounded by [floor, ceil]. Carried here for SB3 fail-loud gating + record;
    # the IsaacLab compiler reads the training_motion node params directly.
    adaptive_motion_enabled: bool = False
    adaptive_update_interval: int = 50
    adaptive_weight_floor: float = 0.03
    adaptive_weight_ceil: float = 0.30
    # Heading-command mode (legged_gym parity). When True, the yaw channel
    # (ang_vel_z) is NOT sampled directly: each resample draws a target
    # heading (world yaw, rad) in ``heading_range``, and every control step
    # the env recomputes ang_vel_z = clip(heading_control_stiffness *
    # wrap_to_pi(target - current_yaw), <item ang_vel_z range>). This is the
    # closed-loop heading tracking legged_gym uses (commands[:,2] derived
    # from heading error). Default off = byte-identical open-loop yaw sampling.
    heading_command: bool = False
    heading_control_stiffness: float = 0.5
    heading_range: Tuple[float, float] = (-3.141592653589793, 3.141592653589793)


@dataclass
class MotionRefConfig:
    """``training_motion`` reference-clip slice consumed by AMP.

    ``consumption_mode`` is REQUIRED — the canvas
    ``training_motion.consumption_mode`` ParamSpec carries the user's
    choice between the three enum values below. The compiler emits a
    ``MISSING_CONSUMPTION_MODE`` validation issue (and skips populating
    ``spec.il.motion_ref``) when the canvas leaves the field blank,
    rather than substituting an ``"amp_discriminator"`` default that
    silently routes the user's tracking clips through the AMP
    discriminator. ``__post_init__`` enforces the non-empty invariant
    at construction time so callers cannot bypass the compiler check.

    Enum values:

      * ``"amp_discriminator"`` — H0 path; clips feed the AMP
        discriminator as the reference distribution.
      * ``"tracking_target"`` — reserved for tracking-style training.
      * ``"hybrid"`` — reserved for mixed-mode training.
    """

    consumption_mode: str                         # amp_discriminator | tracking_target | hybrid
    phase_mode: str = "loop"                      # loop | once | clamp
    motion_fps: float = 50.0
    random_start_phase: bool = True
    # AMP Reference-State Initialization (training_motion node). When
    # ``reference_state_init_enabled``, each episode is seeded from a random
    # expert clip frame (pose + joint state + velocity) with probability
    # ``rsi_prob`` plus ``rsi_joint_noise`` jitter — AMP converges far faster
    # than fixed-pose + uniform-noise resets. BOTH halves of the feature read
    # these SAME canvas fields: the launcher pool population (via
    # isaac_lab/config.py → --unitport_amp_rsi_prob) builds + registers the
    # "default" RSI pool, and env_cfg_compiler emits the
    # reset_from_reference_motion EventTerm (pool_id="default") that consumes
    # it. The spec must transport them so the launcher half is not silently
    # left disabled (pool never registered → EventTerm WARNs + skips RSI).
    reference_state_init_enabled: bool = False
    rsi_prob: float = 0.0
    rsi_joint_noise: float = 0.02
    # task_item_id -> clip_path (resolved from training_items)
    clip_paths: Dict[str, str] = field(default_factory=dict)
    # task_item_id -> (start_frame, end_frame) INCLUSIVE, for items whose clip
    # is a segment (a sub-range of a larger file, encoded canvas-side as
    # ``<ref>#seg=lo-hi``). Absent for whole-file items → the runtime loads the
    # whole clip. Parallel to clip_paths; keyed by the SAME item_id so the
    # positional flattening in isaac_lab/config.py can align files ↔ ranges.
    clip_ranges: Dict[str, Tuple[int, int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.consumption_mode, str) or not self.consumption_mode.strip():
            raise ValueError(
                "MotionRefConfig.consumption_mode is required and must "
                "be a non-empty string (one of 'amp_discriminator' / "
                "'tracking_target' / 'hybrid'). Fix the canvas "
                "training_motion node's consumption_mode ParamSpec — "
                "no silent default."
            )


@dataclass
class DiscriminatorConfig:
    """``discriminator`` — full AMP discriminator config."""

    modules: Dict[str, float] = field(default_factory=lambda: {
        "disc_forward": 1.0,
        "disc_compute_grad_pen": 1.0,
        "disc_predict_reward": 1.0,
    })
    hidden_dims: List[int] = field(default_factory=lambda: [1024, 512])
    amp_reward_coef: float = 2.0
    task_reward_lerp: float = 0.5
    lerp_schedule: str = "none"          # none | slow_anneal | fast_anneal | warmup_then_anneal | custom
    lerp_schedule_json: str = ""
    disc_lr: float = 1e-4
    disc_grad_penalty: float = 10.0
    disc_label_smoothing: float = 0.9
    amp_replay_buffer_size: int = 1_000_000
    # Unified with AMPConfig.num_preload_transitions (was 200_000 — a
    # divergent default that meant the consumer saw a different preload
    # count depending on which dataclass it read. The two are now the
    # same value so the authority single-sourcing (spec_compiler reads the
    # discriminator node into il.amp.*) cannot smuggle in a stale default.
    num_preload_transitions: int = 2_000_000
    amp_obs_fields: str = ""
    auto_inject_ref_obs: bool = True
    # Stability circuit-breakers (P2)
    disc_logit_clamp_max: float = 4.0
    reward_clamp_per_step: float = 50.0
    policy_std_clamp_max: float = 1.5


@dataclass
class AMPConfig:
    """``amp_trainer`` overrides + ``discriminator``.

    Only populated when ``algorithm.training_mode == "AMP_PPO"``.
    """

    amp_reward_coef: float = 2.0
    task_reward_lerp: float = 0.5
    disc_grad_penalty: float = 10.0
    disc_label_smoothing: float = 0.9
    amp_replay_buffer_size: int = 1_000_000
    num_preload_transitions: int = 2_000_000
    disc_lr: float = 1e-4
    lerp_schedule: str = "none"
    lerp_schedule_json: str = ""
    disc: DiscriminatorConfig = field(default_factory=DiscriminatorConfig)


@dataclass
class ImitationLearningConfig:
    """AMP-PPO + future BC/IL+RL fusion config (Stage 9)."""

    amp: AMPConfig = field(default_factory=AMPConfig)
    # motion_ref is Optional because the canvas may not have wired a
    # training_motion node yet; the compiler populates it only when
    # the user has set ``consumption_mode`` explicitly (per the
    # MotionRefConfig.__post_init__ contract). Downstream consumers
    # (validator, IsaacLab config) None-guard.
    motion_ref: Optional[MotionRefConfig] = None
    # BC + IL+RL fusion knobs land in Stage 9; reserved here for forward compat.
    bc_enabled: bool = False
    bc_blend: float = 0.0


# ---------------------------------------------------------------------------
# Domain rand / SB3 env / eval / vis / export
# ---------------------------------------------------------------------------


@dataclass
class DomainRandSchedule:
    mode: str = "none"                   # none | linear
    start_step: int = 0
    end_step: int = 500_000


@dataclass
class DomainRandSb3Config:
    """SB3-side DR knobs.

    Stage H (sim2sim PD framework) — actuator-related DR no longer
    perturbs ``motor_strength`` or ``joint_damping`` at the MJCF level
    (those numbers were a leaky proxy for "PD bandwidth" that diverged
    from the IsaacLab side). Instead, DR perturbs the canonical
    ``(omega_n, zeta)`` parameterization log-uniformly and the env
    re-solves per-joint kp/kd at every reset via
    :func:`application.physics.mujoco_gain_solver.solve`. PhysX-side DR
    applies the same multipliers via an event term emitted by
    :mod:`application.training.isaac_lab.env_cfg_compiler`.
    """

    enabled: bool = True
    mass_range: Tuple[float, float] = (0.8, 1.2)
    friction_range: Tuple[float, float] = (0.5, 1.5)
    # Stage H — replaces motor_strength_range / joint_damping_range.
    # Bounds in linear scale (sampler picks log-uniformly inside).
    # Match the per-step jitter range applied in nodes/domain_rand/node.py.
    omega_n_log_uniform: Tuple[float, float] = (0.8, 1.25)
    zeta_log_uniform: Tuple[float, float] = (0.9, 1.11)
    obs_noise_std: float = 0.01
    push_robot: bool = False
    push_interval_steps: int = 200
    push_force_range: Tuple[float, float] = (50.0, 150.0)
    # Reset-time state randomization (P2 — feature parity with the IsaacLab
    # event DR). init-pose: jitter the base xy position + yaw at episode
    # start; joint-noise: add per-joint position / velocity noise at reset.
    # Disabled by default so legacy SB3 canvases are byte-unchanged.
    init_pose_rand_enabled: bool = False
    init_pos_x_range: Tuple[float, float] = (-0.0, 0.0)
    init_pos_y_range: Tuple[float, float] = (-0.0, 0.0)
    init_yaw_range: Tuple[float, float] = (-0.0, 0.0)
    joint_noise_enabled: bool = False
    joint_pos_noise: float = 0.0
    joint_vel_noise: float = 0.0


@dataclass
class DomainRandIlConfig:
    """Isaac Lab event-driven DR (5 events × {enable, mode, ranges})."""

    # Friction
    friction_enabled: bool = True
    friction_mode: str = "reset"
    static_friction_range: Tuple[float, float] = (0.5, 1.25)
    dynamic_friction_range: Tuple[float, float] = (0.5, 1.25)
    restitution_range: Tuple[float, float] = (0.0, 0.4)
    # Mass
    mass_enabled: bool = True
    mass_mode: str = "reset"
    mass_target_body: str = "base"
    mass_offset_range: Tuple[float, float] = (-0.5, 0.5)
    # External push
    push_enabled: bool = True
    push_mode: str = "interval"
    push_interval_range_s: Tuple[float, float] = (10.0, 15.0)
    push_velocity_x_range: Tuple[float, float] = (-0.5, 0.5)
    push_velocity_y_range: Tuple[float, float] = (-0.5, 0.5)
    # Init pose
    init_pose_enabled: bool = True
    init_pose_mode: str = "reset"
    init_pos_x_range: Tuple[float, float] = (-0.5, 0.5)
    init_pos_y_range: Tuple[float, float] = (-0.5, 0.5)
    init_yaw_range: Tuple[float, float] = (-3.14, 3.14)
    # Joint noise
    joint_noise_enabled: bool = True
    joint_noise_mode: str = "reset"
    joint_pos_noise: float = 0.25
    joint_vel_noise: float = 0.1


@dataclass
class DomainRandConfig:
    """``domain_rand`` — backend-gated."""

    backend: str = "sb3_mujoco"
    schedule: DomainRandSchedule = field(default_factory=DomainRandSchedule)
    sb3: DomainRandSb3Config = field(default_factory=DomainRandSb3Config)
    il: DomainRandIlConfig = field(default_factory=DomainRandIlConfig)


@dataclass
class EnvAssemblerConfig:
    """``env_assembler`` — SB3 VecEnv + wrapper stack."""

    # Hardware-adaptive vectorisation. "auto" derives (n_envs, vec_type) from
    # the training machine's CPU cores + RAM at launch (see
    # application.training.envs.auto_parallelism); "manual" uses the n_envs /
    # vec_type below verbatim. Resolved at runtime on the training box (so
    # cloud workers adapt to their own hardware), never at compile time.
    parallelism_mode: str = "auto"       # auto | manual
    n_envs: int = 8                      # manual override (ignored when auto)
    vec_type: str = "subproc"            # dummy | subproc (ignored when auto)
    obs_normalize: bool = True
    reward_normalize: bool = False
    clip_obs: float = 10.0
    clip_reward: float = 10.0
    action_clip_range: float = 1.0
    enable_monitor: bool = True
    time_limit_override: int = 0
    eval_disable_rand: bool = True


@dataclass
class EvalConfig:
    eval_episodes: int = 20
    deterministic: bool = True
    success_threshold: float = 0.8
    record_video: bool = False
    eval_interval: int = 50_000
    save_best_model: bool = True
    video_dir: str = ""


@dataclass
class VisCheckConfig:
    trigger_mode: str = "step_interval"  # step_interval | episode_split
    vis_start_step: int = 50_000
    vis_interval_steps: int = 50_000
    num_vis_checks: int = 5
    vis_episodes: int = 3
    deterministic: bool = True


@dataclass
class ExportConfig:
    """``export`` — bundle target + review backend."""

    bundle_name: str = "<NEW>"
    version: str = "v1"
    include_onnx: bool = True
    include_torchscript: bool = True
    include_normalization: bool = True
    bundle_targets: List[str] = field(default_factory=lambda: ["runtime_bundle"])
    overwrite: bool = True
    review_backend: str = "mujoco"       # mujoco | isaac_sim | newton
    review_scene_id: str = "flat_ground"
    auto_import: bool = True


# ---------------------------------------------------------------------------
# Robot adapter (mirrors registers.robots.RobotSpec — copied at compile time)
# ---------------------------------------------------------------------------


@dataclass
class RobotSpecRef:
    """Compile-time snapshot of a ``registers.robots.RobotSpec``.

    Lowering copies the registry RobotSpec into the TrainingSpec so the
    submitted spec is self-contained (subprocess launchers in Stage 10
    receive it as JSON without re-reading the registry).

    Per-format schema (Stage 2 upgrade):
        - ``active_format`` is the format the Robot node resolved at
          compile time (mjcf / usd / urdf). Drives every body / joint
          lookup downstream — env_cfg_compiler, JointIRResolver,
          BodyIRMapper, RSI, AMP storage.
        - ``joints_per_format`` / ``bodies_per_format`` mirror the
          registry's per-format dicts so downstream subprocess launchers
          can re-derive the lookup tables without re-reading the
          registry.
        - Legacy ``joint_order`` / ``joint_ir_roles`` / ``body_role_map``
          are kept as derived views of the active format for transitional
          callers; new code should use the format-aware accessors.
    """

    sku: str = ""
    name: str = ""
    brand: str = ""                      # mirrors RobotSpec.brand (registers/robots.py)
    model: str = ""                      # mirrors RobotSpec.model (registers/robots.py)
    families: List[str] = field(default_factory=list)
    # Per-format raw tables (Stage 2): mirror registers.robots.RobotSpec.
    joints_per_format: Dict[str, Dict[str, Dict[str, Any]]] = field(default_factory=dict)
    bodies_per_format: Dict[str, Dict[str, Dict[str, Any]]] = field(default_factory=dict)
    # The active asset format ("MJCF" / "USD" / "URDF") chosen at the
    # Robot node by user override + backend preference. Used by IR
    # resolvers to pick the matching per-format table.
    active_format: str = ""
    # Legacy flat views — populated from the active format at compile
    # time so legacy consumers keep working through the staged rollout.
    joint_order: List[str] = field(default_factory=list)
    joint_ir_roles: List[str] = field(default_factory=list)
    body_role_map: Dict[str, str] = field(default_factory=dict)
    # SKU-recommended actuator defaults (PD gains / effort / vel limits).
    # Mirrors ``registers.robots.RobotSpec.default_actuator_params``.
    # Canvas ActorSetting reconciles its stiffness/damping/effort_limit/
    # velocity_limit fields from this dict whenever the upstream Robot
    # Node changes SKU — eliminates the Go2-style hardcoded canvas
    # defaults that made Spot (42 kg) train under Go2-class PD (15 kg)
    # and learn a ragdoll policy. None means "registry declared none —
    # canvas keeps its own ParamSpec defaults".
    default_actuator_params: Optional[Dict[str, float]] = None
    mjcf_path: Optional[str] = None
    urdf_path: Optional[str] = None
    usd_path: Optional[str] = None
    capabilities: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_joints(self) -> int:
        """Mirror of :attr:`registers.robots.RobotSpec.num_joints` so callers
        that expect a duck-typed RobotSpec (motion validator, action-joint
        check) work uniformly against the spec snapshot."""
        return len(self.joint_order)

    # --- per-format accessors -----------------------------------------------

    def joint_order_for(self, fmt: str) -> List[str]:
        block = self.joints_per_format.get(str(fmt).upper(), {}) or {}
        return [str(j.get("name", "")) for j in block.values()
                if isinstance(j, dict) and j.get("name")]

    def joint_ir_roles_for(self, fmt: str) -> List[str]:
        block = self.joints_per_format.get(str(fmt).upper(), {}) or {}
        out: List[str] = []
        for j in block.values():
            if not isinstance(j, dict):
                continue
            nm = str(j.get("name", ""))
            if nm:
                out.append(str(j.get("ir_role", "")))
        return out

    def joints_role_map_for(self, fmt: str) -> Dict[str, str]:
        """``{joint_name: ir_role}`` for the format (mirrors RobotSpec)."""
        block = self.joints_per_format.get(str(fmt).upper(), {}) or {}
        out: Dict[str, str] = {}
        for j in block.values():
            if not isinstance(j, dict):
                continue
            nm = str(j.get("name", "")).strip()
            rl = str(j.get("ir_role", "")).strip()
            if nm and rl:
                out[nm] = rl
        return out

    def bodies_role_map_for(self, fmt: str) -> Dict[str, str]:
        """``{body_name: ir_role}`` for the format — the actual body mapping."""
        block = self.bodies_per_format.get(str(fmt).upper(), {}) or {}
        out: Dict[str, str] = {}
        for b in block.values():
            if not isinstance(b, dict):
                continue
            nm = str(b.get("name", "")).strip()
            rl = str(b.get("ir_role", "")).strip()
            if nm and rl:
                out[nm] = rl
        return out

    @classmethod
    def from_registry(
        cls, rs: Any, active_format: str = "",
    ) -> "RobotSpecRef":
        # active_format defaults to the registry RobotSpec's preferred_format
        # so legacy callers that don't yet wire it through (Stage 2 -> 3
        # transition) still get a working spec.
        fmt = str(active_format or "").strip().upper()
        if not fmt:
            fmt = getattr(rs, "preferred_format", "MJCF")
        # Derive legacy flat views from the chosen active format so existing
        # consumers of ``joint_order`` / ``joint_ir_roles`` / ``body_role_map``
        # see the format they intend to train against, not whatever the
        # registry happens to prefer.
        # Actuated-joint filter: drop non-motion entries (parked
        # '__out_of_scope__' like an MJCF freejoint, and 'misc'/'sensor*'/'base'
        # buckets) so the flat views feed training / sim2sim / bundle export the
        # joints the policy actually drives. Without this, a leaked freejoint
        # inflates num_joints / action_dim and corrupts joint_sdk_names (the
        # re-trim in bundle_exporter would keep '__out_of_scope__' and drop a
        # real joint). One mask keyed on ir_role keeps the two arrays parallel.
        from registers.robots import is_actuated_ir_role
        _raw_order = list(rs.joint_order_for(fmt))
        _raw_roles = list(rs.joint_ir_roles_for(fmt))
        _keep = [is_actuated_ir_role(r) for r in _raw_roles]
        joint_order_fmt = [n for n, k in zip(_raw_order, _keep) if k]
        joint_ir_roles_fmt = [r for r, k in zip(_raw_roles, _keep) if k]
        joints_role_map_fmt = {
            n: r for n, r in rs.joints_role_map_for(fmt).items()
            if is_actuated_ir_role(r)
        }
        # CLAUDE.md §1.8: previously this fell back to ``rs.joint_order`` /
        # ``rs.joint_ir_roles`` / ``rs.body_role_map`` (which resolve through
        # ``preferred_format``) whenever the requested ``fmt`` had no joint
        # table. For Isaac Lab (active_format="USD") that fallback silently
        # served MJCF joints whenever joints_per_format["USD"] was null —
        # training proceeded with MJCF joint names while Isaac Lab actually
        # spawned the USD asset and reported its own articulation joint
        # order to the policy. The bundle then shipped MJCF-ordered joint
        # names, the policy outputs/observations slotted through the wrong
        # joints at deploy time, and the user saw "electric-shock twitch"
        # behaviour. Refuse to substitute formats here. (Guard on the RAW
        # table, not the actuated filter, so this fires only for a truly
        # un-dumped format.)
        if not _raw_order:
            raise ValueError(
                f"[RobotSpecRef.from_registry] robot sku={rs.sku!r} declares "
                f"no joints under joints_per_format[{fmt!r}] — the format "
                f"the caller asked for. Silent fallback to another format "
                f"is forbidden (CLAUDE.md §1.8): joint orders differ across "
                f"MJCF / USD / URDF, and serving the wrong one ships a "
                f"bundle whose joint_sdk_names don't match the order the "
                f"trained policy expects (the \"twitching policy\" failure "
                f"mode at deploy). Populate the table for format {fmt!r} "
                f"first — for USD, open the Robot Asset card and run "
                f"\"Dump USD\"; for MJCF, run \"Dump MJCF\". Available "
                f"formats for this robot: "
                f"{sorted(getattr(rs, 'available_formats', []) or [])}."
            )
        if not joint_order_fmt:
            raise ValueError(
                f"[RobotSpecRef.from_registry] robot sku={rs.sku!r} declares "
                f"{len(_raw_order)} joint(s) for format {fmt!r} but NONE are "
                f"actuated motion joints — every entry is parked "
                f"('__out_of_scope__') or a bucket ('misc' / 'sensor*'). A "
                f"policy needs at least one actuated joint. Re-open the joint "
                f"mapping for this robot and assign real IR roles."
            )
        dap_raw = getattr(rs, "default_actuator_params", None)
        dap: Optional[Dict[str, float]] = (
            {str(k): float(v) for k, v in dap_raw.items()}
            if isinstance(dap_raw, dict) and dap_raw
            else None
        )
        return cls(
            sku=rs.sku,
            name=rs.name,
            brand=getattr(rs, "brand", "") or "",
            model=getattr(rs, "model", "") or "",
            families=list(rs.families),
            joints_per_format=dict(rs.joints_per_format),
            bodies_per_format=dict(rs.bodies_per_format),
            active_format=fmt,
            joint_order=joint_order_fmt,
            joint_ir_roles=joint_ir_roles_fmt,
            body_role_map=joints_role_map_fmt,   # legacy contract = joint→ir_role
            default_actuator_params=dap,
            mjcf_path=rs.mjcf_path,
            urdf_path=rs.urdf_path,
            usd_path=rs.usd_path,
            capabilities=dict(rs.capabilities),
        )


# ---------------------------------------------------------------------------
# TrainingSpec root
# ---------------------------------------------------------------------------


@dataclass
class TrainingSpec:
    """Canonical canvas → backend training contract.

    See :doc:`MIGRATION_MAP.md` for the per-node field provenance.
    """

    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    robot: RobotSpecRef = field(default_factory=RobotSpecRef)
    actor: ActorConfig = field(default_factory=ActorConfig)
    obs_action: ObsActionContract = field(default_factory=ObsActionContract)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    scene: SceneConfig = field(default_factory=SceneConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    rewards: RewardConfig = field(default_factory=RewardConfig)
    terminations: TerminationConfig = field(default_factory=TerminationConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    il: ImitationLearningConfig = field(default_factory=ImitationLearningConfig)
    domain_rand: DomainRandConfig = field(default_factory=DomainRandConfig)
    stage_schedule: StageScheduleConfig = field(default_factory=StageScheduleConfig)
    env: EnvAssemblerConfig = field(default_factory=EnvAssemblerConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    vis: VisCheckConfig = field(default_factory=VisCheckConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    # Provenance: free-form metadata lowering may attach (run_id, source canvas
    # path, compile timestamp, …) — backends pass-through on serialization.
    meta: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Convert to plain dict for JSON / IPC serialization.

        ``dataclasses.asdict`` recurses through nested dataclasses; tuples
        are converted to lists by JSON anyway.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "TrainingSpec":
        """Inverse of :meth:`to_dict`. Unknown keys raise ``ValueError``."""
        return _spec_from_dict(cls, payload)

    @classmethod
    def minimal(cls) -> "TrainingSpec":
        """Default-construct an instance — useful for tests / smoke compile."""
        return cls()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _spec_from_dict(target_cls, payload):
    """Strict dataclass reconstruction from a dict.

    Walks the type hints; for each field that is itself a dataclass, recurses.
    Unknown keys raise ``ValueError`` — strict-mode contract: the on-disk
    spec must be a 1:1 reflection of the current schema. Forward/backward
    compatibility goes through explicit schema versioning + a migrator, not
    a silent drop.
    """
    import dataclasses
    import typing

    if not dataclasses.is_dataclass(target_cls) or not isinstance(payload, dict):
        return payload

    known = {f.name for f in dataclasses.fields(target_cls)}
    unknown = sorted(set(payload) - known)
    if unknown:
        raise ValueError(
            f"_spec_from_dict({target_cls.__name__}): unknown key(s) "
            f"{unknown!r} — schema mismatch. Migrate the spec source or "
            f"update the dataclass."
        )

    hints = typing.get_type_hints(target_cls)
    kwargs: Dict[str, Any] = {}
    for f in dataclasses.fields(target_cls):
        if f.name not in payload:
            continue
        raw = payload[f.name]
        # ``ActorConfig.pd_param`` is typed ``Any`` (to avoid coupling the
        # spec to the physics layer), so the generic dataclass walker below
        # would leave it as a raw dict — but every consumer (mujoco_gain_solver,
        # bundle_exporter) needs a real ``PDParam`` (``.groups`` etc.). Without
        # this, an IPC round-trip (main app → SB3 subprocess) silently degrades
        # to scalar PD and the bundle export later crashes. Reconstruct it.
        if f.name == "pd_param" and isinstance(raw, dict) and raw:
            from application.physics.pd_param import PDParam
            kwargs[f.name] = PDParam.from_dict(raw)
            continue
        ftype = hints.get(f.name, type(raw))
        # Unwrap Optional[X] / Union[X, None]
        origin = typing.get_origin(ftype)
        args = typing.get_args(ftype)
        if origin is typing.Union and type(None) in args:
            inner = next((a for a in args if a is not type(None)), None)
            if inner is not None:
                ftype = inner
                origin = typing.get_origin(ftype)
                args = typing.get_args(ftype)
        if dataclasses.is_dataclass(ftype) and isinstance(raw, dict):
            kwargs[f.name] = _spec_from_dict(ftype, raw)
        elif origin is tuple and isinstance(raw, (list, tuple)):
            kwargs[f.name] = tuple(raw)
        elif (
            origin is list
            and args
            and dataclasses.is_dataclass(args[0])
            and isinstance(raw, list)
        ):
            inner_cls = args[0]
            kwargs[f.name] = [
                _spec_from_dict(inner_cls, item) if isinstance(item, dict) else item
                for item in raw
            ]
        elif (
            origin is dict
            and len(args) == 2
            and dataclasses.is_dataclass(args[1])
            and isinstance(raw, dict)
        ):
            # ``Dict[str, <dataclass>]`` (e.g. ``MotionConfig.packages``) —
            # the generic walker above only recurses plain-dataclass and
            # list-of-dataclass fields, so without this a dict-of-dataclass
            # would silently round-trip back as raw dicts. Reconstruct values.
            val_cls = args[1]
            kwargs[f.name] = {
                k: _spec_from_dict(val_cls, v) if isinstance(v, dict) else v
                for k, v in raw.items()
            }
        else:
            kwargs[f.name] = raw
    return target_cls(**kwargs)


# ---------------------------------------------------------------------------
# Training package resolution (Method A, Slice 1)
# ---------------------------------------------------------------------------


def _pkg_item_enabled(v: Any) -> bool:
    """Item ``enabled`` flag; a missing key defaults True (aligns with the SB3
    sampler / command-contract convention that an item without ``enabled`` is on)."""
    return bool(v.get("enabled", True)) if isinstance(v, dict) else True


def _pkg_item_package_id(v: Any) -> str:
    """The item's authored ``package_id`` membership (empty string if absent)."""
    return str(v.get("package_id", "")).strip() if isinstance(v, dict) else ""


def resolve_effective_packages(motion: "MotionConfig") -> Dict[str, "TrainingPackage"]:
    """The effective package map for a ``MotionConfig`` (Method A, Slice 1).

    SINGLE source of truth for "what packages exist" — every consumer (validators,
    compile-time lowering, UI preview) calls this and never re-derives membership.

    Presence-gated: when no package layer is authored (empty ``packages`` AND no
    training item carries a ``package_id``) one implicit default package is
    synthesized at ``package_weight=1.0`` wrapping every enabled item — this is
    byte-identical to pre-package behavior. Otherwise the authored packages are
    returned after fail-loud membership validation (§8): every enabled item must
    resolve to exactly one enabled package, and every enabled package must own at
    least one enabled member item. Never defaults membership silently.
    """
    items = motion.training_items or {}
    authored = dict(motion.packages or {})
    any_membership = any(_pkg_item_package_id(v) for v in items.values())

    if not authored and not any_membership:
        # Implicit default: one package wrapping all enabled items.
        return {
            DEFAULT_PACKAGE_ID: TrainingPackage(
                package_id=DEFAULT_PACKAGE_ID,
                name="",
                enabled=True,
                package_weight=1.0,
                reward_terms={},
            )
        }

    if DEFAULT_PACKAGE_ID in authored:
        raise ValueError(
            f"package id {DEFAULT_PACKAGE_ID!r} is reserved for the implicit "
            f"default package and may not be authored explicitly (§8 fail-loud)."
        )

    enabled_pkgs = {pid for pid, p in authored.items() if getattr(p, "enabled", True)}
    for item_id, v in items.items():
        if not _pkg_item_enabled(v):
            continue
        pid = _pkg_item_package_id(v)
        if not pid:
            raise ValueError(
                f"training item {item_id!r} has no package_id but a package layer "
                f"is authored — every enabled item must name exactly one enabled "
                f"package (§8 fail-loud)."
            )
        if pid not in authored:
            raise ValueError(
                f"training item {item_id!r} names package {pid!r} which does not "
                f"exist (known: {sorted(authored)}) (§8 fail-loud)."
            )
        if pid not in enabled_pkgs:
            raise ValueError(
                f"training item {item_id!r} names package {pid!r} which is "
                f"disabled — enable the package or move the item (§8 fail-loud)."
            )

    member_counts: Dict[str, int] = {pid: 0 for pid in authored}
    for v in items.values():
        if _pkg_item_enabled(v):
            pid = _pkg_item_package_id(v)
            if pid in member_counts:
                member_counts[pid] += 1
    for pid, p in authored.items():
        if getattr(p, "enabled", True) and member_counts.get(pid, 0) == 0:
            raise ValueError(
                f"package {pid!r} is enabled but has no enabled member items — a "
                f"package with zero members is invalid; add items or disable it "
                f"(§8 fail-loud)."
            )
    return authored


def package_members(motion: "MotionConfig", package_id: str) -> List[str]:
    """The enabled member item ids of ``package_id`` (implicit-default aware).

    Raises if ``package_id`` is not one of the effective packages (§8 fail-loud).
    """
    packages = resolve_effective_packages(motion)
    if package_id not in packages:
        raise ValueError(
            f"package {package_id!r} not in effective packages {sorted(packages)} "
            f"(§8 fail-loud)."
        )
    items = motion.training_items or {}
    if set(packages) == {DEFAULT_PACKAGE_ID}:
        # Implicit default owns every enabled item (members carry no package_id).
        return [iid for iid, v in items.items() if _pkg_item_enabled(v)]
    return [
        iid
        for iid, v in items.items()
        if _pkg_item_enabled(v) and _pkg_item_package_id(v) == package_id
    ]


__all__ = [
    "TrainingSpec",
    "AlgorithmConfig",
    "IlPpoConfig",
    "PolicyNetConfig",
    "CheckpointConfig",
    "RobotSpecRef",
    "ActorConfig",
    "ContactSensorConfig",
    "ActuatorConfig",
    "ActionScaleCurriculum",
    "InitPoseConfig",
    "ObsActionContract",
    "PhysicsConfig",
    "SceneConfig",
    "RoughTerrainConfig",
    "CustomTerrainConfig",
    "HeightScanConfig",
    "TaskConfig",
    "RewardConfig",
    "TerminationConfig",
    "StageScheduleConfig",
    "StageEntry",
    "DistillLossSpec",
    "STAGE_SCHEDULE_SCHEMA_VERSION",
    "validate_stage_entry_h0",
    "validate_stage_schedule_dict_h0",
    "MotionConfig",
    "TrainingPackage",
    "DEFAULT_PACKAGE_ID",
    "resolve_effective_packages",
    "package_members",
    "MotionRefConfig",
    "GaitConfig",
    "CommandCurriculumConfig",
    "ImitationLearningConfig",
    "AMPConfig",
    "DiscriminatorConfig",
    "DomainRandConfig",
    "DomainRandSchedule",
    "DomainRandSb3Config",
    "DomainRandIlConfig",
    "EnvAssemblerConfig",
    "EvalConfig",
    "VisCheckConfig",
    "ExportConfig",
]
