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
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Algorithm / trainer
# ---------------------------------------------------------------------------


@dataclass
class PolicyNetConfig:
    """``il_policy_network`` actor/critic MLP architecture."""

    actor_hidden_dims: List[int] = field(default_factory=lambda: [128, 64, 32])
    critic_hidden_dims: List[int] = field(default_factory=lambda: [128, 64, 32])
    activation: str = "elu"
    init_noise_std: float = -1.0  # -1 = auto


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
    # Shared (sb3 path; il path uses IlPpoConfig.max_iterations etc.)
    total_timesteps: int = 1_000_000
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
    """``actor_setting`` §4 (legacy) / ``actuator_pd`` (canonical) — PD knobs.

    ``stiffness``/``damping`` are the legacy scalar fallback used when
    no ``actuator_pd`` node is wired. The canonical PD source is
    :attr:`ActorConfig.pd_param` (a :class:`PDParam` built from
    family defaults + canvas overrides via the ``actuator_pd`` node).
    Engine compilers and runtime envs MUST prefer ``pd_param`` over
    ``stiffness``/``damping`` whenever it is populated; the scalar path
    is a single-release back-compat bridge that will be removed once
    every saved canvas has been migrated.
    """

    stiffness: float = 25.0
    damping: float = 0.5
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
    # Populated by ``spec_compiler._compile_pd_param`` from the canvas
    # ``actuator_pd`` node + family defaults. ``None`` means the canvas
    # has no actuator_pd node — engine compilers fall back to the
    # ``actuator.stiffness/damping`` scalar path with a WARN. Once every
    # saved canvas has been migrated, the legacy path will be removed
    # and this field will become non-Optional.
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
    # IL-style (il_observation)
    il_terms: Dict[str, Any] = field(default_factory=dict)
    group_name: str = "policy"
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
    amplitude: float = 0.08
    slope_max_deg: float = 20.0
    curriculum_enabled: bool = False
    difficulty_levels: int = 10


@dataclass
class SceneConfig:
    """``play_ground_setting`` — full §2 Scene block."""

    scene_id: str = "flat_ground"
    scene_type: str = "flat"             # flat | rough
    gravity_z: float = -9.81
    arena_extent_x: float = 10.0
    arena_extent_y: float = 10.0
    friction_static: float = 1.0
    friction_dynamic: float = 0.8
    rough: RoughTerrainConfig = field(default_factory=RoughTerrainConfig)
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

    backend: str = "sb3"                 # "sb3" | "isaac_lab"
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

    backend: str = "sb3"
    conditions: Dict[str, Any] = field(default_factory=dict)
    curriculum_enabled: bool = False
    curriculum_start: float = 0.18
    curriculum_end: float = 0.22
    curriculum_ramp_iters: int = 500
    # See :class:`RewardConfig.inline_source_overrides`.
    inline_source_overrides: Dict[str, str] = field(default_factory=dict)
    # See :class:`RewardConfig.il_params_overrides`.
    il_params_overrides: Dict[str, str] = field(default_factory=dict)


@dataclass
class StageScheduleConfig:
    """``stage_switch`` + ``multigated_reward`` curriculum gate."""

    stages: List[Dict[str, Any]] = field(default_factory=list)
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


# ---------------------------------------------------------------------------
# Motion / IL / AMP
# ---------------------------------------------------------------------------


@dataclass
class GaitConfig:
    """``training_motion`` §3 Walk-These-Ways."""

    enabled: bool = False
    frequency_range_hz: Tuple[float, float] = (1.5, 3.5)
    body_height_range_m: Tuple[float, float] = (0.28, 0.40)
    step_height_range_m: Tuple[float, float] = (0.03, 0.15)
    presets: str = ""


@dataclass
class CommandCurriculumConfig:
    enabled: bool = False
    start: float = 0.25
    end: float = 1.0
    ramp_iters: int = 800


@dataclass
class MotionConfig:
    """``training_motion`` — command envelope only.

    Reference-clip slice goes to ``ImitationLearningConfig.motion_ref``.
    """

    training_items: Dict[str, Any] = field(default_factory=dict)
    mapping_mode: str = "linear"
    deadzone: float = 0.10
    curve_exponent: float = 2.0
    runtime_clip: bool = True
    gait: GaitConfig = field(default_factory=GaitConfig)
    resampling_time_range: Tuple[float, float] = (4.0, 12.0)
    zero_command_probability: float = 0.1
    cmd_step_change_prob: float = 0.01
    command_curriculum: CommandCurriculumConfig = field(default_factory=CommandCurriculumConfig)


@dataclass
class MotionRefConfig:
    """``training_motion`` reference-clip slice consumed by AMP."""

    consumer_mode: str = "amp"           # tracking | amp | both
    phase_mode: str = "loop"             # loop | once | clamp
    motion_fps: float = 50.0
    random_start_phase: bool = True
    # task_item_id -> clip_path (resolved from training_items)
    clip_paths: Dict[str, str] = field(default_factory=dict)


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
    num_preload_transitions: int = 200_000
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
    motion_ref: MotionRefConfig = field(default_factory=MotionRefConfig)
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
    # Defaults match knowledge_base/sim2sim_mass-matrix-adaptive.yaml
    # lines 103-105.
    omega_n_log_uniform: Tuple[float, float] = (0.8, 1.25)
    zeta_log_uniform: Tuple[float, float] = (0.9, 1.11)
    obs_noise_std: float = 0.01
    push_robot: bool = False
    push_interval_steps: int = 200
    push_force_range: Tuple[float, float] = (50.0, 150.0)


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

    backend: str = "sb3"
    schedule: DomainRandSchedule = field(default_factory=DomainRandSchedule)
    sb3: DomainRandSb3Config = field(default_factory=DomainRandSb3Config)
    il: DomainRandIlConfig = field(default_factory=DomainRandIlConfig)


@dataclass
class EnvAssemblerConfig:
    """``env_assembler`` — SB3 VecEnv + wrapper stack."""

    n_envs: int = 8
    vec_type: str = "subproc"            # dummy | subproc
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
    # Isaac Lab's USD-articulation joint loading order (IR roles, by-type
    # grouped — hip × n, thigh × n, calf × n for quadrupeds). Mirrors
    # ``registers.robots.RobotSpec.isaac_lab_joint_order``. Used by
    # ``bundle_finalizer`` to compute the permutation that maps the
    # trained policy's action vector (in USD-articulation order) onto
    # the bundle's joint_sdk_names (in SDK/canonical order) so sim2sim
    # deployment doesn't twitch. None when the registry entry doesn't
    # declare it — finalizer falls back with a WARN.
    isaac_lab_joint_order: Optional[List[str]] = None
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
    target_height: float = 0.0           # robot.target_height override

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
        cls, rs: Any, target_height: float = 0.0, active_format: str = "",
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
        joint_order_fmt = list(rs.joint_order_for(fmt))
        joint_ir_roles_fmt = list(rs.joint_ir_roles_for(fmt))
        joints_role_map_fmt = dict(rs.joints_role_map_for(fmt))
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
        # behaviour. Refuse to substitute formats here.
        if not joint_order_fmt:
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
        il_order_raw = getattr(rs, "isaac_lab_joint_order", None)
        il_order: Optional[List[str]] = (
            [str(x) for x in il_order_raw] if il_order_raw else None
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
            isaac_lab_joint_order=il_order,
            default_actuator_params=dap,
            mjcf_path=rs.mjcf_path,
            urdf_path=rs.urdf_path,
            usd_path=rs.usd_path,
            capabilities=dict(rs.capabilities),
            target_height=float(target_height),
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
        else:
            kwargs[f.name] = raw
    return target_cls(**kwargs)


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
    "HeightScanConfig",
    "TaskConfig",
    "RewardConfig",
    "TerminationConfig",
    "StageScheduleConfig",
    "MotionConfig",
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
