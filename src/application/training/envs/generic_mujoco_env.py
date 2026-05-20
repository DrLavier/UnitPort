"""Generic MuJoCo + Gymnasium env — brand-neutral.

REWRITE-WITH-DEMO-REF from the DEMO brand-coupled gym env (the ~2.8k-line
file targeted for removal — see the red-line list in ``DEMO/CLAUDE.md``),
narrowed to the Stage 6 acceptance contract:

    env = GenericMujocoEnv(robot_spec, scene_config)
    env.reset()
    env.step(env.action_space.sample())

Stage 6 v1 scope — minimal but functional:
    * Loads MJCF from ``RobotSpec.mjcf_path``; falls back to a brand-
      neutral 12-DoF quadruped MJCF (``DEFAULT_QUADRUPED_MJCF``) when
      the spec doesn't carry an asset path.
    * Joint-position PD control via ``CtrlPort -> qpos target`` mapped
      through ``RobotSpec.joint_order``.
    * Proprio observation: ``[base_ang_vel(3), projected_gravity(3),
      joint_pos(N), joint_vel(N), last_action(N), commands(3)]``.
    * Reward / termination are caller-supplied callables; defaults give
      a basic survive + lin_vel-tracking reward and a base-z floor
      termination so the env runs end-to-end without canvas wiring.
    * No domain randomization / height scan / contact history (Stage 6
      v2 will layer those on top once Stage 12 needs them).

**Multi-Brand Inclusiveness red line**: this file may not contain any
brand string. The Stage 6 acceptance grep (regex over the brand list)
must return zero matches across this subtree.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import gymnasium as _gym
    _GymEnvBase = _gym.Env
    _GYM_AVAILABLE = True
except ImportError:  # pragma: no cover
    # Allow import-time inspection of constants (DEFAULT_QUADRUPED_MJCF)
    # without gymnasium installed; instantiation still raises.
    _gym = None
    _GymEnvBase = object
    _GYM_AVAILABLE = False

from unitport_sdk import log_info, log_warning


# ---------------------------------------------------------------------------
# Default MJCF — brand-neutral 12-DoF quadruped (used as fallback when
# RobotSpec.mjcf_path is None). Body/joint names follow the canonical IR
# layout (FL/FR/RL/RR × hip/thigh/calf), not any brand naming.
# ---------------------------------------------------------------------------

DEFAULT_QUADRUPED_MJCF = """\
<mujoco model="canonical_quadruped">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <default>
    <joint damping="0.1" armature="0.01"/>
    <geom contype="1" conaffinity="1" friction="1 0.005 0.0001"/>
  </default>
  <worldbody>
    <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="0 0 1" rgba=".9 .9 .9 1"/>
    <body name="base" pos="0 0 0.35">
      <freejoint name="root"/>
      <geom name="base_geom" type="box" size="0.12 0.06 0.05" mass="6.0"
            rgba=".5 .5 .8 1"/>
      <body name="FL_hip" pos="0.17 0.085 0">
        <joint name="FL_hip" type="hinge" axis="1 0 0" range="-0.8 0.8"/>
        <geom type="capsule" size="0.03" fromto="0 0 0 0.07 0 0" mass="0.6"/>
        <body name="FL_thigh_link" pos="0.07 0 0">
          <joint name="FL_thigh" type="hinge" axis="0 1 0" range="-1.5 3.4"/>
          <geom type="capsule" size="0.03" fromto="0 0 0 0 0 -0.21" mass="0.6"/>
          <body name="FL_calf_link" pos="0 0 -0.21">
            <joint name="FL_calf" type="hinge" axis="0 1 0" range="-2.8 -0.8"/>
            <geom type="capsule" size="0.025" fromto="0 0 0 0 0 -0.19" mass="0.3"/>
          </body>
        </body>
      </body>
      <body name="FR_hip" pos="0.17 -0.085 0">
        <joint name="FR_hip" type="hinge" axis="1 0 0" range="-0.8 0.8"/>
        <geom type="capsule" size="0.03" fromto="0 0 0 0.07 0 0" mass="0.6"/>
        <body name="FR_thigh_link" pos="0.07 0 0">
          <joint name="FR_thigh" type="hinge" axis="0 1 0" range="-1.5 3.4"/>
          <geom type="capsule" size="0.03" fromto="0 0 0 0 0 -0.21" mass="0.6"/>
          <body name="FR_calf_link" pos="0 0 -0.21">
            <joint name="FR_calf" type="hinge" axis="0 1 0" range="-2.8 -0.8"/>
            <geom type="capsule" size="0.025" fromto="0 0 0 0 0 -0.19" mass="0.3"/>
          </body>
        </body>
      </body>
      <body name="RL_hip" pos="-0.17 0.085 0">
        <joint name="RL_hip" type="hinge" axis="1 0 0" range="-0.8 0.8"/>
        <geom type="capsule" size="0.03" fromto="0 0 0 0.07 0 0" mass="0.6"/>
        <body name="RL_thigh_link" pos="0.07 0 0">
          <joint name="RL_thigh" type="hinge" axis="0 1 0" range="-1.5 3.4"/>
          <geom type="capsule" size="0.03" fromto="0 0 0 0 0 -0.21" mass="0.6"/>
          <body name="RL_calf_link" pos="0 0 -0.21">
            <joint name="RL_calf" type="hinge" axis="0 1 0" range="-2.8 -0.8"/>
            <geom type="capsule" size="0.025" fromto="0 0 0 0 0 -0.19" mass="0.3"/>
          </body>
        </body>
      </body>
      <body name="RR_hip" pos="-0.17 -0.085 0">
        <joint name="RR_hip" type="hinge" axis="1 0 0" range="-0.8 0.8"/>
        <geom type="capsule" size="0.03" fromto="0 0 0 0.07 0 0" mass="0.6"/>
        <body name="RR_thigh_link" pos="0.07 0 0">
          <joint name="RR_thigh" type="hinge" axis="0 1 0" range="-1.5 3.4"/>
          <geom type="capsule" size="0.03" fromto="0 0 0 0 0 -0.21" mass="0.6"/>
          <body name="RR_calf_link" pos="0 0 -0.21">
            <joint name="RR_calf" type="hinge" axis="0 1 0" range="-2.8 -0.8"/>
            <geom type="capsule" size="0.025" fromto="0 0 0 0 0 -0.19" mass="0.3"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="FL_hip"   joint="FL_hip"   gear="33.5"/>
    <motor name="FL_thigh" joint="FL_thigh" gear="33.5"/>
    <motor name="FL_calf"  joint="FL_calf"  gear="45.0"/>
    <motor name="FR_hip"   joint="FR_hip"   gear="33.5"/>
    <motor name="FR_thigh" joint="FR_thigh" gear="33.5"/>
    <motor name="FR_calf"  joint="FR_calf"  gear="45.0"/>
    <motor name="RL_hip"   joint="RL_hip"   gear="33.5"/>
    <motor name="RL_thigh" joint="RL_thigh" gear="33.5"/>
    <motor name="RL_calf"  joint="RL_calf"  gear="45.0"/>
    <motor name="RR_hip"   joint="RR_hip"   gear="33.5"/>
    <motor name="RR_thigh" joint="RR_thigh" gear="33.5"/>
    <motor name="RR_calf"  joint="RR_calf"  gear="45.0"/>
  </actuator>
</mujoco>
"""


# ---------------------------------------------------------------------------
# Reward / termination defaults
# ---------------------------------------------------------------------------


def _default_reward(env: "GenericMujocoEnv") -> float:
    """Survive + lin_vel-tracking + action-rate penalty."""
    cmd = env._command  # [vx, vy, wz]
    vx = float(env._lin_vel[0])
    vy = float(env._lin_vel[1])
    wz = float(env._ang_vel[2])

    # Track xy command (Gaussian on error norm)
    err_xy = (vx - cmd[0]) ** 2 + (vy - cmd[1]) ** 2
    track_lin = float(np.exp(-err_xy / max(env._track_sigma, 1e-3)))

    # Track yaw rate
    err_yaw = (wz - cmd[2]) ** 2
    track_ang = float(np.exp(-err_yaw / max(env._track_sigma, 1e-3)))

    # Action-rate penalty
    rate = float(np.sum(np.square(env._action - env._last_action)))
    survive = 1.0
    return survive + 0.5 * track_lin + 0.25 * track_ang - 0.01 * rate


def _default_termination(env: "GenericMujocoEnv") -> bool:
    """Base z below threshold OR non-finite state."""
    if not np.all(np.isfinite(env._qpos)) or not np.all(np.isfinite(env._qvel)):
        return True
    base_z = float(env._base_pos[2])
    return base_z < env._term_base_z


# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------


@dataclass
class _DRState:
    """Resolved SB3-side domain randomization knobs.

    Mirrors :class:`application.training.training_spec.DomainRandSb3Config`
    fields but collapses ``None`` and obviously-default ranges so the
    per-reset / per-step path is a couple of cheap if-checks rather than
    a chain of attribute reads. ``schedule_*`` plumbs in the optional
    linear ramp documented in :class:`DomainRandSchedule`.
    """

    enabled: bool = False
    mass_range: Optional[Tuple[float, float]] = None
    friction_range: Optional[Tuple[float, float]] = None
    # Stage H — replaces motor_strength_range / joint_damping_range.
    # DR perturbs the canonical (omega_n, zeta) and the env re-solves
    # per-joint kp/kd via mujoco_gain_solver at episode reset. Both
    # ranges are log-uniform (linear bounds; sampler is log-scale).
    omega_n_log_uniform: Optional[Tuple[float, float]] = None
    zeta_log_uniform: Optional[Tuple[float, float]] = None
    obs_noise_std: float = 0.0
    push_robot: bool = False
    push_interval_steps: int = 0
    push_force_range: Optional[Tuple[float, float]] = None
    # Schedule — mode "linear" ramps alpha from 0 → 1 between start_step
    # and end_step (global timesteps). "none" pins alpha at 1.0.
    schedule_mode: str = "none"
    schedule_start_step: int = 0
    schedule_end_step: int = 0


def _as_dr_range(value: Any) -> Optional[Tuple[float, float]]:
    """Coerce a (lo, hi) range into a Tuple[float, float]. Skip when the
    range is missing, malformed, or a no-op (lo == hi == 1.0 for scale
    multipliers / lo == hi == 0.0 for force / noise std)."""
    if value is None:
        return None
    try:
        lo = float(value[0])
        hi = float(value[1])
    except (TypeError, ValueError, IndexError):
        return None
    if lo > hi:
        lo, hi = hi, lo
    return (lo, hi)


def _build_dr_state(domain_rand: Any) -> Optional[_DRState]:
    """Translate :class:`DomainRandConfig` → :class:`_DRState`.

    Returns ``None`` when the caller did not supply DR config or the SB3
    sub-block is missing — the env's randomization hooks become a no-op,
    preserving the legacy "DR config absent ⇒ deterministic env" contract.
    """
    if domain_rand is None:
        return None
    sb3 = getattr(domain_rand, "sb3", None)
    if sb3 is None:
        return None
    schedule = getattr(domain_rand, "schedule", None)
    return _DRState(
        enabled=bool(getattr(sb3, "enabled", False)),
        mass_range=_as_dr_range(getattr(sb3, "mass_range", None)),
        friction_range=_as_dr_range(getattr(sb3, "friction_range", None)),
        omega_n_log_uniform=_as_dr_range(getattr(sb3, "omega_n_log_uniform", None)),
        zeta_log_uniform=_as_dr_range(getattr(sb3, "zeta_log_uniform", None)),
        obs_noise_std=float(getattr(sb3, "obs_noise_std", 0.0) or 0.0),
        push_robot=bool(getattr(sb3, "push_robot", False)),
        push_interval_steps=int(getattr(sb3, "push_interval_steps", 0) or 0),
        push_force_range=_as_dr_range(getattr(sb3, "push_force_range", None)),
        schedule_mode=str(getattr(schedule, "mode", "none") or "none").strip().lower(),
        schedule_start_step=int(getattr(schedule, "start_step", 0) or 0),
        schedule_end_step=int(getattr(schedule, "end_step", 0) or 0),
    )


@dataclass
class _Defaults:
    """Internal env knob bag (Stage 6 v1)."""

    sim_dt: float = 5e-3
    control_dt: float = 0.02
    max_episode_steps: int = 1000
    action_scale: float = 0.25
    action_clip: float = 1.0
    obs_clip_range: float = 0.0  # 0 disables per-component obs clipping
    use_default_offset: bool = True
    # Legacy scalar PD (fallback path when no pd_param + JointIRResolver
    # are available). The canonical path uses ``pd_kp_array``/``pd_kd_array``
    # populated from mujoco_gain_solver at construction (and re-solved on
    # every reset when ``resolve_at_reset`` is True).
    pd_kp: float = 25.0
    pd_kd: float = 0.5
    init_pos_x: float = 0.0
    init_pos_y: float = 0.0
    init_pos_z: float = 0.0  # 0 falls through to target_height
    track_sigma: float = 0.25
    term_base_z: float = 0.18
    target_height: float = 0.32


class GenericMujocoEnv(_GymEnvBase):
    """Brand-neutral MuJoCo + Gymnasium env.

    Inherits from :class:`gymnasium.Env` when gymnasium is installed (the
    runtime requirement); otherwise falls back to ``object`` so module
    import succeeds for inspecting constants like
    :data:`DEFAULT_QUADRUPED_MJCF`.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        robot_spec,
        scene_config=None,
        obs_action=None,
        *,
        actor_config=None,
        domain_rand=None,
        reward_fn: Optional[Callable[["GenericMujocoEnv"], float]] = None,
        reward_fn_rich: Optional[
            Callable[["GenericMujocoEnv"], Tuple[float, Dict[str, float]]]
        ] = None,
        termination_fn: Optional[Callable[["GenericMujocoEnv"], bool]] = None,
        sim_dt: Optional[float] = None,
        control_dt: Optional[float] = None,
        max_episode_steps: int = 1000,
        commands: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
        # Composite per-motion-item reward dispatch (CLAUDE.md §1.8 / plan
        # `rewards-stand-walk-run-...`). When ``terms_by_item`` is given,
        # ``reward_fn`` is ignored and each step's reward is a sigmoid-
        # blended sum over per-item closures keyed by an ``ItemResolver``
        # built from ``training_items``. ``blend_width`` controls the
        # transition softness around each item's command range edges.
        terms_by_item: Optional[Dict[str, Dict[str, Any]]] = None,
        training_items: Optional[Dict[str, Any]] = None,
        blend_width: float = 0.10,
        expose_active_item_obs: bool = False,
    ) -> None:
        if not _GYM_AVAILABLE:
            raise ImportError(
                "gymnasium is required for GenericMujocoEnv. Install via "
                "`pip install gymnasium mujoco`."
            )
        # Initialize gym.Env state (sets up np_random etc.).
        super().__init__()
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        elif not hasattr(self, "np_random"):
            self.np_random = np.random.default_rng()

        import mujoco

        self._robot = robot_spec
        self._scene = scene_config
        self._obs_action = obs_action
        self._actor = actor_config
        d = _Defaults()
        if scene_config is not None and getattr(scene_config, "sim_dt", None):
            d.sim_dt = float(scene_config.sim_dt)
        if sim_dt is not None:
            d.sim_dt = float(sim_dt)
        if control_dt is not None:
            d.control_dt = float(control_dt)
        if max_episode_steps:
            d.max_episode_steps = int(max_episode_steps)
        if obs_action is not None:
            d.action_scale = float(getattr(obs_action, "action_scale", d.action_scale))
            d.action_clip = float(getattr(obs_action, "action_clip", d.action_clip))
            d.obs_clip_range = float(getattr(obs_action, "obs_clip_range", d.obs_clip_range))
        if actor_config is not None:
            d.init_pos_x = float(getattr(actor_config, "init_pos_x", d.init_pos_x))
            d.init_pos_y = float(getattr(actor_config, "init_pos_y", d.init_pos_y))
            d.init_pos_z = float(getattr(actor_config, "init_pos_z", d.init_pos_z))
            d.use_default_offset = bool(
                getattr(actor_config, "action_use_default_offset", d.use_default_offset)
            )
            actuator = getattr(actor_config, "actuator", None)
            if actuator is not None:
                stiff = float(getattr(actuator, "stiffness", 0.0) or 0.0)
                damp = float(getattr(actuator, "damping", 0.0) or 0.0)
                if stiff > 0.0:
                    d.pd_kp = stiff
                if damp > 0.0:
                    d.pd_kd = damp
        d.target_height = float(getattr(robot_spec, "target_height", 0.0)) or d.target_height
        self._d = d

        # ── Load MJCF ──
        mjcf_path = getattr(robot_spec, "mjcf_path", None)
        loaded_path: Optional[str] = None
        if mjcf_path:
            p = Path(mjcf_path)
            if p.is_file():
                loaded_path = str(p)
            else:
                log_warning(
                    f"[envs] RobotSpec.mjcf_path {p!r} does not exist; "
                    f"falling back to DEFAULT_QUADRUPED_MJCF"
                )

        if loaded_path:
            self._model = mujoco.MjModel.from_xml_path(loaded_path)
            self._asset_source = loaded_path
        else:
            self._model = mujoco.MjModel.from_xml_string(DEFAULT_QUADRUPED_MJCF)
            self._asset_source = "DEFAULT_QUADRUPED_MJCF"

        self._model.opt.timestep = float(d.sim_dt)
        if scene_config is not None and getattr(scene_config, "gravity_z", None) is not None:
            self._model.opt.gravity[:] = (0.0, 0.0, float(scene_config.gravity_z))
        self._data = mujoco.MjData(self._model)
        self._mujoco = mujoco

        # ── Joint mapping: RobotSpec.joint_order vs MJCF actuators ──
        self._action_dim = int(self._model.nu)
        self._actuator_names = [
            mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            or f"act_{i}"
            for i in range(self._model.nu)
        ]
        # Read MJCF <motor gear="K"> scalars once. PD math produces
        # joint-space torque; the gear multiplier sits between ctrl and
        # joint torque (joint_torque = gear * ctrl). Guard against zero
        # gear in malformed MJCFs to avoid divide-by-zero in step().
        _ag = np.asarray(self._model.actuator_gear[:, 0], dtype=np.float64)
        _ag = np.where(np.abs(_ag) < 1e-12, 1.0, _ag)
        self._actuator_gear: np.ndarray = _ag
        # Phase 5: IR-only joint init wiring.
        # ``actor.joint_init`` arrives from spec.actor as an IR-keyed dict
        # (validated by spec_validator R8). We translate IR → physical via
        # JointIRResolver, then map each physical joint to its MJCF actuator
        # index via the actuator-name → MJCF-joint-name mapping that
        # MuJoCo exposes through actuator_trnid + jnt_id2name.
        self._default_qpos_actuated = np.zeros(self._action_dim, dtype=np.float64)
        joint_init = (
            getattr(actor_config, "joint_init", None)
            if actor_config is not None else None
        )
        if joint_init:
            try:
                from application.training.joint_ir import JointIRResolver
                resolver = JointIRResolver(robot_spec)
                phys_dict = resolver.to_physical_dict(
                    {str(k): float(v) for k, v in joint_init.items()},
                    where="spec.actor.joint_init (SB3 generic_mujoco_env)",
                )
                # Build physical-joint-name → MJCF actuator index map.
                # actuator_trnid[i, 0] is the MJCF joint id this actuator drives.
                phys_to_act_idx: Dict[str, int] = {}
                for i in range(self._action_dim):
                    jid = int(self._model.actuator_trnid[i, 0])
                    if 0 <= jid < self._model.njnt:
                        jname = (
                            mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_JOINT, jid)
                            or ""
                        )
                        if jname:
                            phys_to_act_idx[jname] = i
                missing: List[str] = []
                for phys_name, angle in phys_dict.items():
                    idx = phys_to_act_idx.get(phys_name)
                    if idx is None:
                        missing.append(phys_name)
                        continue
                    self._default_qpos_actuated[idx] = float(angle)
                if missing:
                    raise ValueError(
                        f"actor.joint_init translated IR roles to physical joints "
                        f"{sorted(phys_dict)} but {sorted(missing)} are not present "
                        f"in the MJCF actuator list ({sorted(phys_to_act_idx)}). "
                        f"Likely cause: RobotSpec.joint_order does not match the "
                        f"loaded MJCF — check the asset path and the registry "
                        f"entry for {getattr(robot_spec, 'sku', '?')!r}."
                    )
                log_info(
                    f"[envs] joint_init wired: IR→MJCF "
                    f"{ {ir: phys for ir, phys in zip(joint_init.keys(), phys_dict.keys())} }"
                )
            except Exception as exc:
                log_warning(
                    f"[envs] joint_init wiring failed ({exc!r}); "
                    f"_default_qpos_actuated stays at zeros — episode will start "
                    f"in T-pose / fall pose"
                )

        # Per-actuator joint range — read once from the MJCF for the
        # ``dof_pos_limits`` reward + ``joint_limit_violation`` termination.
        # Unlimited joints emit (0, 0) which the reward / done fns skip.
        soft_limits: List[Tuple[float, float]] = []
        for i in range(self._action_dim):
            jnt_id = int(self._model.actuator_trnid[i, 0])
            if 0 <= jnt_id < self._model.njnt and bool(self._model.jnt_limited[jnt_id]):
                lo = float(self._model.jnt_range[jnt_id, 0])
                hi = float(self._model.jnt_range[jnt_id, 1])
                soft_limits.append((lo, hi))
            else:
                soft_limits.append((0.0, 0.0))
        self._soft_joint_limits: List[Tuple[float, float]] = soft_limits

        # ── Stage E: per-joint mass-matrix-adaptive PD gains ──
        # Build the parallel (physical, IR-role) joint order in actuator
        # index order — this is what the PD math + step() consume. The
        # PhysX side (env_cfg_compiler) emits per-joint dicts; here we
        # produce the matching MuJoCo-side arrays via mj_fullM.
        self._joint_order_physical: List[str] = []
        for i in range(self._action_dim):
            jid = int(self._model.actuator_trnid[i, 0])
            jname = (
                mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_JOINT, jid) or ""
            ) if 0 <= jid < self._model.njnt else ""
            self._joint_order_physical.append(jname)

        self._joint_ir_roles: List[str] = []
        if robot_spec is not None and getattr(robot_spec, "joint_ir_roles", None):
            try:
                from application.training.joint_ir import JointIRResolver
                _resolver = JointIRResolver(robot_spec)
                self._joint_ir_roles = [
                    _resolver.to_ir(p) if p else ""
                    for p in self._joint_order_physical
                ]
            except Exception as exc:
                log_warning(
                    f"[envs] JointIRResolver init failed ({exc!r}); per-joint "
                    f"PD gains will fall back to scalar broadcast."
                )
                self._joint_ir_roles = []

        # Stash for resolve_at_reset (DR-aware re-solve).
        self._pd_param = getattr(actor_config, "pd_param", None) if actor_config is not None else None
        self._pd_resolve_at_reset = bool(
            getattr(actor_config, "resolve_at_reset", False)
            if actor_config is not None else False
        )
        self._mjcf_loaded_path = loaded_path  # None for DEFAULT_QUADRUPED_MJCF

        # Initial gain computation. Falls back to scalar broadcast when:
        # (a) no pd_param is provided (legacy canvas),
        # (b) IR-role mapping is incomplete,
        # (c) MJCF was loaded from a string (no path for the solver).
        self._pd_kp_array, self._pd_kd_array = self._resolve_pd_gains(
            d_pd_kp=d.pd_kp, d_pd_kd=d.pd_kd,
        )

        # ── Spaces ──
        from gymnasium import spaces
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self._action_dim,),
            dtype=np.float32,
        )
        # Composite per-item reward dispatch — must be wired before obs
        # layout because the optional ``cmd_norm + active_item_onehot``
        # tail depends on knowing the item count.
        self._build_per_item_rewards(
            terms_by_item=terms_by_item,
            training_items=training_items,
            blend_width=blend_width,
        )
        self._expose_active_item_obs = bool(
            expose_active_item_obs and self._item_resolver is not None
        )
        n_item_extras = (
            (1 + len(self._item_resolver.item_ids))
            if self._expose_active_item_obs else 0
        )

        # Obs layout: [base_ang_vel(3), projected_gravity(3),
        #              joint_pos(N), joint_vel(N), last_action(N), commands(3),
        #              (optional) cmd_norm(1), active_item_onehot(K)]
        n = self._action_dim
        self._obs_dim = 3 + 3 + n + n + n + 3 + n_item_extras
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self._obs_dim,),
            dtype=np.float32,
        )

        # ── Episode state ──
        self._action = np.zeros(self._action_dim, dtype=np.float32)
        self._last_action = np.zeros_like(self._action)
        self._command = np.asarray(
            commands if commands is not None else (0.5, 0.0, 0.0),
            dtype=np.float32,
        )
        self._step_count = 0

        # ── Reward / termination ──
        # When the per-item dispatcher is active, ``reward_fn`` is ignored
        # and ``step()`` takes the per-item path. A user-supplied reward_fn
        # together with a non-empty terms_by_item is a wiring contradiction
        # — fail loud per CLAUDE.md §1.8 rather than letting one win silently.
        if self._item_resolver is not None and (reward_fn is not None or reward_fn_rich is not None):
            raise ValueError(
                "GenericMujocoEnv: both terms_by_item and a single reward_fn "
                "were supplied. Choose one — composite per-item rewards "
                "override the single-closure path. See CLAUDE.md §1.8."
            )
        self._reward_fn = reward_fn or _default_reward
        self._reward_fn_rich = reward_fn_rich
        self._termination_fn = termination_fn or _default_termination
        self._track_sigma = d.track_sigma
        self._term_base_z = d.term_base_z

        # Cache slots used by reward/termination defaults
        self._base_pos = np.zeros(3, dtype=np.float64)
        self._base_quat = np.array([1.0, 0.0, 0.0, 0.0])  # wxyz
        self._lin_vel = np.zeros(3, dtype=np.float64)
        self._ang_vel = np.zeros(3, dtype=np.float64)
        self._proj_gravity = np.array([0.0, 0.0, -1.0])

        # ── Domain randomization setup ──
        # Snapshot original MJCF physical params so per-episode randomization
        # can re-apply from a clean baseline rather than compounding noise.
        self._dr_orig_body_mass = np.array(self._model.body_mass, dtype=np.float64).copy()
        self._dr_orig_geom_friction = np.array(self._model.geom_friction, dtype=np.float64).copy()
        # Stage H — actuator_gear / dof_damping are no longer DR-rescaled.
        # The (motor_strength_range, joint_damping_range) DR knobs were
        # removed in the hard-replace; PD bandwidth DR now happens via
        # (omega_n_log_uniform, zeta_log_uniform) and re-solving via
        # mujoco_gain_solver. We keep no snapshot because there's no
        # restore-and-rescale to perform.
        self._dr_cfg = _build_dr_state(domain_rand)
        # Find the "base" body driven by the freejoint (qpos[0:7]). When the
        # MJCF doesn't expose a body named "base" we fall back to body id 1
        # (the first non-world body) — push_robot then applies xfrc_applied
        # there, which is what every locomotion DR implementation does.
        self._dr_base_body_id: int = 1 if self._model.nbody > 1 else 0
        if self._dr_cfg is not None and self._dr_cfg.enabled:
            try:
                bid = mujoco.mj_name2id(
                    self._model, mujoco.mjtObj.mjOBJ_BODY, "base"
                )
                if bid > 0:
                    self._dr_base_body_id = int(bid)
            except Exception:                                 # pragma: no cover
                pass
        self._dr_global_step = 0
        self._dr_push_counter = 0

        # Render — lazy because mujoco.Renderer needs a GL context that
        # only some hosts have. Lazily creating on first ``render()`` lets
        # the env construct cheaply on training workers (which never
        # render) and still serves RecordVideo callers on the eval env.
        self.render_mode = render_mode
        self._renderer: Any = None
        self._render_failed: bool = False

        # Reset to a good initial state
        self.reset(seed=seed)

        log_info(
            f"[envs] GenericMujocoEnv ready — robot={getattr(robot_spec,'sku','?')!r} "
            f"asset={Path(self._asset_source).name} "
            f"obs_dim={self._obs_dim} action_dim={self._action_dim} "
            f"sim_dt={d.sim_dt:.4f} control_dt={d.control_dt:.4f}"
        )

    # ------------------------------------------------------------------
    # gymnasium.Env interface
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        self._mujoco.mj_resetData(self._model, self._data)

        # Per-episode domain randomization — restore the model to its MJCF
        # baseline first so noise can never compound across episodes, then
        # sample fresh scaling factors gated by the schedule's alpha.
        self._apply_domain_randomization()

        # Place base at the canvas-set spawn pose. ``init_pos_z`` falling back
        # to ``target_height`` keeps the legacy behaviour for canvases that
        # leave the actor_setting Z at the dataclass default 0.4 while
        # honoring an explicit user override (e.g. 0.7 for tall humanoids).
        # USD↔MJCF anchor offset: the user's init_pos_z is authored
        # against USD/IsaacLab semantics; in MJCF the same value drops
        # the trunk too low (穿地). The per-SKU calibration overlay
        # lifts the spawn at standing pose. See plan
        # curious-nibbling-plum.md and
        # application.training.validation.mjcf_base_calibration.
        if self._model.nq >= 7:
            spawn_z = self._d.init_pos_z if self._d.init_pos_z > 0 else self._d.target_height
            sku = str(getattr(self._robot, "sku", "") or "")
            if sku:
                from application.service.robot_assets.runtime import (
                    read_mjcf_base_offset,
                )
                off_z, _ = read_mjcf_base_offset(sku)
                spawn_z = float(spawn_z) + off_z
            self._data.qpos[:3] = (
                self._d.init_pos_x, self._d.init_pos_y, spawn_z,
            )
            self._data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)  # wxyz identity
        # Apply default joint angles (zeros for now; Stage 4 actor.joint_init
        # would feed RobotSpec joint defaults here)
        if self._model.nq >= 7 + self._action_dim:
            self._data.qpos[7:7 + self._action_dim] = self._default_qpos_actuated
        self._mujoco.mj_forward(self._model, self._data)
        self._step_count = 0
        self._push_step_counter = 0
        self._action.fill(0.0)
        self._last_action.fill(0.0)
        self._refresh_state_cache()
        return self._build_obs(), {"asset_source": self._asset_source}

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (self._action_dim,):
            raise ValueError(
                f"action shape {action.shape} != expected ({self._action_dim},)"
            )
        clip_range = float(self._d.action_clip) if self._d.action_clip > 0 else 1.0
        action = np.clip(action, -clip_range, clip_range)
        self._last_action = self._action
        self._action = action

        # PD-style position control: target qpos = default + scale * action.
        # When the actor explicitly opts out of the default offset, treat the
        # action_scale * action term as the absolute target around zero
        # (matches the IL reference implementation's "delta=False" path).
        if self._d.use_default_offset:
            target = self._default_qpos_actuated + self._d.action_scale * action
        else:
            target = self._d.action_scale * action

        # Apply external-force push (DR). xfrc_applied is a transient force
        # the integrator consumes during the next mj_step; we clear it on
        # the substep after to avoid sustained drift.
        push_applied = self._maybe_apply_push()

        # Run sim_dt sub-steps until we reach control_dt
        # PD math is per-joint in JOINT-torque units: pd_kp_array /
        # pd_kd_array are (n_joints,) arrays populated by _resolve_pd_gains.
        # For the canonical path (with PDParam + mj_fullM) the diagonal
        # of M(q0) appears inside kp_j = M_jj * omega_n^2; for the legacy
        # scalar fallback the arrays are uniform broadcasts.
        #
        # MJCF <motor gear="K"> means joint_torque = gear * ctrl, so the
        # joint-space PD output must be divided by the actuator gear
        # before being written to ctrl. The legacy scalar path produced
        # mild values (pd_kp ~ 25) that the gear=33.5 amplified to ~840 —
        # marginally stable because the policy's action_scale=0.25 kept
        # targets small. The canonical mass-matrix-adaptive path produces
        # gain values consistent with the joint's effective mass, so the
        # gear divide is mandatory or simulation diverges (NaN qacc).
        n_substeps = max(1, int(round(self._d.control_dt / self._d.sim_dt)))
        for sub_i in range(n_substeps):
            q = self._data.qpos[7:7 + self._action_dim]
            qdot = self._data.qvel[6:6 + self._action_dim]
            joint_torque = (
                self._pd_kp_array * (target - q)
                - self._pd_kd_array * qdot
            )
            self._data.ctrl[:self._action_dim] = joint_torque / self._actuator_gear
            self._mujoco.mj_step(self._model, self._data)
            # Push is a single-tick impulse — clear after the first substep
            # so it does not turn into a constant body force.
            if push_applied and sub_i == 0:
                self._data.xfrc_applied[self._dr_base_body_id, :3] = 0.0

        self._step_count += 1
        self._dr_global_step += 1
        self._refresh_state_cache()

        terminated = bool(self._termination_fn(self))
        truncated = self._step_count >= self._d.max_episode_steps
        reward, reward_breakdown, item_breakdown, active_item = self._compute_reward()
        # Cache last per-item weights so _build_obs can append the
        # cmd_norm + onehot tail without recomputing the resolver.
        if self._item_resolver is not None:
            self._last_item_weights = item_breakdown
            self._last_active_item = active_item
        info: Dict[str, Any] = {"step": self._step_count}
        for term_name, contribution in reward_breakdown.items():
            info[f"reward/{term_name}"] = contribution
        for item_id, item_total in item_breakdown.items():
            info[f"reward/item/{item_id}"] = item_total
        if active_item:
            info["reward/active_item"] = active_item
        # Locomotion convention: episode is a "success" iff it ended by the
        # time-limit (truncated) without the termination predicate firing
        # (fall / non-finite state). Emitting on every step keeps Monitor's
        # info_keywords copy cheap; SB3's ep_info_buffer only reads the final
        # transition.
        if terminated or truncated:
            info["is_success"] = bool(truncated and not terminated)
        return self._build_obs(), reward, terminated, truncated, info

    def close(self) -> None:
        # Release the offscreen GL context if one was lazily allocated by
        # ``render()``. MjModel/MjData are GC'd via Python refs.
        if self._renderer is not None:
            try:
                self._renderer.close()
            except Exception:                                # pragma: no cover
                pass
            self._renderer = None
        return None

    def render(self) -> Optional[np.ndarray]:
        """Return an ``(H, W, 3)`` uint8 RGB frame, or ``None`` when the
        env was not built with ``render_mode="rgb_array"`` / the offscreen
        renderer cannot initialise on this host.

        gymnasium.wrappers.RecordVideo (the only intended caller) treats a
        ``None`` return as "nothing to record this step" so a missing
        EGL/GLFW context degrades to "no video", never to a crash.
        """
        if self.render_mode != "rgb_array":
            return None
        if self._render_failed:
            return None
        if self._renderer is None:
            try:
                self._renderer = self._mujoco.Renderer(self._model, height=240, width=320)
            except Exception as exc:                          # pragma: no cover
                log_warning(
                    f"[envs] offscreen renderer init failed ({exc!r}); "
                    f"GenericMujocoEnv.render() will return None for the "
                    f"rest of this env's lifetime."
                )
                self._render_failed = True
                return None
        try:
            self._renderer.update_scene(self._data, camera=-1)
            return np.asarray(self._renderer.render(), dtype=np.uint8)
        except Exception as exc:                              # pragma: no cover
            log_warning(f"[envs] render() failed ({exc!r}); returning None")
            return None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _refresh_state_cache(self) -> None:
        if self._model.nq >= 7:
            self._base_pos = np.asarray(self._data.qpos[:3], dtype=np.float64)
            self._base_quat = np.asarray(self._data.qpos[3:7], dtype=np.float64)  # wxyz
        if self._model.nv >= 6:
            self._lin_vel = np.asarray(self._data.qvel[:3], dtype=np.float64)
            self._ang_vel = np.asarray(self._data.qvel[3:6], dtype=np.float64)
        self._proj_gravity = _project_gravity_wxyz(self._base_quat)
        self._qpos = np.asarray(
            self._data.qpos[7:7 + self._action_dim], dtype=np.float64
        )
        self._qvel = np.asarray(
            self._data.qvel[6:6 + self._action_dim], dtype=np.float64
        )
        # Used by the canvas-driven reward registry (energy / torque /
        # accel penalties). qfrc_actuator is post-step joint-space force.
        try:
            self._qfrc_actuator = np.asarray(
                self._data.qfrc_actuator[6:6 + self._action_dim], dtype=np.float64
            )
        except Exception:
            self._qfrc_actuator = np.zeros(self._action_dim, dtype=np.float64)
        try:
            self._qacc = np.asarray(
                self._data.qacc[6:6 + self._action_dim], dtype=np.float64
            )
        except Exception:
            self._qacc = np.zeros(self._action_dim, dtype=np.float64)

    def _resolve_pd_gains(
        self,
        *,
        d_pd_kp: float,
        d_pd_kd: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute per-joint kp/kd arrays for the current model state.

        Canonical path (when ``self._pd_param`` is set AND we have a valid
        MJCF path AND every actuator has an IR-role mapping): calls
        :func:`application.physics.mujoco_gain_solver.solve` at the current
        ``self._data.qpos`` (the env's nominal stance) so the diagonal of
        ``M(q0)`` reflects domain-randomized masses if DR is active.

        Fallback path (any of the above missing): broadcasts the legacy
        scalar ``d_pd_kp`` / ``d_pd_kd`` to (n_joints,) arrays. Logs WARN
        at construction time so the user knows they're off the canonical
        path; subsequent re-solves at reset() stay silent.

        Returns ``(pd_kp_array, pd_kd_array)`` — both shape (action_dim,),
        dtype float64.
        """
        n = int(self._action_dim)
        can_solve = (
            self._pd_param is not None
            and self._mjcf_loaded_path is not None
            and len(self._joint_ir_roles) == n
            and all(self._joint_ir_roles)
            and all(self._joint_order_physical)
        )
        if not can_solve:
            return (
                np.full(n, float(d_pd_kp), dtype=np.float64),
                np.full(n, float(d_pd_kd), dtype=np.float64),
            )

        try:
            from application.physics.mujoco_gain_solver import solve as _solve
        except Exception as exc:
            log_warning(
                f"[envs] mujoco_gain_solver import failed ({exc!r}); "
                f"falling back to scalar PD."
            )
            return (
                np.full(n, float(d_pd_kp), dtype=np.float64),
                np.full(n, float(d_pd_kd), dtype=np.float64),
            )

        # mj_fullM at the current qpos (already at nominal at __init__
        # time; at reset time the DR block writes the new masses BEFORE
        # we get here, so M reflects the current episode's parameters).
        nominal_qpos = np.array(self._data.qpos, dtype=np.float64)
        try:
            gains = _solve(
                mjcf_path=Path(self._mjcf_loaded_path),
                joint_order_physical=list(self._joint_order_physical),
                joint_ir_roles=list(self._joint_ir_roles),
                nominal_qpos=nominal_qpos,
                pd_param=self._pd_param,
            )
        except Exception as exc:
            log_warning(
                f"[envs] mujoco_gain_solver.solve failed ({exc!r}); "
                f"falling back to scalar PD for this episode."
            )
            return (
                np.full(n, float(d_pd_kp), dtype=np.float64),
                np.full(n, float(d_pd_kd), dtype=np.float64),
            )

        return (
            np.asarray(gains.kp, dtype=np.float64),
            np.asarray(gains.kd, dtype=np.float64),
        )

    def _build_per_item_rewards(
        self,
        *,
        terms_by_item: Optional[Dict[str, Dict[str, Any]]],
        training_items: Optional[Dict[str, Any]],
        blend_width: float,
    ) -> None:
        """Compile composite per-item reward closures + an ItemResolver.

        Sets ``self._item_resolver`` to None when no per-item terms are
        configured (env falls back to the single ``reward_fn`` path). When
        terms_by_item is non-empty, training_items MUST carry the matching
        command_ranges — otherwise raises (CLAUDE.md §1.8: no silent
        fallback to the global rewards bag).
        """
        from application.training.envs.reward_terms import (
            build_reward_fn_with_breakdown,
        )

        self._item_resolver = None
        self._per_item_reward_fns: Dict[str, Any] = {}
        self._last_item_weights: Dict[str, float] = {}
        self._last_active_item: str = ""

        if not terms_by_item:
            return
        if not isinstance(terms_by_item, dict):
            raise TypeError(
                f"GenericMujocoEnv: terms_by_item must be a dict, got "
                f"{type(terms_by_item).__name__}."
            )
        from application.training.envs.item_resolver import ItemResolver

        # Item order is the dict insertion order — already canvas-deterministic
        # because spec_compiler iterates rewards_nodes in topo order. Cast to
        # list so we have a stable index for the obs one-hot.
        item_id_order = list(terms_by_item.keys())
        self._item_resolver = ItemResolver(
            training_items=training_items or {},
            item_id_order=item_id_order,
            blend_width=blend_width,
        )
        for item_id, term_dict in terms_by_item.items():
            self._per_item_reward_fns[item_id] = (
                build_reward_fn_with_breakdown(term_dict)
            )

    def _compute_reward(
        self,
    ) -> Tuple[float, Dict[str, float], Dict[str, float], str]:
        """Return ``(total, term_breakdown, item_breakdown, active_item)``.

        Single-closure path (``_item_resolver is None``) collapses the
        last three slots — caller treats them as empty / "" — so the
        info-dict population in :meth:`step` is uniform.
        """
        if self._item_resolver is None:
            if self._reward_fn_rich is not None:
                total, breakdown = self._reward_fn_rich(self)
                return float(total), dict(breakdown), {}, ""
            return float(self._reward_fn(self)), {}, {}, ""
        weights = self._item_resolver.weights(self._command)
        total = 0.0
        term_breakdown: Dict[str, float] = {}
        item_breakdown: Dict[str, float] = {}
        active_item = ""
        best_alpha = -1.0
        for item_id, alpha in weights.items():
            fn = self._per_item_reward_fns.get(item_id)
            if fn is None:
                continue
            item_total, item_terms = fn(self)
            blended = alpha * float(item_total)
            total += blended
            item_breakdown[item_id] = float(blended)
            for term_name, contribution in item_terms.items():
                term_breakdown[term_name] = (
                    term_breakdown.get(term_name, 0.0) + alpha * float(contribution)
                )
            if alpha > best_alpha:
                best_alpha = alpha
                active_item = item_id
        return float(total), term_breakdown, item_breakdown, active_item

    def _build_obs(self) -> np.ndarray:
        n = self._action_dim
        out = np.zeros(self._obs_dim, dtype=np.float32)
        i = 0
        out[i:i + 3] = self._ang_vel.astype(np.float32); i += 3
        out[i:i + 3] = self._proj_gravity.astype(np.float32); i += 3
        out[i:i + n] = self._qpos.astype(np.float32); i += n
        out[i:i + n] = self._qvel.astype(np.float32); i += n
        out[i:i + n] = self._action.astype(np.float32); i += n
        out[i:i + 3] = self._command.astype(np.float32); i += 3
        if self._expose_active_item_obs and self._item_resolver is not None:
            # cmd_norm: L2 norm of the [vx, vy, wyaw] command. Critic uses
            # this to predict the per-item reward boundary cross-fade.
            cmd_norm = float(np.linalg.norm(self._command))
            out[i] = np.float32(cmd_norm); i += 1
            # active_item_onehot, ordered by ItemResolver.item_ids. We use
            # the last step's weight distribution so the critic sees the
            # *blend*, not just the argmax — that's the signal that matches
            # the reward it actually received this step.
            for iid in self._item_resolver.item_ids:
                out[i] = np.float32(self._last_item_weights.get(iid, 0.0))
                i += 1
        # Per-component obs clip (separate from VecNormalize's whole-vector
        # clip configured on env_assembler). 0 disables clipping.
        if self._d.obs_clip_range > 0.0:
            np.clip(out, -self._d.obs_clip_range, self._d.obs_clip_range, out=out)
        # Domain-randomization additive Gaussian obs noise (after clip so
        # the noise is what the policy actually sees, not what gets clipped
        # off the tails).
        std = self._dr_noise_std_effective()
        if std > 0.0:
            noise = self.np_random.normal(0.0, std, size=out.shape).astype(np.float32)
            out = out + noise
        return out

    # ------------------------------------------------------------------
    # Domain randomization
    # ------------------------------------------------------------------

    def _dr_alpha(self) -> float:
        """Schedule strength in [0, 1].

        ``"none"`` (or absent schedule) → 1.0; ``"linear"`` ramps from 0
        at ``start_step`` to 1 at ``end_step`` so the policy sees a clean
        baseline for the first chunk of training and full DR after the
        ramp completes. Out-of-range steps are clamped at the endpoints.
        """
        cfg = self._dr_cfg
        if cfg is None or not cfg.enabled:
            return 0.0
        mode = cfg.schedule_mode
        if mode != "linear":
            return 1.0
        start = cfg.schedule_start_step
        end = cfg.schedule_end_step
        step = int(self._dr_global_step)
        if end <= start:
            return 1.0
        if step <= start:
            return 0.0
        if step >= end:
            return 1.0
        return float(step - start) / float(end - start)

    def _dr_sample_factor(self, lo: float, hi: float, alpha: float) -> float:
        """Sample a multiplicative scaling factor anchored at 1.0.

        Returns ``1 + alpha * (U(lo, hi) - 1)`` so alpha=0 keeps the
        MJCF baseline and alpha=1 spans the full configured range.
        """
        if hi <= lo:
            sample = lo
        else:
            sample = float(self.np_random.uniform(lo, hi))
        return 1.0 + float(alpha) * (sample - 1.0)

    def _apply_domain_randomization(self) -> None:
        """Restore MJCF baseline, then re-apply per-episode randomized
        scaling per :class:`_DRState`. Invariant: callers (reset()) run
        this BEFORE writing qpos and BEFORE the first :func:`mj_forward`
        so the new physical params drive the initial state correctly.
        """
        # Always restore — even when DR is disabled — so toggling DR off
        # mid-process cannot leak randomized state into the env. Stage H
        # removed the actuator_gear / dof_damping restore lines along
        # with the (motor_strength, joint_damping) DR knobs — neither is
        # rescaled anymore, so there's nothing to restore.
        self._model.body_mass[:] = self._dr_orig_body_mass
        self._model.geom_friction[:] = self._dr_orig_geom_friction
        try:
            self._data.xfrc_applied[:] = 0.0
        except Exception:                                    # pragma: no cover
            pass

        cfg = self._dr_cfg
        if cfg is None or not cfg.enabled:
            return
        alpha = self._dr_alpha()
        if alpha <= 0.0:
            return

        if cfg.mass_range is not None:
            f = self._dr_sample_factor(*cfg.mass_range, alpha=alpha)
            self._model.body_mass[:] = self._dr_orig_body_mass * f
        if cfg.friction_range is not None:
            f = self._dr_sample_factor(*cfg.friction_range, alpha=alpha)
            # Only scale sliding friction (column 0); torsional / rolling
            # columns 1/2 stay at MJCF defaults — they are rarely
            # randomized in locomotion DR and mutating them breaks the
            # MJCF author's tuning.
            geom_f = self._dr_orig_geom_friction.copy()
            geom_f[:, 0] *= f
            self._model.geom_friction[:] = geom_f
        # Stage H — DR on the canonical (omega_n, zeta) parameterization.
        # Sample one multiplier each for omega_n and zeta (log-uniform on
        # the provided linear bounds), build a scaled PDParam, and re-solve
        # per-joint gains via mujoco_gain_solver. The PhysX side gets the
        # same scaling via an event term emitted by env_cfg_compiler.
        #
        # NOTE: the legacy motor_strength_range / joint_damping_range
        # branches (scaling actuator_gear / dof_damping) were deleted in
        # the Stage H hard-replace. Both knobs were a leaky proxy for "PD
        # bandwidth" that diverged from the PhysX side. The PD-bandwidth
        # signal lives in (omega_n, zeta); driving DR there is what we
        # want.
        if self._pd_param is not None and (
            cfg.omega_n_log_uniform is not None
            or cfg.zeta_log_uniform is not None
        ):
            # Log-uniform on (lo, hi): sample u ~ U(log lo, log hi), use
            # exp(u). Equivalent to multiplying by exp(U(log lo, log hi)).
            import math as _math

            def _sample_log_uniform(rng: Tuple[float, float]) -> float:
                lo, hi = rng
                if hi <= lo:
                    return float(lo)
                log_lo, log_hi = _math.log(max(lo, 1e-6)), _math.log(max(hi, 1e-6))
                u = float(self.np_random.uniform(log_lo, log_hi))
                # Apply DR alpha by interpolating between 1.0 (no DR) and
                # the sampled factor.
                factor = _math.exp(u)
                return 1.0 + float(alpha) * (factor - 1.0)

            omega_factor = (
                _sample_log_uniform(cfg.omega_n_log_uniform)
                if cfg.omega_n_log_uniform is not None else 1.0
            )
            zeta_factor = (
                _sample_log_uniform(cfg.zeta_log_uniform)
                if cfg.zeta_log_uniform is not None else 1.0
            )
            try:
                scaled_param = self._pd_param.scaled(
                    omega_n_factor=omega_factor,
                    zeta_factor=zeta_factor,
                )
                # Temporarily swap pd_param so _resolve_pd_gains uses the
                # scaled values, then restore.
                original = self._pd_param
                self._pd_param = scaled_param
                kp_arr, kd_arr = self._resolve_pd_gains(
                    d_pd_kp=self._d.pd_kp, d_pd_kd=self._d.pd_kd,
                )
                self._pd_param = original
                self._pd_kp_array = kp_arr
                self._pd_kd_array = kd_arr
                return  # DR for actuator side fully handled
            except Exception as exc:
                log_warning(
                    f"[envs] DR (omega_n, zeta) re-solve failed ({exc!r}); "
                    f"falling back to nominal gains for this episode."
                )

        # Stage E: re-solve PD gains so kp/kd track the DR-perturbed
        # mass matrix even when DR doesn't touch (omega_n, zeta) directly.
        # The mass DR above changed body_mass; re-solving picks up the
        # new M(q0) diagonal automatically.
        if self._pd_resolve_at_reset:
            kp_arr, kd_arr = self._resolve_pd_gains(
                d_pd_kp=self._d.pd_kp, d_pd_kd=self._d.pd_kd,
            )
            self._pd_kp_array = kp_arr
            self._pd_kd_array = kd_arr

    def _maybe_apply_push(self) -> bool:
        """Apply a transient push force on the base body at the configured
        interval. Returns True when a push was written so step() can
        clear ``xfrc_applied`` after the first substep — otherwise the
        force would persist for every substep in the control_dt window
        and become a sustained body load instead of an impulse.
        """
        cfg = self._dr_cfg
        if cfg is None or not cfg.enabled or not cfg.push_robot:
            return False
        if cfg.push_force_range is None or cfg.push_interval_steps <= 0:
            return False
        alpha = self._dr_alpha()
        if alpha <= 0.0:
            return False

        self._push_step_counter += 1
        if self._push_step_counter < cfg.push_interval_steps:
            return False
        self._push_step_counter = 0

        lo, hi = cfg.push_force_range
        if hi <= lo:
            magnitude = lo
        else:
            magnitude = float(self.np_random.uniform(lo, hi))
        magnitude *= alpha
        # Sample a planar direction so vertical pushes don't dominate the
        # disturbance budget (the policy mostly fights horizontal forces
        # in the real world).
        theta = float(self.np_random.uniform(-np.pi, np.pi))
        fx = magnitude * np.cos(theta)
        fy = magnitude * np.sin(theta)
        try:
            self._data.xfrc_applied[self._dr_base_body_id, 0] = fx
            self._data.xfrc_applied[self._dr_base_body_id, 1] = fy
            self._data.xfrc_applied[self._dr_base_body_id, 2] = 0.0
        except Exception:                                    # pragma: no cover
            return False
        return True

    def _dr_noise_std_effective(self) -> float:
        cfg = self._dr_cfg
        if cfg is None or not cfg.enabled or cfg.obs_noise_std <= 0.0:
            return 0.0
        return float(cfg.obs_noise_std) * self._dr_alpha()

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def num_joints(self) -> int:
        return self._action_dim

    @property
    def asset_source(self) -> str:
        return self._asset_source

    @property
    def actuator_names(self) -> List[str]:
        return list(self._actuator_names)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_env(
    spec,
    *,
    commands=None,
    seed: Optional[int] = None,
    render_mode: Optional[str] = None,
) -> GenericMujocoEnv:
    """Build a :class:`GenericMujocoEnv` from a :class:`TrainingSpec`.

    Reads ``spec.robot``, ``spec.scene``, ``spec.obs_action``,
    ``spec.physics``, ``spec.rewards.terms`` and ``spec.terminations.conditions``
    to construct the env. The reward / termination dicts are compiled by
    :mod:`application.training.envs.reward_terms` into closures that
    :class:`GenericMujocoEnv` calls each step. An empty rewards or
    terminations dict falls back to the env's built-in defaults so the
    smoke runs (no canvas) keep working.
    """
    if spec is None or not getattr(spec, "robot", None):
        raise ValueError("make_env: spec.robot is missing or empty")
    physics = getattr(spec, "physics", None)
    sim_dt = getattr(physics, "sim_dt", None) if physics is not None else None
    control_dt = getattr(physics, "control_dt", None) if physics is not None else None
    max_steps = (
        int(getattr(physics, "episode_max_steps", 1000) or 1000)
        if physics is not None else 1000
    )
    # task_config.truncation_max_steps overrides physics.episode_max_steps
    # when set (>0); this lets the canvas decouple the episode horizon from
    # the sim integrator cadence without having to retune both nodes.
    task_cfg = getattr(spec, "task", None)
    if task_cfg is not None:
        trunc = int(getattr(task_cfg, "truncation_max_steps", 0) or 0)
        if trunc > 0:
            max_steps = trunc
    cmd = commands
    if cmd is None and getattr(spec, "task", None) is not None:
        cmd = np.asarray(
            (spec.task.target_lin_vel_x, spec.task.target_lin_vel_y, spec.task.target_ang_vel_z),
            dtype=np.float32,
        )

    from application.training.envs.reward_terms import (
        build_done_fn,
        build_reward_fn_with_breakdown,
    )

    reward_fn_rich = None
    terms_by_item: Optional[Dict[str, Dict[str, Any]]] = None
    training_items: Optional[Dict[str, Any]] = None
    blend_width = 0.10
    expose_active_item_obs = False
    rewards_cfg = getattr(spec, "rewards", None)
    motion_cfg = getattr(spec, "motion", None)
    if rewards_cfg is not None:
        raw_per_item = getattr(rewards_cfg, "terms_by_item", None) or {}
        terms = getattr(rewards_cfg, "terms", None) or {}
        if raw_per_item:
            ti = (
                getattr(motion_cfg, "training_items", None)
                if motion_cfg is not None else None
            ) or {}
            if not ti:
                raise RuntimeError(
                    "spec.rewards.terms_by_item is non-empty but "
                    "spec.motion.training_items is missing — cannot map the "
                    "command vector to an active item. Either wire a "
                    "TrainingMotion node with command_ranges, or remove the "
                    "per-item reward edges from the canvas. "
                    "(CLAUDE.md §1.8 — no silent fallback to global rewards.)"
                )
            terms_by_item = raw_per_item
            training_items = ti
            if motion_cfg is not None:
                # Reuse motion.deadzone as the soft-blend half-width unless a
                # narrower transition is configured downstream (TrainingMotion
                # currently exposes deadzone as the cmd dead-zone radius; using
                # it as blend width matches "soft-mix around that boundary").
                blend_width = float(getattr(motion_cfg, "deadzone", 0.10) or 0.10)
            expose_active_item_obs = True
        elif terms:
            reward_fn_rich = build_reward_fn_with_breakdown(terms)

    termination_fn = None
    term_cfg = getattr(spec, "terminations", None)
    if term_cfg is not None:
        conditions = getattr(term_cfg, "conditions", None) or {}
        if conditions:
            termination_fn = build_done_fn(conditions)

    return GenericMujocoEnv(
        robot_spec=spec.robot,
        scene_config=getattr(spec, "scene", None),
        obs_action=getattr(spec, "obs_action", None),
        actor_config=getattr(spec, "actor", None),
        domain_rand=getattr(spec, "domain_rand", None),
        sim_dt=sim_dt,
        control_dt=control_dt,
        max_episode_steps=max_steps,
        commands=cmd,
        seed=seed,
        reward_fn_rich=reward_fn_rich,
        termination_fn=termination_fn,
        render_mode=render_mode,
        terms_by_item=terms_by_item,
        training_items=training_items,
        blend_width=blend_width,
        expose_active_item_obs=expose_active_item_obs,
    )


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------


def _project_gravity_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    """Rotate ``[0, 0, -1]`` (world gravity dir) into the body frame
    using a unit quaternion in ``(w, x, y, z)`` order. Returns a (3,)
    float64 vector."""
    w, x, y, z = (
        float(quat_wxyz[0]),
        float(quat_wxyz[1]),
        float(quat_wxyz[2]),
        float(quat_wxyz[3]),
    )
    # R^T * [0, 0, -1]
    return np.array([
        2.0 * (x * z - w * y),
        2.0 * (y * z + w * x),
        -(1.0 - 2.0 * (x * x + y * y)),
    ], dtype=np.float64)


__all__ = [
    "DEFAULT_QUADRUPED_MJCF",
    "GenericMujocoEnv",
    "make_env",
]
