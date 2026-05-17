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
class _Defaults:
    """Internal env knob bag (Stage 6 v1)."""

    sim_dt: float = 5e-3
    control_dt: float = 0.02
    max_episode_steps: int = 1000
    action_scale: float = 0.25
    action_clip: float = 1.0
    obs_clip_range: float = 0.0  # 0 disables per-component obs clipping
    use_default_offset: bool = True
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

    metadata = {"render_modes": []}

    def __init__(
        self,
        robot_spec,
        scene_config=None,
        obs_action=None,
        *,
        actor_config=None,
        reward_fn: Optional[Callable[["GenericMujocoEnv"], float]] = None,
        termination_fn: Optional[Callable[["GenericMujocoEnv"], bool]] = None,
        sim_dt: Optional[float] = None,
        control_dt: Optional[float] = None,
        max_episode_steps: int = 1000,
        commands: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
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

        # ── Spaces ──
        from gymnasium import spaces
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self._action_dim,),
            dtype=np.float32,
        )
        # Obs layout: [base_ang_vel(3), projected_gravity(3),
        #              joint_pos(N), joint_vel(N), last_action(N), commands(3)]
        n = self._action_dim
        self._obs_dim = 3 + 3 + n + n + n + 3
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
        self._reward_fn = reward_fn or _default_reward
        self._termination_fn = termination_fn or _default_termination
        self._track_sigma = d.track_sigma
        self._term_base_z = d.term_base_z

        # Cache slots used by reward/termination defaults
        self._base_pos = np.zeros(3, dtype=np.float64)
        self._base_quat = np.array([1.0, 0.0, 0.0, 0.0])  # wxyz
        self._lin_vel = np.zeros(3, dtype=np.float64)
        self._ang_vel = np.zeros(3, dtype=np.float64)
        self._proj_gravity = np.array([0.0, 0.0, -1.0])

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
        # Place base at the canvas-set spawn pose. ``init_pos_z`` falling back
        # to ``target_height`` keeps the legacy behaviour for canvases that
        # leave the actor_setting Z at the dataclass default 0.4 while
        # honoring an explicit user override (e.g. 0.7 for tall humanoids).
        if self._model.nq >= 7:
            spawn_z = self._d.init_pos_z if self._d.init_pos_z > 0 else self._d.target_height
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

        # Run sim_dt sub-steps until we reach control_dt
        n_substeps = max(1, int(round(self._d.control_dt / self._d.sim_dt)))
        for _ in range(n_substeps):
            q = self._data.qpos[7:7 + self._action_dim]
            qdot = self._data.qvel[6:6 + self._action_dim]
            torque = (
                self._d.pd_kp * (target - q)
                - self._d.pd_kd * qdot
            )
            self._data.ctrl[:self._action_dim] = torque
            self._mujoco.mj_step(self._model, self._data)

        self._step_count += 1
        self._refresh_state_cache()

        terminated = bool(self._termination_fn(self))
        truncated = self._step_count >= self._d.max_episode_steps
        reward = float(self._reward_fn(self))
        info: Dict[str, Any] = {"step": self._step_count}
        # Locomotion convention: episode is a "success" iff it ended by the
        # time-limit (truncated) without the termination predicate firing
        # (fall / non-finite state). Emitting on every step keeps Monitor's
        # info_keywords copy cheap; SB3's ep_info_buffer only reads the final
        # transition.
        if terminated or truncated:
            info["is_success"] = bool(truncated and not terminated)
        return self._build_obs(), reward, terminated, truncated, info

    def close(self) -> None:
        # MuJoCo MjModel/MjData are GC'd via Python refs; nothing to release.
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
        # Per-component obs clip (separate from VecNormalize's whole-vector
        # clip configured on env_assembler). 0 disables clipping.
        if self._d.obs_clip_range > 0.0:
            np.clip(out, -self._d.obs_clip_range, self._d.obs_clip_range, out=out)
        return out

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


def make_env(spec, *, commands=None, seed: Optional[int] = None) -> GenericMujocoEnv:
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

    from application.training.envs.reward_terms import build_done_fn, build_reward_fn

    reward_fn = None
    rewards_cfg = getattr(spec, "rewards", None)
    if rewards_cfg is not None:
        terms = getattr(rewards_cfg, "terms", None) or {}
        if terms:
            reward_fn = build_reward_fn(terms)

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
        sim_dt=sim_dt,
        control_dt=control_dt,
        max_episode_steps=max_steps,
        commands=cmd,
        seed=seed,
        reward_fn=reward_fn,
        termination_fn=termination_fn,
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
