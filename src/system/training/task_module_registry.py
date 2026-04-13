from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Optional, Tuple


# ── Backend / algorithm compatibility constants ───────────────────────
# These tags let the UI filter which reward functions are applicable to
# the user's current training context (engine × algorithm).

BACKEND_SB3 = "sb3"
BACKEND_ISAAC = "isaac_lab"
BACKEND_NEWTON = "newton"
ALL_BACKENDS: FrozenSet[str] = frozenset({BACKEND_SB3, BACKEND_ISAAC, BACKEND_NEWTON})

ALG_PPO = "PPO"
ALG_SAC = "SAC"
ALG_AMP = "AMP"
ALG_TD3 = "TD3"
ALG_ALL = "ALL"
ALL_ALGORITHMS: FrozenSet[str] = frozenset({ALG_PPO, ALG_SAC, ALG_AMP, ALG_TD3, ALG_ALL})

# Isaac Lab module routing constants
IL_MOD_MDP = "mdp"               # isaaclab.envs.mdp (core)
IL_MOD_VEL = "velocity_mdp"      # isaaclab_tasks...velocity.mdp
IL_MOD_INLINE = ""                # emitted inline in compiled config


@dataclass(frozen=True)
class TaskModuleItem:
    key: str
    kind: str
    polarity: str
    title: str
    desc: str
    default: float
    min_value: float
    max_value: float
    step: float
    applicable_families: FrozenSet[str] = field(default_factory=frozenset)
    # Two-layer compatibility filters:
    #   backends   — which training engines support this function
    #                (sb3 / isaac_lab / newton)
    #   algorithms — which RL algorithms it applies to
    #                (PPO / SAC / AMP / TD3 / ALL)
    # "ALL" in algorithms means universally applicable.
    backends: FrozenSet[str] = field(default_factory=lambda: ALL_BACKENDS)
    algorithms: FrozenSet[str] = field(default_factory=lambda: frozenset({ALG_ALL}))

    # ── Isaac Lab compiler metadata ───────────────────────────────────
    # Populated only for IL rewards. The compiler reads these instead of
    # maintaining parallel hardcoded dicts.
    #
    #   il_func   — Isaac Lab function name (e.g. "track_lin_vel_xy_exp")
    #   il_module — module alias: "mdp" | "velocity_mdp" | "" (inline)
    #   il_params — extra RewTerm params template string (may contain
    #               ``{node_std}`` / ``{node_threshold}`` placeholders
    #               that the compiler substitutes from the Rewards node)
    #   il_inline — Python source for an inline function when the reward
    #               is not available in standard Isaac Lab packages.
    #               Empty string → no inline needed.
    il_func: str = ""
    il_module: str = ""
    il_params: str = ""
    il_inline: str = ""


_LOCOMOTION_FAMILIES = frozenset({"quadruped", "biped", "wheeled", "generic_locomotion"})
_LEGGED_FAMILIES = frozenset({"quadruped", "biped", "generic_locomotion"})
_ALL_FAMILIES = frozenset(
    {"quadruped", "biped", "wheeled", "manipulator", "generic_locomotion"}
)


def _reward_item(
    *,
    key: str,
    polarity: str,
    title: str,
    desc: str,
    default: float,
    min_value: float,
    max_value: float,
    step: float,
    applicable_families: FrozenSet[str] = _ALL_FAMILIES,
    backends: FrozenSet[str] = ALL_BACKENDS,
    algorithms: FrozenSet[str] = frozenset({ALG_ALL}),
    il_func: str = "",
    il_module: str = "",
    il_params: str = "",
    il_inline: str = "",
) -> TaskModuleItem:
    return TaskModuleItem(
        key=key,
        kind="reward",
        polarity=polarity,
        title=title,
        desc=desc,
        default=default,
        min_value=min_value,
        max_value=max_value,
        step=step,
        applicable_families=applicable_families,
        backends=backends,
        algorithms=algorithms,
        il_func=il_func,
        il_module=il_module,
        il_params=il_params,
        il_inline=il_inline,
    )


def _termination_item(
    *,
    key: str,
    title: str,
    desc: str,
    default: float,
    min_value: float,
    max_value: float,
    step: float,
    applicable_families: FrozenSet[str] = _ALL_FAMILIES,
    backends: FrozenSet[str] = ALL_BACKENDS,
    algorithms: FrozenSet[str] = frozenset({ALG_ALL}),
) -> TaskModuleItem:
    return TaskModuleItem(
        key=key,
        kind="termination",
        polarity="",
        title=title,
        desc=desc,
        default=default,
        min_value=min_value,
        max_value=max_value,
        step=step,
        applicable_families=applicable_families,
        backends=backends,
        algorithms=algorithms,
    )


REWARD_REGISTRY: Dict[str, TaskModuleItem] = {
    # ── SB3 rewards ── backends=sb3, algorithms as noted ──────────────
    "velocity_tracking": _reward_item(
        key="velocity_tracking",
        polarity="reward",
        title="Velocity Tracking",
        desc="Reward matching commanded forward and lateral velocity.",
        default=1.0, min_value=0.0, max_value=10.0, step=0.05,
        applicable_families=_LOCOMOTION_FAMILIES,
        backends=frozenset({BACKEND_SB3}),
        algorithms=frozenset({ALG_ALL}),
    ),
    "alive": _reward_item(
        key="alive",
        polarity="reward",
        title="Alive Bonus",
        desc="Small survival bonus that encourages staying upright.",
        default=0.5, min_value=0.0, max_value=10.0, step=0.05,
        applicable_families=_LEGGED_FAMILIES,
        backends=frozenset({BACKEND_SB3}),
        algorithms=frozenset({ALG_ALL}),
    ),
    "energy": _reward_item(
        key="energy",
        polarity="penalty",
        title="Energy Penalty",
        desc="Penalty on large actions to reduce wasteful actuation.",
        default=-0.01, min_value=-10.0, max_value=0.0, step=0.01,
        backends=frozenset({BACKEND_SB3}),
        algorithms=frozenset({ALG_ALL}),
    ),
    "yaw_tracking": _reward_item(
        key="yaw_tracking",
        polarity="reward",
        title="Yaw Tracking",
        desc="Reward matching the target yaw-rate command.",
        default=0.5, min_value=0.0, max_value=10.0, step=0.05,
        applicable_families=_LOCOMOTION_FAMILIES,
        backends=frozenset({BACKEND_SB3}),
        algorithms=frozenset({ALG_ALL}),
    ),
    "action_smoothness": _reward_item(
        key="action_smoothness",
        polarity="penalty",
        title="Action Smoothness",
        desc="Penalty for abrupt changes between consecutive actions.",
        default=-0.02, min_value=-10.0, max_value=0.0, step=0.01,
        backends=frozenset({BACKEND_SB3}),
        algorithms=frozenset({ALG_ALL}),
    ),
    "upright": _reward_item(
        key="upright",
        polarity="reward",
        title="Upright Bonus",
        desc="Reward maintaining an upright torso orientation.",
        default=0.3, min_value=0.0, max_value=10.0, step=0.05,
        applicable_families=_LEGGED_FAMILIES,
        backends=frozenset({BACKEND_SB3}),
        algorithms=frozenset({ALG_ALL}),
    ),
    "lateral_penalty": _reward_item(
        key="lateral_penalty",
        polarity="penalty",
        title="Lateral Penalty",
        desc="Penalty for sideways drift when forward tracking is desired.",
        default=-0.1, min_value=-10.0, max_value=0.0, step=0.05,
        applicable_families=_LOCOMOTION_FAMILIES,
        backends=frozenset({BACKEND_SB3}),
        algorithms=frozenset({ALG_ALL}),
    ),
    "foot_clearance": _reward_item(
        key="foot_clearance",
        polarity="reward",
        title="Foot Clearance",
        desc="Reward for lifting feet enough to avoid stumbling during swing.",
        default=0.2, min_value=0.0, max_value=10.0, step=0.05,
        applicable_families=frozenset({"quadruped", "biped"}),
        backends=frozenset({BACKEND_SB3}),
        algorithms=frozenset({ALG_ALL}),
    ),
    "slip_penalty": _reward_item(
        key="slip_penalty",
        polarity="penalty",
        title="Slip Penalty",
        desc="Penalty for excessive foot or wheel slip against the ground.",
        default=-0.1, min_value=-10.0, max_value=0.0, step=0.05,
        applicable_families=_LOCOMOTION_FAMILIES,
        backends=frozenset({BACKEND_SB3}),
        algorithms=frozenset({ALG_ALL}),
    ),
    "collision_penalty": _reward_item(
        key="collision_penalty",
        polarity="penalty",
        title="Collision Penalty",
        desc="Penalty for unwanted collisions with the environment or self.",
        default=-0.2, min_value=-10.0, max_value=0.0, step=0.05,
        backends=frozenset({BACKEND_SB3}),
        algorithms=frozenset({ALG_ALL}),
    ),
    "goal_distance": _reward_item(
        key="goal_distance",
        polarity="reward",
        title="Goal Distance",
        desc="Reward reducing distance to a target pose or position.",
        default=1.0, min_value=-5.0, max_value=10.0, step=0.1,
        applicable_families=frozenset({"manipulator"}),
        backends=frozenset({BACKEND_SB3}),
        algorithms=frozenset({ALG_ALL}),
    ),
    "grasp_success": _reward_item(
        key="grasp_success",
        polarity="reward",
        title="Grasp Success",
        desc="Sparse success bonus for stable grasp or task completion.",
        default=5.0, min_value=0.0, max_value=20.0, step=0.5,
        applicable_families=frozenset({"manipulator"}),
        backends=frozenset({BACKEND_SB3}),
        algorithms=frozenset({ALG_ALL}),
    ),
    "reference_tracking": _reward_item(
        key="reference_tracking",
        polarity="reward",
        title="Reference Tracking",
        desc="Reward for matching a reference motion trajectory keyframe via Gaussian similarity.",
        default=1.0, min_value=0.0, max_value=10.0, step=0.1,
        applicable_families=_LEGGED_FAMILIES,
        backends=frozenset({BACKEND_SB3}),
        algorithms=frozenset({ALG_PPO, ALG_AMP}),
    ),
    "base_height_penalty": _reward_item(
        key="base_height_penalty",
        polarity="penalty",
        title="Base Height Penalty",
        desc="Penalty for base height deviating from the nominal standing height (~0.32 m). "
             "Discourages crouching or over-extension during locomotion.",
        default=-1.0, min_value=-10.0, max_value=0.0, step=0.1,
        applicable_families=_LOCOMOTION_FAMILIES,
        backends=frozenset({BACKEND_SB3}),
        algorithms=frozenset({ALG_ALL}),
    ),
    "angular_rate_penalty": _reward_item(
        key="angular_rate_penalty",
        polarity="penalty",
        title="Angular Rate Penalty",
        desc="Penalty on squared roll and pitch angular rates. "
             "Reduces trunk wobble and promotes smooth, stable locomotion.",
        default=-0.05, min_value=-5.0, max_value=0.0, step=0.01,
        applicable_families=_LEGGED_FAMILIES,
        backends=frozenset({BACKEND_SB3}),
        algorithms=frozenset({ALG_ALL}),
    ),
    "foot_air_time": _reward_item(
        key="foot_air_time",
        polarity="reward",
        title="Foot Air Time",
        desc="Bonus at each foot touchdown proportional to how long the foot was airborne "
             "(capped at 0.5 s). Encourages regular, symmetric gait cycles.",
        default=0.5, min_value=0.0, max_value=5.0, step=0.05,
        applicable_families=frozenset({"quadruped", "biped"}),
        backends=frozenset({BACKEND_SB3}),
        algorithms=frozenset({ALG_ALL}),
    ),
    "joint_pose_tracking": _reward_item(
        key="joint_pose_tracking",
        polarity="reward",
        title="Joint Pose Tracking",
        desc="Gaussian reward for joint positions matching the reference frame. "
             "Falls back to GO2 default standing pose when no reference motion is loaded.",
        default=1.0, min_value=0.0, max_value=10.0, step=0.1,
        applicable_families=_LEGGED_FAMILIES,
        backends=frozenset({BACKEND_SB3}),
        algorithms=frozenset({ALG_PPO, ALG_AMP}),
    ),
    "joint_vel_tracking": _reward_item(
        key="joint_vel_tracking",
        polarity="reward",
        title="Joint Velocity Tracking",
        desc="Gaussian reward for joint velocities matching the reference (finite-diff of "
             "reference frames). Falls back to tracking zero velocity when no reference is loaded.",
        default=0.5, min_value=0.0, max_value=10.0, step=0.05,
        applicable_families=_LEGGED_FAMILIES,
        backends=frozenset({BACKEND_SB3}),
        algorithms=frozenset({ALG_PPO, ALG_AMP}),
    ),
    "foot_pos_tracking": _reward_item(
        key="foot_pos_tracking",
        polarity="reward",
        title="Foot Position Tracking",
        desc="Gaussian reward for foot Cartesian positions matching reference FK positions. "
             "Falls back to nominal standing foot positions when no reference motion is loaded.",
        default=0.5, min_value=0.0, max_value=10.0, step=0.05,
        applicable_families=frozenset({"quadruped", "biped"}),
        backends=frozenset({BACKEND_SB3}),
        algorithms=frozenset({ALG_PPO, ALG_AMP}),
    ),
}


TERMINATION_REGISTRY: Dict[str, TaskModuleItem] = {
    "fall_threshold_roll": _termination_item(
        key="fall_threshold_roll",
        title="Roll Threshold",
        desc="Terminate when roll exceeds the allowed upright margin.",
        default=1.2,
        min_value=0.2,
        max_value=3.14,
        step=0.05,
        applicable_families=_LEGGED_FAMILIES,
    ),
    "fall_threshold_pitch": _termination_item(
        key="fall_threshold_pitch",
        title="Pitch Threshold",
        desc="Terminate when pitch exceeds the allowed upright margin.",
        default=1.2,
        min_value=0.2,
        max_value=3.14,
        step=0.05,
        applicable_families=_LEGGED_FAMILIES,
    ),
    "min_height": _termination_item(
        key="min_height",
        title="Minimum Height",
        desc="Terminate when the robot base drops below this height.",
        default=0.15,
        min_value=0.05,
        max_value=0.5,
        step=0.01,
        applicable_families=_LOCOMOTION_FAMILIES,
    ),
    "max_contact_impulse": _termination_item(
        key="max_contact_impulse",
        title="Contact Impulse",
        desc="Terminate on excessive impact indicating unstable collapse.",
        default=250.0,
        min_value=50.0,
        max_value=500.0,
        step=5.0,
        applicable_families=_LOCOMOTION_FAMILIES,
    ),
    "timeout": _termination_item(
        key="timeout",
        title="Timeout",
        desc="Terminate when the task exceeds its allowed duration.",
        default=1000.0,
        min_value=10.0,
        max_value=100000.0,
        step=10.0,
    ),
    "joint_limit_violation": _termination_item(
        key="joint_limit_violation",
        title="Joint Limit",
        desc="Terminate when joints exceed safe configured limits. "
             "Threshold is a count: 3 = terminate when 3 or more joints exceed their range.",
        default=3.0,
        min_value=1.0,
        max_value=12.0,
        step=1.0,
        applicable_families=_ALL_FAMILIES,
    ),
    "self_collision": _termination_item(
        key="self_collision",
        title="Self Collision",
        desc="Terminate when forbidden self-collision is detected.",
        default=1.0,
        min_value=0.0,
        max_value=5.0,
        step=0.1,
        applicable_families=frozenset({"manipulator"}),
    ),
}


def reward_registry() -> Dict[str, TaskModuleItem]:
    return dict(REWARD_REGISTRY)


def termination_registry() -> Dict[str, TaskModuleItem]:
    return dict(TERMINATION_REGISTRY)


def default_reward_terms() -> Dict[str, float]:
    return {
        key: REWARD_REGISTRY[key].default
        for key in ("velocity_tracking", "alive", "energy")
    }


def default_termination_conditions() -> Dict[str, float]:
    return {
        key: TERMINATION_REGISTRY[key].default
        for key in (
            "fall_threshold_roll",
            "fall_threshold_pitch",
            "min_height",
            "max_contact_impulse",
            "joint_limit_violation",
        )
    }


# ═══════════════════════════════════════════════════════════════════════════
# Isaac Lab registries
# ═══════════════════════════════════════════════════════════════════════════

# ── Inline reward implementations (not in standard Isaac Lab) ─────────
# These are emitted into the compiled config file verbatim.

_INLINE_ENERGY = '''
def _unitport_energy(env, asset_cfg=SceneEntityCfg("robot")):
    """Penalize energy used by joints (|vel| * |torque|)."""
    import torch
    asset = env.scene[asset_cfg.name]
    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)
'''

_INLINE_FOOT_CLEARANCE = '''
def _unitport_foot_clearance_reward(env, asset_cfg=SceneEntityCfg("robot"),
                                     target_height=0.1, std=0.05, tanh_mult=2.0):
    """Reward swinging feet for clearing a specified height."""
    import torch
    asset = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(
        asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(
        tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)
'''

_INLINE_FEET_GAIT = '''
def _unitport_feet_gait(env, period=0.8, offset=None, sensor_cfg=None,
                         threshold=0.5, command_name=None):
    """Enforce periodic gait patterns for legged robots."""
    import torch
    if offset is None:
        offset = [0.0, 0.5, 0.5, 0.0]
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0
    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    phases = []
    for off in offset:
        phases.append((global_phase + off) % 1.0)
    leg_phase = torch.cat(phases, dim=-1)
    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(len(sensor_cfg.body_ids)):
        is_stance = leg_phase[:, i] < threshold
        reward += ~(is_stance ^ is_contact[:, i])
    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        reward *= cmd_norm > 0.1
    return reward
'''

_INLINE_FEET_SLIDE = '''
def _unitport_feet_slide(env, sensor_cfg=None, asset_cfg=SceneEntityCfg("robot")):
    """Penalize feet sliding on the ground while in contact."""
    import torch
    import isaaclab.utils.math as math_utils
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(
        dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]
    cur_footvel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[:, :].unsqueeze(1)
    footvel_body = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footvel_body[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel[:, i, :])
    lateral_vel = torch.sqrt(torch.sum(torch.square(footvel_body[:, :, :2]), dim=2)).view(env.num_envs, -1)
    return torch.sum(lateral_vel * contacts, dim=1)
'''

_INLINE_BASE_HEIGHT = '''
def _unitport_base_height_l2(env, target_height=0.34, asset_cfg=SceneEntityCfg("robot"),
                              sensor_cfg=None):
    """Penalize base height deviation from target using L2 squared kernel."""
    import torch
    asset = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor = env.scene[sensor_cfg.name]
        ray_hits = sensor.data.ray_hits_w[..., 2]
        if torch.isnan(ray_hits).any() or torch.isinf(ray_hits).any():
            adjusted = asset.data.root_link_pos_w[:, 2]
        else:
            adjusted = target_height + torch.mean(ray_hits, dim=1)
    else:
        adjusted = target_height
    return torch.square(asset.data.root_pos_w[:, 2] - adjusted)
'''

_INLINE_UNDESIRED_CONTACTS = '''
def _unitport_undesired_contacts(env, threshold=1.0, sensor_cfg=None):
    """Penalize undesired contacts (non-foot bodies touching ground)."""
    import torch
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(
        torch.norm(net_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=1).float()
'''


# ---------------------------------------------------------------------------
# P2.1 — Walk These Ways parameterised gait rewards
# ---------------------------------------------------------------------------
# These three reward terms read from a :class:`UniformGaitCommand` term
# which the IL compiler emits inline when the canvas Training Commands
# node has gait_enabled. They use ``env.command_manager.get_term(...)``
# to fetch the live phase clock + commanded body/step height per env
# and shape the reward so the policy learns to follow the preset dial.

_INLINE_TRACK_GAIT_PHASE = '''
def _unitport_track_gait_phase(env, command_name="gait_command", sensor_cfg=None):
    """Reward foot contact matching the expected stance/swing phase.

    Walk These Ways §3: each foot has a local phase in [0, 1); it
    should be in stance (on the ground) when phase < 0.5, in swing
    otherwise. This reward is the mean per-foot agreement between
    expected stance and actual contact.
    """
    import torch
    term = env.command_manager.get_term(command_name)
    per_foot = term.per_foot_phase()                            # (n, 4)
    expected_stance = (per_foot < 0.5).float()
    sensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    contact = (torch.norm(forces, dim=-1) > 1.0).float()         # (n, n_feet)
    n_match = min(contact.shape[1], 4)
    match = 1.0 - torch.abs(
        expected_stance[:, :n_match] - contact[:, :n_match]
    )
    return match.mean(dim=-1)
'''

_INLINE_TRACK_BODY_HEIGHT_CMD = '''
def _unitport_track_body_height_cmd(env, command_name="gait_command",
                                     asset_cfg=SceneEntityCfg("robot"), std=0.05):
    """Exponential reward for base height matching commanded body_height."""
    import torch
    term = env.command_manager.get_term(command_name)
    target = term.command[:, 5]
    asset = env.scene[asset_cfg.name]
    height = asset.data.root_pos_w[:, 2]
    error = torch.square(height - target)
    return torch.exp(-error / (std * std))
'''

_INLINE_TRACK_SWING_HEIGHT_CMD = '''
def _unitport_track_swing_height_cmd(env, command_name="gait_command",
                                      asset_cfg=SceneEntityCfg("robot"),
                                      std=0.02):
    """Reward swing-phase foot apex matching commanded step_height."""
    import torch
    term = env.command_manager.get_term(command_name)
    target = term.command[:, 6:7]                                # (n, 1)
    per_foot = term.per_foot_phase()                             # (n, 4)
    is_swing = (per_foot >= 0.5).float()
    asset = env.scene[asset_cfg.name]
    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    n_match = min(foot_z.shape[1], 4)
    foot_z = foot_z[:, :n_match]
    is_swing = is_swing[:, :n_match]
    error = torch.square(foot_z - target)
    reward = torch.exp(-error / (std * std)) * is_swing
    return reward.sum(dim=-1) / is_swing.sum(dim=-1).clamp(min=1.0)
'''

_IL = frozenset({BACKEND_ISAAC})

IL_REWARD_REGISTRY: Dict[str, TaskModuleItem] = {
    # ═══════════════════════════════════════════════════════════════════
    # Isaac Lab reward functions — each entry is the SOLE source of truth
    # for: UI display, compiler function mapping, extra params, and inline
    # implementation. Adding a reward here is ALL that's needed.
    # ═══════════════════════════════════════════════════════════════════

    # ── Velocity tracking (task rewards) ──────────────────────────────
    "track_lin_vel_xy": _reward_item(
        key="track_lin_vel_xy",
        polarity="reward",
        title="Track Lin Vel XY",
        desc="Exponential tracking reward for commanded XY linear velocity.",
        default=1.5, min_value=0.0, max_value=40.0, step=0.1,
        applicable_families=_LOCOMOTION_FAMILIES,
        backends=_IL, algorithms=frozenset({ALG_ALL}),
        il_func="track_lin_vel_xy_exp", il_module=IL_MOD_MDP,
        il_params='"std": {node_std}, "command_name": "base_velocity"',
    ),
    "track_ang_vel_z": _reward_item(
        key="track_ang_vel_z",
        polarity="reward",
        title="Track Ang Vel Z",
        desc="Exponential tracking reward for commanded yaw angular velocity.",
        default=0.75, min_value=0.0, max_value=10.0, step=0.05,
        applicable_families=_LOCOMOTION_FAMILIES,
        backends=_IL, algorithms=frozenset({ALG_ALL}),
        il_func="track_ang_vel_z_exp", il_module=IL_MOD_MDP,
        il_params='"std": {node_std}, "command_name": "base_velocity"',
    ),

    # ── Base velocity penalties ───────────────────────────────────────
    "lin_vel_z_penalty": _reward_item(
        key="lin_vel_z_penalty",
        polarity="penalty",
        title="Lin Vel Z Penalty",
        desc="L2 penalty on vertical linear velocity to discourage bouncing.",
        default=-2.0, min_value=-20.0, max_value=0.0, step=0.1,
        applicable_families=_LOCOMOTION_FAMILIES,
        backends=_IL, algorithms=frozenset({ALG_ALL}),
        il_func="lin_vel_z_l2", il_module=IL_MOD_MDP,
    ),
    "ang_vel_xy_penalty": _reward_item(
        key="ang_vel_xy_penalty",
        polarity="penalty",
        title="Ang Vel XY Penalty",
        desc="L2 penalty on roll/pitch angular velocity.",
        default=-0.05, min_value=-5.0, max_value=0.0, step=0.005,
        applicable_families=_LOCOMOTION_FAMILIES,
        backends=_IL, algorithms=frozenset({ALG_ALL}),
        il_func="ang_vel_xy_l2", il_module=IL_MOD_MDP,
    ),

    # ── Joint / actuation penalties ───────────────────────────────────
    "joint_torque_penalty": _reward_item(
        key="joint_torque_penalty",
        polarity="penalty",
        title="Joint Torque Penalty",
        desc="L2 penalty on joint torques for energy efficiency.",
        default=-0.0002, min_value=-0.01, max_value=0.0, step=0.00005,
        applicable_families=_ALL_FAMILIES,
        backends=_IL, algorithms=frozenset({ALG_ALL}),
        il_func="joint_torques_l2", il_module=IL_MOD_MDP,
    ),
    "joint_accel_penalty": _reward_item(
        key="joint_accel_penalty",
        polarity="penalty",
        title="Joint Accel Penalty",
        desc="L2 penalty on joint accelerations for smooth motion.",
        default=-2.5e-7, min_value=-0.001, max_value=0.0, step=1e-7,
        applicable_families=_ALL_FAMILIES,
        backends=_IL, algorithms=frozenset({ALG_ALL}),
        il_func="joint_acc_l2", il_module=IL_MOD_MDP,
    ),
    "action_rate_penalty": _reward_item(
        key="action_rate_penalty",
        polarity="penalty",
        title="Action Rate Penalty",
        desc="L2 penalty on action rate of change for smooth control.",
        default=-0.05, min_value=-1.0, max_value=0.0, step=0.005,
        applicable_families=_ALL_FAMILIES,
        backends=_IL, algorithms=frozenset({ALG_ALL}),
        il_func="action_rate_l2", il_module=IL_MOD_MDP,
    ),
    "joint_vel_penalty": _reward_item(
        key="joint_vel_penalty",
        polarity="penalty",
        title="Joint Vel Penalty",
        desc="L2 penalty on joint velocities — discourages overly fast joint motion.",
        default=-0.001, min_value=-1.0, max_value=0.0, step=0.0005,
        applicable_families=_ALL_FAMILIES,
        backends=_IL, algorithms=frozenset({ALG_ALL}),
        il_func="joint_vel_l2", il_module=IL_MOD_MDP,
        il_params='"asset_cfg": SceneEntityCfg("robot")',
    ),
    "energy_penalty": _reward_item(
        key="energy_penalty",
        polarity="penalty",
        title="Energy Penalty",
        desc="L2 penalty on torque × velocity product — minimises mechanical energy expenditure.",
        default=-2e-5, min_value=-0.01, max_value=0.0, step=1e-5,
        applicable_families=_ALL_FAMILIES,
        backends=_IL, algorithms=frozenset({ALG_ALL}),
        il_func="_unitport_energy", il_module=IL_MOD_INLINE,
        il_params='"asset_cfg": SceneEntityCfg("robot")',
        il_inline=_INLINE_ENERGY,
    ),

    # ── Orientation / posture ─────────────────────────────────────────
    "flat_orientation": _reward_item(
        key="flat_orientation",
        polarity="penalty",
        title="Flat Orientation",
        desc="L2 penalty on projected gravity deviation from vertical — keeps the base level.",
        default=-5.0, min_value=-20.0, max_value=0.0, step=0.1,
        applicable_families=_LOCOMOTION_FAMILIES,
        backends=_IL, algorithms=frozenset({ALG_ALL}),
        il_func="flat_orientation_l2", il_module=IL_MOD_MDP,
    ),
    "base_height": _reward_item(
        key="base_height",
        polarity="penalty",
        title="Base Height",
        desc="L2 penalty on base height deviating from a target standing height. "
             "Needs params: target_height (default 0.34 for quadruped, 0.78 for biped).",
        default=-10.0, min_value=-50.0, max_value=0.0, step=0.5,
        applicable_families=_LOCOMOTION_FAMILIES,
        backends=_IL, algorithms=frozenset({ALG_ALL}),
        il_func="_unitport_base_height_l2", il_module=IL_MOD_INLINE,
        il_params='"target_height": 0.34, "asset_cfg": SceneEntityCfg("robot")',
        il_inline=_INLINE_BASE_HEIGHT,
    ),
    "alive_reward": _reward_item(
        key="alive_reward",
        polarity="reward",
        title="Alive Bonus",
        desc="Constant per-step survival bonus — encourages the policy to keep the robot alive.",
        default=0.15, min_value=0.0, max_value=5.0, step=0.05,
        applicable_families=_LOCOMOTION_FAMILIES,
        backends=_IL, algorithms=frozenset({ALG_ALL}),
        il_func="is_alive", il_module=IL_MOD_MDP,
    ),

    # ── Feet / gait ───────────────────────────────────────────────────
    "feet_air_time": _reward_item(
        key="feet_air_time",
        polarity="reward",
        title="Feet Air Time",
        desc="Reward for maintaining contact schedule (gait pattern).",
        default=0.125, min_value=0.0, max_value=5.0, step=0.025,
        applicable_families=_LEGGED_FAMILIES,
        backends=_IL, algorithms=frozenset({ALG_ALL}),
        il_func="feet_air_time", il_module=IL_MOD_VEL,
        il_params='"threshold": {node_threshold}, "command_name": "base_velocity", '
                  '"sensor_cfg": SceneEntityCfg("contact_forces", body_names={ir:feet})',
    ),
    "gait": _reward_item(
        key="gait",
        polarity="reward",
        title="Gait Rhythm",
        desc="Periodic gait reward — encourages regular alternating footfalls with a target "
             "period and phase offset. Needs params: period (s), offset (per-foot phase array).",
        default=0.5, min_value=0.0, max_value=5.0, step=0.05,
        applicable_families=_LEGGED_FAMILIES,
        backends=_IL, algorithms=frozenset({ALG_PPO, ALG_AMP}),
        il_func="_unitport_feet_gait", il_module=IL_MOD_INLINE,
        il_params='"period": 0.8, "offset": [0.0, 0.5, 0.5, 0.0], '
                  '"sensor_cfg": SceneEntityCfg("contact_forces", body_names={ir:feet}), '
                  '"threshold": 0.5, "command_name": "base_velocity"',
        il_inline=_INLINE_FEET_GAIT,
    ),
    # ── P2.1 — Walk These Ways parameterised gait (reads from the
    #    UniformGaitCommand term emitted by the IL compiler when the
    #    Training Commands node has gait_enabled). ──
    "track_gait_phase": _reward_item(
        key="track_gait_phase",
        polarity="reward",
        title="Track Gait Phase",
        desc="Reward foot contact matching the expected stance/swing phase from a Walk These "
             "Ways gait command term. Requires Training Commands with gait_enabled.",
        default=0.5, min_value=0.0, max_value=5.0, step=0.05,
        applicable_families=_LEGGED_FAMILIES,
        backends=_IL, algorithms=frozenset({ALG_PPO, ALG_AMP}),
        il_func="_unitport_track_gait_phase", il_module=IL_MOD_INLINE,
        il_params='"command_name": "gait_command", '
                  '"sensor_cfg": SceneEntityCfg("contact_forces", body_names={ir:feet})',
        il_inline=_INLINE_TRACK_GAIT_PHASE,
    ),
    "track_body_height_cmd": _reward_item(
        key="track_body_height_cmd",
        polarity="reward",
        title="Track Body Height Cmd",
        desc="Exponential reward for base height matching the commanded body_height. Pairs with "
             "the Walk These Ways gait command — tightens body_height tracking during training.",
        default=0.3, min_value=0.0, max_value=5.0, step=0.05,
        applicable_families=_LEGGED_FAMILIES,
        backends=_IL, algorithms=frozenset({ALG_PPO, ALG_AMP}),
        il_func="_unitport_track_body_height_cmd", il_module=IL_MOD_INLINE,
        il_params='"command_name": "gait_command", "std": 0.05',
        il_inline=_INLINE_TRACK_BODY_HEIGHT_CMD,
    ),
    "track_swing_height_cmd": _reward_item(
        key="track_swing_height_cmd",
        polarity="reward",
        title="Track Swing Height Cmd",
        desc="Reward swing-phase foot apex height matching the commanded step_height. Requires a "
             "gait command term; feet list auto-resolved via IR mapping.",
        default=0.3, min_value=0.0, max_value=5.0, step=0.05,
        applicable_families=_LEGGED_FAMILIES,
        backends=_IL, algorithms=frozenset({ALG_PPO, ALG_AMP}),
        il_func="_unitport_track_swing_height_cmd", il_module=IL_MOD_INLINE,
        il_params='"command_name": "gait_command", '
                  '"asset_cfg": SceneEntityCfg("robot", body_names={ir:feet}), '
                  '"std": 0.02',
        il_inline=_INLINE_TRACK_SWING_HEIGHT_CMD,
    ),
    "feet_clearance": _reward_item(
        key="feet_clearance",
        polarity="reward",
        title="Feet Clearance",
        desc="Reward for swing-foot height reaching a target clearance above ground. "
             "Needs params: target_height (m, default 0.1).",
        default=1.0, min_value=0.0, max_value=10.0, step=0.1,
        applicable_families=_LEGGED_FAMILIES,
        backends=_IL, algorithms=frozenset({ALG_ALL}),
        il_func="_unitport_foot_clearance_reward", il_module=IL_MOD_INLINE,
        il_params='"asset_cfg": SceneEntityCfg("robot"), "target_height": 0.1, '
                  '"std": 0.05, "tanh_mult": 2.0',
        il_inline=_INLINE_FOOT_CLEARANCE,
    ),
    "feet_slide": _reward_item(
        key="feet_slide",
        polarity="penalty",
        title="Feet Slide",
        desc="L2 penalty on foot velocity while in contact — discourages sliding/skating.",
        default=-0.2, min_value=-5.0, max_value=0.0, step=0.05,
        applicable_families=_LEGGED_FAMILIES,
        backends=_IL, algorithms=frozenset({ALG_ALL}),
        il_func="_unitport_feet_slide", il_module=IL_MOD_INLINE,
        il_params='"sensor_cfg": SceneEntityCfg("contact_forces", body_names={ir:feet}), '
                  '"asset_cfg": SceneEntityCfg("robot")',
        il_inline=_INLINE_FEET_SLIDE,
    ),

    # ── Contact / safety ──────────────────────────────────────────────
    "undesired_contacts": _reward_item(
        key="undesired_contacts",
        polarity="penalty",
        title="Undesired Contacts",
        desc="Penalty when non-foot bodies (torso, thighs, shoulders) make ground contact. "
             "Needs params: sensor_cfg with body_names regex.",
        default=-1.0, min_value=-10.0, max_value=0.0, step=0.1,
        applicable_families=_LOCOMOTION_FAMILIES,
        backends=_IL, algorithms=frozenset({ALG_ALL}),
        il_func="_unitport_undesired_contacts", il_module=IL_MOD_INLINE,
        il_params='"threshold": 1.0, '
                  '"sensor_cfg": SceneEntityCfg("contact_forces", body_names={ir:thighs_hips_base})',
        il_inline=_INLINE_UNDESIRED_CONTACTS,
    ),

    # ── Joint limits ──────────────────────────────────────────────────
    "dof_pos_limits": _reward_item(
        key="dof_pos_limits",
        polarity="penalty",
        title="Joint Pos Limits",
        desc="Penalty when joint positions approach or exceed soft limits — "
             "keeps joints within safe operating range.",
        default=-5.0, min_value=-20.0, max_value=0.0, step=0.5,
        applicable_families=_ALL_FAMILIES,
        backends=_IL, algorithms=frozenset({ALG_ALL}),
        il_func="joint_pos_limits", il_module=IL_MOD_MDP,
        il_params='"asset_cfg": SceneEntityCfg("robot")',
    ),
}

IL_OBS_REGISTRY: Dict[str, TaskModuleItem] = {
    "base_lin_vel": TaskModuleItem(
        key="base_lin_vel",
        kind="observation",
        polarity="",
        title="Base Lin Vel",
        desc="Base linear velocity in body frame (3D).",
        default=1.0,
        min_value=0.0,
        max_value=1.0,
        step=1.0,
        applicable_families=_ALL_FAMILIES,
    ),
    "base_ang_vel": TaskModuleItem(
        key="base_ang_vel",
        kind="observation",
        polarity="",
        title="Base Ang Vel",
        desc="Base angular velocity in body frame (3D).",
        default=1.0,
        min_value=0.0,
        max_value=1.0,
        step=1.0,
        applicable_families=_ALL_FAMILIES,
    ),
    "projected_gravity": TaskModuleItem(
        key="projected_gravity",
        kind="observation",
        polarity="",
        title="Projected Gravity",
        desc="Gravity vector projected into body frame (3D).",
        default=1.0,
        min_value=0.0,
        max_value=1.0,
        step=1.0,
        applicable_families=_ALL_FAMILIES,
    ),
    "velocity_command": TaskModuleItem(
        key="velocity_command",
        kind="observation",
        polarity="",
        title="Velocity Command",
        desc="Commanded velocity target (3D: vx, vy, wz).",
        default=1.0,
        min_value=0.0,
        max_value=1.0,
        step=1.0,
        applicable_families=_LOCOMOTION_FAMILIES,
    ),
    "joint_pos": TaskModuleItem(
        key="joint_pos",
        kind="observation",
        polarity="",
        title="Joint Positions",
        desc="Joint position readings, relative to default pose (N-dim).",
        default=1.0,
        min_value=0.0,
        max_value=1.0,
        step=1.0,
        applicable_families=_ALL_FAMILIES,
    ),
    "joint_vel": TaskModuleItem(
        key="joint_vel",
        kind="observation",
        polarity="",
        title="Joint Velocities",
        desc="Joint velocity readings (N-dim).",
        default=1.0,
        min_value=0.0,
        max_value=1.0,
        step=1.0,
        applicable_families=_ALL_FAMILIES,
    ),
    "last_action": TaskModuleItem(
        key="last_action",
        kind="observation",
        polarity="",
        title="Last Action",
        desc="Previous policy output / action applied (N-dim).",
        default=1.0,
        min_value=0.0,
        max_value=1.0,
        step=1.0,
        applicable_families=_ALL_FAMILIES,
    ),
    "height_scan": TaskModuleItem(
        key="height_scan",
        kind="observation",
        polarity="",
        title="Height Scan",
        desc="Terrain height scanner readings around each foot (M-dim).",
        default=1.0,
        min_value=0.0,
        max_value=1.0,
        step=1.0,
        applicable_families=_LEGGED_FAMILIES,
    ),
}


def il_reward_registry() -> Dict[str, TaskModuleItem]:
    return dict(IL_REWARD_REGISTRY)


def il_obs_registry() -> Dict[str, TaskModuleItem]:
    return dict(IL_OBS_REGISTRY)


def default_il_reward_terms() -> Dict[str, float]:
    return {k: IL_REWARD_REGISTRY[k].default for k in IL_REWARD_REGISTRY}


def default_il_obs_terms() -> Dict[str, float]:
    return {
        k: IL_OBS_REGISTRY[k].default
        for k in (
            "base_lin_vel", "base_ang_vel", "projected_gravity",
            "velocity_command", "joint_pos", "joint_vel", "last_action",
        )
    }


# ─── Isaac Lab termination registry ─────────────────────────────────────────
# These mirror the DoneTerm entries the Isaac Lab compiler used to build from
# standalone toggle params (enable_timeout / enable_illegal_contact /
# enable_base_height). Each item carries a single float threshold so it fits
# the unified _RegistryModuleEditor row layout, exactly like the SB3
# termination registry. illegal_contact's bodies regex list is kept as a
# separate hidden node parameter (illegal_contact_bodies) — the items list
# only owns the contact-force threshold.
IL_TERMINATION_REGISTRY: Dict[str, TaskModuleItem] = {
    "time_out": _termination_item(
        key="time_out",
        title="Episode Timeout",
        desc="Terminate when episode wall-clock duration (s) exceeds this limit. "
             "Mapped to mdp.time_out in the compiled Isaac Lab task.",
        default=20.0,
        min_value=1.0,
        max_value=300.0,
        step=0.5,
        applicable_families=_ALL_FAMILIES,
    ),
    "illegal_contact": _termination_item(
        key="illegal_contact",
        title="Illegal Contact",
        desc="Terminate when net contact force on the configured bodies "
             "exceeds this Newton threshold. Body regex list is configured "
             "via the node's illegal_contact_bodies parameter.",
        default=1.0,
        min_value=0.1,
        max_value=200.0,
        step=0.1,
        applicable_families=_LOCOMOTION_FAMILIES,
    ),
    "base_height": _termination_item(
        key="base_height",
        title="Base Height",
        desc="Terminate when the robot base drops below this minimum height (m).",
        default=0.2,
        min_value=0.05,
        max_value=1.0,
        step=0.01,
        applicable_families=_LOCOMOTION_FAMILIES,
    ),
}


def il_termination_registry() -> Dict[str, TaskModuleItem]:
    return dict(IL_TERMINATION_REGISTRY)


def default_il_termination_conditions() -> Dict[str, float]:
    """Default IL termination items for a fresh canvas — keep timeout and
    illegal-contact on by default to mirror the previous toggle defaults
    (enable_timeout=true, enable_illegal_contact=true, enable_base_height=false).
    """
    return {
        "time_out": IL_TERMINATION_REGISTRY["time_out"].default,
        "illegal_contact": IL_TERMINATION_REGISTRY["illegal_contact"].default,
    }
