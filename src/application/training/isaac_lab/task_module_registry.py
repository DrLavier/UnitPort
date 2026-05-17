from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Iterator, List, Optional, Tuple


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


# ── SB3 inline reward implementations (MuJoCo / numpy) ──────────────
# These are displayed in the Reward Editor for user inspection/override.
# They are NOT emitted by the IL compiler — they document the SB3 env's
# _compute_reward() logic so users can see and modify every function.

_SB3_INLINE_VELOCITY_TRACKING = '''\
def velocity_tracking(lin_vel_x, lin_vel_y, vx_tgt, vy_tgt):
    """Exponential tracking reward for commanded XY velocity."""
    r_vx = exp(-((lin_vel_x - vx_tgt) ** 2) / 0.25)
    r_vy = exp(-((lin_vel_y - vy_tgt) ** 2) / 0.25)
    return r_vx + 0.5 * r_vy
'''

_SB3_INLINE_YAW_TRACKING = '''\
def yaw_tracking(ang_vel_z, yaw_tgt):
    """Exponential tracking reward for commanded yaw rate."""
    return exp(-((ang_vel_z - yaw_tgt) ** 2) / 0.25)
'''

_SB3_INLINE_ALIVE = '''\
def alive():
    """Constant 1.0 survival bonus per step."""
    return 1.0
'''

_SB3_INLINE_ENERGY = '''\
def energy(qvel_joints, torque_joints):
    """Mechanical power: sum(|joint_vel| * |torque|).

    Uses qfrc_actuator (post-step MuJoCo forces) when available,
    otherwise falls back to PD-computed torques:
        tau = Kp * (q_des - q) - Kd * qdot
    """
    return sum(abs(qvel_joints) * abs(torque_joints))
'''

_SB3_INLINE_ACTION_SMOOTHNESS = '''\
def action_smoothness(action, prev_action):
    """L2 penalty on consecutive action difference (reduces jitter)."""
    return sum((action - prev_action) ** 2)
'''

_SB3_INLINE_UPRIGHT = '''\
def upright(projected_gravity):
    """Reward for keeping the torso upright (gravity z-component)."""
    return clip(-projected_gravity[2], 0.0, 1.0)
'''

_SB3_INLINE_LATERAL_PENALTY = '''\
def lateral_penalty(lin_vel_y, vy_tgt):
    """Penalty for sideways drift from commanded lateral velocity."""
    return abs(lin_vel_y - vy_tgt)
'''

_SB3_INLINE_FOOT_CLEARANCE = '''\
def foot_clearance(foot_heights):
    """Reward for average foot height above ground.

    Linearly maps [0.02m, 0.20m] -> [0, 1].
    """
    avg_height = mean(foot_heights)
    return clip((avg_height - 0.02) / 0.18, 0.0, 1.0)
'''

_SB3_INLINE_SLIP_PENALTY = '''\
def slip_penalty(foot_body_ids, xpos, cvel):
    """Penalty for foot lateral velocity when foot is near ground.

    For each foot with height < 0.08m, accumulates lateral speed.
    Returns mean slip speed normalised to [0, 1].
    """
    slip_speeds = []
    for bid in foot_body_ids:
        if xpos[bid][2] > 0.08:
            continue
        lin_vel = cvel[bid][3:6]
        slip_speeds.append(norm(lin_vel[:2]))
    return clip(mean(slip_speeds), 0.0, 1.0) if slip_speeds else 0.0
'''

_SB3_INLINE_FEET_SLIDE = '''\
def feet_slide(foot_body_ids, contacts, root_vel):
    """Contact-force-based penalty on foot lateral velocity while grounded.

    contact_indicator * ||foot_vel_xy - root_vel_xy||^2
    Contact detection uses MuJoCo contact pairs (binary, not force magnitude).
    """
    cost = 0.0
    for bid in contacted_foot_bodies:
        rel_vel = body_vel[bid][:2] - root_vel[:2]
        cost += rel_vel[0]**2 + rel_vel[1]**2
    return cost
'''

_SB3_INLINE_COLLISION_PENALTY = '''\
def collision_penalty(base_body_id, cfrc_ext):
    """Penalty proportional to external contact force on the base body.

    Normalised by 150N to [0, 1] range.
    """
    force_mag = norm(cfrc_ext[base_body_id][:3])
    return clip(force_mag / 150.0, 0.0, 1.0)
'''

_SB3_INLINE_GOAL_DISTANCE = '''\
def goal_distance(current_pos, target_pos):
    """Reward for reducing distance to a target position.

    Returns task-specific success score.
    """
    return task_success_score()
'''

_SB3_INLINE_GRASP_SUCCESS = '''\
def grasp_success(success_streak):
    """Sparse binary reward for stable grasp or task completion."""
    return 1.0 if success_streak > 0 else 0.0
'''

_SB3_INLINE_BASE_HEIGHT_PENALTY = '''\
def base_height_penalty(base_height, target_height=0.32):
    """Squared deviation from nominal standing height."""
    return (base_height - target_height) ** 2
'''

_SB3_INLINE_ANGULAR_RATE_PENALTY = '''\
def angular_rate_penalty(roll_rate, pitch_rate):
    """L2 penalty on roll and pitch angular rates."""
    return roll_rate ** 2 + pitch_rate ** 2
'''

_SB3_INLINE_FOOT_AIR_TIME = '''\
def foot_air_time(foot_body_ids, dt):
    """Landing bonus proportional to swing-phase air duration.

    At each step, for every tracked foot:
    - Height > 0.05m: foot is in swing, increment air counter.
    - Height <= 0.05m: foot is in stance.
      - On touchdown (swing->stance transition):
        bonus = clip(air_steps * dt, 0, 0.5)
    Returns mean bonus across all feet, in [0, 0.5].
    """
    score = 0.0
    for i, bid in enumerate(foot_body_ids):
        h = xpos[bid][2]
        in_air = h > 0.05
        if not in_air and was_in_air[i]:
            air_duration = air_steps[i] * dt
            score += clip(air_duration, 0.0, 0.5)
    return score / n_feet
'''

_SB3_INLINE_REFERENCE_TRACKING = '''\
def reference_tracking(cur_joints, ref_joints, sigma=5.0):
    """Gaussian similarity between current and reference joint positions.

    score = weight * exp(-sigma * ||q_cur - q_ref||^2)
    """
    sq_err = sum((cur_joints - ref_joints) ** 2)
    return exp(-sigma * sq_err)
'''

_SB3_INLINE_JOINT_POSE_TRACKING = '''\
def joint_pose_tracking(cur_joints, ref_joints):
    """Gaussian reward for joint positions matching reference.

    score = exp(-5.0 * ||q_cur - q_ref||^2)
    Falls back to default standing pose when no reference is loaded.
    """
    sq_err = sum((cur_joints - ref_joints) ** 2)
    return exp(-5.0 * sq_err)
'''

_SB3_INLINE_JOINT_VEL_TRACKING = '''\
def joint_vel_tracking(cur_vel, ref_vel):
    """Gaussian reward for joint velocities matching reference.

    score = exp(-0.5 * ||qdot_cur - qdot_ref||^2)
    Falls back to zero velocity when no reference is loaded.
    """
    sq_err = sum((cur_vel - ref_vel) ** 2)
    return exp(-0.5 * sq_err)
'''

_SB3_INLINE_FOOT_POS_TRACKING = '''\
def foot_pos_tracking(foot_positions, ref_foot_positions):
    """Gaussian reward for foot Cartesian positions matching reference FK.

    score = exp(-10.0 * mean(||foot_pos - foot_ref||^2))
    Falls back to nominal standing foot positions when no reference is loaded.
    """
    sq_errs = [sum((fp - rp) ** 2) for fp, rp in zip(foot_positions, ref_foot_positions)]
    return exp(-10.0 * mean(sq_errs))
'''

_SB3_INLINE_LIN_VEL_Z_PENALTY = '''\
def lin_vel_z_penalty(qvel_z):
    """L2 penalty on vertical linear velocity (discourages bouncing)."""
    return qvel_z ** 2
'''

_SB3_INLINE_JOINT_TORQUE_PENALTY = '''\
def joint_torque_penalty(torque_joints):
    """L2 penalty on applied joint torques.

    Uses qfrc_actuator (post-step joint-space actuator forces).
    """
    return sum(torque_joints ** 2)
'''

_SB3_INLINE_JOINT_ACCEL_PENALTY = '''\
def joint_accel_penalty(qacc_joints):
    """L2 penalty on joint accelerations."""
    return sum(qacc_joints ** 2)
'''

_SB3_INLINE_JOINT_VEL_PENALTY = '''\
def joint_vel_penalty(qvel_joints):
    """L2 penalty on joint velocities."""
    return sum(qvel_joints ** 2)
'''

_SB3_INLINE_DOF_POS_LIMITS = '''\
def dof_pos_limits(joint_positions, joint_ranges, soft_margin=0.05):
    """Penalty for joints approaching or exceeding soft limits.

    For each limited joint, computes excess beyond 95% of range on both
    sides and sums squared violations.
    """
    cost = 0.0
    for pos, (lo, hi) in zip(joint_positions, joint_ranges):
        span = hi - lo
        margin = span * soft_margin
        below = max(0.0, (lo + margin) - pos)
        above = max(0.0, pos - (hi - margin))
        cost += below ** 2 + above ** 2
    return cost
'''

_SB3_INLINE_UNDESIRED_CONTACTS = '''\
def undesired_contacts(contacts, foot_body_ids):
    """Count non-foot robot bodies in contact with the ground.

    Iterates MuJoCo contacts; skips foot bodies and world-world pairs.
    Returns float count for multiplying by a negative penalty weight.
    """
    hit = set()
    for c in contacts:
        b1, b2 = geom_bodyid[c.geom1], geom_bodyid[c.geom2]
        if b1 == 0 and b2 not in foot_body_ids:
            hit.add(b2)
        elif b2 == 0 and b1 not in foot_body_ids:
            hit.add(b1)
    return float(len(hit))
'''

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
        il_inline=_SB3_INLINE_VELOCITY_TRACKING,
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
        il_inline=_SB3_INLINE_ALIVE,
    ),
    "energy": _reward_item(
        key="energy",
        polarity="penalty",
        title="Energy Penalty",
        desc="Penalty on mechanical power: Σ|joint_vel|·|torque|. "
             "Consistent with Isaac Lab energy formulation.",
        default=-0.0001, min_value=-0.1, max_value=0.0, step=0.00005,
        backends=frozenset({BACKEND_SB3}),
        algorithms=frozenset({ALG_ALL}),
        il_inline=_SB3_INLINE_ENERGY,
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
        il_inline=_SB3_INLINE_YAW_TRACKING,
    ),
    "action_smoothness": _reward_item(
        key="action_smoothness",
        polarity="penalty",
        title="Action Smoothness",
        desc="Penalty for abrupt changes between consecutive actions.",
        default=-0.02, min_value=-10.0, max_value=0.0, step=0.01,
        backends=frozenset({BACKEND_SB3}),
        algorithms=frozenset({ALG_ALL}),
        il_inline=_SB3_INLINE_ACTION_SMOOTHNESS,
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
        il_inline=_SB3_INLINE_UPRIGHT,
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
        il_inline=_SB3_INLINE_LATERAL_PENALTY,
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
        il_inline=_SB3_INLINE_FOOT_CLEARANCE,
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
        il_inline=_SB3_INLINE_SLIP_PENALTY,
    ),
    "feet_slide": _reward_item(
        key="feet_slide",
        polarity="penalty",
        title="Feet Slide",
        desc="Contact-force-based penalty on foot lateral velocity while in ground contact. "
             "Uses contact detection to weight the sliding cost.",
        default=-0.1, min_value=-5.0, max_value=0.0, step=0.01,
        applicable_families=_LEGGED_FAMILIES,
        backends=frozenset({BACKEND_SB3}),
        algorithms=frozenset({ALG_ALL}),
        il_inline=_SB3_INLINE_FEET_SLIDE,
    ),
    "collision_penalty": _reward_item(
        key="collision_penalty",
        polarity="penalty",
        title="Collision Penalty",
        desc="Penalty for unwanted collisions with the environment or self.",
        default=-0.2, min_value=-10.0, max_value=0.0, step=0.05,
        backends=frozenset({BACKEND_SB3}),
        algorithms=frozenset({ALG_ALL}),
        il_inline=_SB3_INLINE_COLLISION_PENALTY,
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
        il_inline=_SB3_INLINE_GOAL_DISTANCE,
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
        il_inline=_SB3_INLINE_GRASP_SUCCESS,
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
        il_inline=_SB3_INLINE_REFERENCE_TRACKING,
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
        il_inline=_SB3_INLINE_BASE_HEIGHT_PENALTY,
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
        il_inline=_SB3_INLINE_ANGULAR_RATE_PENALTY,
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
        il_inline=_SB3_INLINE_FOOT_AIR_TIME,
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
        il_inline=_SB3_INLINE_JOINT_POSE_TRACKING,
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
        il_inline=_SB3_INLINE_JOINT_VEL_TRACKING,
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
        il_inline=_SB3_INLINE_FOOT_POS_TRACKING,
    ),
    # ── SB3 rewards mirroring IL keys (same key, SB3 backend) ────────
    "lin_vel_z_penalty": _reward_item(
        key="lin_vel_z_penalty",
        polarity="penalty",
        title="Lin Vel Z Penalty",
        desc="L2 penalty on vertical linear velocity to discourage bouncing.",
        default=-2.0, min_value=-20.0, max_value=0.0, step=0.1,
        applicable_families=_LOCOMOTION_FAMILIES,
        backends=frozenset({BACKEND_SB3}),
        algorithms=frozenset({ALG_ALL}),
        il_inline=_SB3_INLINE_LIN_VEL_Z_PENALTY,
    ),
    "joint_torque_penalty": _reward_item(
        key="joint_torque_penalty",
        polarity="penalty",
        title="Joint Torque Penalty",
        desc="L2 penalty on applied joint torques for energy efficiency.",
        default=-0.0002, min_value=-0.01, max_value=0.0, step=0.00005,
        applicable_families=_ALL_FAMILIES,
        backends=frozenset({BACKEND_SB3}),
        algorithms=frozenset({ALG_ALL}),
        il_inline=_SB3_INLINE_JOINT_TORQUE_PENALTY,
    ),
    "joint_accel_penalty": _reward_item(
        key="joint_accel_penalty",
        polarity="penalty",
        title="Joint Accel Penalty",
        desc="L2 penalty on joint accelerations for smooth motion.",
        default=-2.5e-7, min_value=-0.001, max_value=0.0, step=1e-7,
        applicable_families=_ALL_FAMILIES,
        backends=frozenset({BACKEND_SB3}),
        algorithms=frozenset({ALG_ALL}),
        il_inline=_SB3_INLINE_JOINT_ACCEL_PENALTY,
    ),
    "joint_vel_penalty": _reward_item(
        key="joint_vel_penalty",
        polarity="penalty",
        title="Joint Vel Penalty",
        desc="L2 penalty on joint velocities — discourages overly fast joint motion.",
        default=-0.001, min_value=-1.0, max_value=0.0, step=0.0005,
        applicable_families=_ALL_FAMILIES,
        backends=frozenset({BACKEND_SB3}),
        algorithms=frozenset({ALG_ALL}),
        il_inline=_SB3_INLINE_JOINT_VEL_PENALTY,
    ),
    "dof_pos_limits": _reward_item(
        key="dof_pos_limits",
        polarity="penalty",
        title="Joint Pos Limits",
        desc="Penalty when joint positions approach or exceed soft limits.",
        default=-5.0, min_value=-20.0, max_value=0.0, step=0.5,
        applicable_families=_ALL_FAMILIES,
        backends=frozenset({BACKEND_SB3}),
        algorithms=frozenset({ALG_ALL}),
        il_inline=_SB3_INLINE_DOF_POS_LIMITS,
    ),
    "undesired_contacts": _reward_item(
        key="undesired_contacts",
        polarity="penalty",
        title="Undesired Contacts",
        desc="Penalty when non-foot bodies make ground contact.",
        default=-1.0, min_value=-10.0, max_value=0.0, step=0.1,
        applicable_families=_LOCOMOTION_FAMILIES,
        backends=frozenset({BACKEND_SB3}),
        algorithms=frozenset({ALG_ALL}),
        il_inline=_SB3_INLINE_UNDESIRED_CONTACTS,
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


#: Reward terms that almost every legged-locomotion run benefits from.
#: Used both as the default seed when the canvas wires no rewards AND as
#: the keyset checked by ``warn_missing_recommended_reward_terms`` —
#: a partial canvas dict missing any of these will silently degrade to
#: weight 0 inside the env (``self._reward_terms.get(key, 0.0)``), which
#: is exactly how the early-2026 IL training runs ended up with crouching
#: + asymmetric gait. See ``knowledge_base/PROJECT_KNOWLEDGE.md`` if
#: that diagnostic ever needs revisiting.
RECOMMENDED_LOCOMOTION_REWARD_TERMS: Tuple[str, ...] = (
    "velocity_tracking",
    "alive",
    "energy",
    "upright",
    "base_height_penalty",
    "action_smoothness",
    "dof_pos_limits",
)


def default_reward_terms() -> Dict[str, float]:
    """Default reward set seeded into the SB3 env when no canvas wiring exists.

    The previous default contained only ``velocity_tracking / alive /
    energy`` — the three "task" terms. The four safety terms (``upright``,
    ``base_height_penalty``, ``action_smoothness``, ``dof_pos_limits``)
    are now included as well: without them a from-scratch policy quickly
    discovers crouching + jittery stepping as the energy-minimal way to
    keep the velocity-tracking reward.
    """
    return {
        key: REWARD_REGISTRY[key].default
        for key in RECOMMENDED_LOCOMOTION_REWARD_TERMS
    }


def warn_missing_recommended_reward_terms(
    user_terms: Optional[Dict[str, float]],
    *,
    logger=None,
) -> Tuple[str, ...]:
    """Emit a warning when the user-supplied reward dict skips a recommended term.

    Returns the tuple of missing keys (possibly empty). Pass ``logger`` to
    redirect output; otherwise prints to stderr via ``warnings.warn``.

    Only warns when the user *did* pass a non-empty dict — silence when
    the env will fall back to ``default_reward_terms()`` (which already
    includes everything) or when the user explicitly passed an empty
    dict (an explicit "I want zero reward" signal).
    """
    if not user_terms:
        return ()
    missing = tuple(
        k for k in RECOMMENDED_LOCOMOTION_REWARD_TERMS if k not in user_terms
    )
    if not missing:
        return ()
    msg = (
        "[task_module_registry] reward_terms is missing recommended "
        f"safety terms: {', '.join(missing)}. They will default to "
        "weight 0 — the policy may learn crouching / asymmetric gait "
        "to survive without these guards. Add them to the Canvas "
        "Rewards node or the user-passed reward_terms dict."
    )
    if logger is not None:
        try:
            logger.warning(msg)
        except Exception:
            print(msg, flush=True)
    else:
        import warnings
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
    return missing


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

# ── Inline reward implementations ────────────────────────────────────
# Every reward in IL_REWARD_REGISTRY now carries an ``il_inline`` block
# so the Reward Editor always shows real source and the compiler can
# emit a fully self-contained config.  Functions that mirror standard
# Isaac Lab mdp helpers are prefixed ``_unitport_`` to avoid collisions
# when the user's env also imports the original module.

_INLINE_TRACK_LIN_VEL_XY_EXP = '''
def _unitport_track_lin_vel_xy_exp(env, std=0.25, command_name="base_velocity",
                                    asset_cfg=SceneEntityCfg("robot")):
    """Exponential tracking reward for commanded XY linear velocity."""
    import torch
    asset = env.scene[asset_cfg.name]
    vel_xy = asset.data.root_lin_vel_b[:, :2]
    cmd_xy = env.command_manager.get_command(command_name)[:, :2]
    error = torch.sum(torch.square(vel_xy - cmd_xy), dim=1)
    return torch.exp(-error / (std * std))
'''

_INLINE_TRACK_ANG_VEL_Z_EXP = '''
def _unitport_track_ang_vel_z_exp(env, std=0.25, command_name="base_velocity",
                                   asset_cfg=SceneEntityCfg("robot")):
    """Exponential tracking reward for commanded yaw angular velocity."""
    import torch
    asset = env.scene[asset_cfg.name]
    ang_vel_z = asset.data.root_ang_vel_b[:, 2]
    cmd_yaw = env.command_manager.get_command(command_name)[:, 2]
    error = torch.square(ang_vel_z - cmd_yaw)
    return torch.exp(-error / (std * std))
'''

_INLINE_LIN_VEL_Z_L2 = '''
def _unitport_lin_vel_z_l2(env, asset_cfg=SceneEntityCfg("robot")):
    """L2 penalty on vertical linear velocity."""
    import torch
    asset = env.scene[asset_cfg.name]
    vz = asset.data.root_lin_vel_b[:, 2]
    # Cap |vz| at 10 m/s before squaring — protects against physics spikes
    # where the body velocity jumps to hundreds of m/s in one step.
    vz = torch.nan_to_num(vz, nan=0.0, posinf=10.0, neginf=-10.0)
    vz = torch.clamp(vz, min=-10.0, max=10.0)
    return torch.square(vz)
'''

_INLINE_ANG_VEL_XY_L2 = '''
def _unitport_ang_vel_xy_l2(env, asset_cfg=SceneEntityCfg("robot")):
    """L2 penalty on roll/pitch angular velocity."""
    import torch
    asset = env.scene[asset_cfg.name]
    w = asset.data.root_ang_vel_b[:, :2]
    w = torch.nan_to_num(w, nan=0.0, posinf=20.0, neginf=-20.0)
    w = torch.clamp(w, min=-20.0, max=20.0)
    return torch.sum(torch.square(w), dim=1)
'''

_INLINE_LIN_VEL_XY_L2 = '''
def _unitport_lin_vel_xy_l2(env, asset_cfg=SceneEntityCfg("robot")):
    """L2 penalty on horizontal linear velocity — unconditional base-motion penalty.

    Complements track_lin_vel_xy: tracking reward saturates with a Gaussian
    kernel, this penalty grows linearly with |v|^2, giving stronger gradient
    to suppress drift when the commanded velocity is zero (stand) or
    yaw-dominant (turn).
    """
    import torch
    asset = env.scene[asset_cfg.name]
    v = asset.data.root_lin_vel_b[:, :2]
    v = torch.nan_to_num(v, nan=0.0, posinf=10.0, neginf=-10.0)
    v = torch.clamp(v, min=-10.0, max=10.0)
    return torch.sum(torch.square(v), dim=1)
'''

_INLINE_ANG_VEL_Z_L2 = '''
def _unitport_ang_vel_z_l2(env, asset_cfg=SceneEntityCfg("robot")):
    """L2 penalty on yaw angular velocity — unconditional.

    Mirrors ang_vel_xy_penalty for the Z axis. Use for items that
    should not spin (stand / strafe / pace).
    """
    import torch
    asset = env.scene[asset_cfg.name]
    w = asset.data.root_ang_vel_b[:, 2]
    w = torch.nan_to_num(w, nan=0.0, posinf=20.0, neginf=-20.0)
    w = torch.clamp(w, min=-20.0, max=20.0)
    return torch.square(w)
'''

_INLINE_JOINT_TORQUES_L2 = '''
def _unitport_joint_torques_l2(env, asset_cfg=SceneEntityCfg("robot")):
    """L2 penalty on applied joint torques."""
    import torch
    asset = env.scene[asset_cfg.name]
    tq = asset.data.applied_torque[:, asset_cfg.joint_ids]
    tq = torch.nan_to_num(tq, nan=0.0, posinf=1.0e3, neginf=-1.0e3)
    tq = torch.clamp(tq, min=-1.0e3, max=1.0e3)
    return torch.sum(torch.square(tq), dim=1)
'''

_INLINE_JOINT_ACC_L2 = '''
def _unitport_joint_acc_l2(env, asset_cfg=SceneEntityCfg("robot")):
    """L2 penalty on joint accelerations."""
    import torch
    asset = env.scene[asset_cfg.name]
    acc = asset.data.joint_acc[:, asset_cfg.joint_ids]
    # Guard against physics spikes / NaN: cap per-joint |accel| at 1000 rad/s^2
    # before squaring so a single bad step can't blow up the value function.
    acc = torch.nan_to_num(acc, nan=0.0, posinf=1.0e3, neginf=-1.0e3)
    acc = torch.clamp(acc, min=-1.0e3, max=1.0e3)
    return torch.sum(torch.square(acc), dim=1)
'''

_INLINE_ACTION_RATE_L2 = '''
def _unitport_action_rate_l2(env):
    """L2 penalty on action rate of change (consecutive action difference)."""
    import torch
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)
'''

_INLINE_JOINT_VEL_L2 = '''
def _unitport_joint_vel_l2(env, asset_cfg=SceneEntityCfg("robot")):
    """L2 penalty on joint velocities."""
    import torch
    asset = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)
'''

_INLINE_FLAT_ORIENTATION_L2 = '''
def _unitport_flat_orientation_l2(env, asset_cfg=SceneEntityCfg("robot")):
    """L2 penalty on projected gravity deviation from vertical (keeps base level)."""
    import torch
    asset = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
'''

_INLINE_IS_ALIVE = '''
def _unitport_is_alive(env):
    """Constant per-step survival bonus (1.0 for all alive envs)."""
    import torch
    return (~env.termination_manager.terminated).float()
'''

_INLINE_FEET_AIR_TIME = '''
def _unitport_feet_air_time(env, threshold=0.5, command_name="base_velocity",
                             sensor_cfg=None, asset_cfg=SceneEntityCfg("robot")):
    """Reward feet air time when the robot is moving (velocity command above threshold).

    Mirrors velocity_mdp.feet_air_time: gives a bonus proportional to
    how long each foot was in the air at touchdown, gated by whether the
    velocity command magnitude exceeds a minimum.
    """
    import torch
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    cmd = env.command_manager.get_command(command_name)
    is_moving = torch.norm(cmd[:, :2], dim=1) > 0.1
    return reward * is_moving
'''

_INLINE_JOINT_POS_LIMITS = '''
def _unitport_joint_pos_limits(env, asset_cfg=SceneEntityCfg("robot")):
    """Penalty when joint positions approach or exceed soft limits.

    For each joint, computes the excess beyond 95% of the joint range
    on both sides and sums the absolute violations.
    """
    import torch
    asset = env.scene[asset_cfg.name]
    pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    soft_lo = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]
    soft_hi = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]
    out_of_range = -(pos - soft_lo).clamp(max=0.0) + (pos - soft_hi).clamp(min=0.0)
    return torch.sum(out_of_range, dim=1)
'''

_INLINE_JOINT_DEVIATION_L1 = '''
def _unitport_joint_deviation_l1(env, asset_cfg=SceneEntityCfg("robot")):
    """L1 penalty on joint position deviation from the asset's default pose.

    Anchors the policy to the nominal stance encoded in the articulation's
    ``default_joint_pos`` — counters left/right asymmetric drift and
    body-sinking that arises when the policy has no posture reference.
    """
    import torch
    asset = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - \
            asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(angle), dim=1)
'''

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
    """Enforce periodic gait patterns for legged robots.

    Per-leg contribution is +1 when the leg's contact state matches the
    expected stance/swing phase, and -1 when it mismatches. This makes
    "freeze all 4 legs while a velocity command is active" actively
    penalised (= -N_legs) rather than merely unrewarded (= 0), which
    was the prior behaviour and let policies skip gait collection to
    cheat e.g. track_ang_vel_z by twisting joints in place.

    The command_norm gate keeps gait inactive on zero-command stances
    (stand motion), so this stricter shaping does not collide with the
    intent to stay still when commanded.
    """
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
        match = ~(is_stance ^ is_contact[:, i])
        reward += match.float() * 2.0 - 1.0
    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        reward *= (cmd_norm > 0.1).float()
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
def _unitport_base_height_l2(env, target_height=None, asset_cfg=SceneEntityCfg("robot"),
                              sensor_cfg=None):
    """Penalize base height deviation from a brand-neutral standing target.

    target_height resolution (first non-None wins):
      1. The ``target_height`` kwarg passed by the canvas (Robot node
         slider). Caller-controlled override.
      2. ``asset.data.default_root_state[:, 2]`` — the spawn pose's
         z-coordinate, which is the asset's nominal standing height
         declared in its USD/MJCF init_state. Brand-neutral, no per-robot
         tuning needed; works for Go2 (~0.4 m), A1 (~0.32 m), Spot
         (~0.5 m), G1/H1 (~0.7 m).
      3. Final fallback 0.34 m — generic legged-quadruped value used
         only if neither of the above is available (test envs without
         spawn pose).
    """
    import torch
    asset = env.scene[asset_cfg.name]
    # 0.0 / negative is treated as "use auto" sentinel because the canvas
    # compiler emits 0.0 when no Robot.target_height override is set.
    if target_height is None or target_height <= 0.0:
        try:
            spawn_z = asset.data.default_root_state[:, 2]
            target_height = float(spawn_z[0].item())
        except Exception:
            target_height = 0.34
    if sensor_cfg is not None:
        sensor = env.scene[sensor_cfg.name]
        ray_hits = sensor.data.ray_hits_w[..., 2]
        if torch.isnan(ray_hits).any() or torch.isinf(ray_hits).any():
            adjusted = asset.data.root_link_pos_w[:, 2]
        else:
            adjusted = target_height + torch.mean(ray_hits, dim=1)
    else:
        adjusted = target_height
    err = asset.data.root_pos_w[:, 2] - adjusted
    # Cap deviation at 1 m before squaring so a physics-clipping event
    # cannot push the per-step penalty past ~1.0.
    err = torch.nan_to_num(err, nan=0.0, posinf=1.0, neginf=-1.0)
    err = torch.clamp(err, min=-1.0, max=1.0)
    return torch.square(err)
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
        il_func="_unitport_track_lin_vel_xy_exp", il_module=IL_MOD_INLINE,
        il_params='"std": {node_std}, "command_name": "base_velocity"',
        il_inline=_INLINE_TRACK_LIN_VEL_XY_EXP,
    ),
    "track_ang_vel_z": _reward_item(
        key="track_ang_vel_z",
        polarity="reward",
        title="Track Ang Vel Z",
        desc="Exponential tracking reward for commanded yaw angular velocity.",
        default=0.75, min_value=0.0, max_value=10.0, step=0.05,
        applicable_families=_LOCOMOTION_FAMILIES,
        backends=_IL, algorithms=frozenset({ALG_ALL}),
        il_func="_unitport_track_ang_vel_z_exp", il_module=IL_MOD_INLINE,
        il_params='"std": {node_std}, "command_name": "base_velocity"',
        il_inline=_INLINE_TRACK_ANG_VEL_Z_EXP,
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
        il_func="_unitport_lin_vel_z_l2", il_module=IL_MOD_INLINE,
        il_inline=_INLINE_LIN_VEL_Z_L2,
    ),
    "ang_vel_xy_penalty": _reward_item(
        key="ang_vel_xy_penalty",
        polarity="penalty",
        title="Ang Vel XY Penalty",
        desc="L2 penalty on roll/pitch angular velocity.",
        default=-0.05, min_value=-5.0, max_value=0.0, step=0.005,
        applicable_families=_LOCOMOTION_FAMILIES,
        backends=_IL, algorithms=frozenset({ALG_ALL}),
        il_func="_unitport_ang_vel_xy_l2", il_module=IL_MOD_INLINE,
        il_inline=_INLINE_ANG_VEL_XY_L2,
    ),
    "lin_vel_xy_penalty": _reward_item(
        key="lin_vel_xy_penalty",
        polarity="penalty",
        title="Lin Vel XY Penalty",
        desc="L2 penalty on horizontal linear velocity — unconditional. "
             "Use heavy weight for stand (kill drift) or moderate for turn "
             "(discourage forward leak during yaw).",
        default=-0.5, min_value=-20.0, max_value=0.0, step=0.1,
        applicable_families=_LOCOMOTION_FAMILIES,
        backends=_IL, algorithms=frozenset({ALG_ALL}),
        il_func="_unitport_lin_vel_xy_l2", il_module=IL_MOD_INLINE,
        il_inline=_INLINE_LIN_VEL_XY_L2,
    ),
    "ang_vel_z_penalty": _reward_item(
        key="ang_vel_z_penalty",
        polarity="penalty",
        title="Ang Vel Z Penalty",
        desc="L2 penalty on yaw angular velocity — unconditional. "
             "Use for items that should not spin (stand / strafe / pace).",
        default=-0.1, min_value=-10.0, max_value=0.0, step=0.05,
        applicable_families=_LOCOMOTION_FAMILIES,
        backends=_IL, algorithms=frozenset({ALG_ALL}),
        il_func="_unitport_ang_vel_z_l2", il_module=IL_MOD_INLINE,
        il_inline=_INLINE_ANG_VEL_Z_L2,
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
        il_func="_unitport_joint_torques_l2", il_module=IL_MOD_INLINE,
        il_inline=_INLINE_JOINT_TORQUES_L2,
    ),
    "joint_accel_penalty": _reward_item(
        key="joint_accel_penalty",
        polarity="penalty",
        title="Joint Accel Penalty",
        desc="L2 penalty on joint accelerations for smooth motion.",
        default=-2.5e-7, min_value=-0.001, max_value=0.0, step=1e-7,
        applicable_families=_ALL_FAMILIES,
        backends=_IL, algorithms=frozenset({ALG_ALL}),
        il_func="_unitport_joint_acc_l2", il_module=IL_MOD_INLINE,
        il_inline=_INLINE_JOINT_ACC_L2,
    ),
    "action_rate_penalty": _reward_item(
        key="action_rate_penalty",
        polarity="penalty",
        title="Action Rate Penalty",
        desc="L2 penalty on action rate of change for smooth control.",
        default=-0.05, min_value=-1.0, max_value=0.0, step=0.005,
        applicable_families=_ALL_FAMILIES,
        backends=_IL, algorithms=frozenset({ALG_ALL}),
        il_func="_unitport_action_rate_l2", il_module=IL_MOD_INLINE,
        il_inline=_INLINE_ACTION_RATE_L2,
    ),
    "joint_vel_penalty": _reward_item(
        key="joint_vel_penalty",
        polarity="penalty",
        title="Joint Vel Penalty",
        desc="L2 penalty on joint velocities — discourages overly fast joint motion.",
        default=-0.001, min_value=-1.0, max_value=0.0, step=0.0005,
        applicable_families=_ALL_FAMILIES,
        backends=_IL, algorithms=frozenset({ALG_ALL}),
        il_func="_unitport_joint_vel_l2", il_module=IL_MOD_INLINE,
        il_params='"asset_cfg": SceneEntityCfg("robot")',
        il_inline=_INLINE_JOINT_VEL_L2,
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
        il_func="_unitport_flat_orientation_l2", il_module=IL_MOD_INLINE,
        il_inline=_INLINE_FLAT_ORIENTATION_L2,
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
        # ``{robot_target_height}`` is substituted by the IL config compiler
        # from the Robot canvas node (slider override → asset.nominal_height
        # → 0.34 fallback). Never hardcode a per-robot constant here.
        il_params='"target_height": {robot_target_height}, "asset_cfg": SceneEntityCfg("robot")',
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
        il_func="_unitport_is_alive", il_module=IL_MOD_INLINE,
        il_inline=_INLINE_IS_ALIVE,
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
        il_func="_unitport_feet_air_time", il_module=IL_MOD_INLINE,
        il_params='"threshold": {node_threshold}, "command_name": "base_velocity", '
                  '"sensor_cfg": SceneEntityCfg("contact_forces", body_names={ir:feet})',
        il_inline=_INLINE_FEET_AIR_TIME,
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
        il_params='"asset_cfg": SceneEntityCfg("robot", body_names={ir:feet}), '
                  '"target_height": 0.1, "std": 0.05, "tanh_mult": 2.0',
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
                  '"asset_cfg": SceneEntityCfg("robot", body_names={ir:feet})',
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
        il_func="_unitport_joint_pos_limits", il_module=IL_MOD_INLINE,
        il_params='"asset_cfg": SceneEntityCfg("robot")',
        il_inline=_INLINE_JOINT_POS_LIMITS,
    ),
    "joint_deviation_l1": _reward_item(
        key="joint_deviation_l1",
        polarity="penalty",
        title="Joint Deviation L1",
        desc="L1 penalty on joint positions deviating from the asset's default pose. "
             "Anchors the policy to a symmetric nominal stance — counters left/right "
             "asymmetric drift and posture sinking.",
        default=-0.05, min_value=-2.0, max_value=0.0, step=0.01,
        applicable_families=_ALL_FAMILIES,
        backends=_IL, algorithms=frozenset({ALG_ALL}),
        il_func="_unitport_joint_deviation_l1", il_module=IL_MOD_INLINE,
        il_params='"asset_cfg": SceneEntityCfg("robot")',
        il_inline=_INLINE_JOINT_DEVIATION_L1,
    ),
}

# For observation terms, ``default / min_value / max_value / step`` describe the
# Isaac Lab ``ObservationTermCfg.scale`` (a normalisation multiplier applied to
# the term output before it enters the policy input vector). Conventional Isaac
# Lab values: base_lin_vel≈2.0, base_ang_vel≈0.25, joint_vel≈0.05, others≈1.0.
# These are NOT toggles — a term not present in the canvas ``obs_terms`` dict is
# disabled; a term present declares its scale explicitly.
IL_OBS_REGISTRY: Dict[str, TaskModuleItem] = {
    "base_lin_vel": TaskModuleItem(
        key="base_lin_vel",
        kind="observation",
        polarity="",
        title="Base Lin Vel",
        desc="Base linear velocity in body frame (3D). Isaac Lab convention scale=2.0.",
        default=2.0,
        min_value=0.0,
        max_value=5.0,
        step=0.05,
        applicable_families=_ALL_FAMILIES,
    ),
    "base_ang_vel": TaskModuleItem(
        key="base_ang_vel",
        kind="observation",
        polarity="",
        title="Base Ang Vel",
        desc="Base angular velocity in body frame (3D). Isaac Lab convention scale=0.25.",
        default=0.25,
        min_value=0.0,
        max_value=2.0,
        step=0.05,
        applicable_families=_ALL_FAMILIES,
    ),
    "projected_gravity": TaskModuleItem(
        key="projected_gravity",
        kind="observation",
        polarity="",
        title="Projected Gravity",
        desc="Gravity vector projected into body frame (3D). Conventional scale=1.0.",
        default=1.0,
        min_value=0.0,
        max_value=5.0,
        step=0.1,
        applicable_families=_ALL_FAMILIES,
    ),
    "velocity_command": TaskModuleItem(
        key="velocity_command",
        kind="observation",
        polarity="",
        title="Velocity Command",
        desc="Commanded velocity target (3D: vx, vy, wz). Conventional scale=1.0.",
        default=1.0,
        min_value=0.0,
        max_value=5.0,
        step=0.1,
        applicable_families=_LOCOMOTION_FAMILIES,
    ),
    "joint_pos": TaskModuleItem(
        key="joint_pos",
        kind="observation",
        polarity="",
        title="Joint Positions",
        desc="Joint position readings, relative to default pose (N-dim). Conventional scale=1.0.",
        default=1.0,
        min_value=0.0,
        max_value=5.0,
        step=0.05,
        applicable_families=_ALL_FAMILIES,
    ),
    "joint_vel": TaskModuleItem(
        key="joint_vel",
        kind="observation",
        polarity="",
        title="Joint Velocities",
        desc="Joint velocity readings (N-dim). Isaac Lab convention scale=0.05.",
        default=0.05,
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        applicable_families=_ALL_FAMILIES,
    ),
    "last_action": TaskModuleItem(
        key="last_action",
        kind="observation",
        polarity="",
        title="Last Action",
        desc="Previous policy output / action applied (N-dim). Conventional scale=1.0.",
        default=1.0,
        min_value=0.0,
        max_value=5.0,
        step=0.05,
        applicable_families=_ALL_FAMILIES,
    ),
    "height_scan": TaskModuleItem(
        key="height_scan",
        kind="observation",
        polarity="",
        title="Height Scan",
        desc="Terrain height scanner readings around each foot (M-dim). Conventional scale=5.0.",
        default=5.0,
        min_value=0.0,
        max_value=20.0,
        step=0.5,
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


# ---------------------------------------------------------------------------
# Unified Registry — single source of truth for all engines
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Kind-namespaced registry index (collision-safe).
#
# Keys are only unique WITHIN a kind. A reward and a termination may
# legitimately share a semantic name (e.g. ``base_height`` = reward penalty
# on height deviation, AND termination when height drops too low). The
# previous flat ``UNIFIED_REGISTRY`` merged everything into one dict,
# causing the later-merged termination to silently shadow the reward
# (and vice-versa). That abstraction was removed — all lookups now go
# through ``lookup(key, kind=...)`` or ``query_registry(kind=...)``, both
# of which preserve the (kind, key) tuple namespace.
#
# Within a kind, multiple sub-registries may exist (SB3 + IL variants).
# They are iterated in order; ``query_registry`` filters by ``backend``
# so only the implementation matching the engine is returned.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Discriminator function registry (P0.3 — AMP-only)
# ---------------------------------------------------------------------------
# Three editable function bodies that drive AMPDiscriminator behaviour.
# Each entry's ``il_inline`` carries the default Python source — the
# RewardEditorPanel (parameterised on kind) lets the user inspect and
# override these bodies. AMPDiscriminator at __init__ exec's any
# user-edited source into a callable and binds it as a method-replacement
# guard — see _AMP_DISC_SOURCE_DEFAULTS for the canonical defaults this
# registry mirrors.
#
# Why a registry instead of editing discriminator.py directly:
#   - Edits survive UnitPort upgrades that re-vendor discriminator.py
#   - Edits don't pollute the vendored AMP_for_hardware code mirror
#   - Three slots × isolated callables make it obvious which behaviour
#     is being changed
#
# The ``kind="discriminator"`` registry is AMP-specific — no SB3/Isaac
# Lab split (vanilla PPO has no discriminator), so a single sub-registry
# is enough.

_DISC_DEFAULT_FORWARD = '''\
def forward(self, x):
    """Score a batch of concatenated (s_t, s_{t+1}) pairs.

    Default body: trunk MLP → linear head. Override to add residual
    connections, attention, etc. Inputs:
      - self : the AMPDiscriminator instance (has self.trunk, self.amp_linear)
      - x    : torch.Tensor of shape (batch, 2 * amp_obs_dim)
    Returns: torch.Tensor of shape (batch, 1) — raw logits.
    """
    h = self.trunk(x)
    return self.amp_linear(h)
'''

_DISC_DEFAULT_GRAD_PEN = '''\
def compute_grad_pen(self, expert_state, expert_next_state, lambda_=10.0):
    """Zero-centered R1 gradient penalty on expert transitions.

    Default body: matches Peng 2021 / amp-rsl-rl. Override to swap in
    standard WGAN-GP (target = 1.0) or to change the norm.
    """
    import torch
    from torch import autograd
    expert_data = torch.cat([expert_state, expert_next_state], dim=-1)
    expert_data.requires_grad = True
    disc = self.amp_linear(self.trunk(expert_data))
    ones = torch.ones(disc.size(), device=disc.device)
    grad = autograd.grad(
        outputs=disc, inputs=expert_data, grad_outputs=ones,
        create_graph=True, retain_graph=True, only_inputs=True,
    )[0]
    return float(lambda_) * (grad.norm(2, dim=1) - 0.0).pow(2).mean()
'''

_DISC_DEFAULT_PREDICT_REWARD = '''\
def predict_amp_reward(self, state, next_state, normalizer=None):
    """Compute the AMP style reward for a batch of policy transitions.

    Default body: ``r = coef * softplus(d.clamp(max=logit_clamp_max))``
    — Peng 2021's BCE-paired formulation. Override to switch to
    LSGAN-style ``-(d-1)^2``, sigmoid-based, etc.

    Returns (style_reward, disc_score) — first is the per-step reward,
    second is the raw logit for logging.
    """
    import torch
    with torch.no_grad():
        self.eval()
        if normalizer is not None:
            state = normalizer.normalize_torch(state, self.device)
            next_state = normalizer.normalize_torch(next_state, self.device)
        d = self.amp_linear(self.trunk(torch.cat([state, next_state], dim=-1)))
        style_reward = self.amp_reward_coef * torch.nn.functional.softplus(
            d.clamp(max=self.logit_clamp_max)
        )
        self.train()
    return style_reward.squeeze(-1), d
'''


def _disc_item(*, key: str, title: str, desc: str, il_inline: str) -> TaskModuleItem:
    return TaskModuleItem(
        key=key,
        kind="discriminator",
        polarity="",
        title=title,
        desc=desc,
        # default/min/max/step are unused for discriminator entries (they
        # have no scalar weight) — keep dataclass happy with sentinel 0s.
        default=0.0,
        min_value=0.0,
        max_value=0.0,
        step=0.0,
        applicable_families=_ALL_FAMILIES,
        backends=frozenset({BACKEND_ISAAC}),
        algorithms=frozenset({ALG_AMP}),
        il_inline=il_inline,
    )


IL_DISC_REGISTRY: Dict[str, TaskModuleItem] = {
    "disc_forward": _disc_item(
        key="disc_forward",
        title="Forward",
        desc="MLP forward pass — trunk + linear head. Override to add "
             "residuals, attention, dropout, etc.",
        il_inline=_DISC_DEFAULT_FORWARD,
    ),
    "disc_compute_grad_pen": _disc_item(
        key="disc_compute_grad_pen",
        title="Gradient Penalty",
        desc="Zero-centered R1 gradient penalty on expert transitions. "
             "Default targets |grad|=0 (Peng 2021); override for standard "
             "WGAN-GP target=1 or different norm.",
        il_inline=_DISC_DEFAULT_GRAD_PEN,
    ),
    "disc_predict_reward": _disc_item(
        key="disc_predict_reward",
        title="Predict AMP Reward",
        desc="Disc score → style reward. Default: amp_reward_coef × "
             "softplus(d.clamp(max=logit_clamp_max)). Override to switch "
             "to LSGAN ``-(d-1)^2``, sigmoid-based, etc.",
        il_inline=_DISC_DEFAULT_PREDICT_REWARD,
    ),
}


ALL_REGISTRIES: Dict[str, List[Dict[str, TaskModuleItem]]] = {
    "reward": [REWARD_REGISTRY, IL_REWARD_REGISTRY],
    "termination": [TERMINATION_REGISTRY, IL_TERMINATION_REGISTRY],
    "observation": [IL_OBS_REGISTRY],
    "discriminator": [IL_DISC_REGISTRY],
}


def iter_all_items() -> Iterator[Tuple[str, str, TaskModuleItem]]:
    """Yield ``(kind, key, item)`` for every registered module across all kinds."""
    for kind, subs in ALL_REGISTRIES.items():
        for sub in subs:
            for key, item in sub.items():
                yield (kind, key, item)


def lookup(
    key: str,
    *,
    kind: str,
    backend: Optional[str] = None,
) -> Optional[TaskModuleItem]:
    """Kind-aware single-key lookup.

    Returns the item whose key matches within the requested kind's
    sub-registries, or ``None``. Use this instead of a flat lookup — it
    guarantees that a reward key like ``"base_height"`` never resolves
    to the termination entry of the same name (and vice versa).

    When ``backend`` is provided, only items whose ``backends`` include
    that backend are considered — this resolves *within-kind* collisions
    where the same key exists as both an SB3 and an Isaac Lab variant
    (e.g. ``joint_accel_penalty`` is registered with an empty ``il_func``
    on the SB3 side and with ``_unitport_joint_acc_l2`` on the IL side).

    When ``backend`` is None, the LAST matching item wins — this mirrors
    the original dict-merge order where Isaac Lab entries (registered
    later) shadow earlier SB3 entries for the same key.
    """
    match: Optional[TaskModuleItem] = None
    for sub in ALL_REGISTRIES.get(kind, []):
        item = sub.get(key)
        if item is None:
            continue
        if backend is not None and backend not in item.backends:
            continue
        match = item
    return match


def writable_registry(kind: str) -> Dict[str, TaskModuleItem]:
    """Return the primary writable sub-registry for a kind.

    Used by user-facing editors (e.g. RewardEditorPanel) to persist
    custom modules. The last sub-registry in the list is treated as the
    writable target (IL_REWARD_REGISTRY for rewards, etc.).
    """
    subs = ALL_REGISTRIES.get(kind, [])
    if not subs:
        raise KeyError(f"unknown registry kind: {kind!r}")
    return subs[-1]


def query_registry(
    *,
    kind: Optional[str] = None,
    backend: Optional[str] = None,
    algorithm: Optional[str] = None,
    family: Optional[str] = None,
) -> Dict[str, TaskModuleItem]:
    """Filter modules by kind / backend / algorithm / family.

    Parameters
    ----------
    kind : "reward" | "termination" | "observation" | None
        Filter by function kind. None = all kinds.
    backend : "sb3" | "isaac_lab" | None
        Filter by engine compatibility. None = all.
    algorithm : "PPO" | "AMP" | "SAC" | None
        Filter by algorithm compatibility. None = all.
    family : "quadruped" | "biped" | None
        Filter by robot family. None = all.

    Returns
    -------
    Dict[str, TaskModuleItem]
        Filtered copy. When ``kind`` is given, only that kind's sub-
        registries are consulted — cross-kind key collisions cannot occur.
        When ``kind`` is None, items from different kinds sharing a key
        will overwrite each other in the returned flat dict; callers that
        need cross-kind visibility should use :func:`iter_all_items`.
    """
    result: Dict[str, TaskModuleItem] = {}
    kinds = [kind] if kind is not None else list(ALL_REGISTRIES.keys())
    for k in kinds:
        for sub in ALL_REGISTRIES.get(k, []):
            for key, item in sub.items():
                if backend is not None and backend not in item.backends:
                    continue
                if algorithm is not None:
                    if ALG_ALL not in item.algorithms and algorithm not in item.algorithms:
                        continue
                if family is not None:
                    if item.applicable_families and family not in item.applicable_families:
                        continue
                result[key] = item
    return result


def validate_keys(
    keys,
    kind: str,
    backend: str,
) -> tuple:
    """Validate reward/termination keys against the unified registry.

    Returns
    -------
    (matched, unmatched) : (Dict[str, TaskModuleItem], Set[str])
        matched = items found in the registry for this kind+backend.
        unmatched = keys not found (unknown or wrong backend).
    """
    valid = query_registry(kind=kind, backend=backend)
    matched = {k: valid[k] for k in keys if k in valid}
    unmatched = {k for k in keys if k not in valid}
    return matched, unmatched


def emit_inline_overrides(
    kind: str,
    out_path,
    *,
    backend: Optional[str] = None,
) -> int:
    """Serialise sub-registries for ``kind`` to a JSON sidecar.

    Used to bridge the main process (where the editor mutates the
    in-memory registry) and the training subprocess (which re-imports
    the registry fresh and would otherwise lose user edits).

    Walks **all** sub-registries of the given kind (not just the
    writable one) so SB3-side reward edits in ``REWARD_REGISTRY``
    survive alongside IL-side edits in ``IL_REWARD_REGISTRY``.

    The sidecar format is intentionally minimal::

        {
          "kind": "discriminator",
          "entries": {
            "disc_predict_reward": {"il_inline": "def ..."},
            ...
          }
        }

    Parameters
    ----------
    kind:
        Registry kind ("reward" / "termination" / "observation" /
        "discriminator").
    out_path:
        File to write. May be str or Path; parent dirs created on demand.
    backend:
        Optional backend filter ("sb3" / "isaac_lab"). When supplied,
        only entries whose ``backends`` set contains this value are
        emitted — the subprocess on that backend will not see (and
        attempt to exec) source intended for the other backend.

    Returns the number of entries written.
    """
    import json
    from pathlib import Path

    out = Path(str(out_path))
    out.parent.mkdir(parents=True, exist_ok=True)

    entries: Dict[str, dict] = {}
    for sub in ALL_REGISTRIES.get(kind, []):
        for key, item in sub.items():
            if not item.il_inline:
                continue
            if backend is not None and backend not in item.backends:
                continue
            # Later sub-registries (e.g. IL_REWARD_REGISTRY) intentionally
            # override earlier ones (REWARD_REGISTRY) when the same key
            # is present — matches the editor's write-back precedence
            # ("write into whichever sub holds the key"), and keeps the
            # subprocess's view consistent with what the editor showed.
            entries[key] = {"il_inline": item.il_inline}

    payload = {"kind": kind, "entries": entries}
    if backend is not None:
        payload["backend"] = backend
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return len(entries)
