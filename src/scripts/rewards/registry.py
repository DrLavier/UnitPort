"""Reward preset registry — SB3 + Isaac Lab.

Migrated from ``DEMO/src/system/training/task_module_registry.py``
(reward portion only). Each preset is a :class:`TaskModuleItem` carrying
its UI metadata, applicable robot families, supported backends/algorithms,
and (for IL presets) the Isaac Lab compiler routing fields plus an
``il_inline`` block of Python source the compiler can emit verbatim.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from scripts.task_module import (
    ALG_ALL,
    ALG_AMP,
    ALG_PPO,
    ALL_FAMILIES,
    BACKEND_ISAAC,
    BACKEND_SB3,
    IL_MOD_INLINE,
    LEGGED_FAMILIES,
    LOCOMOTION_FAMILIES,
    TaskModuleItem,
    reward_item,
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
    "velocity_tracking": reward_item(
        key="velocity_tracking",
        polarity="reward",
        title="Velocity Tracking",
        desc="Reward matching commanded forward and lateral velocity.",
        default=1.0, min_value=0.0, max_value=10.0, step=0.05,
        applicable_families=LOCOMOTION_FAMILIES,
        backends=frozenset({BACKEND_SB3}),
        algorithms=frozenset({ALG_ALL}),
        il_inline=_SB3_INLINE_VELOCITY_TRACKING,
    ),
    "alive": reward_item(
        key="alive",
        polarity="reward",
        title="Alive Bonus",
        desc="Small survival bonus that encourages staying upright.",
        default=0.5, min_value=0.0, max_value=10.0, step=0.05,
        applicable_families=LEGGED_FAMILIES,
        backends=frozenset({BACKEND_SB3}),
        algorithms=frozenset({ALG_ALL}),
        il_inline=_SB3_INLINE_ALIVE,
    ),
    "energy": reward_item(
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
    "yaw_tracking": reward_item(
        key="yaw_tracking",
        polarity="reward",
        title="Yaw Tracking",
        desc="Reward matching the target yaw-rate command.",
        default=0.5, min_value=0.0, max_value=10.0, step=0.05,
        applicable_families=LOCOMOTION_FAMILIES,
        backends=frozenset({BACKEND_SB3}),
        algorithms=frozenset({ALG_ALL}),
        il_inline=_SB3_INLINE_YAW_TRACKING,
    ),
    "action_smoothness": reward_item(
        key="action_smoothness",
        polarity="penalty",
        title="Action Smoothness",
        desc="Penalty for abrupt changes between consecutive actions.",
        default=-0.02, min_value=-10.0, max_value=0.0, step=0.01,
        backends=frozenset({BACKEND_SB3}),
        algorithms=frozenset({ALG_ALL}),
        il_inline=_SB3_INLINE_ACTION_SMOOTHNESS,
    ),
    "upright": reward_item(
        key="upright",
        polarity="reward",
        title="Upright Bonus",
        desc="Reward maintaining an upright torso orientation.",
        default=0.3, min_value=0.0, max_value=10.0, step=0.05,
        applicable_families=LEGGED_FAMILIES,
        backends=frozenset({BACKEND_SB3}),
        algorithms=frozenset({ALG_ALL}),
        il_inline=_SB3_INLINE_UPRIGHT,
    ),
    "lateral_penalty": reward_item(
        key="lateral_penalty",
        polarity="penalty",
        title="Lateral Penalty",
        desc="Penalty for sideways drift when forward tracking is desired.",
        default=-0.1, min_value=-10.0, max_value=0.0, step=0.05,
        applicable_families=LOCOMOTION_FAMILIES,
        backends=frozenset({BACKEND_SB3}),
        algorithms=frozenset({ALG_ALL}),
        il_inline=_SB3_INLINE_LATERAL_PENALTY,
    ),
    "foot_clearance": reward_item(
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
    "slip_penalty": reward_item(
        key="slip_penalty",
        polarity="penalty",
        title="Slip Penalty",
        desc="Penalty for excessive foot or wheel slip against the ground.",
        default=-0.1, min_value=-10.0, max_value=0.0, step=0.05,
        applicable_families=LOCOMOTION_FAMILIES,
        backends=frozenset({BACKEND_SB3}),
        algorithms=frozenset({ALG_ALL}),
        il_inline=_SB3_INLINE_SLIP_PENALTY,
    ),
    "feet_slide": reward_item(
        key="feet_slide",
        polarity="penalty",
        title="Feet Slide",
        desc="Contact-force-based penalty on foot lateral velocity while in ground contact. "
             "Uses contact detection to weight the sliding cost.",
        default=-0.1, min_value=-5.0, max_value=0.0, step=0.01,
        applicable_families=LEGGED_FAMILIES,
        backends=frozenset({BACKEND_SB3}),
        algorithms=frozenset({ALG_ALL}),
        il_inline=_SB3_INLINE_FEET_SLIDE,
    ),
    "collision_penalty": reward_item(
        key="collision_penalty",
        polarity="penalty",
        title="Collision Penalty",
        desc="Penalty for unwanted collisions with the environment or self.",
        default=-0.2, min_value=-10.0, max_value=0.0, step=0.05,
        backends=frozenset({BACKEND_SB3}),
        algorithms=frozenset({ALG_ALL}),
        il_inline=_SB3_INLINE_COLLISION_PENALTY,
    ),
    "goal_distance": reward_item(
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
    "grasp_success": reward_item(
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
    "reference_tracking": reward_item(
        key="reference_tracking",
        polarity="reward",
        title="Reference Tracking",
        desc="Reward for matching a reference motion trajectory keyframe via Gaussian similarity.",
        default=1.0, min_value=0.0, max_value=10.0, step=0.1,
        applicable_families=LEGGED_FAMILIES,
        backends=frozenset({BACKEND_SB3}),
        algorithms=frozenset({ALG_PPO, ALG_AMP}),
        il_inline=_SB3_INLINE_REFERENCE_TRACKING,
    ),
    "base_height_penalty": reward_item(
        key="base_height_penalty",
        polarity="penalty",
        title="Base Height Penalty",
        desc="Penalty for base height deviating from the nominal standing height (~0.32 m). "
             "Discourages crouching or over-extension during locomotion.",
        default=-1.0, min_value=-10.0, max_value=0.0, step=0.1,
        applicable_families=LOCOMOTION_FAMILIES,
        backends=frozenset({BACKEND_SB3}),
        algorithms=frozenset({ALG_ALL}),
        il_inline=_SB3_INLINE_BASE_HEIGHT_PENALTY,
    ),
    "angular_rate_penalty": reward_item(
        key="angular_rate_penalty",
        polarity="penalty",
        title="Angular Rate Penalty",
        desc="Penalty on squared roll and pitch angular rates. "
             "Reduces trunk wobble and promotes smooth, stable locomotion.",
        default=-0.05, min_value=-5.0, max_value=0.0, step=0.01,
        applicable_families=LEGGED_FAMILIES,
        backends=frozenset({BACKEND_SB3}),
        algorithms=frozenset({ALG_ALL}),
        il_inline=_SB3_INLINE_ANGULAR_RATE_PENALTY,
    ),
    "foot_air_time": reward_item(
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
    "joint_pose_tracking": reward_item(
        key="joint_pose_tracking",
        polarity="reward",
        title="Joint Pose Tracking",
        desc="Gaussian reward for joint positions matching the reference frame. "
             "Falls back to a default standing pose when no reference motion is loaded.",
        default=1.0, min_value=0.0, max_value=10.0, step=0.1,
        applicable_families=LEGGED_FAMILIES,
        backends=frozenset({BACKEND_SB3}),
        algorithms=frozenset({ALG_PPO, ALG_AMP}),
        il_inline=_SB3_INLINE_JOINT_POSE_TRACKING,
    ),
    "joint_vel_tracking": reward_item(
        key="joint_vel_tracking",
        polarity="reward",
        title="Joint Velocity Tracking",
        desc="Gaussian reward for joint velocities matching the reference (finite-diff of "
             "reference frames). Falls back to tracking zero velocity when no reference is loaded.",
        default=0.5, min_value=0.0, max_value=10.0, step=0.05,
        applicable_families=LEGGED_FAMILIES,
        backends=frozenset({BACKEND_SB3}),
        algorithms=frozenset({ALG_PPO, ALG_AMP}),
        il_inline=_SB3_INLINE_JOINT_VEL_TRACKING,
    ),
    "foot_pos_tracking": reward_item(
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
    "lin_vel_z_penalty": reward_item(
        key="lin_vel_z_penalty",
        polarity="penalty",
        title="Lin Vel Z Penalty",
        desc="L2 penalty on vertical linear velocity to discourage bouncing.",
        default=-2.0, min_value=-20.0, max_value=0.0, step=0.1,
        applicable_families=LOCOMOTION_FAMILIES,
        backends=frozenset({BACKEND_SB3}),
        algorithms=frozenset({ALG_ALL}),
        il_inline=_SB3_INLINE_LIN_VEL_Z_PENALTY,
    ),
    "joint_torque_penalty": reward_item(
        key="joint_torque_penalty",
        polarity="penalty",
        title="Joint Torque Penalty",
        desc="L2 penalty on applied joint torques for energy efficiency.",
        default=-0.0002, min_value=-0.01, max_value=0.0, step=0.00005,
        applicable_families=ALL_FAMILIES,
        backends=frozenset({BACKEND_SB3}),
        algorithms=frozenset({ALG_ALL}),
        il_inline=_SB3_INLINE_JOINT_TORQUE_PENALTY,
    ),
    "joint_accel_penalty": reward_item(
        key="joint_accel_penalty",
        polarity="penalty",
        title="Joint Accel Penalty",
        desc="L2 penalty on joint accelerations for smooth motion.",
        default=-2.5e-7, min_value=-0.001, max_value=0.0, step=1e-7,
        applicable_families=ALL_FAMILIES,
        backends=frozenset({BACKEND_SB3}),
        algorithms=frozenset({ALG_ALL}),
        il_inline=_SB3_INLINE_JOINT_ACCEL_PENALTY,
    ),
    "joint_vel_penalty": reward_item(
        key="joint_vel_penalty",
        polarity="penalty",
        title="Joint Vel Penalty",
        desc="L2 penalty on joint velocities — discourages overly fast joint motion.",
        default=-0.001, min_value=-1.0, max_value=0.0, step=0.0005,
        applicable_families=ALL_FAMILIES,
        backends=frozenset({BACKEND_SB3}),
        algorithms=frozenset({ALG_ALL}),
        il_inline=_SB3_INLINE_JOINT_VEL_PENALTY,
    ),
    "dof_pos_limits": reward_item(
        key="dof_pos_limits",
        polarity="penalty",
        title="Joint Pos Limits",
        desc="Penalty when joint positions approach or exceed soft limits.",
        default=-5.0, min_value=-20.0, max_value=0.0, step=0.5,
        applicable_families=ALL_FAMILIES,
        backends=frozenset({BACKEND_SB3}),
        algorithms=frozenset({ALG_ALL}),
        il_inline=_SB3_INLINE_DOF_POS_LIMITS,
    ),
    "undesired_contacts": reward_item(
        key="undesired_contacts",
        polarity="penalty",
        title="Undesired Contacts",
        desc="Penalty when non-foot bodies make ground contact.",
        default=-1.0, min_value=-10.0, max_value=0.0, step=0.1,
        applicable_families=LOCOMOTION_FAMILIES,
        backends=frozenset({BACKEND_SB3}),
        algorithms=frozenset({ALG_ALL}),
        il_inline=_SB3_INLINE_UNDESIRED_CONTACTS,
    ),
}


def reward_registry() -> Dict[str, TaskModuleItem]:
    return dict(REWARD_REGISTRY)


#: Reward terms that almost every legged-locomotion run benefits from.
#: Used both as the default seed when the canvas wires no rewards AND as
#: the keyset checked by ``warn_missing_recommended_reward_terms`` —
#: a partial canvas dict missing any of these will silently degrade to
#: weight 0 inside the env (``self._reward_terms.get(key, 0.0)``).
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
    """Default reward set seeded into the SB3 env when no canvas wiring exists."""
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


# ═══════════════════════════════════════════════════════════════════════════
# Isaac Lab reward registry
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
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - \\
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
# Walk These Ways parameterised gait rewards
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
    "track_lin_vel_xy": reward_item(
        key="track_lin_vel_xy",
        polarity="reward",
        title="Track Lin Vel XY",
        desc="Exponential tracking reward for commanded XY linear velocity.",
        default=1.5, min_value=0.0, max_value=40.0, step=0.1,
        applicable_families=LOCOMOTION_FAMILIES,
        backends=_IL, algorithms=frozenset({ALG_ALL}),
        il_func="_unitport_track_lin_vel_xy_exp", il_module=IL_MOD_INLINE,
        il_params='"std": {node_std}, "command_name": "base_velocity"',
        il_inline=_INLINE_TRACK_LIN_VEL_XY_EXP,
    ),
    "track_ang_vel_z": reward_item(
        key="track_ang_vel_z",
        polarity="reward",
        title="Track Ang Vel Z",
        desc="Exponential tracking reward for commanded yaw angular velocity.",
        default=0.75, min_value=0.0, max_value=10.0, step=0.05,
        applicable_families=LOCOMOTION_FAMILIES,
        backends=_IL, algorithms=frozenset({ALG_ALL}),
        il_func="_unitport_track_ang_vel_z_exp", il_module=IL_MOD_INLINE,
        il_params='"std": {node_std}, "command_name": "base_velocity"',
        il_inline=_INLINE_TRACK_ANG_VEL_Z_EXP,
    ),

    # ── Base velocity penalties ───────────────────────────────────────
    "lin_vel_z_penalty": reward_item(
        key="lin_vel_z_penalty",
        polarity="penalty",
        title="Lin Vel Z Penalty",
        desc="L2 penalty on vertical linear velocity to discourage bouncing.",
        default=-2.0, min_value=-20.0, max_value=0.0, step=0.1,
        applicable_families=LOCOMOTION_FAMILIES,
        backends=_IL, algorithms=frozenset({ALG_ALL}),
        il_func="_unitport_lin_vel_z_l2", il_module=IL_MOD_INLINE,
        il_inline=_INLINE_LIN_VEL_Z_L2,
    ),
    "ang_vel_xy_penalty": reward_item(
        key="ang_vel_xy_penalty",
        polarity="penalty",
        title="Ang Vel XY Penalty",
        desc="L2 penalty on roll/pitch angular velocity.",
        default=-0.05, min_value=-5.0, max_value=0.0, step=0.005,
        applicable_families=LOCOMOTION_FAMILIES,
        backends=_IL, algorithms=frozenset({ALG_ALL}),
        il_func="_unitport_ang_vel_xy_l2", il_module=IL_MOD_INLINE,
        il_inline=_INLINE_ANG_VEL_XY_L2,
    ),
    "lin_vel_xy_penalty": reward_item(
        key="lin_vel_xy_penalty",
        polarity="penalty",
        title="Lin Vel XY Penalty",
        desc="L2 penalty on horizontal linear velocity — unconditional. "
             "Use heavy weight for stand (kill drift) or moderate for turn "
             "(discourage forward leak during yaw).",
        default=-0.5, min_value=-20.0, max_value=0.0, step=0.1,
        applicable_families=LOCOMOTION_FAMILIES,
        backends=_IL, algorithms=frozenset({ALG_ALL}),
        il_func="_unitport_lin_vel_xy_l2", il_module=IL_MOD_INLINE,
        il_inline=_INLINE_LIN_VEL_XY_L2,
    ),
    "ang_vel_z_penalty": reward_item(
        key="ang_vel_z_penalty",
        polarity="penalty",
        title="Ang Vel Z Penalty",
        desc="L2 penalty on yaw angular velocity — unconditional. "
             "Use for items that should not spin (stand / strafe / pace).",
        default=-0.1, min_value=-10.0, max_value=0.0, step=0.05,
        applicable_families=LOCOMOTION_FAMILIES,
        backends=_IL, algorithms=frozenset({ALG_ALL}),
        il_func="_unitport_ang_vel_z_l2", il_module=IL_MOD_INLINE,
        il_inline=_INLINE_ANG_VEL_Z_L2,
    ),

    # ── Joint / actuation penalties ───────────────────────────────────
    "joint_torque_penalty": reward_item(
        key="joint_torque_penalty",
        polarity="penalty",
        title="Joint Torque Penalty",
        desc="L2 penalty on joint torques for energy efficiency.",
        default=-0.0002, min_value=-0.01, max_value=0.0, step=0.00005,
        applicable_families=ALL_FAMILIES,
        backends=_IL, algorithms=frozenset({ALG_ALL}),
        il_func="_unitport_joint_torques_l2", il_module=IL_MOD_INLINE,
        il_inline=_INLINE_JOINT_TORQUES_L2,
    ),
    "joint_accel_penalty": reward_item(
        key="joint_accel_penalty",
        polarity="penalty",
        title="Joint Accel Penalty",
        desc="L2 penalty on joint accelerations for smooth motion.",
        default=-2.5e-7, min_value=-0.001, max_value=0.0, step=1e-7,
        applicable_families=ALL_FAMILIES,
        backends=_IL, algorithms=frozenset({ALG_ALL}),
        il_func="_unitport_joint_acc_l2", il_module=IL_MOD_INLINE,
        il_inline=_INLINE_JOINT_ACC_L2,
    ),
    "action_rate_penalty": reward_item(
        key="action_rate_penalty",
        polarity="penalty",
        title="Action Rate Penalty",
        desc="L2 penalty on action rate of change for smooth control.",
        default=-0.05, min_value=-1.0, max_value=0.0, step=0.005,
        applicable_families=ALL_FAMILIES,
        backends=_IL, algorithms=frozenset({ALG_ALL}),
        il_func="_unitport_action_rate_l2", il_module=IL_MOD_INLINE,
        il_inline=_INLINE_ACTION_RATE_L2,
    ),
    "joint_vel_penalty": reward_item(
        key="joint_vel_penalty",
        polarity="penalty",
        title="Joint Vel Penalty",
        desc="L2 penalty on joint velocities — discourages overly fast joint motion.",
        default=-0.001, min_value=-1.0, max_value=0.0, step=0.0005,
        applicable_families=ALL_FAMILIES,
        backends=_IL, algorithms=frozenset({ALG_ALL}),
        il_func="_unitport_joint_vel_l2", il_module=IL_MOD_INLINE,
        il_params='"asset_cfg": SceneEntityCfg("robot")',
        il_inline=_INLINE_JOINT_VEL_L2,
    ),
    "energy_penalty": reward_item(
        key="energy_penalty",
        polarity="penalty",
        title="Energy Penalty",
        desc="L2 penalty on torque × velocity product — minimises mechanical energy expenditure.",
        default=-2e-5, min_value=-0.01, max_value=0.0, step=1e-5,
        applicable_families=ALL_FAMILIES,
        backends=_IL, algorithms=frozenset({ALG_ALL}),
        il_func="_unitport_energy", il_module=IL_MOD_INLINE,
        il_params='"asset_cfg": SceneEntityCfg("robot")',
        il_inline=_INLINE_ENERGY,
    ),

    # ── Orientation / posture ─────────────────────────────────────────
    "flat_orientation": reward_item(
        key="flat_orientation",
        polarity="penalty",
        title="Flat Orientation",
        desc="L2 penalty on projected gravity deviation from vertical — keeps the base level.",
        default=-5.0, min_value=-20.0, max_value=0.0, step=0.1,
        applicable_families=LOCOMOTION_FAMILIES,
        backends=_IL, algorithms=frozenset({ALG_ALL}),
        il_func="_unitport_flat_orientation_l2", il_module=IL_MOD_INLINE,
        il_inline=_INLINE_FLAT_ORIENTATION_L2,
    ),
    "base_height": reward_item(
        key="base_height",
        polarity="penalty",
        title="Base Height",
        desc="L2 penalty on base height deviating from a target standing height. "
             "Needs params: target_height (default 0.34 for quadruped, 0.78 for biped).",
        default=-10.0, min_value=-50.0, max_value=0.0, step=0.5,
        applicable_families=LOCOMOTION_FAMILIES,
        backends=_IL, algorithms=frozenset({ALG_ALL}),
        il_func="_unitport_base_height_l2", il_module=IL_MOD_INLINE,
        # ``{robot_target_height}`` is substituted by the IL config compiler
        # from the Robot canvas node (slider override -> asset.nominal_height
        # -> 0.34 fallback). Never hardcode a per-robot constant here.
        il_params='"target_height": {robot_target_height}, "asset_cfg": SceneEntityCfg("robot")',
        il_inline=_INLINE_BASE_HEIGHT,
    ),
    "alive_reward": reward_item(
        key="alive_reward",
        polarity="reward",
        title="Alive Bonus",
        desc="Constant per-step survival bonus — encourages the policy to keep the robot alive.",
        default=0.15, min_value=0.0, max_value=5.0, step=0.05,
        applicable_families=LOCOMOTION_FAMILIES,
        backends=_IL, algorithms=frozenset({ALG_ALL}),
        il_func="_unitport_is_alive", il_module=IL_MOD_INLINE,
        il_inline=_INLINE_IS_ALIVE,
    ),

    # ── Feet / gait ───────────────────────────────────────────────────
    "feet_air_time": reward_item(
        key="feet_air_time",
        polarity="reward",
        title="Feet Air Time",
        desc="Reward for maintaining contact schedule (gait pattern).",
        default=0.125, min_value=0.0, max_value=5.0, step=0.025,
        applicable_families=LEGGED_FAMILIES,
        backends=_IL, algorithms=frozenset({ALG_ALL}),
        il_func="_unitport_feet_air_time", il_module=IL_MOD_INLINE,
        il_params='"threshold": {node_threshold}, "command_name": "base_velocity", '
                  '"sensor_cfg": SceneEntityCfg("contact_forces", body_names={ir:feet})',
        il_inline=_INLINE_FEET_AIR_TIME,
    ),
    "gait": reward_item(
        key="gait",
        polarity="reward",
        title="Gait Rhythm",
        desc="Periodic gait reward — encourages regular alternating footfalls with a target "
             "period and phase offset. Needs params: period (s), offset (per-foot phase array).",
        default=0.5, min_value=0.0, max_value=5.0, step=0.05,
        applicable_families=LEGGED_FAMILIES,
        backends=_IL, algorithms=frozenset({ALG_PPO, ALG_AMP}),
        il_func="_unitport_feet_gait", il_module=IL_MOD_INLINE,
        il_params='"period": 0.8, "offset": [0.0, 0.5, 0.5, 0.0], '
                  '"sensor_cfg": SceneEntityCfg("contact_forces", body_names={ir:feet}), '
                  '"threshold": 0.5, "command_name": "base_velocity"',
        il_inline=_INLINE_FEET_GAIT,
    ),
    # ── Walk These Ways parameterised gait (reads from the
    #    UniformGaitCommand term emitted by the IL compiler when the
    #    Training Commands node has gait_enabled). ──
    "track_gait_phase": reward_item(
        key="track_gait_phase",
        polarity="reward",
        title="Track Gait Phase",
        desc="Reward foot contact matching the expected stance/swing phase from a Walk These "
             "Ways gait command term. Requires Training Commands with gait_enabled.",
        default=0.5, min_value=0.0, max_value=5.0, step=0.05,
        applicable_families=LEGGED_FAMILIES,
        backends=_IL, algorithms=frozenset({ALG_PPO, ALG_AMP}),
        il_func="_unitport_track_gait_phase", il_module=IL_MOD_INLINE,
        il_params='"command_name": "gait_command", '
                  '"sensor_cfg": SceneEntityCfg("contact_forces", body_names={ir:feet})',
        il_inline=_INLINE_TRACK_GAIT_PHASE,
    ),
    "track_body_height_cmd": reward_item(
        key="track_body_height_cmd",
        polarity="reward",
        title="Track Body Height Cmd",
        desc="Exponential reward for base height matching the commanded body_height. Pairs with "
             "the Walk These Ways gait command — tightens body_height tracking during training.",
        default=0.3, min_value=0.0, max_value=5.0, step=0.05,
        applicable_families=LEGGED_FAMILIES,
        backends=_IL, algorithms=frozenset({ALG_PPO, ALG_AMP}),
        il_func="_unitport_track_body_height_cmd", il_module=IL_MOD_INLINE,
        il_params='"command_name": "gait_command", "std": 0.05',
        il_inline=_INLINE_TRACK_BODY_HEIGHT_CMD,
    ),
    "track_swing_height_cmd": reward_item(
        key="track_swing_height_cmd",
        polarity="reward",
        title="Track Swing Height Cmd",
        desc="Reward swing-phase foot apex height matching the commanded step_height. Requires a "
             "gait command term; feet list auto-resolved via IR mapping.",
        default=0.3, min_value=0.0, max_value=5.0, step=0.05,
        applicable_families=LEGGED_FAMILIES,
        backends=_IL, algorithms=frozenset({ALG_PPO, ALG_AMP}),
        il_func="_unitport_track_swing_height_cmd", il_module=IL_MOD_INLINE,
        il_params='"command_name": "gait_command", '
                  '"asset_cfg": SceneEntityCfg("robot", body_names={ir:feet}), '
                  '"std": 0.02',
        il_inline=_INLINE_TRACK_SWING_HEIGHT_CMD,
    ),
    "feet_clearance": reward_item(
        key="feet_clearance",
        polarity="reward",
        title="Feet Clearance",
        desc="Reward for swing-foot height reaching a target clearance above ground. "
             "Needs params: target_height (m, default 0.1).",
        default=1.0, min_value=0.0, max_value=10.0, step=0.1,
        applicable_families=LEGGED_FAMILIES,
        backends=_IL, algorithms=frozenset({ALG_ALL}),
        il_func="_unitport_foot_clearance_reward", il_module=IL_MOD_INLINE,
        il_params='"asset_cfg": SceneEntityCfg("robot"), "target_height": 0.1, '
                  '"std": 0.05, "tanh_mult": 2.0',
        il_inline=_INLINE_FOOT_CLEARANCE,
    ),
    "feet_slide": reward_item(
        key="feet_slide",
        polarity="penalty",
        title="Feet Slide",
        desc="L2 penalty on foot velocity while in contact — discourages sliding/skating.",
        default=-0.2, min_value=-5.0, max_value=0.0, step=0.05,
        applicable_families=LEGGED_FAMILIES,
        backends=_IL, algorithms=frozenset({ALG_ALL}),
        il_func="_unitport_feet_slide", il_module=IL_MOD_INLINE,
        il_params='"sensor_cfg": SceneEntityCfg("contact_forces", body_names={ir:feet}), '
                  '"asset_cfg": SceneEntityCfg("robot")',
        il_inline=_INLINE_FEET_SLIDE,
    ),

    # ── Contact / safety ──────────────────────────────────────────────
    "undesired_contacts": reward_item(
        key="undesired_contacts",
        polarity="penalty",
        title="Undesired Contacts",
        desc="Penalty when non-foot bodies (torso, thighs, shoulders) make ground contact. "
             "Needs params: sensor_cfg with body_names regex.",
        default=-1.0, min_value=-10.0, max_value=0.0, step=0.1,
        applicable_families=LOCOMOTION_FAMILIES,
        backends=_IL, algorithms=frozenset({ALG_ALL}),
        il_func="_unitport_undesired_contacts", il_module=IL_MOD_INLINE,
        il_params='"threshold": 1.0, '
                  '"sensor_cfg": SceneEntityCfg("contact_forces", body_names={ir:thighs_hips_base})',
        il_inline=_INLINE_UNDESIRED_CONTACTS,
    ),

    # ── Joint limits ──────────────────────────────────────────────────
    "dof_pos_limits": reward_item(
        key="dof_pos_limits",
        polarity="penalty",
        title="Joint Pos Limits",
        desc="Penalty when joint positions approach or exceed soft limits — "
             "keeps joints within safe operating range.",
        default=-5.0, min_value=-20.0, max_value=0.0, step=0.5,
        applicable_families=ALL_FAMILIES,
        backends=_IL, algorithms=frozenset({ALG_ALL}),
        il_func="_unitport_joint_pos_limits", il_module=IL_MOD_INLINE,
        il_params='"asset_cfg": SceneEntityCfg("robot")',
        il_inline=_INLINE_JOINT_POS_LIMITS,
    ),
    "joint_deviation_l1": reward_item(
        key="joint_deviation_l1",
        polarity="penalty",
        title="Joint Deviation L1",
        desc="L1 penalty on joint positions deviating from the asset's default pose. "
             "Anchors the policy to a symmetric nominal stance — counters left/right "
             "asymmetric drift and posture sinking.",
        default=-0.05, min_value=-2.0, max_value=0.0, step=0.01,
        applicable_families=ALL_FAMILIES,
        backends=_IL, algorithms=frozenset({ALG_ALL}),
        il_func="_unitport_joint_deviation_l1", il_module=IL_MOD_INLINE,
        il_params='"asset_cfg": SceneEntityCfg("robot")',
        il_inline=_INLINE_JOINT_DEVIATION_L1,
    ),
}


def il_reward_registry() -> Dict[str, TaskModuleItem]:
    return dict(IL_REWARD_REGISTRY)


def default_il_reward_terms() -> Dict[str, float]:
    return {k: IL_REWARD_REGISTRY[k].default for k in IL_REWARD_REGISTRY}


__all__ = [
    "REWARD_REGISTRY",
    "IL_REWARD_REGISTRY",
    "RECOMMENDED_LOCOMOTION_REWARD_TERMS",
    "reward_registry",
    "il_reward_registry",
    "default_reward_terms",
    "default_il_reward_terms",
    "warn_missing_recommended_reward_terms",
]
