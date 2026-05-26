# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""SB3 reward & termination registry — numpy implementations.

The canvas's ``rewards`` and ``terminations`` nodes serialize to:

    spec.rewards.terms           = {term_key: float | {"weight": float, ...}, ...}
    spec.terminations.conditions = {kind_key: float, ...}

:func:`build_reward_fn` and :func:`build_done_fn` compile those dicts into
closures ``(env) -> float`` / ``(env) -> bool`` that
:class:`application.training.envs.generic_mujoco_env.GenericMujocoEnv`
calls each step. Unknown term/kind keys are warned once and skipped — the
canvas can carry forward-compat names without crashing the env.

The set of supported reward keys mirrors the SB3 entries in
``scripts.rewards.registry.REWARD_REGISTRY``; the set of supported
termination keys mirrors ``scripts.terminations.registry.TERMINATION_REGISTRY``.
Items that need contact / foot tracking (``foot_clearance``, ``feet_slide``,
``undesired_contacts``, ``foot_air_time``, …) are intentionally not
implemented in v1 — Stage 6 v1 env doesn't expose contact state, so they
would silently degrade to zero. They will land alongside the contact-history
work that fixes those env attributes.
"""
from __future__ import annotations

import functools
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from unitport_sdk import log_warning

if TYPE_CHECKING:
    from application.training.envs.generic_mujoco_env import GenericMujocoEnv

EnvRewardFn = Callable[["GenericMujocoEnv"], float]
EnvDoneFn = Callable[["GenericMujocoEnv"], bool]


# ---------------------------------------------------------------------------
# Weight extraction — terms[k] may be a float or a dict with "weight".
# ---------------------------------------------------------------------------


def _weight_of(value: Any) -> float:
    if isinstance(value, dict):
        try:
            return float(value.get("weight", 0.0))
        except (TypeError, ValueError):
            return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _value_of(value: Any) -> Optional[float]:
    """Per-item "Value" chip (a reward's function-internal param) or None.

    Mirrors :func:`application.compiler.term_payload.parse_item_value`: only
    the structured dict form carries a ``"value"`` key; scalar / absent →
    ``None`` ("unset, use the reward's own default / auto").
    """
    if isinstance(value, dict) and "value" in value:
        try:
            return float(value.get("value"))
        except (TypeError, ValueError):
            return None
    return None


# ---------------------------------------------------------------------------
# Reward primitives — each returns the unweighted scalar.
# Polarity (reward vs penalty) is encoded in the user-supplied weight sign;
# our scripts/rewards/registry.py defaults negative-weight penalties already.
# ---------------------------------------------------------------------------


def _r_velocity_tracking(env) -> float:
    sigma = max(float(getattr(env, "_track_sigma", 0.25)), 1e-3)
    err = (env._lin_vel[0] - env._command[0]) ** 2 + (env._lin_vel[1] - env._command[1]) ** 2
    return float(np.exp(-err / sigma))


def _r_yaw_tracking(env) -> float:
    sigma = max(float(getattr(env, "_track_sigma", 0.25)), 1e-3)
    err = (env._ang_vel[2] - env._command[2]) ** 2
    return float(np.exp(-err / sigma))


def _r_alive(env) -> float:
    return 1.0


def _r_upright(env) -> float:
    return float(np.clip(-env._proj_gravity[2], 0.0, 1.0))


def _r_lateral_penalty(env) -> float:
    return float(abs(env._lin_vel[1] - env._command[1]))


def _r_action_smoothness(env) -> float:
    return float(np.sum(np.square(env._action - env._last_action)))


def _r_lin_vel_z_penalty(env) -> float:
    return float(env._lin_vel[2] ** 2)


def _r_angular_rate_penalty(env) -> float:
    return float(env._ang_vel[0] ** 2 + env._ang_vel[1] ** 2)


def _r_base_height_penalty(env, value: float = 0.0) -> float:
    # ``value`` = the per-item "Value" chip (target standing height, m),
    # bound in by _compile_pairs. 0.0 / unset = "auto" → the env's nominal
    # base z (MJCF keyframe-0 / qpos0), computed once at construction. No
    # silent 0.32 magic (CLAUDE.md §8) and no reach-back to the Robot node.
    target = float(value) if value and value > 0.0 else float(env._nominal_base_z)
    return float((env._base_pos[2] - target) ** 2)


# Reward keys whose function takes a per-item "Value" chip (function-internal
# param). _compile_pairs binds the decoded value into the fn so the runtime
# closure keeps the uniform ``(env) -> float`` contract.
_VALUE_AWARE_REWARDS = frozenset({"base_height_penalty"})


def _r_joint_torque_penalty(env) -> float:
    tau = getattr(env, "_qfrc_actuator", None)
    if tau is None:
        return 0.0
    return float(np.sum(np.square(tau)))


def _r_joint_accel_penalty(env) -> float:
    qacc = getattr(env, "_qacc", None)
    if qacc is None:
        return 0.0
    return float(np.sum(np.square(qacc)))


def _r_joint_vel_penalty(env) -> float:
    return float(np.sum(np.square(env._qvel)))


def _r_energy(env) -> float:
    tau = getattr(env, "_qfrc_actuator", None)
    if tau is None:
        return 0.0
    return float(np.sum(np.abs(env._qvel) * np.abs(tau)))


def _r_dof_pos_limits(env) -> float:
    limits = getattr(env, "_soft_joint_limits", None)
    if not limits:
        return 0.0
    soft_margin = 0.05
    cost = 0.0
    for q, (lo, hi) in zip(env._qpos, limits):
        span = hi - lo
        if span <= 0.0:
            continue
        margin = span * soft_margin
        below = max(0.0, (lo + margin) - float(q))
        above = max(0.0, float(q) - (hi - margin))
        cost += below * below + above * above
    return float(cost)


_REWARD_FNS: Dict[str, EnvRewardFn] = {
    "velocity_tracking": _r_velocity_tracking,
    "yaw_tracking": _r_yaw_tracking,
    "alive": _r_alive,
    "upright": _r_upright,
    "lateral_penalty": _r_lateral_penalty,
    "action_smoothness": _r_action_smoothness,
    "lin_vel_z_penalty": _r_lin_vel_z_penalty,
    "angular_rate_penalty": _r_angular_rate_penalty,
    "base_height_penalty": _r_base_height_penalty,
    "joint_torque_penalty": _r_joint_torque_penalty,
    "joint_accel_penalty": _r_joint_accel_penalty,
    "joint_vel_penalty": _r_joint_vel_penalty,
    "energy": _r_energy,
    "dof_pos_limits": _r_dof_pos_limits,
}


# ---------------------------------------------------------------------------
# Termination primitives — each takes (env, threshold) -> bool
# ---------------------------------------------------------------------------


def _t_min_height(env, threshold: float) -> bool:
    return float(env._base_pos[2]) < float(threshold)


def _t_fall_tilt(env, threshold: float) -> bool:
    # We don't separate roll vs pitch on this env — both terms collapse to a
    # single "tilt magnitude" check (sqrt(g_x^2 + g_y^2) of projected gravity).
    # When upright, proj_gravity == [0, 0, -1]; tilt grows with body lean.
    tilt = float(np.sqrt(env._proj_gravity[0] ** 2 + env._proj_gravity[1] ** 2))
    return tilt > float(threshold)


def _t_joint_limit_violation(env, threshold: float) -> bool:
    limits = getattr(env, "_soft_joint_limits", None)
    if not limits:
        return False
    count = 0
    for q, (lo, hi) in zip(env._qpos, limits):
        if (hi - lo) <= 0.0:
            continue
        if float(q) < lo or float(q) > hi:
            count += 1
    return count >= int(threshold)


_DONE_FNS: Dict[str, Callable[[Any, float], bool]] = {
    # SB3 termination keys
    "min_height": _t_min_height,
    "fall_threshold_roll": _t_fall_tilt,
    "fall_threshold_pitch": _t_fall_tilt,
    "joint_limit_violation": _t_joint_limit_violation,
    # Mirror of the IL key — same semantics on this env (base z below threshold).
    "base_height": _t_min_height,
}


# ---------------------------------------------------------------------------
# Compile-time builders — log_warning once per unknown key, do not crash.
# ---------------------------------------------------------------------------


_warned_unknown_reward_keys: set = set()
_warned_unknown_done_keys: set = set()


def _compile_pairs(terms: Any) -> List[Tuple[str, EnvRewardFn, float]]:
    pairs: List[Tuple[str, EnvRewardFn, float]] = []
    if not isinstance(terms, dict):
        return pairs
    for key, val in terms.items():
        weight = _weight_of(val)
        if weight == 0.0:
            continue
        fn = _REWARD_FNS.get(str(key))
        if fn is None:
            if key not in _warned_unknown_reward_keys:
                _warned_unknown_reward_keys.add(key)
                log_warning(
                    f"[reward_terms] unknown reward kind {key!r}; skipping. "
                    f"Known: {sorted(_REWARD_FNS)}"
                )
            continue
        # Bind the per-item "Value" chip into value-aware reward fns so the
        # runtime closure stays ``(env) -> float``. Unset → 0.0 ("auto").
        if str(key) in _VALUE_AWARE_REWARDS:
            v = _value_of(val)
            fn = functools.partial(fn, value=float(v) if v is not None else 0.0)
        pairs.append((str(key), fn, weight))
    return pairs


def build_reward_fn(terms: Any) -> EnvRewardFn:
    """Compile a ``{term_key: weight | {weight,...}}`` dict into a reward closure.

    Returns a callable ``(env) -> float`` that sums every recognised term × weight.
    Unknown term keys are warned once and contribute zero. An empty / non-dict
    input collapses to a constant-zero closure — callers can detect this case
    with :data:`build_reward_fn.is_empty` if they want to fall back to a
    default reward.
    """
    pairs = _compile_pairs(terms)
    if not pairs:
        return _zero_reward

    def _compose(env) -> float:
        total = 0.0
        for _name, fn, w in pairs:
            total += fn(env) * w
        return float(total)

    return _compose


# Companion to :func:`build_reward_fn` that returns a per-term breakdown
# alongside the scalar total. The breakdown is what feeds the SB3 reward
# logger (info["reward/<term>"]) and the per-item Tensorboard slices —
# see ``sb3_reward_log.RewardBreakdownCallback``.
EnvRewardFnRich = Callable[["GenericMujocoEnv"], Tuple[float, Dict[str, float]]]


def build_reward_fn_with_breakdown(terms: Any) -> EnvRewardFnRich:
    """Like :func:`build_reward_fn` but returns ``(env) -> (total, breakdown)``.

    ``breakdown`` is ``{term_key: weighted_contribution}`` for every term
    that actually fired this step. Unknown / zero-weight keys are omitted.
    Empty input collapses to a closure that returns ``(0.0, {})``.
    """
    pairs = _compile_pairs(terms)
    if not pairs:
        return _zero_reward_rich

    def _compose(env) -> Tuple[float, Dict[str, float]]:
        total = 0.0
        breakdown: Dict[str, float] = {}
        for name, fn, w in pairs:
            contribution = fn(env) * w
            total += contribution
            breakdown[name] = float(contribution)
        return float(total), breakdown

    return _compose


def build_done_fn(conditions: Any) -> EnvDoneFn:
    """Compile a ``{kind_key: threshold}`` dict into a termination closure.

    Always returns True on non-finite physics state. Unknown kinds warned once
    and skipped. Empty input → only the non-finite guard fires.
    """
    pairs: List[Tuple[Callable[[Any, float], bool], float]] = []
    if isinstance(conditions, dict):
        for key, val in conditions.items():
            try:
                threshold = float(val)
            except (TypeError, ValueError):
                continue
            fn = _DONE_FNS.get(str(key))
            if fn is None:
                if key not in _warned_unknown_done_keys:
                    _warned_unknown_done_keys.add(key)
                    log_warning(
                        f"[reward_terms] unknown termination kind {key!r}; skipping. "
                        f"Known: {sorted(_DONE_FNS)}"
                    )
                continue
            pairs.append((fn, threshold))

    def _compose(env) -> bool:
        if not (np.all(np.isfinite(env._qpos)) and np.all(np.isfinite(env._qvel))):
            return True
        for fn, th in pairs:
            try:
                if fn(env, th):
                    return True
            except Exception:  # noqa: BLE001
                continue
        return False

    return _compose


def _zero_reward(env) -> float:
    return 0.0


def _zero_reward_rich(env) -> Tuple[float, Dict[str, float]]:
    return 0.0, {}


def known_reward_kinds() -> Tuple[str, ...]:
    return tuple(sorted(_REWARD_FNS))


def known_termination_kinds() -> Tuple[str, ...]:
    return tuple(sorted(_DONE_FNS))


__all__ = [
    "build_reward_fn",
    "build_reward_fn_with_breakdown",
    "build_done_fn",
    "known_reward_kinds",
    "known_termination_kinds",
]
