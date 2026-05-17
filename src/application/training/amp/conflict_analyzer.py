"""AMP reward-vs-discriminator conflict analyser.

Migrated from DEMO ``src/system/training/amp_conflict_detector.py``.
Pure static analysis — no Qt, no training, no env build. Used by the
``training_motion`` node's ``reward_conflict_table`` UI widget (Analyze
button) to surface reward terms that fight the AMP discriminator on the
selected reference motion clips, and suggest safer weights.

Public surface (UI + spec_compiler):

    evaluate_rewards_on_clips(reward_terms, motion_clips, discriminator_cfg)
        → AMPHelperReport

    AMPHelperReport.items: List[RewardEvalItem]
        Each item carries (key, display_name, current_weight,
        expert_mean_score, verdict, suggested_weight, reason) and is
        what the canvas table renders + writes back into the node's
        ``reward_analysis.analysis_result`` JSON blob.

Verdict semantics:
    * ``"conflict"`` — current weight fights the reference motion;
      suggested_weight is the in-range safe value
    * ``"ok"``       — current weight is aligned with AMP / within
      safe range
    * ``"skip"``     — cannot evaluate offline (contact-dependent or
      missing telemetry)

The dynamics evaluators (compute ``expert_mean_score`` from a real
motion clip's numpy arrays) are stubbed out to ``0.0`` in this initial
RELEASE port. Static-KB verdicts already cover the common conflict
cases for quadruped locomotion. To add real expert-score evaluation:
populate ``_REWARD_EVALUATORS`` and feed clips through them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RewardAMPProfile:
    """AMP conflict metadata for one reward key (static KB entry)."""

    incentivizes: str
    motion_conflicts: Dict[str, str]
    safe_weight_range: Tuple[float, float]
    safe_weight_suggestion: float


@dataclass
class RewardEvalItem:
    """One row of the AMP-helper analysis table.

    Mirrors the JSON record stored in
    ``training_motion.reward_analysis.analysis_result``.
    """

    key: str
    display_name: str
    current_weight: float
    expert_mean_score: float
    verdict: str                    # "conflict" | "ok" | "skip"
    suggested_weight: float
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "display_name": self.display_name,
            "current_weight": self.current_weight,
            "expert_mean_score": self.expert_mean_score,
            "verdict": self.verdict,
            "suggested_weight": self.suggested_weight,
            "reason": self.reason,
        }


@dataclass
class AMPHelperReport:
    """Aggregated result of :func:`evaluate_rewards_on_clips`."""

    items: List[RewardEvalItem] = field(default_factory=list)
    motion_type: str = "unknown"
    n_clips: int = 0
    n_frames_total: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "analysis_result": [it.to_dict() for it in self.items],
            "motion_type": self.motion_type,
            "n_clips": self.n_clips,
            "n_frames_total": self.n_frames_total,
        }


# ---------------------------------------------------------------------------
# Knowledge base (locomotion-relevant rewards). Absent keys yield "skip".
# ---------------------------------------------------------------------------

_KB = RewardAMPProfile

REWARD_AMP_KB: Dict[str, RewardAMPProfile] = {
    # Velocity tracking
    "track_lin_vel_xy": _KB(
        incentivizes="Match commanded XY velocity",
        motion_conflicts={
            "*": "Reference motion has fixed speed profile; command-driven "
                 "velocity tracking pulls the policy away from reference dynamics",
        },
        safe_weight_range=(0.0, 0.8),
        safe_weight_suggestion=0.5,
    ),
    "track_ang_vel_z": _KB(
        incentivizes="Match commanded yaw rate",
        motion_conflicts={
            "*": "Reference motion has fixed heading; yaw-rate tracking at "
                 "non-zero command opposes the reference trajectory",
        },
        safe_weight_range=(0.0, 0.5),
        safe_weight_suggestion=0.3,
    ),
    # Posture / height
    "base_height": _KB(
        incentivizes="Fixed base height (penalises deviation from target)",
        motion_conflicts={
            "walk": "Walking oscillates base height ±5 cm per stride",
            "trot": "Trotting oscillates base height ±3 cm per stride",
            "pace": "Pacing oscillates base height ±4 cm per stride",
        },
        safe_weight_range=(-5.0, 0.0),
        safe_weight_suggestion=-1.0,
    ),
    "flat_orientation": _KB(
        incentivizes="Zero pitch and roll",
        motion_conflicts={
            "walk": "Walking has body pitch oscillation from CoM dynamics",
            "trot": "Trotting has lateral roll from diagonal support",
        },
        safe_weight_range=(-2.5, 0.0),
        safe_weight_suggestion=-0.5,
    ),
    "lin_vel_z_penalty": _KB(
        incentivizes="Zero vertical body velocity",
        motion_conflicts={
            "walk": "Walking produces vertical oscillation at 2× step frequency",
            "trot": "Trotting has vertical velocity from diagonal phase transitions",
        },
        safe_weight_range=(-1.0, 0.0),
        safe_weight_suggestion=-0.2,
    ),
    "ang_vel_xy_penalty": _KB(
        incentivizes="Zero roll/pitch angular velocity",
        motion_conflicts={"walk": "Walking has small pitch-rate oscillation"},
        safe_weight_range=(-0.5, 0.0),
        safe_weight_suggestion=-0.05,
    ),
    # Feet / gait
    "feet_air_time": _KB(
        incentivizes="Maximise time feet spend airborne (longer swing phase)",
        motion_conflicts={
            "*": "Reference gait has specific stance/swing timing; maximising "
                 "air time distorts the natural gait rhythm",
        },
        safe_weight_range=(0.0, 0.15),
        safe_weight_suggestion=0.05,
    ),
    # Joint smoothness penalties (low/no conflict)
    "joint_torque_penalty": _KB(
        incentivizes="Minimise joint torques (energy efficiency)",
        motion_conflicts={},
        safe_weight_range=(-0.01, 0.0),
        safe_weight_suggestion=-0.0002,
    ),
    "joint_accel_penalty": _KB(
        incentivizes="Minimise joint accelerations (smoothness)",
        motion_conflicts={},
        safe_weight_range=(-0.001, 0.0),
        safe_weight_suggestion=-2.5e-7,
    ),
    "action_rate_penalty": _KB(
        incentivizes="Smooth consecutive actions",
        motion_conflicts={},
        safe_weight_range=(-1.0, 0.0),
        safe_weight_suggestion=-0.05,
    ),
    "joint_vel_penalty": _KB(
        incentivizes="Minimise joint velocities",
        motion_conflicts={},
        safe_weight_range=(-1.0, 0.0),
        safe_weight_suggestion=-0.001,
    ),
    "joint_deviation_l1": _KB(
        incentivizes="Stay near default joint pose",
        motion_conflicts={
            "*": "Strong deviation penalty anchors joints to default pose, "
                 "opposing the dynamic joint trajectories in reference motion",
        },
        safe_weight_range=(-0.1, 0.0),
        safe_weight_suggestion=-0.02,
    ),
    "dof_pos_limits": _KB(
        incentivizes="Keep joints within soft limits",
        motion_conflicts={},
        safe_weight_range=(-20.0, 0.0),
        safe_weight_suggestion=-5.0,
    ),
    "alive_reward": _KB(
        incentivizes="Survival bonus (constant per step)",
        motion_conflicts={},
        safe_weight_range=(0.0, 5.0),
        safe_weight_suggestion=0.15,
    ),
    "undesired_contacts": _KB(
        incentivizes="Penalise non-foot body ground contact",
        motion_conflicts={},
        safe_weight_range=(-10.0, 0.0),
        safe_weight_suggestion=-1.0,
    ),
}

_ALWAYS_SAFE = frozenset({"alive_reward", "action_rate_penalty", "dof_pos_limits"})
_CONTACT_DEPENDENT = frozenset({
    "feet_air_time", "feet_slide", "undesired_contacts", "gait",
    "track_gait_phase", "track_body_height_cmd", "track_swing_height_cmd",
    "joint_torque_penalty", "energy_penalty",
})

_DISPLAY_NAMES = {
    "track_lin_vel_xy": "Track Lin Vel XY",
    "track_ang_vel_z": "Track Ang Vel Z",
    "lin_vel_z_penalty": "Lin Vel Z Penalty",
    "ang_vel_xy_penalty": "Ang Vel XY Penalty",
    "base_height": "Base Height",
    "flat_orientation": "Flat Orientation",
    "joint_vel_penalty": "Joint Vel Penalty",
    "joint_accel_penalty": "Joint Accel Penalty",
    "joint_deviation_l1": "Joint Deviation L1",
    "feet_air_time": "Feet Air Time",
    "feet_slide": "Feet Slide",
    "feet_clearance": "Feet Clearance",
    "undesired_contacts": "Undesired Contacts",
    "joint_torque_penalty": "Joint Torque Penalty",
    "energy_penalty": "Energy Penalty",
    "alive_reward": "Alive Reward",
    "action_rate_penalty": "Action Rate Penalty",
    "dof_pos_limits": "Joint Pos Limits",
}

_MOTION_KEYWORDS = {
    "walk": ("walk", "forward", "backward", "stride"),
    "trot": ("trot",),
    "pace": ("pace",),
    "stand": ("stand", "idle"),
    "turn": ("turn", "yaw"),
}


# ---------------------------------------------------------------------------
# Dynamics evaluator registry (stubbed — populate to enable expert scoring)
# ---------------------------------------------------------------------------

_REWARD_EVALUATORS: Dict[str, Callable[..., float]] = {}


def _display_name(key: str) -> str:
    return _DISPLAY_NAMES.get(key, key.replace("_", " ").title())


def _infer_motion_type(filter_tag: str, clip_tags: List[str]) -> str:
    tag = (filter_tag or "").lower()
    for mtype, keywords in _MOTION_KEYWORDS.items():
        if any(kw in tag for kw in keywords):
            return mtype
    for ct in clip_tags:
        t = (ct or "").lower()
        for mtype, keywords in _MOTION_KEYWORDS.items():
            if any(kw in t for kw in keywords):
                return mtype
    return "unknown"


def _classify(key: str, weight: float, motion_type: str) -> Tuple[str, float, str]:
    """Return (verdict, suggested_weight, reason) for a single reward term.

    Pure static — uses ``REWARD_AMP_KB`` + per-key category tables.
    Dynamics-derived ``expert_mean_score`` is filled in separately when an
    evaluator is registered.
    """
    if weight == 0:
        return ("skip", weight, "Weight is zero — not active")

    if key in _ALWAYS_SAFE:
        return ("ok", weight, "Safe — aligned with AMP")

    profile = REWARD_AMP_KB.get(key)
    if profile is None:
        return ("skip", weight, "No AMP knowledge entry — skipped")

    lo, hi = profile.safe_weight_range
    in_range = lo <= weight <= hi

    motion_specific = profile.motion_conflicts.get(motion_type)
    universal = profile.motion_conflicts.get("*")
    has_conflict = motion_specific is not None or universal is not None

    if key in _CONTACT_DEPENDENT:
        if in_range and not has_conflict:
            return ("ok", weight, "Safe — within recommended range")
        if not in_range:
            reason = (motion_specific or universal
                      or f"Weight {weight} outside safe range {profile.safe_weight_range}")
            return ("conflict", profile.safe_weight_suggestion, f"[Static] {reason}")
        return ("skip", weight, "Needs contact/actuator data — cannot evaluate offline")

    if has_conflict and not in_range:
        reason = motion_specific or universal or "Weight outside safe range"
        return ("conflict", profile.safe_weight_suggestion, f"[Static] {reason}")

    if in_range:
        return ("ok", weight, f"Within safe range {profile.safe_weight_range}")

    return ("conflict", profile.safe_weight_suggestion,
            f"Weight {weight} outside safe range {profile.safe_weight_range}")


def evaluate_rewards_on_clips(
    reward_terms: Dict[str, float],
    motion_clips: Optional[List[Path]] = None,
    discriminator_cfg: Optional[Dict[str, Any]] = None,
    motion_task_filter: str = "",
) -> AMPHelperReport:
    """Analyse reward-vs-discriminator conflicts on reference motion clips.

    Args:
        reward_terms: ``{reward_key: weight_float}`` from the upstream
            ``rewards`` node's ``reward_terms`` parameter.
        motion_clips: optional list of motion-clip file paths. Used to
            infer motion type via filename keywords and (when dynamics
            evaluators are registered) to compute ``expert_mean_score``.
            Pass ``None`` or empty list to fall back to static-KB only.
        discriminator_cfg: optional discriminator-node parameter dict.
            Reserved for future lerp_schedule / amp_reward_coef-aware
            severity scoring.
        motion_task_filter: optional motion-task tag string from
            ``reference_motion_config.motion_task_filter``.

    Returns:
        :class:`AMPHelperReport` whose ``items`` populate the
        ``reward_analysis.analysis_result`` JSON list.
    """
    motion_clips = motion_clips or []
    clip_tags = [Path(p).stem for p in motion_clips]
    motion_type = _infer_motion_type(motion_task_filter, clip_tags)

    items: List[RewardEvalItem] = []
    for key, weight in sorted(reward_terms.items(),
                              key=lambda kv: abs(kv[1] or 0.0),
                              reverse=True):
        try:
            w = float(weight) if not isinstance(weight, dict) else float(weight.get("weight", 0.0))
        except (TypeError, ValueError):
            continue
        verdict, suggested, reason = _classify(key, w, motion_type)
        evaluator = _REWARD_EVALUATORS.get(key)
        expert_score = 0.0
        if evaluator is not None and motion_clips:
            try:
                expert_score = float(evaluator(motion_clips))
            except Exception:
                expert_score = 0.0
        items.append(RewardEvalItem(
            key=key,
            display_name=_display_name(key),
            current_weight=w,
            expert_mean_score=expert_score,
            verdict=verdict,
            suggested_weight=suggested,
            reason=reason,
        ))

    return AMPHelperReport(
        items=items,
        motion_type=motion_type,
        n_clips=len(motion_clips),
        n_frames_total=0,
    )
