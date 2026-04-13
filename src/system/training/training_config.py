#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase B — Typed training configuration dataclasses.

Each dataclass mirrors one Training Ground node's output payload.
All fields use Python-native types (int, float, bool, list, dict, str).
String-based node parameters (as stored in TrainingBaseNode.parameters)
are normalized by each ``from_node_dict()`` classmethod.

No Qt dependencies.  No real training backend.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.system.training.task_module_registry import (
    default_reward_terms,
    default_termination_conditions,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bool(v: Any, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes")
    if isinstance(v, int):
        return bool(v)
    return default


def _int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


# Migrate old short-form action_type values (written by pre-Phase-5 canvas
# saves) to the unified runtime vocabulary.  Applied in every from_node_dict()
# that reads action_type so existing experiment JSON round-trips cleanly.
_ACTION_TYPE_MIGRATION: Dict[str, str] = {
    "position": "joint_position",
    "velocity": "joint_velocity",
}


def _action_type(v: Any, default: str = "joint_position") -> str:
    """Normalize and migrate an action_type string."""
    s = str(v).strip().lower() if v else ""
    return _ACTION_TYPE_MIGRATION.get(s, s) if s else default


def resolve_control_timing(sim_dt: Any, control_dt: Any) -> Dict[str, float]:
    """Normalize physics timing and derive control/runtime cadence."""
    sim = max(1e-4, _float(sim_dt, 0.002))
    requested_ctrl = max(sim, _float(control_dt, 0.02))
    decimation = max(1, int(round(requested_ctrl / sim)))
    effective_ctrl = sim * decimation
    return {
        "sim_dt": sim,
        "control_dt": effective_ctrl,
        "decimation": decimation,
        "control_frequency_hz": 1.0 / effective_ctrl if effective_ctrl > 0 else 50.0,
        "timing_exact": math.isclose(
            requested_ctrl,
            effective_ctrl,
            rel_tol=0.0,
            abs_tol=max(1e-6, sim * 0.01),
        ),
    }


def _json_load(v: Any, default: Any = None) -> Any:
    """Parse a JSON string; return ``default`` on any error."""
    if isinstance(v, (dict, list)):
        return v
    if isinstance(v, str):
        try:
            return json.loads(v)
        except (json.JSONDecodeError, ValueError):
            return default
    return default


def _int_list(v: Any, default: Optional[List[int]] = None) -> List[int]:
    """Parse a JSON array of integers (e.g. '[256, 256]')."""
    parsed = _json_load(v, default)
    if isinstance(parsed, list):
        try:
            return [int(x) for x in parsed]
        except (TypeError, ValueError):
            pass
    return default if default is not None else []


def _float_range(v: Any, default: Optional[List[float]] = None) -> List[float]:
    """Parse a JSON array of two floats (e.g. '[0.8, 1.2]')."""
    parsed = _json_load(v, default)
    if isinstance(parsed, list) and len(parsed) >= 2:
        try:
            return [float(parsed[0]), float(parsed[1])]
        except (TypeError, ValueError):
            pass
    return default if default is not None else [0.0, 1.0]


# ---------------------------------------------------------------------------
# RobotSpec
# ---------------------------------------------------------------------------

@dataclass
class RobotSpec:
    """Typed output of the unified Robot node — identity + structure only.

    Populated from the canvas ``robot`` node's parameters via
    :meth:`from_node_dict`. When ``asset_id`` resolves in the
    ``robot_assets`` registry, file paths and ``joint_names`` are
    auto-populated from the registry entry.

    **Pure asset identity / structure.** Editable settings (init pose,
    contact sensor topology, actuator gains) live on :class:`ActorConfig`
    (the sibling dataclass owned by ``ActorSettingNode``).
    """

    robot_type: str = "go2"
    mjcf_path: str = ""
    usd_path: str = ""
    urdf_path: str = ""
    joint_config: Dict[str, Any] = field(default_factory=dict)
    action_dim: int = 0                                        # 0 = unknown; populated at runtime from MJCF
    joint_names: List[str] = field(default_factory=list)       # from active asset file

    # ── Registry linkage (AMP_design.yaml §3.custom_robot_assets) ──
    asset_id: str = ""
    family: str = ""

    # ── Asset-level display metadata ──
    active_file: str = "auto"                                  # auto | mjcf | usd | urdf
    body_mapping: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_node_dict(cls, params: Dict[str, str]) -> "RobotSpec":
        action_dim = 0
        try:
            action_dim = int(params.get("action_dim", 0) or 0)
        except (TypeError, ValueError):
            pass

        # --- Asset registry lookup ---
        asset_id = str(params.get("asset_id", "") or "").strip()
        mjcf_path = str(params.get("mjcf_path", "") or "")
        usd_path = str(params.get("usd_path", "") or "")
        urdf_path = str(params.get("urdf_path", "") or "")
        robot_type = str(params.get("robot_type", "") or "")
        family = ""
        joint_names: List[str] = []

        if asset_id:
            try:
                from src.system.training.robot_assets import (
                    get_asset, rescan,
                )
                rescan()
                asset = get_asset(asset_id)
                if not mjcf_path and asset.mjcf_path is not None:
                    mjcf_path = str(asset.mjcf_path)
                if not usd_path:
                    if asset.usd_path is not None:
                        usd_path = str(asset.usd_path)
                    elif asset.usd_url:
                        usd_path = str(asset.usd_url)
                if not urdf_path and asset.urdf_path is not None:
                    urdf_path = str(asset.urdf_path)
                joint_names = list(asset.joint_names or [])
                family = str(asset.family or "")
                if not robot_type:
                    robot_type = asset_id
            except Exception:
                pass

        if not robot_type:
            robot_type = "go2"

        body_mapping = _json_load(params.get("body_mapping", "{}"), {})

        return cls(
            robot_type=robot_type,
            mjcf_path=mjcf_path,
            usd_path=usd_path,
            urdf_path=urdf_path,
            joint_config=_json_load(params.get("joint_config", "{}"), {}),
            action_dim=action_dim,
            joint_names=joint_names,
            asset_id=asset_id,
            family=family,
            active_file=str(params.get("active_override", "auto") or "auto"),
            body_mapping=body_mapping if isinstance(body_mapping, dict) else {},
        )

    def to_dict(self) -> dict:
        return {
            "robot_type": self.robot_type,
            "mjcf_path": self.mjcf_path,
            "usd_path": self.usd_path,
            "urdf_path": self.urdf_path,
            "joint_config": self.joint_config,
            "action_dim": self.action_dim,
            "joint_names": list(self.joint_names),
            "asset_id": self.asset_id,
            "family": self.family,
            "active_file": self.active_file,
            "body_mapping": dict(self.body_mapping),
        }


# ---------------------------------------------------------------------------
# ActorConfig — partner of RobotSpec, owned by ActorSettingNode
# ---------------------------------------------------------------------------

@dataclass
class ActorConfig:
    """Typed output of :class:`ActorSettingNode` — all user-editable
    settings that attach to a robot.

    Clean split from :class:`RobotSpec`:
      - RobotSpec = asset identity + structural metadata (display)
      - ActorConfig = user settings on top of the asset (editable)
    """

    # §1 Init pose (world-frame spawn)
    init_pose: Dict[str, float] = field(
        default_factory=lambda: {"x": 0.0, "y": 0.0, "z": 0.4}
    )
    # §2 Init joint angles (joint_name/regex → radians)
    init_joint_angles: Dict[str, float] = field(default_factory=dict)
    # §3 Contact sensors
    contact_sensor: Dict[str, Any] = field(
        default_factory=lambda: {
            "body_names": [],
            "history_length": 3,
            "track_air_time": True,
        }
    )
    # §4 Actuator gain overrides (scalar or per-joint dict)
    stiffness: Any = 25.0
    damping: Any = 0.5
    effort_limit: Any = 30.0
    velocity_limit: Any = 30.0
    # §5 Action space (absorbed from IL Action Config)
    action_joint_names_expr: str = "[]"
    action_scale: float = 0.25
    action_use_default_offset: bool = True

    @classmethod
    def from_node_dict(cls, params: Dict[str, str]) -> "ActorConfig":
        init = _json_load(params.get("init_joint_angles", "{}"), {})
        if not isinstance(init, dict):
            init = {}

        def _f(key: str, default: float) -> float:
            try:
                return float(params.get(key, default))
            except (TypeError, ValueError):
                return default

        def _i(key: str, default: int) -> int:
            try:
                return int(params.get(key, default))
            except (TypeError, ValueError):
                return default

        body_names = _json_load(params.get("contact_body_names", "[]"), [])
        if not isinstance(body_names, list):
            body_names = []

        def _scalar_or_dict(raw: Any, default: float) -> Any:
            if raw is None or raw == "":
                return default
            text = str(raw).strip()
            if text.startswith("{"):
                parsed = _json_load(text, None)
                if isinstance(parsed, dict):
                    return parsed
            try:
                return float(text)
            except (TypeError, ValueError):
                return default

        return cls(
            init_pose={
                "x": _f("init_pos_x", 0.0),
                "y": _f("init_pos_y", 0.0),
                "z": _f("init_pos_z", 0.4),
            },
            init_joint_angles=init,
            contact_sensor={
                "body_names": body_names,
                "history_length": _i("contact_history_length", 3),
                "track_air_time": (
                    str(params.get("contact_track_air_time", "true") or "true").lower()
                    == "true"
                ),
            },
            stiffness=_scalar_or_dict(params.get("stiffness"), 25.0),
            damping=_scalar_or_dict(params.get("damping"), 0.5),
            effort_limit=_scalar_or_dict(params.get("effort_limit"), 30.0),
            velocity_limit=_scalar_or_dict(params.get("velocity_limit"), 30.0),
            action_joint_names_expr=str(
                params.get("action_joint_names_expr", "[]") or "[]"
            ),
            action_scale=_f("action_scale", 0.25),
            action_use_default_offset=(
                str(params.get("action_use_default_offset", "true") or "true").lower()
                == "true"
            ),
        )

    def to_dict(self) -> dict:
        return {
            "init_pose": dict(self.init_pose),
            "init_joint_angles": dict(self.init_joint_angles),
            "contact_sensor": dict(self.contact_sensor),
            "stiffness": self.stiffness,
            "damping": self.damping,
            "effort_limit": self.effort_limit,
            "velocity_limit": self.velocity_limit,
            "action_joint_names_expr": self.action_joint_names_expr,
            "action_scale": self.action_scale,
            "action_use_default_offset": self.action_use_default_offset,
        }


# ---------------------------------------------------------------------------
# PhysicsConfig
# ---------------------------------------------------------------------------

@dataclass
class PhysicsConfig:
    """Typed output of PhysicsConfigNode."""

    sim_dt: float = 0.002
    control_dt: float = 0.02
    episode_max_steps: int = 1000
    action_type: str = "joint_position"

    @classmethod
    def from_node_dict(cls, params: Dict[str, str]) -> "PhysicsConfig":
        return cls(
            sim_dt=_float(params.get("sim_dt"), 0.002),
            control_dt=_float(params.get("control_dt"), 0.02),
            episode_max_steps=_int(params.get("episode_max_steps"), 1000),
            action_type=_action_type(params.get("action_type")),
        )

    def to_dict(self) -> dict:
        return {
            "sim_dt": self.sim_dt,
            "control_dt": self.control_dt,
            "episode_max_steps": self.episode_max_steps,
            "action_type": self.action_type,
        }


# ---------------------------------------------------------------------------
# TaskConfig
# ---------------------------------------------------------------------------

@dataclass
class TaskConfig:
    """Typed output of TaskConfigNode."""

    task_type: str = "velocity_tracking"
    command_mode: str = "fixed"
    target_vx: float = 0.5
    target_vy: float = 0.0
    target_wz: float = 0.0
    reward_terms: Dict[str, float] = field(default_factory=dict)
    termination_conditions: Dict[str, float] = field(default_factory=dict)
    curriculum: bool = False
    success_threshold: float = 0.8
    truncation_max_steps: int = 0
    curriculum_schedule: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_node_dict(cls, params: Dict[str, str]) -> "TaskConfig":
        return cls(
            task_type=str(params.get("task_type", "velocity_tracking")),
            command_mode=str(params.get("command_mode", "fixed")),
            target_vx=_float(params.get("target_vx"), 0.5),
            target_vy=_float(params.get("target_vy"), 0.0),
            target_wz=_float(params.get("target_wz"), 0.0),
            reward_terms=_json_load(params.get("reward_terms", "{}"), {}),
            termination_conditions=_json_load(params.get("termination_conditions", "{}"), {}),
            curriculum=_bool(params.get("curriculum"), False),
            success_threshold=_float(params.get("success_threshold"), 0.8),
            truncation_max_steps=_int(params.get("truncation_max_steps"), 0),
            curriculum_schedule=_json_load(
                params.get("curriculum_schedule", "{}"), {}
            ),
        )

    def to_dict(self) -> dict:
        return {
            "task_type": self.task_type,
            "command_mode": self.command_mode,
            "target_vx": self.target_vx,
            "target_vy": self.target_vy,
            "target_wz": self.target_wz,
            "reward_terms": self.reward_terms,
            "termination_conditions": self.termination_conditions,
            "curriculum": self.curriculum,
            "success_threshold": self.success_threshold,
            "truncation_max_steps": self.truncation_max_steps,
            "curriculum_schedule": self.curriculum_schedule,
        }


# ---------------------------------------------------------------------------
# RewardConfig
# ---------------------------------------------------------------------------

@dataclass
class RewardConfig:
    """Typed output of RewardsNode."""

    reward_terms: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_node_dict(cls, params: Dict[str, str]) -> "RewardConfig":
        return cls(
            reward_terms=_json_load(
                params.get("reward_terms", json.dumps(default_reward_terms())),
                {},
            ),
        )

    def to_dict(self) -> dict:
        return {"reward_terms": self.reward_terms}


# ---------------------------------------------------------------------------
# TerminationConfig
# ---------------------------------------------------------------------------

@dataclass
class TerminationConfig:
    """Typed output of TerminationsNode."""

    termination_conditions: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_node_dict(cls, params: Dict[str, str]) -> "TerminationConfig":
        return cls(
            termination_conditions=_json_load(
                params.get("termination_conditions", json.dumps(default_termination_conditions())),
                {},
            ),
        )

    def to_dict(self) -> dict:
        return {"termination_conditions": self.termination_conditions}


# ---------------------------------------------------------------------------
# GatedRewardConfig  (MultiGated Reward curriculum)
# ---------------------------------------------------------------------------

@dataclass
class GatedStage:
    """
    One stage in a gated reward curriculum.

    The first stage (index 0) is always active from step 0.
    Each subsequent stage has a gate that must be satisfied before it
    becomes active.  The final stage has no gate — it is the permanent
    fallback once all earlier gates have opened.

    Gate open condition (for non-final stages):
        step >= min_step
        AND ep_rolling_mean >= best_ever_ep_reward * reward_threshold_ratio
        OR  step >= max_step   (hard timeout — never wait forever)
    """

    reward_terms: Dict[str, float] = field(default_factory=dict)
    """Reward term weights active during this stage."""

    min_step: int = 0
    """Earliest global env-step at which this stage's gate may open.
    Ignored for stage 0 (always starts active)."""

    max_step: int = 0
    """Hard-timeout step: gate opens unconditionally at this step even if
    reward_threshold_ratio is not yet satisfied.  0 = no hard timeout
    (only meaningful for non-final stages)."""

    reward_threshold_ratio: float = 0.75
    """Gate opens only when rolling episode mean >= best_ever * this ratio.
    Prevents transitioning during a decline phase.
    Range [0, 1]; 0.0 = always open once min_ep_window is reached."""

    min_ep_window: int = 10
    """Minimum number of episodes completed in this stage before the gate
    may open.  Replaces min_step as the primary readiness guard."""

    plateau_window: int = 10
    """Number of recent stage episodes used to compute the reward slope
    for plateau detection.  Must be <= min_ep_window to have any effect."""

    plateau_eps: float = 0.005
    """Normalised slope threshold for plateau detection.
    slope / |mean_reward| < this value  →  reward has converged (plateau).
    Smaller values require a flatter curve before declaring convergence."""

    @classmethod
    def from_dict(cls, d: Any) -> "GatedStage":
        if not isinstance(d, dict):
            return cls()
        return cls(
            reward_terms=_json_load(d.get("reward_terms", "{}"), {}),
            min_step=_int(d.get("min_step"), 0),
            max_step=_int(d.get("max_step"), 0),
            reward_threshold_ratio=_float(d.get("reward_threshold_ratio"), 0.75),
            min_ep_window=_int(d.get("min_ep_window"), 10),
            plateau_window=_int(d.get("plateau_window"), 10),
            plateau_eps=_float(d.get("plateau_eps"), 0.005),
        )

    def to_dict(self) -> dict:
        return {
            "reward_terms":            dict(self.reward_terms),
            "min_step":                self.min_step,
            "max_step":                self.max_step,
            "reward_threshold_ratio":  self.reward_threshold_ratio,
            "min_ep_window":           self.min_ep_window,
            "plateau_window":          self.plateau_window,
            "plateau_eps":             self.plateau_eps,
        }


@dataclass
class GatedRewardConfig:
    """
    Typed output of MultiGatedRewardNode.

    Describes a sequence of reward stages with gated transitions.
    Passed to UnitreeGymEnv which handles step counting, rolling reward
    tracking, and linear weight blending at each transition.

    When gated_reward_config is None on a spec the existing flat
    reward_terms dict is used unchanged — full backward compatibility.
    """

    stages: List[GatedStage] = field(default_factory=list)
    """Ordered list of stages.  Must have >= 2 entries to be meaningful."""

    blend_steps: int = 3000
    """Number of env steps over which weights are linearly interpolated
    when a gate opens.  0 = instant switch (not recommended)."""

    stage_behavior: str = "replace"
    """How stage weights combine.
    'replace'    — stage N weights fully replace stage N-1.
    'accumulate' — stage N weights are merged on top of stage N-1
                   (new keys added; existing keys use stage N value)."""

    ep_reward_window: int = 20
    """Number of recent episodes used to compute the rolling mean reward
    for the stability guard."""

    min_ep_window: int = 10
    """Minimum number of episodes completed in the current stage before the
    gate may open.  Primary readiness guard — replaces the old min_step role."""

    plateau_window: int = 10
    """Number of recent stage-episodes used to compute the reward slope for
    plateau detection.  Should be <= min_ep_window."""

    plateau_eps: float = 0.005
    """Normalised slope threshold.  When |slope| / |mean_reward| < this value
    the reward curve is considered to have converged, triggering advance."""

    @classmethod
    def from_node_dict(cls, params: Dict[str, Any]) -> "GatedRewardConfig":
        """Build from a MultiGatedRewardNode parameter dict."""
        raw_stages = params.get("stages", [])
        if isinstance(raw_stages, str):
            raw_stages = _json_load(raw_stages, [])
        stages = [GatedStage.from_dict(s) for s in raw_stages] if isinstance(raw_stages, list) else []
        raw_sb = str(params.get("stage_behavior", "replace")).strip().lower()
        if raw_sb not in ("replace", "accumulate"):
            raw_sb = "replace"
        return cls(
            stages=stages,
            blend_steps=max(0, _int(params.get("blend_steps"), 3000)),
            stage_behavior=raw_sb,
            ep_reward_window=max(1, _int(params.get("ep_reward_window"), 20)),
            min_ep_window=max(1, _int(params.get("min_ep_window"), 10)),
            plateau_window=max(2, _int(params.get("plateau_window"), 10)),
            plateau_eps=max(0.0, _float(params.get("plateau_eps"), 0.005)),
        )

    @classmethod
    def from_dict(cls, d: Any) -> "GatedRewardConfig":
        """Deserialise from a plain dict (e.g. TrainingJobSpec.from_dict)."""
        if not isinstance(d, dict):
            return cls()
        return cls.from_node_dict(d)

    def to_dict(self) -> dict:
        return {
            "stages":           [s.to_dict() for s in self.stages],
            "blend_steps":      self.blend_steps,
            "stage_behavior":   self.stage_behavior,
            "ep_reward_window": self.ep_reward_window,
            "min_ep_window":    self.min_ep_window,
            "plateau_window":   self.plateau_window,
            "plateau_eps":      self.plateau_eps,
        }

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def is_valid(self) -> bool:
        """True when there are at least 2 stages with non-empty reward terms."""
        return len(self.stages) >= 2 and all(s.reward_terms for s in self.stages)


# ---------------------------------------------------------------------------
# DomainRandConfig
# ---------------------------------------------------------------------------

@dataclass
class DomainRandConfig:
    """Typed output of DomainRandNode."""

    enabled: bool = True
    mass_range: List[float] = field(default_factory=lambda: [0.8, 1.2])
    friction_range: List[float] = field(default_factory=lambda: [0.5, 1.5])
    motor_strength_range: List[float] = field(default_factory=lambda: [0.9, 1.1])
    joint_damping_range: List[float] = field(default_factory=lambda: [0.95, 1.05])
    obs_noise_std: float = 0.01
    push_robot: bool = False
    push_interval_steps: int = 200
    push_force_range: List[float] = field(default_factory=lambda: [50.0, 150.0])
    rand_schedule: str = "none"
    rand_schedule_start_step: int = 0
    """Global step at which DR activates. When > 0, domain
    randomization is silently disabled for steps [0, start_step).
    Motivated by RSI-AMP staged training: DR should only engage
    after the gait is locked (stage 6)."""
    rand_schedule_end_step: int = 500000

    @classmethod
    def from_node_dict(cls, params: Dict[str, str]) -> "DomainRandConfig":
        return cls(
            enabled=_bool(params.get("enabled"), True),
            mass_range=_float_range(params.get("mass_range"), [0.8, 1.2]),
            friction_range=_float_range(params.get("friction_range"), [0.5, 1.5]),
            motor_strength_range=_float_range(
                params.get("motor_strength_range"), [0.9, 1.1]
            ),
            joint_damping_range=_float_range(
                params.get("joint_damping_range"), [0.95, 1.05]
            ),
            obs_noise_std=_float(params.get("obs_noise_std"), 0.01),
            push_robot=_bool(params.get("push_robot"), False),
            push_interval_steps=_int(params.get("push_interval_steps"), 200),
            push_force_range=_float_range(
                params.get("push_force_range"), [50.0, 150.0]
            ),
            rand_schedule=str(params.get("rand_schedule", "none")),
            rand_schedule_start_step=_int(
                params.get("rand_schedule_start_step"), 0
            ),
            rand_schedule_end_step=_int(
                params.get("rand_schedule_end_step"), 500000
            ),
        )

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "mass_range": self.mass_range,
            "friction_range": self.friction_range,
            "motor_strength_range": self.motor_strength_range,
            "joint_damping_range": self.joint_damping_range,
            "obs_noise_std": self.obs_noise_std,
            "push_robot": self.push_robot,
            "push_interval_steps": self.push_interval_steps,
            "push_force_range": self.push_force_range,
            "rand_schedule": self.rand_schedule,
            "rand_schedule_start_step": self.rand_schedule_start_step,
            "rand_schedule_end_step": self.rand_schedule_end_step,
        }


# ---------------------------------------------------------------------------
# ObsActionConfig
# ---------------------------------------------------------------------------

@dataclass
class ObsActionConfig:
    """Typed output of ObsActionConfigNode."""

    obs_components: List[str] = field(
        default_factory=lambda: ["joint_pos", "joint_vel", "imu",
                                 "command", "previous_action"]
    )
    obs_clip_range: float = 100.0
    frame_stack: int = 1
    action_type: str = "joint_position"
    action_scale: float = 1.0
    action_clip: float = 1.0
    # Phase 5-B: named obs/action contract preset ("custom" = free-form)
    contract_preset: str = "custom"

    @classmethod
    def from_node_dict(cls, params: Dict[str, str]) -> "ObsActionConfig":
        preset = str(params.get("contract_preset", "custom")).strip()

        # When a named preset is active, override obs_components and action_type
        # from the contract definition so the rest of the pipeline uses the
        # locked values — not whatever the free-form UI fields contain.
        try:
            from src.system.training.obs_contracts import apply_preset_to_params
            params = apply_preset_to_params(preset, params)
        except ImportError:
            pass

        raw_obs = params.get(
            "obs_components",
            "joint_pos joint_vel imu command previous_action",
        )
        obs_list = [c.strip() for c in raw_obs.split() if c.strip()]
        return cls(
            obs_components=obs_list,
            obs_clip_range=_float(params.get("obs_clip_range"), 100.0),
            frame_stack=_int(params.get("frame_stack"), 1),
            action_type=_action_type(params.get("action_type")),
            action_scale=_float(params.get("action_scale"), 1.0),
            action_clip=_float(params.get("action_clip"), 1.0),
            contract_preset=preset,
        )

    def to_dict(self) -> dict:
        return {
            "obs_components":  self.obs_components,
            "obs_clip_range":  self.obs_clip_range,
            "frame_stack":     self.frame_stack,
            "action_type":     self.action_type,
            "action_scale":    self.action_scale,
            "action_clip":     self.action_clip,
            "contract_preset": self.contract_preset,
        }


# ---------------------------------------------------------------------------
# EnvConfig
# ---------------------------------------------------------------------------

@dataclass
class EnvConfig:
    """Typed output of EnvAssemblerNode (combined env config)."""

    n_envs: int = 8
    vec_type: str = "subproc"
    obs_normalize: bool = True
    reward_normalize: bool = False
    clip_obs: float = 10.0
    clip_reward: float = 10.0
    action_clip_range: float = 1.0
    enable_monitor: bool = True
    time_limit_override: int = 0
    eval_disable_rand: bool = True

    @classmethod
    def from_node_dict(cls, params: Dict[str, str]) -> "EnvConfig":
        return cls(
            n_envs=_int(params.get("n_envs"), 8),
            vec_type=str(params.get("vec_type", "subproc")),
            obs_normalize=_bool(params.get("obs_normalize"), True),
            reward_normalize=_bool(params.get("reward_normalize"), False),
            clip_obs=_float(params.get("clip_obs"), 10.0),
            clip_reward=_float(params.get("clip_reward"), 10.0),
            action_clip_range=_float(params.get("action_clip_range"), 1.0),
            enable_monitor=_bool(params.get("enable_monitor"), True),
            time_limit_override=_int(params.get("time_limit_override"), 0),
            eval_disable_rand=_bool(params.get("eval_disable_rand"), True),
        )

    def to_dict(self) -> dict:
        return {
            "n_envs": self.n_envs,
            "vec_type": self.vec_type,
            "obs_normalize": self.obs_normalize,
            "reward_normalize": self.reward_normalize,
            "clip_obs": self.clip_obs,
            "clip_reward": self.clip_reward,
            "action_clip_range": self.action_clip_range,
            "enable_monitor": self.enable_monitor,
            "time_limit_override": self.time_limit_override,
            "eval_disable_rand": self.eval_disable_rand,
        }


# ---------------------------------------------------------------------------
# InitPoseConfig
# ---------------------------------------------------------------------------

@dataclass
class InitPoseConfig:
    """
    Typed output of InitPoseNode.

    Controls how ``UnitreeGymEnv.reset()`` initialises the robot's starting pose
    each episode.  When not provided the env falls back to GO2_DEFAULT_QPOS
    (standing pose) with 0.05 rad noise — identical to the legacy hardcoded path.
    """

    mode: str = "default"
    """
    Init strategy:
      default          — GO2_DEFAULT_QPOS + noise  (always works, no dependencies)
      reference_frame_0 — first frame of reference motion (requires ref motion node)
      keyframe         — named MJCF keyframe (requires keyframe_name)
      custom           — explicit joint angles from custom_qpos list
    """

    noise_scale: float = 0.05
    """Per-joint Gaussian noise added on top of the base pose (rad). 0 = deterministic."""

    base_height: float = -1.0
    """
    Override the robot's Z position at reset (metres).
    -1.0 = auto: 0.32 for GO2 default/custom; read from keyframe/ref-frame otherwise.
    """

    keyframe_name: str = "home"
    """MJCF keyframe name used when mode == 'keyframe'."""

    custom_qpos: List[float] = field(default_factory=list)
    """12-element joint-angle array (rad) used when mode == 'custom'."""

    rsi_prob: float = 1.0
    """Probability [0, 1] that a reset uses reference-state initialization
    when ``mode == 'reference_frame_0'``.  At each reset a coin flip
    decides: with prob *rsi_prob* the robot is placed at a reference frame,
    otherwise the default standing pose is used.  1.0 = always reference
    (current behaviour), 0.0 = never.  Typical staged-training value: 0.9.
    Only effective when mode is ``reference_frame_0``."""

    rsi_sample_mode: str = "frame_0"
    """How to pick the reference frame for RSI:
      frame_0        — always use the first frame (current behaviour)
      uniform_phase  — sample a uniformly random frame from the reference
                       motion clip each reset (APEX-style kinematic RSI)
    Only effective when mode == 'reference_frame_0' and the coin flip
    selects reference init."""

    @classmethod
    def from_node_dict(cls, params: Dict[str, str]) -> "InitPoseConfig":
        raw_qpos = _json_load(params.get("custom_qpos", "[]"), [])
        return cls(
            mode=str(params.get("mode", "default")).strip(),
            noise_scale=_float(params.get("noise_scale"), 0.05),
            base_height=_float(params.get("base_height"), -1.0),
            keyframe_name=str(params.get("keyframe_name", "home")).strip(),
            custom_qpos=[float(v) for v in raw_qpos] if isinstance(raw_qpos, list) else [],
            rsi_prob=_float(params.get("rsi_prob"), 1.0),
            rsi_sample_mode=str(params.get("rsi_sample_mode", "frame_0")).strip(),
        )

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "noise_scale": self.noise_scale,
            "base_height": self.base_height,
            "keyframe_name": self.keyframe_name,
            "custom_qpos": list(self.custom_qpos),
            "rsi_prob": self.rsi_prob,
            "rsi_sample_mode": self.rsi_sample_mode,
        }


# ---------------------------------------------------------------------------
# AlgorithmConfig
# ---------------------------------------------------------------------------

@dataclass
class AlgorithmConfig:
    """Typed output of AlgorithmConfigNode."""

    algorithm: str = "PPO"
    total_timesteps: int = 1_000_000
    learning_rate: float = 3e-4
    batch_size: int = 256
    gamma: float = 0.99
    seed: int = 42
    device: str = "auto"
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    activation: str = "relu"
    gae_lambda: float = 0.95
    n_steps: int = 2048
    ent_coef: str = "auto"
    buffer_size: int = 1_000_000
    learning_starts: int = 10_000
    checkpoint_interval: int = 50_000
    resume_mode: str = "scratch"
    policy_id_out: str = "trained_policy"
    gradient_steps: int = 1
    """SAC/off-policy: gradient updates per env step collected.
    -1 = auto (equals n_envs, keeps GPU busy on multi-env setups).
    Increase to 4–16 on high-end GPUs to extract more training signal
    from each simulation step."""
    n_epochs: int = 10
    """PPO: number of passes over each rollout buffer before discarding.
    Higher values extract more gradient signal per CPU simulation batch."""
    lr_schedule: str = "cosine"
    """Learning-rate decay schedule applied over the full training run.
    'constant' — fixed LR throughout (original behaviour).
    'linear'   — linear decay from lr → lr*0.1 over total_timesteps.
    'cosine'   — cosine annealing from lr → lr*0.1 (smooth, recommended)."""
    clip_range: float = 0.2
    """PPO clip range at the start of training (standard default: 0.2)."""
    clip_range_final: float = 0.05
    """PPO clip range at the end of training.
    Set to -1 to keep clip_range constant throughout.
    Default 0.05 — narrows the trust region as the policy matures, reducing
    the risk of late-training catastrophic forgetting."""

    use_per: bool = False
    """Prioritized Experience Replay (PER) for SAC/TD3 (off-policy only).
    When True, transitions are sampled proportional to their TD-error — the
    replay buffer naturally up-weights high-information transitions and
    down-weights stale/low-quality ones.  Recommended when the policy is
    unstable or the reward curve oscillates without converging."""

    per_alpha: float = 0.6
    """PER priority exponent α ∈ [0, 1].  0 = uniform sampling (standard).
    0.6 is a typical default — higher values bias more aggressively toward
    high-TD-error transitions."""

    per_beta: float = 0.4
    """PER importance-sampling exponent β ∈ [0, 1].  Corrects the bias
    introduced by prioritised sampling.  A common schedule is to anneal
    β → 1 over training; 0.4 is a stable starting value."""

    collapse_guard: bool = True
    """Enable the collapse guard callback.  When True, SB3CollapseGuard
    monitors rolling reward and rolls back to best_model.zip whenever a
    collapse is detected."""

    collapse_threshold: float = 0.65
    """Fraction of best-ever rolling mean below which a check counts as bad.
    Default 0.65 — rollback triggers when reward < 65 % of the best seen."""

    collapse_patience: int = 5
    """Consecutive bad rollout-end checks required before a rollback fires.
    Default 5 — prevents false positives from short dips."""

    collapse_lr_factor: float = 0.3
    """LR multiplier applied immediately after rollback.  Default 0.3 —
    reduces to 30 % while the buffer flushes bad experiences, then linearly
    recovers over 20 rollout-ends."""

    collapse_max_rollbacks: int = 0
    """Hard upper limit on the number of rollbacks allowed in one training run.
    0 = unlimited (legacy behaviour).  When the limit is reached, training
    stops cleanly so results can be inspected rather than looping forever."""

    # SAC-specific stability params
    tau: float = 0.005
    """Polyak averaging coefficient for soft target-network updates (SAC only).
    Default 0.005.  Smaller values (0.001-0.003) stabilise the target Q-network
    more aggressively and reduce Q-value overestimation in locomotion tasks."""

    target_entropy: str = "auto"
    """Desired entropy for the SAC entropy-tuning objective.
    'auto' = -action_dim (SB3 default).  For 12-DOF locomotion with dense
    shaped rewards, setting a less aggressive value such as '-6' (= -0.5 ×
    action_dim) keeps more exploration alive and prevents premature policy
    collapse."""

    # ── Phase_3 additive extension (AMP_design.yaml §4.amp_backend) ─────
    # Permitted values for ``algorithm``:
    #   "PPO"      — stable-baselines3 / rsl_rl PPO (legacy default)
    #   "SAC"      — stable-baselines3 SAC
    #   "AMP_PPO"  — vendored AMP + PPO runner (phase_3, isaac venv only)
    # The AMP_PPO branch in il_train_launcher only fires when algorithm
    # exactly equals "AMP_PPO" — anything else keeps the legacy path.

    @classmethod
    def from_node_dict(cls, params: Dict[str, str]) -> "AlgorithmConfig":
        raw_lr_sched = str(params.get("lr_schedule", "cosine")).strip().lower()
        if raw_lr_sched not in ("constant", "linear", "cosine"):
            raw_lr_sched = "cosine"
        # hidden_dims: support both split keys (hidden_dim_1/2) and legacy single key
        if "hidden_dim_1" in params or "hidden_dim_2" in params:
            d1 = _int(params.get("hidden_dim_1"), 256)
            d2 = _int(params.get("hidden_dim_2"), 256)
            resolved_hidden_dims = [d1, d2] if d2 > 0 else [d1]
        else:
            resolved_hidden_dims = _int_list(params.get("hidden_dims", "[256, 256]"), [256, 256])
        return cls(
            algorithm=str(params.get("algorithm", "PPO")),
            total_timesteps=_int(params.get("total_timesteps"), 1_000_000),
            learning_rate=_float(params.get("learning_rate"), 3e-4),
            batch_size=_int(params.get("batch_size"), 256),
            gamma=_float(params.get("gamma"), 0.99),
            seed=_int(params.get("seed"), 42),
            device=str(params.get("device", "auto")),
            hidden_dims=resolved_hidden_dims,
            activation=str(params.get("activation", "relu")),
            gae_lambda=_float(params.get("gae_lambda"), 0.95),
            n_steps=_int(params.get("n_steps"), 2048),
            ent_coef=str(params.get("ent_coef", "auto")),
            buffer_size=_int(params.get("buffer_size"), 1_000_000),
            learning_starts=_int(params.get("learning_starts"), 10_000),
            checkpoint_interval=_int(params.get("checkpoint_interval"), 50_000),
            resume_mode=str(params.get("resume_mode", "scratch")),
            policy_id_out=str(params.get("policy_id_out", "trained_policy")),
            gradient_steps=_int(params.get("gradient_steps"), 1),
            n_epochs=_int(params.get("n_epochs"), 10),
            lr_schedule=raw_lr_sched,
            use_per=_bool(params.get("use_per", "false")),
            per_alpha=_float(params.get("per_alpha"), 0.6),
            per_beta=_float(params.get("per_beta"), 0.4),
            clip_range=_float(params.get("clip_range"), 0.2),
            clip_range_final=_float(params.get("clip_range_final"), 0.05),
            collapse_guard=_bool(params.get("collapse_guard", "true")),
            collapse_threshold=_float(params.get("collapse_threshold"), 0.65),
            collapse_patience=_int(params.get("collapse_patience"), 5),
            collapse_lr_factor=_float(params.get("collapse_lr_factor"), 0.3),
            collapse_max_rollbacks=_int(params.get("collapse_max_rollbacks"), 0),
            tau=_float(params.get("tau"), 0.005),
            target_entropy=str(params.get("target_entropy", "auto")).strip() or "auto",
        )

    def to_dict(self) -> dict:
        return {
            "algorithm": self.algorithm,
            "total_timesteps": self.total_timesteps,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "gamma": self.gamma,
            "seed": self.seed,
            "device": self.device,
            "hidden_dims": self.hidden_dims,
            "activation": self.activation,
            "gae_lambda": self.gae_lambda,
            "n_steps": self.n_steps,
            "ent_coef": self.ent_coef,
            "buffer_size": self.buffer_size,
            "learning_starts": self.learning_starts,
            "checkpoint_interval": self.checkpoint_interval,
            "resume_mode": self.resume_mode,
            "policy_id_out": self.policy_id_out,
            "gradient_steps": self.gradient_steps,
            "use_per":   self.use_per,
            "per_alpha": self.per_alpha,
            "per_beta":  self.per_beta,
            "n_epochs": self.n_epochs,
            "lr_schedule": self.lr_schedule,
            "clip_range": self.clip_range,
            "clip_range_final": self.clip_range_final,
            "collapse_guard": self.collapse_guard,
            "collapse_threshold": self.collapse_threshold,
            "collapse_patience": self.collapse_patience,
            "collapse_lr_factor": self.collapse_lr_factor,
            "collapse_max_rollbacks": self.collapse_max_rollbacks,
            "tau": self.tau,
            "target_entropy": self.target_entropy,
        }


# ---------------------------------------------------------------------------
# EvalConfig
# ---------------------------------------------------------------------------

@dataclass
class EvalConfig:
    """Typed output of EvalConfigNode."""

    eval_episodes: int = 20
    deterministic: bool = True
    success_threshold: float = 0.8
    record_video: bool = False
    eval_interval: int = 50_000
    save_best_model: bool = True
    video_dir: str = ""

    @classmethod
    def from_node_dict(cls, params: Dict[str, str]) -> "EvalConfig":
        return cls(
            eval_episodes=_int(params.get("eval_episodes"), 20),
            deterministic=_bool(params.get("deterministic"), True),
            success_threshold=_float(params.get("success_threshold"), 0.8),
            record_video=_bool(params.get("record_video"), False),
            eval_interval=_int(params.get("eval_interval"), 50_000),
            save_best_model=_bool(params.get("save_best_model"), True),
            video_dir=str(params.get("video_dir", "")),
        )

    def to_dict(self) -> dict:
        return {
            "eval_episodes": self.eval_episodes,
            "deterministic": self.deterministic,
            "success_threshold": self.success_threshold,
            "record_video": self.record_video,
            "eval_interval": self.eval_interval,
            "save_best_model": self.save_best_model,
            "video_dir": self.video_dir,
        }


# ---------------------------------------------------------------------------
# ExportConfig
# ---------------------------------------------------------------------------

@dataclass
class ExportConfig:
    """Typed output of the unified Export node (§5 in the 6-section layout).

    The ``bundle_name`` + ``version`` pair is the user's ONLY naming
    knob — no more hidden auto-increment. Overwrites are allowed by
    default (user flow is "train, inspect, train again" with the same
    bundle); the UI surfaces a ⚠ "Will Overwrite" tag on files that
    already exist so the user is never surprised.
    """

    bundle_name: str = ""
    version: str = "v1"                          # NEW — explicit user-set version tag
    export_onnx: bool = True
    export_torchscript: bool = True
    include_norm_stats: bool = True
    overwrite: bool = True                       # default True — see class doc
    # Phase 6: what to produce — runtime bundle, training artifact, or both
    export_target: str = "runtime_bundle"        # "runtime_bundle" | "training_artifact" | "both"

    @classmethod
    def from_node_dict(cls, params: Dict[str, str]) -> "ExportConfig":
        raw_target = str(params.get("export_target", "runtime_bundle")).strip()
        if raw_target not in ("runtime_bundle", "training_artifact", "both"):
            raw_target = "runtime_bundle"
        return cls(
            bundle_name=str(
                params.get(
                    "bundle_name",
                    params.get("policy_id_out", ""),
                )
            ),
            version=str(params.get("version", "v1") or "v1"),
            export_onnx=_bool(params.get("export_onnx"), True),
            export_torchscript=_bool(params.get("export_torchscript"), True),
            include_norm_stats=_bool(params.get("include_norm_stats"), True),
            overwrite=_bool(params.get("overwrite"), True),
            export_target=raw_target,
        )

    def to_dict(self) -> dict:
        return {
            "bundle_name":    self.bundle_name,
            "version":        self.version,
            "export_onnx":    self.export_onnx,
            "export_torchscript": self.export_torchscript,
            "include_norm_stats": self.include_norm_stats,
            "overwrite":      self.overwrite,
            "export_target":  self.export_target,
        }


# ---------------------------------------------------------------------------
# SceneConfig
# ---------------------------------------------------------------------------

@dataclass
class SceneConfig:
    """Typed output of :class:`PlayGroundSettingNode` (§2 Scene).

    Unified scene configuration — identity, physics, arena, ground
    material, terrain, and height scanner. Fields populated from the
    canvas ``play_ground_setting`` node via :meth:`from_node_dict`.
    When ``scene_id`` resolves in
    :mod:`src.system.training.scene_registry`, the registry entry
    provides metadata (file paths, supported backends) and the node
    parameters carry the user-editable knobs.
    """

    # --- Identity (registry-backed) ---
    scene_id: str = "flat_ground"
    scene_type: str = "flat"        # flat | rough | stairs | custom
    scene_file: str = ""            # resolved from registry (mjcf/usd path)
    scene_file_url: str = ""        # nucleus URL fallback (Isaac Lab)

    # --- World physics ---
    gravity_z: float = -9.81

    # --- Arena ---
    arena_extent_x: float = 10.0
    arena_extent_y: float = 10.0

    # --- Ground material ---
    friction_static: float = 1.0
    friction_dynamic: float = 0.8
    restitution: float = 0.0

    # --- Rough terrain (only populated when scene_type == "rough") ---
    roughness_amplitude: float = 0.0
    slope_max: float = 0.0
    curriculum_enabled: bool = False
    difficulty_levels: int = 1

    # --- Height scan (backend-agnostic) ---
    height_scan_enabled: bool = False
    scan_resolution: float = 0.1
    scan_size_x: float = 1.6
    scan_size_y: float = 1.0

    # --- Legacy fallback (kept so the old SB3 runner path still works) ---
    custom_scene_path: str = ""

    @classmethod
    def from_node_dict(cls, params: Dict[str, str]) -> "SceneConfig":
        def _b(k: str) -> bool:
            return str(params.get(k, "false") or "false").strip().lower() == "true"

        scene_id = str(params.get("scene_id", "") or "").strip() or "flat_ground"
        scene_type = str(params.get("scene_type", "flat") or "flat").strip()
        scene_file = ""
        scene_file_url = ""

        try:
            from src.system.training.scene_registry import get_scene, rescan
            rescan()
            s = get_scene(scene_id)
            scene_type = s.scene_type
            scene_file = str(s.file_path) if s.file_path else ""
            scene_file_url = s.file_url or ""
        except Exception:
            # Registry miss is non-fatal at compile time; the backend
            # runner will surface a clearer error if the scene_id does
            # not resolve there either.
            pass

        return cls(
            scene_id=scene_id,
            scene_type=scene_type if scene_type in ("flat", "rough", "stairs", "custom") else "flat",
            scene_file=scene_file,
            scene_file_url=scene_file_url,
            gravity_z=_float(params.get("gravity_z"), -9.81),
            arena_extent_x=_float(params.get("arena_extent_x"), 10.0),
            arena_extent_y=_float(params.get("arena_extent_y"), 10.0),
            friction_static=_float(params.get("friction_static"), 1.0),
            friction_dynamic=_float(params.get("friction_dynamic"), 0.8),
            restitution=_float(params.get("restitution"), 0.0),
            roughness_amplitude=_float(params.get("roughness_amplitude"), 0.0),
            slope_max=_float(params.get("slope_max"), 0.0),
            curriculum_enabled=_b("curriculum_enabled"),
            difficulty_levels=_int(params.get("difficulty_levels"), 1),
            height_scan_enabled=_b("height_scan_enabled"),
            scan_resolution=_float(params.get("scan_resolution"), 0.1),
            scan_size_x=_float(params.get("scan_size_x"), 1.6),
            scan_size_y=_float(params.get("scan_size_y"), 1.0),
            custom_scene_path=str(params.get("custom_scene_path", "") or ""),
        )

    def to_dict(self) -> dict:
        return {
            "scene_id": self.scene_id,
            "scene_type": self.scene_type,
            "scene_file": self.scene_file,
            "scene_file_url": self.scene_file_url,
            "gravity_z": self.gravity_z,
            "arena_extent_x": self.arena_extent_x,
            "arena_extent_y": self.arena_extent_y,
            "friction_static": self.friction_static,
            "friction_dynamic": self.friction_dynamic,
            "restitution": self.restitution,
            "roughness_amplitude": self.roughness_amplitude,
            "slope_max": self.slope_max,
            "curriculum_enabled": self.curriculum_enabled,
            "difficulty_levels": self.difficulty_levels,
            "height_scan_enabled": self.height_scan_enabled,
            "scan_resolution": self.scan_resolution,
            "scan_size_x": self.scan_size_x,
            "scan_size_y": self.scan_size_y,
            "custom_scene_path": self.custom_scene_path,
        }


# ---------------------------------------------------------------------------
# VisCheckConfig
# ---------------------------------------------------------------------------

@dataclass
class VisCheckConfig:
    """Typed output of VisCheckNode (optional — Layer D)."""

    trigger_mode: str = "step_interval"
    """'step_interval': fire every vis_interval_steps; 'episode_split': divide total run into num_vis_checks equal slots."""

    vis_start_step: int = 50_000
    """First milestone step at which to open the MuJoCo viewer (step_interval mode only)."""

    vis_interval_steps: int = 50_000
    """Open viewer again every N steps after vis_start_step (step_interval mode only)."""

    num_vis_checks: int = 5
    """Total number of equally-spaced vis checks across the full run (episode_split mode only)."""

    vis_episodes: int = 3
    """Number of episodes to show per visualization milestone."""

    deterministic: bool = True
    """Use deterministic policy actions during visualization."""

    @classmethod
    def from_node_dict(cls, params: Dict[str, str]) -> "VisCheckConfig":
        raw_mode = str(params.get("trigger_mode", "step_interval")).strip()
        if raw_mode not in ("step_interval", "episode_split"):
            raw_mode = "step_interval"
        return cls(
            trigger_mode=raw_mode,
            vis_start_step=_int(params.get("vis_start_step"), 50_000),
            vis_interval_steps=_int(params.get("vis_interval_steps"), 50_000),
            num_vis_checks=_int(params.get("num_vis_checks"), 5),
            vis_episodes=_int(params.get("vis_episodes"), 3),
            deterministic=_bool(params.get("deterministic"), True),
        )

    def to_dict(self) -> dict:
        return {
            "trigger_mode": self.trigger_mode,
            "vis_start_step": self.vis_start_step,
            "vis_interval_steps": self.vis_interval_steps,
            "num_vis_checks": self.num_vis_checks,
            "vis_episodes": self.vis_episodes,
            "deterministic": self.deterministic,
        }


# ---------------------------------------------------------------------------
# ImitationLearningConfig
# ---------------------------------------------------------------------------

@dataclass
class ImitationLearningConfig:
    """Behavioral cloning and imitation learning configuration.

    Controls the 3-phase imitation learning pipeline that runs before (and
    optionally during) the main RL training loop:

        Phase 1 — Pure BC: supervised action matching on reference trajectories.
        Phase 2 — BC + RL: blended loss with decaying BC coefficient.
        Phase 3 — Pure RL: standard RL fine-tuning (handled by SB3Trainer).

    The demo dataset is collected by rolling the reference motion through the
    env and recording (obs, reference_action) pairs.  This is standard practice
    per DeepMimic (Peng et al., 2018) and AMP (Peng et al., 2021).
    """

    enabled: bool = False
    """Master switch.  When False the trainer skips BC entirely."""

    # ── Phase 1: Pure BC ──────────────────────────────────────────────
    bc_epochs: int = 50
    """Number of full passes over the demo dataset in the pure-BC phase."""

    bc_learning_rate: float = 1e-3
    """Adam LR for the BC supervised loss."""

    bc_batch_size: int = 256
    """Mini-batch size for BC gradient steps."""

    bc_loss_type: str = "mse"
    """Loss function: 'mse' (L2) or 'huber' (smooth-L1)."""

    # ── Phase 2: BC + RL blend ────────────────────────────────────────
    bc_blend_steps: int = 200_000
    """Number of RL timesteps during which the BC auxiliary loss is active.
    The BC coefficient decays linearly from ``bc_blend_coef_start`` to 0
    over this many steps.  Set to 0 to skip the blend phase entirely."""

    bc_blend_coef_start: float = 0.5
    """Starting coefficient λ for the BC auxiliary loss during the blend
    phase.  Total actor loss = RL_loss + λ(t) · BC_loss."""

    # ── Demo collection ───────────────────────────────────────────────
    demo_num_trajectories: int = 50
    """Number of reference trajectories to collect for the demo buffer.
    Each trajectory runs for ``episode_max_steps`` env steps (or until
    termination), recording the observation and the reference joint target
    at every control step."""

    demo_noise_std: float = 0.0
    """Gaussian noise σ added to the reference action during demo collection.
    Small values (0.01–0.05) broaden the demo distribution (DAgger-like)
    without degrading quality.  0 means deterministic reference replay."""

    # ── Reference obs injection ───────────────────────────────────────
    auto_inject_ref_obs: bool = True
    """Automatically append reference observation components
    (reference_joint_positions, reference_joint_velocities) to the
    obs_components list when a ReferenceMotionConfig is active.
    Following DeepMimic / AMP standard practice."""

    @classmethod
    def from_node_dict(cls, params: Dict[str, str]) -> "ImitationLearningConfig":
        return cls(
            enabled=_bool(params.get("imitation_enabled"), False),
            bc_epochs=_int(params.get("bc_epochs"), 50),
            bc_learning_rate=_float(params.get("bc_learning_rate"), 1e-3),
            bc_batch_size=_int(params.get("bc_batch_size"), 256),
            bc_loss_type=str(params.get("bc_loss_type", "mse")).strip().lower(),
            bc_blend_steps=_int(params.get("bc_blend_steps"), 200_000),
            bc_blend_coef_start=_float(params.get("bc_blend_coef_start"), 0.5),
            demo_num_trajectories=_int(params.get("demo_num_trajectories"), 50),
            demo_noise_std=_float(params.get("demo_noise_std"), 0.0),
            auto_inject_ref_obs=_bool(params.get("auto_inject_ref_obs"), True),
        )

    def to_dict(self) -> dict:
        return {
            "enabled":               self.enabled,
            "bc_epochs":             self.bc_epochs,
            "bc_learning_rate":      self.bc_learning_rate,
            "bc_batch_size":         self.bc_batch_size,
            "bc_loss_type":          self.bc_loss_type,
            "bc_blend_steps":        self.bc_blend_steps,
            "bc_blend_coef_start":   self.bc_blend_coef_start,
            "demo_num_trajectories": self.demo_num_trajectories,
            "demo_noise_std":        self.demo_noise_std,
            "auto_inject_ref_obs":   self.auto_inject_ref_obs,
        }


# ---------------------------------------------------------------------------
# ReferenceMotionConfig
# ---------------------------------------------------------------------------

@dataclass
class ReferenceMotionConfig:
    """Typed output of ReferenceMotionNode (optional — Layer A).

    Specifies a reference motion trajectory file and tracking parameters for
    motion imitation learning.  When connected to EnvAssemblerNode the trainer
    adds a ``reference_tracking`` reward term to each env step.
    """

    motion_file: str = ""
    """Path to .npy keyframe array of shape (T, num_joints)."""

    motion_source: str = ""
    """Alternative to motion_file.  Accepted formats:
    - ``"loco:<env_name>:<task>"`` — load from loco-mujoco dataset
      (e.g. ``"loco:UnitreeGo2:walk"``)
    - ``"generate:standing"`` — procedural standing-pose reference
    - ``"generate:walk"`` — procedural sinusoidal walk reference
    When non-empty, overrides *motion_file*."""

    phase_mode: str = "loop"
    """'loop': cycle through frames endlessly; 'once': stop at last frame."""

    tracking_weight: float = 1.0
    """Per-node weight scale applied on top of the reward_term coefficient."""

    tracking_sigma: float = 5.0
    """Gaussian sharpness: reward = exp(-sigma * ||q_cur - q_ref||^2)."""

    motion_fps: float = 0.0
    """Playback rate of the reference file in Hz.  0 means 1 frame per control
    step (legacy behaviour).  When set to the actual capture/export fps the env
    automatically resamples to match control_frequency_hz."""

    random_start_phase: bool = True
    """Randomise the starting frame index on each episode reset.
    Prevents the policy from over-fitting to the trajectory opening.
    Default ON — matches AMP_for_hardware reference settings and the
    Canvas-side ReferenceMotionNode init defaults."""

    # ── Phase_2 additive extension (AMP_design.yaml §3.reference_motion) ──
    # These four fields default to values that reproduce pre-phase_2 behavior
    # (tracking-only .npy through unitree_gym_env._init_reference_motion).
    # When ``consumer_mode`` is switched to ``"amp"`` or ``"both"``, the
    # motion flows through src.system.training.motion.loader instead
    # (and eventually through amp.data_provider in phase_3).

    format_id: str = "unitport_npy"
    """Which motion loader parses ``motion_file``. One of:
    - ``"unitport_npy"`` — legacy .npy tracking format (default)
    - ``"amp_legged_gym"`` — DeepMimic-style .txt/.json used by
      AMP_for_hardware and similar repos. Carries full AMP payload.
    """

    consumer_mode: str = "tracking"
    """Downstream consumption strategy. One of:
    - ``"tracking"``  — legacy path (PPO + reference tracking reward).
                        Goes through unitree_gym_env._init_reference_motion.
                        The default, so all existing canvases keep working.
    - ``"amp"``       — feed the motion through
                        src.system.training.motion.loader → AMP data
                        provider → discriminator (phase_3).
    - ``"both"``      — tracking reward + AMP discriminator simultaneously.
    """

    amp_obs_fields: tuple = ()
    """Tuple of field names selecting which MotionClip slots feed the
    discriminator observation. Empty = use the format's full default set
    (joint_pos + toe_pos + lin_vel + ang_vel + joint_vel + root_z for
    amp_legged_gym). Only consulted when consumer_mode != 'tracking'."""

    # amp_reward_coef removed — authoritative knob is on AMPConfig
    # (amp_ppo_trainer node).  Having it here created a confusing
    # duplicate that could silently shadow the trainer value.

    def needs_amp_pipeline(self) -> bool:
        """True iff the motion should flow through the motion/ + amp/
        packages rather than the legacy _init_reference_motion path."""
        return self.consumer_mode in ("amp", "both")

    @classmethod
    def from_node_dict(cls, params: Dict[str, str]) -> "ReferenceMotionConfig":
        raw_mode = str(params.get("phase_mode", "loop")).strip()
        if raw_mode not in ("loop", "once"):
            raw_mode = "loop"

        # Phase_2 additive fields — parse with safe defaults.
        consumer = str(params.get("consumer_mode", "tracking")).strip().lower()
        if consumer not in ("tracking", "amp", "both"):
            consumer = "tracking"

        format_id = str(params.get("format_id", "unitport_npy")).strip()
        if not format_id:
            format_id = "unitport_npy"

        # amp_obs_fields is stored as CSV in the node dict; the dataclass
        # holds it as an immutable tuple so Canvas round-trip tests see
        # deterministic equality.
        fields_csv = str(params.get("amp_obs_fields", "")).strip()
        if fields_csv:
            amp_obs_fields = tuple(
                f.strip() for f in fields_csv.split(",") if f.strip()
            )
        else:
            amp_obs_fields = ()

        return cls(
            motion_file=str(params.get("motion_file", "")),
            motion_source=str(params.get("motion_source", "")),
            phase_mode=raw_mode,
            tracking_weight=_float(params.get("tracking_weight"), 1.0),
            tracking_sigma=_float(params.get("tracking_sigma"), 5.0),
            motion_fps=_float(params.get("motion_fps"), 0.0),
            random_start_phase=_bool(params.get("random_start_phase"), False),
            format_id=format_id,
            consumer_mode=consumer,
            amp_obs_fields=amp_obs_fields,
        )

    def to_dict(self) -> dict:
        return {
            "motion_file":        self.motion_file,
            "motion_source":      self.motion_source,
            "phase_mode":         self.phase_mode,
            "tracking_weight":    self.tracking_weight,
            "tracking_sigma":     self.tracking_sigma,
            "motion_fps":         self.motion_fps,
            "random_start_phase": self.random_start_phase,
            # Phase_2 additive fields
            "format_id":          self.format_id,
            "consumer_mode":      self.consumer_mode,
            "amp_obs_fields":     list(self.amp_obs_fields),
        }


# ---------------------------------------------------------------------------
# AMPConfig (phase_3 of AMP_design.yaml §4.amp_backend)
# ---------------------------------------------------------------------------

@dataclass
class AMPConfig:
    """Typed output of AMPTrainerNode — AMP-specific hyperparameters.

    Everything here is AMP-only; PPO hyperparameters still live in
    ``AlgorithmConfig``. ``TrainingJobSpec.amp`` is ``None`` on the
    legacy PPO path and becomes an ``AMPConfig`` when the user wires
    an ``AMPTrainerNode`` into the canvas.

    The 7 fields correspond 1:1 with ``AMPTrainerNode.params_amp_specific``
    — this is a bijection_check target (phase_3 review checklist).
    """

    amp_reward_coef: float = 2.0
    """Scale applied to the clipped disc reward. Propagates to
    ``AMPDiscriminator.amp_reward_coef`` and the runner's
    ``amp_reward_coef`` cfg key."""

    task_reward_lerp: float = 0.5
    """Blend coefficient between style reward and task reward:
    ``r_total = (1 - lerp) * style_r + lerp * task_r``. Default 0.5
    matches AMP-PPO_design.yaml §1.reward_mixing. Previously defaulted
    to 0.0 (pure style, zero task) which silently disabled the task
    signal — see B3 in knowledge_base/amp_fix.yaml. Mixing is done in
    ``AMPOnPolicyRunner``; the discriminator stores this value as a
    config pass-through only."""

    disc_hidden_dims: List[int] = field(default_factory=lambda: [1024, 512])
    """Hidden widths of the discriminator MLP trunk. Two hidden layers
    are the AMP_for_hardware default for quadrupeds."""

    disc_grad_penalty: float = 10.0
    """Zero-centered gradient penalty lambda on expert transitions
    (see ``AMPDiscriminator.compute_grad_pen``). Default 10.0 matches
    upstream."""

    amp_replay_buffer_size: int = 1_000_000
    """Capacity of the policy-side AMP replay buffer."""

    num_preload_transitions: int = 200_000
    """If > 0, the runner preloads this many expert transitions up
    front via ``MotionAmpData.preload_transitions`` to avoid per-step
    sampling cost. 0 disables preloading (stream sampling).

    Default 200_000: AMP_for_hardware originally shipped 2_000_000 for
    a multi-clip dog mocap dataset (~50k frames). For UnitPort's
    typical pool of 6 clips × ~40 frames = ~240 unique frames, even
    200k oversamples 800×, which is plenty for the discriminator to
    fit a stable distribution. Setting this to 2_000_000 with the
    same small pool burns 8× more preload time without any benefit
    — the extra samples are duplicates of the first 200k."""

    disc_lr: float = 1e-4
    """Dedicated learning rate for the discriminator parameter group
    in the AMPPPO optimizer. Lower than the policy LR by design —
    AMP training is more sensitive to disc over-training than to
    policy under-training."""

    lerp_schedule: Optional[Dict[str, Any]] = None
    """Optional linear anneal schedule for ``task_reward_lerp``.

    When *None* (default), ``task_reward_lerp`` stays constant for the
    entire run — backward-compatible with existing configs.

    When set, the runner linearly interpolates from *start* to *end*
    over *over_iters* training iterations::

        {"start": 0.75, "end": 0.3, "over_iters": 2000}

    At iteration ``i`` the effective lerp is::

        alpha = clamp(i / over_iters, 0, 1)
        lerp  = start + (end - start) * alpha

    Motivated by RSI-AMP staged training: start task-heavy to build
    locomotion, then shift toward style for gait quality.
    """

    @classmethod
    def from_node_dict(cls, params: Dict[str, str]) -> "AMPConfig":
        hidden = _int_list(
            params.get("disc_hidden_dims", "[1024, 512]"),
            [1024, 512],
        )
        # lerp_schedule — optional dict parsed from JSON string or dict
        raw_sched = params.get("lerp_schedule")
        lerp_sched: Optional[Dict[str, Any]] = None
        if raw_sched:
            if isinstance(raw_sched, str):
                try:
                    import json as _json
                    lerp_sched = _json.loads(raw_sched)
                except Exception:
                    lerp_sched = None
            elif isinstance(raw_sched, dict):
                lerp_sched = dict(raw_sched)
        return cls(
            amp_reward_coef=_float(params.get("amp_reward_coef"), 2.0),
            task_reward_lerp=_float(params.get("task_reward_lerp"), 0.5),
            disc_hidden_dims=hidden,
            disc_grad_penalty=_float(params.get("disc_grad_penalty"), 10.0),
            amp_replay_buffer_size=_int(
                params.get("amp_replay_buffer_size"), 1_000_000
            ),
            num_preload_transitions=_int(
                params.get("num_preload_transitions"), 2_000_000
            ),
            disc_lr=_float(params.get("disc_lr"), 1e-4),
            lerp_schedule=lerp_sched,
        )

    def to_dict(self) -> dict:
        d: dict = {
            "amp_reward_coef": self.amp_reward_coef,
            "task_reward_lerp": self.task_reward_lerp,
            "disc_hidden_dims": list(self.disc_hidden_dims),
            "disc_grad_penalty": self.disc_grad_penalty,
            "amp_replay_buffer_size": self.amp_replay_buffer_size,
            "num_preload_transitions": self.num_preload_transitions,
            "disc_lr": self.disc_lr,
        }
        if self.lerp_schedule is not None:
            d["lerp_schedule"] = dict(self.lerp_schedule)
        return d


# ---------------------------------------------------------------------------
# TrainingStage / StageSchedule  (Trainer-internal staged training)
# ---------------------------------------------------------------------------

@dataclass
class TrainingStage:
    """
    One stage in a multi-stage training pipeline.

    Each stage carries a set of parameter *overrides* that are applied on top
    of the trainer's baseline configuration when the stage becomes active.
    Only keys present in ``overrides`` are modified; everything else inherits
    from the baseline (= Trainer node params + upstream node inputs).

    Advance logic (evaluated every ``eval_interval`` iterations):
      - ``advance_mode == "manual"``: never auto-advance; user clicks UI button.
      - ``advance_mode == "auto"``:   all numeric criteria in ``advance_criteria``
        must be satisfied simultaneously.
      - ``advance_mode == "hybrid"``: numeric criteria must be met, then training
        pauses and waits for user confirmation.
    """

    stage_id: str = ""
    """Unique identifier, e.g. '0_ppo_warmup'."""

    display_name: str = ""
    """Human-readable label shown in the UI."""

    iterations: int = 1000
    """Number of training iterations for this stage."""

    overrides: Dict[str, Any] = field(default_factory=dict)
    """Parameter override dict.  Flat keys map to trainer / env params:
      training_mode, reward_terms, amp_reward_coef, task_reward_lerp,
      lerp_schedule, disc_lr, disc_grad_penalty, disc_label_smoothing,
      learning_rate, entropy_coef, clip_param, domain_rand_enabled,
      domain_rand (sub-dict), commands (sub-dict), termination (sub-dict).
    """

    advance_mode: str = "auto"
    """One of: 'auto', 'manual', 'hybrid'."""

    advance_criteria: Dict[str, Any] = field(default_factory=dict)
    """Conditions for auto/hybrid advance.  Recognised keys:
      - min_iterations          (int)   — earliest iter this stage may advance
      - min_mean_episode_length (float)
      - min_mean_reward         (float)
      - disc_accuracy_range     (list[float, float])
      - visual_check_required   (bool)  — pause for user in hybrid mode
    """

    eval_interval: int = 50
    """How often (in iterations) to evaluate advance_criteria."""

    # -- serialisation --------------------------------------------------------

    @classmethod
    def from_dict(cls, d: Any) -> "TrainingStage":
        if not isinstance(d, dict):
            return cls()
        adv_mode = str(d.get("advance_mode", "auto")).strip().lower()
        if adv_mode not in ("auto", "manual", "hybrid"):
            adv_mode = "auto"
        return cls(
            stage_id=str(d.get("stage_id", "")),
            display_name=str(d.get("display_name", "")),
            iterations=max(1, _int(d.get("iterations"), 1000)),
            overrides=_json_load(d.get("overrides", "{}"), {}),
            advance_mode=adv_mode,
            advance_criteria=_json_load(d.get("advance_criteria", "{}"), {}),
            eval_interval=max(1, _int(d.get("eval_interval"), 50)),
        )

    def to_dict(self) -> dict:
        return {
            "stage_id": self.stage_id,
            "display_name": self.display_name,
            "iterations": self.iterations,
            "overrides": dict(self.overrides),
            "advance_mode": self.advance_mode,
            "advance_criteria": dict(self.advance_criteria),
            "eval_interval": self.eval_interval,
        }


@dataclass
class StageSchedule:
    """
    Complete stage pipeline attached to a single ILPPOTrainerNode.

    When ``stages`` is empty the runner falls back to single-stage behaviour
    (full backward compatibility).  When non-empty, the runner iterates
    through stages in order, applying each stage's overrides and respecting
    advance criteria.
    """

    stages: List[TrainingStage] = field(default_factory=list)
    """Ordered list of training stages."""

    checkpoint_strategy: str = "both"
    """How to save checkpoints at stage transitions.
    'save_on_advance'      — save only when advancing to next stage.
    'save_best_per_stage'  — keep best checkpoint per stage (by reward).
    'both'                 — do both."""

    global_max_iterations: int = 0
    """Total iteration budget across all stages.  0 = auto-computed from
    sum(stage.iterations).  Acts as a safety cap."""

    # -- serialisation --------------------------------------------------------

    @classmethod
    def from_dict(cls, d: Any) -> "StageSchedule":
        if not isinstance(d, dict):
            return cls()
        raw_stages = d.get("stages", [])
        if isinstance(raw_stages, str):
            raw_stages = _json_load(raw_stages, [])
        stages = (
            [TrainingStage.from_dict(s) for s in raw_stages]
            if isinstance(raw_stages, list)
            else []
        )
        strat = str(d.get("checkpoint_strategy", "both")).strip().lower()
        if strat not in ("save_on_advance", "save_best_per_stage", "both"):
            strat = "both"
        auto_total = sum(s.iterations for s in stages)
        explicit = _int(d.get("global_max_iterations"), 0)
        return cls(
            stages=stages,
            checkpoint_strategy=strat,
            global_max_iterations=explicit if explicit > 0 else auto_total,
        )

    def to_dict(self) -> dict:
        return {
            "stages": [s.to_dict() for s in self.stages],
            "checkpoint_strategy": self.checkpoint_strategy,
            "global_max_iterations": self.global_max_iterations,
        }

    # -- convenience ----------------------------------------------------------

    def is_valid(self) -> bool:
        """True when there is at least one stage with iterations > 0."""
        return len(self.stages) >= 1 and all(
            s.iterations > 0 for s in self.stages
        )

    @property
    def total_iterations(self) -> int:
        """Sum of all stage iterations."""
        return sum(s.iterations for s in self.stages)
