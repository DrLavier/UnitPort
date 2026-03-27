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

from system.training.task_module_registry import (
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
    """Typed output of RobotMJCFNode."""

    robot_type: str = "go2"
    mjcf_path: str = ""
    joint_config: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_node_dict(cls, params: Dict[str, str]) -> "RobotSpec":
        return cls(
            robot_type=str(params.get("robot_type", "go2")),
            mjcf_path=str(params.get("mjcf_path", "")),
            joint_config=_json_load(params.get("joint_config", "{}"), {}),
        )

    def to_dict(self) -> dict:
        return {
            "robot_type": self.robot_type,
            "mjcf_path": self.mjcf_path,
            "joint_config": self.joint_config,
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
    Range [0, 1]; 0.0 = always open once min_step is reached."""

    @classmethod
    def from_dict(cls, d: Any) -> "GatedStage":
        if not isinstance(d, dict):
            return cls()
        return cls(
            reward_terms=_json_load(d.get("reward_terms", "{}"), {}),
            min_step=_int(d.get("min_step"), 0),
            max_step=_int(d.get("max_step"), 0),
            reward_threshold_ratio=_float(d.get("reward_threshold_ratio"), 0.75),
        )

    def to_dict(self) -> dict:
        return {
            "reward_terms":            dict(self.reward_terms),
            "min_step":                self.min_step,
            "max_step":                self.max_step,
            "reward_threshold_ratio":  self.reward_threshold_ratio,
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
            from system.training.obs_contracts import apply_preset_to_params
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

    @classmethod
    def from_node_dict(cls, params: Dict[str, str]) -> "InitPoseConfig":
        raw_qpos = _json_load(params.get("custom_qpos", "[]"), [])
        return cls(
            mode=str(params.get("mode", "default")).strip(),
            noise_scale=_float(params.get("noise_scale"), 0.05),
            base_height=_float(params.get("base_height"), -1.0),
            keyframe_name=str(params.get("keyframe_name", "home")).strip(),
            custom_qpos=[float(v) for v in raw_qpos] if isinstance(raw_qpos, list) else [],
        )

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "noise_scale": self.noise_scale,
            "base_height": self.base_height,
            "keyframe_name": self.keyframe_name,
            "custom_qpos": list(self.custom_qpos),
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

    @classmethod
    def from_node_dict(cls, params: Dict[str, str]) -> "AlgorithmConfig":
        raw_lr_sched = str(params.get("lr_schedule", "cosine")).strip().lower()
        if raw_lr_sched not in ("constant", "linear", "cosine"):
            raw_lr_sched = "cosine"
        return cls(
            algorithm=str(params.get("algorithm", "PPO")),
            total_timesteps=_int(params.get("total_timesteps"), 1_000_000),
            learning_rate=_float(params.get("learning_rate"), 3e-4),
            batch_size=_int(params.get("batch_size"), 256),
            gamma=_float(params.get("gamma"), 0.99),
            seed=_int(params.get("seed"), 42),
            device=str(params.get("device", "auto")),
            hidden_dims=_int_list(params.get("hidden_dims", "[256, 256]"),
                                  [256, 256]),
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
            clip_range=_float(params.get("clip_range"), 0.2),
            clip_range_final=_float(params.get("clip_range_final"), 0.05),
            collapse_guard=_bool(params.get("collapse_guard", "true")),
            collapse_threshold=_float(params.get("collapse_threshold"), 0.65),
            collapse_patience=_int(params.get("collapse_patience"), 5),
            collapse_lr_factor=_float(params.get("collapse_lr_factor"), 0.3),
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
            "n_epochs": self.n_epochs,
            "lr_schedule": self.lr_schedule,
            "clip_range": self.clip_range,
            "clip_range_final": self.clip_range_final,
            "collapse_guard": self.collapse_guard,
            "collapse_threshold": self.collapse_threshold,
            "collapse_patience": self.collapse_patience,
            "collapse_lr_factor": self.collapse_lr_factor,
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
    """Typed output of ExportNode."""

    bundle_name: str = ""
    export_onnx: bool = True
    export_torchscript: bool = True
    include_norm_stats: bool = True
    overwrite: bool = False
    # Phase 6: what to produce — runtime bundle, training artifact, or both
    export_target: str = "runtime_bundle"  # "runtime_bundle" | "training_artifact" | "both"

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
            export_onnx=_bool(params.get("export_onnx"), True),
            export_torchscript=_bool(params.get("export_torchscript"), True),
            include_norm_stats=_bool(params.get("include_norm_stats"), True),
            overwrite=_bool(params.get("overwrite"), False),
            export_target=raw_target,
        )

    def to_dict(self) -> dict:
        return {
            "bundle_name":    self.bundle_name,
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
    """Typed output of SceneConfigNode (optional — Layer A)."""

    scene_type: str = "flat"
    """'flat': scene.xml (ground + lights + skybox).
    'terrain': scene_terrain.xml (+ heightfield obstacles).
    'custom': use custom_scene_path."""

    custom_scene_path: str = ""
    """Absolute or CWD-relative path to a user-provided scene XML.
    Only used when scene_type='custom'."""

    gravity_z: float = -9.81
    """Effective gravity on world Z axis. Defaults to Earth gravity."""

    @classmethod
    def from_node_dict(cls, params: Dict[str, str]) -> "SceneConfig":
        raw_type = str(params.get("scene_type", "flat")).strip()
        if raw_type not in ("flat", "terrain", "custom"):
            raw_type = "flat"
        return cls(
            scene_type=raw_type,
            custom_scene_path=str(params.get("custom_scene_path", "")),
            gravity_z=_float(params.get("gravity_z"), -9.81),
        )

    def to_dict(self) -> dict:
        return {
            "scene_type": self.scene_type,
            "custom_scene_path": self.custom_scene_path,
            "gravity_z": self.gravity_z,
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

    random_start_phase: bool = False
    """Randomise the starting frame index on each episode reset.
    Prevents the policy from over-fitting to the trajectory opening."""

    @classmethod
    def from_node_dict(cls, params: Dict[str, str]) -> "ReferenceMotionConfig":
        raw_mode = str(params.get("phase_mode", "loop")).strip()
        if raw_mode not in ("loop", "once"):
            raw_mode = "loop"
        return cls(
            motion_file=str(params.get("motion_file", "")),
            phase_mode=raw_mode,
            tracking_weight=_float(params.get("tracking_weight"), 1.0),
            tracking_sigma=_float(params.get("tracking_sigma"), 5.0),
            motion_fps=_float(params.get("motion_fps"), 0.0),
            random_start_phase=_bool(params.get("random_start_phase"), False),
        )

    def to_dict(self) -> dict:
        return {
            "motion_file":        self.motion_file,
            "phase_mode":         self.phase_mode,
            "tracking_weight":    self.tracking_weight,
            "tracking_sigma":     self.tracking_sigma,
            "motion_fps":         self.motion_fps,
            "random_start_phase": self.random_start_phase,
        }
