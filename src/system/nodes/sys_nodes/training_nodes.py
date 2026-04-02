#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training nodes — Layer A-D.

Layer A — Environment:
    PhysicsConfigNode    -> physics_config dict
    TaskConfigNode       -> task_config dict
    RewardsNode          -> rewards dict
    TerminationsNode     -> terminations dict
    DomainRandNode       -> domain_rand_config dict
    ObsActionConfigNode  -> obs_action_config dict

Layer B — Assembly:
    EnvAssemblerNode     -> env_config dict

Layer C — Learning:
    AlgorithmConfigNode  -> algo_config dict
    TrainNode            -> train_result dict

Layer D — Validation/Export:
    EvalConfigNode       -> eval_config dict
    ExportNode           -> bundle_path str

Deprecated (backward-compat aliases only):
    EnvConfigNode    — replaced by PhysicsConfigNode
    TrainConfigNode  — replaced by AlgorithmConfigNode

These node classes define editable, serializable, connectable configuration
payloads for the Training Workspace canvas. The actual training, evaluation,
export, preview, and review execution paths are handled by the Training
Workspace compiler/runtime pipeline rather than these UI-facing node wrappers.
They must NOT be registered in the Mission Canvas node registry
(see nodes/__init__.py).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base_node import BaseNode
from src.system.training.task_module_registry import (
    default_reward_terms,
    default_termination_conditions,
)


# ---------------------------------------------------------------------------
# TrainingBaseNode — shared interface base for all Phase A-D training nodes
# ---------------------------------------------------------------------------

class TrainingBaseNode(BaseNode):
    """
    Base class for all Training Ground canvas nodes (Layers A-D).

    Provides three reserved hook methods that Phase B-D backend code will
    call without knowing the concrete node type:

    get_config_dict()
        Returns a clean copy of self.parameters.
        Phase B uses this to build TrainingJobSpec; Phase C passes it to
        TrainRunThread.

    validate_params()
        Returns a list of human-readable warning/error strings for invalid
        parameter combinations.  Phase A: always [].  Phase D adds real logic
        (e.g. control_dt must be an integer multiple of sim_dt).

    get_port_types()
        Returns {"in": {slot_name: type_str}, "out": {slot_name: type_str}}.
        By design convention all training port slot names equal their type
        names, so this is derived directly from self.inputs / self.outputs.
        Used by the Phase B canvas serializer to reconstruct typed connections
        without importing the concrete class.
    """

    def get_config_dict(self) -> Dict[str, str]:
        """Return a clean copy of node parameters."""
        return dict(self.parameters or {})

    def validate_params(self) -> List[str]:
        """Return a list of warning/error strings.  Phase A: always empty."""
        return []

    def get_port_types(self) -> Dict[str, Dict[str, str]]:
        """Return in/out slot→type mapping (slot name == type name by convention)."""
        return {
            "in":  {k: k for k in (self.inputs  or {})},
            "out": {k: k for k in (self.outputs or {})},
        }


# ---------------------------------------------------------------------------
# EnvConfigNode  (DEPRECATED)
# ---------------------------------------------------------------------------

class EnvConfigNode(BaseNode):
    # DEPRECATED — replaced by PhysicsConfigNode
    # Retained as aliases for backward import compatibility only.
    # Do not add to any node registry or canvas palette.
    """
    Environment configuration node.

    Emits an ``env_config`` dict describing the intended training environment.
    All values are user-editable parameters; no Gymnasium / MuJoCo calls are
    made in S1.
    """

    def __init__(self, node_id: str):
        super().__init__(node_id, "env_config")
        self.inputs = {}
        self.outputs = {"env_config": None}
        self.parameters = {
            "robot_type": "go2",
            "scene_xml": "",
            "obs_components": "joint_pos joint_vel imu",
            "reward_fn": "default",
            "max_steps": "1000",
        }

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "env_config": {
                "status": "configured",
                "robot_type": self.get_parameter("robot_type", "go2"),
                "scene_xml": self.get_parameter("scene_xml", ""),
                "obs_components": self.get_parameter("obs_components", ""),
                "reward_fn": self.get_parameter("reward_fn", "default"),
                "max_steps": self.get_parameter("max_steps", "1000"),
            }
        }

    def get_display_name(self) -> str:
        return "Env Config"

    def get_description(self) -> str:
        return "Training environment configuration"


# ---------------------------------------------------------------------------
# TrainConfigNode
# ---------------------------------------------------------------------------

class TrainConfigNode(BaseNode):
    # DEPRECATED — replaced by AlgorithmConfigNode
    # Retained as aliases for backward import compatibility only.
    # Do not add to any node registry or canvas palette.
    """
    Training configuration node.

    Accepts ``env_config`` and an optional ``checkpoint`` input.
    Emits a ``train_config`` dict.  No trainer validation in S1.
    """

    def __init__(self, node_id: str):
        super().__init__(node_id, "train_config")
        self.inputs = {"env_config": None, "checkpoint": None}
        self.outputs = {"train_config": None}
        self.parameters = {
            "algorithm": "SAC",
            "total_timesteps": "1000000",
            "n_steps": "2048",
            "batch_size": "256",
            "learning_rate": "3e-4",
            "policy_id_out": "trained_policy",
            "save_interval": "10000",
        }

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        env_cfg = inputs.get("env_config") or {}
        checkpoint = inputs.get("checkpoint") or {}
        return {
            "train_config": {
                "status": "configured",
                "algorithm": self.get_parameter("algorithm", "SAC"),
                "total_timesteps": self.get_parameter("total_timesteps", "1000000"),
                "n_steps": self.get_parameter("n_steps", "2048"),
                "batch_size": self.get_parameter("batch_size", "256"),
                "learning_rate": self.get_parameter("learning_rate", "3e-4"),
                "policy_id_out": self.get_parameter("policy_id_out", "trained_policy"),
                "save_interval": self.get_parameter("save_interval", "10000"),
                "env_config": env_cfg,
                "pretrained_policy_id": checkpoint.get("policy_id", ""),
            }
        }

    def get_display_name(self) -> str:
        return "Train Config"

    def get_description(self) -> str:
        return "Training hyperparameter configuration"


# ---------------------------------------------------------------------------
# TrainNode
# ---------------------------------------------------------------------------

class TrainNode(TrainingBaseNode):
    """
    Training execution node (Layer C).

    Accepts env_config (from EnvAssemblerNode), algo_config (from
    AlgorithmConfigNode), and optional eval_config / train_config (backward
    compat).  Returns the compiled training request payload expected by the Training Workspace runtime.
    """

    # train_config is a deprecated backward-compat alias; hidden from canvas UI
    _HIDDEN_PORTS: set = {"train_config"}
    # eval_config is optional input
    _OPTIONAL_INPUTS: set = {"eval_config"}

    def __init__(self, node_id: str):
        super().__init__(node_id, "train")
        self.inputs = {
            "env_config": None,
            "algo_config": None,
            "eval_config": None,
            "train_config": None,  # DEPRECATED backward-compat port — silently ignored when algo_config present
        }
        self.outputs = {
            "train_result": None,
            "vis_check": None,  # triggers VisCheckNode at milestones
        }
        self.parameters = {}

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        algo_cfg = inputs.get("algo_config") or {}
        policy_id_out = algo_cfg.get("policy_id_out", "trained_policy") if algo_cfg else "trained_policy"
        train_result = {
            "status": "configured",
            "bundle_path": f"custom_mods/training/checkpoints/{policy_id_out}",
            "run_id": "configured_run",
            "metrics": {"reward_mean": 0.0, "best_reward": 0.0},
        }
        return {
            "train_result": train_result,
            "result": {"status": "configured", "bundle_path": train_result["bundle_path"]},  # DEPRECATED alias
        }

    def get_display_name(self) -> str:
        return "Train"

    def get_description(self) -> str:
        return "Training execution node"


# ---------------------------------------------------------------------------
# TaskConfigNode
# ---------------------------------------------------------------------------

class TaskConfigNode(TrainingBaseNode):
    """
    Task configuration node.

    Defines task-level semantics: task type, command mode, curriculum, and
    success/truncation policy. Reward and termination modules are injected
    through dedicated input pipes.
    Emits a ``task_config`` dict.  No RL logic in Phase A.
    """

    def __init__(self, node_id: str):
        super().__init__(node_id, "task_config")
        self.inputs = {
            "rewards": None,
            "terminations": None,
        }
        self.outputs = {"task_config": None}
        self.parameters = {
            "task_type": "velocity_tracking",
            "command_mode": "fixed",
            "target_vx": "0.5",
            "target_vy": "0.0",
            "target_wz": "0.0",
            "curriculum": "false",
            "success_threshold": "0.8",
            "truncation_max_steps": "0",
            "curriculum_schedule": "{}",
        }

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        rewards = inputs.get("rewards") or {}
        terminations = inputs.get("terminations") or {}
        return {
            "task_config": {
                "status": "configured",
                "task_type": self.get_parameter("task_type", "velocity_tracking"),
                "command_mode": self.get_parameter("command_mode", "fixed"),
                "target_vx": self.get_parameter("target_vx", "0.5"),
                "target_vy": self.get_parameter("target_vy", "0.0"),
                "target_wz": self.get_parameter("target_wz", "0.0"),
                "reward_terms": rewards.get("reward_terms", default_reward_terms()),
                "termination_conditions": terminations.get("termination_conditions", default_termination_conditions()),
                "curriculum": self.get_parameter("curriculum", "false"),
                "success_threshold": self.get_parameter("success_threshold", "0.8"),
                "truncation_max_steps": self.get_parameter("truncation_max_steps", "0"),
                "curriculum_schedule": self.get_parameter("curriculum_schedule", "{}"),
            }
        }

    def get_display_name(self) -> str:
        return "Task Config"

    def get_description(self) -> str:
        return "RL task definition"


class RewardsNode(TrainingBaseNode):
    """Reward module node injected into TaskConfigNode."""

    def __init__(self, node_id: str):
        super().__init__(node_id, "rewards")
        self.inputs = {}
        self.outputs = {"rewards": None}
        self.parameters = {
            "reward_terms": str(default_reward_terms()).replace("'", '"'),
        }

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "rewards": {
                "status": "configured",
                "reward_terms": self.get_parameter(
                    "reward_terms",
                    str(default_reward_terms()).replace("'", '"'),
                ),
            }
        }

    def get_display_name(self) -> str:
        return "Rewards"

    def get_description(self) -> str:
        return "Reward term module for Task Config"


class TerminationsNode(TrainingBaseNode):
    """Termination module node injected into TaskConfigNode."""

    def __init__(self, node_id: str):
        super().__init__(node_id, "terminations")
        self.inputs = {}
        self.outputs = {"terminations": None}
        self.parameters = {
            "termination_conditions": str(default_termination_conditions()).replace("'", '"'),
        }

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "terminations": {
                "status": "configured",
                "termination_conditions": self.get_parameter(
                    "termination_conditions",
                    str(default_termination_conditions()).replace("'", '"'),
                ),
            }
        }

    def get_display_name(self) -> str:
        return "Terminations"

    def get_description(self) -> str:
        return "Termination condition module for Task Config"


# ---------------------------------------------------------------------------
# ReferenceMotionNode
# ---------------------------------------------------------------------------

class ReferenceMotionNode(TrainingBaseNode):
    """
    Reference motion trajectory + imitation learning node (Layer A, optional).

    Loads a .npy keyframe file and configures both:
    1. The motion imitation reward (Gaussian tracking)
    2. Behavioral cloning pre-training (DeepMimic / AMP / DAPG standard)

    Connect output ``reference_motion_config`` to
    ``EnvAssemblerNode.reference_motion_config``.

    When connected, the compiler automatically:
    - Injects ``reference_joint_positions``, ``reference_joint_velocities``,
      and ``phase_sin_cos`` into the observation space
    - Enables BC pre-training before RL (configurable via the Imitation
      Learning section below)
    """

    def __init__(self, node_id: str):
        super().__init__(node_id, "reference_motion")
        self.inputs = {}
        self.outputs = {"reference_motion_config": None}
        self.parameters = {
            # ── Motion source ─────────────────────────────────────────
            "motion_source":      "",        # "loco:UnitreeGo2:walk" | "generate:standing" | "generate:walk"
            "motion_file":        "",
            "phase_mode":         "loop",   # "loop" | "once"
            "tracking_weight":    "1.0",
            "tracking_sigma":     "5.0",
            "motion_fps":         "50.0",   # default 50 Hz (matches typical control_dt=0.02)
            "random_start_phase": "false",  # randomise start frame each episode
            # ── Imitation learning (BC) ───────────────────────────────
            "imitation_enabled":      "true",    # master switch for BC pipeline
            "bc_epochs":              "50",      # Phase 1: pure BC supervised epochs
            "bc_learning_rate":       "1e-3",    # Adam LR for BC
            "bc_batch_size":          "256",     # mini-batch size
            "bc_loss_type":           "mse",     # "mse" | "huber"
            "bc_blend_steps":         "200000",  # Phase 2: BC+RL blend duration (RL steps)
            "bc_blend_coef_start":    "0.5",     # λ₀ for auxiliary BC loss during RL
            "demo_num_trajectories":  "50",      # trajectories to collect for demo buffer
            "demo_noise_std":         "0.0",     # DAgger-like noise (0 = deterministic)
            "auto_inject_ref_obs":    "true",    # auto-append ref obs to observation space
        }

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        motion_source = self.get_parameter("motion_source", "")
        motion_file = self.get_parameter("motion_file", "")

        # Resolve motion_source to a .npy file if set
        resolved_file = motion_file
        if motion_source:
            resolved_file = self._resolve_motion_source(motion_source) or motion_file

        return {
            "reference_motion_config": {
                "status":             "configured",
                "motion_source":      motion_source,
                "motion_file":        resolved_file,
                "phase_mode":         self.get_parameter("phase_mode", "loop"),
                "tracking_weight":    self.get_parameter("tracking_weight", "1.0"),
                "tracking_sigma":     self.get_parameter("tracking_sigma", "5.0"),
                "motion_fps":         self.get_parameter("motion_fps", "50.0"),
                "random_start_phase": self.get_parameter("random_start_phase", "false"),
                # Imitation learning params (read by compiler)
                "imitation_enabled":     self.get_parameter("imitation_enabled", "true"),
                "bc_epochs":             self.get_parameter("bc_epochs", "50"),
                "bc_learning_rate":      self.get_parameter("bc_learning_rate", "1e-3"),
                "bc_batch_size":         self.get_parameter("bc_batch_size", "256"),
                "bc_loss_type":          self.get_parameter("bc_loss_type", "mse"),
                "bc_blend_steps":        self.get_parameter("bc_blend_steps", "200000"),
                "bc_blend_coef_start":   self.get_parameter("bc_blend_coef_start", "0.5"),
                "demo_num_trajectories": self.get_parameter("demo_num_trajectories", "50"),
                "demo_noise_std":        self.get_parameter("demo_noise_std", "0.0"),
                "auto_inject_ref_obs":   self.get_parameter("auto_inject_ref_obs", "true"),
            }
        }

    @staticmethod
    def _resolve_motion_source(source: str) -> Optional[str]:
        """Resolve a motion_source string to an absolute .npy file path."""
        source = source.strip()
        if not source:
            return None
        try:
            from src.system.training.loco_mujoco_bridge import (
                is_available, load_trajectory_as_npy,
                generate_standing_reference, generate_simple_walk_reference,
            )

            if source.startswith("loco:"):
                # Format: "loco:<env_name>:<task>"
                parts = source.split(":", 2)
                if len(parts) < 3:
                    return None
                env_name, task = parts[1], parts[2]
                if not is_available():
                    return None
                result = load_trajectory_as_npy(env_name, task=task)
                return str(result) if result else None

            if source.startswith("generate:"):
                gen_type = source.split(":", 1)[1].strip().lower()
                if gen_type == "standing":
                    return str(generate_standing_reference("go2", fps=50.0))
                if gen_type == "walk":
                    return str(generate_simple_walk_reference("go2", fps=50.0))
                return None

        except Exception:
            return None
        return None

    def get_display_name(self) -> str:
        return "Reference Motion"

    def get_description(self) -> str:
        return "Reference trajectory for motion imitation reward"


# ---------------------------------------------------------------------------
# InitPoseNode
# ---------------------------------------------------------------------------

class InitPoseNode(TrainingBaseNode):
    """
    Init Pose configuration node (Layer A, optional).

    Controls how the robot is placed at the start of each training episode.
    Connect output ``init_pose_config`` to ``EnvAssemblerNode.init_pose_config``.

    Without this node the env defaults to GO2_DEFAULT_QPOS + 0.05 rad noise —
    which is fine for basic locomotion but suboptimal when a reference motion is
    loaded (the first frame of the reference is a better starting point).

    Modes
    -----
    default
        GO2 nominal standing pose ± noise.  No other nodes required.
    reference_frame_0
        Use the first keyframe of the Reference Motion node as the starting
        joint configuration.  Falls back to ``default`` when no motion is loaded.
    keyframe
        Apply a named keyframe from the MJCF file (e.g. ``"home"``).
        Falls back to ``default`` when the keyframe is not found.
    custom
        Explicit 12-element joint-angle array supplied by the user.

    Parameters
    ----------
    mode : str
        Init strategy (see above).
    noise_scale : float
        Gaussian noise added per joint at reset (rad). 0 = fully deterministic.
    base_height : float
        Override the robot's Z position at reset (m).  -1 = auto (0.32 for
        GO2 default / custom; inferred from keyframe / ref-frame otherwise).
    keyframe_name : str
        MJCF keyframe identifier, used when mode == ``"keyframe"``.
    custom_qpos : str
        JSON array of 12 floats used when mode == ``"custom"``.
    """

    def __init__(self, node_id: str):
        super().__init__(node_id, "init_pose")
        self.inputs = {}
        self.outputs = {"init_pose_config": None}
        self.parameters = {
            "mode":           "default",   # default | reference_frame_0 | keyframe | custom
            "noise_scale":    "0.05",
            "base_height":    "-1.0",      # -1 = auto
            "keyframe_name":  "home",
            "custom_qpos":    "[]",        # JSON array of 12 floats
        }

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "init_pose_config": {
                "status":         "configured",
                "mode":           self.get_parameter("mode", "default"),
                "noise_scale":    self.get_parameter("noise_scale", "0.05"),
                "base_height":    self.get_parameter("base_height", "-1.0"),
                "keyframe_name":  self.get_parameter("keyframe_name", "home"),
                "custom_qpos":    self.get_parameter("custom_qpos", "[]"),
            }
        }

    def get_display_name(self) -> str:
        return "Init Pose"

    def get_description(self) -> str:
        return "Episode starting pose configuration"


# ---------------------------------------------------------------------------
# SceneConfigNode
# ---------------------------------------------------------------------------

class SceneConfigNode(TrainingBaseNode):
    """
    Scene / terrain configuration node (Layer A, optional).

    Selects which MuJoCo scene XML to use for training and visualization.
    The scene file wraps the robot MJCF via ``<include>`` and adds the ground
    plane, lighting, skybox, and optional terrain geometry.

    Three built-in scene types are offered:
        flat     — scene.xml  (flat ground + lights + skybox)
        terrain  — scene_terrain.xml  (flat + heightfield obstacles)
        custom   — user-supplied path to any .xml scene file

    Output ``scene_config`` connects to ``EnvAssemblerNode.scene_config``.
    When this node is absent, ``UnitreeGymEnv`` falls back to auto-detecting
    ``scene.xml`` in the robot's MJCF directory (same as "flat" mode).
    """

    def __init__(self, node_id: str):
        super().__init__(node_id, "scene_config")
        self.inputs = {}
        self.outputs = {"scene_config": None}
        self.parameters = {
            "scene_type":         "flat",  # "flat" | "terrain" | "custom"
            "custom_scene_path":  "",      # path only used when scene_type="custom"
            "gravity_z":          "-9.81",
        }

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "scene_config": {
                "status":             "configured",
                "scene_type":         self.get_parameter("scene_type", "flat"),
                "custom_scene_path":  self.get_parameter("custom_scene_path", ""),
                "gravity_z":          self.get_parameter("gravity_z", "-9.81"),
            }
        }

    def get_display_name(self) -> str:
        return "Scene Config"

    def get_description(self) -> str:
        return "MuJoCo scene / terrain selection for training and visualization"


# ---------------------------------------------------------------------------
# EvalConfigNode
# ---------------------------------------------------------------------------

class EvalConfigNode(TrainingBaseNode):
    """
    Evaluation configuration node (optional).

    Defines post-training evaluation parameters.
    Emits an ``eval_config`` dict.  No evaluation logic in Phase A.
    """

    def __init__(self, node_id: str):
        super().__init__(node_id, "eval_config")
        self.inputs = {}
        self.outputs = {"eval_config": None}
        self.parameters = {
            "eval_episodes": "20",
            "deterministic": "true",
            "success_threshold": "0.8",
            "record_video": "false",
            "eval_interval": "50000",
            "save_best_model": "true",
            "video_dir": "",
        }

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "eval_config": {
                "status": "configured",
                "eval_episodes": self.get_parameter("eval_episodes", "20"),
                "deterministic": self.get_parameter("deterministic", "true"),
                "success_threshold": self.get_parameter("success_threshold", "0.8"),
                "record_video": self.get_parameter("record_video", "false"),
                "eval_interval": self.get_parameter("eval_interval", "50000"),
                "save_best_model": self.get_parameter("save_best_model", "true"),
                "video_dir": self.get_parameter("video_dir", ""),
            }
        }

    def get_display_name(self) -> str:
        return "Eval Config"

    def get_description(self) -> str:
        return "Post-training evaluation configuration"


# ===========================================================================
# Layer A — Environment Layer
# ===========================================================================

# ---------------------------------------------------------------------------
# PhysicsConfigNode
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# RobotMJCFNode
# ---------------------------------------------------------------------------

class RobotMJCFNode(TrainingBaseNode):
    """
    Robot model / MJCF configuration node (Layer A).

    Defines *which robot* to train on and *where* its MJCF scene file lives.
    Emits a ``robot_spec`` dict containing the resolved MJCF path, mock joint
    names, action_dim, and the set of obs components the robot supports.

    The Robot Type dropdown is populated dynamically from the canonical
    ``mujoco_asset_registry`` — every robot with a registered MuJoCo asset
    rule appears as a selectable option.

    Separating robot identity from simulation timing (PhysicsConfigNode) lets
    Training Ground operate as a standalone function centre — independent of
    any Mission Canvas CheckpointNode context.  No file I/O in Phase A.
    """

    # joint_config is an optional input port (delegated to a future JointConfigNode)
    _OPTIONAL_INPUTS: set = {"joint_config"}

    # Default action_dim per model_id.
    # Quadrupeds default 12 (3 joints × 4 legs); bipeds/humanoids vary.
    _ACTION_DIM = {
        "go2": 12, "go2w": 12, "a1": 12,
        "b2": 12, "b2w": 12,
        "g1": 37, "h1": 19, "h1_2": 19,
        "spot": 12,
    }
    # Default available obs components per model_id.
    _DEFAULT_OBS = "joint_pos joint_vel imu base_lin_vel base_ang_vel"
    _OBS_AVAILABLE = {
        "g1":   "joint_pos joint_vel imu base_lin_vel base_ang_vel gravity_vec",
        "h1":   "joint_pos joint_vel imu base_lin_vel base_ang_vel gravity_vec",
        "h1_2": "joint_pos joint_vel imu base_lin_vel base_ang_vel gravity_vec",
    }

    def __init__(self, node_id: str):
        super().__init__(node_id, "robot_mjcf")
        self.inputs = {"joint_config": None}   # optional — from a future JointConfigNode
        self.outputs = {"robot_spec": None}
        self.parameters = {
            "robot_type": "go2",   # dynamically populated by frontend from registry
            "mjcf_path":  "",      # custom MJCF override; empty = auto-resolve via registry
        }

    @staticmethod
    def _resolve_brand(robot_type: str):
        """Look up (brand_id, model_id) from the canonical mujoco_asset_registry."""
        try:
            from src.system.models.mujoco_asset_registry import registered_mujoco_asset_rules
            for rule in registered_mujoco_asset_rules():
                if rule.model_id == robot_type:
                    return (rule.brand_id, rule.model_id)
        except Exception:
            pass
        return None

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        robot_type = self.get_parameter("robot_type", "go2")
        mjcf_path = self.get_parameter("mjcf_path", "")

        # Auto-resolve via mujoco_asset_registry when no override is provided
        resolved_mjcf_path = mjcf_path
        if not mjcf_path:
            brand_info = self._resolve_brand(robot_type)
            if brand_info:
                try:
                    from src.system.models.mujoco_asset_registry import resolve_mujoco_asset
                    loc = resolve_mujoco_asset(brand_info[0], brand_info[1])
                    if loc is not None:
                        resolved_mjcf_path = str(loc.scene_path)
                except Exception:
                    pass  # registry unavailable; UnitreeGymEnv fallback handles this

        # joint_config: prefer connected port, fall back to empty
        joint_config = inputs.get("joint_config") or "{}"

        return {
            "robot_spec": {
                "status": "configured",
                "robot_type": robot_type,
                "mjcf_path": resolved_mjcf_path,
                "joint_config": joint_config,
                "action_dim": self._ACTION_DIM.get(robot_type, 12),
                "obs_components_available": self._OBS_AVAILABLE.get(
                    robot_type, self._DEFAULT_OBS
                ),
            }
        }

    def get_display_name(self) -> str:
        return "Robot / MJCF"

    def get_description(self) -> str:
        return "Robot model and MJCF scene definition"


# ---------------------------------------------------------------------------
# PhysicsConfigNode
# ---------------------------------------------------------------------------

class PhysicsConfigNode(TrainingBaseNode):
    """
    Physics / simulation timing configuration node (Layer A).

    Accepts ``robot_spec`` from RobotMJCFNode and adds simulation timing
    parameters.  Emits a ``physics_config`` dict consumed by EnvAssemblerNode.
    Robot identity (robot_type, MJCF path, joint_config) lives in
    RobotMJCFNode; this node owns only sim-dt, episode length, and
    actuator control mode.  No MuJoCo calls in Phase A.
    """

    def __init__(self, node_id: str):
        super().__init__(node_id, "physics_config")
        self.inputs = {"robot_spec": None}   # required — from RobotMJCFNode
        self.outputs = {"physics_config": None}
        self.parameters = {
            "sim_dt":           "0.002",
            "control_dt":       "0.02",
            "episode_max_steps": "1000",
            "action_type":      "joint_position",  # enum: joint_position | torque | joint_velocity
        }

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "physics_config": {
                "status": "configured",
                "robot_spec": inputs.get("robot_spec") or {},
                "sim_dt": self.get_parameter("sim_dt", "0.002"),
                "control_dt": self.get_parameter("control_dt", "0.02"),
                "episode_max_steps": self.get_parameter("episode_max_steps", "1000"),
                "action_type": self.get_parameter("action_type", "joint_position"),
            }
        }

    def get_display_name(self) -> str:
        return "Physics Config"

    def get_description(self) -> str:
        return "Simulation timing and actuator configuration"


# ---------------------------------------------------------------------------
# DomainRandNode
# ---------------------------------------------------------------------------

class DomainRandNode(TrainingBaseNode):
    """
    Domain randomisation configuration node (Layer A).

    Emits a ``domain_rand_config`` dict.  No randomisation logic in Phase A.
    """

    def __init__(self, node_id: str):
        super().__init__(node_id, "domain_rand")
        self.inputs = {}
        self.outputs = {"domain_rand_config": None}
        self.parameters = {
            "enabled": "true",
            "mass_range": "[0.8, 1.2]",
            "friction_range": "[0.5, 1.5]",
            "motor_strength_range": "[0.9, 1.1]",
            "joint_damping_range": "[0.95, 1.05]",
            "obs_noise_std": "0.01",
            "push_robot": "false",
            "push_interval_steps": "200",
            "push_force_range": "[50, 150]",
            # DR Schedule (Suggestion 4 — curriculum domain randomization for sim-to-real)
            "rand_schedule":          "none",    # enum: none | linear | exponential
            "rand_schedule_end_step": "500000",  # step at which schedule reaches max intensity
        }

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "domain_rand_config": {
                "status": "configured",
                "enabled": self.get_parameter("enabled", "true"),
                "mass_range": self.get_parameter("mass_range", "[0.8, 1.2]"),
                "friction_range": self.get_parameter("friction_range", "[0.5, 1.5]"),
                "motor_strength_range": self.get_parameter("motor_strength_range", "[0.9, 1.1]"),
                "joint_damping_range": self.get_parameter("joint_damping_range", "[0.95, 1.05]"),
                "obs_noise_std": self.get_parameter("obs_noise_std", "0.01"),
                "push_robot": self.get_parameter("push_robot", "false"),
                "push_interval_steps": self.get_parameter("push_interval_steps", "200"),
                "push_force_range": self.get_parameter("push_force_range", "[50, 150]"),
                "rand_schedule": self.get_parameter("rand_schedule", "none"),
                "rand_schedule_end_step": self.get_parameter("rand_schedule_end_step", "500000"),
            }
        }

    def get_display_name(self) -> str:
        return "Domain Rand"

    def get_description(self) -> str:
        return "Domain randomisation configuration"


# ---------------------------------------------------------------------------
# ObsActionConfigNode
# ---------------------------------------------------------------------------

class ObsActionConfigNode(TrainingBaseNode):
    """
    Observation / action space configuration node (Layer A).

    Emits an ``obs_action_config`` dict.  No space validation in Phase A.
    """

    # Valid contract preset names exposed as a class constant so UI code can
    # enumerate them without importing obs_contracts.
    PRESET_NAMES = ("custom", "unitport_go2_v1", "community_go2_sac_34d")

    def __init__(self, node_id: str):
        super().__init__(node_id, "obs_action_config")
        self.inputs = {}
        self.outputs = {"obs_action_config": None}
        self.parameters = {
            "contract_preset": "custom",
            "obs_components": "joint_pos joint_vel imu command previous_action",
            "obs_clip_range": "100.0",
            "frame_stack": "1",
            "action_type": "joint_position",
            "action_scale": "1.0",
            "action_clip": "1.0",
        }

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        preset = self.get_parameter("contract_preset", "custom")

        # When a named preset is active, override the free-form fields so the
        # output always reflects the locked contract values.
        try:
            from src.system.training.obs_contracts import apply_preset_to_params, get_obs_contract
            locked = apply_preset_to_params(preset, self.parameters)
            obs_comp = locked.get("obs_components",
                                  "joint_pos joint_vel imu command previous_action")
            action_type = locked.get("action_type", "joint_position")
        except ImportError:
            obs_comp = self.get_parameter(
                "obs_components", "joint_pos joint_vel imu command previous_action")
            action_type = self.get_parameter("action_type", "joint_position")

        return {
            "obs_action_config": {
                "status":          "configured",
                "contract_preset": preset,
                "obs_components":  obs_comp,
                "obs_clip_range":  self.get_parameter("obs_clip_range", "100.0"),
                "frame_stack":     self.get_parameter("frame_stack", "1"),
                "action_type":     action_type,
                "action_scale":    self.get_parameter("action_scale", "1.0"),
                "action_clip":     self.get_parameter("action_clip", "1.0"),
            }
        }

    def get_display_name(self) -> str:
        return "Obs & Action"

    def get_description(self) -> str:
        return "Observation and action space configuration"


# ---------------------------------------------------------------------------
# MultiGatedRewardNode
# ---------------------------------------------------------------------------

class MultiGatedRewardNode(TrainingBaseNode):
    """
    Multi-stage gated reward curriculum node (Layer A, optional).

    Acts as a router/combiner: each stage's reward weights come from a
    separate ``RewardsNode`` connected to the corresponding stage input port.
    This node owns only the gate logic (timing, stability guard, blend).

    Wiring:
        RewardsNode  →  stage_0  ─┐
        RewardsNode  →  stage_1  ─┤  MultiGatedRewardNode  →  rewards  →  TaskConfig
        ...                       ┘

    The output ``rewards`` carries the Stage 0 terms as the static baseline;
    the full gated schedule is extracted by the compiler from the connected
    stage nodes and injected into the training spec at compile time.

    Parameters (gate logic only — stage reward terms come from connected nodes)
    ----------
    min_step_stage1 : int
        Minimum global env steps before the Stage 1 gate can open.
    max_step_stage0 : int
        Hard timeout — force Stage 1 gate open at this step even when the
        stability guard is not met.  0 = no hard timeout.
    reward_threshold_ratio : float
        Gate condition: rolling episode mean >= best_ever × this ratio.
    blend_steps : int
        Env steps over which weights are linearly interpolated at transition.
    stage_behavior : str
        ``"replace"`` — Stage 1 weights replace Stage 0.
        ``"accumulate"`` — Stage 1 weights stack on top of Stage 0.
    ep_reward_window : int
        Rolling window size (episodes) used for the stability guard.
    """

    def __init__(self, node_id: str):
        super().__init__(node_id, "multigated_reward")
        # Input ports: one per stage.  Each accepts a rewards dict from a RewardsNode.
        self.inputs = {
            "stage_0": None,   # required — Stage 0 reward group (stability phase)
            "stage_1": None,   # required — Stage 1 reward group (full task phase)
        }
        self.outputs = {
            "rewards":     None,   # combined output → TaskConfig.rewards
            "total_steps": None,   # AUTO-computed recommended total env steps → AlgorithmConfigNode
        }
        self.parameters = {
            # Gate timing
            "max_step_stage0":        "0",       # hard timeout; 0 = no hard timeout
            # Stability guard
            "reward_threshold_ratio": "0.75",
            # Dynamic stage progression
            "min_ep_window":          "10",
            "plateau_window":         "10",
            "plateau_eps":            "0.005",
            # Blend & schedule
            "blend_steps":            "3000",
            "stage_behavior":         "replace",  # replace | accumulate
            "ep_reward_window":       "20",
            # Total-steps auto-calculation
            "stage1_ratio":           "1.5",      # stage 1 budget = max_step_stage0 × stage1_ratio
        }

    def compute_recommended_total_steps(self) -> int:
        """Return the AUTO-calculated recommended total env steps.

        Formula: ``max_step_stage0 × (1 + stage1_ratio)``
        Returns 0 when ``max_step_stage0`` is 0 (hard timeout disabled —
        cannot derive a meaningful total without it).
        """
        try:
            max_s0 = int(float(self.get_parameter("max_step_stage0", "0") or 0))
            ratio  = float(self.get_parameter("stage1_ratio", "1.5") or 1.5)
        except (TypeError, ValueError):
            return 0
        if max_s0 <= 0:
            return 0
        return int(max_s0 * (1.0 + max(0.0, ratio)))

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Stage 0 terms serve as the static baseline output for the pipeline.
        # The full gated schedule is built by TrainingSpecCompiler from the
        # connected stage nodes; no encoding needed here at runtime.
        stage0 = dict(inputs.get("stage_0") or {})
        return {
            "rewards":     stage0,
            "total_steps": self.compute_recommended_total_steps(),
        }

    def get_port_types(self) -> Dict[str, Dict[str, str]]:
        """Stage input ports carry rewards data; total_steps output carries int."""
        return {
            "in":  {k: "rewards" for k in self.inputs},
            "out": {
                "rewards":     "rewards",
                "total_steps": "int",
            },
        }

    def get_display_name(self) -> str:
        return "MultiGated (Reward)"

    def get_description(self) -> str:
        return "Routes multiple Rewards nodes into a gated curriculum schedule"


# ===========================================================================
# Layer B — Assembly Layer
# ===========================================================================

# ---------------------------------------------------------------------------
# EnvAssemblerNode
# ---------------------------------------------------------------------------

class EnvAssemblerNode(TrainingBaseNode):
    """
    Environment assembler node (Layer B).

    Merges physics, task, obs/action, and optional domain-rand configs into a
    single ``env_config`` dict consumed by TrainNode.  No real env construction
    in Phase A.
    """

    # domain_rand_config, scene_config, reference_motion_config, init_pose_config are optional
    _OPTIONAL_INPUTS: set = {
        "domain_rand_config", "scene_config", "reference_motion_config", "init_pose_config",
    }

    def __init__(self, node_id: str):
        super().__init__(node_id, "env_assembler")
        self.inputs = {
            "physics_config": None,
            "task_config": None,
            "obs_action_config": None,
            "domain_rand_config": None,       # optional
            "scene_config": None,             # optional
            "reference_motion_config": None,  # optional
            "init_pose_config": None,         # optional
        }
        self.outputs = {"env_config": None}
        self.parameters = {
            # VecEnv settings (Suggestion 2 — general tool needs explicit vec control)
            "n_envs":              "8",        # default 8; 1 = unacceptable for robot RL
            "vec_type":            "subproc",  # enum: dummy | subproc
            # Wrapper stack (Suggestion 1 — expose each wrapper as configurable param)
            "obs_normalize":       "true",
            "reward_normalize":    "false",
            "clip_obs":            "10.0",
            "clip_reward":         "10.0",
            "action_clip_range":   "1.0",      # ActionClip wrapper: clips actions to [-x, x]
            "enable_monitor":      "true",     # Monitor wrapper: records episode stats
            "time_limit_override": "0",        # 0 = inherit from PhysicsConfig
            "eval_disable_rand":   "true",
        }

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        env_config = {
            "status": "configured",
            "physics": inputs.get("physics_config") or {},
            "task": inputs.get("task_config") or {},
            "obs_action": inputs.get("obs_action_config") or {},
            "domain_rand": inputs.get("domain_rand_config") or {},
            "scene": inputs.get("scene_config") or {},
            "reference_motion": inputs.get("reference_motion_config") or {},
            "init_pose": inputs.get("init_pose_config") or {},
            "wrapper": {
                "n_envs":              self.get_parameter("n_envs", "8"),
                "vec_type":            self.get_parameter("vec_type", "subproc"),
                "obs_normalize":       self.get_parameter("obs_normalize", "true"),
                "reward_normalize":    self.get_parameter("reward_normalize", "false"),
                "clip_obs":            self.get_parameter("clip_obs", "10.0"),
                "clip_reward":         self.get_parameter("clip_reward", "10.0"),
                "action_clip_range":   self.get_parameter("action_clip_range", "1.0"),
                "enable_monitor":      self.get_parameter("enable_monitor", "true"),
                "time_limit_override": self.get_parameter("time_limit_override", "0"),
                "eval_disable_rand":   self.get_parameter("eval_disable_rand", "true"),
            },
        }
        return {"env_config": env_config}

    def get_display_name(self) -> str:
        return "Env Assembler"

    def get_description(self) -> str:
        return "Assembles environment config from Layer A nodes"


# ===========================================================================
# Layer C — Learning Layer
# ===========================================================================

# ---------------------------------------------------------------------------
# AlgorithmConfigNode
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# BaseAssetNode  (Layer C.0 — pre-AlgorithmConfig)
# ---------------------------------------------------------------------------

class BaseAssetNode(TrainingBaseNode):
    """
    Start Point selector node (Layer C.0).

    Lets the user choose where training starts from: a fresh run, the latest
    export artifact for the current workspace, or a specific installed
    training asset. The UI populates this node automatically when the user
    clicks an entry in the Training Assets list panel.

    Output ``base_asset`` dict is consumed by AlgorithmConfigNode's optional
    ``base_asset`` input port.  When ``load_mode`` is "scratch" the output is
    ignored by AlgorithmConfigNode.

    Parameters
    ----------
    asset_id : str
        Asset ID from TrainingAssetRegistry (e.g. "sac_unitree_go2_mujoco").
        Empty = no base asset selected.
    checkpoint_file : str
        Relative path within the asset directory to the SB3 .zip file
        (e.g. "best/best_model.zip").  Empty = auto-select primary_checkpoint.
    load_mode : str
        "scratch"            — start from scratch (base_asset ignored)
        "resume_sb3"         — load SB3 weights + optimizer state
        "warm_start_actor"   — copy actor weights only, no optimizer
    """

    _LOAD_MODES = ("scratch", "resume_sb3", "warm_start_actor")

    def __init__(self, node_id: str):
        super().__init__(node_id, "base_asset")
        self.inputs = {}
        self.outputs = {"base_asset": None}
        self.parameters = {
            "start_point":     "__new__",
            "asset_id":        "",
            "checkpoint_file": "",
            "load_mode":       "scratch",
        }

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        start_point     = self.get_parameter("start_point", "__new__")
        asset_id        = self.get_parameter("asset_id", "")
        checkpoint_file = self.get_parameter("checkpoint_file", "")
        load_mode       = self.get_parameter("load_mode", "scratch")

        # Resolve absolute checkpoint path when asset_id is known
        abs_checkpoint_path = ""
        if asset_id and load_mode != "scratch":
            try:
                from src.system.training.training_asset_registry import TrainingAssetRegistry
                entry = TrainingAssetRegistry().get(asset_id)
                ckpt = checkpoint_file or entry.primary_checkpoint
                if ckpt:
                    abs_checkpoint_path = str(entry.asset_path / ckpt)
            except Exception:
                pass

        return {
            "base_asset": {
                "start_point":          start_point,
                "asset_id":             asset_id,
                "checkpoint_file":      checkpoint_file,
                "load_mode":            load_mode,
                "abs_checkpoint_path":  abs_checkpoint_path,
            }
        }

    def get_display_name(self) -> str:
        return "Start Point"

    def get_description(self) -> str:
        return "Choose whether training starts new, from latest export, or a training asset"


class AlgorithmConfigNode(TrainingBaseNode):
    """
    Algorithm / hyperparameter configuration node (Layer C).

    Emits an ``algo_config`` dict.  No SB3 / trainer validation in Phase A.
    """

    _OPTIONAL_INPUTS: set = {"base_asset", "total_steps"}
    _HIDDEN_PORTS: set = {"total_steps"}  # merged into NodeRow for total_timesteps param

    def __init__(self, node_id: str):
        super().__init__(node_id, "algo_config")
        self.inputs = {
            "base_asset":  None,   # optional — from BaseAssetNode; None = scratch
            "total_steps": None,   # optional — from MultiGatedRewardNode; overrides total_timesteps
        }
        self.outputs = {"algo_config": None}
        self.parameters = {
            "algorithm": "PPO",
            "total_timesteps": "1000000",
            "learning_rate": "3e-4",
            "batch_size": "256",
            "gamma": "0.99",
            "seed": "42",
            "device": "auto",
            "hidden_dim_1": "256",
            "hidden_dim_2": "256",
            "activation": "relu",
            "gae_lambda": "0.95",
            "n_steps": "2048",
            "ent_coef": "auto",
            "buffer_size_mode": "auto",   # "auto" | "manual"
            "buffer_size": "1000000",
            "learning_starts": "10000",
            "checkpoint_interval": "50000",
            "resume_mode": "scratch",
            "policy_id_out": "trained_policy",
            # Off-policy gradient updates
            "gradient_steps": "1",
            # PPO-specific
            "n_epochs": "10",
            "lr_schedule": "cosine",   # "cosine" | "linear" | "constant"
            # SAC-specific
            "tau": "0.005",
            "target_entropy": "auto",
            # Prioritized Experience Replay
            "use_per":   "false",
            "per_alpha": "0.6",
            "per_beta":  "0.4",
            # Collapse guard
            "collapse_guard":         "true",
            "collapse_threshold":     "0.65",
            "collapse_patience":      "5",
            "collapse_lr_factor":     "0.3",
            "collapse_max_rollbacks": "0",
        }

    def get_port_types(self) -> Dict[str, Dict[str, str]]:
        """total_steps input carries int; all others use slot-name convention."""
        return {
            "in":  {"base_asset": "base_asset", "total_steps": "int"},
            "out": {"algo_config": "algo_config"},
        }

    def _resolve_buffer_size(self) -> str:
        """Return effective buffer_size string.  AUTO = total_timesteps / 10, rounded to 10 k."""
        mode = self.get_parameter("buffer_size_mode", "auto").strip().lower()
        if mode == "auto":
            try:
                total = int(float(self.get_parameter("total_timesteps", "1000000")))
                raw = max(10_000, total // 10)
                rounded = round(raw / 10_000) * 10_000
                return str(min(10_000_000, max(10_000, rounded)))
            except (ValueError, TypeError):
                pass
        return self.get_parameter("buffer_size", "1000000")

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        base_asset = inputs.get("base_asset") or {}
        # When a BaseAssetNode is connected, its load_mode takes precedence
        # over the local resume_mode parameter.
        resume_mode = (
            base_asset.get("load_mode")
            if base_asset.get("load_mode") and base_asset.get("load_mode") != "scratch"
            else self.get_parameter("resume_mode", "scratch")
        )
        return {
            "algo_config": {
                "status": "configured",
                "algorithm": self.get_parameter("algorithm", "PPO"),
                "total_timesteps": self.get_parameter("total_timesteps", "1000000"),
                "learning_rate": self.get_parameter("learning_rate", "3e-4"),
                "batch_size": self.get_parameter("batch_size", "256"),
                "gamma": self.get_parameter("gamma", "0.99"),
                "seed": self.get_parameter("seed", "42"),
                "device": self.get_parameter("device", "auto"),
                "hidden_dims": str([
                    int(float(self.get_parameter("hidden_dim_1", "256") or "256")),
                    int(float(self.get_parameter("hidden_dim_2", "256") or "256")),
                ]),
                "activation": self.get_parameter("activation", "relu"),
                "gae_lambda": self.get_parameter("gae_lambda", "0.95"),
                "n_steps": self.get_parameter("n_steps", "2048"),
                "ent_coef": self.get_parameter("ent_coef", "auto"),
                "buffer_size": self._resolve_buffer_size(),
                "gradient_steps": self.get_parameter("gradient_steps", "1"),
                "n_epochs": self.get_parameter("n_epochs", "10"),
                "lr_schedule": self.get_parameter("lr_schedule", "cosine"),
                "tau": self.get_parameter("tau", "0.005"),
                "target_entropy": self.get_parameter("target_entropy", "auto"),
                "learning_starts": self.get_parameter("learning_starts", "10000"),
                "checkpoint_interval": self.get_parameter("checkpoint_interval", "50000"),
                "resume_mode": resume_mode,
                "policy_id_out": self.get_parameter("policy_id_out", "trained_policy"),
                "base_asset": base_asset,
                "use_per":   self.get_parameter("use_per",   "false"),
                "per_alpha": self.get_parameter("per_alpha", "0.6"),
                "per_beta":  self.get_parameter("per_beta",  "0.4"),
                "collapse_guard":          self.get_parameter("collapse_guard", "true"),
                "collapse_threshold":      self.get_parameter("collapse_threshold", "0.65"),
                "collapse_patience":       self.get_parameter("collapse_patience", "5"),
                "collapse_lr_factor":      self.get_parameter("collapse_lr_factor", "0.3"),
                "collapse_max_rollbacks":  self.get_parameter("collapse_max_rollbacks", "0"),
            }
        }

    def get_display_name(self) -> str:
        return "Algorithm Config"

    def get_description(self) -> str:
        return "RL algorithm and hyperparameter configuration"


# ===========================================================================
# Layer D — Validation / Export Layer
# ===========================================================================

# ---------------------------------------------------------------------------
# ExportNode
# ---------------------------------------------------------------------------

class ExportNode(TrainingBaseNode):
    """
    Export node (Layer D).

    Accepts ``train_result`` (required) and optional ``eval_config``.
    Returns the effective bundle path emitted by the Training Workspace export pipeline.
    """

    # eval_config is optional — visually marked "(opt)" in TrainingNodeItem
    _OPTIONAL_INPUTS: set = {"eval_config"}

    def __init__(self, node_id: str):
        super().__init__(node_id, "export")
        self.inputs = {
            "train_result": None,
            "eval_config": None,  # optional
        }
        self.outputs = {"bundle_path": None}
        self.parameters = {
            "bundle_name":        "",
            "export_target":      "runtime_bundle",  # runtime_bundle | training_artifact | both
            "export_onnx":        "true",
            "export_torchscript": "true",
            "include_norm_stats": "true",
            "overwrite":          "true",
        }

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        train_result = inputs.get("train_result") or {}
        bundle_name = self.get_parameter("bundle_name", "") or "trained_policy"
        bundle_path = train_result.get(
            "bundle_path", f"mock/custom_mods/training/checkpoints/{bundle_name}"
        )
        return {"bundle_path": bundle_path}

    def get_display_name(self) -> str:
        return "Export"

    def get_description(self) -> str:
        return "Export trained policy to bundle"


# ---------------------------------------------------------------------------
# VisCheckNode
# ---------------------------------------------------------------------------

class VisCheckNode(TrainingBaseNode):
    """
    Visualization check configuration node (Layer D, optional).

    Schedules milestone visualization episodes during training — opens the
    MuJoCo passive viewer at configurable step intervals so the user can watch
    the robot's current policy in action without stopping training.

    Connects to TrainNode's optional ``vis_check`` input port.
    """

    def __init__(self, node_id: str):
        super().__init__(node_id, "vis_check")
        self.inputs = {"vis_check": None}  # receives milestone trigger from TrainNode
        self.outputs = {}
        self.parameters = {
            "trigger_mode":       "step_interval",  # "step_interval" | "episode_split"
            "vis_start_step":    "50000",   # first milestone step (step_interval mode)
            "vis_interval_steps": "50000",  # repeat every N steps (step_interval mode)
            "num_vis_checks":    "5",       # total equally-spaced checks (episode_split mode)
            "vis_episodes":      "3",       # episodes per milestone
            "deterministic":     "true",    # deterministic policy actions
        }

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "vis_check": {
                "trigger_mode":       self.get_parameter("trigger_mode", "step_interval"),
                "vis_start_step":    self.get_parameter("vis_start_step", "50000"),
                "vis_interval_steps": self.get_parameter("vis_interval_steps", "50000"),
                "num_vis_checks":    self.get_parameter("num_vis_checks", "5"),
                "vis_episodes":      self.get_parameter("vis_episodes", "3"),
                "deterministic":     self.get_parameter("deterministic", "true"),
            }
        }

    def get_display_name(self) -> str:
        return "Vis Check"

    def get_description(self) -> str:
        return "Milestone visualization — opens MuJoCo viewer at training checkpoints"
