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
    default_il_termination_conditions,
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
            "train_pipe": None,
            "vis_check": None,  # triggers VisCheckNode at milestones
        }
        self.parameters = {}

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        algo_cfg = inputs.get("algo_config") or {}
        policy_id_out = algo_cfg.get("policy_id_out", "trained_policy") if algo_cfg else "trained_policy"
        train_pipe = {
            "status": "configured",
            "bundle_path": f"training/exported/{policy_id_out}",
            "run_id": "configured_run",
            "metrics": {"reward_mean": 0.0, "best_reward": 0.0},
        }
        return {
            "train_pipe": train_pipe,
            "result": {"status": "configured", "bundle_path": train_pipe["bundle_path"]},  # DEPRECATED alias
        }

    def get_display_name(self) -> str:
        return "SB3 Train"

    def get_description(self) -> str:
        return "SB3 training execution node"


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
            "total_reward": None,
            "termination_config": None,
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
    """Unified reward configuration node — works for both SB3 and Isaac Lab.

    backend="sb3":      uses REWARD_REGISTRY, no input ports, output="total_reward"
    backend="isaac_lab": uses IL_REWARD_REGISTRY, shows IL input ports, extra params (std, threshold)
    """

    # IL-mode optional inputs — unified pipes from the 6-section nodes.
    # RewardsNode reads actor + command streams from the new Actor block
    # and Training Commands node respectively. Contact sensor and action
    # space data are already bundled inside actor_pipe, so we do NOT
    # declare separate ports for them any more.
    _IL_PORTS = {"actor_pipe", "command_pipe"}

    def __init__(self, node_id: str):
        super().__init__(node_id, "rewards")
        # IL input ports always declared; hidden in SB3 mode via _HIDDEN_PORTS
        self.inputs = {
            "actor_pipe": None,
            "command_pipe": None,
        }
        self.outputs = {"total_reward": None}
        self.parameters = {
            "backend": "sb3",
            "reward_terms": str(default_reward_terms()).replace("'", '"'),
            "std": "0.25",
            "threshold": "0.5",
        }

    @property
    def _HIDDEN_PORTS(self) -> set:
        if self.get_parameter("backend", "sb3") == "isaac_lab":
            return set()
        return self._IL_PORTS

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "total_reward": {
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
        return "Reward term configuration (SB3 / Isaac Lab)"


class TerminationsNode(TrainingBaseNode):
    """Unified termination configuration — works for both SB3 and Isaac Lab.

    The frontend is a single ``module_registry_editor`` items list for both
    backends; the registry it draws from is selected by the ``backend``
    parameter (``terminations_auto`` dispatch in
    ``bin/nodes/training_node_items.py``):

    - backend="sb3":       SB3 ``TERMINATION_REGISTRY``
        (fall_threshold_roll, fall_threshold_pitch, min_height,
         max_contact_impulse, joint_limit_violation, ...)
    - backend="isaac_lab": ``IL_TERMINATION_REGISTRY``
        (time_out, illegal_contact, base_height)

    There are intentionally NO IL-specific input ports or toggle widgets on
    this node — every IL termination knob lives inside the same items list,
    so the visual design is identical between backends. The IL compiler
    (``isaac_lab_config_compiler._terminations_cfg``) reads
    ``termination_conditions`` directly and emits the corresponding
    ``DoneTerm`` entries; the bodies regex used by the ``illegal_contact``
    term is hardcoded to a sensible quadruped default in the compiler.
    """

    def __init__(self, node_id: str):
        super().__init__(node_id, "terminations")
        self.inputs = {}
        self.outputs = {"termination_config": None}
        self.parameters = {
            "backend": "sb3",
            # Storage for the items list. The default reflects the SB3
            # registry; when the user (or a preset) flips backend to
            # isaac_lab, the editor swaps in the IL registry and the saved
            # value should be re-keyed to IL items at preset-apply time. See
            # ``IL_PRESETS`` in ``il_presets.py`` for the IL seed values.
            "termination_conditions": str(default_termination_conditions()).replace("'", '"'),
        }

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"termination_config": {"status": "configured", **self.get_config_dict()}}

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
        # Defaults are tuned for the most common case: a quadruped doing
        # AMP-style style imitation. Recommended values come from
        # AMP_for_hardware (NVIDIA / ETH Zurich) defaults + AMPTrainerNode
        # consistency. Fields the user MUST pick (motion_source / motion_file)
        # are intentionally empty.
        self.parameters = {
            # ── Motion source — user must pick ────────────────────────
            "motion_source":      "",   # "pack:..." / "loco:..." / "generate:..." (picker)
            "motion_file":        "",   # absolute path (picker, single-file mode)
            # ── Playback / phase ──────────────────────────────────────
            "phase_mode":         "loop",   # cycle endlessly — works for both gait + AMP
            "motion_fps":         "50.0",   # 50 Hz matches Go2 control rate; click "Auto"
                                            # to extract real native fps from the picked clip
            "random_start_phase": "true",   # ★ recommended ON: AMP_for_hardware standard;
                                            # prevents policy overfitting to clip openings,
                                            # works for both tracking + AMP
            # ── Tracking-only fields (hidden when consumer_mode == amp) ──
            "tracking_weight":    "1.0",
            "tracking_sigma":     "5.0",   # DeepMimic standard Gaussian sharpness
            # ── Imitation learning (BC) — SB3-only opt-in ─────────────
            # Default OFF: BC is the exception, not the rule. AMP path
            # ignores these fields entirely; SB3 users who want BC
            # pre-training can flip the master switch.
            "imitation_enabled":      "false",
            "bc_epochs":              "50",
            "bc_learning_rate":       "1e-3",
            "bc_batch_size":          "256",
            "bc_loss_type":           "mse",
            "bc_blend_steps":         "200000",
            "bc_blend_coef_start":    "0.5",
            "demo_num_trajectories":  "50",
            "demo_noise_std":         "0.0",
            "auto_inject_ref_obs":    "true",
            # UI-only preview toggle — not read by any training code.
            # ON  = physics PD tracking (what the RL agent sees).
            # OFF = pure kinematic replay (pose dropped into qpos).
            "preview_physics":    "true",
            # ── Phase_2 additive extension (AMP_design.yaml §3) ───────
            # Defaults preserve pre-phase_2 behavior (tracking is the
            # legacy path). User flips consumer_mode to "amp" / "both"
            # to engage the AMP discriminator pipeline; the AMP reward
            # coefficient default of 2.0 then matches the AMPTrainerNode
            # default and the AMP_for_hardware reference value.
            "format_id":          "unitport_npy",
            "consumer_mode":      "tracking",
            "amp_obs_fields":     "",
            # amp_reward_coef removed — authoritative knob is on amp_ppo_trainer
        }

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        motion_source = self.get_parameter("motion_source", "")
        motion_file = self.get_parameter("motion_file", "")

        resolved_files: List[str] = []
        if motion_source:
            try:
                resolved_files = self.resolve_motion_source_files(motion_source)
            except Exception as exc:
                try:
                    from src.system.core.logger import log_warning
                    log_warning(
                        f"[ReferenceMotion] failed to resolve motion_source "
                        f"{motion_source!r}: {exc}"
                    )
                except Exception:
                    pass
                resolved_files = []

        if resolved_files:
            resolved_file = resolved_files[0]
        else:
            resolved_file = motion_file
            if motion_file:
                resolved_files = [motion_file]

        return {
            "reference_motion_config": {
                "status":             "configured",
                "motion_source":      motion_source,
                "motion_file":        resolved_file,
                "motion_files":       list(resolved_files),
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
                # Phase_2 additive fields
                "format_id":          self.get_parameter("format_id", "unitport_npy"),
                "consumer_mode":      self.get_parameter("consumer_mode", "tracking"),
                "amp_obs_fields":     self.get_parameter("amp_obs_fields", ""),
            }
        }

    @classmethod
    def resolve_motion_source_files(cls, source: str) -> List[str]:
        """Resolve a ``motion_source`` identifier to absolute file paths.

        Handles three prefix families:

        - ``pack:<package>:<subdir>`` — expands to the full clip list
          inside ``custom_mods/archives/<package>/datasets/<subdir>``.
          Raises :class:`ValueError` when the pack is malformed, missing,
          or empty (callers decide the error policy).
        - ``loco:<env>:<task>`` — single ``.npy`` via the loco_mujoco bridge.
        - ``generate:<kind>`` — single synthetic ``.npy`` from the bridge.

        Empty / unknown sources return an empty list. This is the single
        source of truth for motion_source expansion — the window launcher,
        UI preview panel, and this node's own ``execute()`` all delegate
        here to avoid drift between paths.
        """
        source = (source or "").strip()
        if not source:
            return []

        if source.startswith("pack:"):
            from src.system.training.motion_library import expand_motion_pack
            return [str(p) for p in expand_motion_pack(source)]

        single = cls._resolve_single_bridge_source(source)
        return [single] if single else []

    @staticmethod
    def _resolve_single_bridge_source(source: str) -> Optional[str]:
        """Resolve a single-file ``loco:``/``generate:`` source to a ``.npy``."""
        source = source.strip()
        if not source:
            return None
        try:
            from src.system.training.loco_mujoco_bridge import (
                is_available, load_trajectory_as_npy,
                generate_standing_reference, generate_simple_walk_reference,
            )

            if source.startswith("loco:"):
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

    @classmethod
    def _resolve_motion_source(cls, source: str) -> Optional[str]:
        """Legacy single-path resolver. Prefer ``resolve_motion_source_files``.

        Kept for backward compatibility with callers that only consume a
        single file path (UI preview of a single clip, SB3 tracking path).
        For ``pack:`` sources returns the first clip; soft-fails to None
        on any error.
        """
        try:
            files = cls.resolve_motion_source_files(source)
        except Exception:
            return None
        return files[0] if files else None

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
            # RSI (Reference State Initialization) — APEX-style
            "rsi_prob":        "1.0",     # probability of using reference init
            "rsi_sample_mode": "frame_0", # frame_0 | uniform_phase
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
                "rsi_prob":       self.get_parameter("rsi_prob", "1.0"),
                "rsi_sample_mode": self.get_parameter("rsi_sample_mode", "frame_0"),
            }
        }

    def get_display_name(self) -> str:
        return "Init Pose"

    def get_description(self) -> str:
        return "Episode starting pose configuration"


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

class RobotNode(TrainingBaseNode):
    """Unified Robot asset node (General layer) — information display.

    Single source of truth for robot IDENTITY and STRUCTURAL metadata.
    This node is purely a display surface: ``asset_id`` is the only
    user-editable field (plus the ``active_override`` click toggle);
    everything else the user sees below is a read-only projection of
    the selected asset. **All settings live on the paired**
    :class:`ActorSettingNode` — do not add editable fields here.

    Display panels (read-only):
        §1 Asset files        — MJCF / USD / URDF with Loaded/✔ tags
        §2 Joint & IR list    — every body of the asset with its
                                 auto-resolved IR role (the old
                                 JointsMapping content, as a full
                                 structural view — not filtered by
                                 the current reward selection)
        §3 Init pose default  — the asset's native spawn pose
        §4 Actuator defaults  — built-in stiffness/damping/effort
                                 from the active asset file

    Emits a single ``robot_pipe`` output bundling asset identity,
    joint_names, body list, and default metadata so downstream consumers
    (especially the Actor Setting node's contact body picker) can read
    structural facts without re-querying the registry.
    """

    # Default asset the node is seeded with on first placement. The
    # canvas pipeline picks this up the moment RobotNode is instantiated
    # so users drop the node and immediately see a valid body list /
    # joint mapping rather than an empty state. Chosen as the quadruped
    # baseline that always resolves against the bundled registry.
    _INIT_ASSET_ID = "unitree_go2"

    def __init__(self, node_id: str):
        super().__init__(node_id, "robot")
        self.inputs = {}
        self.outputs = {"robot_pipe": None}
        self.parameters = {
            # The ONLY editable fields. Every other display row on this
            # node is derived from the asset registry entry at runtime.
            "asset_id": self._INIT_ASSET_ID,
            "active_override": "auto",   # auto | mjcf | usd | urdf
            # Auto-resolved IR mapping — stored here so the compiler can
            # read it as a single source of truth. The body_ir_validator
            # widget on this node is display-only; the value is seeded
            # from the asset's joint/body list and cached for downstream.
            "body_mapping": "",
        }

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        cfg = self.get_config_dict()
        asset_id = str(cfg.get("asset_id") or "").strip()

        asset_data: Dict[str, Any] = {}
        if asset_id:
            try:
                from src.system.training.robot_assets import get_asset, rescan
                rescan()
                asset_data = get_asset(asset_id).to_dict()
            except Exception as exc:
                cfg["_asset_error"] = str(exc)

        return {
            "robot_pipe": {
                "status": "configured",
                "asset_id": asset_id,
                "asset": asset_data,
                "active_override": str(cfg.get("active_override", "auto")),
                "body_mapping": str(cfg.get("body_mapping", "")),
            }
        }

    def get_display_name(self) -> str:
        return "Robot"

    def get_description(self) -> str:
        return "Robot asset information display (read-only). Settings live on Actor Setting."


class ActorSettingNode(TrainingBaseNode):
    """Actor-layer state, spawn pose, and actuation settings.

    Partner of :class:`RobotNode` — together they form the complete
    Actor canvas block. Owns EVERY user-editable knob that attaches
    to the robot:

        §1 Init pose           — world-frame spawn X/Y/Z (sliders)
        §2 Init joint angles   — delegated to an optional upstream
                                  :class:`JointInitNode` via the
                                  ``joint_init`` input port. The visual
                                  per-joint editor lives on that node.
        §3 Contact sensors     — which bodies emit contact readings
                                  (picker sourced from connected Robot)
        §4 Actuator overrides  — stiffness/damping/effort/velocity

    Reads ``robot_pipe`` from the Robot node (for the structural body
    list that feeds the contact picker) and re-emits a superset
    ``actor_pipe`` that downstream consumers (rewards, domain rand,
    observation, action, velocity command, …) read from.
    """

    # joint_init is optional — ActorSetting can compile without it.
    # When connected, the upstream :class:`JointInitNode` provides the
    # authoritative per-joint angle dict.
    _OPTIONAL_INPUTS: set = {"joint_init"}

    def __init__(self, node_id: str):
        super().__init__(node_id, "actor_setting")
        self.inputs = {"robot_pipe": None, "joint_init": None}
        self.outputs = {"actor_pipe": None}
        self.parameters = {
            # §1 Init pose
            "init_pos_x": "0.0",
            "init_pos_y": "0.0",
            "init_pos_z": "0.4",
            # §2 Init joint angles — fallback only. When the joint_init
            # input port is connected, the upstream JointInitNode's
            # ``angles`` param wins.
            "init_joint_angles": "{}",
            # §3 Contact sensors
            "contact_body_names": "[]",
            "contact_history_length": "3",
            "contact_track_air_time": "true",
            # §4 Actuator overrides
            "stiffness": "25.0",
            "damping": "0.5",
            "effort_limit": "30.0",
            "velocity_limit": "30.0",
            # §5 Action space (absorbed from IL Action Config)
            "action_joint_names_expr": "[]",
            "action_scale": "0.25",
            "action_use_default_offset": "true",
        }

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        cfg = self.get_config_dict()
        robot_pipe = inputs.get("robot_pipe") or {}

        def _f(key: str, default: float) -> float:
            try:
                return float(cfg.get(key, default))
            except (TypeError, ValueError):
                return default

        # §2: joint_init port wins when connected.
        joint_init_in = inputs.get("joint_init")
        if isinstance(joint_init_in, dict):
            angles_src = joint_init_in.get("angles") or joint_init_in.get(
                "init_joint_angles", ""
            )
        else:
            angles_src = cfg.get("init_joint_angles", "{}")

        return {
            "actor_pipe": {
                "status": "configured",
                "robot_pipe": robot_pipe,
                "init_pose": {
                    "x": _f("init_pos_x", 0.0),
                    "y": _f("init_pos_y", 0.0),
                    "z": _f("init_pos_z", 0.4),
                },
                "init_joint_angles": str(angles_src or "{}"),
                "contact_sensor": {
                    "body_names": str(cfg.get("contact_body_names", "[]")),
                    "history_length": str(cfg.get("contact_history_length", "3")),
                    "track_air_time": str(cfg.get("contact_track_air_time", "true")),
                },
                "stiffness": str(cfg.get("stiffness", "25.0")),
                "damping": str(cfg.get("damping", "0.5")),
                "effort_limit": str(cfg.get("effort_limit", "30.0")),
                "velocity_limit": str(cfg.get("velocity_limit", "30.0")),
                "action_space": {
                    "joint_names_expr": str(cfg.get("action_joint_names_expr", "[]")),
                    "scale": str(cfg.get("action_scale", "0.25")),
                    "use_default_offset": str(cfg.get("action_use_default_offset", "true")),
                },
            }
        }

    def get_display_name(self) -> str:
        return "Actor Setting"

    def get_description(self) -> str:
        return "Actor init pose, contact sensors, and actuator overrides"


class JointInitNode(TrainingBaseNode):
    """Visual per-joint init angle editor (General layer).

    Delegate node connected to :class:`ActorSettingNode` via the
    optional ``joint_init`` input port. Carries a single ``angles``
    parameter — a JSON dict ``{joint_name: radians}`` — but exposes it
    through the ``joint_init_editor`` widget (per-joint popup editor,
    sourced from the Robot node's joint_names), so the user never has
    to hand-edit JSON.

    Emits a ``joint_init`` output which ActorSettingNode treats as the
    authoritative per-joint posture override. When disconnected,
    ActorSetting falls back to its own ``init_joint_angles`` param
    (empty by default = use the asset's native pose).
    """

    def __init__(self, node_id: str):
        super().__init__(node_id, "joint_init")
        self.inputs = {}
        self.outputs = {"joint_init": None}
        self.parameters = {
            "angles": "{}",   # JSON: joint_name → radians
        }

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        cfg = self.get_config_dict()
        return {
            "joint_init": {
                "status": "configured",
                "angles": str(cfg.get("angles", "{}")),
            }
        }

    def get_display_name(self) -> str:
        return "Joint Init"

    def get_description(self) -> str:
        return "Visual per-joint init angle editor — feeds ActorSetting.joint_init"


class PlayGroundSettingNode(TrainingBaseNode):
    """Unified scene / playground setting node (General layer, §2 Scene).

    Single source of truth for everything scene-related. Subsumes the
    former ``SceneConfigNode`` (SB3), ``ILTerrainConfigNode`` (IL) and
    the full ``ILSimulationConfigNode`` (gravity + sim_dt + gpu_*).
    There is no separate §6 Sim node — PlayGround owns every scene-
    scoped engine knob and the IL compiler reads them from here.

    The ``scene_id`` parameter resolves against
    :mod:`src.system.training.scene_registry`; the picker filters the
    registry by training backend (sniffed from the canvas Train node)
    and robot family (sniffed from the canvas Robot node), so users
    only see entries that are actually runnable in the current context.

    Core parameters always visible:
        scene_id, gravity_z, arena_extent_x/y, friction_{static,dynamic},
        height_scan_enabled

    Conditional on ``scene_type == "rough"``:
        roughness_amplitude, slope_max, curriculum_enabled,
        difficulty_levels

    Conditional on ``height_scan_enabled == "true"``:
        scan_resolution, scan_size_x, scan_size_y

    Note: ``height_scan_*`` is deliberately NOT backend-gated — Isaac
    Lab ships it out-of-the-box, and the MuJoCo side can add a
    heightfield ray-cast sensor against the same parameters when the
    integration lands (tracked as a future enhancement in
    UnitreeGymEnv).
    """

    _INIT_SCENE_ID = "flat_ground"

    def __init__(self, node_id: str):
        super().__init__(node_id, "play_ground_setting")
        self.inputs = {}
        self.outputs = {"scene_pipe": None}
        self.parameters = {
            # §2.1 Scene identity
            "scene_id": self._INIT_SCENE_ID,
            "scene_type": "flat",       # auto-synced from scene_id on pick
            # §2.2 World physics
            "gravity_z": "-9.81",
            # §2.3 Arena
            "arena_extent_x": "10.0",
            "arena_extent_y": "10.0",
            # §2.4 Ground material
            "friction_static": "1.0",
            "friction_dynamic": "0.8",
            # §2.5 Rough-terrain knobs (conditional on scene_type)
            "roughness_amplitude": "0.08",
            "slope_max": "20.0",
            "curriculum_enabled": "false",
            "difficulty_levels": "10",
            # §2.6 Height scan (backend-agnostic)
            "height_scan_enabled": "false",
            "scan_resolution": "0.1",
            "scan_size_x": "1.6",
            "scan_size_y": "1.0",
            # §2.7 Sim engine globals (merged from ILSimulationConfigNode)
            "sim_dt": "0.005",
            "gpu_max_rigid_contact_count": "524288",
            "gpu_max_rigid_patch_count": "81920",
        }

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        cfg = self.get_config_dict()

        def _f(key: str, default: float) -> float:
            try:
                return float(cfg.get(key, default))
            except (TypeError, ValueError):
                return default

        def _i(key: str, default: int) -> int:
            try:
                return int(cfg.get(key, default))
            except (TypeError, ValueError):
                return default

        def _b(key: str) -> bool:
            return str(cfg.get(key, "false")).strip().lower() == "true"

        scene_id = str(cfg.get("scene_id") or "").strip()
        scene_type = str(cfg.get("scene_type") or "flat").strip()
        scene_data: Dict[str, Any] = {}
        if scene_id:
            try:
                from src.system.training.scene_registry import get_scene, rescan
                rescan()
                scene_data = get_scene(scene_id).to_dict()
                # Trust the registry for scene_type (prevents drift when
                # the user overrides scene_id via the picker but the
                # hidden scene_type param lags behind).
                scene_type = scene_data.get("scene_type", scene_type)
            except Exception as exc:
                cfg["_scene_error"] = str(exc)

        pipe: Dict[str, Any] = {
            "status": "configured",
            "scene_id": scene_id,
            "scene_type": scene_type,
            "scene": scene_data,
            "gravity": {"x": 0.0, "y": 0.0, "z": _f("gravity_z", -9.81)},
            "arena": {
                "extent_x": _f("arena_extent_x", 10.0),
                "extent_y": _f("arena_extent_y", 10.0),
            },
            "friction": {
                "static": _f("friction_static", 1.0),
                "dynamic": _f("friction_dynamic", 0.8),
                "restitution": 0.0,
            },
            "roughness": None,
            "curriculum": None,
            "height_scan": None,
        }

        if scene_type == "rough":
            pipe["roughness"] = {
                "amplitude": _f("roughness_amplitude", 0.08),
                "slope_max": _f("slope_max", 20.0),
            }
            if _b("curriculum_enabled"):
                pipe["curriculum"] = {
                    "enabled": True,
                    "difficulty_levels": _i("difficulty_levels", 10),
                }

        if _b("height_scan_enabled"):
            pipe["height_scan"] = {
                "enabled": True,
                "resolution": _f("scan_resolution", 0.1),
                "size_x": _f("scan_size_x", 1.6),
                "size_y": _f("scan_size_y", 1.0),
            }

        pipe["sim"] = {
            "dt": _f("sim_dt", 0.005),
            "gpu_max_rigid_contact_count": _i("gpu_max_rigid_contact_count", 524288),
            "gpu_max_rigid_patch_count": _i("gpu_max_rigid_patch_count", 81920),
        }

        return {"scene_pipe": pipe}

    def get_display_name(self) -> str:
        return "Play Ground Setting"

    def get_description(self) -> str:
        return "Unified scene config: registry-backed playground + gravity + arena + friction + terrain + height scan"


class TrainingCommandsNode(TrainingBaseNode):
    """Unified training / sim / sim2real command node (General layer).

    Single source of truth for the ``command`` sub-vector that enters
    the policy observation. Replaces :class:`ILVelocityCommandNode`
    (and the deployment-only :class:`ILGamepadInputNode`) — the same
    schema now drives episode sampling during training AND the
    runtime CommandBus binding at deployment, eliminating the sim2real
    distribution-shift hazard.

    **PR-1 scope** (this iteration): 3 continuous velocity channels
    (``lin_vel_x`` / ``lin_vel_y`` / ``ang_vel_z``) + sampler controls.
    Every field maps 1:1 with the legacy ``il_velocity_cmd`` node, so
    the IL compiler's fallback path accepts both.

    **PR-2** will add gait parameterisation (step frequency, phase
    offsets, body height, step height) and a preset table — the
    Walk These Ways style continuous gait space.

    **PR-3** will add a calibration wizard (controller → auto-detect
    range) plus user-defined discrete / button channels, along with
    consistency checks against the Rewards + Reference Motion nodes.

    Inputs
    ------
    ``reference_motion_pipe`` *(optional)* — reserved port for the
    future Conditional AMP path; not consumed in PR-1 / PR-2. Keeping
    the slot visible now lets users wire it without reshaping the node
    later.

    Outputs
    -------
    ``command_pipe`` — dict carrying the serialised
    :class:`CommandSchema` plus episode-sampler metadata. Consumed by:

      * IL compiler ``_commands_cfg`` (velocity range emission)
      * SB3 runner's command curriculum (future wiring)
      * Bundle exporter → ``deploy_contract.commands`` in manifest.yaml
      * Runtime CommandBus ``gamepad_source.py`` (via manifest)
    """

    _OPTIONAL_INPUTS: set = {"reference_motion_pipe"}

    def __init__(self, node_id: str):
        super().__init__(node_id, "training_commands")
        self.inputs = {"reference_motion_pipe": None}
        self.outputs = {"command_pipe": None}
        self.parameters = {
            # §1 Lever / gain coupling (stick [-1,1] → velocity).
            # Training range is DERIVED from these four gains:
            #   lin_vel_x_range = (-gain_lin_backward, +gain_lin_forward)
            #   lin_vel_y_range = (-gain_lin_lateral, +gain_lin_lateral)
            #   ang_vel_z_range = (-gain_yaw,         +gain_yaw)
            # Deployment CommandBus reads the same gains from manifest
            # and computes vel = stick × gain — single source of truth
            # for training + runtime + sim2real.
            "gain_lin_forward":  "2.0",   # stick=+1 → +this (m/s)
            "gain_lin_backward": "1.0",   # stick=-1 → -this (m/s)
            "gain_lin_lateral":  "0.5",   # stick=±1 → ±this (m/s)
            "gain_yaw":          "1.5",   # stick=±1 → ±this (rad/s)
            # §2 Stick → vel deployment-side mapping curve (training
            # sampler is ALWAYS uniform over the derived range; these
            # parameters only reshape the runtime UX feel).
            "mapping_mode":  "linear",    # linear | deadzone | exponential
            "deadzone":      "0.10",      # honoured when mode == deadzone
            "curve_exponent": "2.00",     # honoured when mode == exponential
            # §3 Walk These Ways gait parameterisation (P2).
            # When gait_enabled, seven extra channels (frequency +
            # 4 phase offsets + body height + step height) are
            # appended to the policy command vector. Training sampler
            # draws them uniformly from the ranges; runtime uses the
            # preset table as D-pad shortcuts.
            "gait_enabled":         "false",
            "gait_frequency_range": "[1.5, 3.5]",
            "body_height_range":    "[0.28, 0.40]",
            "step_height_range":    "[0.03, 0.15]",
            "gait_presets":         "",   # empty ⇒ use bundled DEFAULT_PRESETS
            # §4 Episode sampler controls
            "resampling_time_range": "[10.0, 10.0]",
            "zero_command_probability": "0.1",
            # §5 Runtime guardrails
            "runtime_clip": "true",
        }

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        cfg = self.get_config_dict()
        # Build the schema lazily so the node stays Qt-free and the
        # CommandSchema module (which validates kinds + ranges) is the
        # single point that can reject a malformed param set.
        from src.system.training.command_schema import CommandSchema
        try:
            schema = CommandSchema.from_node_dict(cfg)
            schema_dict = schema.to_dict()
            error = None
        except Exception as exc:
            schema_dict = {}
            error = f"{type(exc).__name__}: {exc}"

        pipe: Dict[str, Any] = {
            "status": "configured" if error is None else "error",
            "schema": schema_dict,
        }
        if error is not None:
            pipe["_error"] = error
        # Leave the (unused in PR-1) reference_motion_pipe slot visible
        # on the wire so downstream CAMP code can pick it up without
        # reshaping the node.
        ref = inputs.get("reference_motion_pipe")
        if ref is not None:
            pipe["reference_motion_pipe"] = ref
        return {"command_pipe": pipe}

    def get_display_name(self) -> str:
        return "Training Commands"

    def get_description(self) -> str:
        return "Unified training/sim/sim2real command schema — single source of truth shared with the deployment CommandBus"


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

    # robot_pipe is now optional — the new RobotNode is wired directly
    # into ActorSettingNode, not Physics. PhysicsConfig no longer depends
    # on robot identity (sim_dt / control_dt / episode_max_steps are all
    # robot-agnostic). The port is kept for canvas compatibility but may
    # be disconnected without compile errors.
    _OPTIONAL_INPUTS: set = {"robot_pipe", "robot_spec"}

    def __init__(self, node_id: str):
        super().__init__(node_id, "physics_config")
        self.inputs = {"robot_pipe": None, "robot_spec": None}  # both optional
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
    """Unified domain randomisation node — works for both SB3 and Isaac Lab.

    backend="sb3":       SB3 DR params (mass_range, friction_range, motor, push, schedule)
    backend="isaac_lab":  IL DR events (friction, mass, push, init_pose, joint_noise with mode toggles)
    """

    _IL_PORTS = {"actor_pipe"}

    def __init__(self, node_id: str):
        super().__init__(node_id, "domain_rand")
        self.inputs = {"actor_pipe": None}
        self.outputs = {"domain_rand_config": None}
        self.parameters = {
            "backend": "sb3",
            # ── SB3 params ──
            "enabled": "true",
            "mass_range": "[0.8, 1.2]",
            "friction_range": "[0.5, 1.5]",
            "motor_strength_range": "[0.9, 1.1]",
            "joint_damping_range": "[0.95, 1.05]",
            "obs_noise_std": "0.01",
            "push_robot": "false",
            "push_interval_steps": "200",
            "push_force_range": "[50, 150]",
            "rand_schedule":          "none",
            "rand_schedule_end_step": "500000",
            # ── IL params ──
            "enable_friction_rand": "true",
            "friction_mode": "reset",
            "static_friction_range": "[0.5, 1.25]",
            "dynamic_friction_range": "[0.5, 1.25]",
            "restitution_range": "[0.0, 0.4]",
            "enable_mass_rand": "true",
            "mass_mode": "reset",
            "mass_target_body": "base",
            "mass_offset_range": "[-0.5, 0.5]",
            "enable_external_push": "true",
            "push_mode": "interval",
            "push_interval_range_s": "[10.0, 15.0]",
            "push_velocity_x_range": "[-0.5, 0.5]",
            "push_velocity_y_range": "[-0.5, 0.5]",
            "enable_init_pose_rand": "true",
            "init_pose_mode": "reset",
            "init_pos_x_range": "[-0.5, 0.5]",
            "init_pos_y_range": "[-0.5, 0.5]",
            "init_yaw_range": "[-3.14, 3.14]",
            "enable_joint_noise": "true",
            "joint_noise_mode": "reset",
            "joint_pos_noise": "0.25",
            "joint_vel_noise": "0.1",
            # DR schedule (RSI-AMP staged training)
            "rand_schedule": "none",           # none | linear
            "rand_schedule_start_step": "0",   # global step DR activates
            "rand_schedule_end_step": "500000", # global step DR reaches full intensity
        }

    @property
    def _HIDDEN_PORTS(self) -> set:
        if self.get_parameter("backend", "sb3") == "isaac_lab":
            return set()
        return self._IL_PORTS

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"domain_rand_config": {"status": "configured", **self.get_config_dict()}}

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
            "total_reward": None,  # combined output → TaskConfig.total_reward
            "total_steps":  None,  # AUTO-computed recommended total env steps → AlgorithmConfigNode
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
            "total_reward": stage0,
            "total_steps":  self.compute_recommended_total_steps(),
        }

    def get_port_types(self) -> Dict[str, Dict[str, str]]:
        """Stage input ports carry total_reward data; total_steps output carries int."""
        return {
            "in":  {k: "total_reward" for k in self.inputs},
            "out": {
                "total_reward": "total_reward",
                "total_steps":  "int",
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
    Environment assembler node (Layer B) — SB3 aggregation point.

    Merges §1 Actor block (actor_pipe), §2 Scene (scene_pipe),
    §4 Training Commands (command_pipe), plus the SB3-only task-layer
    configs (physics / task / obs&action / domain_rand / ref_motion /
    init_pose) into a single ``env_config`` dict consumed by TrainNode.

    scene_pipe / actor_pipe / command_pipe are declared **optional** at
    the node level because legacy SB3 canvases won't have them wired.
    The C6 cross-section validator still requires PlayGroundSetting to
    terminate somewhere on the compile pipeline — either at a trainer
    (IL) or at EnvAssembler.scene_pipe (SB3).
    """

    # Optional inputs: everything that has a sane default when not wired.
    _OPTIONAL_INPUTS: set = {
        "domain_rand_config",
        "reference_motion_config",
        "init_pose_config",
        "actor_pipe",
        "scene_pipe",
        "command_pipe",
    }

    def __init__(self, node_id: str):
        super().__init__(node_id, "env_assembler")
        self.inputs = {
            # Unified 6-section pipes
            "actor_pipe": None,      # optional — §1 Actor block
            "scene_pipe": None,      # optional — §2 Scene (PlayGroundSetting)
            "command_pipe": None,    # optional — §4 Training Commands
            # SB3 task-layer configs
            "physics_config": None,
            "task_config": None,
            "obs_action_config": None,
            "domain_rand_config": None,       # optional
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
            # Unified 6-section pipes (new framework). Consumers should
            # prefer these over the legacy SceneConfig/TaskConfig fields.
            "actor": inputs.get("actor_pipe") or {},
            "scene": inputs.get("scene_pipe") or {},
            "commands": inputs.get("command_pipe") or {},
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
    start_point : str
        The picker's source token. Three shapes are recognized:

        - ``__new__``              — scratch (``load_mode`` ignored)
        - ``__latest_export__``    — resolve to the workspace's newest export
        - ``asset:<asset_id>``     — legacy TrainingAssetRegistry lookup
        - ``run:<abs .pt path>``   — direct run-dir checkpoint (no registry)
    asset_id : str
        Asset ID from TrainingAssetRegistry (e.g. "sac_unitree_go2_mujoco").
        Only consulted for the ``asset:`` / legacy path.
    checkpoint_file : str
        Relative path within the asset directory (e.g. "best/best_model.zip").
        Empty = auto-select ``primary_checkpoint``. Only consulted for the
        legacy path.
    load_mode : str
        - ``scratch``            — start from scratch (base_asset ignored)
        - ``resume``             — load weights + optimizer + discriminator
          (continue a run with the same algorithm)
        - ``warm_start_actor``   — copy actor-critic weights only; optimizer,
          discriminator, and iteration counter are freshly initialized.
          This is the correct mode for seeding an AMP-PPO run from a pure
          PPO checkpoint, or for reusing a policy under a new reward /
          task layout.
    """

    _LOAD_MODES = ("scratch", "resume", "warm_start_actor")

    def __init__(self, node_id: str):
        super().__init__(node_id, "base_asset")
        self.inputs = {}
        # Output slot is named after the semantic payload — "checkpoint".
        # (Node type is still "base_asset" for registry/serialisation.)
        self.outputs = {"checkpoint": None}
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

        abs_checkpoint_path = ""
        if load_mode != "scratch":
            # Path 1: direct run-dir checkpoint, encoded in start_point.
            # Token shape: ``run:<absolute .pt path>``. Bypasses the
            # TrainingAssetRegistry entirely — the picker stores the
            # full path and the resolver just trusts the filesystem.
            sp = str(start_point or "").strip()
            if sp.startswith("run:"):
                abs_checkpoint_path = sp[len("run:"):].strip()
            elif asset_id:
                # Path 2: legacy TrainingAssetRegistry lookup.
                try:
                    from src.system.training.training_asset_registry import TrainingAssetRegistry
                    entry = TrainingAssetRegistry().get(asset_id)
                    ckpt = checkpoint_file or entry.primary_checkpoint
                    if ckpt:
                        abs_checkpoint_path = str(entry.asset_path / ckpt)
                except Exception:
                    pass

        return {
            "checkpoint": {
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

    _OPTIONAL_INPUTS: set = {"checkpoint", "total_steps"}
    _HIDDEN_PORTS: set = {"total_steps"}  # merged into NodeRow for total_timesteps param

    def __init__(self, node_id: str):
        super().__init__(node_id, "algo_config")
        self.inputs = {
            "checkpoint":  None,   # optional — from BaseAssetNode; None = scratch
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
            "in":  {"checkpoint": "checkpoint", "total_steps": "int"},
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
        checkpoint = inputs.get("checkpoint") or {}
        # When a BaseAssetNode is connected, its load_mode takes precedence
        # over the local resume_mode parameter.
        resume_mode = (
            checkpoint.get("load_mode")
            if checkpoint.get("load_mode") and checkpoint.get("load_mode") != "scratch"
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
                "base_asset": checkpoint,
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
    """Unified Export node (§5, General layer — replaces SB3 Export +
    IL Policy Exporter).

    Four responsibilities (see Export Node design doc):

      §1 Output Manifest — user-visible list of every file the next
         training run will produce, with Open buttons and ⚠
         "Will Overwrite" tags on files that already exist. Fields
         ``bundle_name`` + ``version`` are the ONLY naming knobs; no
         more hidden auto-increment.

      §2 Start Point Compatibility — the Start Point → Trainer
         ``checkpoint`` edge is the single carrier. ExportNode reads
         warm-start context out of ``train_pipe`` (the Trainer forwards
         the checkpoint payload it received). No dedicated
         start_point_pipe input — that was a redundant fan-out.

      §3 Review Environment — backend selector (MuJoCo / IsaacSim /
         future Newton), review scene picker filtered by engine
         compatibility, and a Launch Review button that delegates
         subprocess lifetime to the workspace layer. Added in a
         later PR; the param scaffolding is on this node already.

    This iteration (§1) lands the Output Manifest section and keeps
    §2/§3 params as placeholders that the compiler ignores. Old SB3
    fields (``export_target``, ``export_onnx``, ``include_norm_stats``,
    ``overwrite``) stay in place so bundle_exporter.py / isaac_lab_config_compiler.py
    don't need to change yet.
    """

    # eval_config is optional — the node still compiles without it.
    # TrainingNodeItem draws "(opt)" in front of its label.
    _OPTIONAL_INPUTS: set = {"eval_config"}

    def __init__(self, node_id: str):
        super().__init__(node_id, "export")
        self.inputs = {
            "train_pipe": None,
            "eval_config": None,       # optional (legacy SB3 port)
        }
        self.outputs = {"bundle_path": None}
        self.parameters = {
            # §1 Output manifest
            # Default is the <NEW> sentinel; SafetyReview (S6) blocks
            # training launch until the user picks a real name, so fresh
            # canvases can't accidentally write to "trained_policy/".
            "bundle_name":        "<NEW>",
            "version":            "v1",
            "include_onnx":       "true",   # alias → export_onnx for new UI
            "include_torchscript": "true",  # alias → export_torchscript
            "include_normalization": "true",  # alias → include_norm_stats
            # Legacy fields kept for bundle_exporter.py / the IL compiler.
            # UI no longer shows them directly — the toggles above are the
            # user-facing names, but from_node_dict maps both so old
            # canvases and the back-end consumers still work.
            "export_target":      "runtime_bundle",
            "export_onnx":        "true",
            "export_torchscript": "true",
            "include_norm_stats": "true",
            "overwrite":          "true",
            # §3 Review environment
            "review_backend":     "mujoco",    # mujoco | isaac_sim | newton (future)
            "review_scene_id":    "flat_ground",  # filtered by review_backend via scene registry
            "review_launch":      "",          # virtual — owned by review_launch_button widget
            "auto_import":        "true",
        }

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        train_pipe = inputs.get("train_pipe") or {}
        bundle_name = self.get_parameter("bundle_name", "") or "trained_policy"
        version = self.get_parameter("version", "v1") or "v1"
        overwrite = str(self.get_parameter("overwrite", "true")).lower() == "true"
        # Overwrite=ON → flat dir name, successive runs land in the
        # same folder. Overwrite=OFF → append the user-set version
        # tag so each run produces a distinct bundle directory.
        if overwrite:
            full_name = bundle_name
        else:
            full_name = f"{bundle_name}_{version}" if version else bundle_name
        bundle_path = train_pipe.get(
            "bundle_path", f"mock/training/exported/{full_name}"
        )
        return {"bundle_path": bundle_path}

    def get_display_name(self) -> str:
        return "Export"

    def get_description(self) -> str:
        return "Unified policy export — bundle output manifest, Start Point compatibility, review environment"


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


# ===========================================================================
# Isaac Lab Granular Node Graph — IL_A … IL_F layers
# ===========================================================================
# The legacy monolithic IsaacLabTaskNode / IsaacLabTrainNode /
# IsaacLabExportNode have been removed. The granular IL nodes below
# (plus the unified Robot / ActorSetting / PlayGroundSetting / Training
# Commands / Export nodes) cover every configuration knob they used
# to expose, and the IL compiler reads from them directly.
# ===========================================================================

# ILSimulationConfigNode removed — merged into PlayGroundSettingNode.
# PlayGround now owns sim_dt, gpu_max_rigid_contact_count,
# gpu_max_rigid_patch_count, and gravity_z. The IL compiler reads these
# via _find_by_type("play_ground_setting").

# ── GROUP B: Observation Pipeline (IL_B layer) ──────────────────────────

class ILObservationNode(TrainingBaseNode):
    """Isaac Lab observation configuration — all obs terms in one node.

    Stores selected terms as JSON ``{"term_key": 1.0, ...}`` via
    ``module_registry_editor`` (same UX pattern as SB3 RewardsNode).
    """

    # actor_pipe and command_pipe are both optional at the node level —
    # the IL compiler reads obs_terms directly from this node's params
    # and does not traverse these wires. Keeping them optional avoids
    # hard-wiring the canvas layout: users can connect them for
    # visual provenance but nothing breaks if they don't.
    _OPTIONAL_INPUTS: set = {"actor_pipe", "command_pipe"}

    def __init__(self, node_id: str):
        super().__init__(node_id, "il_observation")
        # Ports align with the 6-section unified pipes:
        #   actor_pipe    ← Actor Setting (carries robot_pipe + sensors +
        #                   action space + actuator gains)
        #   command_pipe  ← Training Commands (carries velocity + gait
        #                   command schema)
        # Old ports (robot_entity, command_source, action_source,
        # height_scanner) removed — the scene-level height_scan config
        # lives on the Play Ground Setting node and the IL compiler
        # reads it directly via _find_by_type("play_ground_setting").
        self.inputs = {
            "actor_pipe": None,
            "command_pipe": None,
        }
        self.outputs = {"obs_vector": None}
        from src.system.training.task_module_registry import default_il_obs_terms
        import json as _json
        self.parameters = {
            "obs_terms": _json.dumps(default_il_obs_terms()),
            "group_name": "policy",
            "enable_corruption": "true",
            "corruption_noise_std": "0.05",
        }

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"obs_vector": {"status": "configured", **self.get_config_dict()}}

    def get_display_name(self) -> str:
        return "IL Observation"

    def get_description(self) -> str:
        return "Isaac Lab observation terms (multi-term registry editor)"


# ── GROUP C: Action Space (IL_B layer) ──────────────────────────────────

# ── GROUP G: Training & Export (IL_F layer) ──────────────────────────────

class ILPolicyNetworkNode(TrainingBaseNode):
    """MLP actor-critic architecture config."""

    def __init__(self, node_id: str):
        super().__init__(node_id, "il_policy_network")
        # ``action_entity`` input was removed when ILActionConfigNode
        # was deprecated — the action space is now part of actor_pipe
        # via ActorSetting. The policy network reads obs_vector only
        # and the IL compiler derives the action space from the
        # ActorSettingNode params directly.
        self.inputs = {
            "obs_vector": None,
        }
        self.outputs = {"policy": None}
        self.parameters = {
            "actor_hidden_dims": "[128, 64, 32]",
            "critic_hidden_dims": "[128, 64, 32]",
            "activation": "elu",
            "init_noise_std": "-1.0",
            "disc_hidden_dims": "[1024, 512]",
        }

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"policy": {"status": "configured", **self.get_config_dict()}}

    def get_display_name(self) -> str:
        return "IL Policy Network"

    def get_description(self) -> str:
        return "Isaac Lab MLP actor-critic architecture configuration"


class ILPPOTrainerNode(TrainingBaseNode):
    """Unified Isaac Lab trainer — PPO / AMP_PPO selectable.

    Single node replacing the former IL PPO + AMP PPO split. The
    ``training_mode`` parameter picks the training algorithm and the
    UI layer conditionally shows the AMP-specific hyperparameter rows
    when ``training_mode == "AMP_PPO"``. The reference_motion_config
    input becomes required in that mode; otherwise it is ignored.

    ``task_name`` is no longer a user-facing parameter — the workspace
    infers it from the robot / scene via ``_infer_il_task_name``.
    """

    # ``checkpoint`` (Start Point resume) is always optional. The
    # ``reference_motion_config`` input is topologically always
    # declared so the edge is addressable in PPO mode for users who
    # want it wired preemptively, and made required via the C2
    # cross-section validator when training_mode is AMP_PPO.
    _OPTIONAL_INPUTS: set = {"checkpoint", "reference_motion_config"}

    TRAINING_MODE_CHOICES = ("PPO", "AMP_PPO")

    def __init__(self, node_id: str):
        super().__init__(node_id, "il_ppo_trainer")
        self.inputs = {
            "policy": None,
            "total_reward": None,
            "termination_config": None,
            "domain_rand_config": None,
            "obs_vector": None,
            # §2 Scene — PlayGroundSetting wires its scene_pipe here so
            # the trainer is the single aggregation point for every
            # section that feeds a training run.
            "scene_pipe": None,
            # Optional — Start Point resume/warm-start. Forwarded into
            # train_pipe so ExportNode can read warm-start context.
            "checkpoint": None,
            # Optional in PPO mode, required in AMP_PPO mode — the
            # C2 cross-section validator enforces the mode gate.
            "reference_motion_config": None,
        }
        # Output port name matches ExportNode.inputs["train_pipe"] so the
        # Trainer → Export edge is a name-for-name connection.
        self.outputs = {
            "train_pipe": None,
            "training_metrics": None,
        }
        self.parameters = {
            # §0 Training mode — picks PPO or AMP_PPO. Drives the
            # conditional AMP rows in the UI layer and the C2 validator.
            "training_mode": "PPO",
            # ── 16 shared PPO hyperparams ──
            "max_iterations": "1500",
            "num_steps_per_env": "24",
            "num_learning_epochs": "5",
            "num_minibatches": "4",
            "learning_rate": "0.001",
            "discount_factor": "0.99",
            "gae_lambda": "0.95",
            "clip_param": "0.2",
            "entropy_coef": "0.01",
            "value_loss_coef": "1.0",
            "max_grad_norm": "1.0",
            "schedule": "adaptive",
            "desired_kl": "0.01",
            "save_interval": "100",
            "seed": "42",
            "headless": "true",
            # ── 7 AMP-specific hyperparams (visible only when
            # training_mode == AMP_PPO). Bijective with AMPConfig. ──
            "amp_reward_coef": "2.0",
            "task_reward_lerp": "0.5",
            "disc_grad_penalty": "10.0",
            "disc_label_smoothing": "0.9",
            "amp_replay_buffer_size": "1000000",
            "num_preload_transitions": "2000000",
            "disc_lr": "1e-4",
            # RSI-AMP: lerp schedule preset or "custom"
            "lerp_schedule": "none",
            "lerp_schedule_json": "",
            # ── Staged training (Phase 1 Stage Pipeline) ──
            # When enabled, the runner iterates through stages in order,
            # applying each stage's overrides on top of the baseline config.
            "stages_enabled": "false",
            "stages_json": "[]",
            "stage_checkpoint_strategy": "both",
        }

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        training_mode = str(self.get_parameter("training_mode", "PPO") or "PPO").strip()
        if training_mode not in self.TRAINING_MODE_CHOICES:
            training_mode = "PPO"

        cfg = {
            "status": "configured",
            "algorithm": "AMP_PPO" if training_mode == "AMP_PPO" else "PPO",
            "training_mode": training_mode,
            **self.get_config_dict(),
        }

        # Resolve lerp_schedule preset names to JSON strings
        _LERP_PRESETS = {
            "none":               "",
            "slow_anneal":        '{"start":0.75,"end":0.3,"over_iters":3000}',
            "fast_anneal":        '{"start":0.75,"end":0.3,"over_iters":1000}',
            "warmup_then_anneal": '{"start":0.2,"end":0.6,"over_iters":2000}',
        }
        raw_lerp = str(cfg.get("lerp_schedule", "") or "").strip()
        if raw_lerp in _LERP_PRESETS:
            cfg["lerp_schedule"] = _LERP_PRESETS[raw_lerp]
        elif raw_lerp == "custom":
            cfg["lerp_schedule"] = str(cfg.get("lerp_schedule_json", "") or "").strip()

        if training_mode == "AMP_PPO":
            # Soft check — canonical check lives in
            # training_compatibility.py / the C2 validator.
            ref_cfg = inputs.get("reference_motion_config") or {}
            consumer_mode = "tracking"
            if isinstance(ref_cfg, dict):
                consumer_mode = str(ref_cfg.get("consumer_mode", "tracking"))
            ok = consumer_mode in ("amp", "both")
            if not ok:
                cfg["status"] = "error"
                cfg["error"] = (
                    f"AMP_PPO mode requires the upstream ReferenceMotionNode "
                    f"to have consumer_mode in {{'amp', 'both'}}, got "
                    f"{consumer_mode!r}."
                )

        checkpoint = inputs.get("checkpoint") or {}
        if isinstance(checkpoint, dict):
            load_mode = str(checkpoint.get("load_mode", "scratch"))
            ckpt = str(checkpoint.get("abs_checkpoint_path", "") or "")
            if load_mode != "scratch" and ckpt:
                cfg["resume_checkpoint"] = ckpt
                cfg["load_mode"] = load_mode
            # Forward the full checkpoint payload so ExportNode can read
            # warm-start context off of train_pipe.
            cfg["checkpoint"] = checkpoint

        # ── Staged training pipeline ──
        if str(self.get_parameter("stages_enabled", "false")).strip().lower() in (
            "true", "1", "yes",
        ):
            import json as _json

            raw = self.get_parameter("stages_json", "[]") or "[]"
            try:
                stages_list = _json.loads(raw) if isinstance(raw, str) else raw
            except Exception:
                stages_list = []
            if isinstance(stages_list, list) and stages_list:
                stage_total = sum(
                    int(s.get("iterations", 0)) for s in stages_list
                    if isinstance(s, dict)
                )
                cfg["stage_schedule"] = {
                    "stages": stages_list,
                    "checkpoint_strategy": str(
                        self.get_parameter("stage_checkpoint_strategy", "both")
                    ),
                    "global_max_iterations": stage_total,
                }
                # Stage schedule takes over iteration budget
                cfg["max_iterations"] = str(stage_total)

        return {
            "train_pipe": cfg,
            "training_metrics": {"status": cfg["status"]},
        }

    def get_display_name(self) -> str:
        return "IL Trainer"

    def get_description(self) -> str:
        return "Unified Isaac Lab trainer — pick PPO or AMP_PPO via Training Mode"


# Legacy alias — canvases saved when AMPTrainerNode was a separate node
# type instantiate under the unified ILPPOTrainerNode with
# training_mode pre-set to AMP_PPO. The workspace loader also rewrites
# node_type "amp_ppo_trainer" → "il_ppo_trainer" at load time so edges
# and validators find the merged node.
class AMPTrainerNode(ILPPOTrainerNode):
    """Deprecated: retained only as a legacy instantiation shim."""

    def __init__(self, node_id: str):
        super().__init__(node_id)
        # Force AMP_PPO so canvases referencing this class still work
        # while we phase it out of the palette.
        self.parameters["training_mode"] = "AMP_PPO"
