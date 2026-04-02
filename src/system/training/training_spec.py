#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase B — TrainingJobSpec and graph-to-spec compiler.

TrainingJobSpec
    Typed dataclass representing one complete, runnable training experiment.
    Built from a ``training_canvas_v1`` graph snapshot.

TrainingSpecCompiler
    Reads a serialized graph, indexes nodes by type, resolves connections,
    validates required topology, and assembles a TrainingJobSpec.

    Required topology (from NODE設計.json):
        RobotMJCF → PhysicsConfig → EnvAssembler
        TaskConfig → EnvAssembler
        ObsActionConfig → EnvAssembler
        EnvAssembler → Train
        AlgorithmConfig → Train
        Train → Export

    Optional:
        DomainRandConfig → EnvAssembler
        EvalConfig → Train (or Export)

    If ExportNode is absent the spec is still compilable.
    Required missing nodes/connections raise TrainingSpecError with a clear message.

No Qt.  No real training.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, fields as dataclass_fields
from typing import Any, Dict, List, Optional, Tuple

from src.system.training.training_config import (
    AlgorithmConfig,
    DomainRandConfig,
    EnvConfig,
    EvalConfig,
    ExportConfig,
    GatedRewardConfig,
    GatedStage,
    ImitationLearningConfig,
    InitPoseConfig,
    ObsActionConfig,
    PhysicsConfig,
    ReferenceMotionConfig,
    RewardConfig,
    RobotSpec,
    SceneConfig,
    TaskConfig,
    TerminationConfig,
    VisCheckConfig,
)


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------

class TrainingSpecError(ValueError):
    """Raised when the graph cannot be compiled into a valid TrainingJobSpec."""


# ---------------------------------------------------------------------------
# TrainingJobSpec
# ---------------------------------------------------------------------------

@dataclass
class TrainingJobSpec:
    """
    Fully typed specification for one training experiment.

    All config sub-objects are populated from parsed node parameters.
    Optional fields (domain_rand_config, eval_config, export_config) may
    be None when the corresponding nodes are absent from the graph.
    """

    policy_id: str
    """Base policy_id this workspace is bound to."""

    experiment_id: str = ""
    """Experiment name/id (set by caller; empty = unbound)."""

    schema_version: str = "training_canvas_v1"
    """Canvas schema version the spec was compiled from."""

    robot_spec: RobotSpec = field(default_factory=RobotSpec)
    physics_config: PhysicsConfig = field(default_factory=PhysicsConfig)
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    termination_config: TerminationConfig = field(default_factory=TerminationConfig)
    reward_terms: Dict[str, float] = field(default_factory=dict)
    termination_conditions: Dict[str, float] = field(default_factory=dict)
    task_config: TaskConfig = field(default_factory=TaskConfig)
    domain_rand_config: Optional[DomainRandConfig] = None
    obs_action_config: ObsActionConfig = field(default_factory=ObsActionConfig)
    env_config: EnvConfig = field(default_factory=EnvConfig)
    algorithm_config: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    eval_config: Optional[EvalConfig] = None
    export_config: Optional[ExportConfig] = None
    vis_check_config: Optional[VisCheckConfig] = None
    scene_config: Optional[SceneConfig] = None
    reference_motion_config: Optional[ReferenceMotionConfig] = None
    init_pose_config: Optional[InitPoseConfig] = None
    gated_reward_config: Optional[GatedRewardConfig] = None
    imitation_config: Optional[ImitationLearningConfig] = None

    base_asset_path: str = ""
    """Absolute path to SB3 .zip used for resume/warm-start; empty = scratch."""

    resume_mode: str = "scratch"
    """Top-level resume mode: 'scratch' | 'resume_sb3' | 'warm_start_actor'."""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingJobSpec":
        payload = data if isinstance(data, dict) else {}
        spec = cls(
            policy_id=str(payload.get("policy_id", "")),
            experiment_id=str(payload.get("experiment_id", "")),
            schema_version=str(
                payload.get(
                    "schema_version",
                    payload.get("schema", "training_canvas_v1"),
                )
            ),
            base_asset_path=str(payload.get("base_asset_path", "")),
            resume_mode=str(payload.get("resume_mode", "scratch") or "scratch"),
        )

        spec.robot_spec = _build_config(RobotSpec, payload.get("robot_spec"))
        spec.physics_config = _build_config(
            PhysicsConfig, payload.get("physics_config")
        )
        spec.reward_config = _build_config(RewardConfig, payload.get("reward_config"))
        spec.termination_config = _build_config(
            TerminationConfig, payload.get("termination_config")
        )
        spec.task_config = _build_config(TaskConfig, payload.get("task_config"))
        spec.obs_action_config = _build_config(
            ObsActionConfig, payload.get("obs_action_config")
        )
        spec.env_config = _build_config(EnvConfig, payload.get("env_config"))
        spec.algorithm_config = _build_config(
            AlgorithmConfig, payload.get("algorithm_config")
        )

        if isinstance(payload.get("domain_rand_config"), dict):
            spec.domain_rand_config = _build_config(
                DomainRandConfig, payload.get("domain_rand_config")
            )
        if isinstance(payload.get("eval_config"), dict):
            spec.eval_config = _build_config(EvalConfig, payload.get("eval_config"))
        if isinstance(payload.get("export_config"), dict):
            spec.export_config = _build_config(
                ExportConfig, payload.get("export_config")
            )
        if isinstance(payload.get("vis_check_config"), dict):
            spec.vis_check_config = _build_config(
                VisCheckConfig, payload.get("vis_check_config")
            )
        if isinstance(payload.get("scene_config"), dict):
            spec.scene_config = _build_config(
                SceneConfig, payload.get("scene_config")
            )
        if isinstance(payload.get("reference_motion_config"), dict):
            spec.reference_motion_config = _build_config(
                ReferenceMotionConfig, payload.get("reference_motion_config")
            )
        if isinstance(payload.get("gated_reward_config"), dict):
            spec.gated_reward_config = GatedRewardConfig.from_dict(
                payload.get("gated_reward_config")
            )
        if isinstance(payload.get("imitation_config"), dict):
            spec.imitation_config = _build_config(
                ImitationLearningConfig, payload.get("imitation_config")
            )

        reward_terms, termination_conditions = resolve_effective_task_terms(
            spec,
            top_level_reward_terms=payload.get("reward_terms"),
            top_level_termination_conditions=payload.get(
                "termination_conditions"
            ),
        )
        spec.reward_terms = reward_terms
        spec.termination_conditions = termination_conditions
        spec.reward_config.reward_terms = dict(reward_terms)
        spec.termination_config.termination_conditions = dict(
            termination_conditions
        )
        spec.task_config.reward_terms = dict(reward_terms)
        spec.task_config.termination_conditions = dict(termination_conditions)
        return spec

    def derive_skill_manifest_fields(self) -> Dict[str, Any]:
        """Derive SkillManifest v2 fields from the training configuration.

        Returns a dict suitable for embedding as the ``skill`` section in
        manifest.yaml v2.  Called by ``bundle_exporter.build_manifest()``.
        """
        from src.system.training.robot_family import resolve_robot_family
        from src.system.training.task_templates import TASK_TEMPLATE_FAMILY

        robot_type = self.robot_spec.robot_type.lower()
        robot_family = resolve_robot_family(robot_type)

        # Action space type
        action_type = (
            getattr(self.obs_action_config, "action_type", None)
            or getattr(self.physics_config, "action_type", None)
            or "joint_position"
        )

        # Obs components / keys
        obs_keys: List[str] = list(
            getattr(self.obs_action_config, "obs_components", []) or []
        )

        # Control frequency
        try:
            from src.system.training.training_config import resolve_control_timing
            timing = resolve_control_timing(
                getattr(self.physics_config, "sim_dt", 0.002),
                getattr(self.physics_config, "control_dt", 0.02),
            )
            freq = float(timing["control_frequency_hz"])
        except Exception:
            freq = 50.0

        # Pre/postcondition from task template
        template = TASK_TEMPLATE_FAMILY.get(robot_family, {})
        pre_posture = template.get("precondition_posture", "any")
        post_posture = template.get("postcondition_posture", "standing")

        # Required sensors — derive from obs components
        sensors = ["imu", "joint_encoder"]
        if any("camera" in c or "rgb" in c for c in obs_keys):
            sensors.append("rgb_camera")
        if any("depth" in c for c in obs_keys):
            sensors.append("depth_camera")
        if any("force" in c or "torque" in c for c in obs_keys):
            sensors.append("force_torque")

        # Training source
        algo = getattr(self.algorithm_config, "algorithm", "PPO")
        training_source = f"sb3_{algo.lower()}" if algo else None

        # Robot models
        brand_map = {
            "go2": "unitree", "go1": "unitree", "b1": "unitree",
            "b2": "unitree", "h1": "unitree", "g1": "unitree",
            "spot": "spot", "cyberdog": "cyberdog",
        }
        brand = brand_map.get(robot_type, "unitree")
        target_models = [f"{brand}_{robot_type}"] if robot_type else []

        return {
            "skill_id": "",             # filled by exporter (= policy_id_out)
            "skill_name": "",           # filled by exporter
            "version": "1.0.0",
            "source_type": "sb3_internal",
            "action_space_type": action_type,
            "action_dim": 0,            # filled by exporter (from env)
            "control_frequency_hz": freq,
            "required_sensors": sensors,
            "observation_space_keys": obs_keys,
            "observation_dim": 0,       # filled by exporter (from env)
            "target_robot_family": robot_family,
            "target_robot_models": target_models,
            "precondition": {
                "posture": pre_posture,
                "posture_tolerance_rad": 0.3,
                "velocity_max_mps": 2.0,
            },
            "postcondition": {
                "posture": post_posture,
                "posture_tolerance_rad": 0.3,
                "velocity_max_mps": 0.5,
            },
            "inference_backend": "onnx",
            "normalize_obs": False,
            "training_source": training_source,
            "description": "",
            "tags": [],
        }

    def to_dict(self) -> dict:
        d: dict = {
            "policy_id": self.policy_id,
            "experiment_id": self.experiment_id,
            "schema_version": self.schema_version,
            "robot_spec": self.robot_spec.to_dict(),
            "physics_config": self.physics_config.to_dict(),
            "reward_config": self.reward_config.to_dict(),
            "termination_config": self.termination_config.to_dict(),
            "reward_terms": dict(self.reward_terms),
            "termination_conditions": dict(self.termination_conditions),
            "task_config": self.task_config.to_dict(),
            "obs_action_config": self.obs_action_config.to_dict(),
            "env_config": self.env_config.to_dict(),
            "algorithm_config": self.algorithm_config.to_dict(),
        }
        if self.domain_rand_config is not None:
            d["domain_rand_config"] = self.domain_rand_config.to_dict()
        if self.eval_config is not None:
            d["eval_config"] = self.eval_config.to_dict()
        if self.export_config is not None:
            d["export_config"] = self.export_config.to_dict()
        if self.vis_check_config is not None:
            d["vis_check_config"] = self.vis_check_config.to_dict()
        if self.scene_config is not None:
            d["scene_config"] = self.scene_config.to_dict()
        if self.reference_motion_config is not None:
            d["reference_motion_config"] = self.reference_motion_config.to_dict()
        if self.gated_reward_config is not None:
            d["gated_reward_config"] = self.gated_reward_config.to_dict()
        if self.imitation_config is not None:
            d["imitation_config"] = self.imitation_config.to_dict()
        if self.base_asset_path:
            d["base_asset_path"] = self.base_asset_path
        if self.resume_mode != "scratch":
            d["resume_mode"] = self.resume_mode
        return d


def _build_config(config_cls, raw: Any):
    if not isinstance(raw, dict):
        return config_cls()
    field_names = {f.name for f in dataclass_fields(config_cls)}
    filtered = {k: v for k, v in raw.items() if k in field_names}
    try:
        return config_cls(**filtered)
    except Exception:
        return config_cls()


def _normalize_term_mapping(raw: Any) -> Dict[str, float]:
    if not isinstance(raw, dict) or not raw:
        return {}
    normalized: Dict[str, float] = {}
    for key, value in raw.items():
        try:
            normalized[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return normalized


def resolve_effective_task_terms(
    spec: TrainingJobSpec,
    *,
    top_level_reward_terms: Any = None,
    top_level_termination_conditions: Any = None,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    reward_terms = (
        _normalize_term_mapping(top_level_reward_terms)
        or _normalize_term_mapping(getattr(spec, "reward_terms", None))
        or _normalize_term_mapping(getattr(spec.reward_config, "reward_terms", None))
        or _normalize_term_mapping(getattr(spec.task_config, "reward_terms", None))
    )
    termination_conditions = (
        _normalize_term_mapping(top_level_termination_conditions)
        or _normalize_term_mapping(
            getattr(spec, "termination_conditions", None)
        )
        or _normalize_term_mapping(
            getattr(spec.termination_config, "termination_conditions", None)
        )
        or _normalize_term_mapping(
            getattr(spec.task_config, "termination_conditions", None)
        )
    )
    return reward_terms, termination_conditions


# ---------------------------------------------------------------------------
# Pre-flight validation & auto-fix
# ---------------------------------------------------------------------------

class PreflightWarning:
    """A non-fatal issue detected during pre-flight, with an applied auto-fix."""
    def __init__(self, field: str, message: str, fixed: bool = False):
        self.field = field
        self.message = message
        self.fixed = fixed

    def __repr__(self) -> str:
        tag = "AUTO-FIXED" if self.fixed else "WARNING"
        return f"[{tag}] {self.field}: {self.message}"


def preflight_check(spec: TrainingJobSpec) -> List["PreflightWarning"]:
    """Validate semantic consistency of a compiled spec before training.

    Called after ``TrainingSpecCompiler.compile()`` and before
    ``start_training()``.  The Robot/MJCF node is authoritative — when
    other nodes conflict with it, this function auto-fixes them in-place
    and returns warnings explaining what changed.

    Raises ``TrainingSpecError`` for fatal issues that cannot be auto-fixed.
    Returns a (possibly empty) list of ``PreflightWarning`` for non-fatal
    issues that were auto-corrected.
    """
    import os
    warnings: List[PreflightWarning] = []
    robot = spec.robot_spec
    robot_type = str(getattr(robot, "robot_type", "") or "").strip().lower()

    if not robot_type:
        raise TrainingSpecError("Robot/MJCF node has no robot_type set.")

    # ── 1. Scene config must match robot_type ─────────────────────────
    scene_cfg = spec.scene_config
    if scene_cfg is not None:
        scene_type = str(getattr(scene_cfg, "scene_type", "flat") or "flat")
        custom_path = str(getattr(scene_cfg, "custom_scene_path", "") or "").strip()

        if scene_type == "custom" and custom_path:
            # Check if the custom scene belongs to a different robot
            custom_lower = custom_path.replace("\\", "/").lower()
            _ROBOT_KEYWORDS = {
                "go2": ["go2", "unitree_go2"],
                "a1": ["a1", "unitree_a1"],
                "h1": ["h1", "unitree_h1"],
                "h1_2": ["h1_2", "h1v2"],
                "g1": ["g1", "unitree_g1"],
                "spot": ["spot", "bd_spot"],
            }
            matched_robot = None
            for rtype, keywords in _ROBOT_KEYWORDS.items():
                if any(kw in custom_lower for kw in keywords):
                    matched_robot = rtype
                    break

            if matched_robot and matched_robot != robot_type:
                # Auto-fix: reset to 'flat' so the correct robot scene is auto-detected
                scene_cfg.scene_type = "flat"
                scene_cfg.custom_scene_path = ""
                warnings.append(PreflightWarning(
                    "scene_config",
                    f"Custom scene pointed to '{matched_robot}' model but robot_type "
                    f"is '{robot_type}'. Reset to 'flat' (auto-detect).",
                    fixed=True,
                ))

    # ── 2. Resolve the actual MJCF and verify it matches robot_type ──
    try:
        from src.system.training.unitree_gym_env import _resolve_scene_xml
        resolved_xml = _resolve_scene_xml(
            spec.scene_config,
            getattr(robot, "mjcf_path", "") or "",
            robot_type,
        )
        if resolved_xml:
            resolved_lower = resolved_xml.replace("\\", "/").lower()
            # Check the resolved XML actually contains the right robot
            _ROBOT_SCENE_MARKERS = {
                "go2": "go2", "a1": "a1", "h1": "h1",
                "h1_2": "h1", "g1": "g1", "spot": "spot",
            }
            expected_marker = _ROBOT_SCENE_MARKERS.get(robot_type, robot_type)
            if expected_marker not in resolved_lower:
                raise TrainingSpecError(
                    f"Scene XML '{resolved_xml}' does not match robot_type "
                    f"'{robot_type}'. Check Robot/MJCF and Scene Config nodes."
                )
    except TrainingSpecError:
        raise
    except Exception:
        pass  # resolution failure will be caught later by the trainer

    # ── 3. action_type consistency between Physics and Obs&Action ─────
    physics_at = str(getattr(spec.physics_config, "action_type", "") or "").strip().lower()
    obs_at = str(getattr(spec.obs_action_config, "action_type", "") or "").strip().lower()
    if physics_at and obs_at and physics_at != obs_at:
        # Physics node is authoritative for action_type
        spec.obs_action_config.action_type = physics_at
        warnings.append(PreflightWarning(
            "action_type",
            f"Obs&Action action_type='{obs_at}' conflicted with "
            f"Physics action_type='{physics_at}'. Aligned to Physics.",
            fixed=True,
        ))

    # ── 4. obs_dim sanity check ───────────────────────────────────────
    try:
        from src.system.training.sb3_trainer import get_obs_action_dims
        obs_dim, action_dim = get_obs_action_dims(spec)
        if obs_dim <= 0:
            raise TrainingSpecError(
                f"Computed obs_dim={obs_dim} is invalid. "
                "Check Obs&Action node's obs_components."
            )
        if action_dim <= 0:
            raise TrainingSpecError(
                f"Computed action_dim={action_dim} is invalid."
            )
    except TrainingSpecError:
        raise
    except Exception:
        pass

    # ── 5. Reference motion: warn if tracking reward enabled but
    #       no reference motion node is connected ──────────────────────
    reward_terms = spec.reward_terms or {}
    has_ref_reward = (
        reward_terms.get("reference_tracking", 0) != 0
        or reward_terms.get("joint_pose_tracking", 0) != 0
    )
    if has_ref_reward and spec.reference_motion_config is None:
        warnings.append(PreflightWarning(
            "reference_motion",
            "reference_tracking or joint_pose_tracking reward is enabled "
            "but no Reference Motion node is connected. The tracking reward "
            "will fall back to the default standing pose.",
            fixed=False,
        ))

    return warnings


# ---------------------------------------------------------------------------
# Graph-to-spec compiler
# ---------------------------------------------------------------------------

# Node type identifiers (match TrainingGraphScene._TRAINING_NODE_MAP values)
_NT_ROBOT       = "robot_mjcf"
_NT_PHYSICS     = "physics_config"
_NT_REWARDS     = "rewards"
_NT_TERMS       = "terminations"
_NT_TASK        = "task_config"
_NT_DOMAIN_RAND = "domain_rand"
_NT_OBS_ACTION  = "obs_action_config"
_NT_ENV_ASM     = "env_assembler"
_NT_ALGO        = "algo_config"
_NT_TRAIN       = "train"
_NT_EVAL        = "eval_config"
_NT_EXPORT      = "export"
_NT_VIS_CHECK   = "vis_check"
_NT_BASE_ASSET  = "base_asset"
_NT_SCENE            = "scene_config"
_NT_REF_MOTION       = "reference_motion"
_NT_INIT_POSE        = "init_pose"
_NT_GATED_REWARD     = "multigated_reward"   # drop-in replacement for _NT_REWARDS

# Required node types; presence is validated before compilation.
# _NT_REWARDS and _NT_GATED_REWARD are mutually exclusive alternatives —
# the graph must contain exactly one of them.
_REQUIRED_NODE_TYPES = {
    _NT_ROBOT, _NT_PHYSICS, _NT_TERMS, _NT_TASK, _NT_OBS_ACTION,
    _NT_ENV_ASM, _NT_ALGO, _NT_TRAIN,
    # rewards: validated separately (either "rewards" or "multigated_reward")
}


class TrainingSpecCompiler:
    """
    Compiles a ``training_canvas_v1`` graph snapshot into a TrainingJobSpec.

    Usage::

        compiler = TrainingSpecCompiler()
        spec = compiler.compile(
            graph=scene.serialize_training_graph(),
            policy_id="go2_v1",
            experiment_id="exp_01",
        )

    Raises ``TrainingSpecError`` on invalid graphs.
    """

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def compile(
        self,
        graph: Dict[str, Any],
        policy_id: str = "",
        experiment_id: str = "",
    ) -> TrainingJobSpec:
        """
        Parse *graph* and return a complete TrainingJobSpec.

        Parameters
        ----------
        graph:
            Output of ``TrainingGraphScene.serialize_training_graph()``.
        policy_id:
            Base policy_id for the workspace; carried into the spec.
        experiment_id:
            Experiment name; carried into the spec.

        Raises
        ------
        TrainingSpecError
            If a required node type is missing or if the graph does not
            satisfy the minimum required connection topology.
        """
        schema = graph.get("schema", "")
        if schema != "training_canvas_v1":
            raise TrainingSpecError(
                f"Unsupported graph schema {schema!r}; expected 'training_canvas_v1'"
            )

        nodes_by_type = self._index_by_type(graph.get("nodes", []))
        nodes_by_id   = self._index_by_id(graph.get("nodes", []))
        self._validate_required(nodes_by_type)

        edges = graph.get("edges", [])

        spec = TrainingJobSpec(
            policy_id=policy_id,
            experiment_id=experiment_id,
            schema_version=schema,
        )

        # Required nodes
        spec.robot_spec = RobotSpec.from_node_dict(
            nodes_by_type[_NT_ROBOT]["parameters"]
        )
        spec.physics_config = PhysicsConfig.from_node_dict(
            nodes_by_type[_NT_PHYSICS]["parameters"]
        )
        # Rewards — static RewardsNode OR MultiGatedRewardNode (router).
        # When multigated: stage terms come from the RewardsNodes connected to
        # its stage_0/stage_1 input ports; the gated schedule is compiled from
        # those connected nodes + the gate logic params on the multigated node.
        if _NT_GATED_REWARD in nodes_by_type:
            gated_node = nodes_by_type[_NT_GATED_REWARD]
            spec.gated_reward_config = self._compile_gated_reward_config(
                nodes_by_id, edges, gated_node
            )
            # Stage 0 terms become the static baseline (overridden at runtime)
            stage0_terms = (
                spec.gated_reward_config.stages[0].reward_terms
                if spec.gated_reward_config and spec.gated_reward_config.stages
                else {}
            )
            rc = RewardConfig()
            rc.reward_terms = dict(stage0_terms)
            spec.reward_config = rc
        else:
            spec.reward_config = RewardConfig.from_node_dict(
                nodes_by_type[_NT_REWARDS]["parameters"]
            )

        spec.termination_config = TerminationConfig.from_node_dict(
            nodes_by_type[_NT_TERMS]["parameters"]
        )
        spec.reward_terms = dict(spec.reward_config.reward_terms)
        spec.termination_conditions = dict(spec.termination_config.termination_conditions)
        spec.task_config = TaskConfig.from_node_dict(
            nodes_by_type[_NT_TASK]["parameters"]
        )
        spec.task_config.reward_terms = dict(spec.reward_terms)
        spec.task_config.termination_conditions = dict(spec.termination_conditions)
        spec.obs_action_config = ObsActionConfig.from_node_dict(
            nodes_by_type[_NT_OBS_ACTION]["parameters"]
        )
        spec.env_config = EnvConfig.from_node_dict(
            nodes_by_type[_NT_ENV_ASM]["parameters"]
        )
        spec.algorithm_config = AlgorithmConfig.from_node_dict(
            nodes_by_type[_NT_ALGO]["parameters"]
        )

        # When MultiGatedRewardNode.total_steps → AlgorithmConfigNode.total_steps
        # edge is present, override total_timesteps with the computed value so
        # the trainer always uses the AUTO-calculated budget.
        if _NT_GATED_REWARD in nodes_by_type and self._has_edge(
            nodes_by_type, edges,
            _NT_GATED_REWARD, "total_steps",
            _NT_ALGO, "total_steps",
        ):
            try:
                gp = nodes_by_type[_NT_GATED_REWARD].get("parameters", {})
                max_s0 = int(float(gp.get("max_step_stage0", "0") or 0))
                ratio  = float(gp.get("stage1_ratio", "1.5") or 1.5)
                if max_s0 > 0:
                    recommended = int(max_s0 * (1.0 + max(0.0, ratio)))
                    spec.algorithm_config.total_timesteps = recommended
            except (TypeError, ValueError, AttributeError):
                pass

        # Optional nodes
        if _NT_DOMAIN_RAND in nodes_by_type:
            spec.domain_rand_config = DomainRandConfig.from_node_dict(
                nodes_by_type[_NT_DOMAIN_RAND]["parameters"]
            )
        if _NT_EVAL in nodes_by_type:
            spec.eval_config = EvalConfig.from_node_dict(
                nodes_by_type[_NT_EVAL]["parameters"]
            )
        if _NT_EXPORT in nodes_by_type:
            spec.export_config = ExportConfig.from_node_dict(
                nodes_by_type[_NT_EXPORT]["parameters"]
            )
            bundle_name = (spec.export_config.bundle_name or "").strip()
            if bundle_name:
                spec.algorithm_config.policy_id_out = bundle_name
            else:
                spec.export_config.bundle_name = spec.algorithm_config.policy_id_out
        # BaseAssetNode — optional; when present, override resume_mode on both
        # the top-level spec and algo_config (SB3Trainer reads spec.resume_mode)
        if _NT_BASE_ASSET in nodes_by_type:
            ba_params = nodes_by_type[_NT_BASE_ASSET].get("parameters", {})
            ba_load_mode = str(ba_params.get("load_mode", "scratch"))
            if ba_load_mode != "scratch":
                spec.resume_mode = ba_load_mode
                spec.algorithm_config.resume_mode = ba_load_mode
            # base_asset_path resolution happens in the caller (3-C)

        if (
            _NT_VIS_CHECK in nodes_by_type
            and self._has_edge(
                nodes_by_type,
                edges,
                _NT_TRAIN,
                "vis_check",
                _NT_VIS_CHECK,
                "vis_check",
            )
        ):
            spec.vis_check_config = VisCheckConfig.from_node_dict(
                nodes_by_type[_NT_VIS_CHECK]["parameters"]
            )

        # SceneConfigNode — optional; present whenever SceneConfig is in graph
        if _NT_SCENE in nodes_by_type:
            spec.scene_config = SceneConfig.from_node_dict(
                nodes_by_type[_NT_SCENE]["parameters"]
            )

        # ReferenceMotionNode — only active when wired to EnvAssembler.
        # A node sitting on canvas but not connected has no effect (same as
        # vis_check which also gates on edge presence).
        if TrainingSpecCompiler._has_edge(
            nodes_by_type,
            edges,
            _NT_REF_MOTION,
            "reference_motion_config",
            _NT_ENV_ASM,
            "reference_motion_config",
        ):
            spec.reference_motion_config = ReferenceMotionConfig.from_node_dict(
                nodes_by_type[_NT_REF_MOTION]["parameters"]
            )

        # ── Imitation learning: read BC config from ReferenceMotionNode ─────
        # BC parameters live on the ReferenceMotionNode itself (not a separate
        # node), so the user configures everything in one place.  When the
        # reference motion node is connected, the compiler:
        #   1. Reads BC params from the same node's parameters dict
        #   2. Injects reference obs components into obs_components list
        # This follows DeepMimic / AMP standard: the policy must see the
        # reference target in its observation, and should be pre-trained
        # with behavioral cloning before RL fine-tuning.
        if spec.reference_motion_config is not None:
            # BC params are co-located on the ReferenceMotionNode
            ref_params = nodes_by_type.get(_NT_REF_MOTION, {}).get("parameters", {})
            spec.imitation_config = ImitationLearningConfig.from_node_dict(ref_params)

            # Auto-inject reference observation components (DeepMimic standard)
            imit_cfg = spec.imitation_config
            if imit_cfg is not None and getattr(imit_cfg, "auto_inject_ref_obs", True):
                obs_comps = list(spec.obs_action_config.obs_components)
                existing = set(obs_comps)
                injected = []
                if not existing.intersection({"reference_joint_positions", "ref_joint_pos"}):
                    obs_comps.append("reference_joint_positions")
                    injected.append("reference_joint_positions")
                if not existing.intersection({"reference_joint_velocities", "ref_joint_vel"}):
                    obs_comps.append("reference_joint_velocities")
                    injected.append("reference_joint_velocities")
                if not existing.intersection({"phase_sin_cos", "phase"}):
                    obs_comps.append("phase_sin_cos")
                    injected.append("phase_sin_cos")
                if injected:
                    spec.obs_action_config.obs_components = obs_comps

        # InitPoseNode — optional; activates when wired to EnvAssembler.
        if TrainingSpecCompiler._has_edge(
            nodes_by_type,
            edges,
            _NT_INIT_POSE,
            "init_pose_config",
            _NT_ENV_ASM,
            "init_pose_config",
        ):
            spec.init_pose_config = InitPoseConfig.from_node_dict(
                nodes_by_type[_NT_INIT_POSE]["parameters"]
            )

        # Validate minimum required connection topology
        self._validate_topology(nodes_by_type, edges)

        return spec

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _index_by_type(
        nodes: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Build a mapping from node_type → first matching node dict.

        If multiple nodes of the same type are present, only the first is
        used (the canvas is expected to have exactly one of each required
        type).
        """
        result: Dict[str, Dict[str, Any]] = {}
        for node in nodes:
            ntype = node.get("node_type", "")
            if ntype and ntype not in result:
                result[ntype] = node
        return result

    @staticmethod
    def _index_by_id(nodes: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Build a mapping from node id → node dict (all nodes, not just first-of-type)."""
        return {node["id"]: node for node in nodes if "id" in node}

    @staticmethod
    def _compile_gated_reward_config(
        nodes_by_id: Dict[str, Dict[str, Any]],
        edges: List[Dict[str, Any]],
        gated_node: Dict[str, Any],
    ) -> "GatedRewardConfig":
        """
        Build a GatedRewardConfig by reading stage reward terms from the
        RewardsNodes connected to each stage input port of the multigated node.

        Stage terms are NOT embedded in the multigated node's parameters — they
        come from the connected nodes (stage_0, stage_1, …).  Gate timing and
        schedule settings are read from the multigated node's parameters.
        """
        gated_id = gated_node.get("id", "")
        params   = gated_node.get("parameters", {})

        # Index edges by dst_slot so we can look up which node feeds each stage
        stage_src: Dict[str, Dict[str, Any]] = {}   # slot_name → source node
        for e in edges:
            if e.get("dst_id") != gated_id:
                continue
            slot = str(e.get("dst_slot", ""))
            if slot.startswith("stage_"):
                src_node = nodes_by_id.get(str(e.get("src_id", "")))
                if src_node is not None:
                    stage_src[slot] = src_node

        def _safe_int(v, default: int) -> int:
            try:
                return int(v)
            except (TypeError, ValueError):
                return default

        def _safe_float(v, default: float) -> float:
            try:
                return float(v)
            except (TypeError, ValueError):
                return default

        ratio          = _safe_float(params.get("reward_threshold_ratio", "0.75"), 0.75)
        min_s1         = _safe_int(params.get("min_step_stage1",  "0"),     0)
        max_s0         = _safe_int(params.get("max_step_stage0",  "0"),     0)
        blend_steps    = _safe_int(params.get("blend_steps",      "3000"),  3000)
        stage_beh      = str(params.get("stage_behavior",   "replace") or "replace")
        ep_window      = _safe_int(params.get("ep_reward_window", "20"),    20)
        min_ep_window  = max(1, _safe_int(params.get("min_ep_window",  "10"),   10))
        plateau_window = max(2, _safe_int(params.get("plateau_window", "10"),   10))
        plateau_eps    = max(0.0, _safe_float(params.get("plateau_eps", "0.005"), 0.005))

        stages: List["GatedStage"] = []
        # Build stages in slot order: stage_0, stage_1, …
        for slot_idx in sorted(
            int(s.split("_", 1)[1]) for s in stage_src if s.split("_", 1)[1].isdigit()
        ):
            slot_name = f"stage_{slot_idx}"
            src_node  = stage_src[slot_name]
            reward_config = RewardConfig.from_node_dict(src_node.get("parameters", {}))
            terms = dict(reward_config.reward_terms)

            # Stage 0: active from start, hard-timeout at max_step_stage0
            # Stage 1+: gate opens at min_step_stage1
            if slot_idx == 0:
                stage = GatedStage(
                    reward_terms=terms,
                    min_step=0,
                    max_step=max_s0,
                    reward_threshold_ratio=ratio,
                )
            else:
                stage = GatedStage(
                    reward_terms=terms,
                    min_step=min_s1,
                    max_step=0,
                    reward_threshold_ratio=ratio,
                )
            stages.append(stage)

        return GatedRewardConfig(
            stages=stages,
            blend_steps=blend_steps,
            stage_behavior=stage_beh,
            ep_reward_window=ep_window,
            min_ep_window=min_ep_window,
            plateau_window=plateau_window,
            plateau_eps=plateau_eps,
        )

    @staticmethod
    def _has_edge(
        nodes_by_type: Dict[str, Dict[str, Any]],
        edges: List[Dict[str, Any]],
        src_type: str,
        src_slot: str,
        dst_type: str,
        dst_slot: str,
    ) -> bool:
        """Return True when the graph contains the typed edge."""
        src = nodes_by_type.get(src_type)
        dst = nodes_by_type.get(dst_type)
        if src is None or dst is None:
            return False
        src_id = src["id"]
        dst_id = dst["id"]
        for e in edges:
            if (
                e.get("src_id") == src_id
                and e.get("src_slot") == src_slot
                and e.get("dst_id") == dst_id
                and e.get("dst_slot") == dst_slot
            ):
                return True
        return False

    @staticmethod
    def _validate_required(nodes_by_type: Dict[str, Any]) -> None:
        missing = _REQUIRED_NODE_TYPES - set(nodes_by_type.keys())
        if missing:
            raise TrainingSpecError(
                f"Graph is missing required node types: {sorted(missing)}"
            )
        # Rewards: either static "rewards" OR dynamic "multigated_reward" must exist
        if _NT_REWARDS not in nodes_by_type and _NT_GATED_REWARD not in nodes_by_type:
            raise TrainingSpecError(
                "Graph is missing required node type: 'rewards' or 'multigated_reward'"
            )

    @staticmethod
    def _validate_topology(
        nodes_by_type: Dict[str, Dict[str, Any]],
        edges: List[Dict[str, Any]],
    ) -> None:
        """
        Check that the minimum required connections are present.

        Required connections (src_type.slot → dst_type.slot):
            robot_mjcf.robot_spec          → physics_config.robot_spec
            physics_config.physics_config  → env_assembler.physics_config
            rewards.rewards               → task_config.rewards
            terminations.terminations     → task_config.terminations
            task_config.task_config       → env_assembler.task_config
            obs_action_config.obs_action_config → env_assembler.obs_action_config
            env_assembler.env_config       → train.env_config
            algo_config.algo_config        → train.algo_config
            train.train_result             → export.train_result  (if Export present)
        """
        # Build a quick lookup: (src_id, src_slot, dst_id, dst_slot)
        # Index to sets for fast membership test
        required_edges = [
            (_NT_ROBOT,    "robot_spec",         _NT_PHYSICS,  "robot_spec"),
            (_NT_PHYSICS,  "physics_config",     _NT_ENV_ASM,  "physics_config"),
            (_NT_TERMS,    "terminations",       _NT_TASK,     "terminations"),
            (_NT_TASK,     "task_config",        _NT_ENV_ASM,  "task_config"),
            (_NT_OBS_ACTION, "obs_action_config", _NT_ENV_ASM, "obs_action_config"),
            (_NT_ENV_ASM,  "env_config",          _NT_TRAIN,    "env_config"),
            (_NT_ALGO,     "algo_config",          _NT_TRAIN,    "algo_config"),
        ]
        missing_edges: List[str] = []
        for src_t, src_s, dst_t, dst_s in required_edges:
            if not TrainingSpecCompiler._has_edge(
                nodes_by_type, edges, src_t, src_s, dst_t, dst_s
            ):
                missing_edges.append(f"{src_t}.{src_s} → {dst_t}.{dst_s}")

        # Rewards edge: multigated_reward takes precedence (it is the output node);
        # fall back to the static rewards node when no multigated node is present.
        _rewards_src = _NT_GATED_REWARD if _NT_GATED_REWARD in nodes_by_type else _NT_REWARDS
        if not TrainingSpecCompiler._has_edge(
            nodes_by_type, edges, _rewards_src, "rewards", _NT_TASK, "rewards"
        ):
            missing_edges.append(f"{_rewards_src}.rewards → {_NT_TASK}.rewards")

        # Export connection optional, but must be correct when Export node exists
        if _NT_EXPORT in nodes_by_type:
            if not TrainingSpecCompiler._has_edge(
                nodes_by_type,
                edges,
                _NT_TRAIN,
                "train_result",
                _NT_EXPORT,
                "train_result",
            ):
                missing_edges.append(f"{_NT_TRAIN}.train_result → {_NT_EXPORT}.train_result")

        if missing_edges:
            raise TrainingSpecError(
                "Graph is missing required connections:\n  "
                + "\n  ".join(missing_edges)
            )
