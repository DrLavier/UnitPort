# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""SkillManifest — central contract between Mission layer and Behavior layer.

Every behaviour bundle (RL checkpoint, VLA export, HF import, manual script)
MUST have a SkillManifest.  This is the normalisation layer that makes
heterogeneous policies interoperable within a Mission DAG.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SourceType(enum.Enum):
    """How the skill was produced."""
    SB3_INTERNAL = "sb3_internal"
    HF_IMPORT = "hf_import"
    ONNX_EXTERNAL = "onnx_external"
    VLA_EXPORT = "vla_export"
    MANUAL_SCRIPT = "manual_script"
    ISAAC_LAB = "isaac_lab"


class ActionSpaceType(enum.Enum):
    """Physical interpretation of the action vector."""
    JOINT_TORQUE = "joint_torque"
    JOINT_POSITION = "joint_position"
    CARTESIAN_DELTA = "cartesian_delta"
    END_EFFECTOR_6D = "end_effector_6d"
    HYBRID = "hybrid"


class InferenceBackend(enum.Enum):
    """Runtime used to execute the policy model."""
    ONNX = "onnx"
    TORCHSCRIPT = "torchscript"
    PYTORCH_LIVE = "pytorch_live"
    PYTHON_CALLABLE = "python_callable"


# ---------------------------------------------------------------------------
# Transition sub-contracts
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Precondition:
    """State the robot MUST be in before this skill can begin."""
    posture: str = "any"                    # standing | sitting | prone | any | custom
    posture_tolerance_rad: float = 0.3      # ~17 deg default
    velocity_max_mps: float = 1.0           # max linear velocity at entry
    custom_check: Optional[str] = None      # python callable path for complex checks


@dataclass(frozen=True)
class Postcondition:
    """State the robot is guaranteed to be in after this skill finishes."""
    posture: str = "standing"
    posture_tolerance_rad: float = 0.3
    velocity_max_mps: float = 0.5
    custom_check: Optional[str] = None


# ---------------------------------------------------------------------------
# SkillManifest
# ---------------------------------------------------------------------------
#
# RETIRED (2026-07): the ``command_interface`` sub-contract (CommandField /
# CommandInterface, "v2 reactive execution"). It duplicated what
# ``deploy_contract.commands`` now carries as the v1 CommandContract
# (channels with name/kind/range/default/obs encodings, single emitter in
# application/training/command_schema.py). Its only reader — the opt-in
# ``review_session._resolve_review_command`` helper — reads the contract
# defaults instead; old bundles that still carry the manifest key load
# through that helper's §8(c) legacy branch.

@dataclass(frozen=True)
class SkillManifest:
    """Immutable contract describing a single behaviour skill.

    Fields are grouped by sub-contract:
        identity → action → observation → embodiment → transition → inference → metadata
    """

    # --- identity ---
    skill_id: str                                       # unique, e.g. "unitree_go2_trot_v2"
    skill_name: str                                     # display name
    version: str = "1.0.0"                              # semver
    source_type: SourceType = SourceType.SB3_INTERNAL
    execution_mode: str = "sequential"                  # "sequential" | "reactive"

    # --- action contract ---
    action_space_type: ActionSpaceType = ActionSpaceType.JOINT_POSITION
    action_dim: int = 12
    control_frequency_hz: float = 50.0
    action_range: Optional[Dict[str, List[float]]] = None   # {joint: [min, max]} or None

    # --- observation contract ---
    required_sensors: List[str] = field(default_factory=lambda: ["imu", "joint_encoder"])
    observation_space_keys: List[str] = field(default_factory=list)
    observation_dim: int = 45

    # --- embodiment contract ---
    target_robot_family: str = "quadruped"              # quadruped | biped | manipulator | wheeled
    target_robot_models: List[str] = field(default_factory=list)   # empty = universal
    joint_mapping: Optional[Dict[str, str]] = None      # {policy_joint: robot_joint} or None

    # --- transition contract ---
    precondition: Precondition = field(default_factory=Precondition)
    postcondition: Postcondition = field(default_factory=Postcondition)

    # --- inference contract ---
    inference_backend: InferenceBackend = InferenceBackend.ONNX
    model_path: str = "policy.onnx"
    normalize_obs: bool = False
    normalizer_path: Optional[str] = None

    # --- deploy contract (sim2sim fidelity block) ---
    # Populated by manifest_parser.parse_isaac_lab_env_yaml from env.yaml +
    # spec; serialised verbatim into manifest.yaml::deploy_contract by
    # bundle_finalizer. Runtime ObsBuilder._build_with_contract takes this
    # path when present — per-term scale/clip/history are restored byte-for-
    # byte from training. Typed as plain dict to avoid a cross-layer import
    # of application.service.runtime.policy.deploy_contract from training/.
    deploy_contract: Optional[Dict[str, Any]] = None

    # --- metadata ---
    training_source: Optional[str] = None               # "sb3_ppo", "isaac_lab_rsl_rl", etc.
    description: str = ""
    tags: List[str] = field(default_factory=list)
