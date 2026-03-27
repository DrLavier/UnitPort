from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------

class ManifestValidationError(ValueError):
    """Raised when a bundle manifest fails validation."""

    def __init__(
        self,
        message: str,
        missing_fields: Optional[List[str]] = None,
        invalid_fields: Optional[Dict[str, str]] = None,
        bundle_path: Optional[Path] = None,
    ):
        super().__init__(message)
        self.missing_fields: List[str] = missing_fields or []
        self.invalid_fields: Dict[str, str] = invalid_fields or {}
        self.bundle_path: Optional[Path] = bundle_path


# ---------------------------------------------------------------------------
# Required field list
# ---------------------------------------------------------------------------

MANIFEST_REQUIRED_FIELDS = (
    "name",
    "version",
    "policy.file",
    "policy.format",
    "observation_space.dim",
    "action_space.dim",
    "action_space.type",
    "runtime.control_frequency_hz",
    "runtime.decimation",
    "robot.brand",
    "robot.num_joints",
    "robot.joint_names",
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _get_nested(d: Dict[str, Any], dotted_key: str) -> Any:
    """Walk *d* using dot-separated segments; raise KeyError if missing."""
    parts = dotted_key.split(".")
    node: Any = d
    for part in parts:
        if not isinstance(node, dict) or part not in node:
            raise KeyError(dotted_key)
        node = node[part]
    return node


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_SUPPORTED_FORMATS = {"onnx", "jit"}   # Phase 1: onnx; Phase 2 adds: jit
_REJECTED_FORMATS = {"pt", "safetensors"}


def validate_manifest(raw: Dict[str, Any], bundle_path: Optional[Path] = None) -> None:
    """Validate *raw* manifest dict.

    Raises ManifestValidationError listing ALL problems found in one call.
    """
    # Step 1: collect missing fields
    missing: List[str] = []
    for key in MANIFEST_REQUIRED_FIELDS:
        try:
            _get_nested(raw, key)
        except KeyError:
            missing.append(key)

    if missing:
        raise ManifestValidationError(
            f"Manifest missing required fields: {missing}",
            missing_fields=missing,
            bundle_path=bundle_path,
        )

    # Step 2+: validate field values; collect all errors
    invalid: Dict[str, str] = {}

    # policy.format
    fmt = _get_nested(raw, "policy.format")
    if fmt in _REJECTED_FORMATS:
        invalid["policy.format"] = (
            f"unsupported format '{fmt}'; Phase 1 only supports 'onnx'"
        )
    elif fmt not in _SUPPORTED_FORMATS:
        invalid["policy.format"] = (
            f"unsupported format '{fmt}'; Phase 1 only supports 'onnx'"
        )

    # observation_space.dim — positive int
    obs_dim = _get_nested(raw, "observation_space.dim")
    if not isinstance(obs_dim, int) or obs_dim <= 0:
        invalid["observation_space.dim"] = (
            f"must be a positive integer, got {obs_dim!r}"
        )

    # action_space.dim — positive int
    act_dim = _get_nested(raw, "action_space.dim")
    if not isinstance(act_dim, int) or act_dim <= 0:
        invalid["action_space.dim"] = (
            f"must be a positive integer, got {act_dim!r}"
        )

    # runtime.control_frequency_hz > 0
    freq = _get_nested(raw, "runtime.control_frequency_hz")
    if not isinstance(freq, (int, float)) or freq <= 0:
        invalid["runtime.control_frequency_hz"] = (
            f"must be a positive number, got {freq!r}"
        )

    # runtime.decimation — positive int
    dec = _get_nested(raw, "runtime.decimation")
    if not isinstance(dec, int) or dec <= 0:
        invalid["runtime.decimation"] = (
            f"must be a positive integer, got {dec!r}"
        )

    # robot.num_joints vs len(robot.joint_names)
    num_joints = _get_nested(raw, "robot.num_joints")
    joint_names = _get_nested(raw, "robot.joint_names")
    if isinstance(num_joints, int) and isinstance(joint_names, list):
        if num_joints != len(joint_names):
            invalid["robot.num_joints"] = (
                f"num_joints={num_joints} does not match "
                f"len(joint_names)={len(joint_names)}"
            )

    if invalid:
        raise ManifestValidationError(
            f"Manifest has invalid fields: {list(invalid.keys())}",
            invalid_fields=invalid,
            bundle_path=bundle_path,
        )


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CheckpointBundle:
    """Fully-validated, fully-typed bundle descriptor."""
    name: str
    version: str
    bundle_path: Path
    policy_file: Path          # absolute: bundle_path / manifest["policy"]["file"]
    policy_format: str         # "onnx"
    obs_dim: int
    action_dim: int
    action_type: str
    control_frequency_hz: float
    decimation: int
    robot_brand: str
    num_joints: int
    joint_names: List[str]
    raw_manifest: Dict[str, Any]   # preserved for forward-compat keys


@dataclass
class PolicyInfo:
    """Lightweight summary for CheckpointRegistry listings."""
    policy_id: str          # directory name under custom_checkpoints/
    name: str               # manifest["name"]
    version: str
    robot_brand: str
    bundle_path: Path
    is_valid: bool
    error: Optional[str] = None   # set when is_valid=False
