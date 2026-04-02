"""Load SkillManifest from a bundle's manifest.yaml.

Supports two manifest versions:
  - **v1** (current): flat structure with ``policy``, ``observation_space``,
    ``action_space``, ``runtime``, ``robot`` sections.  Missing v2 fields are
    filled with sensible defaults via :func:`_migrate_v1`.
  - **v2** (new): adds ``skill`` top-level section with full SkillManifest
    fields including ``precondition`` / ``postcondition``.

Callers should use :func:`load_skill_manifest` as the single entry point.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from src.system.skill.skill_manifest import (
    ActionSpaceType,
    InferenceBackend,
    Postcondition,
    Precondition,
    SkillManifest,
    SourceType,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_skill_manifest(
    bundle_path: Path,
    manifest_filename: str = "manifest.yaml",
) -> SkillManifest:
    """Load and return a :class:`SkillManifest` from *bundle_path*.

    Raises ``FileNotFoundError`` if the manifest file is absent and
    ``ValueError`` for un-parseable content.
    """
    manifest_file = Path(bundle_path) / manifest_filename
    if not manifest_file.exists():
        raise FileNotFoundError(f"No {manifest_filename} in {bundle_path}")

    with open(manifest_file, "r", encoding="utf-8") as fh:
        raw: Dict[str, Any] = yaml.safe_load(fh) or {}

    if "skill" in raw:
        return _parse_v2(raw, bundle_path)
    return _migrate_v1(raw, bundle_path)


# ---------------------------------------------------------------------------
# V2 parser — full SkillManifest section present
# ---------------------------------------------------------------------------

def _parse_v2(raw: Dict[str, Any], bundle_path: Path) -> SkillManifest:
    """Parse a v2 manifest that already has a ``skill`` section."""
    s = raw["skill"]

    return SkillManifest(
        skill_id=s.get("skill_id", bundle_path.name),
        skill_name=s.get("skill_name", raw.get("name", bundle_path.name)),
        version=s.get("version", raw.get("version", "1.0.0")),
        source_type=_enum_or_default(SourceType, s.get("source_type"), SourceType.SB3_INTERNAL),
        action_space_type=_enum_or_default(ActionSpaceType, s.get("action_space_type"), ActionSpaceType.JOINT_POSITION),
        action_dim=int(s.get("action_dim", 12)),
        control_frequency_hz=float(s.get("control_frequency_hz", 50.0)),
        action_range=s.get("action_range"),
        required_sensors=s.get("required_sensors", ["imu", "joint_encoder"]),
        observation_space_keys=s.get("observation_space_keys", []),
        observation_dim=int(s.get("observation_dim", 45)),
        target_robot_family=s.get("target_robot_family", "quadruped"),
        target_robot_models=s.get("target_robot_models", []),
        joint_mapping=s.get("joint_mapping"),
        precondition=_parse_condition(s.get("precondition"), Precondition),
        postcondition=_parse_condition(s.get("postcondition"), Postcondition),
        inference_backend=_enum_or_default(InferenceBackend, s.get("inference_backend"), InferenceBackend.ONNX),
        model_path=s.get("model_path", raw.get("policy", {}).get("file", "policy.onnx")),
        normalize_obs=bool(s.get("normalize_obs", False)),
        normalizer_path=s.get("normalizer_path"),
        training_source=s.get("training_source"),
        description=s.get("description", ""),
        tags=s.get("tags", []),
    )


# ---------------------------------------------------------------------------
# V1 migration — derive SkillManifest from legacy flat structure
# ---------------------------------------------------------------------------

# Maps legacy action_space.type strings to ActionSpaceType enum values
_V1_ACTION_TYPE_MAP: Dict[str, ActionSpaceType] = {
    "joint_position": ActionSpaceType.JOINT_POSITION,
    "joint_torque": ActionSpaceType.JOINT_TORQUE,
    "cartesian_delta": ActionSpaceType.CARTESIAN_DELTA,
    "end_effector_6d": ActionSpaceType.END_EFFECTOR_6D,
    "hybrid": ActionSpaceType.HYBRID,
}

# Maps legacy policy.format strings to InferenceBackend enum values
_V1_FORMAT_MAP: Dict[str, InferenceBackend] = {
    "onnx": InferenceBackend.ONNX,
    "jit": InferenceBackend.TORCHSCRIPT,
}


def _migrate_v1(raw: Dict[str, Any], bundle_path: Path) -> SkillManifest:
    """Derive a SkillManifest from a v1 manifest dict with sensible defaults."""
    policy = raw.get("policy", {})
    obs = raw.get("observation_space", {})
    act = raw.get("action_space", {})
    runtime = raw.get("runtime", {})
    robot = raw.get("robot", {})
    norm = raw.get("normalization", {})
    training = raw.get("training", {})

    name = raw.get("name", bundle_path.name)

    # Observation keys from v1 components list
    obs_keys = obs.get("components", [])

    # Normalizer path
    norm_file = norm.get("file")
    normalize = norm_file is not None

    # Training source
    algo = training.get("algorithm")
    training_source = f"sb3_{algo.lower()}" if algo else None

    log.debug("Migrating v1 manifest '%s' to SkillManifest", name)

    return SkillManifest(
        skill_id=bundle_path.name,
        skill_name=name,
        version=raw.get("version", "1.0.0"),
        source_type=SourceType.SB3_INTERNAL,
        action_space_type=_V1_ACTION_TYPE_MAP.get(
            act.get("type", "joint_position"),
            ActionSpaceType.JOINT_POSITION,
        ),
        action_dim=int(act.get("dim", 12)),
        control_frequency_hz=float(runtime.get("control_frequency_hz", 50.0)),
        action_range=None,
        required_sensors=["imu", "joint_encoder"],
        observation_space_keys=obs_keys,
        observation_dim=int(obs.get("dim", 45)),
        target_robot_family="quadruped",
        target_robot_models=[f"{robot.get('brand', 'unitree')}_{robot.get('type', '')}".rstrip("_")],
        joint_mapping=None,
        # v1 bundles get permissive defaults
        precondition=Precondition(posture="any", velocity_max_mps=2.0),
        postcondition=Postcondition(posture="standing"),
        inference_backend=_V1_FORMAT_MAP.get(
            policy.get("format", "onnx"),
            InferenceBackend.ONNX,
        ),
        model_path=policy.get("file", "policy.onnx"),
        normalize_obs=normalize,
        normalizer_path=norm_file,
        training_source=training_source,
        description=f"Auto-migrated from v1 manifest: {name}",
        tags=["v1_migrated"],
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _enum_or_default(enum_cls, value: Optional[str], default):
    """Resolve *value* to an enum member, falling back to *default*."""
    if value is None:
        return default
    try:
        return enum_cls(value)
    except ValueError:
        log.warning("Unknown %s value '%s', using default %s", enum_cls.__name__, value, default)
        return default


def _parse_condition(data: Optional[Dict[str, Any]], cls):
    """Parse a precondition / postcondition dict into its dataclass."""
    if data is None:
        return cls()
    return cls(
        posture=data.get("posture", cls().posture),
        posture_tolerance_rad=float(data.get("posture_tolerance_rad", cls().posture_tolerance_rad)),
        velocity_max_mps=float(data.get("velocity_max_mps", cls().velocity_max_mps)),
        custom_check=data.get("custom_check"),
    )
