"""Infer SkillManifest v2 fields from ONNX model shapes and heuristics.

When importing bundles from HuggingFace (or other external sources) that lack
a ``skill`` section in their manifest, this module inspects the ONNX model's
input/output tensor shapes to derive ``action_dim``, ``observation_dim``, and
``inference_backend``.  Remaining fields are filled via heuristics from the
repository identifier and file-system contents.

Public API
----------
infer_skill_manifest_fields(bundle_path, onnx_path, repo_id) -> dict
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Known robot configurations — used for heuristic fallback
# ---------------------------------------------------------------------------

_KNOWN_CONFIGS: Dict[str, Dict[str, Any]] = {
    "go2": {
        "target_robot_family": "quadruped",
        "target_robot_models": ["unitree_go2"],
        "required_sensors": ["imu", "joint_encoder"],
        "default_obs_dim": 45,
        "default_action_dim": 12,
        "action_space_type": "joint_position",
        "control_frequency_hz": 50.0,
    },
    "go1": {
        "target_robot_family": "quadruped",
        "target_robot_models": ["unitree_go1"],
        "required_sensors": ["imu", "joint_encoder"],
        "default_obs_dim": 45,
        "default_action_dim": 12,
        "action_space_type": "joint_position",
        "control_frequency_hz": 50.0,
    },
    "b1": {
        "target_robot_family": "quadruped",
        "target_robot_models": ["unitree_b1"],
        "required_sensors": ["imu", "joint_encoder"],
        "default_obs_dim": 45,
        "default_action_dim": 12,
        "action_space_type": "joint_position",
        "control_frequency_hz": 50.0,
    },
    "h1": {
        "target_robot_family": "biped",
        "target_robot_models": ["unitree_h1"],
        "required_sensors": ["imu", "joint_encoder"],
        "default_obs_dim": 60,
        "default_action_dim": 19,
        "action_space_type": "joint_position",
        "control_frequency_hz": 50.0,
    },
    "spot": {
        "target_robot_family": "quadruped",
        "target_robot_models": ["boston_dynamics_spot"],
        "required_sensors": ["imu", "joint_encoder"],
        "default_obs_dim": 45,
        "default_action_dim": 12,
        "action_space_type": "joint_position",
        "control_frequency_hz": 50.0,
    },
    "cyberdog": {
        "target_robot_family": "quadruped",
        "target_robot_models": ["xiaomi_cyberdog"],
        "required_sensors": ["imu", "joint_encoder"],
        "default_obs_dim": 45,
        "default_action_dim": 12,
        "action_space_type": "joint_position",
        "control_frequency_hz": 50.0,
    },
}


# ---------------------------------------------------------------------------
# ONNX shape extraction
# ---------------------------------------------------------------------------

def _extract_onnx_shapes(onnx_path: Path) -> Dict[str, Any]:
    """Inspect an ONNX model and return input/output dimensions.

    Returns a dict with keys ``obs_dim``, ``action_dim``, ``input_names``,
    ``output_names``.  Values are ``None`` when extraction fails.
    """
    result: Dict[str, Any] = {
        "obs_dim": None,
        "action_dim": None,
        "input_names": [],
        "output_names": [],
    }

    try:
        import onnxruntime as ort

        session = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )

        inputs = session.get_inputs()
        outputs = session.get_outputs()

        result["input_names"] = [inp.name for inp in inputs]
        result["output_names"] = [out.name for out in outputs]

        # Convention: input shape = (batch, obs_dim), output = (batch, action_dim)
        if inputs:
            shape = inputs[0].shape
            if len(shape) >= 2:
                dim = shape[-1]
                if isinstance(dim, int) and dim > 0:
                    result["obs_dim"] = dim

        if outputs:
            shape = outputs[0].shape
            if len(shape) >= 2:
                dim = shape[-1]
                if isinstance(dim, int) and dim > 0:
                    result["action_dim"] = dim

        log.debug(
            "ONNX shape extraction for %s: obs_dim=%s, action_dim=%s",
            onnx_path.name, result["obs_dim"], result["action_dim"],
        )
    except ImportError:
        log.warning("onnxruntime not available — cannot inspect ONNX shapes")
    except Exception as exc:  # noqa: BLE001
        log.warning("Failed to inspect ONNX model %s: %s", onnx_path, exc)

    return result


# ---------------------------------------------------------------------------
# Heuristic helpers
# ---------------------------------------------------------------------------

def _match_robot_config(repo_id: str) -> Optional[Dict[str, Any]]:
    """Match repo_id keywords against known robot configurations."""
    rid = repo_id.lower()
    for key, config in _KNOWN_CONFIGS.items():
        if key in rid:
            return config
    return None


def _detect_normalizer(bundle_path: Path) -> Optional[str]:
    """Check for common normalizer file names in the bundle."""
    for name in ("obs_normalizer.npz", "normalizer.npz", "obs_rms.npz",
                 "obs_normalizer.pkl", "normalizer.pkl"):
        if (bundle_path / name).exists():
            return name
    return None


def _detect_inference_backend(onnx_path: Optional[Path], bundle_path: Path) -> str:
    """Determine inference backend from available model files."""
    if onnx_path is not None and onnx_path.exists():
        return "onnx"
    # Check for TorchScript
    for pattern in ("*.pt", "policy_jit*.pt"):
        if list(bundle_path.glob(pattern)):
            return "torchscript"
    return "onnx"


def _detect_model_path(onnx_path: Optional[Path], bundle_path: Path) -> str:
    """Return relative model path within the bundle."""
    if onnx_path is not None and onnx_path.exists():
        try:
            return str(onnx_path.relative_to(bundle_path)).replace("\\", "/")
        except ValueError:
            return onnx_path.name
    # Fallback: look for common names
    for name in ("policy.onnx", "policy_jit.pt"):
        if (bundle_path / name).exists():
            return name
    return "policy.onnx"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def infer_skill_manifest_fields(
    bundle_path: Path,
    *,
    onnx_path: Optional[Path] = None,
    repo_id: str = "",
    existing_manifest: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Infer SkillManifest v2 ``skill`` section fields for an imported bundle.

    Inference priority (highest → lowest):

    1. ONNX model shape inspection (``action_dim``, ``observation_dim``)
    2. Existing v1 manifest fields (``existing_manifest``)
    3. Known robot configuration heuristics (from ``repo_id``)
    4. Conservative defaults

    Parameters
    ----------
    bundle_path:
        Root directory of the checkpoint bundle.
    onnx_path:
        Explicit path to the ONNX model file.  When *None*, the function
        searches ``bundle_path`` for common ONNX file names.
    repo_id:
        HuggingFace repository identifier (e.g. ``"owner/sac-unitree-go2-mujoco"``).
        Used for keyword-based heuristics.
    existing_manifest:
        Already-parsed v1 manifest dict (if available).  Fields like
        ``observation_space.dim`` and ``action_space.dim`` are used as fallbacks
        when ONNX inspection is unavailable.

    Returns
    -------
    dict
        A dict suitable for ``manifest["skill"] = result`` in manifest.yaml v2.
    """
    bundle_path = Path(bundle_path)
    existing = existing_manifest or {}

    # Auto-detect ONNX path if not given
    if onnx_path is None:
        for name in ("policy.onnx",):
            candidate = bundle_path / name
            if candidate.exists():
                onnx_path = candidate
                break
        if onnx_path is None:
            for candidate in bundle_path.rglob("*.onnx"):
                onnx_path = candidate
                break

    # 1. ONNX shape extraction
    onnx_info = _extract_onnx_shapes(onnx_path) if onnx_path else {
        "obs_dim": None, "action_dim": None, "input_names": [], "output_names": [],
    }

    # 2. Existing manifest fallbacks
    obs_section = existing.get("observation_space", {}) or {}
    act_section = existing.get("action_space", {}) or {}
    runtime_section = existing.get("runtime", {}) or {}
    robot_section = existing.get("robot", {}) or {}

    # 3. Robot config heuristics
    robot_config = _match_robot_config(repo_id)

    # --- Resolve each field with priority cascade ---

    # action_dim: ONNX > existing manifest > robot config > 12
    action_dim = (
        onnx_info["action_dim"]
        or _safe_int(act_section.get("dim"))
        or (robot_config or {}).get("default_action_dim")
        or 12
    )

    # observation_dim: ONNX > existing manifest > robot config > 45
    observation_dim = (
        onnx_info["obs_dim"]
        or _safe_int(obs_section.get("dim"))
        or (robot_config or {}).get("default_obs_dim")
        or 45
    )

    # action_space_type: existing manifest > robot config > joint_position
    action_space_type = (
        act_section.get("type")
        or (robot_config or {}).get("action_space_type")
        or "joint_position"
    )

    # control_frequency_hz: existing manifest > robot config > 50.0
    control_frequency_hz = (
        _safe_float(runtime_section.get("control_frequency_hz"))
        or (robot_config or {}).get("control_frequency_hz")
        or 50.0
    )

    # target_robot_family: robot config > "quadruped"
    target_robot_family = (robot_config or {}).get("target_robot_family", "quadruped")

    # target_robot_models: robot config > []
    target_robot_models: List[str] = (robot_config or {}).get("target_robot_models", [])
    # Also try to build from existing manifest robot section
    if not target_robot_models:
        brand = str(robot_section.get("brand", "") or "")
        rtype = str(robot_section.get("type", "") or "")
        if brand and rtype:
            target_robot_models = [f"{brand}_{rtype}"]
        elif brand:
            target_robot_models = [brand]

    # required_sensors: robot config > default
    required_sensors = (robot_config or {}).get("required_sensors", ["imu", "joint_encoder"])

    # observation_space_keys: existing > []
    observation_space_keys = obs_section.get("components", [])

    # inference_backend / model_path
    inference_backend = _detect_inference_backend(onnx_path, bundle_path)
    model_path = _detect_model_path(onnx_path, bundle_path)

    # normalizer
    normalizer_path = _detect_normalizer(bundle_path)
    normalize_obs = normalizer_path is not None

    # skill_id / skill_name from repo_id or bundle name
    repo_name = repo_id.split("/")[-1] if repo_id else bundle_path.name
    skill_id = repo_name.replace("-", "_").lower()
    skill_name = existing.get("name", repo_name)

    fields: Dict[str, Any] = {
        "skill_id": skill_id,
        "skill_name": skill_name,
        "version": existing.get("version", "1.0.0"),
        "source_type": "hf_import",
        "action_space_type": action_space_type,
        "action_dim": action_dim,
        "control_frequency_hz": control_frequency_hz,
        "action_range": None,
        "required_sensors": required_sensors,
        "observation_space_keys": observation_space_keys,
        "observation_dim": observation_dim,
        "target_robot_family": target_robot_family,
        "target_robot_models": target_robot_models,
        "joint_mapping": None,
        "precondition": {
            "posture": "any",
            "posture_tolerance_rad": 0.3,
            "velocity_max_mps": 1.0,
        },
        "postcondition": {
            "posture": "standing",
            "posture_tolerance_rad": 0.3,
            "velocity_max_mps": 0.5,
        },
        "inference_backend": inference_backend,
        "model_path": model_path,
        "normalize_obs": normalize_obs,
        "normalizer_path": normalizer_path,
        "training_source": None,
        "description": f"HuggingFace import: {repo_id}" if repo_id else f"Imported bundle: {bundle_path.name}",
        "tags": ["hf_import"],
    }

    log.info(
        "Inferred SkillManifest fields for '%s': action_dim=%d, obs_dim=%d, "
        "action_space=%s, robot_family=%s",
        skill_id, action_dim, observation_dim, action_space_type, target_robot_family,
    )
    return fields


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_int(value: Any) -> Optional[int]:
    """Convert *value* to int, returning None on failure."""
    if value is None:
        return None
    try:
        v = int(value)
        return v if v > 0 else None
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> Optional[float]:
    """Convert *value* to float, returning None on failure."""
    if value is None:
        return None
    try:
        v = float(value)
        return v if v > 0 else None
    except (TypeError, ValueError):
        return None
