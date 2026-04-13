"""Parse Isaac Lab's exported env.yaml into a SkillManifest v2.

Isaac Lab's play.py export produces an ``env.yaml`` that describes the full
environment configuration.  This module extracts the information needed to
create a reactive SkillManifest with correct observation layout, action
config, and command_interface.

Public API
----------
parse_isaac_lab_env_yaml(env_yaml_path) -> SkillManifest
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


# ---------------------------------------------------------------------------
# YAML loader with narrow Python-builtin support
# ---------------------------------------------------------------------------
# Isaac Lab's ``isaaclab.utils.io.dump_yaml`` uses ``yaml.Dumper`` (not
# ``yaml.SafeDumper``), so the exported ``env.yaml`` contains Python-typed
# tags such as ``!python/tuple`` (range-style configs like friction / mass /
# init pose) and ``!python/object/apply:builtins.slice`` (observation index
# ranges). Loading with ``yaml.safe_load`` raises ConstructorError on the
# first such tag.
#
# Switching to ``yaml.unsafe_load`` would let *any* Python class be
# instantiated from the file — too broad for a parser that ingests data from
# an external training tool. Instead we subclass SafeLoader and explicitly
# register constructors for the small set of builtin tags Isaac Lab actually
# emits. Everything else keeps going through SafeLoader's strict path; if
# Isaac Lab starts emitting a new tag in a future version we will see a
# loud ConstructorError pointing at it instead of silently constructing
# arbitrary objects.
class _IsaacLabEnvYamlLoader(yaml.SafeLoader):
    """SafeLoader extended with the Python-builtin tags Isaac Lab emits."""


def _construct_python_tuple(loader: yaml.Loader, node: yaml.Node) -> tuple:
    return tuple(loader.construct_sequence(node))


def _construct_builtin_slice(loader: yaml.Loader, node: yaml.Node) -> slice:
    # Isaac Lab serializes ``slice(start, stop, step)`` as a sequence node
    # under the apply tag; reconstruct by calling the builtin with the same
    # positional arguments.
    args = loader.construct_sequence(node, deep=True)
    return slice(*args)


_IsaacLabEnvYamlLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple",
    _construct_python_tuple,
)
_IsaacLabEnvYamlLoader.add_constructor(
    "tag:yaml.org,2002:python/object/apply:builtins.slice",
    _construct_builtin_slice,
)

from src.system.skill.skill_manifest import (
    ActionSpaceType,
    CommandField,
    CommandInterface,
    InferenceBackend,
    Postcondition,
    Precondition,
    SkillManifest,
    SourceType,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Known observation term dimensions
# ---------------------------------------------------------------------------

# Fallback dim table when env.yaml doesn't specify per-term dims explicitly.
# These match Isaac Lab's common velocity-tracking locomotion tasks.
_DEFAULT_OBS_TERM_DIMS: Dict[str, int] = {
    "base_lin_vel": 3,
    "base_ang_vel": 3,
    "projected_gravity": 3,
    "velocity_commands": 3,
    "joint_pos": 12,      # Go2 default; overridden by actual joint count
    "joint_vel": 12,
    "joint_pos_rel": 12,
    "joint_vel_rel": 12,
    "actions": 12,
    "height_scan": 187,   # rough terrain
}

# Observation term names that represent the command input
_COMMAND_TERM_NAMES = {"velocity_commands", "base_velocity", "commands"}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_isaac_lab_env_yaml(
    env_yaml_path: Path,
    *,
    skill_id: str = "",
    model_path: str = "policy.onnx",
) -> SkillManifest:
    """Parse an Isaac Lab env.yaml and return a fully populated SkillManifest.

    Parameters
    ----------
    env_yaml_path:
        Path to the exported env.yaml.
    skill_id:
        Override for skill_id.  If empty, derived from task name.
    model_path:
        Relative path to the policy file within the bundle.

    Returns
    -------
    SkillManifest
        A v2 manifest with ``execution_mode="reactive"`` and
        ``command_interface`` populated from the observation layout.
    """
    env_yaml_path = Path(env_yaml_path)
    with env_yaml_path.open("r", encoding="utf-8") as fh:
        raw = yaml.load(fh, Loader=_IsaacLabEnvYamlLoader) or {}

    # Navigate Isaac Lab env.yaml structure
    # Typical: {"scene": {...}, "observations": {...}, "actions": {...}, "commands": {...}, ...}
    obs_cfg = _extract_obs_config(raw)
    act_cfg = _extract_action_config(raw)
    cmd_cfg = _extract_command_config(raw)
    robot_cfg = _extract_robot_config(raw)

    # --- Compute observation layout ---
    obs_terms, obs_dim, command_start_idx = _compute_obs_layout(obs_cfg, robot_cfg)
    obs_keys = [t["name"] for t in obs_terms]

    # --- action ---
    num_joints = robot_cfg.get("num_joints", 12)
    action_dim = act_cfg.get("dim", num_joints)
    action_scale = act_cfg.get("scale", 0.25)

    # --- control frequency ---
    sim_dt = _deep_get(raw, "sim.dt", 0.005)
    decimation = _deep_get(raw, "decimation", _deep_get(raw, "sim.decimation", 4))
    control_dt = sim_dt * decimation
    control_freq = 1.0 / control_dt if control_dt > 0 else 50.0

    # --- command interface ---
    command_fields = _build_command_fields(cmd_cfg, command_start_idx)
    cmd_interface = CommandInterface(
        type="velocity_2d",
        fields=tuple(command_fields),
    ) if command_fields else None

    # --- joint mapping ---
    joint_names = robot_cfg.get("joint_names", [])
    joint_mapping = {name: name for name in joint_names} if joint_names else None

    # --- default joint positions (for action offset) ---
    default_joint_pos = robot_cfg.get("default_joint_pos", {})

    # --- skill_id ---
    task_name = raw.get("task_name", "") or raw.get("env_name", "") or ""
    if not skill_id:
        skill_id = task_name.replace("-", "_").lower() if task_name else "isaac_lab_policy"

    # --- tags ---
    tags = ["isaac_lab"]
    if task_name:
        tags.append(task_name)

    return SkillManifest(
        skill_id=skill_id,
        skill_name=task_name or skill_id,
        version="1.0.0",
        source_type=SourceType.ISAAC_LAB,
        execution_mode="reactive",
        action_space_type=ActionSpaceType.JOINT_POSITION,
        action_dim=action_dim,
        control_frequency_hz=control_freq,
        action_range=None,
        required_sensors=["imu", "joint_encoder"],
        observation_space_keys=obs_keys,
        observation_dim=obs_dim,
        target_robot_family=robot_cfg.get("family", "quadruped"),
        target_robot_models=robot_cfg.get("models", []),
        joint_mapping=joint_mapping,
        precondition=Precondition(posture="standing", velocity_max_mps=0.5),
        postcondition=Postcondition(posture="standing"),
        inference_backend=InferenceBackend.ONNX,
        model_path=model_path,
        normalize_obs=False,
        normalizer_path=None,
        command_interface=cmd_interface,
        training_source=f"isaac_lab_{task_name}" if task_name else "isaac_lab",
        description=f"Isaac Lab policy: {task_name}",
        tags=tags,
    )


# ---------------------------------------------------------------------------
# Config extractors
# ---------------------------------------------------------------------------

def _extract_obs_config(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Extract observation configuration from env.yaml."""
    # Isaac Lab nests obs under "observations.policy" or "observations"
    obs = raw.get("observations", {})
    if isinstance(obs, dict) and "policy" in obs:
        obs = obs["policy"]
    return obs if isinstance(obs, dict) else {}


def _extract_action_config(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Extract action configuration."""
    actions = raw.get("actions", {})
    if isinstance(actions, dict) and "joint_pos" in actions:
        act = actions["joint_pos"]
        return {
            "dim": act.get("dim", 12),
            "scale": act.get("scale", 0.25),
            "use_default_offset": act.get("use_default_offset", True),
        }
    return actions if isinstance(actions, dict) else {}


def _extract_command_config(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Extract command ranges configuration."""
    cmds = raw.get("commands", {})
    if isinstance(cmds, dict) and "base_velocity" in cmds:
        return cmds["base_velocity"]
    return cmds if isinstance(cmds, dict) else {}


def _extract_robot_config(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Extract robot/articulation config."""
    result: Dict[str, Any] = {
        "family": "quadruped",
        "models": [],
        "num_joints": 12,
        "joint_names": [],
        "default_joint_pos": {},
    }

    # Scene > articulation > robot
    scene = raw.get("scene", {})
    robot = scene.get("robot", scene.get("articulation", {}))
    if isinstance(robot, dict):
        init_state = robot.get("init_state", {})
        joint_pos = init_state.get("joint_pos", {})
        if isinstance(joint_pos, dict):
            result["default_joint_pos"] = joint_pos
            all_joints = []
            for pattern, _val in joint_pos.items():
                # Isaac Lab uses regex patterns like ".*_hip_joint"
                # For now just store them; real resolution needs the USD/URDF
                all_joints.append(pattern)
            if all_joints:
                result["joint_names"] = all_joints

        actuators = robot.get("actuators", {})
        joint_count = 0
        for _name, act_cfg in actuators.items():
            if isinstance(act_cfg, dict):
                joint_names_expr = act_cfg.get("joint_names_expr", [])
                if isinstance(joint_names_expr, list):
                    joint_count += len(joint_names_expr)
        if joint_count > 0:
            result["num_joints"] = joint_count

    # Robot name heuristic
    raw_str = str(raw).lower()
    if "go2" in raw_str:
        result["models"] = ["unitree_go2"]
    elif "h1" in raw_str:
        result["family"] = "biped"
        result["models"] = ["unitree_h1"]
    elif "go1" in raw_str:
        result["models"] = ["unitree_go1"]

    return result


# ---------------------------------------------------------------------------
# Observation layout computation
# ---------------------------------------------------------------------------

def _compute_obs_layout(
    obs_cfg: Dict[str, Any],
    robot_cfg: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], int, int]:
    """Walk observation terms and compute layout.

    Returns (terms_list, total_dim, command_start_index).
    ``command_start_index`` is -1 if no command term found.
    """
    terms: List[Dict[str, Any]] = []
    total_dim = 0
    command_start_idx = -1
    num_joints = robot_cfg.get("num_joints", 12)

    # obs_cfg may have term entries as sub-dicts with "func" and params
    # or may be a simple ordered list
    if not obs_cfg:
        return terms, 0, -1

    for term_name, term_cfg in obs_cfg.items():
        if term_name.startswith("_") or not isinstance(term_cfg, dict):
            continue

        # Determine dimension
        dim = term_cfg.get("dim", None)
        if dim is None:
            # Heuristic from term name
            base_name = term_name.split(".")[-1] if "." in term_name else term_name
            dim = _DEFAULT_OBS_TERM_DIMS.get(base_name, None)
            if dim is None:
                # Joint-related terms scale with joint count
                if "joint" in base_name or "action" in base_name:
                    dim = num_joints
                else:
                    dim = 3  # conservative default

        # Check if this is the command term
        base_name = term_name.split(".")[-1] if "." in term_name else term_name
        if base_name in _COMMAND_TERM_NAMES:
            command_start_idx = total_dim

        terms.append({
            "name": base_name,
            "start_idx": total_dim,
            "dim": dim,
        })
        total_dim += dim

    return terms, total_dim, command_start_idx


# ---------------------------------------------------------------------------
# Command field builder
# ---------------------------------------------------------------------------

def _build_command_fields(
    cmd_cfg: Dict[str, Any],
    command_start_idx: int,
) -> List[CommandField]:
    """Build CommandField list from command config and obs layout."""
    if command_start_idx < 0:
        return []

    # Extract ranges from command config
    ranges = cmd_cfg.get("ranges", cmd_cfg)
    lin_vel_x = _to_range(ranges.get("lin_vel_x", [-1.0, 1.0]))
    lin_vel_y = _to_range(ranges.get("lin_vel_y", [-0.5, 0.5]))
    ang_vel_z = _to_range(ranges.get("ang_vel_z", [-1.0, 1.0]))

    return [
        CommandField(
            name="vx",
            obs_index=command_start_idx,
            range=lin_vel_x,
            default=0.0,
        ),
        CommandField(
            name="vy",
            obs_index=command_start_idx + 1,
            range=lin_vel_y,
            default=0.0,
        ),
        CommandField(
            name="vyaw",
            obs_index=command_start_idx + 2,
            range=ang_vel_z,
            default=0.0,
        ),
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_range(val: Any) -> Tuple[float, float]:
    """Convert a range value to (min, max) tuple."""
    if isinstance(val, (list, tuple)) and len(val) >= 2:
        return (float(val[0]), float(val[1]))
    return (-1.0, 1.0)


def _deep_get(d: Dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    """Get a nested value using dot-separated keys."""
    keys = dotted_key.split(".")
    node = d
    for k in keys:
        if isinstance(node, dict) and k in node:
            node = node[k]
        else:
            return default
    return node
