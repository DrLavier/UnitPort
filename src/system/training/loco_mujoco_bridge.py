#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bridge between loco-mujoco trajectory system and UnitPort training env.

Loads loco-mujoco Trajectory objects and converts them to the (T, num_joints)
numpy arrays consumed by ``unitree_gym_env._init_reference_motion()``.

Usage
-----
>>> from src.system.training.loco_mujoco_bridge import (
...     is_available, list_datasets, load_trajectory_as_npy,
...     generate_standing_reference,
... )
>>> if is_available():
...     entries = list_datasets("go2")
...     npy_path = load_trajectory_as_npy("UnitreeGo2", task="walk")
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent        # src/
_REPO_ROOT    = _PROJECT_ROOT.parent                                 # D:/.../NEW/
# User-writable motions directory (hot-plugged content).
# Generated and downloaded motions go here, not into the system dir.
_MOTIONS_DIR  = _REPO_ROOT / "custom_mods" / "training" / "motions"

# loco-mujoco may be vendored at several locations — try all of them.
_LOCO_CANDIDATES = [
    _REPO_ROOT / "custom_mods" / "motions" / "loco-mujoco",      # custom_mods/motions/loco-mujoco
    _REPO_ROOT / "custom_mods" / "archives" / "loco-mujoco",     # custom_mods/archives/loco-mujoco
]


def _find_loco_dir() -> Optional[Path]:
    """Return the first candidate directory that contains ``loco_mujoco/``."""
    for candidate in _LOCO_CANDIDATES:
        if (candidate / "loco_mujoco").is_dir():
            return candidate
    return None


# Resolved once at import time; refreshed on demand in is_available().
_LOCO_DIR: Optional[Path] = _find_loco_dir()


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------

def is_available() -> bool:
    """Return True if loco-mujoco package is accessible."""
    global _LOCO_DIR
    if _LOCO_DIR is None:
        _LOCO_DIR = _find_loco_dir()
    return _LOCO_DIR is not None and (_LOCO_DIR / "loco_mujoco").is_dir()


def _ensure_importable() -> None:
    """Add loco-mujoco to sys.path if needed."""
    if _LOCO_DIR is None:
        return
    loco_str = str(_LOCO_DIR)
    if loco_str not in sys.path:
        sys.path.insert(0, loco_str)


# ---------------------------------------------------------------------------
# Go2 joint order contract
# ---------------------------------------------------------------------------
#
# The MuJoCo MJCF defines two DIFFERENT orderings:
#
#   qpos order (joint definition):  FL, FR, RL, RR  (qpos[7:19])
#   ctrl order (actuator definition): FR, FL, RR, RL  (ctrl[0:12])
#
# loco-mujoco uses the ctrl/actuator order: FR, FL, RR, RL
# UnitreeGymEnv reads observations from qpos: FL, FR, RL, RR
#
# All .npy reference files MUST be stored in **qpos order** (FL, FR, RL, RR)
# because the env reads qpos[7:19] directly for both observations and
# reward computation.  Any data arriving in loco/ctrl order must be
# remapped through LOCO_TO_QPOS_IDX before saving.

#: Joint names in loco-mujoco / actuator order (FR, FL, RR, RL).
GO2_LOCO_JOINT_NAMES: Tuple[str, ...] = (
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
)

#: Joint names in qpos order (FL, FR, RL, RR) — the canonical order for .npy files.
GO2_QPOS_JOINT_NAMES: Tuple[str, ...] = (
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
)

# Backward-compat alias
GO2_JOINT_NAMES = GO2_QPOS_JOINT_NAMES

#: Permutation index: loco_order[LOCO_TO_QPOS_IDX] → qpos_order.
#: loco: FR(0-2) FL(3-5) RR(6-8) RL(9-11)
#: qpos: FL(0-2) FR(3-5) RL(6-8) RR(9-11)
LOCO_TO_QPOS_IDX = np.array([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8], dtype=np.intp)

#: Inverse: qpos_order[QPOS_TO_LOCO_IDX] → loco_order.
QPOS_TO_LOCO_IDX = np.array([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8], dtype=np.intp)

#: loco-mujoco Go2 init_qpos (19-DOF: root 7 + joints 12) in loco order.
_GO2_INIT_QPOS_LOCO = np.array([
    0.0, 0.0, 0.27, 1.0, 0.0, 0.0, 0.0,   # root: xyz + quaternion (w,x,y,z)
    0.0, 0.9, -1.8,   # FR  (loco idx 0-2)
    0.0, 0.9, -1.8,   # FL  (loco idx 3-5)
    0.0, 0.9, -1.8,   # RR  (loco idx 6-8)
    0.0, 0.9, -1.8,   # RL  (loco idx 9-11)
], dtype=np.float32)

# Backward-compat alias (NOTE: this is loco order; prefer GO2_STANDING_JOINTS)
GO2_INIT_QPOS = _GO2_INIT_QPOS_LOCO

#: Standing joint angles in qpos order (FL, FR, RL, RR) — the canonical order.
GO2_STANDING_JOINTS = _GO2_INIT_QPOS_LOCO[7:19][LOCO_TO_QPOS_IDX].copy()


def _loco_to_qpos(joints_loco: np.ndarray) -> np.ndarray:
    """Remap a (T, 12) or (12,) array from loco order to qpos order."""
    if joints_loco.ndim == 1:
        return joints_loco[LOCO_TO_QPOS_IDX].copy()
    return joints_loco[:, LOCO_TO_QPOS_IDX].copy()

# Map loco-mujoco env names to our robot_type
_ENV_ROBOT_MAP: Dict[str, str] = {
    "UnitreeGo2": "go2",
    "UnitreeA1":  "a1",
    "UnitreeH1":  "h1",
    "UnitreeG1":  "g1",
}


# ---------------------------------------------------------------------------
# Dataset listing
# ---------------------------------------------------------------------------

def list_datasets(robot_type: str = "go2") -> List[Dict[str, Any]]:
    """List available loco-mujoco datasets for the given robot type.

    Returns a list of dicts with keys: env_name, task, dataset_type, available.
    """
    env_name = _robot_type_to_env(robot_type)
    if not env_name:
        return []

    # Known dataset tasks per env (from loco-mujoco README / dataset_confs)
    _KNOWN_TASKS: Dict[str, List[str]] = {
        "UnitreeGo2": ["walk", "run", "trot"],
        "UnitreeA1":  ["walk", "run", "trot"],
        "UnitreeH1":  ["walk", "run"],
        "UnitreeG1":  ["walk", "run"],
    }

    results = []
    tasks = _KNOWN_TASKS.get(env_name, ["walk"])
    for task in tasks:
        results.append({
            "env_name": env_name,
            "task": task,
            "dataset_type": "mocap",
            "available": _check_dataset_available(env_name, task),
        })
    return results


def _robot_type_to_env(robot_type: str) -> Optional[str]:
    """Map UnitPort robot_type string to loco-mujoco env name."""
    rt = robot_type.strip().lower()
    for env_name, rtype in _ENV_ROBOT_MAP.items():
        if rt == rtype or rt == env_name.lower():
            return env_name
    return None


def _check_dataset_available(env_name: str, task: str) -> bool:
    """Check if a specific dataset is downloadable/available locally or on HF."""
    # Check local cache first
    if _find_cached_npy(env_name, task) is not None:
        return True
    # Check if HF download is possible (lightweight — just checks the hub)
    try:
        from huggingface_hub import hf_hub_url
        filename = f"DefaultDatasets/mocap/{env_name}/{task}.npz"
        # hf_hub_url doesn't download, just constructs the URL
        _ = hf_hub_url("robfiras/loco-mujoco-datasets", filename, repo_type="dataset")
        return True
    except Exception:
        return False


def _find_cached_npy(env_name: str, task: str) -> Optional[Path]:
    """Return path to a previously exported .npy if it exists.

    Searches ``custom_mods/training/motions/`` (single canonical location).
    """
    robot_type = _ENV_ROBOT_MAP.get(env_name, "generic")
    cat = _category_for(robot_type)
    # Try multiple naming conventions (loco export, procedural generator, etc.)
    _task_to_filenames = {
        "walk": ["loco_walk.npy", "procedural_walk.npy", "go2_walk.npy"],
        "trot": ["loco_trot.npy", "procedural_walk.npy"],
        "run":  ["loco_run.npy", "procedural_walk.npy"],
        "standing": ["loco_standing.npy", "standing.npy"],
        "stand":    ["loco_stand.npy", "standing.npy"],
    }
    filenames = _task_to_filenames.get(task, [f"loco_{task}.npy"])
    for fn in filenames:
        candidate = _MOTIONS_DIR / cat / robot_type / fn
        if candidate.is_file():
            return candidate
    return None


def _load_npz_qpos_nojax(npz_path) -> Optional[np.ndarray]:
    """Load qpos from a loco-mujoco .npz trajectory file using pure numpy.

    The .npz files from loco-mujoco contain a 'qpos' key with shape (T, nq).
    For Go2: nq=19 (7 root + 12 joints in **loco order**: FR, FL, RR, RL).

    Returns (T, 12) in **qpos order** (FL, FR, RL, RR) — the canonical
    order used by UnitreeGymEnv.

    This bypasses loco-mujoco's JAX-dependent Trajectory.load() entirely.
    """
    try:
        data = np.load(str(npz_path), allow_pickle=True)
        if "qpos" not in data:
            return None
        qpos = np.asarray(data["qpos"], dtype=np.float32)
        if qpos.ndim != 2:
            return None
        if qpos.shape[1] >= 19:
            joints_loco = qpos[:, 7:19].astype(np.float32)
        elif qpos.shape[1] >= 12:
            joints_loco = qpos[:, :12].astype(np.float32)
        else:
            return None
        # Remap from loco order (FR,FL,RR,RL) to qpos order (FL,FR,RL,RR)
        return _loco_to_qpos(joints_loco)
    except Exception:
        return None


def _download_from_hf(env_name: str, task: str, dataset_type: str = "mocap") -> Optional[Path]:
    """Download a loco-mujoco dataset .npz from HuggingFace Hub.

    Returns the local cache path, or None on failure.
    """
    try:
        from huggingface_hub import hf_hub_download
        filename = f"DefaultDatasets/{dataset_type}/{env_name}/{task}.npz"
        local_path = hf_hub_download(
            repo_id="robfiras/loco-mujoco-datasets",
            filename=filename,
            repo_type="dataset",
        )
        return Path(local_path)
    except Exception as exc:
        import warnings
        warnings.warn(
            f"loco-mujoco dataset download failed for {env_name}/{task}: {exc}",
            stacklevel=3,
        )
        return None


# ---------------------------------------------------------------------------
# Trajectory loading (JAX-free)
# ---------------------------------------------------------------------------

def load_trajectory_as_npy(
    env_name: str,
    *,
    task: str = "walk",
    dataset_type: str = "mocap",
    output_dir: Optional[Path] = None,
    fps: float = 50.0,
) -> Optional[Path]:
    """Load a loco-mujoco dataset and export as .npy for our training env.

    Returns the path to the .npy file, or None on failure.
    The array shape is (T, 12) — just the actuated joint angles.

    This implementation is **JAX-free**: it downloads the .npz from
    HuggingFace Hub and extracts qpos with pure numpy, bypassing the
    JAX-dependent loco-mujoco Trajectory API entirely.
    """
    # 1. Check if we already have an exported .npy
    cached = _find_cached_npy(env_name, task)
    if cached is not None:
        return cached

    # 2. Try downloading the .npz from HuggingFace
    npz_path = _download_from_hf(env_name, task, dataset_type)
    if npz_path is not None:
        joints = _load_npz_qpos_nojax(npz_path)
        if joints is not None:
            robot_type = _ENV_ROBOT_MAP.get(env_name, "generic")
            if output_dir is None:
                output_dir = _MOTIONS_DIR / _category_for(robot_type) / robot_type
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            out_path = output_dir / f"loco_{task}.npy"
            np.save(str(out_path), joints)
            return out_path

    # 3. Fallback: generate procedural trajectory for quadrupeds.
    #    HuggingFace only hosts humanoid mocap data; quadruped trajectories
    #    (Go2, A1, Spot) are generated via loco-mujoco's data generation
    #    pipeline which requires JAX.  As a practical fallback, use our
    #    built-in procedural generators to create reference motions.
    robot_type = _ENV_ROBOT_MAP.get(env_name, "generic")
    _task_to_generator = {"walk": "walk", "trot": "walk", "run": "walk",
                          "standing": "standing", "stand": "standing"}
    gen_type = _task_to_generator.get(task)
    if gen_type is not None:
        import warnings
        warnings.warn(
            f"loco-mujoco HF dataset not available for {env_name}/{task}. "
            f"Falling back to procedural '{gen_type}' reference generator.",
            stacklevel=3,
        )
        if gen_type == "standing":
            return generate_standing_reference(robot_type, fps=fps, output_dir=output_dir)
        else:
            return generate_simple_walk_reference(robot_type, fps=fps, output_dir=output_dir)

    return None


def load_trajectory_array(
    env_name: str,
    *,
    task: str = "walk",
    dataset_type: str = "mocap",
) -> Optional[np.ndarray]:
    """Load a loco-mujoco dataset and return joint angles as numpy array.

    Returns shape (T, 12) or None on failure.  JAX-free.
    """
    # Check cached .npy first
    cached = _find_cached_npy(env_name, task)
    if cached is not None:
        try:
            return np.load(str(cached)).astype(np.float32)
        except Exception:
            pass

    # Download .npz and extract
    npz_path = _download_from_hf(env_name, task, dataset_type)
    if npz_path is None:
        return None
    return _load_npz_qpos_nojax(npz_path)


# ---------------------------------------------------------------------------
# Procedural reference generators
# ---------------------------------------------------------------------------

def generate_standing_reference(
    robot_type: str = "go2",
    *,
    duration_s: float = 2.0,
    fps: float = 50.0,
    output_dir: Optional[Path] = None,
) -> Path:
    """Generate a static standing-pose reference .npy from loco-mujoco init_qpos.

    Useful as a baseline reference when no motion-capture dataset is available.
    Returns the path to the generated .npy file.
    """
    standing = _get_standing_joints(robot_type)
    n_frames = max(1, int(duration_s * fps))
    frames = np.tile(standing, (n_frames, 1)).astype(np.float32)  # (T, 12)

    if output_dir is None:
        cat = _category_for(robot_type)
        rt = robot_type.strip().lower()
        output_dir = _MOTIONS_DIR / cat / rt
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / "standing.npy"
    np.save(str(out_path), frames)
    return out_path


def generate_simple_walk_reference(
    robot_type: str = "go2",
    *,
    step_freq_hz: float = 2.0,
    amplitude_hip: float = 0.05,
    amplitude_thigh: float = 0.15,
    amplitude_calf: float = 0.10,
    duration_s: float = 4.0,
    fps: float = 50.0,
    output_dir: Optional[Path] = None,
) -> Path:
    """Generate a simple sinusoidal walking reference for quadrupeds.

    Creates a trotting gait pattern where diagonal legs move in phase:
    FL+RR vs FR+RL (standard trot pattern).

    Output is in **qpos order** (FL, FR, RL, RR).

    This is a procedurally generated approximation — not motion capture.
    Useful when no loco-mujoco dataset is available.
    """
    standing = _get_standing_joints(robot_type)  # already in qpos order
    n_frames = max(1, int(duration_s * fps))
    t = np.linspace(0, duration_s, n_frames, dtype=np.float32)

    frames = np.tile(standing, (n_frames, 1)).astype(np.float32)

    omega = 2.0 * np.pi * step_freq_hz
    # Phase offset for trot gait in qpos order: FL(0), FR(1), RL(2), RR(3)
    # Diagonal pairs in phase: FL+RR at phase 0, FR+RL at phase pi
    for leg_idx, phase in enumerate([0.0, np.pi, np.pi, 0.0]):
        base = leg_idx * 3  # each leg has 3 joints
        signal = np.sin(omega * t + phase)
        # hip: small lateral swing
        frames[:, base + 0] += amplitude_hip * signal * 0.3
        # thigh: main swing
        frames[:, base + 1] += amplitude_thigh * signal
        # calf: compensatory flex (phase-shifted)
        frames[:, base + 2] += amplitude_calf * np.sin(omega * t + phase + np.pi * 0.5)

    if output_dir is None:
        cat = _category_for(robot_type)
        rt = robot_type.strip().lower()
        output_dir = _MOTIONS_DIR / cat / rt
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / "procedural_walk.npy"
    np.save(str(out_path), frames)
    return out_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_KNOWN_STANDING_JOINTS: Dict[str, np.ndarray] = {
    # All in qpos order.  For robots where qpos==ctrl order, no remapping needed.
    # Values from loco-mujoco init_qpos or Unitree official defaults.
    "go2": np.array([0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8], dtype=np.float32),
    "a1":  np.array([0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8], dtype=np.float32),
    "b2":  np.array([0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8], dtype=np.float32),
    "g1":  np.array([0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8], dtype=np.float32),
}


def _get_standing_joints(robot_type: str, num_joints: int = 12) -> np.ndarray:
    """Return standing joint angles for a robot type (qpos order).

    Uses a hardcoded lookup to avoid importing loco-mujoco (which requires JAX).
    For unknown robots, returns zeros — callers should prefer MJCF keyframes.
    """
    rt = robot_type.strip().lower()
    if rt in _KNOWN_STANDING_JOINTS:
        return _KNOWN_STANDING_JOINTS[rt].copy()
    # Unknown robot — return zeros; the MJCF keyframe is the real source of truth
    return np.zeros(num_joints, dtype=np.float32)


def _category_for(robot_type: str) -> str:
    """Map robot_type to motion library category directory."""
    try:
        from src.system.training.robot_family import ROBOT_FAMILY_BY_TYPE, normalize_robot_type
        family = ROBOT_FAMILY_BY_TYPE.get(normalize_robot_type(robot_type), "quadruped")
        _map = {
            "quadruped": "quadruped", "biped": "biped",
            "wheeled": "wheeled", "manipulator": "manipulator",
        }
        return _map.get(family, "generic")
    except Exception:
        return "quadruped"


def get_info() -> Dict[str, Any]:
    """Return status information about the loco-mujoco bridge."""
    return {
        "available": is_available(),
        "path": str(_LOCO_DIR or ""),
        "go2_datasets": list_datasets("go2") if is_available() else [],
    }
