# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""SensorManager — dynamic sensor activation/deactivation per SkillManifest.

Manages MuJoCo sensor streams, activating and deactivating sensors as
behaviour transitions occur.  Each sensor type has a dedicated reader that
extracts data from MuJoCo's ``mjData`` structure.

Supported sensor types:
  - imu              — angular velocity + linear acceleration from body sensors
  - joint_encoder    — joint positions + velocities from qpos/qvel
  - rgb_camera       — RGB image from mujoco.Renderer (offscreen)
  - depth_camera     — depth buffer from mujoco.Renderer (offscreen)
  - force_torque     — contact forces from foot/end-effector sensors

Public API
----------
SensorManager(mj_model, mj_data)
SensorManager.configure(required_sensors)
SensorManager.read_all() -> Dict[str, Any]
SensorManager.active_sensors -> Set[str]

Ported verbatim from DEMO/src/system/runtime/simulation/sensor_manager.py
(Phase 1). No business I/O — pure mujoco buffer reads.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

import numpy as np

log = logging.getLogger(__name__)

# All sensor types the platform recognises
ALL_SENSOR_TYPES = frozenset({
    "imu", "joint_encoder", "rgb_camera", "depth_camera", "force_torque",
})


class SensorManager:
    """Manages MuJoCo sensor streams with dynamic activation.

    Usage::

        sm = SensorManager(mj_model, mj_data)
        sm.configure(["imu", "joint_encoder"])  # activate for current skill

        readings = sm.read_all()
        # {"imu": {"angular_velocity": [...], "linear_acceleration": [...]},
        #  "joint_encoder": {"positions": [...], "velocities": [...]}}

        sm.configure(["imu", "joint_encoder", "force_torque"])  # skill transition
    """

    def __init__(
        self,
        mj_model: Any,
        mj_data: Any,
        joint_names: Optional[List[str]] = None,
    ):
        self._model = mj_model
        self._data = mj_data
        self._joint_names = joint_names or []
        self._active: Set[str] = set()
        self._renderer: Any = None  # Lazy mujoco.Renderer for camera sensors
        self._camera_width = 640
        self._camera_height = 480

    @property
    def active_sensors(self) -> Set[str]:
        """Currently active sensor types."""
        return set(self._active)

    def configure(self, required_sensors: List[str]) -> None:
        """Activate exactly the sensors in *required_sensors*, deactivating others.

        Called at each behaviour transition to match the incoming
        SkillManifest.required_sensors list.
        """
        requested = set(s for s in required_sensors if s in ALL_SENSOR_TYPES)
        deactivated = self._active - requested
        activated = requested - self._active

        if deactivated:
            log.debug("Deactivating sensors: %s", deactivated)
            for s in deactivated:
                self._deactivate(s)

        if activated:
            log.debug("Activating sensors: %s", activated)
            for s in activated:
                self._activate(s)

        self._active = requested

    def read_all(self) -> Dict[str, Any]:
        """Read all active sensors and return a dict keyed by sensor type."""
        readings: Dict[str, Any] = {}
        for sensor_type in self._active:
            reader = _READERS.get(sensor_type)
            if reader is not None:
                try:
                    readings[sensor_type] = reader(self)
                except Exception as exc:
                    log.debug("Sensor read failed for %s: %s", sensor_type, exc)
                    readings[sensor_type] = None
        return readings

    def read(self, sensor_type: str) -> Any:
        """Read a single sensor type. Returns None if not active or on error."""
        if sensor_type not in self._active:
            return None
        reader = _READERS.get(sensor_type)
        if reader is None:
            return None
        try:
            return reader(self)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Activation / deactivation hooks
    # ------------------------------------------------------------------

    def _activate(self, sensor_type: str) -> None:
        """Hook called when a sensor is newly activated."""
        if sensor_type in ("rgb_camera", "depth_camera"):
            self._ensure_renderer()

    def _deactivate(self, sensor_type: str) -> None:
        """Hook called when a sensor is deactivated."""
        # Camera renderer kept alive (cheap to hold, expensive to recreate)

    def _ensure_renderer(self) -> None:
        """Create offscreen renderer if not yet available."""
        if self._renderer is not None:
            return
        try:
            import mujoco
            self._renderer = mujoco.Renderer(
                self._model, self._camera_height, self._camera_width
            )
        except Exception as exc:
            log.warning("Cannot create MuJoCo renderer for camera sensors: %s", exc)


# ---------------------------------------------------------------------------
# Sensor readers
# ---------------------------------------------------------------------------

def _read_imu(sm: SensorManager) -> Dict[str, Any]:
    """Read IMU data: angular velocity + linear acceleration."""
    data = sm._data
    # MuJoCo stores body angular velocity in cvel (6D) or via sensor API
    # Use qvel for base body angular velocity (first 3 dofs of free joint)
    angular_velocity = np.array(data.qvel[3:6], dtype=np.float32).tolist() if data.qvel.shape[0] >= 6 else [0.0, 0.0, 0.0]

    # Linear acceleration from sensor data (if available) or approximate from qacc
    if hasattr(data, "sensordata") and data.sensordata.shape[0] >= 3:
        linear_acceleration = np.array(data.sensordata[:3], dtype=np.float32).tolist()
    elif data.qacc.shape[0] >= 3:
        linear_acceleration = np.array(data.qacc[:3], dtype=np.float32).tolist()
    else:
        linear_acceleration = [0.0, 0.0, 0.0]

    return {
        "angular_velocity": angular_velocity,
        "linear_acceleration": linear_acceleration,
    }


def _read_joint_encoder(sm: SensorManager) -> Dict[str, Any]:
    """Read joint positions and velocities."""
    model = sm._model
    data = sm._data
    nq = model.nq
    nv = model.nv

    # Skip free-joint DOFs (7 qpos, 6 qvel) for floating-base robots
    has_freejoint = nq > nv  # free joint adds 1 extra qpos (quaternion)
    pos_offset = 7 if has_freejoint else 0
    vel_offset = 6 if has_freejoint else 0

    positions = np.array(data.qpos[pos_offset:], dtype=np.float32).tolist()
    velocities = np.array(data.qvel[vel_offset:], dtype=np.float32).tolist()

    return {
        "positions": positions,
        "velocities": velocities,
        "joint_names": list(sm._joint_names),
    }


def _read_rgb_camera(sm: SensorManager) -> Dict[str, Any]:
    """Read RGB image from offscreen renderer."""
    if sm._renderer is None:
        return {"image": None, "width": 0, "height": 0}
    try:
        sm._renderer.update_scene(sm._data)
        image = sm._renderer.render()
        return {
            "image": np.array(image, dtype=np.uint8),
            "width": sm._camera_width,
            "height": sm._camera_height,
        }
    except Exception:
        return {"image": None, "width": 0, "height": 0}


def _read_depth_camera(sm: SensorManager) -> Dict[str, Any]:
    """Read depth buffer from offscreen renderer."""
    if sm._renderer is None:
        return {"depth": None, "width": 0, "height": 0}
    try:
        sm._renderer.update_scene(sm._data)
        sm._renderer.enable_depth_rendering(True)
        depth = sm._renderer.render()
        sm._renderer.enable_depth_rendering(False)
        return {
            "depth": np.array(depth, dtype=np.float32),
            "width": sm._camera_width,
            "height": sm._camera_height,
        }
    except Exception:
        return {"depth": None, "width": 0, "height": 0}


def _read_force_torque(sm: SensorManager) -> Dict[str, Any]:
    """Read contact forces from MuJoCo contact data."""
    data = sm._data
    contacts: List[Dict[str, Any]] = []
    for i in range(data.ncon):
        contact = data.contact[i]
        force = np.zeros(6, dtype=np.float64)
        try:
            import mujoco
            mujoco.mj_contactForce(sm._model, data, i, force)
        except Exception:
            pass
        contacts.append({
            "geom1": int(contact.geom1),
            "geom2": int(contact.geom2),
            "force": force[:3].tolist(),
            "torque": force[3:].tolist(),
        })
    return {"contacts": contacts, "num_contacts": len(contacts)}


_READERS = {
    "imu": _read_imu,
    "joint_encoder": _read_joint_encoder,
    "rgb_camera": _read_rgb_camera,
    "depth_camera": _read_depth_camera,
    "force_torque": _read_force_torque,
}
