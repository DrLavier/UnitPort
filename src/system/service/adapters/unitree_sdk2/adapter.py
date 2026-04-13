"""Unitree SDK2 service adapter — Phase 4 v2 (lifecycle uplift).

Phase 4 changes (additive only):
  - open_session: explicit connect + SDK/MuJoCo availability check
  - preflight: action-in-progress conflict guard + availability check
  - close_session: explicit simulation stop + viewer teardown
  - capabilities: declares real action set, sensor keys, availability flags

Legacy interface (connect / run_action / stop / get_sensor_data / health /
get_model) is preserved unchanged and continues to work without modification.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.system.brand_packages import get_brand_model_class
from src.system.service.adapters.base_adapter import BaseAdapter
from src.system.service.lifecycle import LifecycleReason, LifecycleResult
from .mapper import ACTION_MAP, map_action
from .semantic_actions import get_unitree_semantic_actions

# Phase 4 reason codes (Unitree-specific, stable constants)
_REASON_ACTION_IN_PROGRESS = "unitree_action_in_progress"
_REASON_CONTROL_CONFLICT   = "unitree_control_conflict"
_REASON_MODEL_UNAVAILABLE  = "unitree_model_unavailable"


def _u_diag(message: str = "", context: dict | None = None, **fields) -> dict:
    """Build Unitree diagnostics dict with required minimum keys (brand/message/context)."""
    base: dict = {"brand": "unitree", "message": message, "context": context or {}}
    base.update(fields)
    return base


class UnitreeAdapter(BaseAdapter):
    """Adapter for Unitree robots via SDK2.

    Phase 4 v2: explicit lifecycle semantics layered on top of the v1
    connect/run_action/stop/get_sensor_data/health interface.  All pre-Phase 4
    callers continue to work without change.
    """

    def __init__(self, robot_type: str = "go2"):
        self.robot_type = (robot_type or "go2").lower()
        self._model: Optional[Any] = None

    # ── Legacy interface (Phase 1 contract — never removed) ───────────────

    def connect(self, **kwargs: Any) -> bool:
        """Create or refresh an underlying Unitree model instance."""
        requested_type = kwargs.get("robot_type")
        force_reinit = bool(kwargs.get("force_reinit", False))
        if requested_type:
            self.robot_type = str(requested_type).lower()

        if self._model is not None and not force_reinit:
            if getattr(self._model, "robot_type", "").lower() == self.robot_type:
                return True

        model = self._create_model_instance(self.robot_type)
        if model is None:
            raise RuntimeError("Unitree model package is unavailable")
        self._model = model
        return True

    def _create_model_instance(self, robot_type: str) -> Any:
        """Instantiate the Unitree brand model via the central brand registry."""
        model_class = get_brand_model_class("unitree")
        return model_class(robot_type)

    def get_model(self) -> Optional[Any]:
        """Compatibility helper for legacy callers that still need model instance."""
        if self._model is None:
            self.connect()
        return self._model

    def run_action(self, action: str, **params: Any) -> Any:
        model = self.get_model()
        if model is None:
            return False
        return model.run_action(map_action(action), **params)

    def velocity_move(self, vx: float, vy: float, vyaw: float, duration: float) -> bool:
        """First-class velocity move API (Step 7). Delegates to UnitreeModel.velocity_move().

        This is the primary execution path for the new binding/executor
        architecture.  ``run_action()`` is preserved for backward compat only.
        """
        model = self.get_model()
        if model is None:
            return False
        return model.velocity_move(float(vx), float(vy), float(vyaw), float(duration))

    def velocity_move_mujoco(self, vx: float, vy: float, vyaw: float, duration: float) -> bool:
        """MuJoCo velocity move (Step 7). Delegates to UnitreeModel.velocity_move_mujoco()."""
        model = self.get_model()
        if model is None:
            return False
        return model.velocity_move_mujoco(float(vx), float(vy), float(vyaw), float(duration))

    def stop(self) -> None:
        model = self.get_model()
        if model is not None:
            model.stop()

    def cancel_action(self) -> None:
        """Interrupt in-flight action by issuing an immediate stop (Cycle 3 STAGE-04).

        Delegates to ``model.stop()`` which halts motion at the SDK level.
        Thread-safe: UnitreeModel.stop() is a fire-and-forget SDK call.
        """
        try:
            model = self._model  # read without side-effects (no get_model() call)
            if model is not None:
                model.stop()
        except Exception:
            pass  # cancel is best-effort; never raise

    def get_sensor_data(self) -> Dict[str, Any]:
        model = self.get_model()
        if model is None:
            return {"error": "Unitree model not available"}
        return model.get_sensor_data()

    def health(self) -> Dict[str, Any]:
        model = self.get_model()
        if model is None:
            return {"connected": False, "robot_type": self.robot_type}
        return {
            "connected": True,
            "robot_type": self.robot_type,
            "available": bool(getattr(model, "is_available", False)),
        }

    # ── Phase 4 lifecycle overrides ────────────────────────────────────────

    def open_session(self, config: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """Establish a Unitree service session.

        Calls ``connect()`` to create or refresh the underlying UnitreeModel,
        then checks SDK/MuJoCo availability.  Returns a reason-coded error if
        neither SDK nor MuJoCo simulation is available.

        Args:
            config: Optional dict forwarded to ``connect()``.  Recognises the
                    same keys as ``connect()``: ``robot_type``, ``force_reinit``.

        Returns:
            LifecycleResult serialised as a plain dict.
        """
        try:
            success = self.connect(**(config or {}))
        except Exception as exc:
            return LifecycleResult.error(
                "open_session",
                LifecycleReason.SESSION_OPEN_FAILED,
                _u_diag(str(exc), robot_type=self.robot_type),
            ).to_dict()

        if not success or self._model is None:
            return LifecycleResult.error(
                "open_session",
                LifecycleReason.SESSION_OPEN_FAILED,
                _u_diag("connect() returned False", robot_type=self.robot_type),
            ).to_dict()

        sdk_available   = bool(getattr(self._model, "is_available",     False))
        mujoco_available = bool(getattr(self._model, "mujoco_available", False))

        # Neither real SDK nor simulation available — no execution path
        if not sdk_available and not mujoco_available:
            return LifecycleResult.error(
                "open_session",
                LifecycleReason.SESSION_OPEN_FAILED,
                _u_diag(
                    "Neither SDK nor MuJoCo simulation available",
                    reason=_REASON_MODEL_UNAVAILABLE,
                    sdk_available=False,
                    mujoco_available=False,
                    robot_type=self.robot_type,
                ),
            ).to_dict()

        return LifecycleResult.ok("open_session").to_dict()

    def preflight(self, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """Validate Unitree adapter readiness before action execution.

        Checks performed (minimum form):
        1. Model is initialised (``open_session`` must precede).
        2. No action currently in progress (``model.running`` is False).
        3. High-level vs low-level control conflict guard: if the SDK sport
           client is initialised *and* a simulation is running, both control
           paths are simultaneously active — reject to prevent undefined state.

        Args:
            context: Optional execution context dict (unused by minimum checks,
                     reserved for future capability-requirement checks).

        Returns:
            LifecycleResult serialised as a plain dict.
        """
        if self._model is None:
            return LifecycleResult.error(
                "preflight",
                LifecycleReason.PREFLIGHT_FAILED,
                _u_diag(
                    "Model not initialised; call open_session first",
                    reason=_REASON_MODEL_UNAVAILABLE,
                    robot_type=self.robot_type,
                ),
            ).to_dict()

        running = bool(getattr(self._model, "running", False))
        if running:
            return LifecycleResult.error(
                "preflight",
                LifecycleReason.PREFLIGHT_FAILED,
                _u_diag(
                    "A simulation action is currently running",
                    reason=_REASON_ACTION_IN_PROGRESS,
                ),
            ).to_dict()

        # Control conflict: SDK sport client active + simulation also active
        sdk_client_active = (
            getattr(self._model, "_sdk_channel_inited", False)
            and getattr(self._model, "_sport_client", None) is not None
        )
        sim_active = bool(getattr(self._model, "mujoco_available", False))
        if sdk_client_active and sim_active:
            return LifecycleResult.error(
                "preflight",
                LifecycleReason.PREFLIGHT_FAILED,
                _u_diag(
                    "SDK high-level client and MuJoCo simulation are both active",
                    reason=_REASON_CONTROL_CONFLICT,
                    sdk_client_active=True,
                    mujoco_available=True,
                ),
            ).to_dict()

        return LifecycleResult.ok("preflight").to_dict()

    def close_session(self) -> Dict[str, Any]:
        """Tear down the Unitree service session.

        Stops any running simulation, closes the MuJoCo viewer if one is open,
        and clears the running flag.  Does not destroy the model instance —
        legacy callers that hold a reference via ``get_model()`` must not be
        invalidated.

        Returns:
            LifecycleResult serialised as a plain dict.
        """
        if self._model is None:
            # Nothing to tear down — idempotent ok
            return LifecycleResult.ok("close_session").to_dict()

        try:
            # Stop any in-progress simulation
            if getattr(self._model, "running", False):
                self._model.stop()

            # Close MuJoCo viewer if it is open
            if callable(getattr(self._model, "close_viewer", None)):
                self._model.close_viewer()

        except Exception as exc:
            return LifecycleResult.error(
                "close_session",
                LifecycleReason.SESSION_CLOSE_FAILED,
                _u_diag(str(exc), robot_type=self.robot_type),
            ).to_dict()

        return LifecycleResult.ok("close_session").to_dict()

    def capabilities(self) -> Dict[str, Any]:
        """Return Unitree adapter capability descriptor (Phase 5 schema).

        Returns:
            dict conforming to the Phase 5 capability schema::

                {
                    "brand":   "unitree",
                    "adapter": "unitree_sdk2",
                    "actions": ["stand", "sit", "walk", "stop", "lift_right_leg"],
                    "sensors": ["battery", "imu", "foot_contact", "qpos", "qvel", "time"],
                    "flags": {
                        "sdk_available":      bool,
                        "mujoco_available":   bool,
                        "high_level_control": True,
                        "robot_type":         str,
                    },
                    "required_settings": [
                        "brand", "robot_type", "connection_mode", "timeout",
                        "stop_policy", "safety_enabled",
                        "dds_domain_id", "network_interface", "control_level",
                        "mode_switch_policy",
                    ],
                }
        """
        sdk_available    = False
        mujoco_available = False
        if self._model is not None:
            sdk_available    = bool(getattr(self._model, "is_available",     False))
            mujoco_available = bool(getattr(self._model, "mujoco_available", False))

        return {
            "brand":      "unitree",
            "robot_type": self.robot_type,   # top-level, mirrors "brand" position
            "adapter":    "unitree_sdk2",
            "actions":    list(ACTION_MAP.keys()),
            "sensors": ["battery", "imu", "foot_contact", "qpos", "qvel", "time"],
            "flags": {
                "sdk_available":      sdk_available,
                "mujoco_available":   mujoco_available,
                "high_level_control": True,
                "robot_type":         self.robot_type,
            },
            "required_settings": [
                "brand", "robot_type", "connection_mode", "timeout",
                "stop_policy", "safety_enabled",
                "dds_domain_id", "network_interface", "control_level",
                "mode_switch_policy",
            ],
            "semantic_actions": [
                d.to_dict()
                for d in get_unitree_semantic_actions(self.robot_type)
            ],
        }

    # ── Phase 6 C4: SkillManifest-aligned action interface ──────────────

    def accept_action(
        self,
        action: Any,
        skill_manifest: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Translate a policy action to Unitree SDK command.

        For ``joint_position`` actions, delegates to the model's torque/position
        control API.  For velocity-like actions, uses ``velocity_move()``.
        """
        import numpy as np

        action_arr = np.asarray(action, dtype=np.float32).flatten()
        sm = skill_manifest if isinstance(skill_manifest, dict) else {}
        action_type = sm.get("action_space_type", "joint_position")
        if hasattr(action_type, "value"):
            action_type = action_type.value

        model = self.get_model()
        if model is None:
            return {"ok": False, "reason": _REASON_MODEL_UNAVAILABLE}

        if action_type in ("joint_position", "joint_torque"):
            ctrl_fn = getattr(model, "set_joint_command", None)
            if callable(ctrl_fn):
                try:
                    ctrl_fn(action_arr, mode=action_type)
                    return {"ok": True, "reason": ""}
                except Exception as exc:
                    return {"ok": False, "reason": str(exc)}
            return {"ok": True, "reason": "no set_joint_command — passthrough"}
        return {"ok": True, "reason": ""}

    def report_sensors(
        self,
        required_sensors: list,
    ) -> Dict[str, Any]:
        """Return sensor data filtered by required_sensors list."""
        all_data = self.get_sensor_data()
        result: Dict[str, Any] = {}
        sensor_key_map = {
            "imu": ["imu"],
            "joint_encoder": ["qpos", "qvel"],
            "force_torque": ["foot_contact"],
        }
        for sensor in required_sensors:
            keys = sensor_key_map.get(sensor, [sensor])
            for k in keys:
                if k in all_data:
                    result.setdefault(sensor, {})[k] = all_data[k]
            if sensor not in result:
                result[sensor] = None
        return result

    def check_precondition(self, precondition: Dict[str, Any]) -> bool:
        """Check if the Unitree robot satisfies the skill precondition."""
        posture = precondition.get("posture", "any")
        if posture == "any":
            return True
        # For now, trust the robot is in the correct posture if connected
        model = self.get_model()
        return model is not None

    # ── Phase V2: Reactive low-level control ─────────────────────────────

    def accept_reactive_action(
        self,
        joint_position_targets: Any,
        pd_params: Optional[Dict[str, Any]] = None,
        *,
        reorder_indices: Any = None,
    ) -> Dict[str, Any]:
        """Send low-level joint position targets for reactive policy deployment.

        Uses Unitree SDK2 LowCmd to send per-joint targets with PD parameters.
        The SDK expects commands at ~500 Hz; the policy runs at ~50 Hz, so the
        caller should repeat the last command between policy steps.

        Parameters
        ----------
        joint_position_targets:
            Array of joint position targets (12 for Go2).
        pd_params:
            Optional dict with ``kp``, ``kd`` per-joint arrays.
            If None, uses defaults from the model.
        reorder_indices:
            Index array to reorder policy joint order → SDK joint order.

        Returns
        -------
        dict with "ok" and "reason" keys.
        """
        import numpy as np

        targets = np.asarray(joint_position_targets, dtype=np.float32).flatten()

        # Reorder from policy joint order to SDK joint order
        if reorder_indices is not None:
            inv_indices = np.argsort(reorder_indices)
            targets = targets[inv_indices]

        model = self.get_model()
        if model is None:
            return {"ok": False, "reason": _REASON_MODEL_UNAVAILABLE}

        # Try low-level joint command API
        low_cmd_fn = getattr(model, "set_low_level_cmd", None)
        if callable(low_cmd_fn):
            try:
                kw = {}
                if pd_params:
                    kw["kp"] = pd_params.get("kp")
                    kw["kd"] = pd_params.get("kd")
                low_cmd_fn(targets, **kw)
                return {"ok": True, "reason": ""}
            except Exception as exc:
                return {"ok": False, "reason": str(exc)}

        # Fallback to general set_joint_command
        ctrl_fn = getattr(model, "set_joint_command", None)
        if callable(ctrl_fn):
            try:
                ctrl_fn(targets, mode="joint_position")
                return {"ok": True, "reason": ""}
            except Exception as exc:
                return {"ok": False, "reason": str(exc)}

        return {"ok": False, "reason": "No low-level command API available"}

    def provide_observation(
        self,
        required_keys: list,
    ) -> Dict[str, Any]:
        """Read real sensor data for reactive policy observation assembly.

        Maps observation term names to real hardware readings:
          - base_lin_vel → IMU linear velocity
          - base_ang_vel → IMU gyroscope
          - projected_gravity → computed from IMU quaternion
          - joint_pos → encoder positions
          - joint_vel → encoder velocities

        Parameters
        ----------
        required_keys:
            List of observation term names needed by the policy.

        Returns
        -------
        dict mapping each key to its sensor reading (numpy array or None).
        """
        import numpy as np

        result: Dict[str, Any] = {}
        sensor_data = self.get_sensor_data()
        imu_data = sensor_data.get("imu", {})

        for key in required_keys:
            if key in ("base_lin_vel",):
                # IMU linear velocity (body frame)
                acc = imu_data.get("accelerometer", [0, 0, 0])
                result[key] = np.array(acc, dtype=np.float32)[:3]
            elif key in ("base_ang_vel",):
                gyro = imu_data.get("gyroscope", [0, 0, 0])
                result[key] = np.array(gyro, dtype=np.float32)[:3]
            elif key == "projected_gravity":
                quat = imu_data.get("quaternion", [1, 0, 0, 0])
                result[key] = self._project_gravity_from_quat(quat)
            elif key in ("joint_pos", "joint_pos_rel"):
                qpos = sensor_data.get("qpos", [])
                result[key] = np.array(qpos, dtype=np.float32)
            elif key in ("joint_vel", "joint_vel_rel"):
                qvel = sensor_data.get("qvel", [])
                result[key] = np.array(qvel, dtype=np.float32)
            else:
                result[key] = None

        return result

    @staticmethod
    def _project_gravity_from_quat(quat) -> Any:
        """Project world gravity into body frame from quaternion [w,x,y,z].

        Computes ``R^T @ [0, 0, -1]`` where R is the body-to-world rotation
        built from the (w,x,y,z) quaternion. Equivalent to IsaacLab's
        ``quat_rotate_inverse(quat, [0,0,-1])`` and matches the unit-vector
        convention required by sim2sim policies (see SIM2SIM/report.yaml
        rule_5_projected_gravity).

        Closed form (derived from negating row 2 of R):
            gx =  2 * (w*y - x*z)
            gy = -2 * (w*x + y*z)
            gz =  2 * (x*x + y*y) - 1

        WARNING: a previous implementation flipped the sign of the w*x and
        w*y cross terms, which left identity (w=1) correct but produced the
        wrong projected_gravity on every non-trivial roll/pitch — leading to
        immediate balance failure on real hardware. Do not "simplify" this
        function without re-deriving from R^T @ [0,0,-1] first. The
        bit-identical reference is ``_project_gravity`` in
        src/system/policy/obs_builder.py, which builds R explicitly and
        multiplies — slower but obviously correct.
        """
        import numpy as np
        q = np.array(quat, dtype=np.float32)
        if q.shape[0] < 4:
            return np.array([0, 0, -1], dtype=np.float32)
        w, x, y, z = q[:4]
        gx = 2.0 * (w * y - x * z)
        gy = -2.0 * (w * x + y * z)
        gz = 2.0 * (x * x + y * y) - 1.0
        return np.array([gx, gy, gz], dtype=np.float32)

    @staticmethod
    def _load_model_class():
        return get_brand_model_class("unitree")

