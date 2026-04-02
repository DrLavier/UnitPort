from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(eq=False)
class SimEnvContext:
    """Lightweight wrapper for a MuJoCo model+data pair and runtime hooks.

    All methods are stubs in Circle 3. Real implementations are provided
    by the CLI or adapter in Circle 5.

    Phase 6 additions:
    - ``sensor_manager``: optional :class:`SensorManager` for dynamic sensor
      activation/deactivation per SkillManifest.required_sensors.
    - ``overlay_info``: mutable dict for viewer overlay rendering (skill name,
      transition state, input mode).
    """

    mj_model: Any                    # mujoco.MjModel — typed Any, no hard import
    mj_data: Any                     # mujoco.MjData
    joint_names: List[str]
    control_frequency_hz: float
    adapter: Optional[Any] = None
    sensor_manager: Optional[Any] = None   # SensorManager instance (lazy)
    overlay_info: Dict[str, str] = field(default_factory=dict)

    def sim_step(self) -> None:
        """Advance simulation by one physics step."""
        raise NotImplementedError("sim_step must be provided by the CLI or adapter")

    def reset(self) -> None:
        """Reset simulation state to initial conditions."""
        raise NotImplementedError("reset must be provided by the CLI or adapter")

    def render(self) -> None:
        """Render the current frame (viewer or offscreen)."""
        raise NotImplementedError("render must be provided by the CLI or adapter")

    def is_terminated(self) -> bool:
        """Phase 1 default: episode never terminates autonomously."""
        return False

    # ── Phase 6: sensor management ────────────────────────────────────────

    def configure_sensors(self, required_sensors: List[str]) -> None:
        """Activate sensors for the current skill's requirements.

        Creates a SensorManager on first call.  Subsequent calls reconfigure
        the active sensor set without recreating the manager.
        """
        if self.sensor_manager is None:
            try:
                from src.system.sim.sensor_manager import SensorManager
                self.sensor_manager = SensorManager(
                    self.mj_model, self.mj_data, self.joint_names,
                )
            except Exception:
                return
        self.sensor_manager.configure(required_sensors)

    def get_sensor_readings(self) -> Dict[str, Any]:
        """Read all active sensors.  Returns empty dict if no manager."""
        if self.sensor_manager is None:
            return {}
        return self.sensor_manager.read_all()

    # ── Phase 6: viewer overlay ───────────────────────────────────────────

    def set_overlay(self, **kwargs: str) -> None:
        """Update viewer overlay fields.

        Common keys: ``skill_name``, ``transition_state``, ``input_mode``.
        """
        self.overlay_info.update(kwargs)
