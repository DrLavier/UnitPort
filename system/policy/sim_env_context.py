from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass(eq=False)
class SimEnvContext:
    """Lightweight wrapper for a MuJoCo model+data pair and runtime hooks.

    All methods are stubs in Circle 3. Real implementations are provided
    by the CLI or adapter in Circle 5.
    """

    mj_model: Any                    # mujoco.MjModel — typed Any, no hard import
    mj_data: Any                     # mujoco.MjData
    joint_names: List[str]
    control_frequency_hz: float
    adapter: Optional[Any] = None

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
