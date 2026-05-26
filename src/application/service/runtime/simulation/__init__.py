# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""MuJoCo + sensor + PD-controller runtime — Phase 1 port from DEMO.

Public surface intended for callers (sim2sim Phase 3, training Phase 4):

- ``PDController`` (.pd_controller) — Isaac ImplicitActuator + DCMotor envelope.
- ``SensorManager`` (.sensor_manager) — IMU / joint encoder / camera readouts.
- ``MjActor`` / ``MjField`` / ``MjSimEnv`` (.mujoco) — actor + scene + env.

The facade names ``PDController`` and ``SensorManager`` are resolved **lazily**
via PEP 562 ``__getattr__`` so that merely importing a submodule under this
package (e.g. ``application.service.runtime.simulation.mujoco.review_session``
from a UI widget at startup) does not pull numpy/mujoco. numpy is provisioned
by ``ProvisioningTask`` in Stage 2; Stage 1 must stay light per CLAUDE.md §3.2.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["PDController", "SensorManager"]

if TYPE_CHECKING:  # static type-checkers only; no runtime import
    from .pd_controller import PDController
    from .sensor_manager import SensorManager


def __getattr__(name: str) -> Any:
    if name == "PDController":
        from .pd_controller import PDController as _PDController
        return _PDController
    if name == "SensorManager":
        from .sensor_manager import SensorManager as _SensorManager
        return _SensorManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
