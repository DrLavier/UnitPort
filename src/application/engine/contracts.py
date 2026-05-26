# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Engine runtime contracts (minimal Phase 3 port).

DEMO ``src/system/engine/contracts.py`` carries an extensive set of
constants — ExecutionPath, DiagnosticsKey, SafetyDecision, etc. — for the
engine/runner / behavior-node infrastructure. RELEASE Phase 3 only needs
the four ``MUJOCO_SIM2SIM_*`` keys used by ``PolicyRunner.run_episode``'s
per-step telemetry hook. The rest of the constants land back in when
``application/engine/`` is properly populated.

Ported strings (DEMO source-of-truth values):
    DiagnosticsKey.MUJOCO_SIM2SIM_STEP_INDEX  = "mujoco_sim2sim_step_index"
    DiagnosticsKey.MUJOCO_SIM2SIM_LAST_ACTION = "mujoco_sim2sim_last_action"
    DiagnosticsKey.MUJOCO_SIM2SIM_LAST_OBS    = "mujoco_sim2sim_last_obs"
    DiagnosticsKey.MUJOCO_SIM2SIM_COMMAND     = "mujoco_sim2sim_command"
"""
from __future__ import annotations


class DiagnosticsKey:
    """Stable key constants for diagnostics / telemetry payloads."""

    # Phase 3 sim2sim per-step telemetry (read by PolicyRunner.run_episode)
    MUJOCO_SIM2SIM_STEP_INDEX = "mujoco_sim2sim_step_index"
    MUJOCO_SIM2SIM_LAST_ACTION = "mujoco_sim2sim_last_action"
    MUJOCO_SIM2SIM_LAST_OBS = "mujoco_sim2sim_last_obs"
    MUJOCO_SIM2SIM_COMMAND = "mujoco_sim2sim_command"


__all__ = ["DiagnosticsKey"]
