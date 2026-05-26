# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Configuration-dependent actuator models (remotized PD).

Currently exposes :class:`TorqueLookupTable` — the canonical loader/evaluator
for the v1 torque-lookup schema (see ``torque_lookup_v1.yaml`` alongside this
module). The table is the single source of truth shared by both engines: the
IsaacLab compiler converts it to a ``RemotizedPDActuatorCfg.joint_parameter_lookup``
array, and the MuJoCo-side ``PDController`` reads it for its per-step torque
clamp.
"""

from .torque_lookup import TorqueLookupTable, TorqueLookupSchemaError

__all__ = ["TorqueLookupTable", "TorqueLookupSchemaError"]
