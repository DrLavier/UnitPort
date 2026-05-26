# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""application.physics — Engine-agnostic gain math for the (omega_n, zeta) PD framework.

This package holds the mass-matrix-adaptive PD solvers that translate the
canonical ``(omega_n, zeta)`` parameterization (from the canvas
``ActuatorPDNode`` or family defaults in ``registers.pd_groups``) into
the engine-specific ``kp``/``kd`` arrays that:

* :mod:`application.training.envs.generic_mujoco_env` uses for Python-side
  PD on top of MJCF ``<motor>`` actuators;
* :mod:`application.training.isaac_lab.env_cfg_compiler` emits as the
  ``stiffness``/``damping`` dicts inside ``ImplicitActuatorCfg``;
* the bundle finalizer derives once at export time and stamps into
  ``deploy_contract`` so loaders on any machine can replay without
  reaching back into local state (RELEASE/CLAUDE.md §1.9).

Design invariants (rule §3 in the mjcf-usd-pd plan):

1. ``kp`` / ``kd`` literal numbers appear ONLY inside the two
   ``*_gain_solver.py`` files. Anywhere else uses the solver output.
2. No engine reads the other engine's derived gains. Both engines read
   the same :class:`PDParam` (the source) and call their own solver.
3. PD parameterization is keyed by IR role / group, never by physical
   joint name. The resolver expands group regex → IR roles at compile
   time, then the physical-name mapping happens at the engine boundary.
4. All failure modes raise (no silent fallback) — see RELEASE/CLAUDE.md §1.8.
"""

from __future__ import annotations

from .pd_param import (
    PDGroup,
    PDParam,
    JointPDGains,
)
from .joint_group_resolver import (
    expand_groups_to_joints,
    UnmappedRoleError,
)


__all__ = [
    "PDGroup",
    "PDParam",
    "JointPDGains",
    "expand_groups_to_joints",
    "UnmappedRoleError",
]
