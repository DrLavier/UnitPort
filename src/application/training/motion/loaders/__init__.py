# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Motion loaders — ``format_id → ReferenceMotionContract``.

Three formats are supported out of the box:

- ``unitport_npy`` — legacy ``.npy`` rows of joint positions; produces a
  tracking-only :class:`MotionClip` with no AMP payload.
- ``amp_legged_gym`` — DeepMimic-style 61-float JSON/txt (root pose +
  joints + toes + velocities). Produces an AMP-capable clip and reorders
  legs FR/FL/RR/RL → FL/FR/RL/RR to match UnitPort's canonical ordering.
- ``loco_mujoco_unitree_h1`` — loco-mujoco trajectory ``.npz`` recorded
  against the 19-DoF Unitree H1 (``h1.xml``) environment. Adapter is
  dependency-free (no ``jax`` / ``flax`` imports) — see ``loco_mujoco.py``
  for the upstream key-layout pin.

Loaders return a :class:`ReferenceMotionContract` that wraps the
:class:`MotionClip` payload with the metadata required to route it to
the right consumer (``consumption_mode``) and to record where it came
from (``source_info``). Callers that only need the numeric payload
extract it via ``contract.clip``.

Registration is explicit
------------------------
Built-in loaders register at package-import time via the two
:func:`register_loader` calls below — never via class side-effects
(``__init_subclass__`` hooks, decorators). This keeps "which loaders
are built-in" greppable in a single place: ``grep "register_loader("
src/application/training/motion/loaders/__init__.py`` lists the set.

External plugins call :func:`register_loader` after importing this
subpackage; :func:`get_loader` is the unified factory entry point.
"""
from __future__ import annotations

from application.training.motion.loaders.amp_legged_gym import AMPLegGymLoader
from application.training.motion.loaders.augment import mirror_motion_clip
from application.training.motion.loaders.base import (
    LOADER_REGISTRY,
    LoaderError,
    MotionLoader,
    get_loader,
    list_loader_formats,
    register_loader,
)
from application.training.motion.loaders.loco_mujoco import (
    LocoMujocoUnitreeH1Loader,
)
from application.training.motion.loaders.npy import NpyLoader


# Explicit factory registration — see the module docstring for why this
# is preferred over class-side-effect (decorator / __init_subclass__).
register_loader(NpyLoader)
register_loader(AMPLegGymLoader)
register_loader(LocoMujocoUnitreeH1Loader)


__all__ = [
    "MotionLoader",
    "LoaderError",
    "LOADER_REGISTRY",
    "register_loader",
    "list_loader_formats",
    "get_loader",
    "NpyLoader",
    "AMPLegGymLoader",
    "LocoMujocoUnitreeH1Loader",
    "mirror_motion_clip",
]
