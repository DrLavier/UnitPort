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
- ``lafan_unitree_g1`` — LAFAN1 Unitree-G1 retargeting ``.csv`` (36 cols:
  root pose + 29 joint angles, 30 FPS, pose-only). Produces an AMP-capable
  clip; joint/root velocities are derived by finite differencing (body
  frame). See ``lafan_g1.py``.

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
from application.training.motion.loaders.lafan_g1 import LAFANUnitreeG1Loader
from application.training.motion.loaders.loco_mujoco import (
    LocoMujocoUnitreeH1Loader,
)
from application.training.motion.loaders.npy import NpyLoader


# Explicit factory registration — see the module docstring for why this
# is preferred over class-side-effect (decorator / __init_subclass__).
register_loader(NpyLoader)
register_loader(AMPLegGymLoader)
register_loader(LocoMujocoUnitreeH1Loader)
register_loader(LAFANUnitreeG1Loader)


# ---------------------------------------------------------------------------
# Format detection (file extension → format_id)
# ---------------------------------------------------------------------------

#: Map each motion file extension to its single registered loader. The
#: amp pipeline reads the format from the bytes (the file extension is
#: the discriminator today — each supported extension has exactly one
#: loader) rather than carrying a redundant ``motion_format`` schema
#: field across the canvas → spec → CLI chain. A new format that shares
#: an extension with an existing one (e.g. an H1 ``.csv``) would make
#: this 1:1 map ambiguous — at that point add a content probe here, not
#: a positional assumption. Loaders still fail loud on a column/shape
#: mismatch, so a wrong-skeleton file under the right extension cannot
#: load silently.
_EXT_TO_FORMAT = {
    ".csv": "lafan_unitree_g1",
    ".npz": "loco_mujoco_unitree_h1",
    ".npy": "unitport_npy",
    ".txt": "amp_legged_gym",
    ".json": "amp_legged_gym",
}


def detect_motion_format(path) -> str:
    """Resolve the registered ``format_id`` for a motion file by extension.

    Raises :class:`LoaderError` on an unknown extension — there is no
    silent default (CLAUDE.md §8). Used by the AMP data provider so a
    canvas can mix robot families (Go2 ``.txt`` vs G1 ``.csv``) without
    a per-node format field: the loader is selected from the file the
    user picked, and its own shape check rejects a mismatched skeleton.
    """
    from pathlib import Path as _Path

    ext = _Path(path).suffix.lower()
    fid = _EXT_TO_FORMAT.get(ext)
    if not fid:
        raise LoaderError(
            f"detect_motion_format: cannot resolve a motion loader for "
            f"{path!r} (extension {ext!r}). Known extensions: "
            f"{sorted(_EXT_TO_FORMAT)}. Add the extension → format_id "
            f"mapping to motion/loaders/__init__._EXT_TO_FORMAT and "
            f"register the matching loader."
        )
    return fid


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
    "LAFANUnitreeG1Loader",
    "mirror_motion_clip",
    "detect_motion_format",
]
