# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Built-in catalog of optional reference-motion / AMP packages.

Historically this was driven by an on-disk ``custom_mods/installs.txt``
manifest. That file was a foot-gun: it was deleted at the ``v1.0.0`` tag
and never restored, so on every install after that the catalog silently
resolved to *empty* -- the wizard rendered no checkboxes and PostSetupTask
cloned nothing. The shipped ``Go2_AMP`` canvas template (which hard-refs
``pack:amp_go2:...`` clips) was therefore broken on a fresh install with
no error anywhere (a textbook CLAUDE.md §8 "illusion of success").

The catalog is now **code-defined and always present** (no file
dependency). The install wizard renders one checkbox per entry inside the
same "Reference Motion Libraries" block as ``loco-mujoco``; PostSetupTask
shallow-clones the selected subset (plus every ``default_install`` entry
on the wizard-skipped path) into ``custom_mods/motions/<name>/``.

Public surface:

* :class:`CustomModEntry` -- one catalog entry (``key``, ``relative_path``,
  ``url``, ``ref``, ``required_for``, ``default_install``, ``target_dir``).
* :data:`BUILTIN_REFERENCE_MOTION_PACKS` -- the catalog itself.
* :func:`load_manifest_entries` -- return the catalog (kept under the old
  name so existing callers don't churn).
* :func:`entry_already_installed` -- cheap on-disk probe for a clone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

from unitport_sdk import Paths


__all__ = [
    "CustomModEntry",
    "BUILTIN_REFERENCE_MOTION_PACKS",
    "load_manifest_entries",
    "entry_already_installed",
]


@dataclass(frozen=True)
class CustomModEntry:
    """One reference-motion package in the built-in catalog.

    ``key`` is the basename of ``relative_path`` and acts as the stable
    identifier echoed into ``selections.backend.custom_mods``. ``target_dir``
    is the absolute on-disk clone destination resolved against the
    user's ``Paths.CUSTOM_MODS_DIR`` (overlay-resolved at module load).

    ``ref`` optionally pins a branch/tag/commit (empty → default branch).

    ``required_for`` is a tuple of ``"<backend>/<template>"`` strings
    pointing into ``custom_mods/canvas/<backend>/[<backend>] <template>.canvas.json``.
    An entry with a non-empty ``required_for`` is pre-checked in the
    install wizard and labelled so the user understands which shipped
    template would silently break without it.

    ``default_install`` marks entries that are part of the default install
    set: pre-checked in the wizard and cloned on the wizard-skipped
    ("legacy default") path. Required entries are always default-install.
    """

    key: str
    relative_path: str
    url: str
    ref: str = ""
    required_for: Tuple[str, ...] = field(default_factory=tuple)
    default_install: bool = False

    @property
    def target_dir(self) -> Path:
        return Path(Paths.CUSTOM_MODS_DIR) / self.relative_path

    @property
    def display_name(self) -> str:
        return self.key.replace("_", " ").replace("-", " ")

    @property
    def is_required(self) -> bool:
        return bool(self.required_for)


# ---------------------------------------------------------------------------
# Built-in catalog -- single source of truth.
#
# Every entry deploys under ``motions/`` so the motion library has one
# managed root (see ``scripts/training_motion/library.py``). Community AMP
# packs expose their clips at ``<pkg>/datasets/<subdir>/*.txt`` -- the path
# ``resolve_pack_ref`` reconstructs from a ``pack:<pkg>:<subdir>/<file>``
# token. ``amp_go2`` ships ``datasets/go2_motion/*.txt`` which the shipped
# ``Go2_AMP`` canvas template references, hence its ``required_for``.
#
# Adding a brand/model URL here does NOT violate CLAUDE.md §1: these are
# hot-pluggable reference-motion assets surfaced in the install wizard
# (application layer), not brand logic in the brand-agnostic core. The
# loco-mujoco URL has always lived alongside this code for the same reason.
# ---------------------------------------------------------------------------
BUILTIN_REFERENCE_MOTION_PACKS: Tuple[CustomModEntry, ...] = (
    CustomModEntry(
        key="amp_go2",
        relative_path="motions/amp_go2",
        url="https://github.com/ak1raljl/amp_go2.git",
        required_for=("isaac_lab/Go2_AMP",),
        default_install=True,
    ),
    CustomModEntry(
        key="AMP_for_hardware",
        relative_path="motions/AMP_for_hardware",
        url="https://github.com/escontra/AMP_for_hardware.git",
    ),
    CustomModEntry(
        key="MetalHead",
        relative_path="motions/MetalHead",
        url="https://github.com/inspirai/MetalHead.git",
    ),
    CustomModEntry(
        key="rl_amp",
        relative_path="motions/rl_amp",
        url="https://github.com/fan-ziqi/rl_amp.git",
    ),
    CustomModEntry(
        key="lifelike-agility-and-play",
        relative_path="motions/lifelike-agility-and-play",
        url="https://github.com/Tencent-RoboticsX/lifelike-agility-and-play.git",
    ),
    CustomModEntry(
        key="IsaacGymEnvs",
        relative_path="motions/IsaacGymEnvs",
        url="https://github.com/isaac-sim/IsaacGymEnvs.git",
    ),
)


def load_manifest_entries() -> List[CustomModEntry]:
    """Return the built-in reference-motion catalog.

    Kept under the historical name so the wizard and PostSetupTask call
    sites don't churn. The catalog is code-defined and always non-empty;
    there is no on-disk manifest to read or fail to read.
    """
    return list(BUILTIN_REFERENCE_MOTION_PACKS)


def entry_already_installed(entry: CustomModEntry) -> bool:
    """Return True iff the entry's target dir already contains a clone.

    Probe is intentionally cheap: presence of ``.git`` (full clone) OR
    of ``HEAD`` (the bare-clone shape git uses when ``--depth`` was
    promoted to a treeless mirror). A non-empty directory without those
    markers is NOT treated as installed -- it may be a stale leftover
    from a previous failed clone, and the post-setup step will error
    loudly if it tries to clone over a non-empty target.
    """
    target = entry.target_dir
    return (target / ".git").exists() or (target / "HEAD").exists()
