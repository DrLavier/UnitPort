# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Asset-browser DTOs — the pure-data contract the Resources UI renders against.

These dataclasses are the *frontend ↔ backend seam*: the Resources panel,
clip table, and crop/label dialog import only these (plus the provider
protocols in :mod:`provider`). The concrete data source — today a thin
adapter over the existing ``ResourceManager`` / motion ``library`` /
``registers.robots`` (:mod:`stub_provider`) — can be swapped wholesale in
the backend round without touching any widget, as long as it keeps
returning these shapes.

No Qt, no I/O here — just frozen records. ``state`` / ``transport`` carry
the *string* values of the resources-package enums (``ResourceState`` /
``TransportKind``) so this module stays decoupled from that package; the
card maps the strings back to theme slots at render time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class AssetCategory(str, Enum):
    """Top-level asset classes shown in the Resources panel.

    Mirrors the meaningful, shareable subtrees of ``custom_mods/`` plus the
    registry-backed robot models. Intentionally excludes ``nodes`` /
    ``runtime`` (app-internal code extensions, not managed data assets).
    """

    MOTION = "motion"
    POLICY = "policy"
    MODEL = "model"
    CANVAS = "canvas"


@dataclass(frozen=True)
class PackageRow:
    """One card in the Resources panel — a downloaded package, a robot
    model, or a canvas template.

    ``package_id`` is the stable handle the UI passes back to
    :meth:`provider.AssetBrowserProvider.list_clips`. For download-registry
    packages it equals the ``ResourceEntry.id`` (so progress signals, which
    carry the entry id, route to the right card); for everything else it is
    a category-prefixed synthetic id (``pack:…`` / ``model:…`` /
    ``canvas:…`` / ``policy:…``).
    """

    package_id: str
    category: AssetCategory
    name: str
    #: ``ResourceState`` value string: ``local`` / ``downloading`` /
    #: ``missing`` / ``error``. Non-download assets are always ``local``.
    state: str = "local"
    #: Ordered display rows for the expanded body (label-key → value). The
    #: card maps each key to an i18n label; insertion order is display order.
    provenance: Dict[str, str] = field(default_factory=dict)
    #: ``TransportKind`` value string (``github_clone`` / ``github_release``
    #: / ``huggingface``); ``None`` for non-download assets (no badge).
    transport: Optional[str] = None
    #: True only for Motion packages — the card embeds a clip table.
    clip_capable: bool = False
    #: ``ResourceEntry.id`` when this row maps to the download registry,
    #: else ``None`` (enables the progress bar + Classify… and routes
    #: download-progress signals to the card).
    raw_entry_id: Optional[str] = None
    #: Download error message when ``state == "error"``, else "".
    error: str = ""
    #: Whether the package resolves to an on-disk folder (→ "Open folder").
    #: True for downloads, scanned community packs, and canvas templates.
    openable: bool = False
    #: Whether the package's files can be deleted (→ "Remove"). True for
    #: downloads and install-bundled community packs; False for registry
    #: models (nothing to delete) and shipped canvas templates.
    removable: bool = False


@dataclass(frozen=True)
class ClipRow:
    """One reference-motion file inside a Motion package.

    ``clip_ref`` is the canonical reference string the Training Motion node
    already understands — either a ``pack:<package>:<subdir>/<file>`` ref
    (community AMP packs) or an absolute path (downloaded datasets like
    LAFAN1 whose layout has no ``datasets/`` subdir). Segment metadata is
    keyed by a hash of this string.
    """

    clip_ref: str
    name: str
    task_tag: str = ""
    #: Frame count; ``-1`` = unknown (the stub provider does not parse the
    #: file — the backend round fills this via ``MotionClip``).
    n_frames: int = -1
    #: Native fps; ``0.0`` = unknown (see ``n_frames``).
    fps: float = 0.0
    segment_count: int = 0


@dataclass(frozen=True)
class SegmentRow:
    """A user-defined sub-clip (in/out window + label) of a ClipRow.

    Persisted as user state under ``USER_CONFIG_DIR`` (see
    :mod:`stub_provider`). ``start_frame`` / ``end_frame`` are inclusive
    frame indices into the parent clip.
    """

    segment_id: str
    clip_ref: str
    name: str
    start_frame: int
    end_frame: int
    task_tag: str = ""


__all__ = [
    "AssetCategory",
    "PackageRow",
    "ClipRow",
    "SegmentRow",
]
