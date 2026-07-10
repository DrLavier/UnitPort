# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""assets_browser — data seam between the Resources UI and asset sources.

The Resources panel, clip table, and crop/label dialog import only from
this package: the DTOs (:mod:`model`), the seam protocols (:mod:`provider`),
and the factory. The concrete source (:mod:`stub_provider`) adapts the
existing ``ResourceManager`` / motion ``library`` / ``registers.robots``
today; the backend round replaces it behind :func:`get_asset_browser_provider`.
"""

from __future__ import annotations

from .factory import get_asset_browser_provider
from .model import AssetCategory, ClipRow, PackageRow, SegmentRow
from .motion_picker import (
    list_downloaded_motion_entries,
    list_pickable_motion_entries,
)
from .provider import AssetBrowserProvider, ClassifyWriter, SegmentWriter

__all__ = [
    "AssetCategory",
    "PackageRow",
    "ClipRow",
    "SegmentRow",
    "AssetBrowserProvider",
    "SegmentWriter",
    "ClassifyWriter",
    "get_asset_browser_provider",
    "list_downloaded_motion_entries",
    "list_pickable_motion_entries",
]
