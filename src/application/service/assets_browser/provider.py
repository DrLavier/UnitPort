# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Asset-browser provider protocols — the read/write seam for the UI.

The Resources UI depends only on these ``Protocol``s (structural typing,
no inheritance required) plus the DTOs in :mod:`model`. The concrete
implementation lives in :mod:`stub_provider` and is obtained via
:func:`factory.get_asset_browser_provider`.

Splitting the seam into three roles keeps the backend round honest about
what is *real* now vs *deferred*:

* :class:`AssetBrowserProvider` — read side. ``list_*`` and ``probe_clip``
  are real today (they adapt the existing ``ResourceManager`` / motion
  ``library`` / ``registers.robots``, and ``probe_clip`` loads the clip for
  its true frame count + fps).
* :class:`SegmentWriter` — write side for clip segments. **Real today** —
  persists under ``USER_CONFIG_DIR`` via the SDK ``DataManager``.
* :class:`ClassifyWriter` — manual asset classification. Stub today
  (records the choice but does not yet route the asset).
"""

from __future__ import annotations

from typing import List, Optional, Protocol, Tuple, runtime_checkable

from .model import AssetCategory, ClipRow, PackageRow, SegmentRow


@runtime_checkable
class AssetBrowserProvider(Protocol):
    """Read side: enumerate categories → packages → clips → segments."""

    def list_categories(self) -> List[AssetCategory]:
        ...

    def list_packages(self, category: AssetCategory) -> List[PackageRow]:
        ...

    def list_clips(self, package_id: str) -> List[ClipRow]:
        """Clips inside a Motion package. ``[]`` for non-clip-capable
        packages (Policy / Model / Canvas)."""
        ...

    def list_segments(self, clip_ref: str) -> List[SegmentRow]:
        ...

    def probe_clip(self, clip_ref: str) -> Tuple[int, float]:
        """``(n_frames, fps)`` for the Clip Motion Editor's timeline range.

        Loads the clip (cached) to report its true frame count + fps.
        Raises (CLAUDE.md §8) on a missing / unparsable clip — the caller
        (the editor) wraps this and degrades to a fallback range on
        failure rather than the provider returning a misleading sentinel.
        """
        ...

    def suggest_sku(self, clip_ref: str) -> Optional[str]:
        """Best render robot for ``clip_ref`` = max IR-role overlap.

        Robot SKUs are opaque ids, so the editor cannot guess the right
        robot from the clip path. This loads the clip and returns the
        registered SKU whose MJCF joints overlap the clip's IR roles the
        most (``None`` if nothing overlaps / the clip can't load). Picking
        by overlap guarantees the default selection actually renders
        instead of failing ``matched==0``.
        """
        ...

    def clip_role_overlap(self, clip_ref: str, sku: str) -> Optional[Tuple[int, int]]:
        """``(overlap, clip_role_count)`` between ``clip_ref``'s IR roles and ``sku``'s.

        The Clip Motion Editor's clip picker uses the ratio to flag a clip the
        bound render robot mostly lacks joints for (``danger_zone``). ``None``
        when the clip can't load or the robot has no joint table. Cached per
        ``(clip_ref, sku)`` in the implementation.
        """
        ...

    def package_folder(self, package_id: str) -> Optional[str]:
        """The package's on-disk folder (``None`` if it has none).

        Powers "Open folder" for any card with files: downloads, scanned
        community packs, canvas templates. Registry models return ``None``.
        """
        ...

    def remove_package(self, package_id: str) -> None:
        """Delete a package's files (downloads + install-bundled packs).

        Raises (CLAUDE.md §8) when the package is not safely removable
        (registry models, shipped canvases) — never a silent no-op.
        """
        ...


@runtime_checkable
class SegmentWriter(Protocol):
    """Write side for clip segments (user state under USER_CONFIG_DIR)."""

    def add_segment(
        self,
        clip_ref: str,
        *,
        name: str,
        start_frame: int,
        end_frame: int,
        task_tag: str = "",
    ) -> SegmentRow:
        ...

    def rename_segment(self, clip_ref: str, segment_id: str, name: str) -> SegmentRow:
        ...

    def set_segment_tag(self, clip_ref: str, segment_id: str, task_tag: str) -> SegmentRow:
        ...

    def delete_segment(self, clip_ref: str, segment_id: str) -> None:
        ...


@runtime_checkable
class ClassifyWriter(Protocol):
    """Manual classification for assets the downloader could not auto-detect."""

    def classify_package(
        self, package_id: str, *, asset_kind: str, robot_family: str
    ) -> None:
        ...


__all__ = [
    "AssetBrowserProvider",
    "SegmentWriter",
    "ClassifyWriter",
]
