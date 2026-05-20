"""application.service.resources — third-party asset downloader & registry.

This package powers the Sidebar > Resources panel: users add a GitHub repo
URL / GitHub Release URL / HuggingFace repo_id and the asset is downloaded
into ``Paths.CUSTOM_MODS_DIR`` under a kind-routed subtree, with the source
metadata recorded in ``custom_mods/.resources.json`` so the download is
reproducible across machines (commit ``custom_mods/.resources.json`` and
the resulting subtree to share).

Public surface (only these names should be imported by callers):

- :class:`ResourceKind` — ``motion`` | ``policy``.
- :class:`TransportKind` — ``github_clone`` | ``github_release`` | ``huggingface``.
- :class:`ResourceState` — ``downloading`` | ``local`` | ``error`` | ``missing``.
- :class:`ResourceSource` — frozen dataclass describing where a resource came
  from (url, revision, subpath/allow_patterns).
- :class:`ResourceEntry` — one record in ``.resources.json``.
- :class:`ResourceManager` — singleton (``get_resource_manager()``) that owns
  the index, queues downloads, and exposes the merged "indexed + scanned" view.
- :class:`PolicyPackScanner` — scans ``custom_mods/policies/<pack>/manifest.yaml``.

The scattered scanner for motion packs (``scripts.training_motion.library``)
is **NOT** re-implemented here; ``ResourceManager.list_motion_packs()``
delegates to it so the Training Motion Node and the Resources Panel see the
exact same set of packs.
"""

from __future__ import annotations

from .index import (
    ResourceEntry,
    ResourceIndex,
    ResourceKind,
    ResourceSource,
    ResourceState,
    TransportKind,
)
from .manager import ResourceManager, get_resource_manager
from .scanner import PolicyPack, PolicyPackScanner, scan_policy_packs


__all__ = [
    "ResourceKind",
    "TransportKind",
    "ResourceState",
    "ResourceSource",
    "ResourceEntry",
    "ResourceIndex",
    "ResourceManager",
    "get_resource_manager",
    "PolicyPack",
    "PolicyPackScanner",
    "scan_policy_packs",
]
