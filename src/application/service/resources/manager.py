"""ResourceManager — singleton glue between UI, index, transports, scanners.

Responsibilities:

- Validate user input from ``AddResourceDialog`` (URL shape, repo_id form).
- Compute the destination subtree under ``Paths.CUSTOM_MODS_DIR`` based
  on the routing rules (transport × asset_kind → ``archives``/``motions``/
  ``policies``).
- Create the index entry up-front (state=downloading) and submit a
  :class:`DownloadResourceTask` to ``TasksManager``.
- Expose merged views (indexed entries + freshly scanned packs) for the
  Resources panel and the policy selector.

The manager is intentionally thin: each operation is one short method,
and complex logic (transport-specific behaviour, scanning, on-disk shape)
lives in the dedicated submodules.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from unitport_sdk import Paths, get_tasks_manager, log_warning

from ..signals import get_app_signals
from .index import (
    ResourceEntry,
    ResourceIndex,
    ResourceKind,
    ResourceSource,
    ResourceState,
    TransportKind,
)
from .scanner import PolicyPack, PolicyPackScanner, scan_policy_packs
from .tasks import DownloadResourceTask
from .transports import (
    repo_name_from_url,
    safe_dir_name,
)
from .transports.huggingface import _normalize_repo_id as _hf_normalize_repo_id


# Routing: (transport, kind) -> subdir under CUSTOM_MODS_DIR.
# git_clone defaults to archives/ because the existing
# scripts/training_motion/library scanner already treats archives/ as a
# motion source, and AMP-style repos historically land there. Release +
# HuggingFace go to the kind-specific subtree.
def _kind_subdir(kind: ResourceKind) -> str:
    return "motions" if kind == ResourceKind.MOTION else "policies"


def _compute_dest_relpath(
    transport: TransportKind,
    kind: ResourceKind,
    source: ResourceSource,
) -> str:
    """Return the path-relative-to-CUSTOM_MODS_DIR for this asset."""
    if transport == TransportKind.GITHUB_CLONE:
        leaf = repo_name_from_url(source.url)
        return f"archives/{leaf}"
    if transport == TransportKind.GITHUB_RELEASE:
        # Use the revision (tag) when present; otherwise use the repo name
        # plus 'latest'. We don't try to reach the network here — that's
        # the transport's job during fetch.
        leaf_repo = repo_name_from_url(source.url)
        tag = source.revision.strip() or "latest"
        leaf = safe_dir_name(f"{leaf_repo}__{tag}")
        return f"{_kind_subdir(kind)}/{leaf}"
    if transport == TransportKind.HUGGINGFACE:
        # Normalize HF URL to bare ``owner/name`` first so the leaf reads
        # as ``owner__name`` instead of including https:// noise.
        try:
            repo_id = _hf_normalize_repo_id(source.url)
        except Exception:
            repo_id = source.url
        leaf = safe_dir_name(repo_id, fallback="hf_resource")
        return f"{_kind_subdir(kind)}/{leaf}"
    raise ValueError(f"unsupported transport: {transport!r}")


@dataclass(frozen=True)
class QueueResult:
    """Outcome of ``ResourceManager.queue_download``."""

    entry_id: str
    task_id: int


class ResourceManager:
    """Singleton facade for the resources package.

    Construct via :func:`get_resource_manager`. Tests can instantiate
    directly with a custom ``ResourceIndex`` for isolation.
    """

    def __init__(self, index: Optional[ResourceIndex] = None) -> None:
        self._index = index if index is not None else ResourceIndex()
        self._lock = threading.RLock()

    # ---- index / view API ----

    @property
    def index(self) -> ResourceIndex:
        return self._index

    def list_entries(self) -> List[ResourceEntry]:
        """All recorded entries, regardless of asset kind."""
        return self._index.list_all()

    def list_motion_entries(self) -> List[ResourceEntry]:
        return self._index.list_by_kind(ResourceKind.MOTION)

    def list_policy_entries(self) -> List[ResourceEntry]:
        return self._index.list_by_kind(ResourceKind.POLICY)

    def refresh_scan_state(self) -> None:
        """Reconcile index entries with their on-disk state.

        Entries whose ``absolute_path()`` no longer exists are flipped to
        ``state="missing"``; entries that exist but were marked
        ``downloading`` from a previous interrupted session and now have
        actual content are flipped back to ``local``. Pure metadata
        refresh — does not touch the asset tree.
        """
        for entry in self._index.list_all():
            disk = entry.absolute_path()
            disk_present = disk.exists() and disk.is_dir() and any(disk.iterdir())
            if entry.state == ResourceState.LOCAL and not disk_present:
                self._index.update(entry.id, state=ResourceState.MISSING)
            elif entry.state == ResourceState.MISSING and disk_present:
                self._index.update(entry.id, state=ResourceState.LOCAL)
            elif entry.state == ResourceState.DOWNLOADING and disk_present:
                # Crashed mid-download last session; user can decide to
                # keep or remove. We mark "local" to surface it in the
                # UI but the size_bytes was never written so the card
                # will display "??" until they re-scan.
                self._index.update(entry.id, state=ResourceState.LOCAL)

    # ---- policy scan ----

    def scan_policy_packs(self) -> List[PolicyPack]:
        """Fresh disk walk under ``custom_mods/policies/``."""
        return scan_policy_packs()

    # ---- motion scan (delegated) ----

    def list_motion_packs(self):
        """Return ``MotionPack`` entries from the existing scanner.

        Delegates to ``scripts.training_motion.library.list_motion_packs``
        so the Resources panel and the Training Motion canvas node agree
        on what packs are visible. Late import to avoid pulling the
        training stack at SDK import time.
        """
        try:
            from scripts.training_motion.library import list_motion_packs
        except ImportError as e:
            log_warning(f"motion library not importable: {e}")
            return []
        return list_motion_packs()

    # ---- queue a new download ----

    def queue_download(
        self,
        *,
        asset_kind: ResourceKind,
        transport: TransportKind,
        name: str,
        source: ResourceSource,
    ) -> QueueResult:
        """Persist a new entry + submit the download task.

        The entry is committed to ``.resources.json`` BEFORE the task
        runs so a crash mid-download leaves a recoverable state on disk.
        ``resource_added`` fires synchronously after the index write so
        the UI can render a card before the first progress tick.
        """
        if not source.url.strip():
            raise ValueError("source.url is required")
        local_rel = _compute_dest_relpath(transport, asset_kind, source)
        display_name = (name or "").strip() or _default_name(transport, source)
        entry = ResourceEntry.new(
            asset_kind=asset_kind,
            transport=transport,
            name=display_name,
            source=source,
            local_path=local_rel,
        )
        with self._lock:
            self._index.add(entry)
        get_app_signals().resource_added.emit(entry.id)

        task = DownloadResourceTask(entry, self._index)
        task_id = get_tasks_manager().submit(task)
        return QueueResult(entry_id=entry.id, task_id=task_id)

    # ---- remove an entry (and optionally its on-disk tree) ----

    def remove_entry(self, entry_id: str, *, delete_files: bool = True) -> None:
        """Delete an entry from the index; optionally remove its tree on disk.

        ``delete_files=False`` is useful when the user manually deleted
        the directory and just wants to clear the stale index row.
        """
        with self._lock:
            entry = self._index.get(entry_id)
            if entry is None:
                return
            self._index.remove(entry_id)
        if delete_files:
            disk = entry.absolute_path()
            if disk.exists():
                import shutil
                try:
                    shutil.rmtree(disk, ignore_errors=True)
                except OSError:
                    pass
        get_app_signals().resource_removed.emit(entry_id)


def _default_name(transport: TransportKind, source: ResourceSource) -> str:
    if transport == TransportKind.GITHUB_CLONE:
        try:
            return repo_name_from_url(source.url)
        except Exception:
            return source.url
    if transport == TransportKind.GITHUB_RELEASE:
        try:
            return f"{repo_name_from_url(source.url)} ({source.revision or 'latest'})"
        except Exception:
            return source.url
    if transport == TransportKind.HUGGINGFACE:
        return source.url
    return source.url


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------


_singleton: Optional[ResourceManager] = None
_singleton_lock = threading.Lock()


def get_resource_manager() -> ResourceManager:
    """Lazy singleton — safe to call from the main thread before / after
    ``QApplication`` is constructed."""
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = ResourceManager()
    return _singleton


__all__ = [
    "QueueResult",
    "ResourceManager",
    "get_resource_manager",
]
