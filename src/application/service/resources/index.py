"""ResourceIndex — read/write ``custom_mods/.resources.json``.

The index is **metadata-only**: it records where each downloaded asset came
from (URL, revision, transport) and where it lives on disk, but the source
of truth for "does this asset actually exist" is always a fresh disk scan.
This matches the user's standing preference for dynamic loading (no
``runs.json``-style index files driving UI lists).

Schema (file format v1)::

    {
      "version": 1,
      "entries": [
        {
          "id": "<uuid4>",
          "asset_kind": "motion" | "policy",
          "transport": "github_clone" | "github_release" | "huggingface",
          "name": "AMP for Hardware",
          "source": {
            "url": "https://github.com/escontrela/AMP_for_hardware.git",
            "revision": "main",
            "subpath": "datasets/mocap_motions"
          },
          "local_path": "archives/AMP_for_hardware",
          "downloaded_at": "2026-05-19T12:00:00Z",
          "size_bytes": 12345678,
          "state": "local" | "downloading" | "error" | "missing",
          "error": "",
          "enabled": true
        }
      ]
    }

``local_path`` is **relative to** ``Paths.CUSTOM_MODS_DIR`` so the index
survives a relocation of ``custom_mods/`` (e.g. via ``[Resources].
custom_mods_dir``).
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from unitport_sdk import Paths, load_data, log_warning, save_data


INDEX_FILE_NAME = ".resources.json"
SCHEMA_VERSION = 1


class ResourceKind(str, Enum):
    MOTION = "motion"
    POLICY = "policy"


class TransportKind(str, Enum):
    GITHUB_CLONE = "github_clone"
    GITHUB_RELEASE = "github_release"
    HUGGINGFACE = "huggingface"


class ResourceState(str, Enum):
    DOWNLOADING = "downloading"
    LOCAL = "local"
    ERROR = "error"
    MISSING = "missing"


@dataclass(frozen=True)
class ResourceSource:
    """Where the asset came from.

    ``url`` is required for github_clone / github_release; HuggingFace uses
    ``repo_id`` plumbed in via ``url`` (we accept either a full
    ``https://huggingface.co/<repo_id>`` URL or a bare ``<owner>/<name>``
    string and normalize at the transport boundary). ``revision`` is the
    branch/tag/commit-sha (git) or the HF revision; empty -> default.
    ``subpath`` is sparse-checkout pattern (git) or
    ``allow_patterns`` glob (HF); empty -> full download.
    """

    url: str = ""
    revision: str = ""
    subpath: str = ""

    @classmethod
    def from_dict(cls, raw: object) -> "ResourceSource":
        if not isinstance(raw, dict):
            return cls()
        return cls(
            url=str(raw.get("url", "") or "").strip(),
            revision=str(raw.get("revision", "") or "").strip(),
            subpath=str(raw.get("subpath", "") or "").strip(),
        )


@dataclass
class ResourceEntry:
    """One asset record."""

    id: str
    asset_kind: ResourceKind
    transport: TransportKind
    name: str
    source: ResourceSource
    local_path: str               # relative to CUSTOM_MODS_DIR
    downloaded_at: str = ""       # ISO-8601 UTC; empty until first finish
    size_bytes: int = 0
    state: ResourceState = ResourceState.DOWNLOADING
    error: str = ""
    enabled: bool = True

    @staticmethod
    def new_id() -> str:
        return uuid.uuid4().hex

    @classmethod
    def new(
        cls,
        *,
        asset_kind: ResourceKind,
        transport: TransportKind,
        name: str,
        source: ResourceSource,
        local_path: str,
    ) -> "ResourceEntry":
        return cls(
            id=cls.new_id(),
            asset_kind=asset_kind,
            transport=transport,
            name=name,
            source=source,
            local_path=local_path,
        )

    def absolute_path(self) -> Path:
        """Resolve ``local_path`` against ``Paths.CUSTOM_MODS_DIR``."""
        return Paths.CUSTOM_MODS_DIR / self.local_path

    def to_dict(self) -> Dict[str, object]:
        d = asdict(self)
        # Enums -> strings.
        d["asset_kind"] = self.asset_kind.value
        d["transport"] = self.transport.value
        d["state"] = self.state.value
        return d

    @classmethod
    def from_dict(cls, raw: object) -> Optional["ResourceEntry"]:
        if not isinstance(raw, dict):
            return None
        try:
            return cls(
                id=str(raw["id"]),
                asset_kind=ResourceKind(str(raw["asset_kind"])),
                transport=TransportKind(str(raw["transport"])),
                name=str(raw.get("name", "") or ""),
                source=ResourceSource.from_dict(raw.get("source")),
                local_path=str(raw.get("local_path", "") or "").replace("\\", "/"),
                downloaded_at=str(raw.get("downloaded_at", "") or ""),
                size_bytes=int(raw.get("size_bytes", 0) or 0),
                state=ResourceState(str(raw.get("state", "local"))),
                error=str(raw.get("error", "") or ""),
                enabled=bool(raw.get("enabled", True)),
            )
        except (KeyError, ValueError) as e:
            log_warning(f"ResourceIndex: dropping malformed entry: {e}")
            return None


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class ResourceIndex:
    """Thread-safe accessor for ``custom_mods/.resources.json``.

    Persistence is intentionally simple: read on first access (lazy), and
    save the whole index after every mutation. The file is tiny (a few KB
    even with hundreds of entries) so the cost is irrelevant compared to
    the I/O latency of the asset downloads themselves.
    """

    def __init__(self, root: Optional[Path] = None) -> None:
        self._root = Path(root) if root is not None else Paths.CUSTOM_MODS_DIR
        self._lock = threading.RLock()
        self._loaded = False
        self._entries: Dict[str, ResourceEntry] = {}

    @property
    def path(self) -> Path:
        return self._root / INDEX_FILE_NAME

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self.path.exists():
            return
        try:
            data = load_data(self.path)
        except Exception as e:
            log_warning(f"ResourceIndex: failed to read {self.path!s}: {e}")
            return
        if not isinstance(data, dict):
            log_warning(f"ResourceIndex: {self.path!s} is not an object; ignoring")
            return
        raw_entries = data.get("entries") or []
        if not isinstance(raw_entries, list):
            log_warning(f"ResourceIndex: 'entries' is not a list; ignoring")
            return
        for raw in raw_entries:
            entry = ResourceEntry.from_dict(raw)
            if entry is not None:
                self._entries[entry.id] = entry

    def _save(self) -> None:
        # Caller owns the lock.
        self._root.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": SCHEMA_VERSION,
            "entries": [e.to_dict() for e in self._entries.values()],
        }
        save_data(self.path, payload)

    # ---- queries ----

    def list_all(self) -> List[ResourceEntry]:
        with self._lock:
            self._ensure_loaded()
            return list(self._entries.values())

    def list_by_kind(self, kind: ResourceKind) -> List[ResourceEntry]:
        with self._lock:
            self._ensure_loaded()
            return [e for e in self._entries.values() if e.asset_kind == kind]

    def get(self, entry_id: str) -> Optional[ResourceEntry]:
        with self._lock:
            self._ensure_loaded()
            return self._entries.get(entry_id)

    # ---- mutations ----

    def add(self, entry: ResourceEntry) -> None:
        with self._lock:
            self._ensure_loaded()
            self._entries[entry.id] = entry
            self._save()

    def update(
        self,
        entry_id: str,
        *,
        state: Optional[ResourceState] = None,
        size_bytes: Optional[int] = None,
        downloaded_at: Optional[str] = None,
        error: Optional[str] = None,
        enabled: Optional[bool] = None,
        local_path: Optional[str] = None,
    ) -> Optional[ResourceEntry]:
        """Patch a subset of fields on an existing entry."""
        with self._lock:
            self._ensure_loaded()
            current = self._entries.get(entry_id)
            if current is None:
                return None
            updated = replace(
                current,
                state=state if state is not None else current.state,
                size_bytes=size_bytes if size_bytes is not None else current.size_bytes,
                downloaded_at=(
                    downloaded_at
                    if downloaded_at is not None
                    else current.downloaded_at
                ),
                error=error if error is not None else current.error,
                enabled=enabled if enabled is not None else current.enabled,
                local_path=(
                    local_path if local_path is not None else current.local_path
                ),
            )
            self._entries[entry_id] = updated
            self._save()
            return updated

    def mark_finished(
        self,
        entry_id: str,
        *,
        size_bytes: int,
    ) -> Optional[ResourceEntry]:
        return self.update(
            entry_id,
            state=ResourceState.LOCAL,
            size_bytes=size_bytes,
            downloaded_at=_now_utc_iso(),
            error="",
        )

    def mark_error(self, entry_id: str, message: str) -> Optional[ResourceEntry]:
        return self.update(
            entry_id,
            state=ResourceState.ERROR,
            error=message,
        )

    def remove(self, entry_id: str) -> Optional[ResourceEntry]:
        with self._lock:
            self._ensure_loaded()
            entry = self._entries.pop(entry_id, None)
            if entry is not None:
                self._save()
            return entry


__all__ = [
    "INDEX_FILE_NAME",
    "SCHEMA_VERSION",
    "ResourceKind",
    "TransportKind",
    "ResourceState",
    "ResourceSource",
    "ResourceEntry",
    "ResourceIndex",
]
