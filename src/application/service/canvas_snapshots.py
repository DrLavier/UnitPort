"""CanvasSnapshotStore -- content-addressed canvas archive for training runs.

When a training run begins, the active canvas is serialized via
``CanvasPage.to_workflow_dict()`` and saved as a SHA-256-keyed file under::

    <project>/canvas/_runs_/<sha256>.canvas.json

The run directory (``<project>/training/runs/<backend_id>/<run_id>/``) carries
a tiny pointer file ``canvas_snapshot.txt`` whose single line is the digest of
the snapshot it references. Garbage collection scans every run directory to
compute the live reference set, then deletes orphaned ``_runs_/<digest>``
files. This gives content-addressed deduplication without an index file
(consistent with the project's "dynamic loading, no index files" rule —
runs are still authoritative on disk; the pointer files are scanned, not
indexed).

Hash contract: ``sha256`` of ``json.dumps(data, sort_keys=True,
ensure_ascii=False, separators=(",", ":"))`` UTF-8 bytes. Same canvas →
same digest; dedup is automatic.

No silent fallbacks (RELEASE/CLAUDE.md §1.8): missing pointer, missing
snapshot file, or hash-mismatched payloads all raise. Callers translate to
user-visible status-bar messages.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set

from unitport_sdk import log_debug, log_info, log_warning, read_data, save_data

from application.service.projects.store import ProjectInfo


_RUNS_DIR_NAME = "_runs_"
_POINTER_FILE_NAME = "canvas_snapshot.txt"
_SNAPSHOT_SUFFIX = ".canvas.json"


# ---------------------------------------------------------------------------
# Hash helper
# ---------------------------------------------------------------------------

def canonical_digest(data: Dict[str, Any]) -> str:
    """SHA-256 of the canvas dict's canonical JSON form.

    ``sort_keys=True`` removes ordering noise so two equal canvases always
    hash the same. ``separators`` removes whitespace ambiguity. UTF-8 bytes
    are stable across platforms.
    """
    payload = json.dumps(
        data, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class CanvasSnapshotStore:
    """Per-project store for canvas snapshots referenced by training runs."""

    def __init__(self, project: ProjectInfo) -> None:
        if project is None:
            raise ValueError("CanvasSnapshotStore: project must not be None")
        self._project = project

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    def runs_root(self) -> Path:
        return self._project.path / "canvas" / _RUNS_DIR_NAME

    def snapshot_path(self, digest: str) -> Path:
        if not digest or not isinstance(digest, str):
            raise ValueError("snapshot_path: digest must be a non-empty string")
        return self.runs_root() / f"{digest}{_SNAPSHOT_SUFFIX}"

    @staticmethod
    def pointer_path(run_dir: Path) -> Path:
        return run_dir / _POINTER_FILE_NAME

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def write(self, data: Dict[str, Any], run_dir: Path) -> str:
        """Save ``data`` content-addressed + write pointer in ``run_dir``.

        - Computes the SHA-256 digest of the canonical JSON.
        - If the snapshot file does not exist yet, writes it via the SDK
          atomic-write path (``save_data``); existing same-digest files are
          left untouched so multiple runs can share the same snapshot.
        - Writes ``run_dir/canvas_snapshot.txt`` with the digest so a
          subsequent scan can find which snapshot this run references.

        Returns the digest. Raises on I/O failure; missing ``run_dir`` is
        created on the spot (it is the trainer task's run_dir, expected to
        exist by the time this is called — the ``mkdir`` is belt-and-braces).
        """
        if not isinstance(data, dict):
            raise TypeError(
                f"CanvasSnapshotStore.write: data must be dict, got {type(data)!r}"
            )
        digest = canonical_digest(data)

        snap_path = self.snapshot_path(digest)
        snap_path.parent.mkdir(parents=True, exist_ok=True)
        if not snap_path.exists():
            save_data(snap_path, data)
            log_info(
                f"[canvas_snapshots] saved {snap_path.name} "
                f"({len(data.get('nodes') or [])} nodes)"
            )
        else:
            log_debug(
                f"[canvas_snapshots] dedup hit on {snap_path.name} — reused"
            )

        run_dir.mkdir(parents=True, exist_ok=True)
        save_data(self.pointer_path(run_dir), digest)
        return digest

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def digest_for_run(self, run_dir: Path) -> str:
        """Read the pointer file inside ``run_dir`` and return the digest.

        Raises ``FileNotFoundError`` if the pointer is absent (run dir
        predates the snapshot feature, or pointer was manually deleted) and
        ``ValueError`` if its content is not a plausible hex digest.
        """
        pointer = self.pointer_path(run_dir)
        if not pointer.exists():
            raise FileNotFoundError(
                f"canvas snapshot pointer missing: {pointer}"
            )
        raw = read_data(pointer)
        if not isinstance(raw, str):
            raise ValueError(
                f"canvas snapshot pointer {pointer} is not a string: "
                f"got {type(raw)!r}"
            )
        digest = raw.strip()
        # SHA-256 hex = 64 lowercase hex chars; allow upper for robustness.
        if len(digest) != 64 or any(c not in "0123456789abcdefABCDEF" for c in digest):
            raise ValueError(
                f"canvas snapshot pointer {pointer} content is not a "
                f"sha256 hex digest: {digest!r}"
            )
        return digest.lower()

    def load_snapshot(self, digest: str) -> Dict[str, Any]:
        """Read the snapshot file for ``digest`` and return its dict body.

        Verifies the on-disk payload still hashes to ``digest`` — a mismatch
        (corruption, manual edit) is a hard error, not a silent best-effort
        load. Raises ``FileNotFoundError`` / ``ValueError`` per
        no-silent-fallback policy.
        """
        path = self.snapshot_path(digest)
        if not path.exists():
            raise FileNotFoundError(
                f"canvas snapshot {path} missing on disk"
            )
        data = read_data(path)
        if not isinstance(data, dict):
            raise ValueError(
                f"canvas snapshot {path} is not a dict: got {type(data)!r}"
            )
        actual = canonical_digest(data)
        if actual != digest.lower():
            raise ValueError(
                f"canvas snapshot {path} content hash {actual} does not "
                f"match its filename digest {digest}"
            )
        return data

    # ------------------------------------------------------------------
    # GC
    # ------------------------------------------------------------------

    def gc(self, surviving_digests: Iterable[str]) -> int:
        """Delete every ``_runs_/*.canvas.json`` whose digest is not in
        ``surviving_digests``. Returns the count deleted."""
        root = self.runs_root()
        if not root.is_dir():
            return 0
        survivors = {str(d).lower() for d in surviving_digests if d}
        deleted = 0
        for entry in root.iterdir():
            if not entry.is_file():
                continue
            name = entry.name
            if not name.endswith(_SNAPSHOT_SUFFIX):
                continue
            stem = name[: -len(_SNAPSHOT_SUFFIX)]
            if stem.lower() in survivors:
                continue
            try:
                entry.unlink()
                deleted += 1
            except OSError as exc:
                log_warning(
                    f"[canvas_snapshots] gc: failed to remove {entry}: {exc}"
                )
        if deleted:
            log_info(f"[canvas_snapshots] gc reclaimed {deleted} orphan snapshot(s)")
        return deleted

    def gc_all(self) -> int:
        """Delete every snapshot file under ``_runs_/``. Returns the count."""
        return self.gc([])


# ---------------------------------------------------------------------------
# Reference scanner
# ---------------------------------------------------------------------------

def collect_referenced_digests(project: ProjectInfo) -> Set[str]:
    """Scan every run directory under ``<project>/training/runs/`` and
    return the union of digests their ``canvas_snapshot.txt`` pointers
    reference.

    Used by :meth:`CanvasSnapshotStore.gc` to determine which snapshots are
    still alive after a run has been deleted. Missing / malformed pointers
    are skipped silently here — they are visible at click time via
    ``digest_for_run`` and reported to the user there; this scan is a
    pure "which snapshots are still in use" set computation.
    """
    if project is None:
        return set()
    runs_dir = project.path / "training" / "runs"
    if not runs_dir.is_dir():
        return set()

    out: Set[str] = set()
    try:
        for level1 in runs_dir.iterdir():
            if not level1.is_dir():
                continue
            # Legacy flat layout: level1 itself is the run dir.
            _collect_from_run_dir(level1, out)
            # New layout: level1 is a backend partition; recurse one level.
            try:
                for run_dir in level1.iterdir():
                    if not run_dir.is_dir():
                        continue
                    _collect_from_run_dir(run_dir, out)
            except OSError as exc:                                # pragma: no cover
                log_warning(
                    f"[canvas_snapshots] scan {level1} failed: {exc}"
                )
    except OSError as exc:                                        # pragma: no cover
        log_warning(f"[canvas_snapshots] scan {runs_dir} failed: {exc}")
    return out


def _collect_from_run_dir(run_dir: Path, out: Set[str]) -> None:
    pointer = run_dir / _POINTER_FILE_NAME
    if not pointer.exists():
        return
    try:
        raw = read_data(pointer)
    except Exception as exc:                                       # noqa: BLE001
        log_warning(
            f"[canvas_snapshots] failed to read pointer {pointer}: {exc!r}"
        )
        return
    if not isinstance(raw, str):
        return
    digest = raw.strip().lower()
    if len(digest) == 64 and all(c in "0123456789abcdef" for c in digest):
        out.add(digest)


__all__ = [
    "CanvasSnapshotStore",
    "canonical_digest",
    "collect_referenced_digests",
]
