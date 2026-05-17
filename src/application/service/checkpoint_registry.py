"""CheckpointRegistry — discover ``.pt`` / ``.zip`` checkpoints under runs/.

REWRITE-WITH-DEMO-REF from DEMO ``checkpoint_watcher.py``, narrowed to
the synchronous filesystem-scan use case Stage 11 needs:

  * ``list_checkpoints(run_id)`` — sorted by step number (parsed from
    filename); falls back to mtime when no step is encoded.
  * ``latest_checkpoint(run_id)`` — convenience wrapper.
  * ``prune_older_than(run_id, keep_last_n)`` — optional retention.

Checkpoints live in ``<project>/training/runs/<run_id>/checkpoints/``.
The naming convention is permissive — any ``.pt`` / ``.zip`` / ``.ckpt``
file is treated as a checkpoint; step is parsed from the first integer
in the filename (matches both ``model_50000_steps.zip`` and
``ckpt_iter000150.pt``).
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from unitport_sdk import log_info, log_warning

from application.service.projects.store import ProjectInfo


_CKPT_SUFFIXES = (".pt", ".zip", ".ckpt", ".pth")
_STEP_RE = re.compile(r"(\d+)")


@dataclass(frozen=True)
class CheckpointInfo:
    """One checkpoint file."""

    path: Path
    run_id: str
    step: int                  # parsed from filename, else -1
    size_bytes: int
    mtime: float               # epoch seconds

    @property
    def name(self) -> str:
        return self.path.name


def _parse_step(name: str) -> int:
    """Best-effort step parse. Returns -1 if no integer present."""
    m = _STEP_RE.search(name)
    if m is None:
        return -1
    try:
        return int(m.group(1))
    except (TypeError, ValueError):
        return -1


# ---------------------------------------------------------------------------
# CheckpointRegistry
# ---------------------------------------------------------------------------


class CheckpointRegistry:
    """Per-project checkpoint discovery."""

    def __init__(self, project: ProjectInfo) -> None:
        if not isinstance(project, ProjectInfo):
            raise TypeError(
                f"CheckpointRegistry: expected ProjectInfo, got "
                f"{type(project).__name__}"
            )
        self.project = project

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    @property
    def runs_dir(self) -> Path:
        return self.project.path / "training" / "runs"

    def run_checkpoints_dir(self, run_id: str) -> Path:
        return self.runs_dir / str(run_id) / "checkpoints"

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def list_checkpoints(self, run_id: str) -> List[CheckpointInfo]:
        """Return all checkpoints for *run_id*, sorted by step ascending.

        Files without a parseable step number are appended at the end
        (sorted by mtime).
        """
        ckpt_dir = self.run_checkpoints_dir(run_id)
        if not ckpt_dir.is_dir():
            return []

        out: List[CheckpointInfo] = []
        for p in ckpt_dir.iterdir():
            if not p.is_file():
                continue
            if p.suffix.lower() not in _CKPT_SUFFIXES:
                continue
            try:
                stat = p.stat()
            except OSError as exc:
                log_warning(f"[ckpt] stat failed for {p}: {exc}")
                continue
            out.append(CheckpointInfo(
                path=p,
                run_id=str(run_id),
                step=_parse_step(p.stem),
                size_bytes=int(stat.st_size),
                mtime=float(stat.st_mtime),
            ))

        # Stable order: stepped first (ascending), then unstepped by mtime
        stepped = [c for c in out if c.step >= 0]
        unstepped = [c for c in out if c.step < 0]
        stepped.sort(key=lambda c: (c.step, c.mtime))
        unstepped.sort(key=lambda c: c.mtime)
        return stepped + unstepped

    def latest_checkpoint(self, run_id: str) -> Optional[CheckpointInfo]:
        """Highest-step checkpoint for *run_id* (or most recent unstepped)."""
        cks = self.list_checkpoints(run_id)
        return cks[-1] if cks else None

    def list_run_ids(self) -> List[str]:
        """Return all run ids that have a checkpoints/ subdir."""
        if not self.runs_dir.is_dir():
            return []
        out: List[str] = []
        for child in self.runs_dir.iterdir():
            if not child.is_dir():
                continue
            if (child / "checkpoints").is_dir():
                out.append(child.name)
        return sorted(out)

    # ------------------------------------------------------------------
    # Retention
    # ------------------------------------------------------------------

    def prune_older_than(self, run_id: str, *, keep_last_n: int) -> int:
        """Delete all but the *keep_last_n* most recent checkpoints.

        Returns the number of files removed. ``keep_last_n <= 0`` is a
        no-op (returns 0). Stepped checkpoints are kept by step; unstepped
        by mtime — same ordering as :meth:`list_checkpoints`.
        """
        if keep_last_n <= 0:
            return 0
        ordered = self.list_checkpoints(run_id)
        if len(ordered) <= keep_last_n:
            return 0
        to_delete = ordered[:-keep_last_n]
        deleted = 0
        for ck in to_delete:
            try:
                ck.path.unlink()
                deleted += 1
            except OSError as exc:
                log_warning(f"[ckpt] failed to remove {ck.path}: {exc}")
        if deleted:
            log_info(
                f"[ckpt] pruned {deleted} old checkpoint(s) under "
                f"{self.run_checkpoints_dir(run_id)} (kept last {keep_last_n})"
            )
        return deleted


__all__ = [
    "CheckpointInfo",
    "CheckpointRegistry",
]
