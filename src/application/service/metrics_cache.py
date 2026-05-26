# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""MetricsCache — process-level in-memory training metrics buffer.

The Mission Control chart panel needs three things that don't quite live
anywhere else:

  1. **Append-only sample stream per run** — the SDK Task thread records
     a sample each time the SB3 subprocess emits ``[UnitPort][METRICS]``;
     the chart prims its line buffers from here when a run is toggled
     visible mid-training.
  2. **Run summary list** — the data-source dropdown unions this cache
     (this-session runs, including the on-going one) with the on-disk
     ``training/runs.json`` (Stage 11 ``TrainingStore``) so historical
     runs are still selectable.
  3. **Status tracking** — ``label`` (display name) plus a status flag
     (``running`` / ``finished`` / ``failed``) so the dropdown can tag
     the on-going run with ``[ON-GOING]`` and finished runs with
     ``[finished]`` / ``[failed]``.

Persistence is **not** in scope here — runs.json + bundles.json own the
durable side, written by ``TrainingStore`` (Stage 11). MetricsCache is
intentionally process-local; closing the app wipes it.

Thread safety: a single ``threading.Lock`` guards every public mutation
and read. Reads return list/dict copies so callers never race with a
concurrent ``record()``.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


_STATUS_RUNNING = "running"
_STATUS_FINISHED = "finished"
_STATUS_FAILED = "failed"


@dataclass
class RunSummary:
    """Lightweight projection used by the data-source dropdown."""

    run_id: str
    label: str                   # human-friendly name (algorithm + short id)
    status: str = _STATUS_RUNNING  # running / finished / failed
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    sample_count: int = 0


class MetricsCache:
    """Singleton — access via :func:`get_metrics_cache`.

    Writers: :class:`SB3TrainingTask` (and any future trainer task that
    speaks the same MSG_METRICS protocol).
    Readers: ``TrainingChartPanel``, ``RunSourceSelector``.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._summaries: Dict[str, RunSummary] = {}
        # Append-only per-run sample buffer. Each sample is the raw dict
        # the trainer subprocess emitted; consumer is responsible for
        # extracting the fields it cares about.
        self._samples: Dict[str, List[dict]] = {}

    # ------------------------------------------------------------------
    # Writer API (SDK Task thread)
    # ------------------------------------------------------------------

    def mark_started(self, run_id: str, label: str) -> None:
        rid = str(run_id or "").strip()
        if not rid:
            return
        with self._lock:
            existing = self._summaries.get(rid)
            if existing is None:
                self._summaries[rid] = RunSummary(
                    run_id=rid,
                    label=str(label or rid),
                    status=_STATUS_RUNNING,
                )
                self._samples.setdefault(rid, [])
            else:
                # Re-start of an already-known run (rare — same run_id
                # resubmitted). Refresh label + flip status back to running
                # but keep the prior sample buffer so the chart continues.
                existing.label = str(label or existing.label)
                existing.status = _STATUS_RUNNING
                existing.started_at = time.time()
                existing.finished_at = None

    def record(self, run_id: str, sample: dict) -> None:
        """Append one metrics sample under *run_id*. Auto-creates a summary."""
        rid = str(run_id or "").strip()
        if not rid or not isinstance(sample, dict):
            return
        with self._lock:
            summary = self._summaries.get(rid)
            if summary is None:
                summary = RunSummary(run_id=rid, label=rid)
                self._summaries[rid] = summary
            buf = self._samples.setdefault(rid, [])
            buf.append(dict(sample))                # defensive copy
            summary.sample_count = len(buf)

    def mark_finished(self, run_id: str, success: bool) -> None:
        rid = str(run_id or "").strip()
        if not rid:
            return
        with self._lock:
            summary = self._summaries.get(rid)
            if summary is None:
                summary = RunSummary(run_id=rid, label=rid)
                self._summaries[rid] = summary
            summary.status = _STATUS_FINISHED if success else _STATUS_FAILED
            summary.finished_at = time.time()

    # ------------------------------------------------------------------
    # Reader API (main thread)
    # ------------------------------------------------------------------

    def list_runs(self) -> List[RunSummary]:
        """Return all known runs, most-recent-first by ``started_at``.

        The returned dataclasses are fresh copies — mutating them does
        not affect cache state.
        """
        with self._lock:
            out = [
                RunSummary(
                    run_id=s.run_id,
                    label=s.label,
                    status=s.status,
                    started_at=s.started_at,
                    finished_at=s.finished_at,
                    sample_count=s.sample_count,
                )
                for s in self._summaries.values()
            ]
        out.sort(key=lambda s: s.started_at, reverse=True)
        return out

    def get_run(self, run_id: str) -> List[dict]:
        """Return the sample buffer for *run_id* (list copy, never the original)."""
        rid = str(run_id or "").strip()
        if not rid:
            return []
        with self._lock:
            buf = self._samples.get(rid)
            if not buf:
                return []
            return list(buf)

    def get_summary(self, run_id: str) -> Optional[RunSummary]:
        rid = str(run_id or "").strip()
        if not rid:
            return None
        with self._lock:
            s = self._summaries.get(rid)
            if s is None:
                return None
            return RunSummary(
                run_id=s.run_id,
                label=s.label,
                status=s.status,
                started_at=s.started_at,
                finished_at=s.finished_at,
                sample_count=s.sample_count,
            )

    def clear(self) -> None:
        """Drop all cached state. Test helper; production code shouldn't call."""
        with self._lock:
            self._summaries.clear()
            self._samples.clear()


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_singleton: Optional[MetricsCache] = None
_singleton_lock = threading.Lock()


def get_metrics_cache() -> MetricsCache:
    """Lazy singleton — safe to call from any thread."""
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = MetricsCache()
    return _singleton


__all__ = [
    "RunSummary",
    "MetricsCache",
    "get_metrics_cache",
]
