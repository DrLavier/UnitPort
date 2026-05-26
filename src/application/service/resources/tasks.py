# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""SDK Task wrapper for the resources downloader.

One ``DownloadResourceTask`` per asset download. The task:

1. Picks the transport implementation for ``entry.transport``.
2. Wires its ``on_progress`` callback to ``self.progress_line`` AND to
   ``AppSignals.resource_progress`` so the Resources panel can render a
   per-card progress bar without subscribing to the SDK task signal bus.
3. On success: updates the index entry (state=local, size_bytes,
   downloaded_at) and emits ``resource_finished(id, True, "")``.
4. On failure / cancellation: cleans up the partial directory tree,
   updates the entry (state=error / removes entry), and emits the
   appropriate terminal signal.

Per the project's logging contract, ``progress_line`` is NOT interleaved
with ``log_info`` / ``log_warning``. Failure paths use ``log_error`` once,
at the end of the task.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from unitport_sdk import Paths, Task, TaskCancelledException, log_error

from ..signals import get_app_signals
from .index import (
    ResourceEntry,
    ResourceIndex,
    ResourceState,
    TransportKind,
)
from .transports import (
    GitHubCloneTransport,
    GitHubReleaseTransport,
    HuggingFaceTransport,
    Transport,
    TransportContext,
    TransportError,
)


def _build_transport(entry: ResourceEntry) -> Transport:
    if entry.transport == TransportKind.GITHUB_CLONE:
        return GitHubCloneTransport()
    if entry.transport == TransportKind.GITHUB_RELEASE:
        return GitHubReleaseTransport()
    if entry.transport == TransportKind.HUGGINGFACE:
        # Use kind to seed the HF repo_type default; a typed URL with
        # /datasets/ overrides this at fetch time.
        return HuggingFaceTransport(
            repo_type_hint="dataset" if entry.asset_kind.value == "motion" else "model",
        )
    raise TransportError(f"unsupported transport: {entry.transport!r}")


class DownloadResourceTask(Task):
    """Run one download to completion (or cancellation)."""

    def __init__(self, entry: ResourceEntry, index: ResourceIndex) -> None:
        super().__init__(f"Download {entry.name or entry.id[:8]}")
        self.entry_id = entry.id
        self._index = index
        self._entry = entry
        self._dest = entry.absolute_path()

    def run(self):
        transport = _build_transport(self._entry)
        signals = get_app_signals()

        def _on_progress(frac: float, line: str) -> None:
            # SDK contract: progress in [0, 1]; line is a single non-empty status.
            self.progress_line(max(0.0, min(frac, 1.0)), line)
            signals.resource_progress.emit(self.entry_id, frac, line)

        def _cancel_check() -> None:
            # Re-uses Task's own cancel flag; raises TaskCancelledException.
            self.check_cancelled()

        ctx = TransportContext(
            source=self._entry.source,
            dest_dir=self._dest,
            on_progress=_on_progress,
            cancel_check=_cancel_check,
        )

        try:
            size = transport.fetch(ctx)
        except TaskCancelledException:
            # User asked to abort. Clean up partial tree + remove entry.
            self._cleanup_partial()
            self._index.remove(self.entry_id)
            signals.resource_removed.emit(self.entry_id)
            raise
        except TransportError as e:
            self._cleanup_partial()
            self._index.mark_error(self.entry_id, str(e))
            log_error(f"resource download failed [{self.entry_id[:8]}]: {e}")
            signals.resource_finished.emit(self.entry_id, False, str(e))
            return False
        except Exception as e:
            # Unexpected — preserve the partial tree so the user can
            # inspect it manually, but mark the entry as errored. The
            # crash hook will have logged the traceback already.
            self._index.mark_error(self.entry_id, f"{type(e).__name__}: {e}")
            log_error(
                f"resource download crashed [{self.entry_id[:8]}]: "
                f"{type(e).__name__}: {e}"
            )
            signals.resource_finished.emit(
                self.entry_id, False, f"{type(e).__name__}: {e}"
            )
            return False

        self._index.mark_finished(self.entry_id, size_bytes=int(size or 0))
        signals.resource_finished.emit(self.entry_id, True, "")
        return True

    def _cleanup_partial(self) -> None:
        """Remove the partial destination tree, swallowing FS errors.

        We accept best-effort cleanup here: if a stray file is locked by
        another process the user can delete it manually later, but we
        should not crash the task on the cleanup path.
        """
        dest = self._dest
        if not dest.exists():
            return
        try:
            shutil.rmtree(dest, ignore_errors=True)
        except OSError:
            pass


__all__ = ["DownloadResourceTask"]
