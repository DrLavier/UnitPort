#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HFDownloadThread — QThread worker for HuggingFace checkpoint download.

Runs ``system.training.hf_downloader.download_hf_checkpoint`` off the
main UI thread and reports progress / completion / failure via Qt signals.

Usage::

    thread = HFDownloadThread("https://huggingface.co/owner/repo/tree/main")
    thread.progress.connect(my_label.setText)
    thread.finished.connect(on_download_finished)   # receives bundle path str
    thread.failed.connect(on_download_failed)        # receives error message
    thread.start()

    # To cancel:
    thread.cancel()
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QThread, Signal


class HFDownloadThread(QThread):
    """Worker thread that drives HuggingFace checkpoint download.

    Signals
    -------
    progress(str)
        Human-readable status update emitted during the download.
    finished(str)
        Emitted on successful completion with the path to the normalized
        bundle directory (temporary; caller must install and then clean up).
    failed(str)
        Emitted on error or cancellation with the error message string.
    """

    progress = Signal(str)
    finished = Signal(str)
    failed = Signal(str)

    def __init__(self, url: str, parent=None) -> None:
        super().__init__(parent)
        self._url: str = url
        self._cancelled: bool = False
        self._bundle_path: Optional[str] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def cancel(self) -> None:
        """Request cancellation.  The thread will stop at the next checkpoint."""
        self._cancelled = True

    @property
    def bundle_path(self) -> Optional[str]:
        """Path to the normalized bundle directory, or None if not complete."""
        return self._bundle_path

    # ------------------------------------------------------------------
    # QThread entry point
    # ------------------------------------------------------------------

    def run(self) -> None:  # noqa: D102
        try:
            from src.system.training.hf_downloader import (
                HFDownloadError,
                download_hf_checkpoint,
            )

            bundle = download_hf_checkpoint(
                self._url,
                progress_callback=self._on_progress,
                cancel_check=self._is_cancelled,
            )

            if self._cancelled:
                # Clean up any bundle that was created before cancel detected
                shutil.rmtree(bundle.parent, ignore_errors=True)
                self.failed.emit("Download cancelled.")
                return

            self._bundle_path = str(bundle)
            self.finished.emit(str(bundle))

        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _on_progress(self, msg: str) -> None:
        self.progress.emit(msg)

    def _is_cancelled(self) -> bool:
        return self._cancelled
