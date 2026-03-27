from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QThread, Signal


class HFTrainingAssetDownloadThread(QThread):
    """QThread wrapper around raw HF training-asset download."""

    progress = Signal(str)
    finished = Signal(str)
    failed = Signal(str)

    def __init__(self, url: str, parent=None) -> None:
        super().__init__(parent)
        self._url = url
        self._cancelled = False
        self._asset_path: Optional[str] = None

    def cancel(self) -> None:
        self._cancelled = True

    @property
    def asset_path(self) -> Optional[str]:
        return self._asset_path

    def run(self) -> None:
        try:
            from system.training.hf_training_asset_downloader import download_hf_training_asset

            asset_dir = download_hf_training_asset(
                self._url,
                progress_callback=self.progress.emit,
                cancel_check=lambda: self._cancelled,
            )
            if self._cancelled:
                shutil.rmtree(Path(asset_dir).parent, ignore_errors=True)
                self.failed.emit("Download cancelled.")
                return
            self._asset_path = str(asset_dir)
            self.finished.emit(str(asset_dir))
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))
