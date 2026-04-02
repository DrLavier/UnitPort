from __future__ import annotations

import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Callable, Optional

from .hf_downloader import HFDownloadError, parse_hf_url


def _asset_id_from_repo(repo_id: str) -> str:
    repo_name = repo_id.split("/")[-1]
    return re.sub(r"[^a-z0-9_]", "_", repo_name.lower())


def download_hf_training_asset(
    url: str,
    *,
    progress_callback: Optional[Callable[[str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Path:
    """Download a raw HF training repository snapshot for Training Ground."""
    repo_id, revision = parse_hf_url(url)

    def _emit(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)

    def _cancelled() -> bool:
        return bool(cancel_check and cancel_check())

    if _cancelled():
        raise HFDownloadError("Download cancelled.")

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise HFDownloadError(
            "huggingface_hub is not installed.\nRun: pip install huggingface_hub"
        ) from exc

    tmp_snapshot = Path(tempfile.mkdtemp(prefix="unitport_hf_training_snapshot_"))
    asset_parent = Path(tempfile.mkdtemp(prefix="unitport_hf_training_asset_"))
    asset_dir = asset_parent / _asset_id_from_repo(repo_id)

    try:
        _emit(f"Parsed: {repo_id} @ {revision}")
        downloaded_path = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=str(tmp_snapshot),
        )
        if _cancelled():
            raise HFDownloadError("Download cancelled.")
        shutil.copytree(str(downloaded_path), str(asset_dir))
        source_meta = {
            "type": "huggingface_training_asset",
            "repo_id": repo_id,
            "revision": revision,
            "url": url,
        }
        (asset_dir / "source.json").write_text(
            json.dumps(source_meta, indent=2),
            encoding="utf-8",
        )
        _emit("Training asset snapshot ready.")
        return asset_dir
    except Exception as exc:
        shutil.rmtree(asset_parent, ignore_errors=True)
        if isinstance(exc, HFDownloadError):
            raise
        raise HFDownloadError(f"Download failed: {exc}") from exc
    finally:
        shutil.rmtree(tmp_snapshot, ignore_errors=True)
