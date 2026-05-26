# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Atomic IO helpers for training checkpoints + bundle artifacts.

Routes everything through the SDK to honour the "no business open()" rule:

- text/structured (.json/.yaml/.toml/.ini/.csv/.parquet/.xlsx/.pkl/.txt) →
  ``DataManager.save_data`` / ``load_data`` (atomic temp + os.replace).
- binary blobs not in DataManager's handler set (.pt / .zip / .onnx) →
  :func:`atomic_write_bytes` (same atomic pattern; same volume guarantee).

Bundle artifacts that should land outside the project tree go through
``push_data(rel_path, value)`` which writes under ``Paths.USER_CONFIG_DIR``.
For bytes targets, prefer :func:`push_bundle_bytes` which composes the
``USER_CONFIG_DIR`` anchor and the atomic write.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Optional

from unitport_sdk import Paths, load_data, log_warning, push_data, save_data


def save_artifact(path: Path, value: Any) -> bool:
    """Atomic save dispatched by extension via DataManager.

    Use for ``.json`` / ``.yaml`` / ``.toml`` / ``.ini`` / ``.csv`` /
    ``.parquet`` / ``.pkl`` / ``.txt``. Returns True on success.
    """
    return save_data(path, value)


def load_artifact(path: Path) -> Any:
    """Read+cache via DataManager. Returns the parsed payload or None."""
    return load_data(path)


def atomic_write_bytes(path: Path, data: bytes) -> bool:
    """Atomic binary write — same temp+os.replace pattern as DataManager
    but for blob types DataManager doesn't natively dispatch (``.pt`` /
    ``.zip`` / ``.onnx``).

    The temp file is created in the **same parent directory** so the final
    ``os.replace`` is atomic on the same volume.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd: Optional[int] = None
    tmp_path: Optional[str] = None
    try:
        fd, tmp_path = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=str(path.parent),
        )
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        fd = None  # closed by context manager
        os.replace(tmp_path, str(path))
        tmp_path = None
        return True
    except OSError as exc:
        log_warning(f"[checkpoint_io] atomic_write_bytes failed for {path}: {exc}")
        return False
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def push_bundle_bytes(rel_path: str, data: bytes) -> bool:
    """Write binary bundle artifact under ``USER_CONFIG_DIR`` atomically.

    ``rel_path`` is relative to ``Paths.USER_CONFIG_DIR`` (mirrors the
    ``Storage.push_data`` convention). Used for the ONNX policy bytes
    inside an exported bundle, where the destination is outside the
    project tree.
    """
    target = Paths.USER_CONFIG_DIR / rel_path
    return atomic_write_bytes(target, data)


def push_bundle_artifact(rel_path: str, value: Any) -> bool:
    """Structured bundle artifact (manifest.yaml, source.json) via Storage.

    Thin pass-through to ``push_data`` for symmetry with
    :func:`push_bundle_bytes`. Cloud channel (``Storage.push("...", ...,
    channel="cloud")``) is intentionally not exposed here — bundle export
    today is local-only.
    """
    return push_data(rel_path, value)


__all__ = [
    "save_artifact",
    "load_artifact",
    "atomic_write_bytes",
    "push_bundle_bytes",
    "push_bundle_artifact",
]
