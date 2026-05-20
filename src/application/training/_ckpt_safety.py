"""Defensive ``torch.load`` helper for user-resumable checkpoints.

Background
----------
``torch.load(..., weights_only=False)`` invokes ``pickle``, which is a remote
code-execution primitive when the file content is attacker-controlled. PyTorch
2.6+ flipped the default to ``weights_only=True`` for exactly that reason, but
UnitPort's trainer checkpoints legitimately contain non-tensor objects
(``Normalizer`` pickle, optimizer state with custom keys, stage metadata) and
need the unrestricted loader.

This module is the single gate for **untrusted-possible** call sites — the
ones whose ``path`` argument originates in the GUI's resume / warm-start
picker, where the user could have pointed at a third-party ``.pt``. Direct
``torch.load(..., weights_only=False)`` is only allowed in trusted-internal
sites whose ``path`` is produced by this same process's own trainer; each
such site must carry a ``# WHY KEPT:`` comment per CLAUDE.md §1.8 (c).

Sidecar convention
------------------
The integrity proof is a sibling file ``<path>.sha256`` containing the
lowercase hex digest followed by a newline. Bundle finalization writes these
for its own artifacts (plan finding P2-1); the same convention is reused for
arbitrary checkpoints. A passing sidecar unlocks ``weights_only=False`` with
no further prompts.

When no sidecar is present this loader first tries ``weights_only=True``
(safe, tensor-only). If PyTorch rejects the file because it contains pickled
objects, the user is asked once for explicit consent before the unrestricted
load runs. In headless contexts (no ``QApplication``) consent is hard-No.

Plan finding P1-1.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Optional

from unitport_sdk import log_info, log_warning


def _sidecar_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".sha256")


def read_sidecar(path: Path) -> Optional[str]:
    """Return the lowercase hex digest from ``<path>.sha256`` or ``None``."""
    sidecar = _sidecar_path(path)
    if not sidecar.is_file():
        return None
    try:
        first_token = sidecar.read_text(encoding="utf-8").strip().split()
    except OSError:
        return None
    if not first_token:
        return None
    return first_token[0].lower()


def file_sha256(path: Path) -> str:
    """Stream the file in 1 MiB chunks and return its hex digest."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_sidecar(path: Path) -> Path:
    """Compute SHA-256 of ``path`` and write ``<path>.sha256`` atomically."""
    digest = file_sha256(path)
    sidecar = _sidecar_path(path)
    tmp = sidecar.with_suffix(sidecar.suffix + ".tmp")
    tmp.write_text(digest + "\n", encoding="utf-8")
    import os
    os.replace(str(tmp), str(sidecar))
    return sidecar


def _ask_user_to_proceed(path: Path) -> bool:
    """Modal confirmation when no sha256 sidecar exists.

    Returns True iff the user explicitly accepts the risk of loading a pickle
    that may execute arbitrary code. The dialog only appears when PyQt6 is
    importable and a ``QApplication`` is alive; in headless contexts (CI,
    training subprocess) the answer is hard-No.
    """
    try:
        from PyQt6.QtWidgets import QApplication, QMessageBox
    except ImportError:
        return False
    app = QApplication.instance()
    if app is None:
        return False
    box = QMessageBox()
    box.setIcon(QMessageBox.Icon.Warning)
    box.setWindowTitle("Checkpoint integrity not verified")
    box.setText(
        f"The checkpoint at\n  {path}\n\n"
        "has no SHA-256 sidecar. Loading it will deserialize Python objects "
        "(pickle), which can execute arbitrary code if the file came from an "
        "untrusted source.\n\nContinue anyway?"
    )
    box.setStandardButtons(
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
    )
    box.setDefaultButton(QMessageBox.StandardButton.No)
    return box.exec() == QMessageBox.StandardButton.Yes


def load_checkpoint_safely(path: Any, *, map_location: Any = "cpu") -> Any:
    """Load a torch checkpoint with explicit integrity gating.

    Policy:
      1. ``<path>.sha256`` exists → verify; mismatch raises; match unlocks
         ``weights_only=False``.
      2. No sidecar → first try ``weights_only=True`` (tensor-only safe
         load). If PyTorch rejects because the checkpoint legitimately needs
         custom classes, prompt the user before falling back to
         ``weights_only=False``. Headless contexts treat the prompt as
         refused and raise.

    Raises ``RuntimeError`` on integrity failure or refused consent.
    """
    import torch

    p = Path(path)
    expected = read_sidecar(p)
    if expected is not None:
        got = file_sha256(p)
        if got.lower() != expected:
            raise RuntimeError(
                f"checkpoint SHA-256 mismatch for {p}: "
                f"expected {expected}, got {got.lower()}"
            )
        log_info(f"[ckpt_safety] {p.name} sha256 verified")
        return torch.load(str(p), map_location=map_location, weights_only=False)

    needs_pickle = False
    try:
        return torch.load(str(p), map_location=map_location, weights_only=True)
    except TypeError:
        needs_pickle = True
    except Exception as exc:
        log_warning(
            f"[ckpt_safety] {p.name}: strict load rejected by torch "
            f"({type(exc).__name__}: {exc}); the file contains pickled objects."
        )
        needs_pickle = True

    if not needs_pickle:
        return None

    if not _ask_user_to_proceed(p):
        raise RuntimeError(
            f"refused to load checkpoint {p} — no SHA-256 sidecar and pickle "
            f"load was not authorized"
        )
    log_warning(
        f"[ckpt_safety] {p.name}: loading with weights_only=False after user consent"
    )
    return torch.load(str(p), map_location=map_location, weights_only=False)


__all__ = [
    "file_sha256",
    "load_checkpoint_safely",
    "read_sidecar",
    "write_sidecar",
]
