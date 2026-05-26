# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Async downloader + on-disk cache for the signed-in user's avatar.

Cache lives at ``Paths.USER_CONFIG_DIR / "avatars" / "<md5(url)>.img"`` so it
survives across restarts but never lands in the project tree.

OAuth providers (Google, GitHub, ...) hand Supabase an ``avatar_url`` that
points at a public HTTPS image. We fetch it exactly once per URL, off the UI
thread, then cache the bytes by ``md5(url)`` so the entry rotates
automatically whenever the provider rolls the avatar URL.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

# NOTE: ``httpx`` is intentionally NOT imported at module top.
# avatar_cache is pulled into the import graph eagerly by the InstallConfigWizard
# chain (wizard → menagerie_card → ui.widgets → mission_control_panel →
# real_robot_connection_card → connection_settings_dialog → ui.dialogs.__init__
# → email_identity_dialog → ``from application.service.auth import
# get_auth_manager`` → auth/__init__ ``__getattr__`` → auth_manager →
# avatar_cache). On a first launch the wizard is constructed BEFORE
# ProvisioningTask installs requirements.txt, so httpx is not yet on disk.
# Importing it eagerly at module top crashes the wizard construction with
# ``ModuleNotFoundError: No module named 'httpx'``. The actual HTTP call is
# only ever issued by ``_AvatarDownloadWorker.run`` (off the UI thread, well
# after Stage 3), so the import is deferred to call time.
from PyQt6.QtCore import QObject, QRectF, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPainterPath, QPixmap

from unitport_sdk import Config, Paths, log_debug, log_warning

_DOWNLOAD_TIMEOUT = 10.0
_MAX_BYTES = 2 * 1024 * 1024   # 2 MiB — plenty for provider thumbnails


def _cache_dir() -> Path:
    """Lazy resolve of the avatar cache dir inside the LIVE USER_CONFIG_DIR.

    Resolved at call time (not module load) so hot workspace switches via
    ``user_workspace.set_user_workspace`` are picked up without restarting
    the process.
    """
    return Paths.USER_CONFIG_DIR / "avatars"


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


class _AvatarDownloadWorker(QThread):
    """Fetch avatar_url bytes off the UI thread; emit once."""

    fetched = pyqtSignal(str, str, bytes)   # user_id, avatar_url, body (empty on fail)

    def __init__(self, user_id: str, avatar_url: str, parent=None) -> None:
        super().__init__(parent)
        self._user_id = user_id
        self._avatar_url = avatar_url

    def run(self) -> None:  # type: ignore[override]
        body = b""
        # Imported here, not at module top: see the note above the imports
        # block. ProvisioningTask has long since installed httpx by the time
        # any avatar fetch is issued (avatars only fire after Stage 3 when
        # AuthManager emits ``avatar_updated``).
        import httpx
        try:
            with httpx.stream(
                "GET", self._avatar_url,
                timeout=_DOWNLOAD_TIMEOUT, follow_redirects=True,
            ) as resp:
                if resp.status_code == 200:
                    chunks = []
                    total = 0
                    for chunk in resp.iter_bytes():
                        total += len(chunk)
                        if total > _MAX_BYTES:
                            log_warning(
                                f"[avatar] {self._avatar_url} exceeds {_MAX_BYTES}B — aborting"
                            )
                            chunks = []
                            break
                        chunks.append(chunk)
                    body = b"".join(chunks)
                else:
                    log_warning(f"[avatar] HTTP {resp.status_code} for {self._avatar_url}")
        except Exception as exc:
            # Covers httpx.RequestError as well — kept under a single
            # except so the local httpx import above doesn't need to be
            # bound to a name in the surrounding scope.
            log_warning(f"[avatar] network error fetching {self._avatar_url}: {exc}")
        self.fetched.emit(self._user_id, self._avatar_url, body)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


class AvatarCache(QObject):
    """Owner of the avatar fetch pipeline.

    De-dupes concurrent (user_id, url) requests; emits :attr:`avatar_ready`
    with a QPixmap constructed on the main thread.
    """

    avatar_ready = pyqtSignal(str, QPixmap)   # user_id, pixmap

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        # NOTE: do NOT mkdir _cache_dir() here — the live USER_CONFIG_DIR
        # may swap under us (workspace hot-switch), and we don't want to
        # leave behind an empty avatars/ dir in every account that has
        # ever cold-started UnitPort. Lazy-mkdir at first write instead.
        self._inflight: set[str] = set()
        self._workers: list[QThread] = []

    def fetch(self, user_id: str, avatar_url: str) -> None:
        """Serve the avatar via :attr:`avatar_ready`. Cache hit -> sync; miss -> worker."""
        if not user_id or not avatar_url:
            return

        # Ensure cache dir exists under the LIVE USER_CONFIG_DIR.
        try:
            _cache_dir().mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            log_warning(f"[avatar] could not create cache dir: {exc}")

        cache_path = self._cache_path(avatar_url)
        if cache_path.exists():
            pm = QPixmap(str(cache_path))
            if not pm.isNull():
                log_debug(f"[avatar] cache hit {cache_path.name}")
                self.avatar_ready.emit(user_id, pm)
                return
            try:
                cache_path.unlink()
            except OSError:
                pass

        key = f"{user_id}::{avatar_url}"
        if key in self._inflight:
            return
        self._inflight.add(key)
        log_debug(f"[avatar] fetching {avatar_url[:80]} -> {cache_path.name}")

        worker = _AvatarDownloadWorker(user_id, avatar_url, parent=self)
        worker.fetched.connect(self._on_fetched)
        worker.finished.connect(lambda w=worker, k=key: self._drop_worker(w, k))
        self._workers.append(worker)
        worker.start()

    def load_cached(self, avatar_url: str) -> Optional[QPixmap]:
        """Synchronous cache-only lookup. Returns None on miss."""
        if not avatar_url:
            return None
        cache_path = self._cache_path(avatar_url)
        if not cache_path.exists():
            return None
        pm = QPixmap(str(cache_path))
        return pm if not pm.isNull() else None

    @staticmethod
    def _cache_path(avatar_url: str) -> Path:
        digest = hashlib.md5(avatar_url.encode("utf-8")).hexdigest()
        # Neutral .img suffix — QPixmap auto-detects format from the header.
        return _cache_dir() / f"{digest}.img"

    def _on_fetched(self, user_id: str, avatar_url: str, body: bytes) -> None:
        if not body:
            log_warning(f"[avatar] empty body from {avatar_url[:80]}")
            return
        cache_path = self._cache_path(avatar_url)
        try:
            cache_path.write_bytes(body)
            log_debug(f"[avatar] cached {len(body)}B -> {cache_path.name}")
        except OSError as exc:
            log_warning(f"[avatar] cache write failed: {exc}")
        pm = QPixmap()
        if pm.loadFromData(body):
            self.avatar_ready.emit(user_id, pm)
        else:
            log_warning(f"[avatar] Qt could not parse image from {avatar_url}")

    def _drop_worker(self, worker: QThread, key: str) -> None:
        self._inflight.discard(key)
        try:
            self._workers.remove(worker)
        except ValueError:
            pass
        worker.deleteLater()


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def round_pixmap(pm: QPixmap, size: int) -> QPixmap:
    """Return a center-cropped, circular size x size copy of pm."""
    if pm.isNull() or size <= 0:
        return QPixmap()
    scaled = pm.scaled(
        size, size,
        Qt.AspectRatioMode.KeepAspectRatioByExpanding,
        Qt.TransformationMode.SmoothTransformation,
    )
    crop_x = max(0, (scaled.width() - size) // 2)
    crop_y = max(0, (scaled.height() - size) // 2)
    cropped = scaled.copy(crop_x, crop_y, size, size)

    out = QPixmap(size, size)
    out.fill(Qt.GlobalColor.transparent)
    painter = QPainter(out)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
    path = QPainterPath()
    path.addEllipse(QRectF(0, 0, size, size))
    painter.setClipPath(path)
    painter.drawPixmap(0, 0, cropped)
    painter.end()
    return out


def _initials(label: str) -> str:
    if not label:
        return "?"
    if "@" in label:
        label = label.split("@", 1)[0]
    import re as _re
    parts = [p for p in _re.split(r"[\s_.\-]+", label.strip()) if p]
    if len(parts) >= 2:
        return (parts[0][0] + parts[-1][0]).upper()
    if parts:
        return parts[0][0].upper()
    return "?"


def _hash_color(seed: str) -> QColor:
    digest = hashlib.md5((seed or "?").encode("utf-8")).digest()
    hue = digest[0] / 256.0
    return QColor.fromHsvF(hue, 0.55, 0.78)


def make_initials_pixmap(seed: str, label: str, size: int) -> QPixmap:
    """Render a circular avatar with label's initials on a deterministic colour."""
    if size <= 0:
        return QPixmap()
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pm)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setBrush(_hash_color(seed))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawEllipse(0, 0, size, size)

    initials = _initials(label)
    font = QFont(painter.font())
    font.setPointSize(max(9, int(size * 0.38)))
    font.setBold(True)
    painter.setFont(font)
    painter.setPen(QColor(Config.get_color("main_t2")))
    painter.drawText(pm.rect(), Qt.AlignmentFlag.AlignCenter, initials)
    painter.end()
    return pm


__all__ = ["AvatarCache", "round_pixmap", "make_initials_pixmap"]
