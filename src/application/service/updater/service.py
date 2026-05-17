"""UpdateService — singleton orchestrator for the in-app update flow.

Responsibilities:

- On ``check()``: hit the GitHub Releases API, compare ``tag_name`` to
  ``Config.get_value("System", "version", ...)``, persist the result
  into ``user.ini[App]``, emit ``update_check_complete`` on the
  AppSignals bus. Honors a 6-hour throttle so repeated app launches
  don't hammer the GitHub rate limit.
- On ``apply()``: detect the active channel, call its ``apply()``,
  forward progress through ``update_apply_progress`` and the terminal
  ``update_apply_complete``.
- ``latest_release()``: cached release info from the most recent
  successful check, for UI consumers that arrive after the signal has
  already fired (e.g. the sidebar built post-boot).

Per CLAUDE.md §1.6, state lives in ``user.ini[App]`` only:
``update_last_check``, ``update_available_version``,
``update_skipped_version``, ``update_include_prereleases``.
"""

from __future__ import annotations

import datetime as _dt
import threading
from typing import Optional

from packaging.version import InvalidVersion, Version

from unitport_sdk import Config, log_debug, log_info, log_success, log_warning

from .channels import detect_channel
from .channels.base import ProgressCallback, UpdateChannel
from .release_api import (
    UpdateNetworkError,
    UpdateParseError,
    fetch_latest_release,
)
from .release_info import ApplyResult, ReleaseInfo


# Skip the API call if we have a fresh-enough cached result. 6 h keeps
# us well under GitHub's 60 req/h/IP unauth budget across many launches.
_THROTTLE_HOURS = 6

_REPO = "DrLavier/UnitPort"


def _safe_version(v: str) -> Optional[Version]:
    """Parse ``v`` to a Version or return None (caller logs + falls back)."""
    s = (v or "").strip()
    if not s:
        return None
    try:
        return Version(s)
    except InvalidVersion:
        return None


def _isoformat_now() -> str:
    return _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"


class UpdateService:
    """Singleton facade — call ``get_update_service()`` to obtain it."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: Optional[ReleaseInfo] = None
        self._channel: Optional[UpdateChannel] = None

    # ------------------------------------------------------------------
    # State accessors
    # ------------------------------------------------------------------
    def latest_release(self) -> Optional[ReleaseInfo]:
        """Most recent successful check's release info, or None."""
        return self._latest

    def current_version(self) -> str:
        return str(Config.get_value("System", "version", "0.0.0") or "0.0.0")

    def channel(self) -> Optional[UpdateChannel]:
        """Cache + return the detected channel for this install."""
        with self._lock:
            if self._channel is None:
                self._channel = detect_channel()
            return self._channel

    # ------------------------------------------------------------------
    # Check
    # ------------------------------------------------------------------
    def _should_skip_throttled(self) -> bool:
        """Whether the last successful check is too recent to repeat."""
        last = str(Config.get_value("App", "update_last_check", "") or "").strip()
        if not last:
            return False
        try:
            last_dt = _dt.datetime.fromisoformat(last.rstrip("Z"))
        except ValueError:
            return False
        delta = _dt.datetime.utcnow() - last_dt
        return delta.total_seconds() < _THROTTLE_HOURS * 3600

    def _emit_check_complete(self, info: Optional[ReleaseInfo]) -> None:
        try:
            from application.service.signals import get_app_signals
            get_app_signals().update_check_complete.emit(info)
        except Exception as exc:
            log_warning(f"[updater] failed to emit update_check_complete: {exc}")

    def _replay_cached_release(self) -> Optional[ReleaseInfo]:
        """Reconstruct a minimal ReleaseInfo from ``user.ini[App]`` cache.

        Used when the throttle short-circuits the network call. Only the
        version is preserved — the release notes are not cached.
        """
        cached_version = str(
            Config.get_value("App", "update_available_version", "") or ""
        ).strip()
        if not cached_version:
            return None
        tag = f"v{cached_version}"
        # We don't have body/html_url cached, but a sidebar-only consumer
        # only needs ``.version`` to decide button state. The dialog
        # branch will trigger a fresh check if .html_url is empty.
        return ReleaseInfo(tag=tag, version=cached_version)

    def check(self, *, force: bool = False) -> Optional[ReleaseInfo]:
        """Run a single update check. Returns the release if newer than local.

        ``force=True`` bypasses the 6-hour throttle. The throttled path
        still emits ``update_check_complete`` with the cached result so
        the sidebar's signal-driven button state stays consistent.
        """
        try:
            from application.service.signals import get_app_signals
            get_app_signals().update_check_started.emit()
        except Exception:
            pass

        include_pre = bool(
            Config.get_value(
                "App", "update_include_prereleases", False, value_type=bool,
            )
        )

        if not force and self._should_skip_throttled():
            cached = self._replay_cached_release()
            with self._lock:
                self._latest = cached
            log_debug(
                f"[updater] check throttled (<{_THROTTLE_HOURS}h since last); "
                f"cached available_version="
                f"{Config.get_value('App', 'update_available_version', '')!r}"
            )
            self._emit_check_complete(cached)
            return cached

        try:
            release = fetch_latest_release(
                _REPO,
                include_prereleases=include_pre,
            )
        except UpdateNetworkError as exc:
            log_warning(f"[updater] check failed (network): {exc}")
            self._emit_check_complete(self._replay_cached_release())
            return None
        except UpdateParseError as exc:
            log_warning(f"[updater] check failed (parse): {exc}")
            self._emit_check_complete(self._replay_cached_release())
            return None
        except Exception as exc:  # noqa: BLE001 — defensive
            log_warning(f"[updater] check failed (unexpected): {exc}")
            self._emit_check_complete(self._replay_cached_release())
            return None

        # Successful API call — record the timestamp regardless of result.
        Config.set_value("App", "update_last_check", _isoformat_now())

        local = _safe_version(self.current_version())
        remote = _safe_version(release.version)
        if local is None or remote is None:
            log_warning(
                f"[updater] malformed version: local={self.current_version()!r} "
                f"remote={release.version!r} — treating as no-update"
            )
            Config.set_value("App", "update_available_version", "")
            with self._lock:
                self._latest = None
            self._emit_check_complete(None)
            return None

        if remote > local:
            with self._lock:
                self._latest = release
            Config.set_value("App", "update_available_version", release.version)
            skipped = str(
                Config.get_value("App", "update_skipped_version", "") or ""
            ).strip()
            if skipped == release.version:
                log_info(
                    f"[updater] new version {release.version} available "
                    f"(user previously skipped this version)"
                )
            else:
                log_success(
                    f"新版本 {release.version} 可用 / new version "
                    f"{release.version} available"
                )
            self._emit_check_complete(release)
            return release

        # Already on or ahead of latest.
        with self._lock:
            self._latest = None
        Config.set_value("App", "update_available_version", "")
        log_debug(
            f"[updater] up to date "
            f"(local={local}, latest={remote})"
        )
        self._emit_check_complete(None)
        return None

    # ------------------------------------------------------------------
    # Apply
    # ------------------------------------------------------------------
    def apply(
        self,
        release: ReleaseInfo,
        *,
        progress_cb: Optional[ProgressCallback] = None,
    ) -> ApplyResult:
        """Run the active channel's apply() on the calling thread.

        Designed to be called from inside ``ApplyUpdateTask.run`` so
        the long-running work happens on the TasksManager worker, never
        on the UI thread. Emits ``update_apply_progress`` via the signal
        bus in addition to invoking any caller-supplied callback.
        """
        ch = self.channel()
        if ch is None:
            res = ApplyResult(ok=False, message="no update channel available")
            self._emit_apply_complete(res)
            return res

        log_info(f"[updater] applying {release.tag} via {ch.id} channel")

        def cb(frac: float, label: str) -> None:
            try:
                from application.service.signals import get_app_signals
                get_app_signals().update_apply_progress.emit(float(frac), str(label))
            except Exception:
                pass
            if progress_cb is not None:
                try:
                    progress_cb(frac, label)
                except Exception:
                    pass

        try:
            result = ch.apply(release, progress_cb=cb)
        except Exception as exc:  # noqa: BLE001
            log_warning(f"[updater] channel raised: {exc}")
            result = ApplyResult(ok=False, message=f"channel error: {exc}")

        if result.ok:
            # Clear the "available" flag so the next launch's check
            # finds nothing new (the update WAS applied; the new tag
            # is now the local version after restart).
            Config.set_value("App", "update_available_version", "")
            log_success(f"[updater] apply complete: {result.message}")
        else:
            log_warning(f"[updater] apply failed: {result.message}")

        self._emit_apply_complete(result)
        return result

    def _emit_apply_complete(self, result: ApplyResult) -> None:
        try:
            from application.service.signals import get_app_signals
            get_app_signals().update_apply_complete.emit(
                bool(result.ok), str(result.message)
            )
        except Exception as exc:
            log_warning(f"[updater] failed to emit update_apply_complete: {exc}")

    # ------------------------------------------------------------------
    # User actions
    # ------------------------------------------------------------------
    def skip_version(self, version: str) -> None:
        """User dismissed a specific version via the Skip button."""
        Config.set_value("App", "update_skipped_version", str(version or ""))
        log_info(f"[updater] user skipped version {version!r}")


_singleton: Optional[UpdateService] = None


def get_update_service() -> UpdateService:
    """Return the process-wide :class:`UpdateService` singleton."""
    global _singleton
    if _singleton is None:
        _singleton = UpdateService()
    return _singleton


__all__ = ["UpdateService", "get_update_service"]
