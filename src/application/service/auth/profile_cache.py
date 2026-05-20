"""Non-sensitive user profile cache — JSON file at ``<USER_CONFIG_DIR>/session.json``.

Split from the sensitive ``refresh_token`` (which lives in the OS keyring via
:class:`~application.service.auth.secure.SecureCredentialStore`). This module
handles ONLY the non-sensitive cached profile fields (``email``, ``display_name``,
``avatar_url``, ``user_id``, ``provider``).

The cache is kept under ``Paths.USER_CONFIG_DIR`` so it lives outside the
project tree (per CLAUDE.md: user state never goes in src/).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from unitport_sdk import Paths, log_warning, push_data, read_data

_SESSION_REL = "session.json"


def _session_file() -> Path:
    """Lazy resolve of the session.json path inside the LIVE USER_CONFIG_DIR.

    Resolved at call time (not module load) so hot workspace switches via
    ``user_workspace.set_user_workspace`` are picked up without restarting
    the process.
    """
    return Paths.USER_CONFIG_DIR / _SESSION_REL


@dataclass
class CachedProfile:
    user_id: str = ""
    email: str = ""
    display_name: str = ""
    avatar_url: str = ""
    provider: str = ""

    @property
    def is_empty(self) -> bool:
        return not (self.user_id or self.email)


def save_cached_profile(profile: CachedProfile) -> None:
    if not push_data(_SESSION_REL, asdict(profile)):
        log_warning(f"[auth:profile_cache] failed to write {_session_file()}")


def load_cached_profile() -> CachedProfile:
    if not _session_file().exists():
        return CachedProfile()
    data = read_data(_session_file())
    if not isinstance(data, dict):
        return CachedProfile()
    return CachedProfile(
        user_id=str(data.get("user_id", "")),
        email=str(data.get("email", "")),
        display_name=str(data.get("display_name", "")),
        avatar_url=str(data.get("avatar_url", "")),
        provider=str(data.get("provider", "")),
    )


def clear_cached_profile() -> None:
    if _session_file().exists():
        try:
            _session_file().unlink()
        except OSError as exc:
            log_warning(f"[auth:profile_cache] cannot delete {_session_file()}: {exc}")


__all__ = [
    "CachedProfile",
    "save_cached_profile",
    "load_cached_profile",
    "clear_cached_profile",
]
