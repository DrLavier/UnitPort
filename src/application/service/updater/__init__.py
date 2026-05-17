"""In-app update checker + applier.

Public API:

- :func:`get_update_service` — process-wide singleton facade.
- :class:`CheckUpdateTask` — startup-pipeline FuncTask sibling.
- :class:`ApplyUpdateTask` — user-triggered apply task.
- :class:`ReleaseInfo`, :class:`ApplyResult` — typed payloads
  exchanged with the UI.

See ``CLAUDE.md`` §1.6 for the version contract.
"""

from __future__ import annotations

from .apply_task import ApplyUpdateTask
from .check_task import CheckUpdateTask
from .release_api import UpdateNetworkError, UpdateParseError, fetch_latest_release
from .release_info import ApplyResult, ReleaseAsset, ReleaseInfo
from .service import UpdateService, get_update_service

__all__ = [
    "ApplyResult",
    "ApplyUpdateTask",
    "CheckUpdateTask",
    "ReleaseAsset",
    "ReleaseInfo",
    "UpdateNetworkError",
    "UpdateParseError",
    "UpdateService",
    "fetch_latest_release",
    "get_update_service",
]
