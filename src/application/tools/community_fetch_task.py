"""CommunityReleasesFetchTask — fetch GitHub Releases for the homepage.

Background worker for :class:`HomepageCommunityCard`. Wraps
``release_api.fetch_releases`` so the network round-trip happens on a
TasksManager worker thread; the card listens to ``task_finished`` and
populates its right column with the returned list of :class:`ReleaseInfo`.

Both ``UpdateNetworkError`` and ``UpdateParseError`` are downgraded to
``log_warning`` + empty-list returns: a missing news feed is not a fatal
condition and the card should render its "could not load" placeholder
rather than propagating the exception into the UI thread.
"""

from __future__ import annotations

from typing import Any

from unitport_sdk import Task

from application.service.updater.release_api import (
    UpdateNetworkError,
    UpdateParseError,
    fetch_releases,
)


class CommunityReleasesFetchTask(Task):
    """One-shot release-list fetch for the homepage Community card."""

    def __init__(self, *, n: int = 3) -> None:
        super().__init__(name="community-releases-fetch")
        self._n = n

    def run(self) -> Any:  # type: ignore[override]
        self.log_debug(f"fetching latest {self._n} releases")
        try:
            releases = fetch_releases(n=self._n)
        except UpdateNetworkError as exc:
            self.log_warning(f"network: {exc}")
            return []
        except UpdateParseError as exc:
            self.log_warning(f"parse: {exc}")
            return []
        self.log_debug(f"got {len(releases)} release(s)")
        return releases


__all__ = ["CommunityReleasesFetchTask"]
