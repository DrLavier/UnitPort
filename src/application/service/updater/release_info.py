# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Frozen dataclasses describing a GitHub release payload.

Single source of truth for the shape of the GitHub Releases API response
as consumed by the rest of the updater package. The release_api module
parses raw JSON into these; channels and the UI receive only these
typed objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class ReleaseAsset:
    """A single binary asset attached to a GitHub release.

    Used by the (future) BinaryInstallerChannel to pick the OS-matching
    download. SourceGitChannel + SourceZipChannel ignore assets and pull
    the source zipball instead.
    """

    name: str
    browser_download_url: str
    size: int = 0
    content_type: str = ""


@dataclass(frozen=True)
class ReleaseInfo:
    """One GitHub release.

    ``tag`` carries the leading ``v`` as returned by the API; ``version``
    is the same string with the ``v`` stripped, normalized for semver
    comparison via ``packaging.version.Version``.
    """

    tag: str
    version: str
    name: str = ""
    body: str = ""
    html_url: str = ""
    published_at: str = ""
    zipball_url: str = ""
    tarball_url: str = ""
    prerelease: bool = False
    assets: List[ReleaseAsset] = field(default_factory=list)


@dataclass(frozen=True)
class ApplyResult:
    """Outcome of UpdateChannel.apply()."""

    ok: bool
    requires_restart: bool = False
    message: str = ""


__all__ = ["ReleaseAsset", "ReleaseInfo", "ApplyResult"]
