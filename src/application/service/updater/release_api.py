"""GitHub Releases REST wrapper.

Pure-httpx, no Qt. Mirrors the error-handling shape of
``application.service.auth.supabase_client`` (try/except
``httpx.RequestError`` raising a domain exception, explicit timeout).
"""

from __future__ import annotations

from typing import List, Optional

import httpx

from unitport_sdk import Config

from .release_info import ReleaseAsset, ReleaseInfo


# Public GitHub REST endpoint. The project is public so no auth header is
# required (unauthenticated calls share a 60 req/h/IP budget — the
# updater service throttles to once per 6 h, well below the cap).
_GITHUB_API_BASE = "https://api.github.com"


class UpdateNetworkError(RuntimeError):
    """Network-side failure during a release-API call (timeout, DNS, TLS, ...).

    Distinct from parse errors so callers (UpdateService) can log a
    warning instead of an error for the common "offline at startup" case.
    """


class UpdateParseError(RuntimeError):
    """Server responded but the payload is not a valid release JSON."""


def _user_agent() -> str:
    """Build a polite User-Agent string identifying the local app build.

    GitHub asks every unauthenticated client to send a UA. The local
    version doubles as a build identifier in their server logs.
    """
    ver = Config.get_value("System", "version", "0.0.0")
    return f"UnitPort/{ver} (in-app updater)"


def _normalize_version(tag: str) -> str:
    """Strip the leading ``v`` from a tag name (``v0.9.0`` -> ``0.9.0``).

    The git tag convention is ``v<version>`` per CLAUDE.md §1.6, and the
    GitHub API returns the tag with the prefix. ``packaging.version``
    accepts both forms but we normalize here so the rest of the package
    only sees the bare semver string.
    """
    t = (tag or "").strip()
    if t.lower().startswith("v"):
        return t[1:]
    return t


def _parse_release(payload: dict) -> ReleaseInfo:
    tag = str(payload.get("tag_name") or "").strip()
    if not tag:
        raise UpdateParseError("release payload missing tag_name")
    assets_raw = payload.get("assets") or []
    assets = []
    for a in assets_raw:
        try:
            assets.append(
                ReleaseAsset(
                    name=str(a.get("name") or ""),
                    browser_download_url=str(a.get("browser_download_url") or ""),
                    size=int(a.get("size") or 0),
                    content_type=str(a.get("content_type") or ""),
                )
            )
        except (TypeError, ValueError):
            continue
    return ReleaseInfo(
        tag=tag,
        version=_normalize_version(tag),
        name=str(payload.get("name") or ""),
        body=str(payload.get("body") or ""),
        html_url=str(payload.get("html_url") or ""),
        published_at=str(payload.get("published_at") or ""),
        zipball_url=str(payload.get("zipball_url") or ""),
        tarball_url=str(payload.get("tarball_url") or ""),
        prerelease=bool(payload.get("prerelease") or False),
        assets=assets,
    )


def fetch_latest_release(
    repo: str = "DrLavier/UnitPort",
    *,
    timeout: float = 8.0,
    include_prereleases: bool = False,
) -> ReleaseInfo:
    """Fetch the latest published release.

    Args:
        repo: ``owner/name`` slug; defaults to the project repo.
        timeout: Request timeout in seconds. 8 s covers cross-region
            jitter while staying well under any UI freeze threshold.
        include_prereleases: When False (default), hit
            ``/releases/latest`` which already excludes drafts +
            pre-releases server-side. When True, list all releases and
            pick the first non-draft (which may be a pre-release).

    Returns:
        A populated :class:`ReleaseInfo`.

    Raises:
        UpdateNetworkError: any ``httpx.RequestError`` (timeout, DNS, TLS).
        UpdateParseError: the response was not a valid JSON release object
            or the status code was not 2xx.
    """
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": _user_agent(),
        "X-GitHub-Api-Version": "2022-11-28",
    }
    url = f"{_GITHUB_API_BASE}/repos/{repo}/releases"
    if not include_prereleases:
        url = f"{url}/latest"
    try:
        resp = httpx.get(url, headers=headers, timeout=timeout)
    except httpx.RequestError as exc:
        raise UpdateNetworkError(str(exc)) from exc
    if resp.status_code // 100 != 2:
        raise UpdateParseError(
            f"GitHub Releases API returned HTTP {resp.status_code}: "
            f"{resp.text[:200]}"
        )
    try:
        payload = resp.json()
    except Exception as exc:
        raise UpdateParseError(f"invalid JSON: {exc}") from exc
    if include_prereleases:
        if not isinstance(payload, list):
            raise UpdateParseError("expected a list of releases")
        for entry in payload:
            if isinstance(entry, dict) and not entry.get("draft"):
                return _parse_release(entry)
        raise UpdateParseError("no non-draft releases found")
    if not isinstance(payload, dict):
        raise UpdateParseError("expected a release object")
    return _parse_release(payload)


def fetch_releases(
    repo: str = "DrLavier/UnitPort",
    n: int = 3,
    *,
    timeout: float = 8.0,
    include_prereleases: bool = False,
) -> List[ReleaseInfo]:
    """Fetch up to ``n`` most-recent releases, newest first.

    Hits ``/repos/{repo}/releases?per_page=<page>`` and filters draft +
    pre-release entries client-side. ``page`` is sized a bit larger than
    ``n`` so the loop still has candidates after skipping drafts. Raises
    the same exceptions as :func:`fetch_latest_release` (the homepage
    fetch task catches both and degrades to an empty list).
    """
    if n <= 0:
        return []
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": _user_agent(),
        "X-GitHub-Api-Version": "2022-11-28",
    }
    page = max(n * 2, 5)
    url = f"{_GITHUB_API_BASE}/repos/{repo}/releases?per_page={page}"
    try:
        resp = httpx.get(url, headers=headers, timeout=timeout)
    except httpx.RequestError as exc:
        raise UpdateNetworkError(str(exc)) from exc
    if resp.status_code // 100 != 2:
        raise UpdateParseError(
            f"GitHub Releases API returned HTTP {resp.status_code}: "
            f"{resp.text[:200]}"
        )
    try:
        payload = resp.json()
    except Exception as exc:
        raise UpdateParseError(f"invalid JSON: {exc}") from exc
    if not isinstance(payload, list):
        raise UpdateParseError("expected a list of releases")
    out: List[ReleaseInfo] = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        if entry.get("draft"):
            continue
        if not include_prereleases and entry.get("prerelease"):
            continue
        try:
            out.append(_parse_release(entry))
        except UpdateParseError:
            continue
        if len(out) >= n:
            break
    return out


__all__ = [
    "UpdateNetworkError",
    "UpdateParseError",
    "fetch_latest_release",
    "fetch_releases",
]
