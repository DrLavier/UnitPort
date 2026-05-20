"""GitHub Release tarball/zipball transport.

Accepts either:

- a full ``https://github.com/<owner>/<repo>/releases/tag/<tag>`` URL —
  we resolve to the release's source zipball/tarball.
- a direct asset URL (``...releases/download/<tag>/<name>``) — we fetch
  that asset by-name.

The downloaded archive is streamed to a temp file under ``dest_dir``
(stay on the same volume so ``os.replace`` works for atomic moves),
then unpacked in-place. Cancellation closes the httpx stream mid-flight.
"""

from __future__ import annotations

import re
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Optional, Tuple

import httpx

from unitport_sdk import Config

from .base import (
    Transport,
    TransportContext,
    TransportError,
    directory_size_bytes,
)


_GITHUB_API_BASE = "https://api.github.com"

_RELEASE_TAG_URL_RE = re.compile(
    r"^https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/releases/tag/(?P<tag>[^/?#]+)/?$"
)
_RELEASE_LATEST_URL_RE = re.compile(
    r"^https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/releases/latest/?$"
)
_ASSET_URL_RE = re.compile(
    r"^https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/releases/download/"
    r"(?P<tag>[^/]+)/(?P<asset>[^/?#]+)$"
)


def _user_agent() -> str:
    ver = Config.get_value("System", "version", "0.0.0")
    return f"UnitPort/{ver} (resources-downloader)"


def _parse_release_url(url: str) -> Tuple[str, str, str, Optional[str]]:
    """Return ``(owner, repo, tag_or_'latest', asset_name_or_None)``."""
    s = (url or "").strip()
    if not s:
        raise TransportError("GitHub Release fetch requires a non-empty URL")
    m = _ASSET_URL_RE.match(s)
    if m:
        return m.group("owner"), m.group("repo"), m.group("tag"), m.group("asset")
    m = _RELEASE_TAG_URL_RE.match(s)
    if m:
        return m.group("owner"), m.group("repo"), m.group("tag"), None
    m = _RELEASE_LATEST_URL_RE.match(s)
    if m:
        return m.group("owner"), m.group("repo"), "latest", None
    raise TransportError(
        "URL must look like https://github.com/<owner>/<repo>/releases/{tag/<tag>"
        ",latest,download/<tag>/<asset>}"
    )


def _resolve_download_url(
    owner: str, repo: str, tag: str, asset_name: Optional[str], *,
    revision_override: str = "",
) -> Tuple[str, int]:
    """Hit the Releases API to resolve a download URL + Content-Length hint.

    ``revision_override`` (from ``source.revision``) takes precedence over
    a ``tag`` parsed from the URL when both are present — the UI may let
    a user paste ``.../releases`` and select a tag separately.

    When ``asset_name`` is None we return the source zipball URL; the
    archive contains a single top-level directory ``<owner>-<repo>-<sha>``
    which we strip during extraction. Size is 0 when unknown (zipball
    endpoints don't expose Content-Length on the API response).
    """
    effective_tag = (revision_override or tag).strip() or "latest"
    if effective_tag == "latest":
        api_url = f"{_GITHUB_API_BASE}/repos/{owner}/{repo}/releases/latest"
    else:
        api_url = (
            f"{_GITHUB_API_BASE}/repos/{owner}/{repo}/releases/tags/{effective_tag}"
        )
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": _user_agent(),
        "X-GitHub-Api-Version": "2022-11-28",
    }
    try:
        resp = httpx.get(api_url, headers=headers, timeout=15.0)
    except httpx.RequestError as e:
        raise TransportError(f"GitHub API request failed: {e}") from e
    if resp.status_code // 100 != 2:
        raise TransportError(
            f"GitHub API returned HTTP {resp.status_code} for {api_url!r}"
        )
    try:
        payload = resp.json()
    except Exception as e:
        raise TransportError(f"GitHub API returned non-JSON: {e}") from e
    if not isinstance(payload, dict):
        raise TransportError("GitHub API response is not an object")

    if asset_name:
        for a in payload.get("assets") or []:
            if not isinstance(a, dict):
                continue
            if str(a.get("name", "")) == asset_name:
                return (
                    str(a.get("browser_download_url", "") or ""),
                    int(a.get("size", 0) or 0),
                )
        raise TransportError(
            f"release {effective_tag!r} has no asset named {asset_name!r}"
        )

    # No specific asset → use source zipball.
    zip_url = str(payload.get("zipball_url", "") or "")
    if not zip_url:
        raise TransportError("release payload missing zipball_url")
    return zip_url, 0


class GitHubReleaseTransport(Transport):
    name = "github_release"

    def fetch(self, ctx: TransportContext) -> int:
        owner, repo, tag, asset_name = _parse_release_url(ctx.source.url)
        download_url, content_length = _resolve_download_url(
            owner, repo, tag, asset_name, revision_override=ctx.source.revision,
        )
        ctx.cancel_check()

        dest = ctx.dest_dir
        if dest.exists() and any(dest.iterdir()):
            raise TransportError(
                f"destination already exists and is non-empty: {dest!s}"
            )
        dest.mkdir(parents=True, exist_ok=True)

        # Stream into a sibling temp file so partial downloads don't pollute dest.
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=_archive_suffix(asset_name, download_url),
            dir=str(dest.parent), delete=False,
        ) as tmp_fh:
            tmp_path = Path(tmp_fh.name)
            try:
                self._stream_to(tmp_fh, download_url, content_length, ctx)
            except BaseException:
                tmp_fh.close()
                tmp_path.unlink(missing_ok=True)
                raise

        try:
            ctx.on_progress(0.9, f"extracting {tmp_path.name}")
            _extract_archive(tmp_path, dest, strip_top_level=asset_name is None)
        finally:
            tmp_path.unlink(missing_ok=True)

        ctx.cancel_check()
        size = directory_size_bytes(dest)
        ctx.on_progress(1.0, f"{size // (1024 * 1024)} MiB on disk")
        return size

    def _stream_to(self, fh, url: str, content_length: int, ctx: TransportContext) -> None:
        headers = {"User-Agent": _user_agent()}
        try:
            with httpx.stream(
                "GET", url, headers=headers,
                timeout=httpx.Timeout(30.0, connect=15.0),
                follow_redirects=True,
            ) as resp:
                if resp.status_code // 100 != 2:
                    raise TransportError(
                        f"download URL returned HTTP {resp.status_code}: {url}"
                    )
                # When the API didn't expose a size, fall back to the
                # Content-Length on the actual download (browser_download_url
                # for assets does populate it).
                total = content_length or int(resp.headers.get("Content-Length", 0) or 0)
                downloaded = 0
                for chunk in resp.iter_bytes(chunk_size=1 << 16):
                    if not chunk:
                        continue
                    ctx.cancel_check()
                    fh.write(chunk)
                    downloaded += len(chunk)
                    frac = (downloaded / total) * 0.9 if total > 0 else 0.0
                    ctx.on_progress(
                        min(frac, 0.9),
                        _format_bytes_progress(downloaded, total),
                    )
        except httpx.RequestError as e:
            raise TransportError(f"download stream failed: {e}") from e


def _archive_suffix(asset_name: Optional[str], url: str) -> str:
    s = (asset_name or url).lower()
    for suf in (".tar.gz", ".tgz", ".zip", ".tar"):
        if s.endswith(suf):
            return suf
    # Zipball default.
    return ".zip"


def _format_bytes_progress(done: int, total: int) -> str:
    mib = 1024 * 1024
    if total > 0:
        return f"{done // mib} / {total // mib} MiB"
    return f"{done // mib} MiB"


def _extract_archive(archive: Path, dest: Path, *, strip_top_level: bool) -> None:
    """Unpack ``archive`` into ``dest``.

    ``strip_top_level=True`` peels the single root directory that GitHub
    zipballs / source tarballs wrap their contents in (e.g.
    ``escontrela-AMP_for_hardware-abc123/``).
    """
    suffix = archive.suffix.lower()
    name_lower = archive.name.lower()
    if name_lower.endswith(".tar.gz") or name_lower.endswith(".tgz") or suffix == ".tar":
        mode = "r:gz" if name_lower.endswith((".tar.gz", ".tgz")) else "r:"
        with tarfile.open(archive, mode) as tf:
            _safe_extract_tar(tf, dest, strip_top_level=strip_top_level)
        return
    if suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            _safe_extract_zip(zf, dest, strip_top_level=strip_top_level)
        return
    raise TransportError(f"unsupported archive format: {archive.name}")


def _safe_extract_tar(tf: tarfile.TarFile, dest: Path, *, strip_top_level: bool) -> None:
    members = tf.getmembers()
    prefix = _common_prefix(m.name for m in members) if strip_top_level else ""
    for m in members:
        name = m.name
        if prefix and name == prefix:
            continue
        if prefix and name.startswith(prefix + "/"):
            name = name[len(prefix) + 1:]
        out = dest / name
        _guard_path_traversal(out, dest)
        if m.isdir():
            out.mkdir(parents=True, exist_ok=True)
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        fobj = tf.extractfile(m)
        if fobj is None:
            continue
        with open(out, "wb") as fh:
            shutil.copyfileobj(fobj, fh)


def _safe_extract_zip(zf: zipfile.ZipFile, dest: Path, *, strip_top_level: bool) -> None:
    names = zf.namelist()
    prefix = _common_prefix(names) if strip_top_level else ""
    for name in names:
        rel = name
        if prefix and rel == prefix + "/":
            continue
        if prefix and rel.startswith(prefix + "/"):
            rel = rel[len(prefix) + 1:]
        if not rel:
            continue
        out = dest / rel
        _guard_path_traversal(out, dest)
        if name.endswith("/"):
            out.mkdir(parents=True, exist_ok=True)
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(name) as src, open(out, "wb") as fh:
            shutil.copyfileobj(src, fh)


def _common_prefix(names) -> str:
    """Return the single top-level directory shared by all members.

    Returns ``""`` if the archive has multiple roots or no root structure
    (i.e. files at the archive's root); the extractor then falls through
    to a non-stripping extract.
    """
    roots = set()
    for n in names:
        head = n.split("/", 1)[0]
        if head:
            roots.add(head)
    if len(roots) == 1:
        return next(iter(roots))
    return ""


def _guard_path_traversal(out: Path, dest: Path) -> None:
    """Refuse archive members that resolve outside ``dest``."""
    try:
        out.resolve().relative_to(dest.resolve())
    except ValueError:
        raise TransportError(
            f"archive contains path traversal entry: {out!s}"
        )


__all__ = ["GitHubReleaseTransport"]
