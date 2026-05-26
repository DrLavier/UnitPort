# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""SourceZipChannel — download release zipball, smart-merge into PROJECT_ROOT.

Used when no ``.git/`` exists or no ``git`` binary is on PATH. The
zipball download is hit on GitHub's ``codeload`` redirect; the streaming
write surfaces progress per chunk.

The merge step is where user data is preserved: any path matching the
``PRESERVE_PATHS`` allowlist is **skipped** (existing file untouched,
new file under that prefix not installed). The list is intentionally
broad — it errs on the side of preserving rather than overwriting.
"""

from __future__ import annotations

import os
import shutil
import stat
import zipfile
from pathlib import Path
from typing import Iterable, List, Optional

import httpx

from unitport_sdk import Paths

from ..release_info import ApplyResult, ReleaseInfo
from .base import ProgressCallback, UpdateChannel


# Project-relative path prefixes that the update flow must never touch.
# Anything matching one of these (file or directory) is silently skipped
# during the merge. Trailing slash distinguishes a directory prefix
# from a literal-file match.
#
# Note: only path prefixes are checked. Adding a new top-level
# user-content directory later requires appending it here. A future
# release_manifest.json shipped in the repo would let releases ship
# their own preserve list, but that's deferred.
PRESERVE_PATHS = (
    "custom_mods/",
    "localisation/EN/",
    "localisation/ZH/",
    "projects/",
    "knowledge_base/",
    ".venv311/",
    ".pip-cache/",
    ".pip-tmp/",
    "log/",
    "src/runtime/",
    "runtime/",
    "__pycache__/",
    ".git/",
    ".pytest_cache/",
    ".cache/",
)


# Download chunk size — 256 KB strikes a balance between progress
# granularity (~hundreds of callbacks for a 100 MB zip) and syscall
# overhead.
_CHUNK_BYTES = 256 * 1024


def _should_preserve(rel_posix: str) -> bool:
    """Whether a project-relative POSIX path is on the preserve allowlist."""
    p = rel_posix.replace("\\", "/")
    if p.startswith("./"):
        p = p[2:]
    for prefix in PRESERVE_PATHS:
        if p == prefix.rstrip("/") or p.startswith(prefix):
            return True
    return False


def _gc_stale_downloads(root: Path, max_age_hours: int = 24) -> None:
    """Best-effort cleanup of yesterday's zipballs."""
    if not root.exists():
        return
    import time
    cutoff = time.time() - max_age_hours * 3600
    for entry in root.iterdir():
        try:
            if entry.stat().st_mtime >= cutoff:
                continue
            if entry.is_dir():
                shutil.rmtree(entry, ignore_errors=True)
            else:
                entry.unlink(missing_ok=True)
        except Exception:
            continue


def _atomic_write(dest: Path, src: Path) -> None:
    """Replace ``dest`` with ``src`` atomically.

    ``src`` and ``dest`` live in the same directory by construction
    (caller writes the temp file next to the final destination), so
    ``os.replace`` is guaranteed in-volume and never raises EXDEV.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    os.replace(str(src), str(dest))


class SourceZipChannel(UpdateChannel):
    id = "source-zip"

    def is_available(self) -> bool:
        # No preconditions beyond a writable PROJECT_ROOT and httpx
        # being importable. Both are guaranteed by the install/bootstrap
        # so this channel is effectively the always-available fallback.
        return True

    def describe(self) -> str:
        return "download release zip + smart-merge (preserves user content)"

    def apply(
        self,
        release: ReleaseInfo,
        progress_cb: Optional[ProgressCallback] = None,
    ) -> ApplyResult:
        def report(frac: float, label: str) -> None:
            if progress_cb is not None:
                try:
                    progress_cb(frac, label)
                except Exception:
                    pass

        if not release.zipball_url:
            return ApplyResult(
                ok=False,
                message=f"release {release.tag} has no zipball_url",
            )

        work_root = Paths.USER_CONFIG_DIR / "_update"
        work_root.mkdir(parents=True, exist_ok=True)
        _gc_stale_downloads(work_root)

        zip_path = work_root / f"unitport-{release.version}.zip"
        extract_dir = work_root / f"unitport-{release.version}"

        # ---- 1. Download (streaming) -----------------------------------
        try:
            self._stream_download(release.zipball_url, zip_path, report)
        except httpx.RequestError as exc:
            return ApplyResult(ok=False, message=f"download failed: {exc}")
        except Exception as exc:
            return ApplyResult(ok=False, message=f"download error: {exc}")

        # ---- 2. Extract to temp dir ------------------------------------
        report(0.55, "extracting archive")
        if extract_dir.exists():
            shutil.rmtree(extract_dir, ignore_errors=True)
        extract_dir.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)
        except zipfile.BadZipFile as exc:
            return ApplyResult(ok=False, message=f"corrupt zip: {exc}")

        # GitHub zipballs unpack into a single top-level dir like
        # ``DrLavier-UnitPort-<sha>/``. Resolve dynamically rather
        # than guessing — the SHA changes per release.
        try:
            top = next(p for p in extract_dir.iterdir() if p.is_dir())
        except StopIteration:
            return ApplyResult(ok=False, message="zipball is empty")

        # ---- 3. Smart-merge into PROJECT_ROOT --------------------------
        report(0.65, "merging files")
        try:
            stats = self._merge_tree(top, Paths.PROJECT_ROOT, report)
        except PermissionError as exc:
            return ApplyResult(
                ok=False,
                message=(
                    f"permission denied while merging: {exc}. "
                    "Close any tool that may be holding a file in the "
                    "project tree and retry."
                ),
            )
        except Exception as exc:
            return ApplyResult(ok=False, message=f"merge failed: {exc}")

        # ---- 4. Cleanup the extracted dir (keep the zip for caching) ---
        shutil.rmtree(extract_dir, ignore_errors=True)

        report(1.0, "done")
        return ApplyResult(
            ok=True,
            requires_restart=True,
            message=(
                f"applied {release.tag}: {stats['written']} file(s) "
                f"updated, {stats['preserved']} preserved"
            ),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _stream_download(
        self,
        url: str,
        dest: Path,
        report,
    ) -> None:
        """Streaming GET with progress callbacks per chunk."""
        report(0.06, "connecting")
        dest.parent.mkdir(parents=True, exist_ok=True)
        # ``follow_redirects=True`` is required: GitHub returns a 302
        # to codeload.github.com.
        with httpx.stream(
            "GET", url, timeout=60.0, follow_redirects=True,
            headers={"User-Agent": "UnitPort/updater"},
        ) as resp:
            if resp.status_code // 100 != 2:
                raise RuntimeError(
                    f"zipball download HTTP {resp.status_code}"
                )
            total = 0
            try:
                content_length = int(resp.headers.get("content-length") or 0)
            except (TypeError, ValueError):
                content_length = 0
            with open(dest, "wb") as fh:
                for chunk in resp.iter_bytes(chunk_size=_CHUNK_BYTES):
                    if not chunk:
                        continue
                    fh.write(chunk)
                    total += len(chunk)
                    if content_length > 0:
                        # Map download to [0.06, 0.50].
                        frac = 0.06 + 0.44 * min(1.0, total / content_length)
                        report(frac, f"downloading {total // 1024} KB")
                    else:
                        report(0.30, f"downloading {total // 1024} KB")
        report(0.50, "download complete")

    def _merge_tree(
        self,
        src_root: Path,
        dst_root: Path,
        report,
    ) -> dict:
        """Walk ``src_root`` and atomically replace files into ``dst_root``.

        Returns a small stats dict ``{written, preserved}``.
        """
        all_files: List[Path] = [p for p in src_root.rglob("*") if p.is_file()]
        total = max(1, len(all_files))
        written = 0
        preserved = 0
        for i, src_file in enumerate(all_files, start=1):
            rel = src_file.relative_to(src_root)
            rel_posix = rel.as_posix()
            if _should_preserve(rel_posix):
                preserved += 1
                continue
            dest = dst_root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(dest.suffix + ".unitport-new")
            try:
                shutil.copy2(src_file, tmp)
                # POSIX: zipballs strip exec bits; restore for shell
                # scripts so ``./start.sh`` keeps working after an
                # update.
                if os.name == "posix" and src_file.suffix == ".sh":
                    try:
                        mode = tmp.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
                        tmp.chmod(mode)
                    except OSError:
                        pass
                _atomic_write(dest, tmp)
            except PermissionError:
                # Clean up the temp before re-raising so the next
                # attempt doesn't trip over leftovers.
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass
                raise
            written += 1
            # Map [0.65, 0.98].
            frac = 0.65 + 0.33 * (i / total)
            if i % 20 == 0 or i == total:
                report(frac, f"merging {i}/{total}")
        return {"written": written, "preserved": preserved}


__all__ = ["SourceZipChannel", "PRESERVE_PATHS"]
