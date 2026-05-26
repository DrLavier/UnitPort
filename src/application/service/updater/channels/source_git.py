# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""SourceGitChannel — update by switching to a release tag in a git checkout.

Preconditions: ``PROJECT_ROOT/.git/`` exists AND ``git`` is on PATH.
Refuses to operate when the working tree is in detached HEAD (user is
mid-experiment — destroying that state is worse than refusing to update).
Auto-stashes dirty trees with a recognizable marker so manual recovery
is straightforward.
"""

from __future__ import annotations

import datetime as _dt
import shutil
import subprocess
from typing import List, Optional, Tuple

from unitport_sdk import Paths

from ..release_info import ApplyResult, ReleaseInfo
from .base import ProgressCallback, UpdateChannel


_STASH_MARKER_PREFIX = "unitport-updater-autostash"


def _run_git(args: List[str], *, timeout: float = 60.0) -> Tuple[int, str]:
    """Run ``git -C PROJECT_ROOT <args>`` and return (returncode, stdout+stderr).

    Never raises on non-zero exit — the channel itself decides whether
    a non-zero rc is fatal (e.g. ``status --porcelain`` returning 0
    is success regardless of output).
    """
    cmd = ["git", "-C", str(Paths.PROJECT_ROOT), *args]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return 124, f"git {args[0] if args else ''} timed out after {timeout}s"
    except FileNotFoundError:
        return 127, "git not on PATH"
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, out.strip()


class SourceGitChannel(UpdateChannel):
    id = "source-git"

    def is_available(self) -> bool:
        if shutil.which("git") is None:
            return False
        if not (Paths.PROJECT_ROOT / ".git").is_dir():
            return False
        return True

    def describe(self) -> str:
        return "git fetch + checkout release tag (preserves untracked files)"

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

        report(0.05, "fetching tags")
        rc, out = _run_git(["fetch", "--tags", "--prune", "origin"])
        if rc != 0:
            return ApplyResult(
                ok=False,
                message=f"git fetch failed (rc={rc}): {out[:200]}",
            )

        report(0.35, "checking working tree")
        rc, branch = _run_git(["symbolic-ref", "--short", "-q", "HEAD"])
        if rc != 0 or not branch.strip():
            return ApplyResult(
                ok=False,
                message=(
                    "repo is in detached HEAD; refusing to update so the "
                    "current state isn't lost. Check out a branch first, "
                    "then retry."
                ),
            )

        rc, dirty = _run_git(["status", "--porcelain"])
        if rc != 0:
            return ApplyResult(
                ok=False,
                message=f"git status failed (rc={rc}): {dirty[:200]}",
            )
        autostash = False
        if dirty.strip():
            stamp = _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            marker = f"{_STASH_MARKER_PREFIX}-{stamp}"
            report(0.45, "stashing local changes")
            rc, msg = _run_git(["stash", "push", "-u", "-m", marker])
            if rc != 0:
                return ApplyResult(
                    ok=False,
                    message=f"git stash failed (rc={rc}): {msg[:200]}",
                )
            autostash = True

        report(0.65, f"checking out v{release.version}")
        rc, msg = _run_git(["checkout", release.tag])
        if rc != 0:
            # Roll the autostash back so user's work isn't trapped.
            if autostash:
                _run_git(["stash", "pop"])
            return ApplyResult(
                ok=False,
                message=f"git checkout {release.tag} failed (rc={rc}): {msg[:200]}",
            )

        report(1.0, "checkout complete")
        tail = (
            " (local changes are kept in `git stash list` under "
            f"{_STASH_MARKER_PREFIX}-*; run `git stash pop` to recover)"
            if autostash else ""
        )
        return ApplyResult(
            ok=True,
            requires_restart=True,
            message=f"checked out {release.tag}{tail}",
        )


__all__ = ["SourceGitChannel"]
