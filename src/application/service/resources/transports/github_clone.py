# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""GitHub clone transport.

Models the same pattern as ``service.models.menagerie_manager``:
``git clone --filter=blob:none --depth 1`` for full clones, optionally
followed by ``sparse-checkout set <subpath>`` when the user supplies a
``source.subpath``.

Cancellation: the subprocess Popen handle is held in the transport
instance for the duration of the fetch; ``cancel_check`` raises
``TaskCancelledException``, which the caller propagates up to
``DownloadResourceTask`` where the ``finally`` clause terminates the
process and cleans the partial tree.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Optional

from .base import (
    Transport,
    TransportContext,
    TransportError,
    directory_size_bytes,
)


_PROGRESS_RE = re.compile(r"(\d+)\s*%")


def _git_available() -> bool:
    return shutil.which("git") is not None


class GitHubCloneTransport(Transport):
    """``git clone`` into ``dest_dir``, optionally sparse-checking a subpath.

    The wire protocol uses ``--filter=blob:none --depth 1 --progress``;
    blobs are fetched lazily on checkout so a sparse-checkout of a single
    subdir does not pull the full history. Total bytes are computed by
    walking the work tree after checkout (git itself doesn't expose the
    materialised size, only blob counts).
    """

    name = "github_clone"

    def __init__(self) -> None:
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()

    def fetch(self, ctx: TransportContext) -> int:
        if not _git_available():
            raise TransportError(
                "`git` is not on PATH — install Git for Windows and retry"
            )
        url = ctx.source.url.strip()
        if not url:
            raise TransportError("GitHub clone requires a non-empty URL")
        revision = ctx.source.revision.strip()
        subpath = ctx.source.subpath.strip()

        dest = ctx.dest_dir
        if dest.exists() and any(dest.iterdir()):
            raise TransportError(
                f"destination already exists and is non-empty: {dest!s}"
            )
        dest.parent.mkdir(parents=True, exist_ok=True)

        ctx.cancel_check()
        ctx.on_progress(0.0, f"git clone {url}")

        # Step 1: clone (no checkout if a subpath is requested — so we can
        # configure sparse-checkout before pulling blobs).
        clone_args = [
            "git", "clone",
            "--progress",
            "--filter=blob:none",
            "--depth", "1",
        ]
        if subpath:
            clone_args.append("--no-checkout")
        if revision:
            # `--branch` works for branches and lightweight tags. For a
            # bare commit SHA, fall back to a full fetch + checkout.
            clone_args.extend(["--branch", revision])
        clone_args.extend([url, str(dest)])

        self._run_streaming(clone_args, ctx, span_start=0.0, span_end=0.6)

        if subpath:
            ctx.cancel_check()
            ctx.on_progress(0.6, f"sparse-checkout: {subpath}")
            self._run_streaming(
                [
                    "git", "-C", str(dest),
                    "sparse-checkout", "init", "--cone",
                ],
                ctx, span_start=0.6, span_end=0.65,
            )
            # ``sparse-checkout set`` accepts top-level path patterns.
            self._run_streaming(
                ["git", "-C", str(dest), "sparse-checkout", "set", subpath],
                ctx, span_start=0.65, span_end=0.7,
            )
            self._run_streaming(
                ["git", "-C", str(dest), "checkout"],
                ctx, span_start=0.7, span_end=0.95,
            )

        ctx.cancel_check()
        size = directory_size_bytes(dest)
        ctx.on_progress(1.0, f"{size // (1024 * 1024)} MiB on disk")
        return size

    def _run_streaming(
        self,
        args,
        ctx: TransportContext,
        *,
        span_start: float,
        span_end: float,
    ) -> None:
        """Run a git command, streaming progress to ``ctx.on_progress``.

        ``span_start`` / ``span_end`` are the [0,1] fraction range this
        single command occupies in the overall fetch progress. Inside that
        span, we interpolate based on the percentage git itself reports
        on stderr (``Receiving objects:  42% (...)``); when no percentage
        is parseable we hold the start of the span and let the textual
        status line carry the progress.
        """
        with self._lock:
            proc = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            self._proc = proc

        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                msg = line.strip()
                if not msg:
                    continue
                # Cooperative cancellation point.
                try:
                    ctx.cancel_check()
                except BaseException:
                    self._terminate_locked()
                    raise
                pct = _parse_percent(msg)
                if pct is not None:
                    frac = span_start + (span_end - span_start) * (pct / 100.0)
                    ctx.on_progress(min(frac, span_end), msg)
                else:
                    ctx.on_progress(span_start, msg)
            rc = proc.wait()
            if rc != 0:
                raise TransportError(
                    f"`{' '.join(args[:3])}` exited with code {rc}"
                )
        finally:
            with self._lock:
                self._proc = None

    def _terminate_locked(self) -> None:
        """Kill the running subprocess on cancel. Idempotent."""
        with self._lock:
            proc = self._proc
            if proc is None:
                return
            try:
                proc.terminate()
            except OSError:
                pass


def _parse_percent(line: str) -> Optional[float]:
    m = _PROGRESS_RE.search(line)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


__all__ = ["GitHubCloneTransport"]
