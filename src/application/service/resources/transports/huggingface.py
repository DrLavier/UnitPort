# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""HuggingFace Hub transport.

Wraps :func:`huggingface_hub.snapshot_download`. The hub library handles
authentication, partial-file resume, and retries on its own; we only need
to plumb the ``allow_patterns`` filter (from ``source.subpath``) and the
``revision`` field, then translate its tqdm progress into our
``on_progress`` callback.

Cancellation: HuggingFace's downloader does not expose a clean abort
hook. We approximate by raising ``TaskCancelledException`` from inside
our tqdm wrapper's ``update()``; that aborts the active file mid-chunk
but in-flight HTTP requests for sibling files in the same snapshot will
still complete before the exception unwinds. The DownloadResourceTask's
``finally`` clause cleans up the partial directory afterward.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Optional

from .base import (
    Transport,
    TransportContext,
    TransportError,
    directory_size_bytes,
)


_HF_URL_RE = re.compile(
    r"^https?://huggingface\.co/(?:datasets/)?(?P<repo>[^/]+/[^/?#]+)"
)


def _normalize_repo_id(raw: str) -> str:
    """Accept a bare ``owner/name`` or a full HF URL; return ``owner/name``."""
    s = (raw or "").strip()
    if not s:
        raise TransportError("HuggingFace fetch requires a repo_id or HF URL")
    m = _HF_URL_RE.match(s)
    if m:
        return m.group("repo")
    # Treat as bare repo_id (HF allows org/name and user/name).
    if s.count("/") != 1:
        raise TransportError(
            f"HuggingFace repo_id must be 'owner/name', got {s!r}"
        )
    return s


def _split_patterns(raw: str) -> Optional[List[str]]:
    s = (raw or "").strip()
    if not s:
        return None
    return [p.strip() for p in re.split(r"[\s,]+", s) if p.strip()]


class HuggingFaceTransport(Transport):
    """Snapshot a HuggingFace repo into ``dest_dir``.

    The transport defaults to ``repo_type='model'`` for ``policy`` and
    ``repo_type='dataset'`` for ``motion``, but the URL form
    ``https://huggingface.co/datasets/<repo>`` overrides this regardless
    of the manager's asset_kind hint.
    """

    name = "huggingface"

    def __init__(self, *, repo_type_hint: str = "dataset") -> None:
        # ``repo_type_hint`` lets the manager set "dataset" for motion
        # downloads and "model" for policy downloads; the URL form
        # (``/datasets/<repo>``) overrides this at fetch time when the
        # user pasted a typed URL.
        self.repo_type_hint = repo_type_hint

    def fetch(self, ctx: TransportContext) -> int:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as e:  # WHY KEPT: optional-import pattern per §1.8(a)
            raise TransportError(
                "huggingface_hub is not installed (run install.bat)"
            ) from e

        repo_id = _normalize_repo_id(ctx.source.url)
        allow_patterns = _split_patterns(ctx.source.subpath)
        revision = ctx.source.revision.strip() or None
        repo_type = self._infer_repo_type(ctx.source.url)

        dest = ctx.dest_dir
        if dest.exists() and any(dest.iterdir()):
            raise TransportError(
                f"destination already exists and is non-empty: {dest!s}"
            )
        dest.mkdir(parents=True, exist_ok=True)

        ctx.cancel_check()
        ctx.on_progress(
            0.05,
            f"hf snapshot: {repo_id} ({repo_type})"
            + (f" @ {revision}" if revision else "")
        )

        # huggingface_hub uses environment variable HF_HUB_DISABLE_TELEMETRY for
        # noise reduction in CI; we don't enforce it here.
        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type=repo_type,
                revision=revision,
                allow_patterns=allow_patterns,
                local_dir=str(dest),
                tqdm_class=_make_tqdm_class(ctx),
            )
        except Exception as e:
            # huggingface_hub raises a mix of RepositoryNotFoundError /
            # RevisionNotFoundError / HfHubHTTPError; surface the type +
            # message so the UI card hint is actionable.
            raise TransportError(
                f"{type(e).__name__}: {e}"
            ) from e

        ctx.cancel_check()
        size = directory_size_bytes(dest)
        ctx.on_progress(1.0, f"{size // (1024 * 1024)} MiB on disk")
        return size

    def _infer_repo_type(self, url: str) -> str:
        s = (url or "").lower()
        if "huggingface.co/datasets/" in s:
            return "dataset"
        if "huggingface.co/spaces/" in s:
            return "space"
        if self.repo_type_hint in ("model", "dataset", "space"):
            return self.repo_type_hint
        return "model"


def _make_tqdm_class(ctx: TransportContext):
    """Build a tqdm-compatible class that pumps ``ctx.on_progress``."""

    class _Tqdm:
        def __init__(self, *args, **kwargs):
            self.total = kwargs.get("total") or 0
            self.desc = kwargs.get("desc") or ""
            self.n = 0

        def update(self, inc: int = 1) -> None:
            self.n += int(inc or 0)
            ctx.cancel_check()
            if self.total > 0:
                frac = max(0.05, min(self.n / self.total, 0.98))
            else:
                frac = 0.5
            label = (
                f"{self.desc}: {self.n} / {self.total}"
                if self.total > 0
                else f"{self.desc}: {self.n}"
            )
            ctx.on_progress(frac, label)

        def close(self) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.close()
            return False

        # Some HF versions iterate ``tqdm(iter)`` — keep a minimal __iter__.
        def __iter__(self):
            return iter(())

        def set_description(self, d, refresh: bool = True) -> None:  # noqa: D401
            self.desc = str(d or "")

        def set_postfix(self, *args, **kwargs) -> None:
            pass

        def refresh(self) -> None:
            pass

        @staticmethod
        def write(msg, *_, **__) -> None:  # tqdm.write
            return

    return _Tqdm


__all__ = ["HuggingFaceTransport"]
