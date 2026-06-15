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
    """Build a ``tqdm`` subclass that pumps ``ctx.on_progress``.

    We subclass the real ``tqdm`` rather than fake it. ``huggingface_hub``
    drives downloads through ``tqdm.contrib.concurrent.thread_map``, which
    calls ``tqdm_class.get_lock()`` / ``set_lock()`` (the classmethods a
    hand-rolled fake lacks — the source of the ``has no 'get_lock'`` crash)
    and iterates the bar to consume the worker pool's result generator (so a
    fake ``__iter__`` that yields nothing would swallow per-file download
    exceptions — an illusion-of-success failure, §8). Inheriting from
    ``tqdm`` gives correct ``get_lock``/``set_lock``/``__iter__``/``total``
    bookkeeping and exception propagation for free; we only override
    ``update``/``refresh`` to mirror progress into the UI callback (and to
    raise on cancellation), and redirect terminal output to a null sink.
    """
    # WHY KEPT: tqdm ships as a hard dependency of huggingface_hub, so it is
    # importable on every path that reaches this transport (§1.8(a)).
    from tqdm import tqdm as _BaseTqdm

    class _NullSink:
        """Swallow tqdm's terminal writes — we render via ``ctx.on_progress``."""

        def write(self, *_args, **_kwargs) -> int:
            return 0

        def flush(self, *_args, **_kwargs) -> None:
            pass

    class _Tqdm(_BaseTqdm):
        def __init__(self, *args, **kwargs):
            # huggingface_hub passes ``name=`` (its own tqdm subclass pops it);
            # the base tqdm rejects it. Force the bar enabled and silent so our
            # callback is the sole renderer regardless of global HF settings.
            kwargs.pop("name", None)
            kwargs["disable"] = False
            kwargs.setdefault("file", _NullSink())
            super().__init__(*args, **kwargs)
            self._emit()

        def update(self, n=1):
            ctx.cancel_check()
            ret = super().update(n)
            self._emit()
            return ret

        def refresh(self, *args, **kwargs):
            ret = super().refresh(*args, **kwargs)
            self._emit()
            return ret

        def _emit(self) -> None:
            total = self.total or 0
            n = self.n or 0
            desc = (self.desc or "").strip().rstrip(":")
            if total > 0:
                frac = max(0.05, min(n / total, 0.98))
                pct = int(min(n / total, 1.0) * 100)
                label = f"{desc} {pct}%".strip()
            else:
                frac = 0.5
                label = desc or f"{n}"
            ctx.on_progress(frac, label)

    return _Tqdm


__all__ = ["HuggingFaceTransport"]
