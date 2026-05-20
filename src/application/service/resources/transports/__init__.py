"""Transport implementations for the resources downloader.

A transport knows how to materialise a single asset (one ``ResourceEntry``)
on disk given its ``ResourceSource``. Each transport is a thin wrapper
around an underlying client (git CLI / httpx / huggingface_hub) and is
**stateless** — instantiated per download by ``ResourceManager``.
"""

from __future__ import annotations

from .base import (
    Transport,
    TransportError,
    TransportContext,
    ProgressFn,
    CancelCheckFn,
    repo_name_from_url,
    safe_dir_name,
)
from .github_clone import GitHubCloneTransport
from .github_release import GitHubReleaseTransport
from .huggingface import HuggingFaceTransport


__all__ = [
    "Transport",
    "TransportError",
    "TransportContext",
    "ProgressFn",
    "CancelCheckFn",
    "repo_name_from_url",
    "safe_dir_name",
    "GitHubCloneTransport",
    "GitHubReleaseTransport",
    "HuggingFaceTransport",
]
