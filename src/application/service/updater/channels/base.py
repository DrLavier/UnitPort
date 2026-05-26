# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""UpdateChannel abstract base class.

Each concrete channel knows how to apply a release in one specific
install mode (source-with-git, source-zip, binary installer). The
service layer picks the first available channel and calls ``apply()``;
no business logic lives in the base.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional

from ..release_info import ApplyResult, ReleaseInfo


# Channel.apply() takes an optional progress callback. fraction is in
# [0, 1]; label is a short human-readable status ("downloading",
# "extracting", "git checkout"). Channels may call this any number of
# times, but at least once before returning is good UX practice.
ProgressCallback = Callable[[float, str], None]


class UpdateChannel(ABC):
    """Strategy interface for applying an update in a specific install mode."""

    #: Stable short identifier ("source-git", "source-zip", "binary").
    #: Used in log lines and persisted state — never change a value
    #: once shipped.
    id: str = ""

    @abstractmethod
    def is_available(self) -> bool:
        """Whether this channel's preconditions hold *right now*.

        Cheap probes only (filesystem, ``shutil.which``); no network
        calls. The result is read by ``detect_channel()`` on every
        check, so transient changes (git installed mid-session, .git
        added/removed) are picked up without restart.
        """

    @abstractmethod
    def describe(self) -> str:
        """One-line human description, e.g. ``git checkout v0.9.0``.

        Used in the apply-update dialog so the user sees what's about
        to happen before clicking Apply.
        """

    @abstractmethod
    def apply(
        self,
        release: ReleaseInfo,
        progress_cb: Optional[ProgressCallback] = None,
    ) -> ApplyResult:
        """Perform the update. Blocking; runs on a TasksManager worker.

        Implementations must:
        - emit at least one progress callback (>0.0 and <1.0 minimum)
        - return an :class:`ApplyResult` on both success and recoverable
          failure (do NOT raise for "branch is dirty" / "detached HEAD"
          / "permission denied on one file"; raise only for unexpected
          programming errors)
        - never call back into Qt directly; the orchestrator owns the
          signal bus
        """


__all__ = ["UpdateChannel", "ProgressCallback"]
