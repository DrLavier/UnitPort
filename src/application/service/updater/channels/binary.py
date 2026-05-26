# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""BinaryInstallerChannel — stub for the (future) packaged-build update path.

Today this channel is never available: ``sys.frozen`` is False under
the venv'd Python launcher and no OS-specific installer assets are
attached to any release. The class is kept in the registry so that when
a binary build pipeline lands later, only ``binary.py`` needs to grow a
concrete ``apply()`` body — channel detection and the service layer
already know to prefer it.
"""

from __future__ import annotations

import sys
from typing import Optional

from ..release_info import ApplyResult, ReleaseInfo
from .base import ProgressCallback, UpdateChannel


class BinaryInstallerChannel(UpdateChannel):
    """Reserved-for-future packaged-build channel.

    Future contract (when implemented):

    - ``is_available()``: True iff ``sys.frozen`` AND a release asset
      whose name matches the current OS naming convention exists
      (``*-win64.exe`` / ``*-linux-x64.AppImage`` / ``*-macos.dmg``).
    - ``apply()``: download the matching asset to
      ``Paths.USER_CONFIG_DIR/_update/``, spawn the installer detached
      (``subprocess.Popen`` with ``DETACHED_PROCESS`` on Windows; mark
      AppImage executable + re-exec on Linux), then signal the
      ``RelauncherStrategy`` so the current process exits cleanly.
    """

    id = "binary"

    def is_available(self) -> bool:
        return bool(getattr(sys, "frozen", False))

    def describe(self) -> str:
        return "download OS-specific installer (not yet implemented)"

    def apply(
        self,
        release: ReleaseInfo,
        progress_cb: Optional[ProgressCallback] = None,
    ) -> ApplyResult:
        # Defensive: even if some future caller bypasses is_available()
        # this returns a clean ApplyResult rather than tossing an
        # exception across the worker boundary.
        return ApplyResult(
            ok=False,
            message=(
                "binary installer mode is reserved for future packaged "
                "releases and is not implemented in this build"
            ),
        )


__all__ = ["BinaryInstallerChannel"]
