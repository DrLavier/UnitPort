# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Lazy singleton accessor for the asset-browser provider.

Mirrors ``resources.manager.get_resource_manager``. The UI obtains its
provider exclusively through here, so the backend round swaps the concrete
class in one place (return a different implementation) without touching any
widget.
"""

from __future__ import annotations

import threading
from typing import Optional

from .stub_provider import DefaultAssetBrowserProvider

_singleton: Optional[DefaultAssetBrowserProvider] = None
_singleton_lock = threading.Lock()


def get_asset_browser_provider() -> DefaultAssetBrowserProvider:
    """Return the process-wide asset-browser provider (lazy, thread-safe)."""
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = DefaultAssetBrowserProvider()
    return _singleton


__all__ = ["get_asset_browser_provider"]
