# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""UnitPort user authentication — Supabase-backed.

Public entry points:

    from application.service.auth import get_auth_manager, UserProfile

    mgr = get_auth_manager()
    mgr.authenticated.connect(lambda user: print(user.email))
    mgr.try_restore_session()

Session persistence:
    - refresh token  -> OS keyring via secure.SecureCredentialStore
    - cached profile -> Paths.USER_CONFIG_DIR / "session.json"

Import policy: ``deeplink_handler`` is eager (it only depends on PyQt and the
SDK, ~ms cost) so the Stage 0b single-instance guard in ``main.py`` can run
before the LoadingScreen paints. ``auth_manager`` / ``secure`` pull httpx +
keyring (~400 ms on Windows) and are loaded lazily via PEP 562 ``__getattr__``
the first time UserPanel / LoginDialog / ``_finalize`` references them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .deeplink_handler import (
    DeeplinkHandler,
    find_deeplink_url,
    install_single_instance_guard,
)

if TYPE_CHECKING:
    # Type-checker visibility without runtime cost.
    from .auth_manager import AuthManager, UserProfile, get_auth_manager
    from .secure import SecureCredentialStore, get_secure_store
    from .supabase_storage import (
        RemoteObject,
        StorageError,
        StorageNetworkError,
        SupabaseStorageClient,
    )

__all__ = [
    "AuthManager",
    "UserProfile",
    "get_auth_manager",
    "DeeplinkHandler",
    "install_single_instance_guard",
    "find_deeplink_url",
    "SecureCredentialStore",
    "get_secure_store",
    "SupabaseStorageClient",
    "RemoteObject",
    "StorageError",
    "StorageNetworkError",
]


_LAZY = {
    "AuthManager": ("auth_manager", "AuthManager"),
    "UserProfile": ("auth_manager", "UserProfile"),
    "get_auth_manager": ("auth_manager", "get_auth_manager"),
    "SecureCredentialStore": ("secure", "SecureCredentialStore"),
    "get_secure_store": ("secure", "get_secure_store"),
    "SupabaseStorageClient": ("supabase_storage", "SupabaseStorageClient"),
    "RemoteObject": ("supabase_storage", "RemoteObject"),
    "StorageError": ("supabase_storage", "StorageError"),
    "StorageNetworkError": ("supabase_storage", "StorageNetworkError"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY:
        raise AttributeError(
            f"module 'application.service.auth' has no attribute {name!r}"
        )
    module_name, attr = _LAZY[name]
    from importlib import import_module
    mod = import_module(f"{__name__}.{module_name}")
    value = getattr(mod, attr)
    globals()[name] = value  # cache on the package so future lookups are O(1)
    return value
