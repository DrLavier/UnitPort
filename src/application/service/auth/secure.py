# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Secure credential storage via the OS credential manager.

The only module in RELEASE that imports ``keyring``. Every consumer routes
through :class:`SecureCredentialStore` (or its module-level singleton via
:func:`get_secure_store`).

Backends (auto-selected by the ``keyring`` library):
    Windows: Windows Credential Manager
    macOS:   Keychain
    Linux:   Secret Service (gnome-keyring / kwallet) or ``pass``

Failure mode: if ``keyring`` is unavailable (import error or backend unusable),
all ops become no-ops with a single warning logged.
"""

from __future__ import annotations

from typing import Optional

from unitport_sdk import log_warning

try:
    import keyring as _keyring
    _KEYRING_OK = True
except ImportError:
    _keyring = None  # type: ignore[assignment]
    _KEYRING_OK = False


def _resolve_user_scope() -> str:
    """Return the active workspace id ("" → guest), used as a keyring prefix.

    Lazy-imports user_workspace to avoid a circular import: user_workspace
    imports unitport_sdk's logger which the SecureCredentialStore also uses.
    """
    try:
        from application.service.user_workspace import read_active_user
        return read_active_user() or "_guest"
    except Exception:
        # Defensive: anything goes wrong reading the shim, fall through to
        # the guest scope so SSH passwords still land in a predictable slot
        # rather than crashing the caller.
        return "_guest"


class _SecretHandle:
    """Typed (service, account) pair — exposes get/set/delete."""

    __slots__ = ("_store", "_service", "_account")

    def __init__(self, store: "SecureCredentialStore", service: str, account: str) -> None:
        self._store = store
        self._service = service
        self._account = account

    def get(self) -> Optional[str]:
        return self._store.get(self._service, self._account)

    def set(self, value: str) -> None:
        self._store.set(self._service, self._account, value)

    def delete(self) -> None:
        self._store.delete(self._service, self._account)


class SecureCredentialStore:
    """Single authorized chokepoint for keyring access."""

    AUTH_SERVICE  = "unitport-auth"
    SSH_SERVICE   = "unitport-ssh"
    ROBOT_SERVICE = "unitport-robot"
    LLM_SERVICE   = "unitport-llm"

    def __init__(self) -> None:
        self._warned_unavailable = False

    def is_available(self) -> bool:
        return _KEYRING_OK

    # ── core three-op API ────────────────────────────────────────────────

    def set(self, service: str, account: str, secret: str) -> None:
        if not secret:
            self.delete(service, account)
            return
        if not self._ensure_available():
            return
        try:
            _keyring.set_password(service, account, secret)
        except Exception as exc:
            log_warning(
                f"[storage:secure] set_password({service!r}, {account!r}) failed: {exc}"
            )

    def get(self, service: str, account: str) -> Optional[str]:
        if not self._ensure_available():
            return None
        try:
            return _keyring.get_password(service, account)
        except Exception as exc:
            log_warning(
                f"[storage:secure] get_password({service!r}, {account!r}) failed: {exc}"
            )
            return None

    def delete(self, service: str, account: str) -> None:
        if not self._ensure_available():
            return
        try:
            _keyring.delete_password(service, account)
        except Exception:
            # Deleting a non-existent entry raises; expected on unconditional sign-out.
            pass

    # ── typed sub-handles ────────────────────────────────────────────────
    # OS keyring is shared across all UnitPort accounts on the same OS user.
    # Per-account isolation is realised by namespacing the keyring ``account``
    # field with the active workspace id; otherwise the second account to
    # log in would overwrite the first's refresh token, and two accounts
    # that happen to use the same SSH server name would share the password.

    def auth_refresh_token(self, user_id: str = "") -> _SecretHandle:
        """Per-user Supabase refresh token slot.

        ``user_id`` is the Supabase user id (UUID). When empty, falls back to
        the legacy account name ``"refresh_token"`` — covers the bootstrap
        window before any user has ever signed in on this OS account, and
        any pre-isolation install where the previous schema is still in place.
        """
        account = user_id or "refresh_token"
        return _SecretHandle(self, self.AUTH_SERVICE, account)

    def ssh_password(
        self, server_name: str, user_scope: str = "",
    ) -> _SecretHandle:
        """Per-account SSH password slot.

        ``user_scope`` defaults to the active workspace (resolved lazily via
        ``user_workspace.read_active_user``) so callers needn't thread it
        through. Pass it explicitly when the active user can't be trusted
        (tests, headless flows).
        """
        scope = user_scope or _resolve_user_scope()
        return _SecretHandle(self, self.SSH_SERVICE, f"{scope}/{server_name}")

    def ssh_sudo_password(
        self, server_name: str, user_scope: str = "",
    ) -> _SecretHandle:
        scope = user_scope or _resolve_user_scope()
        return _SecretHandle(
            self, self.SSH_SERVICE, f"{scope}/{server_name}/sudo"
        )

    def robot_credential(self, brand: str, robot: str, key: str) -> _SecretHandle:
        return _SecretHandle(self, self.ROBOT_SERVICE, f"{brand}/{robot}/{key}")

    def llm_api_key(self, provider: str) -> _SecretHandle:
        return _SecretHandle(self, self.LLM_SERVICE, provider)

    # ── internals ────────────────────────────────────────────────────────

    def _ensure_available(self) -> bool:
        if _KEYRING_OK:
            return True
        if not self._warned_unavailable:
            log_warning(
                "[storage:secure] keyring unavailable — credentials will not "
                "persist across restarts. Install `keyring` or the appropriate "
                "platform backend."
            )
            self._warned_unavailable = True
        return False


_default_store: Optional[SecureCredentialStore] = None


def get_secure_store() -> SecureCredentialStore:
    """Return the process-wide :class:`SecureCredentialStore`."""
    global _default_store
    if _default_store is None:
        _default_store = SecureCredentialStore()
    return _default_store


__all__ = ["SecureCredentialStore", "get_secure_store"]
