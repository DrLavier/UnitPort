"""CRUD store for remote SSH server configurations.

Thin delegation layer over the unified ``EngineRegistry``.
All data lives in ``src/config/engine_registry.json`` under each engine's
``cloud.servers`` block.  This module preserves the existing API so callers
(``remote_server_dialog.py``, ``training_workspace_window.py``, etc.) need
no changes beyond import paths — which already point here.

Passwords are NEVER persisted to disk.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from src.system.training.remote_ssh_config import RemoteServerConfig

log = logging.getLogger(__name__)

# Engine ID used for Isaac Lab cloud servers.
_ISAAC_ENGINE = "isaac_lab"


class RemoteServerStore:
    """Persistent CRUD for saved remote server configurations.

    Delegates to ``EngineRegistry`` for the ``isaac_lab`` engine.
    """

    def __init__(self, engine_id: str = _ISAAC_ENGINE) -> None:
        self._engine_id = engine_id

    def _registry(self):
        from src.system.engines.registry import get_engine_registry
        return get_engine_registry()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def list_servers(self) -> List[RemoteServerConfig]:
        raw = self._registry().list_servers(self._engine_id)
        return [RemoteServerConfig.from_dict(s) for s in raw]

    def get_server(self, name: str) -> Optional[RemoteServerConfig]:
        raw = self._registry().get_server(self._engine_id, name)
        return RemoteServerConfig.from_dict(raw) if raw else None

    def get_default_server(self) -> Optional[RemoteServerConfig]:
        raw = self._registry().get_default_server(self._engine_id)
        return RemoteServerConfig.from_dict(raw) if raw else None

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_server(self, server: RemoteServerConfig) -> None:
        self._registry().add_server(
            self._engine_id, server.to_dict(include_sensitive=False),
        )

    def update_server(self, old_name: str, server: RemoteServerConfig) -> None:
        self._registry().update_server(
            self._engine_id, old_name, server.to_dict(include_sensitive=False),
        )

    def remove_server(self, name: str) -> None:
        self._registry().remove_server(self._engine_id, name)

    def set_default(self, name: str) -> None:
        self._registry().set_default_server(self._engine_id, name)

    # ------------------------------------------------------------------
    # Connection test (unchanged — still uses paramiko directly)
    # ------------------------------------------------------------------

    def test_connection(self, server: RemoteServerConfig) -> Tuple[bool, str]:
        """Test SSH connectivity and verify remote Isaac Lab installation.

        Returns ``(success, message)``.
        """
        try:
            import paramiko
        except ImportError:
            return False, "paramiko is not installed. Run: pip install paramiko"

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            connect_kwargs = {
                "hostname": server.host,
                "port": server.port,
                "username": server.username,
                "timeout": 10,
            }
            if server.auth_method == "key" and server.private_key_path:
                connect_kwargs["key_filename"] = server.private_key_path
            elif server.auth_method == "password" and server.password:
                connect_kwargs["password"] = server.password
            else:
                return False, "No valid authentication credentials configured."

            client.connect(**connect_kwargs)

            # Verify remote Isaac Lab exists.
            launcher = server.remote_isaac_lab_launcher or (
                server.remote_isaac_lab_path.rstrip("/") + "/isaaclab.sh"
            )
            cmd = f'test -f "{launcher}" && echo "OK" || echo "NOT_FOUND"'
            _, stdout, _ = client.exec_command(cmd, timeout=10)
            result = stdout.read().decode("utf-8", errors="replace").strip()

            if result == "OK":
                return True, f"Connected. Isaac Lab launcher found at {launcher}"
            else:
                return True, (
                    f"Connected to {server.host}, but Isaac Lab launcher "
                    f"not found at {launcher}. Check remote_isaac_lab_path."
                )

        except paramiko.AuthenticationException:
            return False, "Authentication failed. Check username/key/password."
        except paramiko.SSHException as exc:
            return False, f"SSH error: {exc}"
        except OSError as exc:
            return False, f"Connection error: {exc}"
        finally:
            client.close()


# ---------------------------------------------------------------------------
# Module-level convenience singleton
# ---------------------------------------------------------------------------

_store: Optional[RemoteServerStore] = None


def get_remote_server_store() -> RemoteServerStore:
    """Return the module-level RemoteServerStore singleton."""
    global _store
    if _store is None:
        _store = RemoteServerStore()
    return _store
