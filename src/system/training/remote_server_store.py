"""CRUD store for remote SSH server configurations.

Thin delegation layer over the unified ``EngineRegistry``.
All data lives in ``src/config/engine_registry.json`` under each engine's
``cloud.servers`` block.  This module preserves the existing API so callers
(``remote_server_dialog.py``, ``training_workspace_window.py``, etc.) need
no changes beyond import paths — which already point here.

Passwords are stored in the OS credential manager via ``keyring``.
They are NEVER persisted to the engine_registry JSON file.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from src.system.training.remote_ssh_config import RemoteServerConfig

log = logging.getLogger(__name__)

# Engine ID used for Isaac Lab cloud servers.
_ISAAC_ENGINE = "isaac_lab"

# keyring service namespace — all UnitPort SSH passwords live under this key.
_KEYRING_SERVICE = "unitport-ssh"


# ---------------------------------------------------------------------------
# Keyring helpers (best-effort — never crash if keyring is unavailable)
# ---------------------------------------------------------------------------

def _keyring_set(server_name: str, password: str) -> None:
    """Store *password* for *server_name* in the OS credential manager."""
    try:
        import keyring
        keyring.set_password(_KEYRING_SERVICE, server_name, password)
    except Exception as exc:  # noqa: BLE001
        log.warning("keyring: failed to store password for %r: %s", server_name, exc)


def _keyring_get(server_name: str) -> str:
    """Retrieve the password for *server_name*, or ``""`` if not found."""
    try:
        import keyring
        pw = keyring.get_password(_KEYRING_SERVICE, server_name)
        return pw or ""
    except Exception:  # noqa: BLE001
        return ""


def _keyring_delete(server_name: str) -> None:
    """Remove the credential for *server_name* (best-effort)."""
    try:
        import keyring
        keyring.delete_password(_KEYRING_SERVICE, server_name)
    except Exception:  # noqa: BLE001
        pass


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
        servers = [RemoteServerConfig.from_dict(s) for s in raw]
        # Hydrate passwords from keyring.
        for srv in servers:
            if srv.auth_method == "password" and not srv.password:
                srv.password = _keyring_get(srv.name)
        return servers

    def get_server(self, name: str) -> Optional[RemoteServerConfig]:
        raw = self._registry().get_server(self._engine_id, name)
        if raw is None:
            return None
        srv = RemoteServerConfig.from_dict(raw)
        if srv.auth_method == "password" and not srv.password:
            srv.password = _keyring_get(srv.name)
        return srv

    def get_default_server(self) -> Optional[RemoteServerConfig]:
        raw = self._registry().get_default_server(self._engine_id)
        if raw is None:
            return None
        srv = RemoteServerConfig.from_dict(raw)
        if srv.auth_method == "password" and not srv.password:
            srv.password = _keyring_get(srv.name)
        return srv

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_server(self, server: RemoteServerConfig) -> None:
        # Persist password to keyring (never to JSON).
        if server.auth_method == "password" and server.password:
            _keyring_set(server.name, server.password)
        self._registry().add_server(
            self._engine_id, server.to_dict(include_sensitive=False),
        )

    def update_server(self, old_name: str, server: RemoteServerConfig) -> None:
        # If the server was renamed, migrate the credential.
        if old_name != server.name:
            _keyring_delete(old_name)
        if server.auth_method == "password" and server.password:
            _keyring_set(server.name, server.password)
        self._registry().update_server(
            self._engine_id, old_name, server.to_dict(include_sensitive=False),
        )

    def remove_server(self, name: str) -> None:
        _keyring_delete(name)
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
            if server.docker_enabled and server.docker_image:
                # Docker mode: check inside the container image / running container.
                container_launcher = "/workspace/isaaclab/isaaclab.sh"
                if server.docker_container_name:
                    # Existing running container → docker exec
                    cmd = (
                        f"docker exec {server.docker_container_name} "
                        f'test -f "{container_launcher}" '
                        f'&& echo "OK" || echo "NOT_FOUND"'
                    )
                else:
                    # Ephemeral mode → docker run --rm to probe the image
                    cmd = (
                        f"docker run --rm {server.docker_image} "
                        f'test -f "{container_launcher}" '
                        f'&& echo "OK" || echo "NOT_FOUND"'
                    )
                _, stdout, _ = client.exec_command(cmd, timeout=30)
                result = stdout.read().decode("utf-8", errors="replace").strip()
                if result == "OK":
                    return True, (
                        f"Connected. Isaac Lab launcher found at "
                        f"{container_launcher} (inside Docker)."
                    )
                else:
                    return True, (
                        f"Connected to {server.host}, but Isaac Lab launcher "
                        f"not found at {container_launcher} inside Docker image "
                        f"{server.docker_image}."
                    )
            else:
                # Bare-metal mode: check on the host filesystem.
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
