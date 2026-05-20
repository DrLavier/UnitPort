"""EngineService — thin facade over registers.backends + per-engine user state.

Reads availability + version + module path from ``registers.backends`` (the
canonical "what's installed on this machine" source). Persists user-editable
state — local root, enabled flag, cloud server list, default server — under
``Paths.USER_CONFIG_DIR / "engines" / "<engine_id>.json"`` via ``push_data``.

This split honours the registers/ contract: ``data/backends_installed.json``
is the only runtime-writable file inside the registers tree, and even that
only carries detection results — never user choices like SSH server lists.

Per-engine user state schema (one file per engine_id under <USER_CONFIG_DIR>/engines/):

    {
      "local": {
        "enabled":   true,
        "root":      "/home/user/UnitPort/engines/isaac/IsaacLab",
        "registered": true
      },
      "cloud": {
        "default_server": "Lab GPU",
        "servers": [
          { "name": "Lab GPU", "host": "10.0.0.1", "user": "researcher", ... }
        ]
      }
    }

Passwords are NEVER persisted here — keyring (via SecureCredentialStore.ssh_password)
is the only acceptable home for SSH credentials.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from PyQt6.QtCore import QObject, pyqtSignal

from registers import backends
from unitport_sdk import Paths, log_info, log_warning, push_data, read_data

# Sensitive fields stripped from server dicts before writing to disk.
_SENSITIVE_FIELDS = frozenset({"password", "sudo_password"})


def _engine_state_rel(engine_id: str) -> str:
    """Relative path used with push_data — resolved under USER_CONFIG_DIR."""
    return f"engines/{engine_id}.json"


def _engine_state_path(engine_id: str) -> Path:
    return Paths.USER_CONFIG_DIR / "engines" / f"{engine_id}.json"


def _strip_sensitive(server: dict) -> dict:
    return {k: v for k, v in server.items() if k not in _SENSITIVE_FIELDS}


def _empty_state() -> Dict[str, Any]:
    return {
        "local": {},
        "cloud": {"servers": [], "default_server": ""},
    }


class EngineService(QObject):
    """Process-wide facade for the User panel's Engines section."""

    # engine_id ("" = all engines / global refresh)
    changed = pyqtSignal(str)

    # ----- registry-derived (read-only) -------------------------------------

    def list_known_engines(self) -> List[str]:
        """Engines that registers.backends knows about (sb3, isaac_lab, ...).

        Empty list before backends.refresh_engine_availability() runs at least
        once — caller should treat this as "no engines yet" and render the
        empty section quietly.
        """
        return sorted(backends.list_installed().keys())

    def status(self, engine_id: str) -> Dict[str, Any]:
        """Detection result from registers.backends.

        Returns ``{"available": False, "enabled": False, "version": "", "path": ""}``
        for unknown engines (never raises).
        """
        rec = backends.get_installed(engine_id) or {}
        return {
            "available": bool(rec.get("available", False)),
            "enabled": bool(rec.get("enabled", False)),
            "version": str(rec.get("version", "")),
            "path": str(rec.get("path", "")),
        }

    def refresh(self) -> None:
        """Re-scan engine availability and notify listeners."""
        backends.refresh_engine_availability()
        self.changed.emit("")

    # ----- per-engine user state -------------------------------------------

    def _load_state(self, engine_id: str) -> Dict[str, Any]:
        path = _engine_state_path(engine_id)
        if not path.exists():
            return _empty_state()
        data = read_data(path)
        if not isinstance(data, dict):
            return _empty_state()
        data.setdefault("local", {})
        cloud = data.setdefault("cloud", {})
        cloud.setdefault("servers", [])
        cloud.setdefault("default_server", "")
        return data

    def _save_state(self, engine_id: str, state: Dict[str, Any]) -> None:
        if not push_data(_engine_state_rel(engine_id), state):
            log_warning(f"[engines] failed to save state for {engine_id}")

    # local block

    def get_local(self, engine_id: str) -> Dict[str, Any]:
        return dict(self._load_state(engine_id).get("local", {}))

    def set_local(self, engine_id: str, **kwargs: Any) -> None:
        """Merge kwargs into the engine's local block."""
        state = self._load_state(engine_id)
        state["local"].update(kwargs)
        self._save_state(engine_id, state)
        self.changed.emit(engine_id)

    def register_isaac_local(
        self,
        root: str,
        *,
        source: str = "manual",
    ) -> bool:
        """Validate + register an Isaac Lab installation root.

        Markers (mirrors DEMO EngineRegistry.register_isaac_local logic):
        - isaaclab.sh OR isaaclab.bat present at the root
        - source/ subdirectory present

        ``source`` records *how* the path was registered so downstream
        audit / repair tools can tell apart user-located vs in-app-installed
        registrations. Recognised values: ``"manual"`` (locate flow from
        the wizard or sidebar), ``"install"`` (in-app installer succeeded),
        ``"import_from_demo"`` (one-shot DEMO→RELEASE bridge).

        Returns True if validation passed and state was written; False otherwise.
        """
        p = Path(root).expanduser().resolve()
        markers_ok = (
            (p / "isaaclab.sh").exists() or (p / "isaaclab.bat").exists()
        ) and (p / "source").is_dir()
        if not markers_ok:
            log_warning(
                f"[engines] {p} does not look like an Isaac Lab root "
                f"(missing isaaclab.sh|isaaclab.bat and/or source/)"
            )
            return False
        self.set_local(
            "isaac_lab",
            root=str(p),
            registered=True,
            enabled=True,
            source=str(source),
        )
        log_info(
            f"[engines] Isaac Lab local registered: {p} (source={source})"
        )
        return True

    def import_isaac_lab_path_from_demo(
        self, demo_setup_state_path: Optional[Path] = None
    ) -> bool:
        """One-shot import of the Isaac Lab path that DEMO already registered.

        DEMO persists the path the user located during install at
        ``DEMO/src/config/setup_state.json`` under
        ``selections.backend.isaaclab_path``. RELEASE's user state lives in
        ``<USER_CONFIG_DIR>/engines/isaac_lab.json`` (`local.root`). This method
        bridges the two so a user who already ran DEMO install does not have
        to re-locate Isaac Lab in RELEASE.

        Discovery order for ``demo_setup_state_path`` when not provided:
          1. ``Paths.PROJECT_ROOT.parent / "DEMO" / "src" / "config" / "setup_state.json"``
             (sibling layout, matches the working repo)

        Returns True if a path was read AND it validated as an Isaac Lab root
        (delegates to ``register_isaac_local``). False on any miss/mismatch
        (already-registered RELEASE state is left intact).
        """
        if demo_setup_state_path is None:
            demo_setup_state_path = (
                Paths.PROJECT_ROOT.parent
                / "DEMO" / "src" / "config" / "setup_state.json"
            )
        if not demo_setup_state_path.exists():
            log_warning(
                f"[engines] DEMO setup_state.json not found at "
                f"{demo_setup_state_path} — skip Isaac Lab path import"
            )
            return False
        data = read_data(demo_setup_state_path)
        if not isinstance(data, dict):
            log_warning(
                f"[engines] {demo_setup_state_path} did not parse as dict — skip"
            )
            return False
        backend = (
            data.get("selections", {})
                .get("backend", {})
        )
        root = str(backend.get("isaaclab_path", "")).strip()
        if not root:
            log_warning(
                f"[engines] DEMO setup_state.json has no isaaclab_path — skip"
            )
            return False
        log_info(f"[engines] importing Isaac Lab path from DEMO: {root}")
        return self.register_isaac_local(root, source="import_from_demo")

    # cloud block

    def list_servers(self, engine_id: str) -> List[Dict[str, Any]]:
        return list(self._load_state(engine_id).get("cloud", {}).get("servers", []))

    def get_server(self, engine_id: str, server_name: str) -> Optional[Dict[str, Any]]:
        for s in self.list_servers(engine_id):
            if s.get("name") == server_name:
                return s
        return None

    def get_default_server_name(self, engine_id: str) -> str:
        return str(self._load_state(engine_id).get("cloud", {}).get("default_server", ""))

    def get_default_server(self, engine_id: str) -> Optional[Dict[str, Any]]:
        name = self.get_default_server_name(engine_id)
        if name:
            srv = self.get_server(engine_id, name)
            if srv is not None:
                return srv
        servers = self.list_servers(engine_id)
        return servers[0] if servers else None

    def add_server(self, engine_id: str, server: Dict[str, Any]) -> None:
        """Append a cloud server. Raises ValueError on duplicate name."""
        state = self._load_state(engine_id)
        servers = state["cloud"]["servers"]
        name = server.get("name", "")
        if not name:
            raise ValueError("Server must have a 'name' field.")
        if any(s.get("name") == name for s in servers):
            raise ValueError(f"Server '{name}' already exists under engine '{engine_id}'.")
        servers.append(_strip_sensitive(server))
        self._save_state(engine_id, state)
        self.changed.emit(engine_id)

    def update_server(self, engine_id: str, old_name: str, server: Dict[str, Any]) -> None:
        state = self._load_state(engine_id)
        servers = state["cloud"]["servers"]
        for i, s in enumerate(servers):
            if s.get("name") == old_name:
                servers[i] = _strip_sensitive(server)
                if state["cloud"]["default_server"] == old_name:
                    state["cloud"]["default_server"] = server.get("name", old_name)
                self._save_state(engine_id, state)
                self.changed.emit(engine_id)
                return
        raise KeyError(f"Server '{old_name}' not found under engine '{engine_id}'.")

    def remove_server(self, engine_id: str, server_name: str) -> None:
        state = self._load_state(engine_id)
        state["cloud"]["servers"] = [
            s for s in state["cloud"]["servers"] if s.get("name") != server_name
        ]
        if state["cloud"]["default_server"] == server_name:
            state["cloud"]["default_server"] = ""
        self._save_state(engine_id, state)
        self.changed.emit(engine_id)

    def set_default_server(self, engine_id: str, server_name: str) -> None:
        state = self._load_state(engine_id)
        state["cloud"]["default_server"] = server_name
        self._save_state(engine_id, state)
        self.changed.emit(engine_id)


_instance: Optional[EngineService] = None


def get_engine_service() -> EngineService:
    """Return the process-wide EngineService."""
    global _instance
    if _instance is None:
        _instance = EngineService()
    return _instance


__all__ = ["EngineService", "get_engine_service"]
