"""Unified engine registry — single source of truth for all engine backends.

Storage: ``~/UnitPort/engine_registry.json``  (outside project tree for SSH security)

Layout::

    {
      "schema_version": 2,
      "engines": {
        "sb3": {
          "local": { "enabled": true },
          "cloud": { "servers": [] }
        },
        "isaac_lab": {
          "local": {
            "enabled": true,
            "root": "/home/user/UnitPort/engines/isaac/IsaacLab",
            "registered": true
          },
          "cloud": {
            "default_server": "Lab GPU",
            "servers": [
              { "name": "Lab GPU", "host": "10.0.0.1", ... }
            ]
          }
        }
      }
    }

Each engine is a top-level key under ``engines``.  Inside each engine:
  - ``local``  — local installation info (paths, flags).
  - ``cloud``  — remote server list + default pointer.

Passwords are NEVER persisted (security).
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

from src.system.engines.base import UNITPORT_USER_ROOT

# Registry lives under ~/UnitPort/ — outside the project tree.
# SSH credentials never end up in a git-tracked directory.
_REGISTRY_PATH = UNITPORT_USER_ROOT / "engine_registry.json"

# Legacy location (project-internal, pre-migration)
_LEGACY_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"

_SCHEMA_VERSION = 2

# Sensitive fields stripped from server dicts before writing to disk.
_SENSITIVE_FIELDS = frozenset({"password"})


def _default_data() -> dict:
    return {"schema_version": _SCHEMA_VERSION, "engines": {}}


def _ensure_engine(data: dict, engine_id: str) -> dict:
    """Return the engine block, creating it if absent."""
    engines = data.setdefault("engines", {})
    if engine_id not in engines:
        engines[engine_id] = {"local": {}, "cloud": {"servers": [], "default_server": ""}}
    blk = engines[engine_id]
    blk.setdefault("local", {})
    blk.setdefault("cloud", {"servers": [], "default_server": ""})
    blk["cloud"].setdefault("servers", [])
    blk["cloud"].setdefault("default_server", "")
    return blk


def _strip_sensitive(server_dict: dict) -> dict:
    return {k: v for k, v in server_dict.items() if k not in _SENSITIVE_FIELDS}


# =========================================================================
# EngineRegistry
# =========================================================================

class EngineRegistry:
    """Unified CRUD for engine local paths + cloud servers.

    Thread-safety: callers are expected to run on the main thread or use
    external locking.  The file is written atomically (temp + rename).
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        self._path = path or _REGISTRY_PATH

    # ------------------------------------------------------------------
    # Raw I/O
    # ------------------------------------------------------------------

    def _load(self) -> dict:
        if not self._path.is_file():
            return _default_data()
        try:
            text = self._path.read_text(encoding="utf-8")
            data = json.loads(text)
            if not isinstance(data, dict):
                return _default_data()
            data.setdefault("schema_version", _SCHEMA_VERSION)
            data.setdefault("engines", {})
            return data
        except Exception:
            log.warning("Failed to load %s, using defaults", self._path)
            return _default_data()

    def _save(self, data: dict) -> None:
        data["schema_version"] = _SCHEMA_VERSION
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            dir=str(self._path.parent), suffix=".tmp", prefix="engine_reg_",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                f.write("\n")
            try:
                os.replace(tmp, self._path)
            except OSError:
                if self._path.exists():
                    self._path.unlink()
                os.rename(tmp, self._path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    # ==================================================================
    # LOCAL — per-engine local installation info
    # ==================================================================

    def get_local(self, engine_id: str) -> dict:
        """Return the ``local`` block for *engine_id* (empty dict if absent)."""
        data = self._load()
        return data.get("engines", {}).get(engine_id, {}).get("local", {})

    def set_local(self, engine_id: str, local_info: dict) -> None:
        """Replace the ``local`` block for *engine_id*."""
        data = self._load()
        blk = _ensure_engine(data, engine_id)
        blk["local"] = local_info
        self._save(data)

    def update_local(self, engine_id: str, **kwargs: Any) -> None:
        """Merge key-value pairs into the ``local`` block."""
        data = self._load()
        blk = _ensure_engine(data, engine_id)
        blk["local"].update(kwargs)
        self._save(data)

    # ==================================================================
    # CLOUD — per-engine server list
    # ==================================================================

    def list_servers(self, engine_id: str) -> List[dict]:
        """Return the cloud server list for *engine_id*."""
        data = self._load()
        return (
            data.get("engines", {})
            .get(engine_id, {})
            .get("cloud", {})
            .get("servers", [])
        )

    def get_server(self, engine_id: str, server_name: str) -> Optional[dict]:
        for s in self.list_servers(engine_id):
            if s.get("name") == server_name:
                return s
        return None

    def get_default_server_name(self, engine_id: str) -> str:
        data = self._load()
        return (
            data.get("engines", {})
            .get(engine_id, {})
            .get("cloud", {})
            .get("default_server", "")
        )

    def get_default_server(self, engine_id: str) -> Optional[dict]:
        name = self.get_default_server_name(engine_id)
        if name:
            srv = self.get_server(engine_id, name)
            if srv:
                return srv
        servers = self.list_servers(engine_id)
        return servers[0] if servers else None

    def add_server(self, engine_id: str, server_dict: dict) -> None:
        """Add a cloud server.  Raises ValueError on duplicate name."""
        data = self._load()
        blk = _ensure_engine(data, engine_id)
        servers = blk["cloud"]["servers"]
        name = server_dict.get("name", "")
        if any(s.get("name") == name for s in servers):
            raise ValueError(f"Server '{name}' already exists under engine '{engine_id}'.")
        servers.append(_strip_sensitive(server_dict))
        self._save(data)

    def update_server(self, engine_id: str, old_name: str, server_dict: dict) -> None:
        data = self._load()
        blk = _ensure_engine(data, engine_id)
        servers = blk["cloud"]["servers"]
        found = False
        for i, s in enumerate(servers):
            if s.get("name") == old_name:
                servers[i] = _strip_sensitive(server_dict)
                found = True
                break
        if not found:
            raise KeyError(f"Server '{old_name}' not found under engine '{engine_id}'.")
        if blk["cloud"]["default_server"] == old_name:
            blk["cloud"]["default_server"] = server_dict.get("name", old_name)
        self._save(data)

    def remove_server(self, engine_id: str, server_name: str) -> None:
        data = self._load()
        blk = _ensure_engine(data, engine_id)
        blk["cloud"]["servers"] = [
            s for s in blk["cloud"]["servers"] if s.get("name") != server_name
        ]
        if blk["cloud"]["default_server"] == server_name:
            blk["cloud"]["default_server"] = ""
        self._save(data)

    def set_default_server(self, engine_id: str, server_name: str) -> None:
        data = self._load()
        blk = _ensure_engine(data, engine_id)
        blk["cloud"]["default_server"] = server_name
        self._save(data)

    # ==================================================================
    # Convenience — Isaac Lab shortcuts
    # ==================================================================

    def get_isaac_local_root(self) -> Optional[str]:
        """Return the registered Isaac Lab local root, or None."""
        loc = self.get_local("isaac_lab")
        if loc.get("registered") and loc.get("root"):
            return str(loc["root"])
        return None

    def register_isaac_local(self, root: str) -> None:
        """Register a local Isaac Lab installation."""
        self.set_local("isaac_lab", {
            "root": str(Path(root).resolve()),
            "registered": True,
        })
        log.info("[engine-registry] Isaac Lab local registered: %s", root)

    # ==================================================================
    # Engine enumeration
    # ==================================================================

    def list_engines(self) -> List[str]:
        """Return all registered engine IDs."""
        data = self._load()
        return list(data.get("engines", {}).keys())

    # ==================================================================
    # Migration from legacy files
    # ==================================================================

    def migrate_legacy(self) -> bool:
        """Import data from legacy files into ~/UnitPort/engine_registry.json,
        then delete the old files.

        Legacy sources (all under src/config/):
          - isaaclab.json           → engines.isaac_lab.local
          - remote_servers.json     → engines.isaac_lab.cloud
          - engine_registry.json    → merge into new location

        Safe to call multiple times — old files only exist once.
        Returns True if any migration occurred.
        """
        migrated = False

        # 0. Old-location engine_registry.json (was in src/config/ before
        #    the move to ~/UnitPort/).  Merge its content into the new file.
        old_registry = _LEGACY_CONFIG_DIR / "engine_registry.json"
        if old_registry.is_file() and old_registry != self._path:
            try:
                old_data = json.loads(old_registry.read_text(encoding="utf-8"))
                for eid, eblock in old_data.get("engines", {}).items():
                    # Merge local
                    old_local = eblock.get("local", {})
                    if old_local and not self.get_local(eid):
                        self.set_local(eid, old_local)
                        migrated = True
                    # Merge cloud servers
                    old_servers = eblock.get("cloud", {}).get("servers", [])
                    existing_names = {s.get("name") for s in self.list_servers(eid)}
                    for srv in old_servers:
                        if srv.get("name") not in existing_names:
                            self.add_server(eid, srv)
                            migrated = True
                    old_default = eblock.get("cloud", {}).get("default_server", "")
                    if old_default and not self.get_default_server_name(eid):
                        self.set_default_server(eid, old_default)
                old_registry.unlink()
                log.info("[engine-registry] Migrated src/config/engine_registry.json → ~/UnitPort/")
            except Exception as exc:
                log.warning("[engine-registry] old registry migration failed: %s", exc)

        # 1. isaaclab.json → engines.isaac_lab.local
        legacy_isaac = _LEGACY_CONFIG_DIR / "isaaclab.json"
        if legacy_isaac.is_file():
            try:
                old = json.loads(legacy_isaac.read_text(encoding="utf-8"))
                if old.get("registered") and old.get("root"):
                    current = self.get_local("isaac_lab")
                    if not current.get("registered"):
                        self.register_isaac_local(old["root"])
                        migrated = True
                legacy_isaac.unlink()
                log.info("[engine-registry] Migrated and removed isaaclab.json")
            except Exception as exc:
                log.warning("[engine-registry] isaaclab.json migration failed: %s", exc)

        # 2. remote_servers.json → engines.isaac_lab.cloud
        legacy_remote = _LEGACY_CONFIG_DIR / "remote_servers.json"
        if legacy_remote.is_file():
            try:
                old = json.loads(legacy_remote.read_text(encoding="utf-8"))
                old_servers = old.get("servers", [])
                if old_servers:
                    existing = self.list_servers("isaac_lab")
                    existing_names = {s.get("name") for s in existing}
                    for srv in old_servers:
                        if srv.get("name") not in existing_names:
                            self.add_server("isaac_lab", srv)
                            migrated = True
                    if old.get("default_server") and not self.get_default_server_name("isaac_lab"):
                        self.set_default_server("isaac_lab", old["default_server"])
                legacy_remote.unlink()
                log.info("[engine-registry] Migrated and removed remote_servers.json")
            except Exception as exc:
                log.warning("[engine-registry] remote_servers.json migration failed: %s", exc)

        return migrated


# =========================================================================
# Module-level singleton
# =========================================================================

_registry: Optional[EngineRegistry] = None


def get_engine_registry() -> EngineRegistry:
    """Return the module-level EngineRegistry singleton."""
    global _registry
    if _registry is None:
        _registry = EngineRegistry()
    return _registry
