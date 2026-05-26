# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""EngineService — thin facade over registers.backends + per-engine cloud state.

Two storage tiers, by design:

* **Local installation** — root path, registered flag, enabled flag, source
  tag — is a per-installation physical fact and lives in
  ``backends_installed.json`` (the SDK tree's runtime-writable registry file,
  see CLAUDE.md §3). EngineService delegates these to
  :mod:`registers.backends` via ``get_local_registration`` /
  ``set_local_registration``. **USER_CONFIG_DIR is never read or written for
  local-installation state.** The earlier ``<USER_CONFIG_DIR>/engines/<eid>.json``
  layout that carried a ``local`` block has been retired — at startup any
  legacy file is silently migrated into the registry and the ``local`` block
  is stripped.

* **Cloud servers** — SSH endpoint list + default-server pick — is user
  preference (per-account, cloud-synced) and stays in
  ``<USER_CONFIG_DIR>/engines/<eid>.json`` (cloud-only schema below).

Per-engine cloud-only state schema (one file per engine_id under
<USER_CONFIG_DIR>/engines/):

    {
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
from unitport_sdk import (
    Config,
    Paths,
    log_info,
    log_warning,
    push_data,
    read_data,
    tr,
)

# Sensitive fields stripped from server dicts before writing to disk.
_SENSITIVE_FIELDS = frozenset({"password", "sudo_password"})

# ---------------------------------------------------------------------------
# user.ini-backed training preferences (per-user, survives device switch).
#
# The *list* of installed Isaac Lab versions is a per-installation physical
# fact (backends_installed.json). Which one a user prefers as their local
# training target, and whether they train Local vs Cloud, is a per-user
# preference — it rides in user.ini[Training] so a user who switches machines
# keeps the choice, and we fall back loudly when the pinned root is absent on
# the new box (CLAUDE.md §8: no silent substitution, surface the fallback).
# ---------------------------------------------------------------------------
_TRAINING_SECTION = "Training"
_KEY_LOCAL_ROOT = "local_isaac_root"        # selected isaac install root ("" = primary)
_KEY_LINK_TARGET = "link_target"            # "local" | "cloud"
_KEY_PROMPT_DISMISSED = "local_no_isaac_dismissed"  # "1" once user picks cloud-only

_ISAAC_LAB = "isaac_lab"


def _engine_state_rel(engine_id: str) -> str:
    """Relative path used with push_data — resolved under USER_CONFIG_DIR."""
    return f"engines/{engine_id}.json"


def _engine_state_path(engine_id: str) -> Path:
    return Paths.USER_CONFIG_DIR / "engines" / f"{engine_id}.json"


def _strip_sensitive(server: dict) -> dict:
    return {k: v for k, v in server.items() if k not in _SENSITIVE_FIELDS}


def _empty_cloud_state() -> Dict[str, Any]:
    return {"cloud": {"servers": [], "default_server": ""}}


def _migrate_legacy_local_block(
    engine_id: str, data: Dict[str, Any]
) -> Dict[str, Any]:
    """One-shot migrator: pull ``local`` out of the legacy combined file.

    Older builds wrote both ``local`` (install root + flags) and ``cloud``
    (server list) into ``<USER_CONFIG_DIR>/engines/<eid>.json``. The local
    half is now owned by ``backends_installed.json``. If we see a populated
    ``local`` block here, push it through the registry and rewrite the
    file with the cloud half only.

    Returns the cloud-only dict to use going forward. Never raises — a
    migration failure logs a warning and leaves the file alone so the
    user can recover manually.
    """
    legacy = data.get("local")
    if not isinstance(legacy, dict) or not legacy:
        return data
    root = str(legacy.get("root") or "").strip()
    try:
        backends.set_local_registration(
            engine_id,
            root=root,
            registered=bool(legacy.get("registered", False)) if root else None,
            enabled=bool(legacy.get("enabled", False)) if root else None,
            source=str(legacy.get("source") or "") or None,
        )
        log_info(
            f"[engines] migrated legacy local-state for {engine_id!r} "
            f"from USER_CONFIG_DIR/engines/{engine_id}.json into "
            f"backends_installed.json (root={root or '<empty>'})"
        )
    except Exception as exc:                                  # noqa: BLE001
        log_warning(
            f"[engines] failed to migrate legacy local-state for "
            f"{engine_id!r}: {exc!r} — leaving file untouched"
        )
        return data
    cloud_only = {"cloud": dict(data.get("cloud") or {})}
    cloud_only["cloud"].setdefault("servers", [])
    cloud_only["cloud"].setdefault("default_server", "")
    # Persist the rewrite so the legacy block is gone for good.
    if not push_data(_engine_state_rel(engine_id), cloud_only):
        log_warning(
            f"[engines] migrated {engine_id!r} local-state into registry "
            f"but failed to rewrite the cloud-only file"
        )
    return cloud_only


# ---------------------------------------------------------------------------
# Active-install resolution (module-level, QObject-free).
#
# The User panel writes the user's pick to user.ini[Training].local_isaac_root;
# the *list* of installs is a per-machine physical fact in
# backends_installed.json. Resolving "which install do we actually use" must
# combine the two with a loud fallback (CLAUDE.md §8). Exposed as free
# functions — not just EngineService methods — so non-UI callers (notably the
# training launcher's IsaacLabConfig.from_registers) resolve through the SAME
# selection without instantiating a QObject on a worker thread.
# ---------------------------------------------------------------------------
def read_local_selection_root() -> str:
    """User's pinned local Isaac install root ("" = use primary/base)."""
    return str(
        Config.get_value(_TRAINING_SECTION, _KEY_LOCAL_ROOT, "") or ""
    ).strip()


def resolve_active_local_isaac() -> Optional[Dict[str, Any]]:
    """Resolve which Isaac install to use, with a surfaced fallback.

    Order (CLAUDE.md §8 — surface the fallback, never silently wrong):
      1. The user.ini-pinned root, if it is still a registered install.
      2. The base install (``Engines/isaac_lab/``), if present.
      3. The first registered install, if any.
      4. ``None`` — nothing local registered (caller prompts the user).

    Logs loudly when the pinned root is missing so a device switch surfaces as
    a visible fallback rather than a phantom selection.
    """
    installs = backends.list_isaac_installations(ensure_base=True)
    if not installs:
        return None
    pinned = read_local_selection_root()
    if pinned:
        for e in installs:
            if backends.install_roots_equal(e.get("root", ""), pinned):
                return e
        log_warning(
            f"[engines] pinned local Isaac root {pinned!r} is not "
            f"registered on this machine — falling back to base/first"
        )
    for e in installs:
        if e.get("is_base"):
            return e
    return installs[0]


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

    # ----- local-installation block (delegated to backends_installed.json) ---

    def get_local(self, engine_id: str) -> Dict[str, Any]:
        """Return the local-installation block for ``engine_id``.

        Backed by ``backends_installed.json`` (per-installation SDK
        registry). USER_CONFIG_DIR is never touched here.

        Shape matches the legacy USER_CONFIG_DIR layout for caller
        compatibility: ``{enabled, root, registered, source}``.
        """
        return backends.get_local_registration(engine_id)

    def set_local(self, engine_id: str, **kwargs: Any) -> None:
        """Patch the local-installation block of ``engine_id``.

        Accepted keys: ``root``, ``registered``, ``enabled``, ``source``.
        Any other kwarg is ignored — the only place that should write
        free-form local-state today is :meth:`register_isaac_local`.
        """
        backends.set_local_registration(
            engine_id,
            root=kwargs.get("root"),
            registered=kwargs.get("registered"),
            enabled=kwargs.get("enabled"),
            source=kwargs.get("source"),
        )
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
        ``"import_from_demo"`` (one-shot DEMO→RELEASE bridge),
        ``"boot_repair"`` (re-register during startup probe),
        ``"auto"`` (auto-discovered under ``PROJECT_ROOT/Engines/`` by
        :func:`registers.backends.refresh_engine_availability`).

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
        # Multi-version registry: add (or update) this root in the install
        # list. The version is filled lazily by IsaacInstallProbeTask — we do
        # not probe synchronously here to keep the UI responsive.
        backends.register_isaac_installation(str(p), source=str(source))
        log_info(
            f"[engines] Isaac Lab local registered: {p} (source={source})"
        )
        self.changed.emit("isaac_lab")
        return True

    # ----- multi-version Isaac Lab installation list ----------------------

    def list_isaac_installations(self) -> List[Dict[str, Any]]:
        """All registered Isaac Lab installs (base + external), decorated.

        Delegates to :func:`registers.backends.list_isaac_installations`; each
        entry carries ``root / version / source / available / is_base / drive /
        exists``. Read-only and cheap (no subprocess) — versions come from the
        last probe; call :meth:`refresh_isaac_installations` (background) to
        refresh them.
        """
        return backends.list_isaac_installations(ensure_base=True)

    def refresh_isaac_installations(self) -> List[Dict[str, Any]]:
        """Synchronously re-probe all installs. Prefer the background task.

        Heavy (one ``import isaaclab`` subprocess per root). The management
        dialog submits :class:`IsaacInstallProbeTask` instead of calling this on
        the UI thread; this synchronous form exists for headless / test callers.
        """
        items = backends.refresh_isaac_installations()
        self.changed.emit("isaac_lab")
        return items

    def unbind_isaac_installation(self, root: str) -> bool:
        """Unbind (registry-only remove) an external Isaac Lab install.

        For the project-owned base, true removal means deleting the directory —
        callers must route that through ``IsaacUninstallTask``, not here. We
        refuse a base unbind so a stray call can't leave the on-disk base
        unreferenced while it still auto-discovers on next read.
        """
        if backends.is_base_isaac_root(root):
            log_warning(
                f"[engines] refusing to unbind base Isaac Lab root {root!r} — "
                f"use the uninstall flow to delete it from disk"
            )
            return False
        ok = backends.unregister_isaac_installation(root)
        if ok:
            # If the unbound root was the user's pinned default, drop the pin so
            # resolution falls back loudly rather than pointing at a ghost.
            if backends.install_roots_equal(self.get_local_selection_root(), root):
                self.set_local_selection_root("")
            self.changed.emit("isaac_lab")
        return ok

    # ----- local-engine selection (user.ini, per-user preference) ----------

    def get_local_selection_root(self) -> str:
        """User's pinned local Isaac install root ("" = use primary/base)."""
        return read_local_selection_root()

    def set_local_selection_root(self, root: str) -> None:
        """Persist the pinned local install root to user.ini (DataManager-safe)."""
        Config.set_value(_TRAINING_SECTION, _KEY_LOCAL_ROOT, str(root or ""))
        self.changed.emit("isaac_lab")

    def resolve_active_local_isaac(self) -> Optional[Dict[str, Any]]:
        """Resolve which install to use as the local target, with fallback.

        Order (CLAUDE.md §8 — surface the fallback, never silently wrong):
          1. The user.ini-pinned root, if it is still a registered install.
          2. The base install (``Engines/isaac_lab/``), if present — the
             "fall back to root" behaviour the user asked for on device switch.
          3. The first registered install, if any.
          4. ``None`` — nothing local registered (caller prompts the user).

        Logs loudly when the pinned root is missing so a device switch surfaces
        as a visible fallback rather than a phantom selection.
        """
        # Delegate to the module-level resolver so UI and the training
        # launcher (IsaacLabConfig.from_registers) share one resolution path.
        return resolve_active_local_isaac()

    # ----- link target (Local vs Cloud) + no-isaac prompt opt-out ----------

    def get_link_target(self) -> str:
        """Persisted train target: ``"local"`` (default) or ``"cloud"``."""
        v = str(
            Config.get_value(_TRAINING_SECTION, _KEY_LINK_TARGET, "local") or "local"
        ).strip().lower()
        return "cloud" if v == "cloud" else "local"

    def set_link_target(self, target: str) -> None:
        v = "cloud" if str(target or "").strip().lower() == "cloud" else "local"
        Config.set_value(_TRAINING_SECTION, _KEY_LINK_TARGET, v)

    def is_no_isaac_prompt_dismissed(self) -> bool:
        """True once the user chose "cloud only" — stop nagging to register."""
        return str(
            Config.get_value(_TRAINING_SECTION, _KEY_PROMPT_DISMISSED, "") or ""
        ).strip() in ("1", "true", "True")

    def set_no_isaac_prompt_dismissed(self, dismissed: bool) -> None:
        Config.set_value(
            _TRAINING_SECTION, _KEY_PROMPT_DISMISSED, "1" if dismissed else ""
        )

    def import_isaac_lab_path_from_demo(
        self, demo_setup_state_path: Optional[Path] = None
    ) -> bool:
        """One-shot import of the Isaac Lab path that DEMO already registered.

        DEMO persists the path the user located during install at
        ``DEMO/src/config/setup_state.json`` under
        ``selections.backend.isaaclab_path``. RELEASE's local-installation
        registry lives in ``backends_installed.json`` (per-installation,
        SDK-tree). This method bridges the two so a user who already ran
        DEMO install does not have to re-locate Isaac Lab in RELEASE.

        Discovery order for ``demo_setup_state_path`` when not provided:
          1. ``Paths.PROJECT_ROOT.parent / "DEMO" / "src" / "config" / "setup_state.json"``
             (sibling layout, matches the working repo)

        Returns True if a path was read AND it validated as an Isaac Lab root
        (delegates to ``register_isaac_local``). False on any miss/mismatch
        (already-registered local state is left intact).
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

    # ----- cloud-server block (USER_CONFIG_DIR-backed) ---------------------

    def _load_cloud_state(self, engine_id: str) -> Dict[str, Any]:
        path = _engine_state_path(engine_id)
        if not path.exists():
            return _empty_cloud_state()
        data = read_data(path)
        if not isinstance(data, dict):
            return _empty_cloud_state()
        # Legacy combined file? Migrate the local half into the registry
        # and rewrite this file with cloud-only content. Idempotent — once
        # migrated, subsequent calls see no ``local`` key and skip.
        if isinstance(data.get("local"), dict) and data["local"]:
            data = _migrate_legacy_local_block(engine_id, data)
        cloud = data.setdefault("cloud", {})
        cloud.setdefault("servers", [])
        cloud.setdefault("default_server", "")
        return data

    def _save_cloud_state(self, engine_id: str, state: Dict[str, Any]) -> None:
        if not push_data(_engine_state_rel(engine_id), state):
            log_warning(f"[engines] failed to save cloud state for {engine_id}")

    def list_servers(self, engine_id: str) -> List[Dict[str, Any]]:
        return list(self._load_cloud_state(engine_id).get("cloud", {}).get("servers", []))

    def get_server(self, engine_id: str, server_name: str) -> Optional[Dict[str, Any]]:
        for s in self.list_servers(engine_id):
            if s.get("name") == server_name:
                return s
        return None

    def get_default_server_name(self, engine_id: str) -> str:
        return str(self._load_cloud_state(engine_id).get("cloud", {}).get("default_server", ""))

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
        state = self._load_cloud_state(engine_id)
        servers = state["cloud"]["servers"]
        name = server.get("name", "")
        if not name:
            raise ValueError("Server must have a 'name' field.")
        if any(s.get("name") == name for s in servers):
            raise ValueError(f"Server '{name}' already exists under engine '{engine_id}'.")
        servers.append(_strip_sensitive(server))
        self._save_cloud_state(engine_id, state)
        self.changed.emit(engine_id)

    def update_server(self, engine_id: str, old_name: str, server: Dict[str, Any]) -> None:
        state = self._load_cloud_state(engine_id)
        servers = state["cloud"]["servers"]
        for i, s in enumerate(servers):
            if s.get("name") == old_name:
                servers[i] = _strip_sensitive(server)
                if state["cloud"]["default_server"] == old_name:
                    state["cloud"]["default_server"] = server.get("name", old_name)
                self._save_cloud_state(engine_id, state)
                self.changed.emit(engine_id)
                return
        raise KeyError(f"Server '{old_name}' not found under engine '{engine_id}'.")

    def remove_server(self, engine_id: str, server_name: str) -> None:
        state = self._load_cloud_state(engine_id)
        state["cloud"]["servers"] = [
            s for s in state["cloud"]["servers"] if s.get("name") != server_name
        ]
        if state["cloud"]["default_server"] == server_name:
            state["cloud"]["default_server"] = ""
        self._save_cloud_state(engine_id, state)
        self.changed.emit(engine_id)

    def set_default_server(self, engine_id: str, server_name: str) -> None:
        state = self._load_cloud_state(engine_id)
        state["cloud"]["default_server"] = server_name
        self._save_cloud_state(engine_id, state)
        self.changed.emit(engine_id)


# ---------------------------------------------------------------------------
# Shared label / option builders for the train-target combos.
#
# The User-panel local dropdown, the MainRow combo and the Mission-Control card
# combo must all show identical option text, so the formatting lives here (one
# source of truth) rather than being duplicated three times. The data token is
# the routing key the combos hand back: ``"cloud"``, ``"local::<root>"``, or
# the sentinel ``"local::__none__"`` (no install registered → prompt the user).
# ---------------------------------------------------------------------------

LOCAL_PREFIX = "local::"
LOCAL_NONE = "local::__none__"
CLOUD_TOKEN = "cloud"


def format_isaac_install_label(entry: Dict[str, Any]) -> str:
    """Short drive/version label for one install: ``[Root]0.54.2`` / ``D: 0.54.2``.

    ``entry`` is one item from :meth:`EngineService.list_isaac_installations`.
    A missing version renders as ``?`` (probe pending / failed) rather than an
    empty parenthetical.
    """
    ver = str(entry.get("version") or "").strip() or "?"
    if entry.get("is_base"):
        return tr("engines.local_base_label", "[Root]{ver}").format(ver=ver)
    drive = str(entry.get("drive") or "").strip() or "?"
    return tr("engines.local_ext_label", "{drive} {ver}").format(
        drive=drive, ver=ver
    )


def build_local_target_options(svc: EngineService) -> List[Dict[str, str]]:
    """Build the ordered combo options for [Local(ver)…, Cloud].

    Returns a list of ``{"data", "label"}`` dicts:
      * one ``local::<root>`` entry per registered Isaac install, labelled
        ``Local (<drive/ver>)``;
      * when none are registered AND the user has not opted into cloud-only, a
        single ``local::__none__`` entry labelled ``Local`` that the combo wires
        to the "no local Isaac" prompt;
      * always a trailing ``cloud`` entry labelled ``Cloud``.
    """
    options: List[Dict[str, str]] = []
    installs = svc.list_isaac_installations()
    if installs:
        for e in installs:
            short = format_isaac_install_label(e)
            options.append({
                "data": f"{LOCAL_PREFIX}{e.get('root', '')}",
                "label": tr("progressrow.target.local_ver", "Local ({short})")
                .format(short=short),
            })
    elif not svc.is_no_isaac_prompt_dismissed():
        options.append({
            "data": LOCAL_NONE,
            "label": tr("progressrow.target.local", "Local"),
        })
    options.append({
        "data": CLOUD_TOKEN,
        "label": tr("progressrow.target.cloud", "Cloud"),
    })
    return options


def link_kind_for_data(data: str) -> str:
    """Map a combo data token back to the legacy ``"local"`` / ``"cloud"`` kind."""
    return CLOUD_TOKEN if str(data or "").strip().lower() == CLOUD_TOKEN else "local"


_instance: Optional[EngineService] = None


def get_engine_service() -> EngineService:
    """Return the process-wide EngineService."""
    global _instance
    if _instance is None:
        _instance = EngineService()
    return _instance


__all__ = [
    "EngineService",
    "get_engine_service",
    "format_isaac_install_label",
    "build_local_target_options",
    "link_kind_for_data",
    "LOCAL_PREFIX",
    "LOCAL_NONE",
    "CLOUD_TOKEN",
]
