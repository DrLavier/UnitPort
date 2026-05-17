"""RobotAssetService — registers.robots-driven asset registry.

Reads the (brand, model) catalog from registers.robots; per-asset selection +
custom paths + family tags live in
``Paths.USER_CONFIG_DIR / "robot_assets" / "state.json"`` via push_data.

The catalog comes from the canonical robot register (registers.robots), not from
hard-coded brand strings — honouring the Multi-Brand Inclusiveness rule. When
the canonical register is empty (current state), :meth:`list_assets` returns an
empty list and the panel renders an empty-state hint.

Asset kinds: MJCF / USD / URDF / XACRO. Kinds are open-set strings — adapter
packages may register additional kinds in the future.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from PyQt6.QtCore import QObject, pyqtSignal

from registers import robots as _robots_registry
from unitport_sdk import Paths, log_warning, push_data, read_data

_STATE_REL = "robot_assets/state.json"


def _state_path() -> Path:
    """Lazy resolve of the asset state path inside the LIVE USER_CONFIG_DIR.

    Resolved at call time so workspace hot-switches are picked up without
    restart.
    """
    return Paths.USER_CONFIG_DIR / "robot_assets" / "state.json"

ASSET_KINDS = ("MJCF", "USD", "URDF", "XACRO")

# AssetRecord.status values — kept as bare strings so the panel can use them
# directly as Theme slot suffixes (robot_asset_status_<value>).
STATUS_LOCAL = "local"
STATUS_REMOTE = "remote"
STATUS_MISSING = "missing"


# ---------------------------------------------------------------------------
# Canonical-asset search roots — used by RobotAssetService.resolve()
# when a robot's canonical entry stores a relative path
# (e.g. "menagerie/unitree_go2/scene.xml").
#
# Order matters: first match wins. Project-relative locations come first,
# then the DEMO sibling tree (so RELEASE can reuse DEMO's already-cloned
# menagerie without copying), then ASSETS_DIR.
# ---------------------------------------------------------------------------
def _canonical_asset_roots() -> List[Path]:
    return [
        Paths.PROJECT_ROOT / "custom_mods" / "models",
        Paths.PROJECT_ROOT.parent / "DEMO" / "custom_mods" / "models",
        Paths.ASSETS_DIR,
    ]


class AssetStatus(Enum):
    """Tri-state availability of one asset kind for one robot.

    - LOCAL:   path resolved AND file exists on disk → ready to load.
    - REMOTE:  canonical path declared but file is missing locally — would
               need a fetch (HF / git lfs) before it can be loaded. Phase 1
               does not implement the fetch; UI surfaces this as
               "download required".
    - MISSING: nothing in user state and nothing in canonical → user must
               register a custom path first.
    """

    LOCAL = "local"
    REMOTE = "remote"
    MISSING = "missing"


@dataclass
class RobotAsset:
    """Resolved asset record for a single robot SKU.

    Output of :meth:`RobotAssetService.resolve`. Captures everything the
    sim/training stack needs about the robot's on-disk assets PLUS the
    joint-name → IR-role map from ``registers.robots``.

    Path resolution priority (per kind, computed at resolve() time):
      1. user-state ``selected[kind]`` (UI-selected path)
      2. user-state ``paths[kind][0]`` (custom-registered path)
      3. canonical ``registers.robots[sku].assets[kind]`` resolved against
         :func:`_canonical_asset_roots` candidates.
    """

    sku: str
    brand: str
    model: str
    name: str
    families: List[str] = field(default_factory=list)
    joints: Dict[str, str] = field(default_factory=dict)   # joint_name → ir_role

    mjcf_path: Optional[Path] = None
    urdf_path: Optional[Path] = None
    usd_path: Optional[Path] = None
    xacro_path: Optional[Path] = None
    # Cloud-only USD asset (Nucleus URL marker) — non-empty for IsaacLab-served
    # robots when no local USD is available. ``asset_status("USD")`` returns
    # REMOTE in that case.
    usd_url: Optional[str] = None

    # Per-kind source tag — useful for UI to show "(menagerie)" vs "(custom)".
    _path_sources: Dict[str, str] = field(default_factory=dict)

    def path_for(self, kind: str) -> Optional[Path]:
        attr = f"{str(kind).lower()}_path"
        return getattr(self, attr, None)

    def has_asset(self, kind: str) -> bool:
        """True iff the resolved path for ``kind`` exists on disk."""
        p = self.path_for(kind)
        return p is not None and p.exists()

    def asset_status(self, kind: str) -> AssetStatus:
        """LOCAL / REMOTE / MISSING for the given kind. See :class:`AssetStatus`."""
        p = self.path_for(kind)
        if p is not None:
            return AssetStatus.LOCAL if p.exists() else AssetStatus.REMOTE
        # USD-only fallback: cloud URL counts as REMOTE even with no path.
        if str(kind).upper() == "USD" and self.usd_url:
            return AssetStatus.REMOTE
        return AssetStatus.MISSING


@dataclass
class AssetRecord:
    sku: str
    brand: str
    model: str
    name: str
    families: List[str] = field(default_factory=list)
    family_tags: List[str] = field(default_factory=list)
    paths: Dict[str, List[str]] = field(default_factory=dict)        # kind -> [path, ...] (user custom only)
    selected: Dict[str, str] = field(default_factory=dict)           # kind -> selected path
    # Canonical relative paths declared in registers.robots[sku].assets[kind].
    # These are read-only display entries — the panel renders them with a
    # [canonical] tag and disables the delete button. Kept separate from
    # ``paths`` (user custom) so the two never get conflated.
    canonical: Dict[str, Optional[str]] = field(default_factory=dict)
    # Cloud-only marker URLs (currently USD via Nucleus). Display-only;
    # not selectable. None for kinds without a cloud asset.
    canonical_url: Dict[str, Optional[str]] = field(default_factory=dict)
    # Per-kind status: STATUS_LOCAL / STATUS_REMOTE / STATUS_MISSING. Drives
    # the status pill in the panel. Mirrors RobotAsset.asset_status().
    status: Dict[str, str] = field(default_factory=dict)
    # Body→IR-role overrides the user applied on top of BodyIRMapper auto-detection.
    # Shape matches application.training.body_ir.extract_user_overrides:
    # {"manual_roles": {role_id: body_link, ...}, "out_of_scope": [body, ...]}.
    body_ir_overrides: Dict[str, Any] = field(default_factory=dict)


class RobotAssetService(QObject):
    """Per-asset selection + family tags + custom path registration."""

    changed = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

    # ----- on-disk state ---------------------------------------------------

    def _load_state(self) -> Dict[str, Any]:
        if not _state_path().exists():
            return {"assets": {}}
        data = read_data(_state_path())
        if not isinstance(data, dict):
            return {"assets": {}}
        data.setdefault("assets", {})
        return data

    def _save_state(self, state: Dict[str, Any]) -> None:
        if not push_data(_STATE_REL, state):
            log_warning("[robot_assets] failed to persist state")

    def _asset_block(self, state: Dict[str, Any], sku: str) -> Dict[str, Any]:
        assets = state.setdefault("assets", {})
        block = assets.setdefault(sku, {})
        block.setdefault("paths", {})
        block.setdefault("selected", {})
        block.setdefault("family_tags", [])
        block.setdefault("body_ir_overrides", {})
        return block

    # ----- public API -----------------------------------------------------

    def list_assets(self) -> List[AssetRecord]:
        """One AssetRecord per (brand, model) known to registers.robots.

        The record bundles three layers — canonical (read-only registry),
        custom (user-added paths in state.json), status (resolve() result) —
        so panel and node render the same data without the panel needing to
        peek into the registry on its own.
        """
        try:
            skus = list(_robots_registry.list_skus())
        except Exception as exc:
            log_warning(f"[robot_assets] registers.robots load failed: {exc}")
            return []
        state = self._load_state()
        out: List[AssetRecord] = []
        for sku in skus:
            entry = _robots_registry.get_robot(sku) or {}
            block = state.get("assets", {}).get(sku, {})
            canonical_assets = entry.get("assets", {}) or {}
            canonical = {
                k: (canonical_assets.get(k) or None) for k in ASSET_KINDS
            }
            canonical_url = {
                "USD": canonical_assets.get("USD_URL") or None,
            }
            # resolve() to compute the per-kind status — same code path the
            # canvas Robot Node hits, so panel and node never disagree.
            asset = self.resolve(sku)
            status: Dict[str, str] = {}
            for k in ASSET_KINDS:
                if asset is None:
                    status[k] = STATUS_MISSING
                    continue
                ast = asset.asset_status(k)
                if ast == AssetStatus.LOCAL:
                    status[k] = STATUS_LOCAL
                elif ast == AssetStatus.REMOTE:
                    status[k] = STATUS_REMOTE
                else:
                    status[k] = STATUS_MISSING
            out.append(AssetRecord(
                sku=sku,
                brand=str(entry.get("brand", "")),
                model=str(entry.get("model", "")),
                name=str(entry.get("name", entry.get("model", ""))),
                families=list(entry.get("families", []) or []),
                family_tags=list(block.get("family_tags", []) or []),
                paths={
                    str(k).upper(): list(v or [])
                    for k, v in dict(block.get("paths", {}) or {}).items()
                },
                selected={
                    str(k).upper(): str(v or "")
                    for k, v in dict(block.get("selected", {}) or {}).items()
                },
                canonical=canonical,
                canonical_url=canonical_url,
                status=status,
                body_ir_overrides=dict(block.get("body_ir_overrides", {}) or {}),
            ))
        return out

    def get_asset(self, brand: str, model: str) -> Optional[AssetRecord]:
        from registers import resolve_robot_sku
        sku = resolve_robot_sku(brand, model)
        for rec in self.list_assets():
            if rec.sku == sku:
                return rec
        return None

    def resolve(self, sku: str) -> Optional[RobotAsset]:
        """Resolve a SKU into a fully-populated :class:`RobotAsset`.

        This is the canonical training/sim entry point — replaces the DEMO
        ``UnifiedRobotAsset.has_asset(kind)`` / ``asset_status(kind)`` API
        with one resolve() call that returns a typed record.

        Returns ``None`` if ``sku`` is unknown to ``registers.robots``.
        Otherwise returns a :class:`RobotAsset` whose path fields point at
        the highest-priority source for each asset kind (see
        :class:`RobotAsset` docstring for priority rules).
        """
        entry = _robots_registry.get_robot(sku)
        if entry is None:
            return None

        state = self._load_state()
        block = state.get("assets", {}).get(sku, {}) or {}
        block_selected = {
            str(k).upper(): str(v or "")
            for k, v in dict(block.get("selected", {}) or {}).items()
        }
        block_paths = {
            str(k).upper(): list(v or [])
            for k, v in dict(block.get("paths", {}) or {}).items()
        }

        canonical_assets = entry.get("assets", {}) or {}
        roots = _canonical_asset_roots()

        path_for_kind: Dict[str, Optional[Path]] = {}
        source_for_kind: Dict[str, str] = {}

        for kind in ASSET_KINDS:
            # Priority 1: user-selected path (if non-empty and exists)
            sel = block_selected.get(kind, "").strip()
            if sel:
                p = Path(sel).expanduser()
                if p.exists():
                    path_for_kind[kind] = p
                    source_for_kind[kind] = "selected"
                    continue
            # Priority 2: first custom-registered path that exists
            for cp in block_paths.get(kind, []):
                p = Path(str(cp)).expanduser()
                if p.exists():
                    path_for_kind[kind] = p
                    source_for_kind[kind] = "custom"
                    break
            else:
                # Priority 3: canonical relative path resolved against roots
                rel = canonical_assets.get(kind)
                resolved: Optional[Path] = None
                if rel:
                    rel_path = Path(str(rel))
                    for root in roots:
                        candidate = root / rel_path
                        if candidate.exists():
                            resolved = candidate
                            break
                    if resolved is None:
                        # Declare the first candidate even if missing — that
                        # gives REMOTE-state semantics ("would be at path X
                        # if downloaded").
                        resolved = roots[0] / rel_path
                        source_for_kind[kind] = "canonical_remote"
                    else:
                        source_for_kind[kind] = "canonical"
                path_for_kind[kind] = resolved

        joints_map: Dict[str, str] = {}
        for joint_block in (entry.get("joints", {}) or {}).values():
            if not isinstance(joint_block, dict):
                continue
            jname = str(joint_block.get("name", "")).strip()
            jrole = str(joint_block.get("ir_role", "")).strip()
            if jname:
                joints_map[jname] = jrole

        return RobotAsset(
            sku=sku,
            brand=str(entry.get("brand", "")),
            model=str(entry.get("model", "")),
            name=str(entry.get("name", entry.get("model", ""))),
            families=list(entry.get("families", []) or []),
            joints=joints_map,
            mjcf_path=path_for_kind.get("MJCF"),
            urdf_path=path_for_kind.get("URDF"),
            usd_path=path_for_kind.get("USD"),
            xacro_path=path_for_kind.get("XACRO"),
            usd_url=str(canonical_assets.get("USD_URL") or "") or None,
            _path_sources=source_for_kind,
        )

    def set_selected_path(self, brand: str, model: str, kind: str, path: str) -> None:
        from registers import resolve_robot_sku
        sku = resolve_robot_sku(brand, model)
        state = self._load_state()
        block = self._asset_block(state, sku)
        block["selected"][str(kind).upper()] = str(path)
        self._save_state(state)
        self.changed.emit()

    def add_custom_path(self, brand: str, model: str, kind: str, path: str) -> None:
        from registers import resolve_robot_sku
        sku = resolve_robot_sku(brand, model)
        if not Path(path).exists():
            log_warning(f"[robot_assets] cannot add missing path {path!r}")
            return
        state = self._load_state()
        block = self._asset_block(state, sku)
        kind_u = str(kind).upper()
        paths = block["paths"].setdefault(kind_u, [])
        if path not in paths:
            paths.append(path)
        # Auto-select the freshly added path if nothing was selected for this kind yet.
        if not block["selected"].get(kind_u):
            block["selected"][kind_u] = path
        self._save_state(state)
        self.changed.emit()

    # ----- body-IR overrides (per-asset, persists across canvases) ----------

    def get_body_ir_overrides(self, sku: str) -> Dict[str, Any]:
        """Return the persisted manual_roles + out_of_scope payload for ``sku``.

        Shape: ``{"manual_roles": {role_id: body_link, ...}, "out_of_scope": [body, ...]}``,
        matching :func:`application.training.body_ir.extract_user_overrides`.
        Returns an empty dict if nothing has been persisted for this SKU.
        """
        if not sku:
            return {}
        state = self._load_state()
        block = state.get("assets", {}).get(sku, {}) or {}
        overrides = block.get("body_ir_overrides", {}) or {}
        return dict(overrides) if isinstance(overrides, dict) else {}

    def set_body_ir_overrides(
        self, sku: str, overrides: Optional[Dict[str, Any]],
    ) -> None:
        """Persist (or clear) the user's manual body→IR-role overrides for ``sku``.

        Pass ``None`` or an empty dict to clear the block. Otherwise stores the
        ``extract_user_overrides`` payload as-is. State.json is the single source
        of truth — every canvas that resolves this SKU will replay these overrides
        on top of fresh auto-detection.
        """
        if not sku:
            return
        state = self._load_state()
        block = self._asset_block(state, sku)
        if not overrides:
            block["body_ir_overrides"] = {}
        else:
            block["body_ir_overrides"] = {
                "manual_roles": dict(overrides.get("manual_roles", {}) or {}),
                "out_of_scope": list(overrides.get("out_of_scope", []) or []),
            }
        self._save_state(state)
        self.changed.emit()

    def clear_body_ir_overrides(self, sku: str) -> None:
        self.set_body_ir_overrides(sku, None)

    def set_family_tags(self, brand: str, model: str, tags: List[str]) -> None:
        from registers import resolve_robot_sku
        sku = resolve_robot_sku(brand, model)
        state = self._load_state()
        block = self._asset_block(state, sku)
        block["family_tags"] = [str(t).strip() for t in (tags or []) if str(t).strip()]
        self._save_state(state)
        self.changed.emit()

    def scan_and_merge(self) -> int:
        """Backwards-compatible alias for :meth:`scan_and_merge_assets`.

        Older callers (panel refresh button) still use this name; it now
        runs the full filesystem scan instead of just reloading.
        """
        return self.scan_and_merge_assets()

    def scan_and_merge_assets(self) -> int:
        """Scan canonical roots for USD/URDF/XACRO and merge into the registry.

        Discovery results are written to ``~/UnitPort/registers/robots_custom.json``
        via :func:`registers.robots.persist_user_robot` — the same overlay
        layer that ``RegistryHub.load_all`` already merges. This keeps
        ``registers/`` as the **single source of truth** for asset facts;
        nothing is written to ``state.json`` (which is reserved for
        user-preference data: which path is selected, family tags, body-IR
        overrides).

        Fill-only-empty-slots semantics: discovery never overwrites a
        canonical-declared path. If ``robots_canonical.json`` already
        sets ``assets.USD = "<rel>"`` (or someone has already persisted a
        USD into the user overlay), discovery skips that slot.

        Returns the number of (sku, kind) pairs newly merged.
        """
        from .discovery import discover_local_assets
        try:
            from registers import RegistryHub
        except Exception as exc:  # noqa: BLE001
            log_warning(f"[robot_assets] RegistryHub import failed: {exc}")
            self.changed.emit()
            return 0

        try:
            discovered = discover_local_assets()
        except Exception as exc:  # noqa: BLE001
            log_warning(f"[robot_assets] discover_local_assets failed: {exc}")
            discovered = {}

        merged_pairs = 0
        for sku, kind_paths in discovered.items():
            existing = _robots_registry.get_robot(sku)
            if existing is None:
                continue
            cur_assets = dict(existing.get("assets") or {})
            changed = False
            for kind, rel in kind_paths.items():
                if cur_assets.get(kind):
                    continue  # registry already declares this kind — leave alone
                cur_assets[kind] = rel
                changed = True
                merged_pairs += 1
            if changed:
                patch = dict(existing)
                patch["assets"] = cur_assets
                _robots_registry.persist_user_robot(sku, patch)

        if merged_pairs:
            try:
                RegistryHub.reload()
            except Exception as exc:  # noqa: BLE001
                log_warning(f"[robot_assets] RegistryHub.reload failed: {exc}")

        self.changed.emit()
        return merged_pairs

    # ----- user-layer robot register CRUD ----------------------------------

    def is_user_extension(self, sku: str) -> bool:
        """True iff ``sku`` came from ``~/UnitPort/registers/robots_custom.json``."""
        try:
            return _robots_registry.is_user_extension(sku)
        except Exception:
            return False

    def add_user_robot(
        self,
        brand: str,
        model: str,
        name: str,
        families: List[str],
        adapter: str,
        assets: Dict[str, str],
        joints: Dict[str, Dict[str, str]],
    ) -> str:
        """Persist a new user-layer robot definition then reload + validate.

        Returns the resolved SKU. Caller-supplied ``joints`` shape:
        ``{joint_sku: {"name": "<raw_joint_name>", "ir_role": "<role_id>"}}``.
        """
        from registers import resolve_robot_sku
        sku = resolve_robot_sku(brand, model)
        entry: Dict[str, Any] = {
            "sku": sku,
            "brand": str(brand),
            "model": str(model),
            "name": str(name) or str(model),
            "families": [str(f) for f in (families or []) if str(f).strip()],
            "adapter": str(adapter) if adapter else "",
            "assets": {
                str(k).upper(): (str(v) if v else None)
                for k, v in (assets or {}).items()
            },
            "joints": dict(joints or {}),
            "sensors": {},
            "capabilities": {},
        }
        ok = _robots_registry.persist_user_robot(sku, entry)
        if not ok:
            log_warning(f"[robot_assets] persist_user_robot failed for sku={sku}")
            return sku
        self._reload_and_validate()
        self.changed.emit()
        return sku

    def update_user_robot(self, sku: str, patch: Dict[str, Any]) -> bool:
        """Read existing user-layer entry → merge ``patch`` → persist."""
        if not self.is_user_extension(sku):
            log_warning(f"[robot_assets] cannot update canonical sku={sku}")
            return False
        existing = _robots_registry.get_robot(sku) or {}
        merged = dict(existing)
        for k, v in (patch or {}).items():
            merged[k] = v
        ok = _robots_registry.persist_user_robot(sku, merged)
        if not ok:
            return False
        self._reload_and_validate()
        self.changed.emit()
        return True

    def remove_user_robot(self, sku: str) -> bool:
        """Delete a user-layer robot AND its per-asset state.json block."""
        if not self.is_user_extension(sku):
            log_warning(f"[robot_assets] cannot remove canonical sku={sku}")
            return False
        ok = _robots_registry.delete_user_robot(sku)
        if not ok:
            return False
        # Also drop the per-asset user state for this SKU so a re-added
        # entry starts clean.
        state = self._load_state()
        if sku in state.get("assets", {}):
            state["assets"].pop(sku, None)
            self._save_state(state)
        self._reload_and_validate()
        self.changed.emit()
        return True

    def _reload_and_validate(self) -> None:
        # CLAUDE.md §1.2 red line: ``RegistryValidationError`` (unmapped
        # bodies / IR-role mismatches) MUST surface to the user, not be
        # silently logged. The UI dialogs that drive add/update wrap the
        # service call in try/except → QMessageBox; re-raising lets that
        # surface to the user instead of vanishing into a warning line.
        from registers import RegistryHub, RegistryValidationError
        try:
            RegistryHub.reload()
        except Exception as exc:  # noqa: BLE001
            log_warning(f"[robot_assets] reload failed: {exc}")
            return
        try:
            RegistryHub.validate()
        except RegistryValidationError:
            raise


_instance: Optional[RobotAssetService] = None


def get_robot_asset_service() -> RobotAssetService:
    global _instance
    if _instance is None:
        _instance = RobotAssetService()
    return _instance


# ---------------------------------------------------------------------------
# Globally selected robot — used by the mission_panel "robot hardware info"
# card. Distinct from RobotAssetService.changed (which fires on any registry
# / state.json mutation): this is purely "which robot is the user currently
# looking at?". Empty string ('') means no robot selected.
# ---------------------------------------------------------------------------

_selected_sku: str = ""


def selected_robot_sku() -> str:
    return _selected_sku


def selected_robot() -> Optional[RobotAsset]:
    """Resolve the currently selected SKU into a RobotAsset, or None."""
    if not _selected_sku:
        return None
    try:
        return get_robot_asset_service().resolve(_selected_sku)
    except Exception:
        return None


def set_selected_robot(sku: str) -> None:
    """Update the global selection and emit ``robot_selection_changed``.

    Empty / None clears the selection. No-op when the value is unchanged.
    """
    global _selected_sku
    new = str(sku or "")
    if new == _selected_sku:
        return
    _selected_sku = new
    # Lazy import to avoid circular: signals.py imports nothing from here,
    # but robot_assets/__init__.py is imported during early bootstrap.
    from application.service.signals import get_app_signals
    get_app_signals().robot_selection_changed.emit(new)


__all__ = [
    "RobotAssetService",
    "get_robot_asset_service",
    "selected_robot_sku",
    "selected_robot",
    "set_selected_robot",
    "AssetRecord",
    "RobotAsset",
    "AssetStatus",
    "ASSET_KINDS",
    "STATUS_LOCAL",
    "STATUS_REMOTE",
    "STATUS_MISSING",
]
