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
from unitport_sdk import (
    Paths,
    Task,
    get_tasks_manager,
    log_info,
    log_warning,
    push_data,
    read_data,
)

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
        Paths.CUSTOM_MODS_DIR / "models",
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
    per-format joint/body → IR-role tables from ``registers.robots``.

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
    # Per-format joint/body tables (Stage 1 schema). Each entry is
    # ``{joint_name: ir_role}`` for the format. Absent format = format
    # not declared for this robot.
    joints_per_format: Dict[str, Dict[str, str]] = field(default_factory=dict)
    bodies_per_format: Dict[str, Dict[str, str]] = field(default_factory=dict)
    # Legacy flat joint→ir_role map (kept for transitional callers; merged
    # across formats with MJCF preference). New code should pick a format
    # explicitly via ``joints_for(fmt)`` / ``bodies_for(fmt)``.
    joints: Dict[str, str] = field(default_factory=dict)

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

    # --- per-format accessors -----------------------------------------------

    def joints_for(self, fmt: str) -> Dict[str, str]:
        """Return ``{joint_name: ir_role}`` for the given format, or ``{}``."""
        return dict(self.joints_per_format.get(str(fmt).upper(), {}) or {})

    def bodies_for(self, fmt: str) -> Dict[str, str]:
        """Return ``{body_name: ir_role}`` for the given format, or ``{}``."""
        return dict(self.bodies_per_format.get(str(fmt).upper(), {}) or {})

    def available_formats(self) -> List[str]:
        """Asset formats with a non-empty joints OR bodies declaration."""
        out: List[str] = []
        for fmt in ASSET_KINDS:
            if self.joints_per_format.get(fmt) or self.bodies_per_format.get(fmt):
                out.append(fmt)
        return out


@dataclass
class DumpResult:
    """Structured result of :meth:`RobotAssetService.dump_and_persist`.

    On success: ``ok=True``, ``joints``/``bodies`` populated, ``error=None``,
    ``error_kind=""``. The user-overlay registry has been written, reloaded,
    and ``RobotAssetService.changed`` has been emitted — callers only need to
    rebuild their own widgets.

    On failure: ``ok=False``, ``error`` holds a human-readable message, and
    ``error_kind`` identifies which step broke:

    - ``"no_asset"``    — SKU empty or asset unresolvable
    - ``"discovery"``   — :class:`DiscoveryError` from MJCF/USD dump (missing
                          path/URL, subprocess crash, parse failure, timeout)
    - ``"validation"``  — :class:`RegistryValidationError` from persist
                          (unknown ir_role in payload)
    - ``"persist"``     — write returned False (atomic write failed, etc.)
    - ``"unexpected"``  — any other exception
    """

    ok: bool
    joints: Dict[str, str] = field(default_factory=dict)
    bodies: Dict[str, str] = field(default_factory=dict)
    error: Optional[str] = None
    error_kind: str = ""

    @property
    def n_joints_resolved(self) -> int:
        return sum(1 for ir in self.joints.values() if ir)

    @property
    def n_bodies_resolved(self) -> int:
        return sum(1 for ir in self.bodies.values() if ir)

    @property
    def n_unresolved(self) -> int:
        return (
            (len(self.joints) - self.n_joints_resolved)
            + (len(self.bodies) - self.n_bodies_resolved)
        )


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
    # Variant hierarchy fields (Stage 2 schema):
    #   model_family : groups variants under one UI card; defaults to ``model``.
    #   variant_label: chip-display label ("Standard" / "Air" / ...).
    #   parent_sku   : non-empty iff this record inherits from another robot.
    model_family: str = ""
    variant_label: str = ""
    parent_sku: Optional[str] = None


@dataclass
class AssetRecordGroup:
    """One (brand, model_family) bucket — fed to the panel for card rendering.

    A group with a single record renders as the existing flat :class:`_AssetCard`.
    A group with multiple records (base + variants) renders as a
    :class:`_ModelGroupCard` with a variant chip strip.
    """

    brand: str
    model_family: str
    records: List[AssetRecord] = field(default_factory=list)


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
            model = str(entry.get("model", ""))
            family_default = model
            out.append(AssetRecord(
                sku=sku,
                brand=str(entry.get("brand", "")),
                model=model,
                name=str(entry.get("name", model)),
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
                model_family=str(entry.get("model_family") or family_default),
                variant_label=str(entry.get("variant_label") or "Standard"),
                parent_sku=_robots_registry.get_parent_sku(sku),
            ))
        return out

    def get_asset(self, brand: str, model: str) -> Optional[AssetRecord]:
        from registers import resolve_robot_sku
        sku = resolve_robot_sku(brand, model)
        for rec in self.list_assets():
            if rec.sku == sku:
                return rec
        return None

    # ----- Menagerie package matching (Stage 4) ------------------------

    @staticmethod
    def _split_dir_to_brand_model(dir_name: str) -> List[tuple]:
        """All plausible (brand, model) splits for a menagerie dir name.

        Snake-case packages can fold the brand-model boundary anywhere
        (``unitree_go2`` -> single split; ``franka_fr3_v2`` -> two splits).
        Callers try each candidate in order; the first that resolves to a
        registered SKU wins. The full-name fallback is the dir itself,
        used to drive an alias lookup as a last resort.
        """
        parts = [p for p in dir_name.split("_") if p]
        out: List[tuple] = []
        for i in range(1, len(parts)):
            out.append(("_".join(parts[:i]), "_".join(parts[i:])))
        return out

    def try_match_menagerie_package(self, dir_name: str) -> Optional[str]:
        """Resolve a menagerie dir name to a registered SKU, or None.

        Lookup order:
          1. Every (brand, model) split → ``resolve_robot_sku`` → registry hit.
          2. ``resolve_id(dir_name)``  → alias / display-name fold.
          3. ``resolve_id(model_token)`` for the last split's model token.

        Returns the resolved SKU on the first hit; ``None`` if nothing
        matches. Callers should then open the editor for new registration.
        """
        from registers import resolve_robot_sku
        clean = (dir_name or "").strip().lower()
        if not clean:
            return None
        for brand, model in self._split_dir_to_brand_model(clean):
            sku = resolve_robot_sku(brand, model)
            if _robots_registry.get_robot(sku) is not None:
                return sku
        # Alias / display-name fold
        sku = _robots_registry.resolve_id(clean)
        if sku is not None:
            return sku
        # Last-fallback: alias-lookup of the trailing model token alone.
        parts = clean.split("_")
        if len(parts) >= 2:
            sku = _robots_registry.resolve_id(parts[-1])
            if sku is not None:
                return sku
        return None

    @staticmethod
    def menagerie_register_hint(dir_name: str) -> Dict[str, str]:
        """Pre-fill payload for ``RobotEditorDialog`` when registering an
        unmatched menagerie package as a new robot.

        Returns a flat dict with keys: ``brand``, ``model``, ``name``, and
        ``mjcf`` (relative path under ``menagerie/<dir>/scene.xml``).
        Uses the last-token split heuristic — the user can override
        anything in the dialog before saving.
        """
        clean = (dir_name or "").strip()
        if not clean:
            return {"brand": "", "model": "", "name": "", "mjcf": ""}
        parts = clean.split("_")
        if len(parts) >= 2:
            brand = parts[0]
            model = "_".join(parts[1:])
        else:
            brand = ""
            model = clean
        return {
            "brand": brand,
            "model": model,
            "name": clean.replace("_", " ").title(),
            "mjcf": f"menagerie/{clean}/scene.xml",
        }

    def list_assets_grouped(self) -> List["AssetRecordGroup"]:
        """Return AssetRecords grouped by (brand, model_family).

        Each group's records are ordered: base entry first (no parent),
        then variants sorted by variant_label. Groups themselves are
        sorted by (brand, family) ascending — same ordering the registry
        uses for ``list_model_families()`` — so the panel renders the
        catalogue alphabetically.

        Each group is rendered as either a flat :class:`_AssetCard`
        (single record) or a :class:`_ModelGroupCard` (multiple records)
        in the sidebar panel.
        """
        records = self.list_assets()
        groups: Dict[tuple, List[AssetRecord]] = {}
        for rec in records:
            key = (
                (rec.brand or "").strip().lower(),
                (rec.model_family or rec.model or "").strip().lower(),
            )
            groups.setdefault(key, []).append(rec)
        out: List[AssetRecordGroup] = []
        for key in sorted(groups.keys()):
            bucket = groups[key]
            bucket.sort(key=lambda r: (
                # base (no parent) first; variants sorted by label
                0 if not r.parent_sku else 1,
                (r.variant_label or "").lower(),
                r.sku,
            ))
            out.append(AssetRecordGroup(
                brand=key[0],
                model_family=key[1],
                records=bucket,
            ))
        return out

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

        # Per-format schema (Stage 1): joints/bodies live under
        # ``joints_per_format[<FORMAT>]`` / ``bodies_per_format[<FORMAT>]``.
        # Convert raw {uid: {name, ir_role}} dicts to flat {name: ir_role}.
        def _flatten(block: Optional[Dict[str, Any]]) -> Dict[str, str]:
            out: Dict[str, str] = {}
            if not isinstance(block, dict):
                return out
            for spec in block.values():
                if not isinstance(spec, dict):
                    continue
                nm = str(spec.get("name", "")).strip()
                rl = str(spec.get("ir_role", "")).strip()
                if nm:
                    out[nm] = rl
            return out

        raw_joints_pf = entry.get("joints_per_format", {}) or {}
        raw_bodies_pf = entry.get("bodies_per_format", {}) or {}
        joints_per_format: Dict[str, Dict[str, str]] = {}
        bodies_per_format: Dict[str, Dict[str, str]] = {}
        for fmt in ASSET_KINDS:
            j = _flatten(raw_joints_pf.get(fmt))
            if j:
                joints_per_format[fmt] = j
            b = _flatten(raw_bodies_pf.get(fmt))
            if b:
                bodies_per_format[fmt] = b

        # Legacy flat ``joints`` view: merge across formats with MJCF
        # priority (MJCF last → wins on conflict). Same shape downstream
        # callers expect during the transition.
        joints_map: Dict[str, str] = {}
        for fmt in ("URDF", "USD", "MJCF"):
            joints_map.update(joints_per_format.get(fmt, {}))

        return RobotAsset(
            sku=sku,
            brand=str(entry.get("brand", "")),
            model=str(entry.get("model", "")),
            name=str(entry.get("name", entry.get("model", ""))),
            families=list(entry.get("families", []) or []),
            joints_per_format=joints_per_format,
            bodies_per_format=bodies_per_format,
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

    # ----- body-IR overrides (per-asset × per-format) -----------------------
    #
    # State.json shape (Stage 1 upgrade):
    #     "body_ir_overrides": {
    #         "MJCF": {"manual_roles": {...}, "out_of_scope": [...]},
    #         "USD":  {"manual_roles": {...}, "out_of_scope": [...]}
    #     }
    #
    # Legacy shape (pre-migration) was a single flat dict. The reader
    # auto-migrates on access by wrapping the legacy dict under "MJCF"
    # (the safest default — every robot with overrides had them based on
    # MJCF body discovery before this refactor).

    @staticmethod
    def _is_legacy_overrides_shape(block: Dict[str, Any]) -> bool:
        """A flat block has top-level ``manual_roles`` / ``out_of_scope``;
        the per-format block has top-level format keys (MJCF/USD/URDF)."""
        if not isinstance(block, dict) or not block:
            return False
        # If any top-level key is a known format, treat as already per-format.
        for k in block.keys():
            if str(k).upper() in ASSET_KINDS:
                return False
        # Treat as legacy iff it has the legacy shape keys
        return "manual_roles" in block or "out_of_scope" in block

    def _read_overrides_block(self, sku: str) -> Dict[str, Dict[str, Any]]:
        """Read overrides for ``sku`` in normalised per-format shape."""
        if not sku:
            return {}
        state = self._load_state()
        block = state.get("assets", {}).get(sku, {}) or {}
        raw = block.get("body_ir_overrides", {}) or {}
        if not isinstance(raw, dict):
            return {}
        if self._is_legacy_overrides_shape(raw):
            # Wrap legacy flat overrides as MJCF
            return {
                "MJCF": {
                    "manual_roles": dict(raw.get("manual_roles", {}) or {}),
                    "out_of_scope": list(raw.get("out_of_scope", []) or []),
                }
            }
        out: Dict[str, Dict[str, Any]] = {}
        for fmt, sub in raw.items():
            fmt_u = str(fmt).upper()
            if fmt_u not in ASSET_KINDS or not isinstance(sub, dict):
                continue
            out[fmt_u] = {
                "manual_roles": dict(sub.get("manual_roles", {}) or {}),
                "out_of_scope": list(sub.get("out_of_scope", []) or []),
            }
        return out

    def get_body_ir_overrides(
        self, sku: str, fmt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return persisted manual_roles + out_of_scope for ``sku``.

        Shape: ``{"manual_roles": {role_id: body_link, ...}, "out_of_scope": [body, ...]}``,
        matching :func:`application.training.body_ir.extract_user_overrides`.

        Calling with ``fmt=None`` returns the **MJCF** overrides (legacy
        compatibility — callers that haven't migrated yet expect a flat
        block). New code should pass an explicit ``fmt``. Pass
        ``fmt="<FORMAT>"`` to retrieve overrides for that specific format.
        Returns an empty dict if nothing is persisted.
        """
        normalised = self._read_overrides_block(sku)
        if fmt is None:
            return dict(normalised.get("MJCF", {}))
        fmt_u = str(fmt).upper()
        return dict(normalised.get(fmt_u, {}))

    def get_body_ir_overrides_all(self, sku: str) -> Dict[str, Dict[str, Any]]:
        """Return the full per-format overrides map (``{FORMAT: block}``)."""
        return self._read_overrides_block(sku)

    def set_body_ir_overrides(
        self,
        sku: str,
        overrides: Optional[Dict[str, Any]],
        fmt: Optional[str] = None,
    ) -> None:
        """Persist (or clear) the user's body→IR-role overrides for one format.

        Pass ``overrides=None`` (or empty dict) to clear the format's
        overrides. ``fmt=None`` writes to MJCF for legacy compatibility;
        new code should pass an explicit format.

        State.json is the single source of truth — every canvas that resolves
        this SKU will replay these overrides on top of fresh auto-detection.
        """
        if not sku:
            return
        fmt_u = (str(fmt).upper() if fmt else "MJCF")
        if fmt_u not in ASSET_KINDS:
            log_warning(f"[robot_assets] unknown format {fmt!r}; ignored")
            return
        state = self._load_state()
        block = self._asset_block(state, sku)
        raw = block.get("body_ir_overrides", {}) or {}
        if not isinstance(raw, dict):
            raw = {}
        # Auto-migrate legacy flat shape to per-format before merging.
        if self._is_legacy_overrides_shape(raw):
            raw = {
                "MJCF": {
                    "manual_roles": dict(raw.get("manual_roles", {}) or {}),
                    "out_of_scope": list(raw.get("out_of_scope", []) or []),
                }
            }
        if not overrides:
            raw.pop(fmt_u, None)
        else:
            raw[fmt_u] = {
                "manual_roles": dict(overrides.get("manual_roles", {}) or {}),
                "out_of_scope": list(overrides.get("out_of_scope", []) or []),
            }
        block["body_ir_overrides"] = raw
        self._save_state(state)
        self.changed.emit()

    def clear_body_ir_overrides(self, sku: str, fmt: Optional[str] = None) -> None:
        """Clear overrides for one format (default MJCF) or all formats."""
        if fmt == "*":
            if not sku:
                return
            state = self._load_state()
            block = self._asset_block(state, sku)
            block["body_ir_overrides"] = {}
            self._save_state(state)
            self.changed.emit()
            return
        self.set_body_ir_overrides(sku, None, fmt=fmt)

    # ----- MJCF base spawn-Z offset overlay (USD↔MJCF anchor compensation) --
    #
    # state.json shape:
    #     "mjcf_base_height_offset": {
    #         "schema": "unitport.mjcf_base_offset/v1",
    #         "status": "calibrated" | "not_applicable",
    #         "offset_z": float,
    #         "calibration": { ...metadata, landmark_z, sha hashes... }
    #     }
    #
    # Computed by application.training.validation.mjcf_base_calibration
    # at standing pose with multi-landmark IR assessment; applied at every
    # MuJoCo spawn site (Review Pose, generic_mujoco_env, mj_sim_env).

    def get_mjcf_base_offset(self, sku: str) -> Optional[Dict[str, Any]]:
        """Return the raw overlay dict for ``sku`` or ``None`` if not present.

        Callers that just need the float should prefer
        :func:`application.service.robot_assets.runtime.read_mjcf_base_offset`,
        which centralises the WARN-on-missing contract (§1.8).
        """
        if not sku:
            return None
        state = self._load_state()
        block = state.get("assets", {}).get(sku, {}) or {}
        overlay = block.get("mjcf_base_height_offset")
        if not isinstance(overlay, dict) or not overlay:
            return None
        return overlay

    def set_mjcf_base_offset(self, sku: str, result: Any) -> None:
        """Persist a :class:`MjcfBaseOffsetResult` (or its overlay dict) for ``sku``."""
        if not sku:
            return
        # Accept either the dataclass or a pre-serialised dict — keeps
        # callers (calibration task, tests, future schema migration) from
        # needing to import the dataclass.
        if hasattr(result, "to_overlay_dict"):
            payload = result.to_overlay_dict()
        elif isinstance(result, dict):
            payload = dict(result)
        else:
            log_warning(
                f"[robot_assets] set_mjcf_base_offset({sku!r}): unsupported "
                f"result type {type(result).__name__}; ignored"
            )
            return
        state = self._load_state()
        block = self._asset_block(state, sku)
        block["mjcf_base_height_offset"] = payload
        self._save_state(state)
        self.changed.emit()

    def clear_mjcf_base_offset(self, sku: str) -> None:
        """Wipe the offset overlay for ``sku`` (forces recalibration on next
        Robot Node load)."""
        if not sku:
            return
        state = self._load_state()
        block = self._asset_block(state, sku)
        block.pop("mjcf_base_height_offset", None)
        self._save_state(state)
        self.changed.emit()

    # ----- per-format discovered tables (Dump MJCF / Dump USD button) -------

    def set_discovered_bodies(
        self, sku: str, fmt: str, joints: Dict[str, str], bodies: Dict[str, str],
    ) -> bool:
        """Persist a Dump-button payload into the user-overlay robot register.

        Used by the Robot Asset card's "Dump USD/MJCF" buttons to back-fill
        ``joints_per_format[fmt]`` / ``bodies_per_format[fmt]`` from a live
        asset parse (USD via Isaac Sim subprocess, MJCF via mujoco). The
        write lands in ``<USER_CONFIG_DIR>/registers/robots_custom.json`` so it
        survives reinstall and doesn't mutate the factory canonical.

        ``joints`` / ``bodies`` are flat ``{name: ir_role}`` dicts. We
        convert to the canonical ``{uid: {name, ir_role}}`` shape before
        persisting.
        """
        if not sku:
            return False
        fmt_u = str(fmt).upper()
        if fmt_u not in ASSET_KINDS:
            log_warning(f"[robot_assets] unknown format {fmt!r}; discovery ignored")
            return False
        from registers import resolve_joint_sku
        existing = _robots_registry.get_robot(sku) or {}
        patch = dict(existing)
        # joints_per_format
        jpf = dict(patch.get("joints_per_format", {}) or {})
        if joints:
            jpf[fmt_u] = {
                resolve_joint_sku(sku, nm): {"name": nm, "ir_role": rl}
                for nm, rl in joints.items()
            }
        patch["joints_per_format"] = {
            f: jpf.get(f) for f in ASSET_KINDS
        }
        # bodies_per_format
        bpf = dict(patch.get("bodies_per_format", {}) or {})
        if bodies:
            bpf[fmt_u] = {
                resolve_joint_sku(sku, nm): {"name": nm, "ir_role": rl}
                for nm, rl in bodies.items()
            }
        patch["bodies_per_format"] = {
            f: bpf.get(f) for f in ASSET_KINDS
        }
        ok = _robots_registry.persist_user_robot(sku, patch)
        if not ok:
            log_warning(f"[robot_assets] set_discovered_bodies failed for sku={sku}")
            return False
        self._reload_and_validate()
        self.changed.emit()
        return True

    def update_ir_role(
        self, sku: str, fmt: str, kind: str, uid: str, ir_role: str,
    ) -> bool:
        """Patch a single ``ir_role`` field in the user-overlay registry.

        Used by IRRoleAssignmentDialog (boot-time self-check) and any
        future caller that needs to set ``joints_per_format[fmt][uid].ir_role``
        or ``bodies_per_format[fmt][uid].ir_role`` *without* rewriting the
        full per-format table (which would clobber concurrent assignments
        — :meth:`set_discovered_bodies` does that wholesale rewrite).

        Distinct from :meth:`set_body_ir_overrides` — that one writes the
        ``manual_roles`` / ``out_of_scope`` blob the canvas mapper uses
        to overlay a re-mapping on top of the registry's ir_role; this
        method writes the registry's ir_role itself.

        Args:
            sku: robot SKU (already resolved).
            fmt: ``"MJCF"`` / ``"USD"`` / ``"URDF"``.
            kind: ``"joint"`` or ``"body"``.
            uid: the per-format entry uid (the key in joints_per_format[fmt]).
            ir_role: new role id; empty string clears the assignment.

        Returns True iff the write+reload succeeded.
        """
        if not sku:
            return False
        fmt_u = str(fmt).upper()
        if fmt_u not in ASSET_KINDS:
            log_warning(f"[robot_assets] update_ir_role: bad fmt {fmt!r}")
            return False
        kind_l = str(kind).lower()
        if kind_l not in ("joint", "body"):
            log_warning(f"[robot_assets] update_ir_role: bad kind {kind!r}")
            return False
        if not uid:
            return False

        block_key = "joints_per_format" if kind_l == "joint" else "bodies_per_format"
        existing = _robots_registry.get_robot(sku) or {}
        patch = dict(existing)

        blocks = dict(patch.get(block_key, {}) or {})
        fmt_table = dict(blocks.get(fmt_u, {}) or {})
        entry = dict(fmt_table.get(uid, {}) or {})
        if not entry:
            log_warning(
                f"[robot_assets] update_ir_role: uid {uid!r} not present "
                f"in {block_key}[{fmt_u!r}] for sku={sku!r}"
            )
            return False
        entry["ir_role"] = str(ir_role or "")
        fmt_table[uid] = entry
        blocks[fmt_u] = fmt_table
        # Preserve all formats — never collapse other fmts to None.
        patch[block_key] = {f: blocks.get(f) for f in ASSET_KINDS}

        if not _robots_registry.persist_user_robot(sku, patch):
            log_warning(f"[robot_assets] update_ir_role: persist failed for sku={sku!r}")
            return False
        self._reload_and_validate()
        self.changed.emit()
        return True

    def bulk_update_ir_roles(
        self, patches: List[tuple],
    ) -> int:
        """Apply many ``ir_role`` patches to the user-overlay registry in
        one pass — one ``persist_user_robot`` per SKU, **one** final
        ``_reload_and_validate`` + ``changed.emit`` for the whole batch.

        Required for any UI that confirms a list of assignments at once
        (e.g. :class:`IRRoleAssignmentDialog`). Calling
        :meth:`update_ir_role` in a loop triggers a registry reload AND
        a full ``RegistryHub.validate()`` per call — for a 30-entry
        confirm that's 30 reloads + 30 validations spraying the same
        warning lines, plus a ``changed`` signal storm that re-rebuilds
        every listener N times.

        Args:
            patches: list of ``(sku, fmt, kind, uid, ir_role)`` tuples.
                Order doesn't matter; the method groups by sku before
                writing. Empty list is a no-op.

        Returns:
            Count of entries successfully patched. Failures (bad
            kind/fmt, missing uid, persist failure) are logged at
            WARNING level and skipped — the rest of the batch still
            commits.
        """
        if not patches:
            return 0
        from collections import defaultdict

        # Group by sku so each user-overlay JSON gets at most one write
        # this batch.
        by_sku: Dict[str, List[tuple]] = defaultdict(list)
        for tup in patches:
            try:
                sku, fmt, kind, uid, ir_role = tup
            except (TypeError, ValueError):
                log_warning(
                    f"[robot_assets] bulk_update_ir_roles: bad tuple {tup!r}"
                )
                continue
            if not sku:
                continue
            fmt_u = str(fmt).upper()
            if fmt_u not in ASSET_KINDS:
                log_warning(f"[robot_assets] bulk: bad fmt {fmt!r}")
                continue
            kind_l = str(kind).lower()
            if kind_l not in ("joint", "body"):
                log_warning(f"[robot_assets] bulk: bad kind {kind!r}")
                continue
            if not uid:
                continue
            by_sku[sku].append((fmt_u, kind_l, str(uid), str(ir_role or "")))

        n_patched = 0
        for sku, items in by_sku.items():
            existing = _robots_registry.get_robot(sku) or {}
            patch = dict(existing)
            # Copy the per-format containers down to per-uid level so
            # multiple patches against the same fmt don't lose each
            # other.
            jpf_raw = dict(patch.get("joints_per_format", {}) or {})
            bpf_raw = dict(patch.get("bodies_per_format", {}) or {})
            jpf = {f: dict(jpf_raw.get(f) or {}) for f in ASSET_KINDS}
            bpf = {f: dict(bpf_raw.get(f) or {}) for f in ASSET_KINDS}
            for fmt_u, kind_l, uid, ir_role in items:
                container = jpf if kind_l == "joint" else bpf
                fmt_table = container[fmt_u]
                entry = dict(fmt_table.get(uid) or {})
                if not entry:
                    log_warning(
                        f"[robot_assets] bulk: uid {uid!r} not present in "
                        f"{('joints' if kind_l=='joint' else 'bodies')}_per_format"
                        f"[{fmt_u!r}] for sku={sku!r}"
                    )
                    continue
                entry["ir_role"] = ir_role
                fmt_table[uid] = entry
                n_patched += 1
            # Preserve all ASSET_KINDS slots so unrelated formats don't
            # get nulled out by accident.
            patch["joints_per_format"] = jpf
            patch["bodies_per_format"] = bpf
            if not _robots_registry.persist_user_robot(sku, patch):
                log_warning(
                    f"[robot_assets] bulk: persist failed for sku={sku!r}"
                )
        # Single reload + single signal emit for the whole batch.
        if n_patched:
            self._reload_and_validate()
            self.changed.emit()
        return n_patched

    def dump_and_persist(self, sku: str, fmt: str) -> DumpResult:
        """Dump bodies+joints for ``fmt`` from the live asset and persist
        to the user-overlay registry.

        MJCF runs in-process via mujoco (~0.1s); USD spawns the Isaac Lab
        venv subprocess to run ``bootstrap/discover_usd_bodies.py`` (5-30s
        depending on whether pxr alone or the kit fallback handles the
        asset). Both paths land in
        ``<USER_CONFIG_DIR>/registers/robots_custom.json`` via
        :meth:`set_discovered_bodies`; on success the registry is reloaded
        in-place and ``changed`` is emitted, so listeners (canvas Body
        Mapping table, Mission Control table) rebuild automatically.

        The caller owns UI feedback (busy cursor, dialogs, log prefix) —
        this method only returns a :class:`DumpResult` describing what
        happened.
        """
        from application.service.robot_assets.discovery_subprocess import (
            DiscoveryError,
            dump_format,
        )
        from registers import RegistryValidationError

        fmt_u = str(fmt or "").upper()
        if not sku:
            return DumpResult(ok=False, error="empty SKU", error_kind="no_asset")
        if fmt_u not in ASSET_KINDS:
            return DumpResult(
                ok=False, error=f"unknown format {fmt!r}", error_kind="no_asset",
            )

        asset = self.resolve(sku)
        if asset is None:
            return DumpResult(
                ok=False,
                error=f"cannot resolve robot asset for SKU {sku!r}",
                error_kind="no_asset",
            )

        mjcf_joints = asset.joints_for("MJCF") or {}
        families = list(getattr(asset, "families", []) or [])
        family = families[0] if families else "generic"

        try:
            joints, bodies = dump_format(asset, fmt_u, mjcf_joints, family=family)
        except DiscoveryError as exc:
            return DumpResult(ok=False, error=str(exc), error_kind="discovery")
        except Exception as exc:
            return DumpResult(
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
                error_kind="unexpected",
            )

        try:
            ok = self.set_discovered_bodies(sku, fmt_u, joints, bodies)
        except RegistryValidationError as exc:
            return DumpResult(
                ok=False, joints=joints, bodies=bodies,
                error=str(exc), error_kind="validation",
            )
        except Exception as exc:
            return DumpResult(
                ok=False, joints=joints, bodies=bodies,
                error=f"{type(exc).__name__}: {exc}",
                error_kind="unexpected",
            )

        if not ok:
            return DumpResult(
                ok=False, joints=joints, bodies=bodies,
                error="persist returned False",
                error_kind="persist",
            )

        return DumpResult(ok=True, joints=joints, bodies=bodies)

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

        Discovery results are written to ``<USER_CONFIG_DIR>/registers/robots_custom.json``
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
        """True iff ``sku`` came from ``<USER_CONFIG_DIR>/registers/robots_custom.json``."""
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
        joints_format: str = "MJCF",
        bodies: Optional[Dict[str, Dict[str, str]]] = None,
        bodies_format: Optional[str] = None,
    ) -> str:
        """Persist a new user-layer robot definition then reload + validate.

        Returns the resolved SKU. Caller-supplied ``joints`` shape:
        ``{joint_sku: {"name": "<raw_joint_name>", "ir_role": "<role_id>"}}``.

        The ``joints`` block is parked under ``joints_per_format[joints_format]``
        (default MJCF). Other formats stay null until the user runs Dump
        MJCF / Dump USD against the asset.

        ``bodies`` (optional) is the parallel body-side mapping for the
        same format; ``bodies_format`` defaults to ``joints_format``.
        """
        from registers import resolve_robot_sku
        sku = resolve_robot_sku(brand, model)
        jfmt = str(joints_format).upper() if joints_format else "MJCF"
        bfmt = str(bodies_format).upper() if bodies_format else jfmt
        if jfmt not in ASSET_KINDS:
            log_warning(f"[robot_assets] unknown joints_format {joints_format!r}; coerced to MJCF")
            jfmt = "MJCF"
        if bfmt not in ASSET_KINDS:
            bfmt = jfmt

        joints_pf: Dict[str, Any] = {f: None for f in ASSET_KINDS}
        if joints:
            joints_pf[jfmt] = dict(joints)
        bodies_pf: Dict[str, Any] = {f: None for f in ASSET_KINDS}
        if bodies:
            bodies_pf[bfmt] = dict(bodies)

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
            "joints_per_format": joints_pf,
            "bodies_per_format": bodies_pf,
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
        """Read existing user-layer entry → merge ``patch`` → persist.

        Round-trip uses the **raw** registry view (pre-inheritance) so saving
        an edit on a variant never writes the parent's inherited fields back
        into the user overlay (which would break inheritance).
        """
        if not self.is_user_extension(sku):
            log_warning(f"[robot_assets] cannot update canonical sku={sku}")
            return False
        existing = _robots_registry.get_raw_robot(sku) or _robots_registry.get_robot(sku) or {}
        merged = dict(existing)
        for k, v in (patch or {}).items():
            merged[k] = v
        ok = _robots_registry.persist_user_robot(sku, merged)
        if not ok:
            return False
        self._reload_and_validate()
        self.changed.emit()
        return True

    def add_user_variant_robot(
        self,
        parent_sku: str,
        variant_label: str,
        name: str,
        *,
        brand: Optional[str] = None,
        model: Optional[str] = None,
        adapter: str = "",
        sdk_paths: Optional[List[str]] = None,
        capabilities: Optional[Dict[str, Any]] = None,
        aliases: Optional[List[str]] = None,
    ) -> str:
        """Persist a new user-layer variant under ``parent_sku``.

        Default brand / model derive from the parent so the variant SKU is
        unique per (brand, model). When ``model`` is omitted, the script
        appends ``"_<slug(variant_label)>"`` to the parent's model so SKU
        collisions are avoided (e.g. Go2 -> Go2 Field-Test -> "go2_ft").
        """
        from registers import resolve_robot_sku
        parent = _robots_registry.get_robot(parent_sku) or {}
        if not parent:
            raise ValueError(f"parent_sku {parent_sku} is not a registered robot")
        if _robots_registry.is_variant(parent_sku):
            raise ValueError(
                f"parent_sku {parent_sku} is itself a variant; chains forbidden"
            )

        v_brand = (brand or parent.get("brand") or "").strip()
        if not v_brand:
            raise ValueError("variant requires a brand (inherited or explicit)")

        if model:
            v_model = str(model).strip()
        else:
            slug = "".join(c if c.isalnum() else "_" for c in variant_label.lower()).strip("_")
            v_model = f"{parent.get('model', '')}_{slug}" if slug else str(parent.get("model", ""))
        if not v_model:
            raise ValueError("variant requires a non-empty model")

        sku = resolve_robot_sku(v_brand, v_model)
        if sku == parent_sku:
            raise ValueError(
                "variant resolves to the same SKU as its parent; "
                "pick a distinct variant_label / model slug"
            )

        family = parent.get("model_family") or parent.get("model") or v_model
        entry: Dict[str, Any] = {
            "sku": sku,
            "brand": v_brand,
            "model": v_model,
            "name": str(name) or f"{parent.get('name', '')} {variant_label}".strip(),
            "model_family": str(family),
            "variant_label": str(variant_label) or "Variant",
            "inherits_from": parent_sku,
            "aliases": list(aliases or []),
            "families": list(parent.get("families", []) or []),
            "adapter": str(adapter or parent.get("adapter") or ""),
            "sdk_paths": list(sdk_paths or []),
            "capabilities": dict(capabilities or {}),
        }
        ok = _robots_registry.persist_user_robot(sku, entry)
        if not ok:
            log_warning(f"[robot_assets] persist_user_robot failed for sku={sku}")
            return sku
        self._reload_and_validate()
        self.changed.emit()
        return sku

    def detach_variant(self, sku: str) -> bool:
        """Materialise a variant's merged view into the user overlay.

        After ``detach_variant`` runs, the SKU no longer inherits from its
        parent — every field (joints/bodies/assets/init_pose) is materialised
        from the merged view. Used so the user can delete the parent without
        orphaning the variant.
        """
        if not _robots_registry.is_variant(sku):
            return False
        merged = _robots_registry.get_robot(sku) or {}
        if not merged:
            return False
        flat = {k: v for k, v in merged.items() if k != "inherits_from"}
        ok = _robots_registry.persist_user_robot(sku, flat)
        if not ok:
            return False
        self._reload_and_validate()
        self.changed.emit()
        return True

    def remove_user_robot(self, sku: str) -> bool:
        """Delete a user-layer robot AND its per-asset state.json block.

        Refuses with :class:`~registers.robots.CascadeProtectionError` when
        the SKU has user-layer variants pointing at it; UI catches and
        prompts the user to detach or delete the variants first.
        """
        if not self.is_user_extension(sku):
            log_warning(f"[robot_assets] cannot remove canonical sku={sku}")
            return False
        try:
            ok = _robots_registry.delete_user_robot(sku)
        except _robots_registry.CascadeProtectionError:
            raise
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
# MJCF base spawn-Z offset — calibration task + freshness gate.
# ---------------------------------------------------------------------------
# Active calibration dedupe: prevents the same SKU from being submitted
# twice while the first task is still running (multiple RobotNode.execute()
# invocations during canvas open / param change can otherwise queue
# redundant work).
_active_offset_calibrations: set = set()


class _MjcfBaseCalibrationTask(Task):
    """Background task wrapping
    :func:`application.training.validation.mjcf_base_calibration.calibrate_mjcf_base_offset`.

    Submitted by :func:`ensure_mjcf_base_offset_cached` when the overlay is
    missing or stale. On completion, writes the result into state.json via
    :meth:`RobotAssetService.set_mjcf_base_offset` so subsequent MuJoCo
    spawns pick up the corrected offset automatically.

    Failures (FileNotFoundError, ValueError from missing registry blueprint
    or empty landmark set) are logged loudly per §1.8 — the task does NOT
    write a placeholder zero-offset overlay; absence of an overlay is the
    UI signal "uncalibrated, please address the registry issue".
    """

    def __init__(self, sku: str, reference_init_z: float = 0.40) -> None:
        super().__init__(f"Calibrate MJCF base offset ({sku})")
        self._sku = str(sku)
        self._ref_z = float(reference_init_z)

    def run(self) -> str:
        from application.training.validation.mjcf_base_calibration import (
            calibrate_mjcf_base_offset,
        )
        try:
            result = calibrate_mjcf_base_offset(
                self._sku, reference_init_z=self._ref_z,
            )
        except (FileNotFoundError, ValueError) as exc:
            # §1.8: don't write a fake "calibrated with offset=0" overlay.
            # The next runtime spawn will WARN about uncalibrated state and
            # the UI badge will show "uncalibrated", prompting the user to
            # fix the underlying registry data.
            self.log_warning(
                f"calibration aborted for sku={self._sku!r}: {exc}"
            )
            return "failed"
        finally:
            _active_offset_calibrations.discard(self._sku)
        svc = get_robot_asset_service()
        svc.set_mjcf_base_offset(self._sku, result)
        self.log_info(
            f"calibration committed for sku={self._sku!r}: "
            f"status={result.status} offset_z={result.offset_z:.4f}m "
            f"({len(result.landmark_z)} landmarks)"
        )
        return result.status


def ensure_mjcf_base_offset_cached(sku: str) -> None:
    """Trigger one-shot calibration for ``sku`` if the overlay is missing or stale.

    Called from :meth:`RobotNode.execute` after IR mapping resolves. Cheap
    when the overlay is fresh — only file SHA + registry blueprint hash
    comparisons happen on the calling thread; the actual MuJoCo load runs
    in a background ``Task``.

    Freshness key: ``(mjcf_sha256, standing_pose_sha256)``. If the user
    swaps the MJCF file or the registry blueprint's
    ``default_init_joint_angles`` changes, the next Robot Node refresh
    automatically re-calibrates.
    """
    if not sku:
        return
    if sku in _active_offset_calibrations:
        return  # already queued — skip duplicate submit

    svc = get_robot_asset_service()
    overlay = svc.get_mjcf_base_offset(sku)

    # not_applicable (manipulator, etc.) is a terminal state — the family
    # doesn't have a spawn-z drift problem, so we never re-run.
    if isinstance(overlay, dict) and overlay.get("status") == "not_applicable":
        return

    needs_calibration = True
    if isinstance(overlay, dict) and overlay.get("status") == "calibrated":
        cached = overlay.get("calibration", {}) or {}
        cached_mjcf = str(cached.get("mjcf_sha256", "") or "")
        cached_pose = str(cached.get("standing_pose_sha256", "") or "")
        if cached_mjcf and cached_pose:
            try:
                from application.training.validation.mjcf_base_calibration import (
                    hash_file_sha256,
                    hash_standing_pose,
                )

                asset = svc.resolve(sku)
                mjcf_path = getattr(asset, "mjcf_path", None) if asset else None
                if mjcf_path is None or not Path(mjcf_path).exists():
                    return  # no MJCF on disk → cannot calibrate; runtime WARN will flag this
                cur_mjcf = hash_file_sha256(Path(mjcf_path))
                entry = _robots_registry.get_robot(sku) or {}
                blueprint = entry.get("default_init_joint_angles") or {}
                if not blueprint:
                    # Registry blueprint went missing since last calibration.
                    # Don't re-trigger (would just fail loudly). Let the
                    # next manual recalibrate / registry fix surface it.
                    return
                cur_pose = hash_standing_pose(
                    {str(k): float(v) for k, v in blueprint.items()}
                )
                if cur_mjcf == cached_mjcf and cur_pose == cached_pose:
                    needs_calibration = False
                else:
                    log_info(
                        f"[robot_assets] mjcf_base_offset stale for sku={sku!r} "
                        f"(mjcf_sha matches={cur_mjcf == cached_mjcf}, "
                        f"pose_sha matches={cur_pose == cached_pose}); "
                        f"resubmitting calibration."
                    )
            except Exception as exc:  # noqa: BLE001
                # WHY KEPT (§1.8 (c)): freshness check failure should not
                # block calibration — log and fall through to re-submit.
                # The submitted task itself will fail loudly if there's a
                # real problem (missing MJCF, missing blueprint).
                log_warning(
                    f"[robot_assets] freshness check for sku={sku!r} failed "
                    f"({exc}); resubmitting calibration as a precaution"
                )

    if not needs_calibration:
        return

    try:
        task = _MjcfBaseCalibrationTask(sku)
    except Exception as exc:
        log_warning(
            f"[robot_assets] could not construct calibration task for "
            f"sku={sku!r}: {exc}"
        )
        return
    _active_offset_calibrations.add(sku)
    try:
        mgr = get_tasks_manager()
        mgr.submit(task)
    except Exception as exc:
        _active_offset_calibrations.discard(sku)
        log_warning(
            f"[robot_assets] could not submit calibration task for "
            f"sku={sku!r}: {exc}"
        )


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
