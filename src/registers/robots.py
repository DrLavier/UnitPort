# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""registers.robots — 机器人模板注册表（含 joints + bodies + sensors）/ Canonical robot register.

合并自原 robots/ + sensor_catalog/（plan §11.3 / §11.5）/ Merged from former:
- robots/canonical.json    → data/robots_canonical.json
- robots/__init__.py       → 本文件（含 SKU/alias 折叠 + family 组合查询）
- sensor_catalog/__init__.py → 内联（sensor 是 robot 属性，无独立必要）

Schema 见 plan.md §11.3。每条 robot 以 SKU（``build_sku(brand, model)``）为 key；
``name`` 字段供 UI 显示，**不可作为查询键**。

Per-format schema (post-migration; legacy ``joints`` key rejected by validator):
    "joints_per_format": {
        "MJCF": {uid: {name, ir_role}} | null,
        "USD":  {uid: {name, ir_role}} | null,
        "URDF": {uid: {name, ir_role}} | null
    },
    "bodies_per_format": {
        "MJCF": {uid: {name, ir_role}} | null,
        "USD":  {uid: {name, ir_role}} | null,
        "URDF": {uid: {name, ir_role}} | null
    }

``null`` = "该格式未声明 / 不可用"。每个 robot 至少一个非 null format。

API:
    load() -> int
    get_robot(sku) / list_skus() / get_version()
    resolve_id(user_input) -> str | None        # 大小写/分隔符折叠 → SKU
    list_by_family(*families)                   # superset family 匹配
    list_by_brand(brand)
    list_joints(robot_sku, fmt)                 # 该 format 下的 joint 列表
    get_joints(sku, fmt) / get_bodies(sku, fmt) # raw dict {uid: {name, ir_role}}
    get_joint_order(sku, fmt) / get_joint_ir_roles(sku, fmt)
    get_body_role_map(sku, fmt) -> {name: ir_role}
    list_available_formats(sku) -> List[str]    # 该 robot 已声明的 format
    get_joints_by_ir_role(robot_sku, ir_role, fmt)
    list_sensors(robot_sku)                     # robot.sensors 直读

⚠ 多品牌包容性：禁止在 robot 字段下增加 ``if_brand_is_X`` 分支。
品牌差异通过 ``adapter`` 字段指向 application/service/adapters/<brand>/。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from unitport_sdk import Paths, log_warning, push_data, read_data

from . import RegistryValidationError, resolve_robot_sku


# Canonical asset format keys, in display / preference order.
_FORMATS: tuple = ("MJCF", "USD", "URDF")


# Sentinel ir_role value the IR-role assignment dialog writes when the
# user marks an entry as "(Out of Scope)" — typically sensors / fixtures
# that should be visible in the registry (so the user is reminded they
# decided to skip them) but must NOT participate in IR-role mapping or
# training. Validator rule 3 (in registers/__init__.py) special-cases
# this constant so it doesn't warn "ir_role is not in the IR canonical".
# BodyIRMapper.from_robot_asset can't find it in any family catalog,
# which gives the desired "kinematic-only, ignored at training time"
# behaviour without code changes downstream.
OUT_OF_SCOPE_IR_ROLE: str = "__out_of_scope__"


def _normalize_format(fmt: str) -> str:
    """Normalise a format string to one of the canonical keys.

    Accepts both ``"mjcf"`` and ``"MJCF"`` style. Returns the canonical key
    or raises ``ValueError`` for unknown formats.
    """
    if not fmt:
        raise ValueError("format key is required (one of MJCF/USD/URDF)")
    up = str(fmt).strip().upper()
    if up not in _FORMATS:
        raise ValueError(f"unknown format {fmt!r}; expected one of {_FORMATS}")
    return up


_DATA_DIR = Paths.REGISTERS_DIR / "data"
_CANONICAL_PATH = _DATA_DIR / "robots_canonical.json"
_USER_CUSTOM_REL = "registers/robots_custom.json"


def _user_custom_path() -> Path:
    """Lazy resolve of the user robot overlay path inside the LIVE
    USER_CONFIG_DIR. Resolved at call time so workspace hot-switches are
    picked up without restart."""
    return Paths.USER_CONFIG_DIR / "registers" / "robots_custom.json"

_state: Dict[str, Any] = {
    "loaded": False,
    "version": "",
    "robots": {},          # sku → robot dict (post inheritance merge)
    "raw_robots": {},      # sku → robot dict (pre inheritance merge; used by editor round-trip)
    "alias_to_sku": {},
    "user_extensions": {},
    "variant_skus": set(),  # SKUs whose inherits_from is non-empty
}


# =============================================================================
# Inheritance (model_family / variant_label / inherits_from)
# =============================================================================
#
# Three optional fields layered on top of the flat SKU table:
#   - model_family : groups variants under one UI card; default = entry["model"]
#   - variant_label: chip-display label; default "Standard"
#   - inherits_from: parent SKU; resolver merges parent fields into the variant
#
# Inheritance is resolved ONCE at load (and after every user-overlay merge) into
# _state["robots"]. All existing consumers read from _state["robots"] and get
# the merged view transparently — no call-site refactor needed. The pre-merge
# view is preserved in _state["raw_robots"] for editor round-trip (so saving an
# edit doesn't accidentally write inherited fields back into the overlay).
#
# Hard rules:
#   * No chains: parent.inherits_from MUST be None/missing.
#   * variant.model_family MUST match parent.model_family (validator enforces).
#   * Variant's own non-null fields always win on merge.

# Fields that propagate from parent → variant when the variant doesn't declare them.
_INHERITABLE_FIELDS: tuple = (
    "joints_per_format",
    "bodies_per_format",
    "assets",
    "init_pose_presets",
    "default_actuator_params",
    "default_init_joint_angles",
    "sensors",
    "families",
    "canonical_version",
)


def _has_declared(value: Any) -> bool:
    """True iff a variant-level field counts as 'declared' (so it wins over parent).

    Treats None / missing as not-declared; everything else (including empty
    dicts / lists) is declared. Variants that want to explicitly null out an
    inherited field should set it to ``None`` — which means inherit, not erase.
    """
    if value is None:
        return False
    return True


def _merge_variant_onto_parent(parent: Dict[str, Any], variant: Dict[str, Any]) -> Dict[str, Any]:
    """Return a new dict: copy of ``parent``, with every variant-declared
    inheritable field overridden, plus all variant-own fields layered on top.

    Whole-block override (not per-key deep merge). See plan §1.3.
    """
    merged: Dict[str, Any] = dict(parent)
    # Inheritable fields: variant wins iff declared.
    for key in _INHERITABLE_FIELDS:
        if _has_declared(variant.get(key, None)):
            merged[key] = variant[key]
    # Variant-own fields (always taken from variant): everything else in variant.
    for key, value in variant.items():
        if key in _INHERITABLE_FIELDS:
            continue
        merged[key] = value
    return merged


def _resolve_inheritance_in_place(robots: Dict[str, Dict[str, Any]]) -> None:
    """Walk every entry; for each with non-empty ``inherits_from``, replace
    its dict in-place with the parent-merged view.

    Idempotent: safe to call after every user-overlay merge. Detects chains
    (variant whose parent itself has inherits_from) by surfacing a log_error
    and leaving the variant entry as-is; the cross-registry validator then
    raises RegistryValidationError when run.
    """
    variant_skus: set = set()
    for sku, entry in robots.items():
        parent_sku = entry.get("inherits_from")
        if not parent_sku:
            continue
        variant_skus.add(sku)
        parent_entry = robots.get(parent_sku)
        if parent_entry is None:
            log_warning(
                f"[robots] inherits_from target missing: variant={sku} parent={parent_sku}"
            )
            continue
        if parent_entry.get("inherits_from"):
            log_warning(
                f"[robots] inheritance chain detected: variant={sku} parent={parent_sku} "
                f"is itself a variant — chains are forbidden, validator will raise"
            )
            continue
        merged = _merge_variant_onto_parent(parent_entry, entry)
        robots[sku] = merged
    _state["variant_skus"] = variant_skus


_SEP_RE = re.compile(r"[\s\-_/.]+")


def _strip_separators(s: str) -> str:
    """与 build_sku 的归一化策略一致：剥除所有分隔符."""
    return _SEP_RE.sub("", str(s).strip().lower())


def _alias_keys(brand: str, model: str, name: str, extra: List[str]) -> List[str]:
    raw = [f"{brand}.{model}", model, name, *extra]
    return list(dict.fromkeys(_strip_separators(r) for r in raw if r))


def load() -> int:
    if _state["loaded"]:
        return len(_state["robots"])

    payload = read_data(_CANONICAL_PATH)
    if not isinstance(payload, dict):
        raise RuntimeError(f"robots_canonical.json 解析失败：{_CANONICAL_PATH}")

    _state["version"] = str(payload.get("version", ""))
    raw_robots = payload.get("robots", {})

    robots: Dict[str, Dict[str, Any]] = {}
    alias: Dict[str, str] = {}
    for sku, entry in raw_robots.items():
        if not isinstance(entry, dict) or "brand" not in entry or "model" not in entry:
            log_warning(f"[robots] 跳过非法条目：{sku}")
            continue
        expected = resolve_robot_sku(entry["brand"], entry["model"])
        if expected != sku:
            log_warning(
                f"[robots] SKU 不一致：canonical key={sku} but build_sku({entry['brand']},{entry['model']})={expected}"
            )
        robots[sku] = entry
        for k in _alias_keys(
            entry["brand"],
            entry["model"],
            entry.get("name", ""),
            list(entry.get("aliases", []) or []),
        ):
            alias.setdefault(k, sku)

    # Preserve pre-merge view for editor round-trip; resolve inheritance into
    # the public ``robots`` map so every existing consumer sees merged fields.
    _state["raw_robots"] = {k: dict(v) for k, v in robots.items()}
    _resolve_inheritance_in_place(robots)
    _state["robots"] = robots
    _state["alias_to_sku"] = alias
    _state["loaded"] = True
    return len(robots)


def _autoreshape_legacy_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Lenient in-memory upgrade for user-overlay entries still using the
    legacy flat ``joints`` field. The canonical (factory) file is migrated
    once via ``bootstrap/migrate_canonical_to_per_format.py`` and the
    validator rejects flat ``joints`` there. User overlays may pre-date the
    migration, so we silently reshape them as MJCF so user data is not lost.

    Reshape rules:
        - ``entry["joints"]`` → ``entry["joints_per_format"]["MJCF"]``
        - leaves USD / URDF as null
        - bodies_per_format stays empty (no auto-discovery here; UI Dump
          MJCF / Dump USD buttons can fill later)
        - Does NOT touch entries already in per-format shape.
    """
    if not isinstance(entry, dict):
        return entry
    if "joints_per_format" in entry:
        return entry
    legacy = entry.get("joints")
    if not isinstance(legacy, dict):
        return entry
    out = dict(entry)
    out.pop("joints", None)
    out["joints_per_format"] = {
        "MJCF": dict(legacy),
        "USD": None,
        "URDF": None,
    }
    out.setdefault("bodies_per_format", {"MJCF": None, "USD": None, "URDF": None})
    return out


def merge_user_extensions(extensions: Dict[str, Dict[str, Any]]) -> int:
    added = 0
    for sku, entry in (extensions or {}).items():
        if not isinstance(entry, dict):
            log_warning(f"[robots] 跳过非法 robots_custom 条目：{sku}")
            continue
        entry = _autoreshape_legacy_entry(entry)
        _state["user_extensions"][sku] = entry
        # SHALLOW field-level merge: user-overlay keys override canonical;
        # canonical fields NOT declared in the overlay are preserved. This
        # lets future releases add new RobotSpec fields (e.g.
        # default_init_joint_angles in Stage A of the sim2sim PD framework)
        # without users having to re-export every overlay entry to pick up
        # the new field. Editor round-trip still writes the full entry
        # back so explicit edits are durable; the merge just stops the
        # overlay from shadowing canonical fields the user never touched.
        base_canonical = _state["raw_robots"].get(sku, {}) if sku in _state["raw_robots"] else {}
        merged = dict(base_canonical)
        merged.update(entry)
        _state["raw_robots"][sku] = merged
        _state["robots"][sku] = dict(merged)
        added += 1
    _resolve_inheritance_in_place(_state["robots"])
    return added


def persist_user_robot(sku: str, entry: Dict[str, Any]) -> bool:
    """写入/更新 user 层 robot 至 <USER_CONFIG_DIR>/registers/robots_custom.json.

    Shape: ``{"robots": {sku: entry, ...}}`` — 与 RegistryHub._merge_user_overlays 完全兼容。
    本函数只负责落盘；调用方完成多个写入后应自行 RegistryHub.reload()。
    """
    payload: Dict[str, Any] = {}
    if _user_custom_path().exists():
        existing = read_data(_user_custom_path())
        if isinstance(existing, dict):
            payload = dict(existing)
    robots_blk = dict(payload.get("robots", {}) or {})
    robots_blk[sku] = entry
    payload["robots"] = robots_blk
    return bool(push_data(_USER_CUSTOM_REL, payload))


class CascadeProtectionError(RuntimeError):
    """Raised when deleting a robot would orphan user-layer variants."""


def list_user_variants_of(parent_sku: str) -> List[str]:
    """Return SKUs in the user-overlay whose ``inherits_from`` points at the
    given parent. Used for cascade-delete protection.
    """
    if not _state["loaded"]:
        load()
    out: List[str] = []
    for sku, entry in (_state.get("user_extensions") or {}).items():
        if isinstance(entry, dict) and entry.get("inherits_from") == parent_sku:
            out.append(sku)
    return out


def delete_user_robot(sku: str, *, force: bool = False) -> bool:
    """Remove the SKU entry from ``<USER_CONFIG_DIR>/registers/robots_custom.json``.

    Refuses to delete a robot that still has user-layer variants pointing
    at it via ``inherits_from`` — unless ``force=True``. The variants would
    otherwise become orphans (parent missing) and break validation.
    """
    if not force:
        children = list_user_variants_of(sku)
        if children:
            raise CascadeProtectionError(
                f"robot {sku} has user variants {children}; "
                f"detach or delete those first"
            )
    if not _user_custom_path().exists():
        return False
    existing = read_data(_user_custom_path())
    if not isinstance(existing, dict):
        return False
    payload = dict(existing)
    robots_blk = dict(payload.get("robots", {}) or {})
    if sku not in robots_blk:
        return False
    robots_blk.pop(sku, None)
    payload["robots"] = robots_blk
    return bool(push_data(_USER_CUSTOM_REL, payload))


def is_user_extension(sku: str) -> bool:
    """canonical(出厂) vs user(运行时新增) 的判别。"""
    return sku in (_state.get("user_extensions") or {})


def get_version() -> str:
    return _state["version"]


def list_skus() -> List[str]:
    if not _state["loaded"]:
        load()
    return list(_state["robots"].keys())


def get_robot(sku: str) -> Optional[Dict[str, Any]]:
    if not _state["loaded"]:
        load()
    return _state["robots"].get(sku)


def get_raw_robot(sku: str) -> Optional[Dict[str, Any]]:
    """Return the pre-inheritance view of the entry (variant fields only).

    Use this in the RobotEditorDialog so saving an edit doesn't accidentally
    write inherited fields back into the user overlay and break inheritance.
    All other callers should keep using :func:`get_robot` for the merged view.
    """
    if not _state["loaded"]:
        load()
    raw = _state.get("raw_robots") or {}
    entry = raw.get(sku)
    return dict(entry) if isinstance(entry, dict) else None


def get_parent_sku(sku: str) -> Optional[str]:
    """Return the unmerged ``inherits_from`` value, or None for non-variants."""
    raw = get_raw_robot(sku) or {}
    parent = raw.get("inherits_from")
    return str(parent) if parent else None


def is_variant(sku: str) -> bool:
    """True iff this SKU was loaded with a non-empty ``inherits_from``."""
    return bool(get_parent_sku(sku))


def get_variant_label(sku: str) -> str:
    """Return entry.variant_label, defaulting to ``"Standard"``."""
    entry = get_robot(sku) or {}
    label = entry.get("variant_label")
    return str(label) if label else "Standard"


def get_model_family(sku: str) -> str:
    """Return entry.model_family, defaulting to entry.model.

    For flat (non-variant) robots, this means each robot is its own one-member
    family — UI grouping by ``(brand, model_family)`` degrades gracefully to a
    single-card layout.
    """
    entry = get_robot(sku) or {}
    fam = entry.get("model_family")
    if fam:
        return str(fam)
    return str(entry.get("model", "") or "")


def list_model_families() -> List[tuple]:
    """Return a sorted, deduped list of ``(brand, model_family)`` tuples."""
    if not _state["loaded"]:
        load()
    out = set()
    for entry in _state["robots"].values():
        brand = str(entry.get("brand", "") or "").strip().lower()
        fam = entry.get("model_family") or entry.get("model") or ""
        fam = str(fam).strip().lower()
        if brand and fam:
            out.add((brand, fam))
    return sorted(out)


def list_variants(brand: str, model_family: str) -> List[str]:
    """Return SKUs for ``(brand, model_family)`` — base first, then variants
    sorted by variant_label.

    "Base" = the entry with no ``inherits_from``. If multiple bases exist for
    the same family (shouldn't happen in practice; validator warns), they are
    all returned first in registry order.
    """
    if not _state["loaded"]:
        load()
    brand_l = str(brand or "").strip().lower()
    fam_l = str(model_family or "").strip().lower()
    bases: List[str] = []
    variants: List[tuple] = []
    for sku, entry in _state["robots"].items():
        b = str(entry.get("brand", "") or "").strip().lower()
        f = str(entry.get("model_family") or entry.get("model") or "").strip().lower()
        if b != brand_l or f != fam_l:
            continue
        parent = (_state.get("raw_robots", {}) or {}).get(sku, {}).get("inherits_from")
        if parent:
            variants.append((str(entry.get("variant_label", "") or ""), sku))
        else:
            bases.append(sku)
    variants.sort(key=lambda kv: (kv[0].lower(), kv[1]))
    return bases + [sku for _, sku in variants]


def resolve_id(user_input: str) -> Optional[str]:
    """大小写/分隔符折叠 + 别名表查找；返回 SKU 或 None.

    Examples:
        resolve_id("Mini Pupper") == resolve_id("mini-pupper") == resolve_id("MINI_PUPPER")
        resolve_id("unitree.go2w") == resolve_id("Unitree Go2-W")
    """
    if not _state["loaded"]:
        load()
    if not user_input:
        return None
    raw = str(user_input).strip().lower()
    folded = _strip_separators(raw)
    if folded in _state["alias_to_sku"]:
        return _state["alias_to_sku"][folded]
    if raw in _state["robots"]:
        return raw
    return None


def resolve_to_sku(user_input: str) -> Optional[str]:
    """Three-step lookup: literal SKU → alias/name → ``brand.model``.

    The single resolution helper for everywhere the codebase converts a
    user-facing ``asset_id`` string (as stored on ``RobotNode.params``) into
    the canonical SKU stored in artifacts / canvas top-level ``robot_id``.

    Accepts:
        * Canonical SKU      — ``"a3f9c1d2..."``
        * Alias / name       — ``"Unitree Go2"``, ``"go2"``, ``"unitree.go2"``
        * ``brand.model``    — ``"unitree.go2w"`` (rebuilt via ``build_sku``)

    Returns the canonical SKU, or ``None`` if the input cannot be matched
    in the loaded registry. Always validates the third-step result against
    the registry so we never return a hash for an unregistered robot.
    """
    if not _state["loaded"]:
        load()
    raw = str(user_input or "").strip()
    if not raw:
        return None
    # Step 1: literal SKU?
    if raw in _state["robots"]:
        return raw
    # Step 2: alias / display name (case / separator-insensitive)
    sku = resolve_id(raw)
    if sku is not None:
        return sku
    # Step 3: ``brand.model`` form → build_sku, then verify it exists
    if "." in raw:
        from . import resolve_robot_sku
        brand, _, model = raw.partition(".")
        brand = brand.strip()
        model = model.strip()
        if brand and model:
            candidate = resolve_robot_sku(brand, model)
            if candidate in _state["robots"]:
                return candidate
    return None


def list_by_family(*families: str) -> List[Dict[str, Any]]:
    """返回包含全部给定 family 的 robots（superset 匹配）。"""
    if not _state["loaded"]:
        load()
    want = set(families)
    out = []
    for entry in _state["robots"].values():
        fams = set(entry.get("families", []) or [])
        if want.issubset(fams):
            out.append(entry)
    return out


def list_by_brand(brand: str) -> List[Dict[str, Any]]:
    if not _state["loaded"]:
        load()
    target = str(brand or "").strip().lower()
    return [e for e in _state["robots"].values() if e.get("brand") == target]


def list_available_formats(robot_sku: str) -> List[str]:
    """Return the asset formats for which this robot has a joints/bodies dict
    declared (not null). Ordering follows the canonical _FORMATS tuple.
    """
    entry = get_robot(robot_sku)
    if entry is None:
        return []
    pf = entry.get("joints_per_format", {}) or {}
    return [f for f in _FORMATS if isinstance(pf.get(f), dict)]


def get_joints(robot_sku: str, fmt: str) -> Dict[str, Dict[str, Any]]:
    """Return ``{uid: {name, ir_role}}`` for the given format, or ``{}`` if
    the format is not declared for this robot."""
    fmt_key = _normalize_format(fmt)
    entry = get_robot(robot_sku)
    if entry is None:
        return {}
    pf = entry.get("joints_per_format", {}) or {}
    block = pf.get(fmt_key)
    if not isinstance(block, dict):
        return {}
    return dict(block)


def get_bodies(robot_sku: str, fmt: str) -> Dict[str, Dict[str, Any]]:
    """Return ``{uid: {name, ir_role}}`` for the given format, or ``{}`` if
    the format is not declared for this robot."""
    fmt_key = _normalize_format(fmt)
    entry = get_robot(robot_sku)
    if entry is None:
        return {}
    pf = entry.get("bodies_per_format", {}) or {}
    block = pf.get(fmt_key)
    if not isinstance(block, dict):
        return {}
    return dict(block)


def get_joint_order(robot_sku: str, fmt: str) -> List[str]:
    """Joint names in the format's natural (dict insertion) order."""
    out: List[str] = []
    for jspec in get_joints(robot_sku, fmt).values():
        if isinstance(jspec, dict):
            nm = str(jspec.get("name", ""))
            if nm:
                out.append(nm)
    return out


def get_joint_ir_roles(robot_sku: str, fmt: str) -> List[str]:
    """IR roles parallel to ``get_joint_order(sku, fmt)``."""
    out: List[str] = []
    for jspec in get_joints(robot_sku, fmt).values():
        if isinstance(jspec, dict):
            nm = str(jspec.get("name", ""))
            ir = str(jspec.get("ir_role", ""))
            if nm:
                out.append(ir)
    return out


def get_body_role_map(robot_sku: str, fmt: str) -> Dict[str, str]:
    """Return ``{body_name: ir_role}`` for the given format."""
    out: Dict[str, str] = {}
    for bspec in get_bodies(robot_sku, fmt).values():
        if isinstance(bspec, dict):
            nm = str(bspec.get("name", ""))
            ir = str(bspec.get("ir_role", ""))
            if nm and ir:
                out[nm] = ir
    return out


def list_joints(robot_sku: str, fmt: str) -> List[Dict[str, Any]]:
    """Backwards-compat list view; equivalent to ``list(get_joints(sku, fmt).values())``."""
    return list(get_joints(robot_sku, fmt).values())


def get_joints_by_ir_role(
    robot_sku: str, ir_role: str, fmt: str
) -> Optional[Dict[str, Any]]:
    for j in list_joints(robot_sku, fmt):
        if j.get("ir_role") == ir_role:
            return j
    return None


def get_robot_init_pose_presets(robot_sku: str) -> List[Dict[str, Any]]:
    """Return factory init pose presets shipped with the canonical robot entry.

    Each preset is a raw dict with shape:
        {"name": str, "description": str,
         "base_pos": [x, y, z],
         "joint_pos_by_ir": {ir_role: angle_rad, ...}}

    Returns an empty list when the SKU is unknown OR has no init pose
    presets declared in robots_canonical.json. User-defined presets live
    in ``<USER_CONFIG_DIR>/robot_presets/init_poses.json`` and are merged by
    :class:`application.service.robot_init_poses.service.InitPoseService`.
    """
    entry = get_robot(robot_sku)
    if entry is None:
        return []
    presets = entry.get("init_pose_presets")
    if not isinstance(presets, list):
        return []
    return list(presets)


# ---------------------------------------------------------------------------
# Typed RobotSpec accessor — per-format aware (Stage 1 upgrade)
#
# Each robot now carries body/joint→IR-role tables for every asset format it
# declares (MJCF / USD / URDF). The runtime selects the right table by
# ``active_format`` — wired through ``RobotSpecRef.active_format`` in Stage 2
# and the BodyIRMapper / JointIRResolver factories in Stage 3.
#
# Legacy fields ``joint_order`` / ``joint_ir_roles`` / ``body_role_map`` are
# kept as DERIVED views (resolved via ``preferred_format`` — MJCF first when
# declared, else first non-null format) so existing call-sites keep working
# during the staged rollout. New code should call the per-format accessors
# (``joints_for / bodies_for / joint_order_for / joint_ir_roles_for``).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RobotSpec:
    """Snapshot of one canonical robot, typed for training-side consumers."""

    sku: str
    name: str
    brand: str
    model: str
    families: List[str]
    # Per-format raw tables: {format: {uid: {name, ir_role}}}. Format values
    # are EITHER a dict (declared) OR missing/empty (not declared for this
    # robot). Use ``available_formats`` for the declared set.
    joints_per_format: Dict[str, Dict[str, Dict[str, Any]]]
    bodies_per_format: Dict[str, Dict[str, Dict[str, Any]]]
    mjcf_path: Optional[str]
    urdf_path: Optional[str]
    usd_path: Optional[str]
    # Cloud-only USD asset (Nucleus URL marker for IsaacLab built-in USD library).
    # Distinct from ``usd_path`` because pathlib mis-parses ``scheme://`` on Windows.
    usd_url: Optional[str]
    capabilities: Dict[str, Any]
    sensors: List[Dict[str, Any]]
    adapter: str
    sdk_paths: List[str]
    # SKU-specific PD / effort / velocity defaults the canvas ActorSetting
    # node reconciles to when this robot is selected (canvas-derived-keys
    # rule). Without this, every robot defaulted to Go2-shaped Kp=25 /
    # Kd=0.5 / effort=30 → training a 42 kg Spot with Go2-class PD made
    # legs collapse under self-weight and the policy never learned a
    # stable stance ("ragdoll feel + bouncing on contact"). Keys are the
    # canonical IsaacLab ImplicitActuator field names; missing keys
    # inherit whatever defaults the canvas node ParamSpec declares.
    default_actuator_params: Optional[Dict[str, float]] = None
    # SKU-specific standing-pose blueprint (IR role → joint angle in
    # radians). Sourced by the ActorSetting canvas reconcile when the
    # user switches SKU and the canvas needs to seed new IR-role keys
    # the previous robot didn't expose. Without this, the reconcile
    # filled new keys with 0.0 — which silently corrupted bundles for
    # any robot whose joint range doesn't include 0 (Spot's knee bends
    # only backward in [-2.79, -0.25]; 0.0 is out of range). Keys are
    # IR-role names from this robot's primary IR family; missing IR
    # roles fall back to 0.0 (legacy behaviour) and the env_cfg compiler
    # joint-range validator catches the resulting out-of-range pose with
    # an actionable error pointing back at this field.
    default_init_joint_angles: Optional[Dict[str, float]] = None

    # --- per-format accessors -----------------------------------------------

    @property
    def available_formats(self) -> List[str]:
        """Asset formats with a non-empty joints OR bodies declaration."""
        out: List[str] = []
        for fmt in _FORMATS:
            j = self.joints_per_format.get(fmt)
            b = self.bodies_per_format.get(fmt)
            if isinstance(j, dict) or isinstance(b, dict):
                out.append(fmt)
        return out

    @property
    def preferred_format(self) -> str:
        """The default format to fall back to when no active_format is set.

        MJCF first (most robots have it), then USD, then URDF. Used by the
        legacy ``joint_order`` / ``joint_ir_roles`` / ``body_role_map``
        derived properties so old call-sites keep working.
        """
        for fmt in _FORMATS:
            if isinstance(self.joints_per_format.get(fmt), dict):
                return fmt
        for fmt in _FORMATS:
            if isinstance(self.bodies_per_format.get(fmt), dict):
                return fmt
        return _FORMATS[0]  # MJCF as last-resort sentinel

    def joints_for(self, fmt: str) -> Dict[str, Dict[str, Any]]:
        block = self.joints_per_format.get(_normalize_format(fmt))
        return dict(block) if isinstance(block, dict) else {}

    def bodies_for(self, fmt: str) -> Dict[str, Dict[str, Any]]:
        block = self.bodies_per_format.get(_normalize_format(fmt))
        return dict(block) if isinstance(block, dict) else {}

    def joint_order_for(self, fmt: str) -> List[str]:
        out: List[str] = []
        for jspec in self.joints_for(fmt).values():
            nm = str(jspec.get("name", ""))
            if nm:
                out.append(nm)
        return out

    def joint_ir_roles_for(self, fmt: str) -> List[str]:
        out: List[str] = []
        for jspec in self.joints_for(fmt).values():
            nm = str(jspec.get("name", ""))
            if nm:
                out.append(str(jspec.get("ir_role", "")))
        return out

    def joints_role_map_for(self, fmt: str) -> Dict[str, str]:
        """Return ``{joint_name: ir_role}`` for the given format.

        Despite the legacy ``body_role_map`` property's misleading name,
        the historical contract has been **joint name → ir_role** (see
        ``model_registry.get_ir_role_to_joint`` / ``IsaacLabConfig
        .body_mapping_json``). New code should call this explicitly when
        it needs joint mapping, and ``bodies_role_map_for`` when it
        truly needs body mapping.
        """
        out: Dict[str, str] = {}
        for jspec in self.joints_for(fmt).values():
            nm = str(jspec.get("name", ""))
            ir = str(jspec.get("ir_role", ""))
            if nm and ir:
                out[nm] = ir
        return out

    def bodies_role_map_for(self, fmt: str) -> Dict[str, str]:
        """Return ``{body_name: ir_role}`` for the given format — the real
        body mapping (as opposed to the misnamed legacy ``body_role_map``).
        """
        out: Dict[str, str] = {}
        for bspec in self.bodies_for(fmt).values():
            nm = str(bspec.get("name", ""))
            ir = str(bspec.get("ir_role", ""))
            if nm and ir:
                out[nm] = ir
        return out

    # --- legacy derived views (Stage 1 backwards-compat) --------------------
    # Old call-sites referencing ``.joint_order`` / ``.joint_ir_roles`` /
    # ``.body_role_map`` / ``.num_joints`` keep working; they resolve via
    # ``preferred_format``. Stage 2-3 migrates these call-sites to the
    # format-aware variants and these properties become deletion candidates.
    #
    # NOTE: ``body_role_map`` is named for historical reasons — its actual
    # contract is JOINT name → ir_role (see joints_role_map_for above).

    @property
    def joint_order(self) -> List[str]:
        return self.joint_order_for(self.preferred_format)

    @property
    def joint_ir_roles(self) -> List[str]:
        return self.joint_ir_roles_for(self.preferred_format)

    @property
    def body_role_map(self) -> Dict[str, str]:
        # Legacy contract: joint name → ir_role (historically misnamed).
        return self.joints_role_map_for(self.preferred_format)

    @property
    def num_joints(self) -> int:
        return len(self.joint_order)

    @property
    def action_dim(self) -> int:
        """Action dim defaults to joint count; overridable via capabilities."""
        override = self.capabilities.get("action_dim") if self.capabilities else None
        if isinstance(override, int) and override > 0:
            return override
        return self.num_joints


def get_robot_spec(robot_id: str) -> Optional[RobotSpec]:
    """Resolve ``robot_id`` (SKU, alias, or model name) to a :class:`RobotSpec`.

    Returns ``None`` if the id cannot be resolved. Callers must surface this to
    the UI rather than guessing — see Multi-Brand Inclusiveness red line.
    """
    sku = resolve_id(robot_id) or robot_id
    entry = get_robot(sku)
    if entry is None:
        return None

    # Per-format tables: read raw dicts, normalise null/missing → {} so the
    # ``available_formats`` accessor reports a clean set.
    raw_joints_pf = entry.get("joints_per_format", {}) or {}
    raw_bodies_pf = entry.get("bodies_per_format", {}) or {}
    joints_per_format: Dict[str, Dict[str, Dict[str, Any]]] = {}
    bodies_per_format: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for fmt in _FORMATS:
        j_block = raw_joints_pf.get(fmt)
        if isinstance(j_block, dict):
            joints_per_format[fmt] = dict(j_block)
        b_block = raw_bodies_pf.get(fmt)
        if isinstance(b_block, dict):
            bodies_per_format[fmt] = dict(b_block)

    assets = entry.get("assets", {}) or {}

    # Legacy ``isaac_lab_joint_order`` field is intentionally not read.
    # ``joints_per_format["USD"]`` insertion order is the sole source of
    # truth for Isaac Lab's USD-articulation joint ordering. The hand-
    # authored field used to mirror that order, but drifted from the
    # dumped USD table whenever both were present and silently shipped
    # bundles with the wrong joint permutation. Any value still carried
    # in factory data or user overlay entries is silently ignored at
    # load time — there is no migration warning because the field has
    # no consumer left.

    dap_raw = entry.get("default_actuator_params")
    default_actuator_params: Optional[Dict[str, float]]
    if isinstance(dap_raw, dict) and dap_raw:
        default_actuator_params = {}
        for k, v in dap_raw.items():
            try:
                default_actuator_params[str(k)] = float(v)
            except (TypeError, ValueError):
                log_warning(
                    f"[robots] sku={sku!r} default_actuator_params[{k!r}]="
                    f"{v!r} is not a number; dropped."
                )
    else:
        default_actuator_params = None

    dija_raw = entry.get("default_init_joint_angles")
    default_init_joint_angles: Optional[Dict[str, float]]
    if isinstance(dija_raw, dict) and dija_raw:
        default_init_joint_angles = {}
        for k, v in dija_raw.items():
            try:
                default_init_joint_angles[str(k)] = float(v)
            except (TypeError, ValueError):
                log_warning(
                    f"[robots] sku={sku!r} default_init_joint_angles[{k!r}]="
                    f"{v!r} is not a number; dropped."
                )
    else:
        default_init_joint_angles = None

    return RobotSpec(
        sku=str(entry.get("sku", sku)),
        name=str(entry.get("name", "")),
        brand=str(entry.get("brand", "")),
        model=str(entry.get("model", "")),
        families=list(entry.get("families", []) or []),
        joints_per_format=joints_per_format,
        bodies_per_format=bodies_per_format,
        mjcf_path=assets.get("MJCF") or None,
        urdf_path=assets.get("URDF") or None,
        usd_path=assets.get("USD") or None,
        usd_url=assets.get("USD_URL") or None,
        capabilities=dict(entry.get("capabilities", {}) or {}),
        sensors=list_sensors(sku),
        adapter=str(entry.get("adapter", "")),
        sdk_paths=list(entry.get("sdk_paths", []) or []),
        default_actuator_params=default_actuator_params,
        default_init_joint_angles=default_init_joint_angles,
    )


# ---------------------------------------------------------------------------
# Sensor catalog（合并自原 sensor_catalog/__init__.py）
# 当前传感器声明 inline 在 robot.sensors；待 sim2real 流程铺设时再考虑独立词汇表。
# ---------------------------------------------------------------------------

def list_sensors(robot_sku: str) -> List[Dict[str, Any]]:
    """返回 robot 的传感器列表 / Sensors attached to the given robot."""
    entry = get_robot(robot_sku)
    if entry is None:
        return []
    sensors = entry.get("sensors", {}) or {}
    out: List[Dict[str, Any]] = []
    for name, spec in sensors.items():
        if spec is None:
            continue
        out.append({"name": name, **(spec if isinstance(spec, dict) else {})})
    return out


__all__ = [
    "load",
    "merge_user_extensions",
    "persist_user_robot",
    "delete_user_robot",
    "list_user_variants_of",
    "CascadeProtectionError",
    "is_user_extension",
    "get_version",
    "list_skus",
    "get_robot",
    "get_raw_robot",
    "resolve_id",
    "resolve_to_sku",
    "list_by_family",
    "list_by_brand",
    # Variant hierarchy
    "list_model_families",
    "list_variants",
    "get_parent_sku",
    "is_variant",
    "get_variant_label",
    "get_model_family",
    # Per-format accessors
    "list_available_formats",
    "get_joints",
    "get_bodies",
    "get_joint_order",
    "get_joint_ir_roles",
    "get_body_role_map",
    "list_joints",
    "get_joints_by_ir_role",
    "get_robot_init_pose_presets",
    "list_sensors",
    "RobotSpec",
    "get_robot_spec",
]
