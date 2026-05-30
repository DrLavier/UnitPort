# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""registers.ir — IR 角色 / 意图 / 动作 / 编译器 schema 规范源.

合并自原 ir_catalog/ 子目录（plan §11.4）/ Merged from former ir_catalog/ subdir:
- canonical.json       → data/ir_canonical.json
- intent_vocab.json    → data/ir_intent_vocab.json
- ir_actions.py        → 内联进本文件
- compiler_schemas/    → data/compiler_schemas/（保留子目录：每节点一文件）

⚠ 多品牌包容性铁则：``ir_canonical.json`` 永远手编，禁止从 URDF/USDA 自动扩展。
未识别 body 名以 ``unmapped_bodies`` 形式上抛 UI。

API:
    load() -> int
    get_role(family, role_id) / list_roles(family)
    list_roles_for_families(*families, include_user_extensions=False)
        # Go2w=quadruped+wheeled、Spot+Arm=quadruped+manipulator
    is_known_role(family, role_id)
    list_families() / get_version()
    get_intent(intent_id) / list_intents()
    get_action(intent_id) / list_actions()      # ir_actions（阶段 C 填）
    merge_user_extensions(extensions)            # 来自 <USER_CONFIG_DIR>/registers/ir_custom.json
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pathlib import Path

from unitport_sdk import Paths, log_warning, push_data, read_data


_DATA_DIR = Paths.REGISTERS_DIR / "data"
_CANONICAL_PATH = _DATA_DIR / "ir_canonical.json"
_INTENTS_PATH = _DATA_DIR / "ir_intent_vocab.json"
_USER_CUSTOM_REL = "registers/ir_custom.json"


def _user_custom_path() -> Path:
    """Lazy resolve of the user IR overlay path inside the LIVE USER_CONFIG_DIR.

    Resolved at call time so workspace hot-switches are picked up without
    restart (mirrors :func:`registers.robots._user_custom_path`).
    """
    return Paths.USER_CONFIG_DIR / "registers" / "ir_custom.json"


_state: Dict[str, Any] = {
    "loaded": False,
    "version": "",
    "families": {},          # family → list[role dict]
    "user_extensions": [],
    # User-level family-level overlay (set of fam_id added through
    # ``merge_user_family_extensions``). The merged role list lives in
    # ``families`` alongside canonical entries; ``user_family_specs`` keeps
    # the on-disk spec (alias_of + explicit roles) for round-trip editing.
    "user_families": set(),
    "user_family_specs": {},
    "intents": {},
    "actions": {},           # intent_id → action descriptor (阶段 C 填)
}


class CascadeProtectionError(RuntimeError):
    """Raised when deleting a user-overlay family would orphan robots that
    still reference it via :attr:`RobotSpec.families`."""


def _resolve_alias(family: str, raw: Dict[str, Any]) -> str:
    """Follow ``alias_of`` chain to a concrete family name."""
    seen = set()
    cur = family
    while cur not in seen:
        seen.add(cur)
        spec = raw.get(cur, {})
        target = spec.get("alias_of")
        if not target:
            return cur
        cur = target
    return cur


def load() -> int:
    """加载 ir_canonical.json + ir_intent_vocab.json；返回总角色数。"""
    if _state["loaded"]:
        return sum(len(rs) for rs in _state["families"].values())

    payload = read_data(_CANONICAL_PATH)
    if not isinstance(payload, dict):
        raise RuntimeError(f"ir_canonical.json 解析失败：{_CANONICAL_PATH}")

    _state["version"] = str(payload.get("version", ""))
    raw_families = payload.get("families", {})

    resolved: Dict[str, List[Dict[str, Any]]] = {}
    for fam_id in raw_families:
        target = _resolve_alias(fam_id, raw_families)
        roles = raw_families.get(target, {}).get("roles", [])
        resolved[fam_id] = [r for r in roles if isinstance(r, dict) and "id" in r]
    _state["families"] = resolved

    intents_payload = read_data(_INTENTS_PATH)
    intents: Dict[str, Dict[str, Any]] = {}
    if isinstance(intents_payload, dict):
        for entry in intents_payload.get("intents", []):
            if isinstance(entry, dict) and "id" in entry:
                intents[entry["id"]] = entry
    _state["intents"] = intents

    # ir_actions：阶段 C 填入；当前空集
    _state["actions"] = {}

    # RegistryHub.reload() flips _state["loaded"]=False; on the next load()
    # we rebuild families from the canonical JSON, so any user-family entry
    # from a prior session must be cleared here too — otherwise the
    # subsequent merge_user_family_extensions call would double-stack on
    # the stale roles. user_extensions (role-level) is cleared by reload().
    _state["user_families"] = set()
    _state["user_family_specs"] = {}

    _state["loaded"] = True
    return sum(len(rs) for rs in resolved.values())


def merge_user_extensions(extensions: List[Dict[str, Any]]) -> int:
    """合入用户级 ir_custom 拓展 / Merge user-level IR role extensions.

    每条拓展 dict 需含 ``id`` ``category`` ``axis``；可选 ``required`` ``family``。
    返回成功合入数。
    """
    added = 0
    for ext in extensions or []:
        if not isinstance(ext, dict) or "id" not in ext:
            log_warning(f"[ir] 跳过非法 ir_custom 条目：{ext}")
            continue
        _state["user_extensions"].append(ext)
        added += 1
    return added


def merge_user_family_extensions(families: Dict[str, Dict[str, Any]]) -> int:
    """Merge a ``{fam_id: spec}`` block of user families into the canonical map.

    Each spec may carry:
        * ``label``    — display name (informational; consumers read it via
          :func:`get_family_label`).
        * ``alias_of`` — existing family id whose roles are inherited. The
          base is resolved at merge time so a deletion of the base after
          merge does not retroactively empty this family.
        * ``roles``    — list of role dicts layered on top of the inherited
          set. Dedupe by ``id``; explicit overrides inherited.

    Returns the number of families successfully merged. Families with both
    an empty alias and empty roles are still registered (so the user can
    iteratively edit), but :func:`RegistryHub.validate` will refuse robots
    that point at an empty family.
    """
    if not _state["loaded"]:
        load()
    added = 0
    for raw_id, spec in (families or {}).items():
        if not isinstance(spec, dict):
            log_warning(f"[ir] skip invalid user family entry: {raw_id!r}")
            continue
        fid = str(raw_id).strip().lower()
        if not fid:
            log_warning("[ir] skip user family with empty id")
            continue
        alias = str(spec.get("alias_of") or "").strip().lower()
        base_roles: List[Dict[str, Any]] = []
        if alias:
            base_roles = list(_state["families"].get(alias, []))
            if not base_roles:
                log_warning(
                    f"[ir] user family {fid!r} alias_of {alias!r} but base "
                    f"family has no roles loaded; merged result will only "
                    f"contain explicit roles"
                )
        explicit_roles = [
            r for r in (spec.get("roles") or [])
            if isinstance(r, dict) and r.get("id")
        ]
        merged: Dict[str, Dict[str, Any]] = {r["id"]: dict(r) for r in base_roles}
        for r in explicit_roles:
            merged[r["id"]] = dict(r)
        _state["families"][fid] = list(merged.values())
        _state["user_families"].add(fid)
        _state["user_family_specs"][fid] = dict(spec)
        added += 1
    return added


def get_family_label(fam_id: str) -> str:
    """Display label for ``fam_id``; falls back to the id itself.

    User families carry an explicit ``label`` in their on-disk spec;
    canonical families have no label and are surfaced by id.
    """
    if not _state["loaded"]:
        load()
    fid = str(fam_id or "").strip().lower()
    spec = _state["user_family_specs"].get(fid)
    if isinstance(spec, dict):
        lab = str(spec.get("label") or "").strip()
        if lab:
            return lab
    return fid


def is_user_family(fam_id: str) -> bool:
    """True iff ``fam_id`` was merged from ``ir_custom.json`` (user overlay)."""
    if not _state["loaded"]:
        load()
    return str(fam_id or "").strip().lower() in _state["user_families"]


def list_user_families() -> List[str]:
    """List of family ids registered through the user overlay."""
    if not _state["loaded"]:
        load()
    return sorted(_state["user_families"])


def get_user_family_spec(fam_id: str) -> Optional[Dict[str, Any]]:
    """Return the raw on-disk spec for a user family (label/alias_of/roles),
    or ``None`` if the family is not user-defined."""
    if not _state["loaded"]:
        load()
    fid = str(fam_id or "").strip().lower()
    spec = _state["user_family_specs"].get(fid)
    return dict(spec) if isinstance(spec, dict) else None


def find_robots_using_family(fam_id: str) -> List[str]:
    """SKUs whose ``families`` list contains ``fam_id``. Used for cascade
    protection on :func:`delete_user_family`."""
    fid = str(fam_id or "").strip().lower()
    if not fid:
        return []
    # Lazy import — avoid circular at module load (robots → ir for validate).
    from . import robots as _robots
    out: List[str] = []
    for sku in _robots.list_skus():
        entry = _robots.get_robot(sku) or {}
        fams = [str(f).strip().lower() for f in (entry.get("families") or [])]
        if fid in fams:
            out.append(sku)
    return out


def persist_user_family(fam_id: str, spec: Dict[str, Any]) -> bool:
    """Write/update a user-overlay family entry in ``ir_custom.json``.

    Shape: ``{"families": {fam_id: spec, ...}, "extensions": [...]}`` —
    compatible with :meth:`RegistryHub._merge_user_overlays`. Pre-existing
    keys (including role-level ``extensions``) are preserved untouched.
    Caller should call :meth:`RegistryHub.reload` after persist to refresh
    the in-memory registry.
    """
    fid = str(fam_id or "").strip().lower()
    if not fid:
        return False
    payload: Dict[str, Any] = {}
    if _user_custom_path().exists():
        existing = read_data(_user_custom_path())
        if isinstance(existing, dict):
            payload = dict(existing)
    families_blk = dict(payload.get("families", {}) or {})
    families_blk[fid] = dict(spec)
    payload["families"] = families_blk
    return bool(push_data(_USER_CUSTOM_REL, payload))


def delete_user_family(fam_id: str, *, force: bool = False) -> bool:
    """Remove a user-overlay family from ``ir_custom.json``.

    Refuses to delete a family still referenced by any robot's ``families``
    list (raises :class:`CascadeProtectionError`) unless ``force=True``;
    the UI catches that and prompts the user to remove the family from the
    robots first. Caller should call :meth:`RegistryHub.reload` afterwards.
    """
    fid = str(fam_id or "").strip().lower()
    if not fid:
        return False
    if fid not in _state["user_families"]:
        return False
    if not force:
        users = find_robots_using_family(fid)
        if users:
            raise CascadeProtectionError(
                f"family {fid!r} is referenced by robots {users}; "
                f"remove the family from those robots first"
            )
    if not _user_custom_path().exists():
        return False
    existing = read_data(_user_custom_path())
    if not isinstance(existing, dict):
        return False
    payload = dict(existing)
    families_blk = dict(payload.get("families", {}) or {})
    if fid not in families_blk:
        return False
    families_blk.pop(fid, None)
    payload["families"] = families_blk
    return bool(push_data(_USER_CUSTOM_REL, payload))


def get_version() -> str:
    return _state["version"]


def list_roles(family: str) -> List[Dict[str, Any]]:
    if not _state["loaded"]:
        load()
    return list(_state["families"].get(family, []))


def list_roles_for_families(
    *families: str,
    include_user_extensions: bool = False,
) -> List[Dict[str, Any]]:
    """组合 family 角色，去重保序 / Union roles across families."""
    if not _state["loaded"]:
        load()

    seen_ids: set[str] = set()
    out: List[Dict[str, Any]] = []
    for fam in families:
        for role in _state["families"].get(fam, []):
            rid = role["id"]
            if rid in seen_ids:
                continue
            seen_ids.add(rid)
            out.append(role)

    if include_user_extensions:
        for ext in _state["user_extensions"]:
            rid = ext["id"]
            if rid in seen_ids:
                continue
            seen_ids.add(rid)
            out.append(ext)
    return out


def get_role(family: str, role_id: str) -> Optional[Dict[str, Any]]:
    for r in list_roles(family):
        if r.get("id") == role_id:
            return r
    return None


def is_known_role(family: str, role_id: str) -> bool:
    return get_role(family, role_id) is not None


def list_families() -> List[str]:
    if not _state["loaded"]:
        load()
    return list(_state["families"].keys())


def get_intent(intent_id: str) -> Optional[Dict[str, Any]]:
    if not _state["loaded"]:
        load()
    return _state["intents"].get(intent_id)


def list_intents() -> List[Dict[str, Any]]:
    if not _state["loaded"]:
        load()
    return list(_state["intents"].values())


# ---------------------------------------------------------------------------
# Typed body-role accessors (Stage 2)
# Used by motion validator (Stage 5) + obs/action contract (Stage 5) to refuse
# silent role auto-extension. canonical_body_map() returns role_id → BodyRole
# for one or more families; collisions across families are deduped by first-
# write-wins (mirrors ``list_roles_for_families``).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BodyRole:
    """One canonical body role for a given family."""

    id: str
    category: str
    label: str
    position: str
    required: bool
    family: str  # the family the role was sourced from

    @classmethod
    def from_dict(cls, family: str, d: Dict[str, Any]) -> "BodyRole":
        return cls(
            id=str(d.get("id", "")),
            category=str(d.get("category", "")),
            label=str(d.get("label", "")),
            position=str(d.get("position", "")),
            required=bool(d.get("required", False)),
            family=family,
        )


def canonical_body_map(*families: str) -> Dict[str, BodyRole]:
    """Return ``role_id -> BodyRole`` for the union of the given families.

    If no families are passed, returns an empty mapping (callers must declare
    their morphology — § Multi-Brand Inclusiveness red line). Roles are
    deduplicated by ``id``; first family wins on collision (mirrors
    :func:`list_roles_for_families`).
    """
    if not _state["loaded"]:
        load()
    out: Dict[str, BodyRole] = {}
    for fam in families:
        for role in _state["families"].get(fam, []):
            rid = role.get("id")
            if not rid or rid in out:
                continue
            out[rid] = BodyRole.from_dict(fam, role)
    return out


# ---------------------------------------------------------------------------
# IR action descriptors（合并自原 ir_actions.py）
# 阶段 B 占位：阶段 C 随 application/compiler/behavior 迁移时填入真实数据。
# ---------------------------------------------------------------------------

def get_action(intent_id: str) -> Optional[Dict[str, Any]]:
    if not _state["loaded"]:
        load()
    return _state["actions"].get(intent_id)


def list_actions() -> List[Dict[str, Any]]:
    if not _state["loaded"]:
        load()
    return list(_state["actions"].values())


__all__ = [
    "load",
    "merge_user_extensions",
    "merge_user_family_extensions",
    "persist_user_family",
    "delete_user_family",
    "is_user_family",
    "list_user_families",
    "get_user_family_spec",
    "get_family_label",
    "find_robots_using_family",
    "CascadeProtectionError",
    "get_version",
    "list_roles",
    "list_roles_for_families",
    "get_role",
    "is_known_role",
    "list_families",
    "get_intent",
    "list_intents",
    "get_action",
    "list_actions",
    "BodyRole",
    "canonical_body_map",
]
