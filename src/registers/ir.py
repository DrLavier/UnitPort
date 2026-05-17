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
    merge_user_extensions(extensions)            # 来自 ~/UnitPort/registers/ir_custom.json
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from unitport_sdk import Paths, log_warning, read_data


_DATA_DIR = Paths.REGISTERS_DIR / "data"
_CANONICAL_PATH = _DATA_DIR / "ir_canonical.json"
_INTENTS_PATH = _DATA_DIR / "ir_intent_vocab.json"

_state: Dict[str, Any] = {
    "loaded": False,
    "version": "",
    "families": {},          # family → list[role dict]
    "user_extensions": [],
    "intents": {},
    "actions": {},           # intent_id → action descriptor (阶段 C 填)
}


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
