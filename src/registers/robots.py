"""registers.robots — 机器人模板注册表（含 joints + sensors）/ Canonical robot register.

合并自原 robots/ + sensor_catalog/（plan §11.3 / §11.5）/ Merged from former:
- robots/canonical.json    → data/robots_canonical.json
- robots/__init__.py       → 本文件（含 SKU/alias 折叠 + family 组合查询）
- sensor_catalog/__init__.py → 内联（sensor 是 robot 属性，无独立必要）

Schema 见 plan.md §11.3。每条 robot 以 SKU（``build_sku(brand, model)``）为 key；
``name`` 字段供 UI 显示，**不可作为查询键**。

API:
    load() -> int
    get_robot(sku) / list_skus() / get_version()
    resolve_id(user_input) -> str | None        # 大小写/分隔符折叠 → SKU
    list_by_family(*families)                   # superset family 匹配
    list_by_brand(brand)
    list_joints(robot_sku)
    get_joints_by_ir_role(robot_sku, ir_role)
    list_sensors(robot_sku)                     # robot.sensors 直读

⚠ 多品牌包容性：禁止在 robot 字段下增加 ``if_brand_is_X`` 分支。
品牌差异通过 ``adapter`` 字段指向 application/service/adapters/<brand>/。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from unitport_sdk import Paths, log_warning, push_data, read_data

from . import resolve_robot_sku


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
    "robots": {},          # sku → robot dict
    "alias_to_sku": {},
    "user_extensions": {},
}


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

    _state["robots"] = robots
    _state["alias_to_sku"] = alias
    _state["loaded"] = True
    return len(robots)


def merge_user_extensions(extensions: Dict[str, Dict[str, Any]]) -> int:
    added = 0
    for sku, entry in (extensions or {}).items():
        if not isinstance(entry, dict):
            log_warning(f"[robots] 跳过非法 robots_custom 条目：{sku}")
            continue
        _state["user_extensions"][sku] = entry
        _state["robots"][sku] = entry  # user 层覆盖出厂同 SKU
        added += 1
    return added


def persist_user_robot(sku: str, entry: Dict[str, Any]) -> bool:
    """写入/更新 user 层 robot 至 ~/UnitPort/registers/robots_custom.json.

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


def delete_user_robot(sku: str) -> bool:
    """从 ~/UnitPort/registers/robots_custom.json 中移除 sku 条目。"""
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


def list_joints(robot_sku: str) -> List[Dict[str, Any]]:
    entry = get_robot(robot_sku)
    if entry is None:
        return []
    return list(entry.get("joints", {}).values())


def get_joints_by_ir_role(robot_sku: str, ir_role: str) -> Optional[Dict[str, Any]]:
    for j in list_joints(robot_sku):
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
    in ``~/UnitPort/robot_presets/init_poses.json`` and are merged by
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
# Typed RobotSpec accessor (Stage 2)
# Stage 3 TrainingSpec compiler + Stage 6 generic MuJoCo env consume this:
# joint_order pins URDF / MJCF actuator order; body_role_map drives motion
# validator + reward shaping; mjcf_path / urdf_path feed env asset loader.
# action_dim mirrors the actuated joint count (assumes joint-position
# control; over-actuated rigs override via ``capabilities.action_dim``).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RobotSpec:
    """Snapshot of one canonical robot, typed for training-side consumers."""

    sku: str
    name: str
    brand: str
    model: str
    families: List[str]
    joint_order: List[str]                # raw joint names, registration order
    joint_ir_roles: List[str]             # parallel to joint_order
    body_role_map: Dict[str, str]         # raw_joint_name -> ir_role
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
    # Isaac Lab USD articulation 原生 joint 顺序（IR-role 形式）。
    # 与 ``joint_ir_roles`` 的 SDK canonical 顺序解耦：IL 训练时 obs/action
    # vector 按 USD prim 遍历序排列；对四足 Unitree 是按 type 分组
    # (all hips → all thighs → all calves)，与 SDK canonical 的按腿分组不一致。
    # ``None`` 表示该机器人 USD 顺序未确认 — bundle_finalizer 会回退到
    # ``joint_ir_roles`` 并打 warning。
    isaac_lab_joint_order: Optional[List[str]] = None

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

    joints_dict = entry.get("joints", {}) or {}
    joint_order: List[str] = []
    joint_ir_roles: List[str] = []
    body_role_map: Dict[str, str] = {}
    for jspec in joints_dict.values():
        if not isinstance(jspec, dict):
            continue
        raw = str(jspec.get("name", ""))
        ir_role = str(jspec.get("ir_role", ""))
        if not raw:
            continue
        joint_order.append(raw)
        joint_ir_roles.append(ir_role)
        if ir_role:
            body_role_map[raw] = ir_role

    assets = entry.get("assets", {}) or {}

    il_order_raw = entry.get("isaac_lab_joint_order")
    isaac_lab_joint_order: Optional[List[str]]
    if isinstance(il_order_raw, list) and il_order_raw:
        isaac_lab_joint_order = [str(x) for x in il_order_raw if x]
    else:
        isaac_lab_joint_order = None

    return RobotSpec(
        sku=str(entry.get("sku", sku)),
        name=str(entry.get("name", "")),
        brand=str(entry.get("brand", "")),
        model=str(entry.get("model", "")),
        families=list(entry.get("families", []) or []),
        joint_order=joint_order,
        joint_ir_roles=joint_ir_roles,
        body_role_map=body_role_map,
        mjcf_path=assets.get("MJCF") or None,
        urdf_path=assets.get("URDF") or None,
        usd_path=assets.get("USD") or None,
        usd_url=assets.get("USD_URL") or None,
        capabilities=dict(entry.get("capabilities", {}) or {}),
        sensors=list_sensors(sku),
        adapter=str(entry.get("adapter", "")),
        sdk_paths=list(entry.get("sdk_paths", []) or []),
        isaac_lab_joint_order=isaac_lab_joint_order,
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
    "is_user_extension",
    "get_version",
    "list_skus",
    "get_robot",
    "resolve_id",
    "list_by_family",
    "list_by_brand",
    "list_joints",
    "get_joints_by_ir_role",
    "get_robot_init_pose_presets",
    "list_sensors",
    "RobotSpec",
    "get_robot_spec",
]
