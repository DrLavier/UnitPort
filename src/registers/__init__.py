"""registers — 全局注册表层 / Global registry layer.

职责 / Purpose:
- 把 DEMO 中分散的 30 个注册表收口到一个目录下。
- 整层扁平化为 8 个 ``.py`` + 1 个 ``data/`` 数据目录（数据按 ``<域>_<键>.json`` 命名）。
- 业务代码只能通过 ``RegistryHub`` 或各域 ``.py`` 公开 API 查询，禁止绕过直读 ``data/``。

加载时机 / Loading:
- ``UnitPortMain.data_load()`` 调用 ``RegistryHub.load_all()`` 一次性读入并校验。
- 后续运行只查不重读（除非显式 ``reload()``）。

写入路径 / Write path:
- 用户运行时只能写 ``USER_CONFIG_DIR`` 下的层（通过 ``Storage.push_data``）。
- 出厂数据 ``data/*`` **禁止运行时修改**。
- 唯一例外：``data/backends_installed.json``（per-installation 物理事实，由
  ``backends.refresh_engine_availability()`` 主动写入）。

校验 / Validation:
- ``validate()`` 跨注册表检查：robot.joints[*].ir_role 必须在 IR canonical 之内。
- 失败抛 ``RegistryValidationError``，由 UnitPortMain 决定是阻塞启动还是降级。

USER overlay（plan.md §11.1）/ User-level overlays merged on load:
- ``~/UnitPort/registers/ir_custom.json``      → ir.merge_user_extensions
- ``~/UnitPort/registers/robots_custom.json``  → robots.merge_user_extensions

SKU helpers（数据库级唯一键）/ Database-level unique keys:
- ``build_sku("Unitree", "Go2-W") == build_sku("unitree", "go2w")``
- 后端互引一律用 SKU；前端/UI/日志显示 ``name`` 字段，永不显示 SKU。
- 12 位 hex（48 bit）≈ 2.8e14 空间，出厂百级机器人 + 千级关节安全。
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict

from unitport_sdk import Paths, log_debug, log_error, log_success, read_data


# =============================================================================
# SKU helpers（合并自 _sku.py / Inlined from former _sku.py）
# =============================================================================

_SLUG_RE = re.compile(r"[\s\-_/.]+")


def _normalize_sku_part(s: str) -> str:
    """大小写折叠 + 剥除所有分隔符 / Case-fold + strip all separators.

    使 "Go2-W" / "Go2 W" / "Go2_W" / "Go2.W" / "Go2W" 折叠为同一形态 "go2w"。
    """
    return _SLUG_RE.sub("", s.strip().lower())


def build_sku(*parts: str, length: int = 12) -> str:
    """根据多段语义键生成稳定 SKU / Build a stable SKU from semantic key parts."""
    raw = ".".join(_normalize_sku_part(p) for p in parts if p)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:length]


def resolve_robot_sku(brand: str, model: str) -> str:
    return build_sku(brand, model)


def resolve_joint_sku(robot_sku: str, raw_joint_name: str) -> str:
    return build_sku(robot_sku, raw_joint_name)


# =============================================================================
# RegistryHub — 统一加载/校验/查询入口
# =============================================================================

class RegistryValidationError(RuntimeError):
    """注册表跨域完整性校验失败 / Cross-registry validation failed."""


class RegistryHub:
    """所有子注册表的统一入口 / Unified entry to all sub-registries."""

    _loaded: bool = False
    _summary: Dict[str, int] = {}

    DOMAINS = ("ir", "robots", "nodes", "manifests", "backends", "brands", "services", "commands", "motion_phases")

    @classmethod
    def load_all(cls) -> None:
        """按依赖顺序加载所有子注册表 / Load every sub-registry in dependency order.

        顺序：
        1. ir（IR 角色枚举，无依赖）
        2. robots（依赖 IR 角色用作 joint.ir_role 值域）
        3. nodes / manifests / backends / brands / services
        4. 合入 USER_CONFIG_DIR 的拓展（ir_custom / robots_custom）
        """
        if cls._loaded:
            return

        cls._summary = {}

        from . import ir
        cls._summary["ir"] = ir.load()
        log_debug(f"[registers] ir loaded: {cls._summary['ir']} roles")

        from . import robots
        cls._summary["robots"] = robots.load()
        log_debug(f"[registers] robots loaded: {cls._summary['robots']} entries")

        from . import nodes
        cls._summary["nodes"] = nodes.load()

        from . import manifests
        cls._summary["manifests"] = manifests.load()

        from . import backends
        cls._summary["backends"] = backends.load()

        from . import brands
        cls._summary["brands"] = brands.load()

        from . import services
        cls._summary["services"] = services.load()

        from . import commands
        cls._summary["commands"] = commands.load()

        from . import motion_phases
        cls._summary["motion_phases"] = motion_phases.load()

        cls._merge_user_overlays()

        cls._loaded = True
        log_success(f"[registers] load_all complete: {cls._summary}")

    @classmethod
    def _merge_user_overlays(cls) -> None:
        """合入 USER_CONFIG_DIR 的 ir_custom + robots_custom（缺失=空集，不报错）。"""
        user_root = Paths.USER_CONFIG_DIR / "registers"

        ir_custom_path = user_root / "ir_custom.json"
        if ir_custom_path.exists():
            from . import ir
            payload = read_data(ir_custom_path) or {}
            extensions = list(payload.get("extensions", []) or [])
            if extensions:
                added = ir.merge_user_extensions(extensions)
                cls._summary["ir"] = cls._summary.get("ir", 0) + added
                log_debug(f"[registers] ir_custom merged: +{added} roles")

        robots_custom_path = user_root / "robots_custom.json"
        if robots_custom_path.exists():
            from . import robots
            payload = read_data(robots_custom_path) or {}
            extensions = dict(payload.get("robots", {}) or {})
            if extensions:
                added = robots.merge_user_extensions(extensions)
                cls._summary["robots"] = cls._summary.get("robots", 0) + added
                log_debug(f"[registers] robots_custom merged: +{added} entries")

        motion_phases_custom_path = user_root / "motion_phases_custom.json"
        if motion_phases_custom_path.exists():
            from . import motion_phases
            payload = read_data(motion_phases_custom_path) or {}
            if isinstance(payload, dict):
                added = motion_phases.merge_user_extensions(payload)
                if added:
                    cls._summary["motion_phases"] = cls._summary.get("motion_phases", 0) + added
                    log_debug(f"[registers] motion_phases_custom merged: +{added} phases")

    @classmethod
    def validate(cls) -> None:
        """跨注册表完整性校验 / Cross-registry integrity check.

        当前规则：
        1. 每个 robot 的 joints[*].ir_role 必须在该 robot families 对应的 IR roles 之内
        2. canonical_version 字段一致（提示用）

        失败抛 RegistryValidationError；调用方决定阻塞启动或降级。
        """
        if not cls._loaded:
            raise RuntimeError("RegistryHub.load_all() 必须先调用")

        from . import ir, robots

        problems: list[str] = []
        for sku in robots.list_skus():
            entry = robots.get_robot(sku)
            if entry is None:
                continue
            families = list(entry.get("families", []) or [])
            if not families:
                continue
            valid_role_ids = {r["id"] for r in ir.list_roles_for_families(*families)}
            joints = entry.get("joints", {}) or {}
            for jsku, jspec in joints.items():
                ir_role = (jspec or {}).get("ir_role", "")
                # CLAUDE.md §1.2 red line: every joint must have a non-empty
                # ir_role mapped into the canonical catalog. The earlier
                # truthy guard (``if ir_role and ...``) silently let
                # ir_role="" slip through — that path is the same family of
                # bug the il_manifest_compat.py auto-extension fix targets.
                role_str = str(ir_role).strip() if ir_role is not None else ""
                if not role_str:
                    problems.append(
                        f"  robot={sku}({entry.get('name','')}) "
                        f"joint={jsku} 的 ir_role 为空; 每个关节必须显式映射到 families={families} 的 IR canonical role"
                    )
                    continue
                if role_str not in valid_role_ids:
                    problems.append(
                        f"  robot={sku}({entry.get('name','')}) "
                        f"joint={jsku} ir_role={role_str!r} 不在 families={families} 的 IR canonical 内"
                    )

        if problems:
            msg = "[registers] validate failed:\n" + "\n".join(problems)
            log_error(msg)
            raise RegistryValidationError(msg)
        log_success("[registers] validate passed")

    @classmethod
    def reload(cls) -> None:
        """Drop every per-domain cache then re-run :meth:`load_all`.

        Each sub-registry keeps its own ``_state["loaded"]`` flag, so just
        flipping ``cls._loaded`` is not enough — the per-domain ``load()``
        short-circuits on its own cache. We reset both layers so the
        ``user_extensions`` overlay (e.g. ``robots_custom.json``) is re-read
        from disk and stale entries are dropped.
        """
        cls._loaded = False
        cls._summary.clear()
        for mod_name in cls.DOMAINS:
            try:
                mod = __import__(f"{__name__}.{mod_name}", fromlist=[mod_name])
            except Exception:
                continue
            mod_state = getattr(mod, "_state", None)
            if not isinstance(mod_state, dict):
                continue
            mod_state["loaded"] = False
            # Drop merged-in user overlays so prior persists/deletes do not
            # linger across reload boundaries.
            if "user_extensions" in mod_state:
                ue = mod_state["user_extensions"]
                if isinstance(ue, dict):
                    ue.clear()
                elif isinstance(ue, list):
                    ue.clear()
        cls.load_all()

    @classmethod
    def summary(cls) -> Dict[str, int]:
        """每个子注册表的条目数（调试用）/ Per-domain entry count for diagnostics."""
        return dict(cls._summary)

    @classmethod
    def is_loaded(cls) -> bool:
        return cls._loaded


__all__ = [
    "RegistryHub",
    "RegistryValidationError",
    "build_sku",
    "resolve_robot_sku",
    "resolve_joint_sku",
]
