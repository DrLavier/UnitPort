# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""registers.brands — 品牌 / 适配器 索引 / Brand & adapter index.

合并自原 brands/ 子目录 / Merged from former brands/ subdir:
- packages.json     → data/brands_packages.json   （出厂品牌包清单）
- adapters.yaml     → data/brands_adapters.yaml   （ROS2 adapter dotted-path 映射）

每个机器人的 ``adapter`` 字段在 ``registers.robots.<robot>.adapter``；本注册表
管的是品牌级元数据。

API:
    load() -> int
    get_brand(brand_id) / list_brands()
    list_built_in_adapters() -> dict[str, str]
    get_adapter_class(adapter_id) -> str | None
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from unitport_sdk import Paths, log_warning, read_data


_DATA_DIR = Paths.REGISTERS_DIR / "data"
_PACKAGES = _DATA_DIR / "brands_packages.json"
_ADAPTERS = _DATA_DIR / "brands_adapters.yaml"
# Per-robot brand DATA tree (remotized-actuator tables + manifests, etc.).
# DATA ONLY — see registers/brands/<brand>/<model>/README.md.
_BRANDS_TREE = Paths.REGISTERS_DIR / "brands"

_state: Dict[str, Any] = {
    "loaded": False,
    "brands": {},
    "adapters_built_in": {},
    "host_adapters_built_in": {},
}


def load() -> int:
    if _state["loaded"]:
        return len(_state["brands"])

    pkgs = read_data(_PACKAGES) or {}
    _state["brands"] = dict(pkgs.get("brands", {}) or {})

    try:
        adapters = read_data(_ADAPTERS) or {}
        _state["adapters_built_in"] = dict(adapters.get("built_in", {}) or {})
        # Host-side adapters (run inside the Studio process; bridge brand
        # semantics to NativeDDSBridge / vendor SDKs) are a separate map
        # from the robot-side built_in adapters above.
        _state["host_adapters_built_in"] = dict(
            adapters.get("host_built_in", {}) or {}
        )
    except Exception as e:  # noqa: BLE001
        log_warning(f"[brands] adapters.yaml 读取失败：{e}")
        _state["adapters_built_in"] = {}
        _state["host_adapters_built_in"] = {}

    _state["loaded"] = True
    return len(_state["brands"])


def get_brand(brand_id: str) -> Optional[Dict[str, Any]]:
    if not _state["loaded"]:
        load()
    return _state["brands"].get(str(brand_id or "").strip().lower())


def list_brands() -> List[Dict[str, Any]]:
    if not _state["loaded"]:
        load()
    return list(_state["brands"].values())


def list_built_in_adapters() -> Dict[str, str]:
    if not _state["loaded"]:
        load()
    return dict(_state["adapters_built_in"])


def get_adapter_class(adapter_id: str) -> Optional[str]:
    if not _state["loaded"]:
        load()
    return _state["adapters_built_in"].get(adapter_id)


def remotized_manifest(brand: str, model: str) -> Optional[Dict[str, Any]]:
    """Load ``registers/brands/<brand>/<model>/manifest.yaml`` or ``None``.

    Returns the parsed remotized-actuator manifest for a robot, with each
    ``remotized_joints[*]`` entry augmented with an absolute
    ``lookup_table_path`` (resolved against the manifest directory) so callers
    need not reconstruct the brand-data layout. ``None`` when the robot has no
    such declaration — that is the normal case for non-remotized robots, NOT
    an error.

    DATA ONLY: this reads a data file; it carries no brand-specific logic
    (CLAUDE.md §1). The mechanism is generic ("remotized actuator with a
    torque lookup table"); the brand contributes only the table + manifest.
    """
    b = str(brand or "").strip().lower()
    m = str(model or "").strip().lower()
    if not b or not m:
        return None
    bdir = _BRANDS_TREE / b / m
    mpath = bdir / "manifest.yaml"
    if not mpath.is_file():
        return None
    data = read_data(mpath)
    if not isinstance(data, dict):
        log_warning(f"[brands] remotized manifest is not a mapping: {mpath}")
        return None
    for entry in data.get("remotized_joints", []) or []:
        if isinstance(entry, dict) and entry.get("lookup_table"):
            entry["lookup_table_path"] = str(bdir / entry["lookup_table"])
    return data


def get_host_adapter_class(adapter_id: str) -> Optional[str]:
    """Return the host-side dotted-path for ``adapter_id`` or None.

    Host-side adapters live in ``application/service/adapters/<brand>/`` and
    are resolved by :class:`application.service.adapters.AdapterFactory`.
    Distinct from :func:`get_adapter_class`, which returns ROBOT-side bridge
    adapter paths shipped inside the UnitPort ROS2 workspace.
    """
    if not _state["loaded"]:
        load()
    return _state["host_adapters_built_in"].get(adapter_id)


def list_host_built_in_adapters() -> Dict[str, str]:
    if not _state["loaded"]:
        load()
    return dict(_state["host_adapters_built_in"])


__all__ = [
    "load",
    "get_brand",
    "list_brands",
    "list_built_in_adapters",
    "get_adapter_class",
    "get_host_adapter_class",
    "list_host_built_in_adapters",
    "remotized_manifest",
]
