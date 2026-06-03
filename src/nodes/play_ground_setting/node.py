# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""PlayGroundSettingNode — 统一场景配置节点 (General §2 Scene).

DEMO 对应：``src/system/nodes/sys_nodes/training_nodes.py:PlayGroundSettingNode``.

行为契约：
    - 单一场景真理源：subsumes SceneConfigNode (SB3) + ILTerrainConfigNode +
      ILSimulationConfigNode (gravity / sim_dt / gpu_*)
    - scene_id 走 picker_scene widget，按训练 backend + robot family 过滤
    - rough 地形参数（curriculum_enabled / difficulty_levels / prop_* /
      max_init_terrain_level）只在 scene_type == "rough" 可见
    - height_scan_* 只在 height_scan_enabled == true 可见
    - height_scan 不做 backend gating — Isaac Lab 原生支持，MuJoCo 端将来用
      heightfield ray-cast 对齐同一组参数

Phase 2 frontend skeleton — ``execute()`` 留接口给 Stage C 后端 wiring。
"""

from __future__ import annotations

from typing import Any, Dict

from application.compiler.nodes import (
    manifest_from_toml,
    NODE_MANIFEST_SCHEMA,
    BaseNode,
    NodeKind,
    NodeManifest,
    ParamSpec,
    PortSpec,
)


class PlayGroundSettingNode(BaseNode):
    """Layer A — Unified scene configuration node."""

    # ---- DEMO 行为契约（class-level 提示，画布 UI 读取） ----
    _INIT_SCENE_ID = "flat_ground"

    # 条件可见参数集合（与 manifest meta.conditional_on 一一对应）
    _ROUGH_ONLY_KEYS: tuple = (
        "curriculum_enabled", "difficulty_levels",
        "max_init_terrain_level",
        "prop_pyramid_stairs", "prop_pyramid_stairs_inv", "prop_boxes",
        "prop_random_rough", "prop_slope", "prop_slope_inv",
    )
    _HEIGHT_SCAN_ONLY_KEYS: tuple = (
        "scan_resolution", "scan_size_x", "scan_size_y",
    )
    # Visible only when scene_type == "custom" (user-imported heightfield).
    _CUSTOM_ONLY_KEYS: tuple = (
        "custom_terrain_path", "custom_terrain_vertical_scale",
    )

    MANIFEST = manifest_from_toml(__file__)

    # ---- 画布 UI 钩子（动态隐藏条件参数） ----

    @property
    def hidden_params(self) -> set:
        """根据 scene_type / height_scan_enabled 动态隐藏条件参数."""
        hidden: set = set()
        scene_type = str(self.get_parameter("scene_type", "flat") or "flat").strip()
        if scene_type != "rough":
            hidden.update(self._ROUGH_ONLY_KEYS)
        if scene_type != "custom":
            hidden.update(self._CUSTOM_ONLY_KEYS)
        height_scan = self.get_parameter("height_scan_enabled", False)
        # bool 与 "true"/"True" 字符串都视为开
        if not (height_scan is True or str(height_scan).strip().lower() == "true"):
            hidden.update(self._HEIGHT_SCAN_ONLY_KEYS)
        return hidden