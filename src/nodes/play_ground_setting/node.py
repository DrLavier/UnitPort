# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""PlayGroundSettingNode — 统一场景配置节点 (General §2 Scene).

DEMO 对应：``src/system/nodes/sys_nodes/training_nodes.py:PlayGroundSettingNode``.

行为契约：
    - 单一场景真理源：subsumes SceneConfigNode (SB3) + ILTerrainConfigNode +
      ILSimulationConfigNode (gravity / sim_dt / gpu_*)
    - scene_id 走 picker_scene widget，按训练 backend + robot family 过滤
    - rough 地形参数（roughness_amplitude / slope_max / curriculum_enabled /
      difficulty_levels）只在 scene_type == "rough" 可见
    - height_scan_* 只在 height_scan_enabled == true 可见
    - height_scan 不做 backend gating — Isaac Lab 原生支持，MuJoCo 端将来用
      heightfield ray-cast 对齐同一组参数

Phase 2 frontend skeleton — ``execute()`` 留接口给 Stage C 后端 wiring。
"""

from __future__ import annotations

from typing import Any, Dict

from application.compiler.nodes import (
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
        "roughness_amplitude", "slope_max",
        "curriculum_enabled", "difficulty_levels",
    )
    _HEIGHT_SCAN_ONLY_KEYS: tuple = (
        "scan_resolution", "scan_size_x", "scan_size_y",
    )

    MANIFEST = NodeManifest(
        schema=NODE_MANIFEST_SCHEMA,
        id="play_ground_setting",
        kind=NodeKind.CUSTOM,
        version="1.0.0",
        category="config",
        layer="A",
        display_name_key="node.play_ground_setting.display",
        description_key="node.play_ground_setting.desc",
        icon="play_ground.svg",
        inputs=[],
        outputs=[
            PortSpec(name="scene_pipe", type="scene_pipe"),
        ],
        parameters=[
            # §2.1 Scene identity
            ParamSpec(key="scene_id", type="string", default="flat_ground",
                      widget="picker_scene",
                      description="场景 id（registry-backed picker）"),
            ParamSpec(key="scene_type", type="enum", default="flat",
                      choices=["flat", "rough"],
                      description="场景类型（auto-synced from scene_id on pick）"),
            # §2.2 World physics
            ParamSpec(key="gravity_z", type="float", default=-9.81,
                      description="重力 Z 分量"),
            # §2.3 Arena
            ParamSpec(key="arena_extent_x", type="float", default=10.0,
                      widget="range", description="竞技场 X 范围",
                      meta={"min": 1.0, "max": 1000.0, "step": 0.1}),
            ParamSpec(key="arena_extent_y", type="float", default=10.0,
                      widget="range", description="竞技场 Y 范围",
                      meta={"min": 1.0, "max": 1000.0, "step": 0.1}),
            # §2.4 Ground material
            ParamSpec(key="friction_static", type="float", default=1.0,
                      widget="range", description="静摩擦",
                      meta={"min": 0.0, "max": 5.0, "step": 0.01}),
            ParamSpec(key="friction_dynamic", type="float", default=0.8,
                      widget="range", description="动摩擦",
                      meta={"min": 0.0, "max": 5.0, "step": 0.01}),
            # §2.5 Rough-terrain knobs (conditional on scene_type == "rough")
            ParamSpec(key="roughness_amplitude", type="float", default=0.08,
                      widget="range", description="粗糙地面幅度",
                      meta={"conditional_on": {"key": "scene_type", "op": "==", "value": "rough"},
                            "min": 0.0, "max": 1.0, "step": 0.01}),
            ParamSpec(key="slope_max", type="float", default=20.0,
                      widget="range", description="最大坡度（度）",
                      meta={"conditional_on": {"key": "scene_type", "op": "==", "value": "rough"},
                            "min": 0.0, "max": 60.0, "step": 0.5}),
            ParamSpec(key="curriculum_enabled", type="bool", default=False,
                      description="启用地形难度 curriculum",
                      meta={"conditional_on": {"key": "scene_type", "op": "==", "value": "rough"}}),
            ParamSpec(key="difficulty_levels", type="int", default=10,
                      widget="index", description="curriculum 难度层数",
                      meta={"conditional_on": {"key": "scene_type", "op": "==", "value": "rough"},
                            "min": 1, "max": 100}),
            # §2.6 Height scan (backend-agnostic)
            ParamSpec(key="height_scan_enabled", type="bool", default=False,
                      description="启用 height scan"),
            ParamSpec(key="scan_resolution", type="float", default=0.1,
                      widget="range", description="扫描分辨率（m）",
                      meta={"conditional_on": {"key": "height_scan_enabled", "op": "==", "value": True},
                            "min": 0.01, "max": 1.0, "step": 0.01}),
            ParamSpec(key="scan_size_x", type="float", default=1.6,
                      widget="range", description="扫描区域 X（m）",
                      meta={"conditional_on": {"key": "height_scan_enabled", "op": "==", "value": True},
                            "min": 0.1, "max": 10.0, "step": 0.1}),
            ParamSpec(key="scan_size_y", type="float", default=1.0,
                      widget="range", description="扫描区域 Y（m）",
                      meta={"conditional_on": {"key": "height_scan_enabled", "op": "==", "value": True},
                            "min": 0.1, "max": 10.0, "step": 0.1}),
            # §2.7 Sim engine globals (merged from ILSimulationConfigNode)
            ParamSpec(key="sim_dt", type="float", default=0.005,
                      description="物理步长（s）"),
            # NOTE: gpu_max_rigid_patch_count / gpu_max_rigid_contact_count
            # were intentionally REMOVED 2026-05-10. They are now computed
            # by env_cfg_compiler._simulation_cfg_literal as a function of
            # num_envs × terrain_density, and should never be user-edited
            # — the original sliders allowed configurations that either
            # overflowed (too small) or wasted GPU memory (too large) with
            # no way for the user to know which they had picked. See
            # project_release_il_compiler_guards memory.
        ],
    )

    # ---- 画布 UI 钩子（动态隐藏条件参数） ----

    @property
    def hidden_params(self) -> set:
        """根据 scene_type / height_scan_enabled 动态隐藏条件参数."""
        hidden: set = set()
        scene_type = str(self.get_parameter("scene_type", "flat") or "flat").strip()
        if scene_type != "rough":
            hidden.update(self._ROUGH_ONLY_KEYS)
        height_scan = self.get_parameter("height_scan_enabled", False)
        # bool 与 "true"/"True" 字符串都视为开
        if not (height_scan is True or str(height_scan).strip().lower() == "true"):
            hidden.update(self._HEIGHT_SCAN_ONLY_KEYS)
        return hidden