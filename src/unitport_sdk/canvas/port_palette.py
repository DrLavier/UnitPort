# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""unitport_sdk.canvas.port_palette — 端口类型 → 颜色 / Port type → color.

合并自 RELEASE Phase 1 (`application/ui/canvas/port_palette.py`, 39 entries) +
DEMO `bin/nodes/training_node_items.py:110-163` 的 18 个 Isaac Lab 颗粒类型，
合计 ~50 项可识别 data_type。

颜色优先级（从高到低）：
    1. ``Config.get_color(f"canvas_port_{data_type}", ...)`` —— theme 用户覆盖
    2. ``PORT_TYPE_COLORS`` 字典默认值
    3. 通用 fallback ``"#9CA3AF"``（中性灰）

为什么不在 `system.ini` 里塞 ~50 个 ``canvas_port_*`` slot：
- type 字典是产品契约的一部分（节点 manifest 的 PortSpec.type 用同名引用），
  让用户随便改色会让两个原本相同 type 的端口画成不同颜色，逻辑断链。
- system.ini ``[Theme]`` 是给整体视觉风格的，~50 个细分 slot 会污染 theme
  panel。Config.get_color 仍提供 override 入口给确实要定制的用户。
"""

from __future__ import annotations

from typing import Dict

from PyQt6.QtGui import QColor

from unitport_sdk.sys import Config


# =============================================================================
# 默认色板 / Default palette
# =============================================================================

PORT_TYPE_COLORS: Dict[str, str] = {
    # ---- 通用基础类型 / generic primitives ----
    "any":          "#9CA3AF",   # 灰
    "bool":         "#22C55E",   # 绿
    "int":          "#3B82F6",   # 蓝
    "float":        "#06B6D4",   # 青
    "string":       "#F59E0B",   # 琥珀
    "dict":         "#EC4899",   # 粉
    "list":         "#F97316",   # 橙
    "function":     "#A855F7",   # 紫
    "flow":         "#888888",   # 灰（控制流）

    # ---- Layer A 输出类型（环境配置层）----
    "task_config":          "#81C784",
    "rewards":              "#38BDF8",
    "reward_pipe":          "#38BDF8",   # RewardsNode v2 per-item 输出
    "terminations":         "#F472B6",
    "init_pose_config":     "#FFB74D",
    "robot_config":         "#FF8C42",
    "robot_spec":           "#FF8C42",   # 同 robot_config（DEMO 兼容）
    "actor_setting":        "#9575CD",
    "joint_init_config":    "#4DD0E1",
    "playground_setting":   "#AED581",
    "physics_config":       "#4FC3F7",
    "domain_rand_config":   "#BA68C8",
    "obs_action_config":    "#FFD54F",
    "multi_gated_reward":   "#7986CB",
    "asset_config":         "#A1887F",

    # ---- Layer B 输出类型（环境组装）----
    "training_motion_config":   "#FF7043",
    "env_config":               "#4DB6AC",

    # ---- Layer C 输出类型（algorithm + train）----
    "algo_config":      "#F48FB1",
    "train_config":     "#F48FB1",   # 同 algo_config（AlgorithmConfigNode 输出名）
    "train_pipe":       "#FF7043",
    "train_result":     "#26A69A",

    # ---- Layer D 输出类型（eval / export / check）----
    "eval_config":          "#A5D6A7",
    "bundle_path":          "#90A4AE",
    "vis_check_result":     "#CE93D8",
    "checkpoint":           "#CE93D8",   # 同 vis_check_result

    # ---- Layer IL 输出类型（imitation learning）----
    "il_obs_config":            "#7E57C2",
    "il_network_config":        "#5C6BC0",
    "disc_config":              "#42A5F5",
    "discriminator_config":     "#42A5F5",   # 同 disc_config（DiscriminatorNode 输出名）
    "amp_helper_config":        "#26C6DA",
    "il_ppo_result":            "#EC407A",
    "amp_result":               "#AB47BC",
    "stage_switch_config":      "#FFA726",

    # ---- Isaac Lab 颗粒类型（DEMO training_node_items.py:110-163）----
    "sim_context":              "#78909C",
    "terrain":                  "#8D6E63",
    "robot_entity":             "#EF5350",
    "joint_info":               "#EF9A9A",
    "body_info":                "#FFCDD2",
    "actuator_config":          "#FF8A65",
    "obs_vector":               "#0288D1",
    "action_entity":            "#66BB6A",
    "policy":                   "#7E57C2",
    "training_metrics":         "#9575CD",
    "exported_model":           "#4DB6AC",
    "termination_config":       "#E57373",
    "contact_sensor":           "#CE93D8",

    # ---- 6-section 统一管道（DEMO）----
    "robot_pipe":               "#00695C",
    "actor_pipe":               "#00695C",
    "joint_init":               "#26A69A",   # 注意：与 joint_init_config 是不同 type
    "scene_pipe":               "#2D6A2D",
    "command_pipe":             "#1565C0",
    "reference_motion_pipe":    "#7E57C2",

    # ---- 其他 DEMO 残留类型 ----
    "vis_check":                "#CE93D8",
    "reference_motion_config":  "#7E57C2",
    "init_pose":                "#FFB74D",
    "joint_config":             "#4DD0E1",
    "export_config":            "#90A4AE",
    "velocity_command":         "#1565C0",
    "command_source":           "#1565C0",
    "action_source":            "#66BB6A",
    "body_mapping":             "#FFCDD2",
    "height_scanner":           "#0288D1",
}

_DEFAULT_FALLBACK = "#9CA3AF"


# =============================================================================
# 查询 API
# =============================================================================

def default_color_hex(data_type: str) -> str:
    """返回某 data_type 的默认色（hex）；未知则返回中性灰."""
    return PORT_TYPE_COLORS.get(data_type, _DEFAULT_FALLBACK)


def get_port_color(data_type: str) -> QColor:
    """端口颜色查询 / Port color lookup.

    用户可通过 ``user.ini [Theme] canvas_port_<type> = #XXXXXX`` 覆盖；
    未覆盖则回落到 ``PORT_TYPE_COLORS``；都不命中则中性灰。
    """
    fallback = default_color_hex(data_type)
    slot = f"canvas_port_{data_type}"
    return QColor(Config.get_color(slot, fallback))


def known_types() -> tuple:
    """返回内置色板中所有已知 data_type 名（用于自检 / 文档生成）."""
    return tuple(PORT_TYPE_COLORS.keys())


__all__ = [
    "PORT_TYPE_COLORS",
    "default_color_hex",
    "get_port_color",
    "known_types",
]
