# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""TaskConfigNode — RL 任务定义 / RL task definition.

DEMO 对应：``src/system/nodes/sys_nodes/training_nodes.py:TaskConfigNode``.

Layer A / category=config. 定义任务级语义：任务类型、命令模式、
curriculum、success & truncation 策略；Reward 与 termination 模块由
专用 input pipe 注入。

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


class TaskConfigNode(BaseNode):
    """Layer A — Task configuration node."""

    MANIFEST = NodeManifest(
        schema=NODE_MANIFEST_SCHEMA,
        id="task_config",
        kind=NodeKind.CUSTOM,
        version="1.0.0",
        category="config",
        layer="A",
        display_name_key="node.task_config.display",
        description_key="node.task_config.desc",
        icon="task_config.svg",
        inputs=[
            PortSpec(name="total_reward", type="total_reward"),
            PortSpec(name="termination_config", type="termination_config"),
        ],
        outputs=[
            PortSpec(name="task_config", type="task_config"),
        ],
        parameters=[
            ParamSpec(
                key="task_type", type="enum", default="velocity_tracking",
                choices=["velocity_tracking"], widget="picker_task_type",
                description="任务类型 / Task type",
            ),
            ParamSpec(
                key="command_mode", type="enum", default="fixed",
                choices=["fixed", "random"],
                description="命令模式 / Command mode",
            ),
            ParamSpec(
                key="target_lin_vel_x", type="float", default=0.5,
                description="目标 X 速度 / Target linear velocity X",
            ),
            ParamSpec(
                key="target_lin_vel_y", type="float", default=0.0,
                description="目标 Y 速度 / Target linear velocity Y",
            ),
            ParamSpec(
                key="target_ang_vel_z", type="float", default=0.0,
                description="目标偏航角速度 / Target yaw rate",
            ),
            ParamSpec(
                key="curriculum", type="bool", default=False,
                description="启用 curriculum / Enable curriculum",
            ),
            ParamSpec(
                key="success_threshold", type="float", default=0.8,
                widget="range",
                description="成功阈值 / Success threshold",
                meta={"min": 0.0, "max": 1.0, "step": 0.01},
            ),
            ParamSpec(
                key="truncation_max_steps", type="int", default=0,
                widget="index",
                description="截断最大步数（0=禁用） / Truncation max steps (0 = disabled)",
                meta={"min": 0, "max": 1_000_000},
            ),
            ParamSpec(
                key="curriculum_schedule", type="json", default="{}",
                widget="code",
                description="curriculum 调度 (JSON) / Curriculum schedule",
                meta={"language": "json"},
            ),
            ParamSpec(
                key="resampling_time_range", type="list", default="[0.0, 0.0]",
                widget="range_list",
                description="命令重采样时间窗 / Command resampling time range",
                meta={"min": 0.0, "max": 30.0, "step": 0.1},
            ),
        ],
    )