# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""application — UnitPort 业务代码命名空间 / Business code namespace.

子包（高度集中化布局，plan §3）:
- ui/         所有 UI：main_window/页面/控件/画布/对话框/资源
- compiler/   编译期产物：IR + behavior + nodes-logic + policy + skill
- training/   训练计算管线（待阶段 C 填）
- engine/     运行期执行层：IR runtime + safety + sim 后端 + ROS2 桥（待阶段 C 填）
- service/    I/O + 状态：adapters/brands/connection/deployment/auth/telemetry/storage

跨层公共类型（合并自原 tools/types/common_types.py）/ Common type aliases:
"""

from __future__ import annotations

from typing import Any, Dict


RobotId = str
ActionName = str
AdapterName = str
SensorPayload = Dict[str, Any]
Params = Dict[str, Any]


__all__ = [
    "AdapterName",
    "ActionName",
    "Params",
    "RobotId",
    "SensorPayload",
]
