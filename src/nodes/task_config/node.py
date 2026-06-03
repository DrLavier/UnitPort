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
    manifest_from_toml,
    NODE_MANIFEST_SCHEMA,
    BaseNode,
    NodeKind,
    NodeManifest,
    ParamSpec,
    PortSpec,
)


class TaskConfigNode(BaseNode):
    """Layer A — Task configuration node."""

    MANIFEST = manifest_from_toml(__file__)