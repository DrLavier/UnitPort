# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""StageSwitchNode — 时间多路选择器（最多 6 阶段）.

DEMO 对应：``src/system/nodes/sys_nodes/training_nodes.py:StageSwitchNode``.

把多个 train_pipe 顺序拼接成 staged run；本节点本身**不**修改 train_pipe
内容，只附加 ``stage_schedule`` payload 供 runner 迭代。

Phase 2 frontend skeleton — ``execute()`` 留接口给 Stage C 后端 wiring。
"""

from __future__ import annotations

from typing import Any, Dict, List

from application.compiler.nodes import (
    manifest_from_toml,
    NODE_MANIFEST_SCHEMA,
    BaseNode,
    NodeKind,
    NodeManifest,
    ParamSpec,
    PortSpec,
)


class StageSwitchNode(BaseNode):
    """Layer IL — Temporal multiplexer for staged training."""

    _MAX_STAGES = 6
    _OPTIONAL_INPUTS: set = {"stage_2", "stage_3", "stage_4", "stage_5"}

    MANIFEST = manifest_from_toml(__file__)

    # ---- 校验 ----

    def validate(self) -> List[str]:
        errors = super().validate()
        # stage_0 / stage_1 必连；其他可选。Cross-edge 校验由 canvas 引擎层做。
        return errors