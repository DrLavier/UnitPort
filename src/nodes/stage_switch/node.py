"""StageSwitchNode — 时间多路选择器（最多 6 阶段）.

DEMO 对应：``src/system/nodes/sys_nodes/training_nodes.py:StageSwitchNode``.

把多个 train_pipe 顺序拼接成 staged run；本节点本身**不**修改 train_pipe
内容，只附加 ``stage_schedule`` payload 供 runner 迭代。

Phase 2 frontend skeleton — ``execute()`` 留接口给 Stage C 后端 wiring。
"""

from __future__ import annotations

from typing import Any, Dict, List

from application.compiler.nodes import (
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

    MANIFEST = NodeManifest(
        schema=NODE_MANIFEST_SCHEMA,
        id="stage_switch",
        kind=NodeKind.CUSTOM,
        version="1.0.0",
        category="training",
        layer="IL",
        display_name_key="node.stage_switch.display",
        description_key="node.stage_switch.desc",
        icon="stage_switch.svg",
        inputs=[
            PortSpec(name="stage_0", type="train_pipe"),
            PortSpec(name="stage_1", type="train_pipe"),
            PortSpec(name="stage_2", type="train_pipe", optional=True),
            PortSpec(name="stage_3", type="train_pipe", optional=True),
            PortSpec(name="stage_4", type="train_pipe", optional=True),
            PortSpec(name="stage_5", type="train_pipe", optional=True),
        ],
        outputs=[
            PortSpec(name="train_pipe", type="train_pipe"),
        ],
        parameters=[
            ParamSpec(
                key="stages_config", type="list", default="[]",
                widget="stage_editor",
                description="阶段调度 / Stage schedule (Node_StageSwitchEditorWidget)",
                meta={"max_stages": _MAX_STAGES},
            ),
            ParamSpec(
                key="checkpoint_strategy", type="enum", default="both",
                choices=["both", "best", "last"],
                description="checkpoint 保存策略",
            ),
        ],
    )

    # ---- 校验 ----

    def validate(self) -> List[str]:
        errors = super().validate()
        # stage_0 / stage_1 必连；其他可选。Cross-edge 校验由 canvas 引擎层做。
        return errors