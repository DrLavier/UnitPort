"""EvalConfigNode — 训练后评估配置.

DEMO 对应：``training_nodes.py:EvalConfigNode``.
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


class EvalConfigNode(BaseNode):
    """Layer D — Post-training evaluation configuration."""

    MANIFEST = NodeManifest(
        schema=NODE_MANIFEST_SCHEMA,
        id="eval_config",
        kind=NodeKind.CUSTOM,
        version="1.0.0",
        category="config",
        layer="D",
        display_name_key="node.eval_config.display",
        description_key="node.eval_config.desc",
        icon="eval_config.svg",
        inputs=[],
        outputs=[PortSpec(name="eval_config", type="eval_config")],
        parameters=[
            ParamSpec(key="eval_episodes", type="int", default=20, widget="index",
                      description="评估回合数",
                      meta={"min": 1, "max": 10000}),
            ParamSpec(key="deterministic", type="bool", default=True,
                      description="确定性策略"),
            ParamSpec(key="success_threshold", type="float", default=0.8, widget="range",
                      description="成功阈值",
                      meta={"min": 0.0, "max": 1.0, "step": 0.01}),
            ParamSpec(key="record_video", type="bool", default=False,
                      description="录制视频"),
            ParamSpec(key="eval_interval", type="int", default=50000, widget="index",
                      description="训练间隔评估步数",
                      meta={"min": 1, "max": 100_000_000}),
            ParamSpec(key="save_best_model", type="bool", default=True,
                      description="保存最佳模型"),
            ParamSpec(key="video_dir", type="string", default="", widget="path",
                      description="视频输出目录",
                      meta={"is_dir": True}),
        ],
    )