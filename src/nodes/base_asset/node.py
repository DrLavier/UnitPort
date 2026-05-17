"""BaseAssetNode — Start Point selector (3 semantic categories).

start_point token grammar (top-tier, exactly 3 entries):
    __new__      scratch — random init
    __latest__   cumulative training from THIS canvas's last produced
                 run (resolved at compile time from ``last_run_id``)
    __load__     transfer learning — pick any project-local checkpoint;
                 secondary ``checkpoint_id`` field carries the choice

``checkpoint_id`` grammar (only meaningful when start_point=__load__):
    run:<abs .pt path>     run-dir checkpoint (.pt) — resume-compatible
    export:<abs .onnx>     exported policy — warm_start_actor only

``last_run_id`` is written by ``MainWindow._on_start_training`` after
``submit_canvas_training`` succeeds; the picker reads it to decide
whether to surface the Latest option.
"""

from __future__ import annotations

from application.compiler.nodes import (
    NODE_MANIFEST_SCHEMA,
    BaseNode,
    NodeKind,
    NodeManifest,
    ParamSpec,
    PortSpec,
)


class BaseAssetNode(BaseNode):
    """Layer C.0 — Training start point selector."""

    _START_POINTS = ("__new__", "__latest__", "__load__")
    _LOAD_MODES = ("scratch", "resume", "warm_start_actor")

    MANIFEST = NodeManifest(
        schema=NODE_MANIFEST_SCHEMA,
        id="base_asset",
        kind=NodeKind.CUSTOM,
        version="2.0.0",
        category="config",
        layer="C",
        display_name_key="node.base_asset.display",
        description_key="node.base_asset.desc",
        icon="base_asset.svg",
        inputs=[],
        outputs=[PortSpec(name="checkpoint", type="checkpoint")],
        parameters=[
            ParamSpec(key="start_point", type="enum", default="__new__",
                      choices=list(_START_POINTS),
                      widget="picker_start_point",
                      description="起点语义：New / Latest（本画布最近 run） / Load（项目内任意 checkpoint）"),
            ParamSpec(key="checkpoint_id", type="string", default="",
                      widget="picker_project_checkpoint",
                      description="二级 checkpoint 选择，仅 start_point=__load__ 时可见；存储格式 run:<abs> 或 export:<abs>",
                      meta={"conditional_on": {"key": "start_point", "op": "==", "value": "__load__"}}),
            ParamSpec(key="load_mode", type="enum", default="scratch",
                      choices=list(_LOAD_MODES),
                      description="加载模式（start_point=__new__ 时该行隐藏，patch 自动落回 scratch）",
                      meta={"conditional_on": {"key": "start_point", "op": "!=", "value": "__new__"}}),
            ParamSpec(key="last_run_id", type="string", default="",
                      description="本画布最近一次训练的 run_id（trainer 启动时回写，UI 不渲染）",
                      meta={"hidden": True}),
        ],
    )
