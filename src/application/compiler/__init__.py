# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""application.compiler — 编译期产物层 / Compile-time artifact layer.

合并自原 application/tools/{ir,compiler,behavior,nodes,policy,skill}/ —
"compiler" 与运行期 ``application.engine`` 对偶；这里是无 I/O 的纯计算层。

实搬 / Migrated:
- ir.py            分层 IR 契约 + 桥/序列化器（合并自 tools/ir/{layered_contracts,layered_interfaces}.py）
- nodes.py         节点契约总线：BaseNode + NodeManifest + PortSpec + ParamSpec + NodeKind
- lowering.py      画布 ↔ IR 双向转换：WorkflowIR 数据契约 + canvas_to_ir / ir_to_canvas

节点实现下沉到 ``RELEASE/src/nodes/<id>/`` 与 ``RELEASE/custom_mods/nodes/<id>/``，
本层只持有契约。Discovery 由 ``registers.nodes.load()`` 完成。

待搬 / Pending（阶段 C，按依赖序）:
- parser.py / semantic.py / codegen.py    < DEMO/src/system/compiler/
- behavior.py    < DEMO/src/system/behavior/（去除已搬到 registers/ 的 *_catalog 文件）
- policy.py      < DEMO/src/system/policy/（删除 il_manifest_compat.py）
- skill.py       < DEMO/src/system/skill/

不搬 / Not migrated:
- workbench/    IR 工作台输出 → 由 src/runtime/workbench/ 承担（Paths.RUNTIME_DIR）
"""

from __future__ import annotations

from .ir import (
    BehaviorIR,
    BehaviorStateTransition,
    CommandIR,
    DefaultLayeredIRBridge,
    DefaultLayeredIRSerializer,
    ExecutionIR,
    LayeredIRBridge,
    LayeredIRBundle,
    LayeredIRSerializer,
    MigrationInfo,
    MissionIR,
    MotionParamSpec,
    ObservabilityIR,
    PackageMetadata,
    PackageSemanticValidator,
    SafetyIR,
    SemanticValidationResult,
    SensorBinding,
    SubgraphIR,
    SubgraphPort,
)
from .lowering import (
    EdgeType,
    IREdge,
    IRNode,
    IRNodeUI,
    IRParam,
    WorkflowIR,
    canvas_to_ir,
    ir_to_canvas,
)
from .nodes import (
    NODE_MANIFEST_SCHEMA,
    BaseNode,
    NodeKind,
    NodeManifest,
    ParamSpec,
    PortSpec,
)


__all__ = [
    # 节点契约（compiler.nodes）
    "BaseNode",
    "NodeKind",
    "NodeManifest",
    "PortSpec",
    "ParamSpec",
    "NODE_MANIFEST_SCHEMA",
    # WorkflowIR 数据契约（compiler.lowering）
    "EdgeType",
    "IRParam",
    "IRNodeUI",
    "IRNode",
    "IREdge",
    "WorkflowIR",
    # 画布 ↔ IR 转换函数（compiler.lowering）
    "canvas_to_ir",
    "ir_to_canvas",
    # 分层 IR contracts（compiler.ir）
    "BehaviorIR",
    "BehaviorStateTransition",
    "CommandIR",
    "ExecutionIR",
    "LayeredIRBundle",
    "MigrationInfo",
    "MissionIR",
    "MotionParamSpec",
    "ObservabilityIR",
    "PackageMetadata",
    "PackageSemanticValidator",
    "SafetyIR",
    "SemanticValidationResult",
    "SensorBinding",
    "SubgraphIR",
    "SubgraphPort",
    # IR bridge / serializer（compiler.ir）
    "DefaultLayeredIRBridge",
    "DefaultLayeredIRSerializer",
    "LayeredIRBridge",
    "LayeredIRSerializer",
]
