"""application.compiler.nodes — 节点契约总线 / Node contract bus.

UnitPort 画布节点的强类型契约层：
- ``BaseNode``        所有节点必须继承的 ABC
- ``NodeManifest``    与 ``manifest.toml`` schema 一一对齐的 dataclass
- ``PortSpec`` / ``ParamSpec``    端口与参数的强类型规约
- ``NodeKind``        枚举（迁自 DEMO ``src/system/compiler/ir/workflow_ir.py:17-44`` 的 24 变体）

节点实现一律下沉到 ``RELEASE/src/nodes/<id>/`` 或 ``RELEASE/custom_mods/nodes/<id>/``，
每个文件夹含 ``__init__.py`` (暴露 ``NODE_CLASSES``) + ``manifest.toml`` + ``node.py``。
本模块**不**做发现也**不**持有注册表——发现在 ``registers.nodes.load()`` 内完成。
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# =============================================================================
# NodeKind — 节点语义分类
# =============================================================================

class NodeKind(Enum):
    """节点的语义类别 / Semantic kind of a node.

    Mission canvas 的 24 个 FSM 控制流变体（START/END/IF/SWITCH/PARALLEL/...）
    在 RELEASE 已废弃；当前训练画布的所有节点 100% 走 ``CUSTOM`` kind，
    具体业务区分由 ``NodeManifest.layer`` (A/B/C/D/IL) 与 ``category`` 字段承载。
    """

    CUSTOM = "custom"

    @classmethod
    def from_string(cls, s: str) -> "NodeKind":
        """字符串 → ``NodeKind``；不识别归 ``CUSTOM``."""
        try:
            return cls(s.lower())
        except ValueError:
            return cls.CUSTOM


# =============================================================================
# PortSpec / ParamSpec — 强类型端口与参数规约
# =============================================================================

@dataclass(frozen=True)
class PortSpec:
    """节点端口规约 / Node port specification.

    - ``type`` 取值: ``"flow"`` / ``"bool"`` / ``"int"`` / ``"float"`` / ``"string"`` /
      ``"dict"`` / ``"list"`` / ``"any"`` / 自定义协议名（如 ``"protocol.cmd_vel"``）
    - ``multi`` 允许多连接（默认单连接）
    - ``optional`` 允许悬空（默认必连）
    - ``meta`` 端口附加元数据（与 ``ParamSpec.meta`` 对齐）。当前已用 key:
        * ``conditional_on``: ``{"key": str, "op": "=="|"!="|"in"|"not in", "value": Any}``
          — UI 据 host params 决定端口显隐（与 ParamRow conditional 同协议）
    """

    name: str
    type: str
    multi: bool = False
    optional: bool = False
    description: str = ""
    meta: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class ParamSpec:
    """节点参数规约 / Node parameter specification.

    - ``type`` 取值: ``"string"`` / ``"int"`` / ``"float"`` / ``"bool"`` / ``"enum"`` / ``"json"``
    - ``choices`` 仅在 ``type == "enum"`` 时给出
    - ``widget`` 可选 UI 行类型 hint：``"path"`` / ``"code"`` / ``"range"`` / ``"index"`` /
      ``"badge"`` / ``"table_readonly"``；缺省由 ``type`` 推断
    - ``meta`` 行特定附加参数（如 range 的 ``{"min": 0, "max": 1, "step": 0.01}``）；
      具体 key 由各 ParamRow 子类定义
    """

    key: str
    type: str
    default: Any = None
    choices: Optional[List[Any]] = None
    description: str = ""
    widget: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None


# =============================================================================
# NodeManifest — manifest.toml 的 Python 镜像
# =============================================================================

NODE_MANIFEST_SCHEMA = "unitport.node/v1"


_VALID_LAYERS = ("A", "B", "C", "D", "IL", "TOOLS")


@dataclass
class NodeManifest:
    """与 ``manifest.toml`` schema 一一对齐 / Mirror of ``manifest.toml`` schema.

    必填: ``schema`` / ``id`` / ``kind`` / ``version``
    选填: ``category`` / ``layer`` / ``display_name_key`` / ``description_key`` /
          ``icon`` / ``inputs`` / ``outputs`` / ``parameters``

    ``schema`` 字段固定为 ``"unitport.node/v1"``；后续破坏性升级走 v2 路径。

    ``layer`` 取值 ``A`` / ``B`` / ``C`` / ``D`` / ``IL`` 之一（或 ``None``）；
    决定画布上节点标题栏配色，不参与 IR 语义。
    """

    schema: str
    id: str
    kind: NodeKind
    version: str = "1.0.0"
    category: str = "misc"
    layer: Optional[str] = None
    display_name_key: str = ""
    description_key: str = ""
    icon: str = ""
    inputs: List[PortSpec] = field(default_factory=list)
    outputs: List[PortSpec] = field(default_factory=list)
    parameters: List[ParamSpec] = field(default_factory=list)
    # True 表示该节点承载训练超参 — mission_panel 训练配置卡靠这个 flag 在画布
    # 上动态发现 trainer 节点，再从其 ParamSpec.meta 收集带 mission_expose=True
    # 的字段；rule §1.1：core 不写死任何具体 trainer schema 名。
    is_trainer: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NodeManifest":
        """从 TOML/JSON 解析 dict 构造 ``NodeManifest`` 并校验.

        Raises:
            ValueError: schema 缺失或不识别；id 为空；kind 不在 ``NodeKind`` 枚举内；
                        layer 不在 ``A|B|C|D|IL`` 内
        """
        schema = data.get("schema", "")
        if schema != NODE_MANIFEST_SCHEMA:
            raise ValueError(
                f"manifest schema {schema!r} 不识别；期望 {NODE_MANIFEST_SCHEMA!r}"
            )
        node_id = (data.get("id") or "").strip()
        if not node_id:
            raise ValueError("manifest.id 不能为空 / cannot be empty")

        kind_str = (data.get("kind") or "CUSTOM").strip().lower()
        try:
            kind = NodeKind(kind_str)
        except ValueError:
            valid = [k.value for k in NodeKind]
            raise ValueError(
                f"manifest.kind {kind_str!r} 不在 NodeKind 枚举内；可选值 {valid}"
            )

        layer_raw = data.get("layer")
        layer: Optional[str] = None
        if layer_raw is not None:
            layer_norm = str(layer_raw).strip().upper()
            if layer_norm and layer_norm not in _VALID_LAYERS:
                raise ValueError(
                    f"manifest.layer {layer_raw!r} 不在 {_VALID_LAYERS} 内"
                )
            layer = layer_norm or None

        return cls(
            schema=schema,
            id=node_id,
            kind=kind,
            version=str(data.get("version", "1.0.0")),
            category=str(data.get("category", "misc")),
            layer=layer,
            display_name_key=str(data.get("display_name_key", "")),
            description_key=str(data.get("description_key", "")),
            icon=str(data.get("icon", "")),
            inputs=[_port_from_dict(p) for p in (data.get("inputs") or [])],
            outputs=[_port_from_dict(p) for p in (data.get("outputs") or [])],
            parameters=[_param_from_dict(p) for p in (data.get("parameters") or [])],
            is_trainer=bool(data.get("is_trainer", False)),
        )

    def to_dict(self) -> Dict[str, Any]:
        """导出为 dict（用于写 ``data/nodes_system.json`` 快照）。"""
        return {
            "schema": self.schema,
            "id": self.id,
            "kind": self.kind.value,
            "version": self.version,
            "category": self.category,
            "layer": self.layer,
            "display_name_key": self.display_name_key,
            "description_key": self.description_key,
            "icon": self.icon,
            "inputs": [_port_to_dict(p) for p in self.inputs],
            "outputs": [_port_to_dict(p) for p in self.outputs],
            "parameters": [_param_to_dict(p) for p in self.parameters],
            "is_trainer": self.is_trainer,
        }


def _port_from_dict(d: Dict[str, Any]) -> PortSpec:
    meta = d.get("meta")
    return PortSpec(
        name=str(d.get("name", "")),
        type=str(d.get("type", "any")),
        multi=bool(d.get("multi", False)),
        optional=bool(d.get("optional", False)),
        description=str(d.get("description", "")),
        meta=dict(meta) if isinstance(meta, dict) else None,
    )


def _port_to_dict(p: PortSpec) -> Dict[str, Any]:
    return {
        "name": p.name,
        "type": p.type,
        "multi": p.multi,
        "optional": p.optional,
        "description": p.description,
        "meta": dict(p.meta) if p.meta else None,
    }


def _param_from_dict(d: Dict[str, Any]) -> ParamSpec:
    choices = d.get("choices")
    meta = d.get("meta")
    widget = d.get("widget")
    return ParamSpec(
        key=str(d.get("key", "")),
        type=str(d.get("type", "string")),
        default=d.get("default"),
        choices=list(choices) if choices is not None else None,
        description=str(d.get("description", "")),
        widget=str(widget) if widget else None,
        meta=dict(meta) if isinstance(meta, dict) else None,
    )


def _param_to_dict(p: ParamSpec) -> Dict[str, Any]:
    return {
        "key": p.key,
        "type": p.type,
        "default": p.default,
        "choices": list(p.choices) if p.choices is not None else None,
        "description": p.description,
        "widget": p.widget,
        "meta": dict(p.meta) if p.meta else None,
    }


# =============================================================================
# BaseNode — 所有节点必须继承的 ABC
# =============================================================================

class BaseNode(ABC):
    """UnitPort 画布节点基类 / Base class for every UnitPort canvas node.

    生命周期 / Lifecycle:
        ``__init__(node_id, params)``         实例化（engine 在创建画布实例时调用）
        ``validate() -> list[str]``            静态校验（参数完整性，空列表=合法）
        ``execute(inputs) -> outputs``         运行时（engine 在工作流执行时调用）

    每个子类必须挂上类级 ``MANIFEST`` 属性（``NodeManifest`` 实例），由
    ``src/nodes/<id>/__init__.py`` 的 ``NODE_CLASSES`` 列表暴露给 discovery。
    """

    # 子类必须覆盖：与 manifest.toml 等价的 Python 表达
    MANIFEST: NodeManifest

    def __init__(self, node_id: str, params: Optional[Dict[str, Any]] = None) -> None:
        self.node_id = node_id
        self.params: Dict[str, Any] = dict(params or {})

    # ---- 静态查询（默认走 MANIFEST） ----

    @classmethod
    def manifest(cls) -> NodeManifest:
        return cls.MANIFEST

    @classmethod
    def schema_id(cls) -> str:
        return cls.MANIFEST.id

    @classmethod
    def kind(cls) -> NodeKind:
        return cls.MANIFEST.kind

    def get_input_ports(self) -> List[PortSpec]:
        return list(self.MANIFEST.inputs)

    def get_output_ports(self) -> List[PortSpec]:
        return list(self.MANIFEST.outputs)

    def get_parameter(self, key: str, default: Any = None) -> Any:
        if key in self.params:
            return self.params[key]
        for spec in self.MANIFEST.parameters:
            if spec.key == key:
                return spec.default
        return default

    def set_parameter(self, key: str, value: Any) -> None:
        self.params[key] = value

    # ---- 子类可选覆盖 ----

    def validate(self) -> List[str]:
        """静态校验，返回错误列表（空=合法）。

        默认实现：检查 ``ParamSpec.type == "enum"`` 时取值在 ``choices`` 内。
        子类可叠加额外规则（必填 inputs 等）。
        """
        errors: List[str] = []
        for spec in self.MANIFEST.parameters:
            if spec.type == "enum" and spec.choices is not None:
                value = self.get_parameter(spec.key)
                if value is not None and value not in spec.choices:
                    errors.append(
                        f"node={self.node_id} param={spec.key}={value!r} 不在 enum choices {spec.choices}"
                    )
        return errors

    # ---- 默认 execute（子类按需覆盖） ----

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Runtime execution.

        Default behaviour: passthrough — emits one payload dict per declared
        output port that bundles the node's resolved parameter values with
        every value supplied through ``inputs``. Suitable for the 25
        config-only nodes (Layer A/B/C/D except trainer/sink); trainer
        nodes (il_ppo_trainer, amp_trainer, train) and asset-resolving
        nodes (robot) override.

        See Stage 4 of ``plans/release-canva-todo-yaml-canva-demo-rele-iridescent-hellman.md``.
        """
        return self._passthrough_outputs(inputs)

    # ---- 默认 execute 辅助器 ----

    def _resolved_params(self) -> Dict[str, Any]:
        """All param values, defaulting to manifest defaults for keys the
        user has not overridden. Excludes hidden_params (UI gating)."""
        out: Dict[str, Any] = {}
        hidden = set(getattr(self, "hidden_params", set()) or set())
        for spec in self.MANIFEST.parameters:
            if spec.key in hidden:
                continue
            out[spec.key] = self.params.get(spec.key, spec.default)
        return out

    def _passthrough_outputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Emit one payload dict per output port.

        Payload = ``{node_id, schema_id, **resolved_params, _inputs:{...}}``.
        Single-output ports get the payload unwrapped one level; multi-output
        ports each receive an identical payload (downstream picks fields).
        """
        if not self.MANIFEST.outputs:
            return {}
        params = self._resolved_params()
        payload: Dict[str, Any] = {
            "node_id": self.node_id,
            "schema_id": self.MANIFEST.id,
            **params,
        }
        if isinstance(inputs, dict) and inputs:
            payload["_inputs"] = dict(inputs)
        return {port.name: payload for port in self.MANIFEST.outputs}


__all__ = [
    "BaseNode",
    "NodeKind",
    "NodeManifest",
    "PortSpec",
    "ParamSpec",
    "NODE_MANIFEST_SCHEMA",
]
