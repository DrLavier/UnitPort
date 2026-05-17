"""unitport_sdk.canvas.node_builder — 节点构造糖 / Node convenience metaclass.

借鉴 ComfyUI 的 ``INPUT_TYPES`` 风格，让节点作者用紧凑 dict 写完 manifest，
避免每个节点都展开 ``NodeManifest(inputs=[PortSpec(...), ...], ...)`` 样板。

用法对比：

    # 老写法（完全合法，不强求迁移）
    class TaskConfigNode(BaseNode):
        MANIFEST = NodeManifest(
            schema=NODE_MANIFEST_SCHEMA,
            id="task_config",
            kind=NodeKind.CUSTOM,
            version="1.0.0",
            layer="A",
            inputs=[
                PortSpec(name="total_reward", type="total_reward", optional=True),
                PortSpec(name="termination_config", type="termination_config", optional=True),
            ],
            outputs=[PortSpec(name="task_config", type="task_config")],
            parameters=[
                ParamSpec(key="task_type", type="enum", default="velocity_tracking",
                          choices=["velocity_tracking"]),
                ParamSpec(key="target_lin_vel_x", type="float", default=0.5),
                ...
            ],
        )

        def execute(self, inputs):
            raise NotImplementedError(...)

    # 新写法（NodeBuilder 自动转 MANIFEST）
    class TaskConfigNode(BaseNode, metaclass=NodeBuilder):
        LAYER = "A"
        INPUTS = {
            "total_reward":       ("total_reward",       {"optional": True}),
            "termination_config": ("termination_config", {"optional": True}),
        }
        OUTPUTS = {
            "task_config": ("task_config",),
        }
        PARAMETERS = {
            "task_type": ("enum", "velocity_tracking", {"choices": ["velocity_tracking"]}),
            "target_lin_vel_x": ("float", 0.5),
            ...
        }

        def execute(self, inputs):
            raise NotImplementedError(...)

值元组协议：

    INPUTS / OUTPUTS:
        "<port_name>": (type_str, opts_dict_optional)
        opts 可含: optional / multi / description

    PARAMETERS:
        "<param_key>": (type_str, default, opts_dict_optional)
        opts 可含: widget / meta / choices / description

类级别还可设：
    NODE_ID         默认从类名 PascalCase → snake_case 推导（去掉 Node 后缀）
    LAYER           "A" / "B" / "C" / "D" / "IL" / None
    VERSION         默认 "1.0.0"
    CATEGORY        默认 "misc"
    DISPLAY_NAME_KEY / DESCRIPTION_KEY / ICON
    SCHEMA          默认走 ``NODE_MANIFEST_SCHEMA``

老写法仍合法：手写 ``MANIFEST = NodeManifest(...)`` 时本 metaclass 不覆盖。
"""

from __future__ import annotations

import re
from abc import ABCMeta
from typing import Any, Dict, Optional, Tuple


# =============================================================================
# 工具：类名 → snake_case node_id
# =============================================================================

_CAMEL_RE = re.compile(r"(?<!^)(?=[A-Z])")


def _classname_to_node_id(name: str) -> str:
    """``TaskConfigNode`` → ``task_config``；``AMPTrainerNode`` → ``amp_trainer``."""
    base = name
    if base.endswith("Node"):
        base = base[:-4]
    # 处理连续大写：AMPTrainer → AMP_Trainer → amp_trainer
    base = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", base)
    base = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", base)
    return base.lower().strip("_")


# =============================================================================
# Tuple → spec 转换
# =============================================================================

def _normalize_port_tuple(name: str, value: Any) -> "PortSpec":  # noqa: F821 (forward ref)
    from application.compiler.nodes import PortSpec

    if isinstance(value, PortSpec):
        return value
    if not isinstance(value, tuple):
        raise TypeError(
            f"port {name!r} 必须是 (type, opts?) 元组或 PortSpec；got {type(value).__name__}"
        )
    if len(value) == 1:
        type_str = value[0]
        opts: Dict[str, Any] = {}
    elif len(value) == 2:
        type_str, opts = value
        if opts is None:
            opts = {}
        if not isinstance(opts, dict):
            raise TypeError(f"port {name!r} 第二元素必须是 dict")
    else:
        raise ValueError(f"port {name!r} 元组长度应为 1 或 2；got {len(value)}")
    return PortSpec(
        name=name,
        type=str(type_str),
        multi=bool(opts.get("multi", False)),
        optional=bool(opts.get("optional", False)),
        description=str(opts.get("description", "")),
    )


def _normalize_param_tuple(key: str, value: Any) -> "ParamSpec":  # noqa: F821
    from application.compiler.nodes import ParamSpec

    if isinstance(value, ParamSpec):
        return value
    if not isinstance(value, tuple):
        raise TypeError(
            f"param {key!r} 必须是 (type, default, opts?) 元组或 ParamSpec；"
            f"got {type(value).__name__}"
        )
    if len(value) == 2:
        type_str, default = value
        opts: Dict[str, Any] = {}
    elif len(value) == 3:
        type_str, default, opts = value
        if opts is None:
            opts = {}
        if not isinstance(opts, dict):
            raise TypeError(f"param {key!r} 第三元素必须是 dict")
    else:
        raise ValueError(
            f"param {key!r} 元组长度应为 2 或 3；got {len(value)}"
        )
    # widget / meta / choices / description 都从 opts 拿；其余 key 视为 meta
    widget = opts.get("widget")
    description = str(opts.get("description", ""))
    choices = opts.get("choices")
    if choices is not None:
        choices = list(choices)
    # meta：剩下的 keys（除 widget/choices/description 之外的）会原样落进 meta
    explicit_meta = opts.get("meta")
    if explicit_meta is not None:
        meta = dict(explicit_meta)
    else:
        meta = {
            k: v for k, v in opts.items()
            if k not in ("widget", "choices", "description", "meta")
        }
        meta = meta or None
    return ParamSpec(
        key=key,
        type=str(type_str),
        default=default,
        choices=choices,
        description=description,
        widget=widget,
        meta=meta,
    )


# =============================================================================
# Metaclass
# =============================================================================

class NodeBuilder(ABCMeta):
    """节点 metaclass / Node convenience metaclass.

    在类创建时把 ``INPUTS`` / ``OUTPUTS`` / ``PARAMETERS`` dict 转成正经
    ``NodeManifest`` 写到 ``cls.MANIFEST``。已有 ``MANIFEST`` 的类不动，保证老
    写法依然合法。

    继承 ``ABCMeta`` 是因为 ``BaseNode`` 是 ABC，metaclass 必须兼容。
    """

    def __new__(
        mcs,
        name: str,
        bases: Tuple[type, ...],
        ns: Dict[str, Any],
    ) -> type:
        cls = super().__new__(mcs, name, bases, ns)

        # 已显式定义 MANIFEST → 不覆盖（含 BaseNode 自身 + 任何手写 MANIFEST 的子类）
        if "MANIFEST" in ns and ns["MANIFEST"] is not None:
            return cls

        # 没声明任何 INPUTS/OUTPUTS/PARAMETERS → 视作中间抽象类，不构 MANIFEST
        has_dsl = any(k in ns for k in ("INPUTS", "OUTPUTS", "PARAMETERS"))
        if not has_dsl:
            return cls

        # 延迟 import，避免 SDK ↔ application 循环
        from application.compiler.nodes import (
            NODE_MANIFEST_SCHEMA,
            NodeKind,
            NodeManifest,
        )

        node_id: str = ns.get("NODE_ID") or _classname_to_node_id(name)
        layer: Optional[str] = ns.get("LAYER")
        version: str = str(ns.get("VERSION", "1.0.0"))
        category: str = str(ns.get("CATEGORY", "misc"))
        display_name_key: str = str(ns.get("DISPLAY_NAME_KEY", f"node.{node_id}.display"))
        description_key: str = str(ns.get("DESCRIPTION_KEY", f"node.{node_id}.desc"))
        icon: str = str(ns.get("ICON", ""))
        schema: str = str(ns.get("SCHEMA", NODE_MANIFEST_SCHEMA))

        inputs_dict: Dict[str, Any] = ns.get("INPUTS") or {}
        outputs_dict: Dict[str, Any] = ns.get("OUTPUTS") or {}
        params_dict: Dict[str, Any] = ns.get("PARAMETERS") or {}

        try:
            inputs = [_normalize_port_tuple(n, v) for n, v in inputs_dict.items()]
            outputs = [_normalize_port_tuple(n, v) for n, v in outputs_dict.items()]
            parameters = [_normalize_param_tuple(k, v) for k, v in params_dict.items()]
        except (TypeError, ValueError) as e:
            raise TypeError(
                f"NodeBuilder 构造 {name} 的 MANIFEST 失败：{e}"
            ) from e

        cls.MANIFEST = NodeManifest(  # type: ignore[attr-defined]
            schema=schema,
            id=node_id,
            kind=NodeKind.CUSTOM,
            version=version,
            category=category,
            layer=layer,
            display_name_key=display_name_key,
            description_key=description_key,
            icon=icon,
            inputs=inputs,
            outputs=outputs,
            parameters=parameters,
        )
        return cls


__all__ = ["NodeBuilder"]
