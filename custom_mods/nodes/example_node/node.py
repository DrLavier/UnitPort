"""ExampleNode — 三方节点起步样例 / Community node starter.

最小三件套样例：声明一个 ``message`` 字符串参数。``BaseNode.execute`` 默认
passthrough 已足够，无需重写。
"""

from __future__ import annotations

from application.compiler.nodes import (
    NODE_MANIFEST_SCHEMA,
    BaseNode,
    NodeKind,
    NodeManifest,
    ParamSpec,
)


class ExampleNode(BaseNode):
    MANIFEST = NodeManifest(
        schema=NODE_MANIFEST_SCHEMA,
        id="example.print_hello",
        kind=NodeKind.CUSTOM,
        version="0.1.0",
        category="custom",
        display_name_key="node.example.print_hello.display",
        description_key="node.example.print_hello.desc",
        parameters=[
            ParamSpec(
                key="message",
                type="string",
                default="hello from example_node",
                description="要打印的消息 / Message to print",
            ),
        ],
    )
