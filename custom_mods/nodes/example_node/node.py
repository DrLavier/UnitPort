"""ExampleNode — 三方节点起步样例 / Community node starter.

打印 ``message`` 参数到日志，演示最小三件套。
"""

from __future__ import annotations

from typing import Any, Dict

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

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError(
            f"{self.MANIFEST.id}.execute pending backend wiring"
        )
