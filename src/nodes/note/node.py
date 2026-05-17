"""NoteNode — Canvas annotation card.

不参与 IR lowering / 编译执行；编译器对未知 schema_id 的 dispatch 是静默跳过。
Non-executing — compiler's unknown-schema dispatch silently skips this node.
"""

from __future__ import annotations

from application.compiler.nodes import (
    NODE_MANIFEST_SCHEMA,
    BaseNode,
    NodeKind,
    NodeManifest,
    ParamSpec,
)


class NoteNode(BaseNode):
    """Annotation node — title + multi-line body, no I/O, no execution."""

    MANIFEST = NodeManifest(
        schema=NODE_MANIFEST_SCHEMA,
        id="note",
        kind=NodeKind.CUSTOM,
        version="1.0.0",
        category="annotation",
        layer="TOOLS",
        display_name_key="node.note.display",
        description_key="node.note.desc",
        icon="note.svg",
        inputs=[],
        outputs=[],
        parameters=[
            ParamSpec(key="title", type="string", default="Note",
                      description="Note 标题"),
            ParamSpec(key="body", type="string", default="", widget="code",
                      description="Note 正文（多行）",
                      meta={"language": "text"}),
        ],
    )
