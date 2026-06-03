# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""NoteNode — Canvas annotation card.

不参与 IR lowering / 编译执行；编译器对未知 schema_id 的 dispatch 是静默跳过。
Non-executing — compiler's unknown-schema dispatch silently skips this node.
"""

from __future__ import annotations

from application.compiler.nodes import (
    manifest_from_toml,
    NODE_MANIFEST_SCHEMA,
    BaseNode,
    NodeKind,
    NodeManifest,
    ParamSpec,
)


class NoteNode(BaseNode):
    """Annotation node — title + multi-line body, no I/O, no execution."""

    MANIFEST = manifest_from_toml(__file__)
