# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""src.nodes.note — Annotation node (non-executing canvas comment card)."""

from .node import NoteNode


NODE_CLASSES = [NoteNode]


__all__ = ["NODE_CLASSES", "NoteNode"]
