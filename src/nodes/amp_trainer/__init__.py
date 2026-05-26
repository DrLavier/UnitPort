# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""src.nodes.amp_trainer — Layer IL."""

from .node import AMPTrainerNode


NODE_CLASSES = [AMPTrainerNode]


__all__ = ["NODE_CLASSES", "AMPTrainerNode"]
