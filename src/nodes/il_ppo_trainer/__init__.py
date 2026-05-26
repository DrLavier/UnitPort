# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""src.nodes.il_ppo_trainer — Layer IL."""

from .node import ILPPOTrainerNode


NODE_CLASSES = [ILPPOTrainerNode]


__all__ = ["NODE_CLASSES", "ILPPOTrainerNode"]
