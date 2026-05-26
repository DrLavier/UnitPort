# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""src.nodes.multigated_reward — Layer A."""

from .node import MultiGatedRewardNode


NODE_CLASSES = [MultiGatedRewardNode]


__all__ = ["NODE_CLASSES", "MultiGatedRewardNode"]
