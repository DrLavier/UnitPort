# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""src.nodes.task_config — Layer A."""

from .node import TaskConfigNode


NODE_CLASSES = [TaskConfigNode]


__all__ = ["NODE_CLASSES", "TaskConfigNode"]
