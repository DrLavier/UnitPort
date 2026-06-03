# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""EnvAssemblerNode — Layer B SB3 aggregation point.

DEMO 对应：``src/system/nodes/sys_nodes/training_nodes.py:EnvAssemblerNode``.

Merges §1 Actor block (actor_pipe), §2 Scene (scene_pipe), §4 Training
Commands (command_pipe), plus the SB3-only task-layer configs
(physics / task / obs&action / domain_rand / ref_motion / init_pose) into a
single ``env_config`` dict consumed by TrainNode.

scene_pipe / actor_pipe / command_pipe are declared optional at the node
level because legacy SB3 canvases may not have them wired. The C6
cross-section validator still requires PlayGroundSetting to terminate
somewhere on the compile pipeline — either at a trainer (IL) or at
EnvAssembler.scene_pipe (SB3).

Phase 2 frontend skeleton — ``execute()`` 留接口给 Stage C 后端 wiring。
"""

from __future__ import annotations

from typing import Any, Dict

from application.compiler.nodes import (
    manifest_from_toml,
    NODE_MANIFEST_SCHEMA,
    BaseNode,
    NodeKind,
    NodeManifest,
    ParamSpec,
    PortSpec,
)


class EnvAssemblerNode(BaseNode):
    """Layer B — SB3 environment assembler node."""

    # ---- DEMO 行为契约（class-level 提示，画布 UI 读取） ----
    # Optional inputs: 任何有 sane default 的端口都允许悬空。
    _OPTIONAL_INPUTS: set = {
        "domain_rand_config",
        "reference_motion_config",
        "actor_pipe",
        "scene_pipe",
        "command_pipe",
    }

    MANIFEST = manifest_from_toml(__file__)