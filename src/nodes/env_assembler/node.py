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

    MANIFEST = NodeManifest(
        schema=NODE_MANIFEST_SCHEMA,
        id="env_assembler",
        kind=NodeKind.CUSTOM,
        version="1.0.0",
        category="config",
        layer="B",
        display_name_key="node.env_assembler.display",
        description_key="node.env_assembler.desc",
        icon="env_assembler.svg",
        inputs=[
            # Unified 6-section pipes
            PortSpec(name="actor_pipe", type="actor_pipe", optional=True,
                     description="§1 Actor block (optional in legacy SB3 canvases)"),
            PortSpec(name="scene_pipe", type="scene_pipe", optional=True,
                     description="§2 Scene from PlayGroundSetting (optional)"),
            PortSpec(name="command_pipe", type="command_pipe", optional=True,
                     description="§4 Training Commands (optional)"),
            # SB3 task-layer configs
            PortSpec(name="physics_config", type="physics_config"),
            PortSpec(name="task_config", type="task_config"),
            PortSpec(name="obs_action_config", type="obs_action_config"),
            PortSpec(name="domain_rand_config", type="domain_rand_config", optional=True),
            PortSpec(name="reference_motion_config", type="reference_motion_config",
                     optional=True),
        ],
        outputs=[
            PortSpec(name="env_config", type="env_config",
                     description="Aggregated SB3 env config consumed by TrainNode"),
        ],
        parameters=[
            # VecEnv settings
            ParamSpec(key="n_envs", type="int", default=8, widget="index",
                      description="并行 env 数 (1 不可接受) / Parallel envs",
                      meta={"min": 1, "max": 4096}),
            ParamSpec(key="vec_type", type="enum", default="subproc",
                      choices=["dummy", "subproc"],
                      description="向量化类型 / Vectorisation type"),
            # Wrapper stack
            ParamSpec(key="obs_normalize", type="bool", default=True,
                      description="观测归一化 / Observation normalisation"),
            ParamSpec(key="reward_normalize", type="bool", default=False,
                      description="奖励归一化 / Reward normalisation"),
            ParamSpec(key="clip_obs", type="float", default=10.0, widget="range",
                      description="观测裁剪范围 / Observation clip",
                      meta={"min": 0.0, "max": 1000.0, "step": 0.1}),
            ParamSpec(key="clip_reward", type="float", default=10.0, widget="range",
                      description="奖励裁剪范围 / Reward clip",
                      meta={"min": 0.0, "max": 1000.0, "step": 0.1}),
            ParamSpec(key="action_clip_range", type="float", default=1.0, widget="range",
                      description="ActionClip 包装范围 / ActionClip wrapper range [-x, x]",
                      meta={"min": 0.0, "max": 100.0, "step": 0.01}),
            ParamSpec(key="enable_monitor", type="bool", default=True,
                      description="启用 Monitor 包装 / Monitor wrapper"),
            ParamSpec(key="time_limit_override", type="int", default=0, widget="index",
                      description="TimeLimit 覆盖 (0 = 继承 PhysicsConfig) / TimeLimit override",
                      meta={"min": 0, "max": 100_000_000}),
            ParamSpec(key="eval_disable_rand", type="bool", default=True,
                      description="评估时禁用域随机化 / Disable DR during eval"),
        ],
    )