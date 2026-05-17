"""DiscriminatorNode — 独立 AMP 判别器配置 / Standalone AMP discriminator.

DEMO 对应：``src/system/nodes/sys_nodes/training_nodes.py:DiscriminatorNode``.

Layer IL / category=config. 集中所有 adversarial motion prior 参数（先前散落
在 ILPolicyNetworkNode / ILPPOTrainerNode / TrainingMotionNode 三处），
让用户在一处推理 Rewards vs. Discriminator 的信号重叠。

Wiring::

    [TrainingMotion] ── reference_motion_config ──► [Discriminator]
                                                        │
                            discriminator_config ───────┘──► [IL Trainer]

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


class DiscriminatorNode(BaseNode):
    """Layer IL — Standalone AMP discriminator configuration."""

    MANIFEST = NodeManifest(
        schema=NODE_MANIFEST_SCHEMA,
        id="discriminator",
        kind=NodeKind.CUSTOM,
        version="1.0.0",
        category="config",
        layer="IL",
        display_name_key="node.discriminator.display",
        description_key="node.discriminator.desc",
        icon="discriminator.svg",
        inputs=[
            PortSpec(name="reference_motion_config", type="reference_motion_config",
                     description="Motion clip metadata; forwarded into discriminator_config"),
        ],
        outputs=[
            PortSpec(name="discriminator_config", type="discriminator_config"),
        ],
        parameters=[
            # ---- Source code (P0.3) ----
            ParamSpec(
                key="disc_modules", type="json",
                default='{"disc_forward": 1.0, "disc_compute_grad_pen": 1.0, "disc_predict_reward": 1.0}',
                widget="code",
                description="判别器模块字典 (JSON {key:1.0}) / Active discriminator functions",
                meta={"language": "json"},
            ),
            # ---- Network architecture ----
            ParamSpec(
                key="disc_hidden_dims", type="json", default="[1024, 512]",
                widget="code",
                description="判别器 MLP 隐藏维度 (JSON list) / Discriminator hidden dims",
                meta={"language": "json"},
            ),
            # ---- Reward shaping ----
            ParamSpec(
                key="amp_reward_coef", type="float", default=2.0,
                description="AMP discriminator 奖励权重 / AMP reward coefficient",
            ),
            ParamSpec(
                key="task_reward_lerp", type="float", default=0.5, widget="range",
                description="task vs disc 奖励 lerp / Task vs discriminator reward lerp",
                meta={"min": 0.0, "max": 1.0, "step": 0.01},
            ),
            ParamSpec(
                key="lerp_schedule", type="enum", default="none",
                choices=["none", "slow_anneal", "fast_anneal",
                         "warmup_then_anneal", "custom"],
                description="lerp 预设调度 / Lerp schedule preset",
            ),
            ParamSpec(
                key="lerp_schedule_json", type="string", default="",
                widget="code",
                description="custom lerp 调度 (JSON) / Custom lerp schedule",
                meta={"language": "json"},
            ),
            # ---- Training hyperparams ----
            ParamSpec(
                key="disc_lr", type="float", default=0.0001,
                description="判别器学习率 / Discriminator learning rate",
            ),
            ParamSpec(
                key="disc_grad_penalty", type="float", default=10.0,
                description="梯度惩罚 / Discriminator gradient penalty",
            ),
            ParamSpec(
                key="disc_label_smoothing", type="float", default=0.9, widget="range",
                description="判别器标签平滑 / Discriminator label smoothing",
                meta={"min": 0.0, "max": 1.0, "step": 0.01},
            ),
            # ---- Replay buffer ----
            ParamSpec(
                key="amp_replay_buffer_size", type="int", default=1_000_000,
                widget="index",
                description="AMP replay buffer 容量 / AMP replay buffer size",
                meta={"min": 1000, "max": 100_000_000},
            ),
            ParamSpec(
                key="num_preload_transitions", type="int", default=200_000,
                widget="index",
                description="预加载 transition 数 / Number of preload transitions",
                meta={"min": 0, "max": 100_000_000},
            ),
            # ---- Observation fields ----
            ParamSpec(
                key="amp_obs_fields", type="string", default="",
                description="AMP 观测字段（逗号分隔） / AMP observation fields (comma-separated)",
            ),
            ParamSpec(
                key="auto_inject_ref_obs", type="bool", default=True,
                description="自动注入参考观测 / Auto-inject reference observations",
            ),
            # ---- Stability circuit-breakers (P2) ----
            ParamSpec(
                key="disc_logit_clamp_max", type="float", default=4.0,
                description="discriminator logit clamp 上限 / Discriminator logit clamp max",
            ),
            ParamSpec(
                key="reward_clamp_per_step", type="float", default=50.0,
                description="单步 reward clamp 上限 / Per-step reward clamp",
            ),
            ParamSpec(
                key="policy_std_clamp_max", type="float", default=1.5,
                description="policy std clamp 上限 / Policy std clamp max",
            ),
        ],
    )