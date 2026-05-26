# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""ILPPOTrainerNode — Isaac Lab 统一 trainer (PPO / AMP_PPO).

DEMO 对应：``src/system/nodes/sys_nodes/training_nodes.py:ILPPOTrainerNode``.

行为契约：
    training_mode == "PPO"      — 标准 PPO；7 个 AMP 专属参数 + AMP 端口隐藏
    training_mode == "AMP_PPO"  — AMP-PPO；reference_motion_config /
                                  discriminator_config 端口必连；7 个 AMP
                                  hyperparams 写入 train_pipe

Phase 2 frontend skeleton — ``execute()`` 留接口给 Stage C 后端 wiring。
"""

from __future__ import annotations

from typing import Any, Dict, List

from application.compiler.nodes import (
    NODE_MANIFEST_SCHEMA,
    BaseNode,
    NodeKind,
    NodeManifest,
    ParamSpec,
    PortSpec,
)


# AMP 专属 ParamSpec keys — 抽出供 PPO 模式裁剪（与 DEMO `_AMP_ONLY_KEYS` 同集合）
_AMP_ONLY_KEYS = (
    "amp_reward_coef",
    "task_reward_lerp",
    "disc_grad_penalty",
    "disc_label_smoothing",
    "amp_replay_buffer_size",
    "num_preload_transitions",
    "disc_lr",
    "lerp_schedule",
    "lerp_schedule_json",
)

# Conditional 端口提示 (AMP_PPO 模式才显示，PPO 模式隐藏)
_CONDITIONAL_AMP_PORTS = ("reference_motion_config", "discriminator_config")


class ILPPOTrainerNode(BaseNode):
    """Layer IL — Isaac Lab unified PPO / AMP-PPO trainer."""

    # ---- DEMO 行为契约（class-level 提示，画布 UI 读取） ----
    TRAINING_MODE_CHOICES = ("PPO", "AMP_PPO")
    _OPTIONAL_INPUTS: set = {
        "checkpoint", "reference_motion_config", "discriminator_config",
    }
    _CONDITIONAL_PORTS: dict = {
        "reference_motion_config": {"key": "training_mode", "op": "==", "value": "AMP_PPO"},
        "discriminator_config":    {"key": "training_mode", "op": "==", "value": "AMP_PPO"},
    }
    _AMP_ONLY_KEYS: tuple = _AMP_ONLY_KEYS

    MANIFEST = NodeManifest(
        schema=NODE_MANIFEST_SCHEMA,
        id="il_ppo_trainer",
        kind=NodeKind.CUSTOM,
        version="1.0.0",
        category="training",
        layer="IL",
        display_name_key="node.il_ppo_trainer.display",
        description_key="node.il_ppo_trainer.desc",
        icon="il_trainer.svg",
        # mission_panel 训练配置卡靠 is_trainer 在画布上动态发现承载训练超参的节点。
        is_trainer=True,
        inputs=[
            PortSpec(name="policy", type="policy"),
            PortSpec(name="command_pipe", type="command_pipe",
                     description="Aggregated command + per-item rewards stream (from training_motion)"),
            PortSpec(name="termination_config", type="termination_config"),
            PortSpec(name="domain_rand_config", type="domain_rand_config"),
            PortSpec(name="obs_vector", type="obs_vector"),
            PortSpec(name="scene_pipe", type="scene_pipe"),
            PortSpec(name="checkpoint", type="checkpoint", optional=True,
                     description="Start Point resume/warm-start checkpoint (optional)"),
            PortSpec(name="reference_motion_config", type="reference_motion_config",
                     optional=True,
                     description="Required in AMP_PPO mode (auto-hidden in PPO mode)"),
            PortSpec(name="discriminator_config", type="discriminator_config",
                     optional=True,
                     description="Required in AMP_PPO mode (auto-hidden in PPO mode)"),
        ],
        outputs=[
            PortSpec(name="train_pipe", type="train_pipe"),
            PortSpec(name="training_metrics", type="dict"),
        ],
        parameters=[
            # ---- Mode + scale ----
            ParamSpec(key="training_mode", type="enum", default="PPO",
                      choices=["PPO", "AMP_PPO"],
                      description="训练模式 / Training mode",
                      meta={"mode_gate": True, "pin_on_collapse": True}),
            ParamSpec(key="num_envs", type="int", default=4096, widget="index",
                      description="并行 env / Parallel envs",
                      meta={"min": 1, "max": 65536,
                            "mission_expose": True,
                            "mission_label_i18n": "mission.train.field.num_envs",
                            "pin_on_collapse": True}),
            # ---- 16 shared PPO hyperparams ----
            ParamSpec(key="max_iterations", type="int", default=1500, widget="index",
                      description="最大迭代",
                      meta={"min": 1, "max": 1_000_000,
                            "mission_expose": True,
                            "mission_label_i18n": "mission.train.field.max_iterations",
                            "pin_on_collapse": True}),
            ParamSpec(key="num_steps_per_env", type="int", default=24, widget="index",
                      description="每 env 步数",
                      meta={"min": 1, "max": 1024, "pin_on_collapse": True}),
            ParamSpec(key="num_learning_epochs", type="int", default=5, widget="index",
                      description="PPO 学习 epoch",
                      meta={"min": 1, "max": 64, "pin_on_collapse": True}),
            ParamSpec(key="num_minibatches", type="int", default=4, widget="index",
                      description="PPO mini-batch 数",
                      meta={"min": 1, "max": 64, "pin_on_collapse": True}),
            ParamSpec(key="learning_rate", type="float", default=0.001,
                      description="学习率",
                      meta={"mission_expose": True,
                            "mission_label_i18n": "mission.train.field.lr",
                            "pin_on_collapse": True}),
            ParamSpec(key="discount_factor", type="float", default=0.99, widget="range",
                      description="折扣因子 γ",
                      meta={"min": 0.0, "max": 1.0, "step": 0.001,
                            "mission_expose": True,
                            "mission_label_i18n": "mission.train.field.gamma"}),
            ParamSpec(key="gae_lambda", type="float", default=0.95, widget="range",
                      description="GAE λ",
                      meta={"min": 0.0, "max": 1.0, "step": 0.001}),
            ParamSpec(key="clip_param", type="float", default=0.2, widget="range",
                      description="PPO clip ε",
                      meta={"min": 0.0, "max": 1.0, "step": 0.01}),
            ParamSpec(key="entropy_coef", type="float", default=0.01,
                      description="熵系数",
                      meta={"mission_expose": True,
                            "mission_label_i18n": "mission.train.field.ent_coef"}),
            ParamSpec(key="value_loss_coef", type="float", default=1.0,
                      description="value loss 权重"),
            ParamSpec(key="max_grad_norm", type="float", default=1.0,
                      description="梯度裁剪上限"),
            ParamSpec(key="schedule", type="enum", default="adaptive",
                      choices=["adaptive", "fixed"],
                      description="学习率调度 / LR schedule"),
            ParamSpec(key="desired_kl", type="float", default=0.01,
                      description="adaptive KL 目标"),
            ParamSpec(key="save_interval", type="int", default=100, widget="index",
                      description="checkpoint 保存间隔",
                      meta={"min": 1, "max": 100_000}),
            ParamSpec(key="seed", type="int", default=42, widget="index",
                      description="随机种子",
                      meta={"min": 0, "max": 100_000_000,
                            "mission_expose": True,
                            "mission_label_i18n": "mission.train.field.seed"}),
            ParamSpec(key="headless", type="bool", default=True,
                      description="无头模式（仅在生产训练用）"),
            # ---- 7 AMP-only hyperparams (visible only when training_mode == AMP_PPO) ----
            ParamSpec(key="amp_reward_coef", type="float", default=2.0,
                      description="[AMP] discriminator 奖励权重",
                      meta={"conditional_on": {"key": "training_mode", "op": "==", "value": "AMP_PPO"}}),
            ParamSpec(key="task_reward_lerp", type="float", default=0.5, widget="range",
                      description="[AMP] task vs disc 奖励 lerp",
                      meta={"conditional_on": {"key": "training_mode", "op": "==", "value": "AMP_PPO"},
                            "min": 0.0, "max": 1.0, "step": 0.01}),
            ParamSpec(key="disc_grad_penalty", type="float", default=10.0,
                      description="[AMP] discriminator 梯度惩罚",
                      meta={"conditional_on": {"key": "training_mode", "op": "==", "value": "AMP_PPO"}}),
            ParamSpec(key="disc_label_smoothing", type="float", default=0.9, widget="range",
                      description="[AMP] discriminator 标签平滑",
                      meta={"conditional_on": {"key": "training_mode", "op": "==", "value": "AMP_PPO"},
                            "min": 0.0, "max": 1.0, "step": 0.01}),
            ParamSpec(key="amp_replay_buffer_size", type="int", default=1_000_000,
                      widget="buffer_size",
                      description="[AMP] replay buffer 容量",
                      meta={"conditional_on": {"key": "training_mode", "op": "==", "value": "AMP_PPO"},
                            "min": 1000, "max": 100_000_000}),
            ParamSpec(key="num_preload_transitions", type="int", default=2_000_000,
                      widget="index",
                      description="[AMP] 预加载 transition 数",
                      meta={"conditional_on": {"key": "training_mode", "op": "==", "value": "AMP_PPO"},
                            "min": 0, "max": 100_000_000}),
            ParamSpec(key="disc_lr", type="float", default=0.0001,
                      description="[AMP] discriminator 学习率",
                      meta={"conditional_on": {"key": "training_mode", "op": "==", "value": "AMP_PPO"}}),
            # ---- RSI-AMP lerp schedule ----
            ParamSpec(key="lerp_schedule", type="enum", default="none",
                      choices=["none", "slow_anneal", "fast_anneal",
                               "warmup_then_anneal", "custom"],
                      description="[AMP] lerp 预设",
                      meta={"conditional_on": {"key": "training_mode", "op": "==", "value": "AMP_PPO"}}),
            ParamSpec(key="lerp_schedule_json", type="string", default="",
                      widget="code",
                      description="[AMP] custom lerp 调度（JSON）",
                      meta={"conditional_on": {"key": "training_mode", "op": "==", "value": "AMP_PPO"},
                            "language": "json"}),
            # ---- Entropy schedule ----
            ParamSpec(key="entropy_schedule_enabled", type="bool", default=False,
                      description="启用熵系数退火"),
            ParamSpec(key="entropy_schedule_start", type="float", default=0.01,
                      description="熵起始值"),
            ParamSpec(key="entropy_schedule_end", type="float", default=0.003,
                      description="熵终值"),
            ParamSpec(key="entropy_schedule_ramp_iters", type="int", default=1500,
                      widget="index",
                      description="熵退火持续 iter 数",
                      meta={"min": 1, "max": 1_000_000}),
        ],
    )

    # ---- 画布 UI 钩子（动态隐藏 PPO 模式下的 AMP 端口与参数） ----

    @property
    def hidden_ports(self) -> set:
        mode = str(self.get_parameter("training_mode", "PPO") or "PPO").strip()
        if mode != "AMP_PPO":
            return set(_CONDITIONAL_AMP_PORTS)
        return set()

    @property
    def hidden_params(self) -> set:
        mode = str(self.get_parameter("training_mode", "PPO") or "PPO").strip()
        if mode != "AMP_PPO":
            return set(_AMP_ONLY_KEYS)
        return set()

    # ---- 校验 ----

    def validate(self) -> List[str]:
        errors = super().validate()
        mode = str(self.get_parameter("training_mode", "PPO") or "PPO").strip()
        if mode not in self.TRAINING_MODE_CHOICES:
            errors.append(
                f"node={self.node_id} training_mode={mode!r} 不在 {self.TRAINING_MODE_CHOICES}"
            )
        return errors

    # ---- 运行时 ----

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Refuse to launch from canvas IR execution — Play button only.

        Canvas-IR-driven execution can't supply the source canvas dict
        that ``IsaacLabTrainingTask`` needs to compile ``UnitPortEnvCfg``,
        so this path used to silently fall back to Isaac Lab's stock
        ``Isaac-Velocity-Flat-Unitree-Go2-v0`` task and ignore the
        user's canvas entirely (see 2026-05-10 debug). The single legal
        IL training entry point is the top Play button
        (``MainWindow._on_start_training`` → ``submit_canvas_training``).
        Mission Control's Start button forwards to the same Play click.
        """
        raise RuntimeError(
            f"[{self.MANIFEST.id}] node.execute() does not launch IL "
            f"training. Use the top Play button — it routes through "
            f"submit_canvas_training, which carries the canvas dict "
            f"required to compile UnitPortEnvCfg. The previous "
            f"silent-fallback to Isaac Lab's stock task was removed "
            f"because the user couldn't tell which env was actually "
            f"running."
        )
