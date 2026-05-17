"""AlgorithmConfigNode — RL 算法 + 超参 (Layer C).

DEMO 对应：``src/system/nodes/sys_nodes/training_nodes.py:AlgorithmConfigNode``.

行为契约：
    端口：
        checkpoint   (optional) — 来自 BaseAssetNode；scratch 时可悬空
        total_steps  (optional) — 来自 MultiGatedRewardNode；NodeRow 合并到
                                  total_timesteps 行（_HIDDEN_PORTS 隐藏端口）

    算法 gating（DEMO ``_PPO_ONLY_KEYS`` / ``_OFF_POLICY_ONLY_KEYS``）：
        algorithm == "PPO":
            visible: gae_lambda / n_steps / n_epochs / lr_schedule
            hidden:  gradient_steps / tau / target_entropy /
                     buffer_size / buffer_size_mode / learning_starts /
                     use_per / per_alpha / per_beta
        algorithm in ("SAC", "TD3"):
            visible: 上述 off-policy 集合（per_alpha / per_beta 仅 use_per == true）
            hidden:  gae_lambda / n_steps / n_epochs / lr_schedule

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


# ---- DEMO 行为契约（class-level 常量） ----

# total_steps 端口在 NodeRow 中合并到 total_timesteps 行，画布上隐藏
_HIDDEN_PORTS = ("total_steps",)

# PPO 专属参数 keys（off-policy 模式下隐藏）
_PPO_ONLY_KEYS = (
    "gae_lambda", "n_steps", "n_epochs", "lr_schedule",
)

# Off-policy (SAC / TD3) 专属参数 keys（PPO 模式下隐藏）
_OFF_POLICY_ONLY_KEYS = (
    "gradient_steps", "tau", "target_entropy",
    "buffer_size", "buffer_size_mode", "learning_starts",
    "use_per", "per_alpha", "per_beta",
)

# PER 专属 keys（仅 off-policy + use_per == true 时可见）
_PER_ONLY_KEYS = ("per_alpha", "per_beta")


class AlgorithmConfigNode(BaseNode):
    """Layer C — RL algorithm and hyperparameter configuration."""

    # ---- DEMO 行为契约（class-level 提示，画布 UI 读取） ----
    _OPTIONAL_INPUTS: set = {"checkpoint", "total_steps"}
    _HIDDEN_PORTS: tuple = _HIDDEN_PORTS
    _PPO_ONLY_KEYS: tuple = _PPO_ONLY_KEYS
    _OFF_POLICY_ONLY_KEYS: tuple = _OFF_POLICY_ONLY_KEYS

    ALGORITHM_CHOICES = ("PPO", "SAC", "TD3")

    MANIFEST = NodeManifest(
        schema=NODE_MANIFEST_SCHEMA,
        id="algorithm_config",
        kind=NodeKind.CUSTOM,
        version="1.0.0",
        category="config",
        layer="C",
        display_name_key="node.algorithm_config.display",
        description_key="node.algorithm_config.desc",
        icon="algorithm.svg",
        # mission_panel 训练配置卡靠 is_trainer + ParamSpec.meta["mission_expose"]
        # 在画布上动态发现承载训练超参的节点。
        is_trainer=True,
        inputs=[
            PortSpec(name="checkpoint", type="checkpoint", optional=True,
                     description="来自 BaseAssetNode；scratch 时可悬空"),
            PortSpec(name="total_steps", type="int", optional=True,
                     description="来自 MultiGatedRewardNode；覆盖 total_timesteps（NodeRow 合并显示）"),
        ],
        outputs=[
            PortSpec(name="algo_config", type="algo_config"),
        ],
        parameters=[
            # ---- Algorithm gate ----
            ParamSpec(key="algorithm", type="enum", default="PPO",
                      choices=["PPO", "SAC", "TD3"],
                      description="算法 / Algorithm",
                      meta={"algorithm_gate": True, "pin_on_collapse": True}),
            # ---- Shared ----
            ParamSpec(key="total_timesteps", type="int", default=1_000_000,
                      widget="index", description="总训练步数",
                      meta={"min": 1, "max": 1_000_000_000,
                            "mission_expose": True,
                            "mission_label_i18n": "mission.train.field.total_timesteps",
                            "pin_on_collapse": True}),
            ParamSpec(key="learning_rate", type="float", default=3e-4,
                      description="学习率",
                      meta={"mission_expose": True,
                            "mission_label_i18n": "mission.train.field.lr",
                            "pin_on_collapse": True}),
            ParamSpec(key="batch_size", type="int", default=256,
                      widget="index", description="Batch size",
                      meta={"min": 1, "max": 1_048_576,
                            "mission_expose": True,
                            "mission_label_i18n": "mission.train.field.batch_size",
                            "pin_on_collapse": True}),
            ParamSpec(key="gamma", type="float", default=0.99,
                      widget="range", description="折扣因子 γ",
                      meta={"min": 0.0, "max": 1.0, "step": 0.001,
                            "mission_expose": True,
                            "mission_label_i18n": "mission.train.field.gamma"}),
            ParamSpec(key="seed", type="int", default=42,
                      widget="index", description="随机种子",
                      meta={"min": 0, "max": 100_000_000,
                            "mission_expose": True,
                            "mission_label_i18n": "mission.train.field.seed"}),
            ParamSpec(key="device", type="enum", default="auto",
                      choices=["auto", "cpu", "cuda"], description="设备"),
            ParamSpec(key="hidden_dim_1", type="int", default=256,
                      widget="index", description="隐藏层 1 维度",
                      meta={"min": 1, "max": 8192}),
            ParamSpec(key="hidden_dim_2", type="int", default=256,
                      widget="index", description="隐藏层 2 维度",
                      meta={"min": 1, "max": 8192}),
            ParamSpec(key="activation", type="enum", default="relu",
                      choices=["relu", "tanh", "elu"], description="激活函数"),
            ParamSpec(key="ent_coef", type="string", default="auto",
                      description='熵系数 ("auto" 或浮点字符串)',
                      meta={"mission_expose": True,
                            "mission_label_i18n": "mission.train.field.ent_coef"}),
            ParamSpec(key="checkpoint_interval", type="int", default=50_000,
                      widget="index", description="Checkpoint 保存间隔",
                      meta={"min": 1, "max": 100_000_000}),
            ParamSpec(key="resume_mode", type="enum", default="scratch",
                      choices=["scratch", "resume", "warm_start_actor"],
                      description="Resume 模式（被 checkpoint.load_mode 覆盖）"),
            ParamSpec(key="policy_id_out", type="string", default="trained_policy",
                      description="输出策略 id"),
            # ---- PPO-only ----
            ParamSpec(key="gae_lambda", type="float", default=0.95,
                      widget="range", description="[PPO] GAE λ",
                      meta={"conditional_on": {"key": "algorithm", "op": "==", "value": "PPO"},
                            "min": 0.0, "max": 1.0, "step": 0.001}),
            ParamSpec(key="n_steps", type="int", default=2048,
                      widget="index", description="[PPO] n_steps",
                      meta={"conditional_on": {"key": "algorithm", "op": "==", "value": "PPO"},
                            "min": 1, "max": 1_000_000}),
            ParamSpec(key="n_epochs", type="int", default=10,
                      widget="index", description="[PPO] n_epochs",
                      meta={"conditional_on": {"key": "algorithm", "op": "==", "value": "PPO"},
                            "min": 1, "max": 1024}),
            ParamSpec(key="lr_schedule", type="enum", default="cosine",
                      choices=["cosine", "linear", "constant"],
                      description="[PPO] 学习率调度",
                      meta={"conditional_on": {"key": "algorithm", "op": "==", "value": "PPO"}}),
            # ---- Off-policy (SAC / TD3) ----
            ParamSpec(key="gradient_steps", type="int", default=1,
                      widget="index", description="[Off-policy] 梯度更新步数",
                      meta={"conditional_on": {"key": "algorithm", "op": "in", "value": ["SAC", "TD3"]},
                            "min": 1, "max": 1024}),
            ParamSpec(key="tau", type="float", default=0.005,
                      description="[Off-policy] target 网络软更新系数",
                      meta={"conditional_on": {"key": "algorithm", "op": "in", "value": ["SAC", "TD3"]}}),
            ParamSpec(key="target_entropy", type="string", default="auto",
                      description='[SAC/TD3] target entropy ("auto" 或浮点字符串)',
                      meta={"conditional_on": {"key": "algorithm", "op": "in", "value": ["SAC", "TD3"]}}),
            ParamSpec(key="buffer_size_mode", type="enum", default="auto",
                      choices=["auto", "manual"],
                      description="[Off-policy] Buffer 大小模式",
                      meta={"conditional_on": {"key": "algorithm", "op": "in", "value": ["SAC", "TD3"]}}),
            ParamSpec(key="buffer_size", type="int", default=1_000_000,
                      widget="buffer_size",
                      description="[Off-policy] Replay buffer 容量",
                      meta={"conditional_on": {"key": "algorithm", "op": "in", "value": ["SAC", "TD3"]},
                            "min": 10_000, "max": 100_000_000}),
            ParamSpec(key="learning_starts", type="int", default=10_000,
                      widget="index", description="[Off-policy] 学习起始步",
                      meta={"conditional_on": {"key": "algorithm", "op": "in", "value": ["SAC", "TD3"]},
                            "min": 0, "max": 100_000_000}),
            # PER (off-policy + use_per gate)
            ParamSpec(key="use_per", type="bool", default=False,
                      description="[Off-policy] 启用 Prioritized Experience Replay",
                      meta={"conditional_on": {"key": "algorithm", "op": "in", "value": ["SAC", "TD3"]}}),
            ParamSpec(key="per_alpha", type="float", default=0.6,
                      widget="range", description="[PER] α (优先级强度)",
                      meta={"conditional_on": {"key": "algorithm", "op": "in", "value": ["SAC", "TD3"]},
                            "min": 0.0, "max": 1.0, "step": 0.01}),
            ParamSpec(key="per_beta", type="float", default=0.4,
                      widget="range", description="[PER] β (importance sampling 校正)",
                      meta={"conditional_on": {"key": "algorithm", "op": "in", "value": ["SAC", "TD3"]},
                            "min": 0.0, "max": 1.0, "step": 0.01}),
            # ---- Collapse guard (always visible) ----
            ParamSpec(key="collapse_guard", type="bool", default=True,
                      description="崩溃保护"),
            ParamSpec(key="collapse_threshold", type="float", default=0.65,
                      widget="range", description="崩溃阈值（best ratio）",
                      meta={"min": 0.0, "max": 1.0, "step": 0.01}),
            ParamSpec(key="collapse_patience", type="int", default=5,
                      widget="index", description="崩溃容忍 epoch 数",
                      meta={"min": 1, "max": 1000}),
            ParamSpec(key="collapse_lr_factor", type="float", default=0.3,
                      widget="range", description="崩溃后学习率乘子",
                      meta={"min": 0.0, "max": 1.0, "step": 0.01}),
            ParamSpec(key="collapse_max_rollbacks", type="int", default=0,
                      widget="index", description="崩溃最大回滚次数（0 = 无限制）",
                      meta={"min": 0, "max": 1000}),
        ],
    )

    # ---- 画布 UI 钩子（动态隐藏端口与参数） ----

    @property
    def hidden_ports(self) -> set:
        """total_steps 端口被 NodeRow 合并到 total_timesteps 行，始终隐藏."""
        return set(self._HIDDEN_PORTS)

    @property
    def hidden_params(self) -> set:
        """根据 algorithm 切换 PPO ↔ off-policy 专属参数；
        off-policy 下若 use_per == false 再隐藏 PER 子集."""
        algorithm = str(self.get_parameter("algorithm", "PPO") or "PPO").strip()
        if algorithm == "PPO":
            # PPO 模式：隐藏所有 off-policy keys
            return set(self._OFF_POLICY_ONLY_KEYS)
        # Off-policy 模式：隐藏 PPO keys
        hidden: set = set(self._PPO_ONLY_KEYS)
        # PER 子集仅在 use_per == true 时可见
        use_per = self.get_parameter("use_per", False)
        if not (use_per is True or str(use_per).strip().lower() == "true"):
            hidden.update(_PER_ONLY_KEYS)
        return hidden

    # ---- 校验 ----

    def validate(self) -> List[str]:
        errors = super().validate()
        algorithm = str(self.get_parameter("algorithm", "PPO") or "PPO").strip()
        if algorithm not in self.ALGORITHM_CHOICES:
            errors.append(
                f"node={self.node_id} algorithm={algorithm!r} 不在 {self.ALGORITHM_CHOICES}"
            )
        return errors