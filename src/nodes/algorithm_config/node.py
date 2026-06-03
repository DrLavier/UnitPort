# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

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
    manifest_from_toml,
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

    MANIFEST = manifest_from_toml(__file__)

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