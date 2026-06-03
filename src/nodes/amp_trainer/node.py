# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""AMPTrainerNode — Legacy alias of ILPPOTrainerNode (training_mode=AMP_PPO).

DEMO 对应：``src/system/nodes/sys_nodes/training_nodes.py:AMPTrainerNode``.

DEMO 类签名：``class AMPTrainerNode(ILPPOTrainerNode)`` —— 继承父 trainer
的全部参数 / 端口，仅在 ``__init__`` 末尾 force ``training_mode = "AMP_PPO"``。
RELEASE 这里不复用 Python 继承（节点 manifest 是数据契约，不是类继承），而是
**复制** ILPPOTrainerNode 的 manifest 并把 ``training_mode`` 默认改成
``"AMP_PPO"``，行为契约（hidden_ports / hidden_params 切换、validate）保持
完全一致。

行为契约：
    training_mode == "PPO"      — 标准 PPO；7 个 AMP 专属参数 + AMP 端口隐藏
    training_mode == "AMP_PPO"  — AMP-PPO（本节点默认）；reference_motion_config /
                                  discriminator_config 端口必连；7 个 AMP
                                  hyperparams 写入 train_pipe

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


class AMPTrainerNode(BaseNode):
    """Layer IL — Legacy AMP-PPO trainer alias (forced training_mode=AMP_PPO)."""

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

    MANIFEST = manifest_from_toml(__file__)

    # ---- 画布 UI 钩子（动态隐藏 PPO 模式下的 AMP 端口与参数） ----

    @property
    def hidden_ports(self) -> set:
        """When user flips this alias back to PPO, hide the AMP-only ports."""
        mode = str(self.get_parameter("training_mode", "AMP_PPO") or "AMP_PPO").strip()
        if mode != "AMP_PPO":
            return set(_CONDITIONAL_AMP_PORTS)
        return set()

    @property
    def hidden_params(self) -> set:
        """When user flips this alias back to PPO, hide the 7 AMP-only params."""
        mode = str(self.get_parameter("training_mode", "AMP_PPO") or "AMP_PPO").strip()
        if mode != "AMP_PPO":
            return set(_AMP_ONLY_KEYS)
        return set()

    # ---- 校验 ----

    def validate(self) -> List[str]:
        errors = super().validate()
        mode = str(self.get_parameter("training_mode", "AMP_PPO") or "AMP_PPO").strip()
        if mode not in self.TRAINING_MODE_CHOICES:
            errors.append(
                f"node={self.node_id} training_mode={mode!r} 不在 {self.TRAINING_MODE_CHOICES}"
            )
        return errors

    # ---- 运行时 ----

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Refuse to launch from canvas IR execution — Play button only.

        See :meth:`ILPPOTrainerNode.execute` for the rationale; the
        canvas-IR-driven path doesn't carry the canvas dict that
        ``IsaacLabTrainingTask`` needs, and the previous silent fallback
        to Isaac Lab's stock task hid this from the user.
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
