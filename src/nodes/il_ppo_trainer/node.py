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

    MANIFEST = manifest_from_toml(__file__)

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
