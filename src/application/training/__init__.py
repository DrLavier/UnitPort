# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""application.training — 训练计算管线 / Training compute pipeline.

来源 / Source: DEMO/src/system/training/（去掉已搬到 ``registers/backends.py`` 的
``*_registry / *_catalog`` 类文件，仅留训练管线本身）。

阶段 C 落地文件（待迁移；现在不预创空 .py 以保 "能用文件分隔不开新目录" 初衷）/
Stage C target files (created when content arrives, not as empty placeholders):

- trainer.py        sb3 trainer（< DEMO/src/system/training/sb3_trainer.py）
- amp.py            AMP-PPO（< DEMO/src/system/training/amp/）
- motion.py         motion clip 管线（< DEMO/src/system/training/motion/）
- archive.py        archive_converter（< DEMO/src/system/training/archive_converter/）
- exporter.py       bundle_exporter（< DEMO/src/system/training/bundle_exporter.py）
- discovery.py      robot_assets scanner（< DEMO/src/system/training/robot_assets/discovery/）

⚠ 训练算法 / 场景 / reward 预设清单已搬到 ``registers/backends.py``；
本包只承载 *管线代码*，**不**重复定义注册表数据。
"""

from __future__ import annotations
