# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""application.engine — 运行期执行层 / Runtime execution layer.

⭐ 命名整改：本目录**单数** ``engine``。原 DEMO 中 ``engine + engines + runtime``
三块在 RELEASE 合并到此，解决 ``engine`` vs ``engines`` 起名冲突；
注册表层（可用后端清单）则改名为 ``registers/backends.py``。

阶段 C 落地文件（待迁移；现在不预创空 .py 以保 "能用文件分隔不开新目录" 初衷）/
Stage C target files:

- core.py           IR runtime + safety + interception（< DEMO/src/system/engine/）
- sim_isaac.py      Isaac Sim 后端（< DEMO/src/system/engines/）
- sim_mujoco.py     MuJoCo 后端（< DEMO/src/system/runtime/mujoco/）
- bridge_ros2.py    ROS2 bridge（< DEMO/src/system/runtime/ros2/）
- input_sources.py  控制器 / 输入设备（< DEMO/src/system/runtime/input_sources/）

⚠ 与编译期 ``application.compiler`` 对偶 —— compiler 是无 I/O 的纯计算层，
engine 是实时 I/O / 有状态的执行层。
"""

from __future__ import annotations
