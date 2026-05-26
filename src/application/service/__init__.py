# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""application.service — I/O + 状态服务层 / Stateful I/O service layer.

布局（高度集中化，plan §3）/ Layout (highly consolidated):
- telemetry.py           事件总线 + 数据契约 + 主线程 marshaller（已实搬）
- adapters/, brands/     ⭐ 唯一保留的子目录（per-brand 子包；多品牌包容性契约要求）
                         待阶段 C 第一个品牌包落地时建立；之前不创建空占位
- connection.py          连接管理（待阶段 C）
- deployment.py          部署管线（待阶段 C；含 Sim/Real 切换）
- models.py              机器人模型适配器（去掉已搬到 registers/robots 的注册表）
- auth.py                认证 / 凭据管理（cloud channel 走 SDK Storage stub）
- storage.py             本地 Storage backend 业务侧扩展（cloud 走 SDK Storage stub）

⚠ 与编译期 ``application.compiler`` 对偶 —— compiler 是无 I/O 的纯计算层，
service 是有 I/O / 有状态的运行时层。

⚠ DEMO/src/system/core/{log,config,localisation} 已被 SDK 覆盖，整体丢弃；
仅 robot_context 类等业务级 service 待阶段 C 按需迁入相应文件。

⚠ DEMO/src/system/{engine,engines,runtime} 三块在 RELEASE 中合并到
``application/engine/``（单数；plan §3）。本目录不再承载执行内核与实时 I/O。
"""

from __future__ import annotations
