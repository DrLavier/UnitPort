"""application.service.brands — Per-brand 运行时包容器 / Per-brand package container.

来源 / Source: DEMO/src/system/brand_packages/

与 ``service/adapters/`` 的区别 / Distinction from adapters/:
- ``adapters/<brand>/``  仅 SDK ↔ UnitPort core 的薄翻译层（IR command → 厂家 SDK 调用）
- ``brands/<brand>/``    完整品牌运行时：joint mapping 已搬到 ``registers/robots.py``，
                         本目录留 brand 私有的 sim 配置 / 烧录脚本 / 厂家服务发现等

⚠ 多品牌包容性铁则同 ``adapters/``：核心代码层禁止 brand 字符串。

阶段 C 落地：第一个品牌包随该品牌的 sim/real 验证流程一并加入。
现在保持空容器，结构在位但**不**预创空品牌占位。
"""

from __future__ import annotations
