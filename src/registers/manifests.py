# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""registers.manifests — Skill / Policy / Deployment manifest schemas.

合并自原 manifests/ 子目录 / Merged from former manifests/ subdir:
- skill_schema.py       < DEMO/src/system/skill/skill_manifest.py
- policy_schema.py      < DEMO/src/system/policy/manifest_schema.py
- deployment_schema.py  < DEMO/src/system/service/deployment/manifest.py

⚠ DEMO 中的 il_manifest_compat.py 不迁移 —— 违反多品牌包容性铁则。

阶段 B 占位：当前仅暴露 ``load()``；schema 类的具体迁移随阶段 C
``application/compiler/{policy,skill}.py`` 与 ``application/service/deployment.py``
一并完成。
"""

from __future__ import annotations

from typing import Any, Dict


_state: Dict[str, Any] = {"loaded": False}


def load() -> int:
    """阶段 B 占位 / Stage B placeholder."""
    _state["loaded"] = True
    return 0


__all__ = ["load"]
