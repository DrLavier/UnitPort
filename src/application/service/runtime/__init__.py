# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""application.service.runtime — live MuJoCo / ROS2 / input I/O subtree.

Distinct from ``src/runtime/`` (the on-disk cache directory pointed to by
``Paths.RUNTIME_DIR``). This package holds the *source code* for the
runtime layer; the cache directory holds artifacts.

Subpackages:
- ``simulation/``   — MuJoCo wrappers (mj_sim_env / mj_actor / mj_field /
                      pd_controller / sensor_manager).
- ``policy/``       — IL Policy → MuJoCo sim2sim adapter (Phase 3 lands the
                      bulk; ``sim_env_context`` is the only Phase 1 piece).
"""

from __future__ import annotations
