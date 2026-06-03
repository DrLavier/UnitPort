# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""BaseAssetNode — Start Point selector (3 semantic categories).

start_point token grammar (top-tier, exactly 3 entries):
    __new__      scratch — random init
    __latest__   cumulative training from THIS canvas's last produced
                 run (resolved at compile time from ``last_run_id``)
    __load__     transfer learning — pick any project-local checkpoint;
                 secondary ``checkpoint_id`` field carries the choice

``checkpoint_id`` grammar (only meaningful when start_point=__load__):
    run:<abs .pt path>     run-dir checkpoint (.pt) — resume-compatible
    export:<abs .onnx>     exported policy — warm_start_actor only

``last_run_id`` is written by ``MainWindow._on_start_training`` after
``submit_canvas_training`` succeeds; the picker reads it to decide
whether to surface the Latest option.
"""

from __future__ import annotations

from application.compiler.nodes import (
    manifest_from_toml,
    NODE_MANIFEST_SCHEMA,
    BaseNode,
    NodeKind,
    NodeManifest,
    ParamSpec,
    PortSpec,
)


class BaseAssetNode(BaseNode):
    """Layer C.0 — Training start point selector."""

    _START_POINTS = ("__new__", "__latest__", "__load__")
    _LOAD_MODES = ("scratch", "resume", "warm_start_actor")

    MANIFEST = manifest_from_toml(__file__)
