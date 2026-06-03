# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""ExportNode — 统一策略导出 / Unified policy export.

DEMO 对应：``src/system/nodes/sys_nodes/training_nodes.py:ExportNode``
+ ``bin/nodes/training_node_ui.json#export``。

Layer D / category=config. 四大职责（与 DEMO 完全对齐）：
    §1 Output Manifest          — bundle_name + overwrite + version + include_*
                                   外加 output_files 文件清单（含 Open 按钮）
    §2 Start Point Compatibility — start_point_compat 兼容性 diff 面板
    §3 Review Environment       — review_backend / review_scene_id / review_launch

行为契约：``eval_config`` 端口为 optional（DEMO ``_OPTIONAL_INPUTS``）。

UI 行顺序（DEMO ``training_node_ui.json#export`` 1:1）：
    bundle_name → overwrite → version (overwrite=False 时显隐) →
    include_onnx → include_torchscript → include_normalization →
    output_files → start_point_compat →
    review_backend → review_scene_id → review_launch

Legacy ``ParamSpec`` 字段（``bundle_targets`` / ``export_target`` /
``export_onnx`` / ``export_torchscript`` / ``include_norm_stats``）继续保留，
但通过 ``meta={"hidden": True}`` 让 NodeBuilder 跳过 UI 渲染 — 它们仍被
``application.training.spec_compiler`` / ``training_spec`` 消费。
"""

from __future__ import annotations

from typing import Any, Dict

from application.compiler.nodes import (
    manifest_from_toml,
    NODE_MANIFEST_SCHEMA,
    BaseNode,
    NodeKind,
    NodeManifest,
    ParamSpec,
    PortSpec,
)


class ExportNode(BaseNode):
    """Layer D — Unified Export node (replaces SB3 Export + IL Policy Exporter)."""

    # ---- DEMO 行为契约（class-level 提示，画布 UI 读取） ----
    _OPTIONAL_INPUTS: set = {"eval_config"}

    MANIFEST = manifest_from_toml(__file__)

    # ---- 运行时 ----

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Write the trained policy to a v1 bundle under ``<project>/training/exported/``.

        Phase 3 wiring scope (MVP): the upstream ``train_pipe`` is expected
        to carry the artifacts already in the right shape — typically:

            train_pipe = {
                "policy_onnx_path": "<abs path to policy.onnx>",
                "manifest": {<v1 manifest dict>},
                # OR a pre-baked manifest path:
                "manifest_path": "<abs path to manifest.yaml>",
                # plus the usual run identifiers for source.json:
                "run_id": "...", "run_dir": "...",
            }

        The full pipeline that compiles raw RSL-RL ``.pt`` checkpoints +
        ``env.yaml`` into this artifact set lives in Phase 4 (SB3 path) and
        Phase 2.5 (Isaac Lab → contract path). When ``train_pipe`` carries
        only a raw ``.pt`` file (no ``policy_onnx_path``), this node
        explicitly raises NotImplementedError so callers see the gap
        instead of producing an unloadable bundle.

        Returns ``{"bundle_path": <str>}``. The legacy ``"review_launch"``
        echo has been removed — review_launch is now a button widget, not
        a data field.
        """
        from application.training.bundle_exporter import BundleExporter

        params = self.params if hasattr(self, "params") else {}
        train_pipe = inputs.get("train_pipe") if isinstance(inputs, dict) else None
        if not isinstance(train_pipe, dict):
            raise ValueError(
                "ExportNode.execute: required input 'train_pipe' missing or not a dict"
            )

        bundle_name = str(params.get("bundle_name", "<NEW>") or "<NEW>").strip()
        if bundle_name in ("", "<NEW>"):
            raise ValueError(
                "ExportNode.execute: bundle_name is the <NEW> sentinel — "
                "set a concrete name before exporting"
            )
        version = str(params.get("version", "v1") or "v1")
        overwrite = bool(params.get("overwrite", True))

        manifest = train_pipe.get("manifest")
        manifest_path = train_pipe.get("manifest_path")
        if manifest is None and manifest_path:
            import yaml
            with open(manifest_path, "r", encoding="utf-8") as fh:
                manifest = yaml.safe_load(fh)
        if not isinstance(manifest, dict):
            raise NotImplementedError(
                "ExportNode.execute: train_pipe carries no v1 manifest. "
                "Raw .pt → ONNX + manifest compilation is Phase 4 work "
                "(SB3 pipeline) / Phase 2.5 (Isaac Lab compiler). For now "
                "the upstream node must provide either 'manifest' (dict) "
                "or 'manifest_path' (existing v1 manifest.yaml)."
            )

        onnx_source = train_pipe.get("policy_onnx_path")
        if not onnx_source:
            raise NotImplementedError(
                "ExportNode.execute: train_pipe carries no 'policy_onnx_path'. "
                "Phase 3 ExportNode requires the upstream trainer to emit a "
                "ready-to-copy ONNX file."
            )

        source_meta = {
            "type": "training",
            "name": bundle_name,
            "version": version,
            "run_id": str(train_pipe.get("run_id", "")),
            "run_dir": str(train_pipe.get("run_dir", "")),
            "task_id": str(train_pipe.get("task_id", "")),
        }

        result = BundleExporter.export_from_artifacts(
            name=bundle_name,
            version=version,
            manifest=manifest,
            onnx_source=onnx_source,
            source_meta=source_meta,
            overwrite=overwrite,
        )

        return {"bundle_path": str(result.bundle_path)}
