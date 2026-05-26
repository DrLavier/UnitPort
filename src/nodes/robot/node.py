# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""RobotNode — 机器人资产 / Robot asset (information display).

DEMO 对应：``src/system/nodes/sys_nodes/training_nodes.py:RobotNode``.

Single source of truth for robot IDENTITY and STRUCTURAL metadata. ``asset_id``
is the only user-editable identity field (plus the ``active_override`` toggle);
everything else displayed on the canvas (joint list, init pose default, actuator
defaults) is a read-only projection of the selected asset. Settings live on the
paired :class:`ActorSettingNode`. (The base_height reward target moved to the
Rewards node's per-item "Value" chip; spawn height comes from actor init_pos_z /
the asset nominal — see CLAUDE.md decouple.)

Phase 2 frontend skeleton — ``execute()`` 留接口给 Stage C 后端 wiring。
"""

from __future__ import annotations

from typing import Any, Dict

from application.compiler.nodes import (
    NODE_MANIFEST_SCHEMA,
    BaseNode,
    NodeKind,
    NodeManifest,
    ParamSpec,
    PortSpec,
)


class RobotNode(BaseNode):
    """Layer A — Robot asset node (read-only display surface)."""

    # ---- DEMO 行为契约（class-level 提示，画布 UI 读取） ----
    # Default asset seeded on first placement so users immediately see a
    # valid body list / joint mapping. Mirrors DEMO ``_INIT_ASSET_ID``.
    _INIT_ASSET_ID = "unitree_go2"

    MANIFEST = NodeManifest(
        schema=NODE_MANIFEST_SCHEMA,
        id="robot",
        kind=NodeKind.CUSTOM,
        version="1.0.0",
        category="config",
        layer="A",
        display_name_key="node.robot.display",
        description_key="node.robot.desc",
        icon="robot.svg",
        inputs=[],
        outputs=[
            PortSpec(
                name="robot_pipe", type="robot_pipe",
                description="Bundles asset identity, joint_names, body list, defaults",
            ),
        ],
        parameters=[
            ParamSpec(
                key="asset_id", type="string", default="unitree_go2",
                widget="picker_robot_asset",
                description="资产 id / Robot asset id (registry-backed picker)",
                meta={"registry_id": "robots", "pin_on_collapse": True},
            ),
            ParamSpec(
                key="active_override", type="enum", default="auto",
                choices=["auto", "mjcf", "usd", "urdf"],
                widget="active_file_table",
                description="格式覆盖 / Active asset format override (3-row file-type table)",
            ),
            ParamSpec(
                key="body_mapping", type="json", default="",
                widget="body_mapping_table",
                description="body→IR-role 映射 (auto-resolved) / Body→IR-role mapping",
            ),
            ParamSpec(
                key="_mjcf_base_offset_status", type="string", default="",
                widget="mjcf_offset_badge",
                description="MJCF 落地偏移 / MJCF base spawn-Z offset (auto-calibrated; read-only status)",
            ),
            # ── PD parameterization (sim2sim_mass-matrix-adaptive) ──
            # Merged from ex-ActuatorPDNode (which lived as a separate
            # canvas node for one iteration before consolidating onto
            # RobotNode — the params are robot-bound by nature and the
            # extra node was redundant graph topology).
            #
            # ALL FIELDS ARE OPTIONAL. Empty pd_groups ({}) means "use
            # family defaults" for every group, loaded from
            # registers/data/pd_groups_defaults.json (e.g. quadruped:
            # hip_x ωn=25/ζ=1.0, hip_y ωn=35/ζ=0.8, knee ωn=35/ζ=0.8).
            # To override a single group, write
            # {"knee": {"omega_n": 40}} — other groups still use defaults.
            ParamSpec(
                key="pd_param_mode", type="enum", default="natural_freq",
                choices=["natural_freq"],
                description="PD 参数化模式 / PD parameterization mode",
            ),
            ParamSpec(
                key="pd_groups", type="json", default="{}",
                widget="pd_param_table",
                description="按关节组覆盖 (omega_n, zeta)；空 {} = 全用 family 默认 / Per-group overrides; empty = family defaults",
                meta={"language": "json"},
            ),
            # effort_limit / velocity_limit intentionally NOT defined here.
            # The single canvas source of truth is the ActorSetting node
            # (§4 Actuator overrides); de-duplicated off RobotNode so both
            # engines read one value. RobotNode owns only (omega_n, zeta)
            # + the PD-process toggles below.
            ParamSpec(
                key="pd_resolve_at_reset", type="bool", default=True,
                description="每次 reset 重算 PD / Re-solve gains after each DR reset",
            ),
            ParamSpec(
                key="pd_calibration_blocking", type="bool", default=True,
                description="校准失败阻断 bundle / Block bundle export on calibration fail",
            ),
            ParamSpec(
                key="pd_skip_calibration", type="bool", default=False,
                description="跳过 step-response 校准 / Skip calibration",
            ),
        ],
    )

    # ---- 运行时 ----

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve ``asset_id`` (alias or SKU) → :class:`RobotAsset` +
        :class:`BodyIRMapper`, return them under ``robot_pipe``.

        Downstream nodes consume the pipe dict directly; ``physics_config``
        / ``actor_setting`` etc. read ``robot_pipe['joints']`` and
        ``robot_pipe['mjcf_path']`` for their own configuration.
        """
        import json

        from application.service.robot_assets.service import (
            get_robot_asset_service,
        )
        from application.training.body_ir import (
            BodyIRMapper,
            apply_user_overrides,
        )
        from registers import robots as _robots_registry

        params = self.params if hasattr(self, "params") else {}
        raw_id = str(params.get("asset_id", self._INIT_ASSET_ID) or "").strip()

        # Single resolution path (SKU / alias / brand.model) shared with
        # canvas-level ``robot_id`` derivation.
        sku = _robots_registry.resolve_to_sku(raw_id)

        svc = get_robot_asset_service()
        asset = svc.resolve(sku) if sku else None
        if asset is None:
            raise KeyError(
                f"RobotNode: asset_id {raw_id!r} did not resolve to a "
                f"known SKU in registers.robots"
            )

        # Per-kind asset status — same data the sidebar Robot Asset panel
        # surfaces, so panel and node are guaranteed in sync (single
        # source: registers.robots → RobotAssetService.resolve()).
        asset_status = {
            kind: asset.asset_status(kind).value
            for kind in ("MJCF", "USD", "URDF", "XACRO")
        }

        # Resolve ``active_override`` ∈ {"auto", "mjcf", "usd", "urdf"} into
        # a concrete chosen format. Cloud-only USD is selectable when
        # ``usd_url`` is present (e.g. IsaacLab Nucleus assets).
        # Hoisted above mapper construction (Stage 3): the mapper is now
        # format-bound, so we need active_format before building it.
        override = str(params.get("active_override", "auto") or "auto").strip().lower()
        backend = str(params.get("backend", "sb3") or "sb3").strip()

        def _avail(kind: str) -> bool:
            if getattr(asset, f"{kind}_path", None) is not None:
                return True
            return kind == "usd" and bool(asset.usd_url)

        if override in ("mjcf", "usd", "urdf"):
            active_format = override
        else:
            order = ("usd", "urdf", "mjcf") if backend == "isaac_lab" else ("mjcf", "usd", "urdf")
            active_format = next((k for k in order if _avail(k)), "")

        # Per-format mapper: read bodies_per_format[active_format] from
        # the registry (no runtime asset parsing). Empty when the format
        # isn't declared yet — the canvas widget surfaces that state.
        mapper = BodyIRMapper.from_robot_asset(
            asset, active_format=active_format.upper() if active_format else None,
        )

        # Replay user's per-format body↔role overrides on top of the
        # per-format auto-detection.
        if active_format:
            overrides = svc.get_body_ir_overrides(asset.sku, fmt=active_format.upper())
        else:
            overrides = svc.get_body_ir_overrides(asset.sku)
        if overrides:
            apply_user_overrides(mapper, overrides)

        # One-shot MJCF base spawn-Z calibration (USD↔MJCF anchor offset).
        # Fires only when an MJCF asset is active — the offset is irrelevant
        # for pure USD/Isaac Lab runs. The gate is no-op when a fresh
        # overlay already exists (sha256 match on MJCF + standing pose);
        # otherwise it submits a background Task that runs mujoco.mj_forward
        # at standing pose and writes the result into
        # state.json :: assets[sku].mjcf_base_height_offset. The next MuJoCo
        # spawn (Review Pose / generic_mujoco_env / mj_sim_env) reads that
        # value via application.service.robot_assets.runtime.read_mjcf_base_offset
        # and lifts qpos[base_z] accordingly. See plan
        # C:/Users/84373/.claude/plans/curious-nibbling-plum.md.
        if active_format == "mjcf":
            try:
                from application.service.robot_assets.service import (
                    ensure_mjcf_base_offset_cached,
                )
                ensure_mjcf_base_offset_cached(asset.sku)
            except Exception as exc:  # noqa: BLE001
                # WHY KEPT (§1.8 (c)): calibration is best-effort — a
                # transient failure (TasksManager not yet ready during
                # boot, mjcf path race) must not block Robot Node
                # execution. The runtime reader will WARN about
                # uncalibrated state on the next spawn.
                from unitport_sdk import log_warning
                log_warning(
                    f"[robot_node] ensure_mjcf_base_offset_cached failed "
                    f"for sku={asset.sku!r}: {exc}"
                )

        # Transient mirror: keep the params["body_mapping"] string in sync so any
        # downstream consumer that reads the parameter as JSON sees the resolved
        # mapping. NEVER persisted by execute() — persistence is the editing UI's
        # job, via set_body_ir_overrides(sku, ..., fmt=fmt).
        mapping_dict = mapper.to_dict()
        params["body_mapping"] = json.dumps(mapping_dict, ensure_ascii=False)

        # Pick the path/URL that backs ``active_format``. For cloud-only
        # USD, ``active_path`` is "" and ``active_url`` carries the
        # nucleus: marker so the launcher knows to expand it inside the
        # IsaacLab venv.
        if active_format == "mjcf":
            active_path = str(asset.mjcf_path) if asset.mjcf_path else ""
            active_url = ""
        elif active_format == "urdf":
            active_path = str(asset.urdf_path) if asset.urdf_path else ""
            active_url = ""
        elif active_format == "usd":
            active_path = str(asset.usd_path) if asset.usd_path else ""
            active_url = "" if asset.usd_path else (asset.usd_url or "")
        else:
            active_path = ""
            active_url = ""

        return {
            "robot_pipe": {
                "asset": asset,
                "sku": asset.sku,
                "brand": asset.brand,
                "model": asset.model,
                "name": asset.name,
                "families": list(asset.families),
                "mjcf_path": str(asset.mjcf_path) if asset.mjcf_path else "",
                "urdf_path": str(asset.urdf_path) if asset.urdf_path else "",
                "usd_path": str(asset.usd_path) if asset.usd_path else "",
                "xacro_path": str(asset.xacro_path) if asset.xacro_path else "",
                "usd_url": asset.usd_url or "",
                "asset_status": asset_status,
                "active_override": override,
                "active_format": active_format,
                "active_path": active_path,
                "active_url": active_url,
                "joint_names": list(asset.joints.keys()),
                "joint_ir_roles": dict(asset.joints),
                "body_mapper": mapper,
                "body_mapping": mapping_dict,
                "ir_family": mapper.family,
            },
        }
