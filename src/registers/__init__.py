# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""registers — 全局注册表层 / Global registry layer.

职责 / Purpose:
- 把 DEMO 中分散的 30 个注册表收口到一个目录下。
- 整层扁平化为 8 个 ``.py`` + 1 个 ``data/`` 数据目录（数据按 ``<域>_<键>.json`` 命名）。
- 业务代码只能通过 ``RegistryHub`` 或各域 ``.py`` 公开 API 查询，禁止绕过直读 ``data/``。

加载时机 / Loading:
- ``UnitPortMain.data_load()`` 调用 ``RegistryHub.load_all()`` 一次性读入并校验。
- 后续运行只查不重读（除非显式 ``reload()``）。

写入路径 / Write path:
- 用户运行时只能写 ``USER_CONFIG_DIR`` 下的层（通过 ``Storage.push_data``）。
- 出厂数据 ``data/*`` **禁止运行时修改**。
- 唯一例外：``data/backends_installed.json``（per-installation 物理事实，由
  ``backends.refresh_engine_availability()`` 主动写入）。

校验 / Validation:
- ``validate()`` 跨注册表检查：robot.joints[*].ir_role 必须在 IR canonical 之内。
- 失败抛 ``RegistryValidationError``，由 UnitPortMain 决定是阻塞启动还是降级。

USER overlay（plan.md §11.1）/ User-level overlays merged on load:
- ``~/UnitPort/registers/ir_custom.json``       → ir.merge_user_extensions
- ``~/UnitPort/registers/robots_custom.json``   → robots.merge_user_extensions
- ``~/UnitPort/registers/pd_groups_custom.json`` → pd_groups.merge_user_extensions
- ``~/UnitPort/registers/families_custom.json`` → families.merge_user_extensions
- ``~/UnitPort/registers/symmetry_custom.json`` → symmetry.merge_user_extensions
- ``~/UnitPort/registers/pd_calibration_canaries_custom.json`` → pd_calibration_canaries.merge_user_extensions
- ``~/UnitPort/registers/amp_obs_templates_custom.json`` → amp_obs_templates.merge_user_extensions
- ``~/UnitPort/registers/gait_commands_custom.json`` → gait_commands.merge_user_extensions
- ``~/UnitPort/registers/motion_phases_custom.json`` → motion_phases.merge_user_extensions

SKU helpers（数据库级唯一键）/ Database-level unique keys:
- ``build_sku("Unitree", "Go2-W") == build_sku("unitree", "go2w")``
- 后端互引一律用 SKU；前端/UI/日志显示 ``name`` 字段，永不显示 SKU。
- 12 位 hex（48 bit）≈ 2.8e14 空间，出厂百级机器人 + 千级关节安全。
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict

from unitport_sdk import Paths, log_debug, log_error, log_success, log_warning, read_data


# =============================================================================
# SKU helpers（合并自 _sku.py / Inlined from former _sku.py）
# =============================================================================

_SLUG_RE = re.compile(r"[\s\-_/.]+")


def _normalize_sku_part(s: str) -> str:
    """大小写折叠 + 剥除所有分隔符 / Case-fold + strip all separators.

    使 "Go2-W" / "Go2 W" / "Go2_W" / "Go2.W" / "Go2W" 折叠为同一形态 "go2w"。
    """
    return _SLUG_RE.sub("", s.strip().lower())


def build_sku(*parts: str, length: int = 12) -> str:
    """根据多段语义键生成稳定 SKU / Build a stable SKU from semantic key parts."""
    raw = ".".join(_normalize_sku_part(p) for p in parts if p)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:length]


def resolve_robot_sku(brand: str, model: str) -> str:
    return build_sku(brand, model)


def resolve_joint_sku(robot_sku: str, raw_joint_name: str) -> str:
    return build_sku(robot_sku, raw_joint_name)


# =============================================================================
# RegistryHub — 统一加载/校验/查询入口
# =============================================================================

class RegistryValidationError(RuntimeError):
    """注册表跨域完整性校验失败 / Cross-registry validation failed."""


class RegistryHub:
    """所有子注册表的统一入口 / Unified entry to all sub-registries."""

    _loaded: bool = False
    _summary: Dict[str, int] = {}

    # Dependency-ordered load chain. Each entry must follow the modules it
    # references at load() time. Specifically:
    #   * ir      — no deps (root vocabulary).
    #   * pd_groups — calls ir._resolve_alias semantically; pd_groups.load
    #     itself is self-contained but pd_groups.validate() (run from the
    #     hub later) reads ir, so for symmetry + readability we keep it
    #     adjacent to ir. (Historical layout had pd_groups at the tail —
    #     that was a chronological artefact of pd_groups being added
    #     last; the current order is by dependency, not insertion date.)
    #   * families — depends on the catalog file alone; needs to be ready
    #     BEFORE robots so the hub validate Rule C5 can verify each SKU's
    #     family strings against the catalog.
    #   * robots — depends on ir (IR roles used as joint.ir_role value range)
    #     and on families (Rule C5). robots.load itself does not call into
    #     families/ir; only the hub validator does, after every module is
    #     loaded.
    #   * symmetry — mirror op declarative data. Sits AFTER robots
    #     deliberately:
    #       (1) symmetry.load reads ONLY symmetry_catalog.json — no
    #           dependency on robots, families, or ir at load time.
    #       (2) symmetry.validate is called from the hub validator and
    #           uses ONLY ir.list_roles(family) for family-slice partition
    #           checks. It does NOT cross-reference robots: symmetry data
    #           is family-keyed, not SKU-keyed, and intentionally decoupled
    #           from the SKU table.
    #       (3) No SKU-side check needs symmetry to be loaded before
    #           robots, so placing symmetry after robots keeps the
    #           "ir/pd_groups/families/robots cluster" intact (they are
    #           cross-validated by the hub) and groups the leaf consumers
    #           (symmetry, nodes, manifests, ...) together.
    #   * pd_calibration_canaries — per-family PD calibration canary group
    #     recipe. Same shape rationale as symmetry:
    #       (1) load reads ONLY pd_calibration_canaries.json — no other
    #           registry dependency.
    #       (2) validate consults pd_groups (for the S1 group-id existence
    #           check) and families (for the S2 catalog-coverage check),
    #           both already loaded by this point.
    #       (3) Data is family-keyed (one entry per family), NOT SKU-keyed,
    #           so it does not need to precede robots. Placed after
    #           symmetry to keep the family-data leaf cluster contiguous.
    # Modules after robots are leaf consumers in load order — they do not
    # influence robots / families / ir / pd_groups validation.
    DOMAINS = (
        "ir",
        "pd_groups",
        "families",
        "robots",
        "symmetry",
        "pd_calibration_canaries",
        "amp_obs_templates",
        "gait_commands",
        "nodes",
        "manifests",
        "backends",
        "brands",
        "services",
        "commands",
        "motion_phases",
    )

    @classmethod
    def load_all(cls) -> None:
        """按依赖顺序加载所有子注册表 / Load every sub-registry in dependency order.

        顺序：
        1. ir         — IR 角色枚举，无依赖
        2. pd_groups  — PD 关节组（其 validate() 引用 ir，load 自给自足）
        3. families   — family 元数据 + 继承链（RegistryHub.validate Rule C5 依赖此）
        4. robots     — 依赖 ir + families（Rule 3 + Rule C5）
        5. symmetry   — mirror op 数据 + partition 校验（validate 只用
                        ir.list_roles，与 robots 解耦——故置 robots 后）
        6. 其他后续注册表（nodes / manifests / backends / brands / services /
           commands / motion_phases）
        7. 合入 USER_CONFIG_DIR 的拓展（ir_custom / pd_groups_custom /
           families_custom / symmetry_custom / robots_custom /
           motion_phases_custom）
        """
        if cls._loaded:
            return

        cls._summary = {}

        from . import ir
        cls._summary["ir"] = ir.load()
        log_debug(f"[registers] ir loaded: {cls._summary['ir']} roles")

        from . import pd_groups
        cls._summary["pd_groups"] = pd_groups.load()
        log_debug(f"[registers] pd_groups loaded: {cls._summary['pd_groups']} groups")

        from . import families
        cls._summary["families"] = families.load()
        log_debug(f"[registers] families loaded: {cls._summary['families']} families")

        from . import robots
        cls._summary["robots"] = robots.load()
        log_debug(f"[registers] robots loaded: {cls._summary['robots']} entries")

        from . import symmetry
        cls._summary["symmetry"] = symmetry.load()
        log_debug(f"[registers] symmetry loaded: {cls._summary['symmetry']} families")

        from . import pd_calibration_canaries
        cls._summary["pd_calibration_canaries"] = pd_calibration_canaries.load()
        log_debug(
            f"[registers] pd_calibration_canaries loaded: "
            f"{cls._summary['pd_calibration_canaries']} families"
        )

        from . import amp_obs_templates
        cls._summary["amp_obs_templates"] = amp_obs_templates.load()
        log_debug(
            f"[registers] amp_obs_templates loaded: "
            f"{cls._summary['amp_obs_templates']} families"
        )

        from . import gait_commands
        cls._summary["gait_commands"] = gait_commands.load()
        log_debug(
            f"[registers] gait_commands loaded: "
            f"{cls._summary['gait_commands']} families"
        )

        from . import init_pose_presets
        cls._summary["init_pose_presets"] = init_pose_presets.load()
        log_debug(
            f"[registers] init_pose_presets loaded: "
            f"{cls._summary['init_pose_presets']} families"
        )

        from . import nodes
        cls._summary["nodes"] = nodes.load()

        from . import manifests
        cls._summary["manifests"] = manifests.load()

        from . import backends
        cls._summary["backends"] = backends.load()

        from . import brands
        cls._summary["brands"] = brands.load()

        from . import services
        cls._summary["services"] = services.load()

        from . import commands
        cls._summary["commands"] = commands.load()

        from . import motion_phases
        cls._summary["motion_phases"] = motion_phases.load()

        # Factory-build overlay layer (FACTORY_BUILD_DIR/robots_factory_build.json):
        # locally-built factory data (USD body dumps from a local IsaacLab
        # installation, asset extractions for standard SKUs). Merged BEFORE
        # the user overlay so user-private SKUs still win on conflict; merged
        # AFTER the SDK-ship factory layer so locally-dumped data overrides
        # any null placeholders in robots_canonical.json. See robots.py
        # ``_factory_build_path`` docstring for the §9 portable-artifacts
        # justification.
        cls._merge_factory_build_overlays()

        cls._merge_user_overlays()

        cls._loaded = True
        log_success(f"[registers] load_all complete: {cls._summary}")

    @classmethod
    def _merge_factory_build_overlays(cls) -> None:
        """合入 FACTORY_BUILD_DIR 下的工厂自构建数据（缺失=空集，不报错）。

        当前只有 robots 域支持工厂自构建层（USD body 自动 dump 落盘）；
        其他注册表如有类似需求，按相同模式扩展即可（独立 _state[
        "factory_build_extensions"] dict + ``target_layer="factory_build"``
        参数）。
        """
        from . import robots
        fb_path = robots._factory_build_path()
        if not fb_path.exists():
            return
        payload = read_data(fb_path) or {}
        if not isinstance(payload, dict):
            return
        extensions = dict(payload.get("robots", {}) or {})
        if extensions:
            added = robots.merge_user_extensions(
                extensions, target_layer="factory_build"
            )
            cls._summary["robots"] = cls._summary.get("robots", 0) + added
            log_debug(
                f"[registers] robots_factory_build merged: +{added} entries"
            )

    @classmethod
    def _merge_user_overlays(cls) -> None:
        """合入 USER_CONFIG_DIR 的 ir_custom + robots_custom（缺失=空集，不报错）。"""
        user_root = Paths.USER_CONFIG_DIR / "registers"

        ir_custom_path = user_root / "ir_custom.json"
        if ir_custom_path.exists():
            from . import ir
            payload = read_data(ir_custom_path) or {}
            # families MUST be merged before role-level extensions so an
            # extension can target a freshly-registered user family via its
            # optional ``family`` field.
            user_families = dict(payload.get("families", {}) or {})
            if user_families:
                added_fams = ir.merge_user_family_extensions(user_families)
                if added_fams:
                    log_debug(f"[registers] ir_custom merged: +{added_fams} families")
            extensions = list(payload.get("extensions", []) or [])
            if extensions:
                added = ir.merge_user_extensions(extensions)
                cls._summary["ir"] = cls._summary.get("ir", 0) + added
                log_debug(f"[registers] ir_custom merged: +{added} roles")

        robots_custom_path = user_root / "robots_custom.json"
        if robots_custom_path.exists():
            from . import robots
            payload = read_data(robots_custom_path) or {}
            extensions = dict(payload.get("robots", {}) or {})
            if extensions:
                added = robots.merge_user_extensions(extensions)
                cls._summary["robots"] = cls._summary.get("robots", 0) + added
                log_debug(f"[registers] robots_custom merged: +{added} entries")

        motion_phases_custom_path = user_root / "motion_phases_custom.json"
        if motion_phases_custom_path.exists():
            from . import motion_phases
            payload = read_data(motion_phases_custom_path) or {}
            if isinstance(payload, dict):
                added = motion_phases.merge_user_extensions(payload)
                if added:
                    cls._summary["motion_phases"] = cls._summary.get("motion_phases", 0) + added
                    log_debug(f"[registers] motion_phases_custom merged: +{added} phases")

        pd_groups_custom_path = user_root / "pd_groups_custom.json"
        if pd_groups_custom_path.exists():
            from . import pd_groups
            payload = read_data(pd_groups_custom_path) or {}
            if isinstance(payload, dict):
                added = pd_groups.merge_user_extensions(payload)
                if added:
                    log_debug(f"[registers] pd_groups_custom merged: +{added} overrides")

        families_custom_path = user_root / "families_custom.json"
        if families_custom_path.exists():
            from . import families
            payload = read_data(families_custom_path) or {}
            if isinstance(payload, dict):
                added = families.merge_user_extensions(payload)
                if added:
                    cls._summary["families"] = cls._summary.get("families", 0) + added
                    log_debug(f"[registers] families_custom merged: +{added} families")

        symmetry_custom_path = user_root / "symmetry_custom.json"
        if symmetry_custom_path.exists():
            from . import symmetry
            payload = read_data(symmetry_custom_path) or {}
            if isinstance(payload, dict):
                added = symmetry.merge_user_extensions(payload)
                if added:
                    cls._summary["symmetry"] = cls._summary.get("symmetry", 0) + added
                    log_debug(f"[registers] symmetry_custom merged: +{added} families")

        pd_canaries_custom_path = user_root / "pd_calibration_canaries_custom.json"
        if pd_canaries_custom_path.exists():
            from . import pd_calibration_canaries
            payload = read_data(pd_canaries_custom_path) or {}
            if isinstance(payload, dict):
                added = pd_calibration_canaries.merge_user_extensions(payload)
                if added:
                    cls._summary["pd_calibration_canaries"] = (
                        cls._summary.get("pd_calibration_canaries", 0) + added
                    )
                    log_debug(
                        f"[registers] pd_calibration_canaries_custom merged: "
                        f"+{added} families"
                    )

        amp_obs_templates_custom_path = user_root / "amp_obs_templates_custom.json"
        if amp_obs_templates_custom_path.exists():
            from . import amp_obs_templates
            payload = read_data(amp_obs_templates_custom_path) or {}
            if isinstance(payload, dict):
                added = amp_obs_templates.merge_user_extensions(payload)
                if added:
                    cls._summary["amp_obs_templates"] = (
                        cls._summary.get("amp_obs_templates", 0) + added
                    )
                    log_debug(
                        f"[registers] amp_obs_templates_custom merged: "
                        f"+{added} families"
                    )

        gait_commands_custom_path = user_root / "gait_commands_custom.json"
        if gait_commands_custom_path.exists():
            from . import gait_commands
            payload = read_data(gait_commands_custom_path) or {}
            if isinstance(payload, dict):
                added = gait_commands.merge_user_extensions(payload)
                if added:
                    cls._summary["gait_commands"] = (
                        cls._summary.get("gait_commands", 0) + added
                    )
                    log_debug(
                        f"[registers] gait_commands_custom merged: "
                        f"+{added} families"
                    )

        init_pose_presets_custom_path = (
            user_root / "init_pose_presets_custom.json"
        )
        if init_pose_presets_custom_path.exists():
            from . import init_pose_presets
            payload = read_data(init_pose_presets_custom_path) or {}
            if isinstance(payload, dict):
                added = init_pose_presets.merge_user_extensions(payload)
                if added:
                    cls._summary["init_pose_presets"] = (
                        cls._summary.get("init_pose_presets", 0) + added
                    )
                    log_debug(
                        f"[registers] init_pose_presets_custom merged: "
                        f"+{added} families"
                    )

    @classmethod
    def validate(cls) -> None:
        """Cross-registry integrity check.

        Rules:
        1. Every robot must use the per-format schema (``joints_per_format``
           + ``bodies_per_format``); the legacy flat ``joints`` field is
           rejected (already migrated via
           ``bootstrap/migrate_canonical_to_per_format.py``).
        2. Each robot must declare at least one non-null format (joints
           or bodies). All-null = warning only — registered but not
           trainable; training launcher surfaces the error at use-time.
        3. Under every declared format, every joints[*].ir_role and
           bodies[*].ir_role must be non-empty AND a member of the IR
           canonical role set for that robot's families.
        4. Inheritance integrity (model_family / variant_label /
           inherits_from): variant chains forbidden, family must match
           parent, parent must exist.
        5. pd_groups integrity (delegated to ``pd_groups.validate()``).
        6. physics_per_format / body_inertials_per_format /
           joint_frames_per_format cache integrity vs joints/bodies tables.
        7. families catalog integrity (delegated to ``families.validate()``):
           every non-null ``inherits_from`` targets an existing family;
           no inheritance cycle.
        8. symmetry catalog cross-family partition (delegated to
           ``symmetry.validate()``): every IR role in each
           non-alias family's slice partitions cleanly across
           ``role_swaps`` / ``role_self_map`` / ``role_unmapped``
           (S1); referenced roles exist in the family's IR slice (S3);
           ``alias_of`` chain has no missing target / no cycle (S4).
        9. pd_calibration_canaries catalog cross-family integrity
           (delegated to ``pd_calibration_canaries.validate()``):
           every canary group id is a defined ``pd_groups`` group for
           the family (S1); every catalog family is declared in
           ``families_catalog.json`` (S2); ``alias_of`` chain has no
           missing target / no cycle.
       10. amp_obs_templates catalog cross-family integrity (delegated
           to ``amp_obs_templates.validate()``): every term name in
           ``template_terms`` is registered in
           ``application.training.amp.obs_terms`` (A1); every catalog
           family is declared in ``families_catalog.json`` (A2);
           ``alias_of`` chain has no missing target / no cycle (A3); a
           families_catalog family with no obs templates entry emits a
           soft warning (A4) -- omission is a coverage gap, not a
           corruption, so it does not block boot.
       11. gait_commands catalog cross-family integrity (delegated to
           ``gait_commands.validate()``): every non-alias entry declares
           non-empty ``class_name`` (G1); ``phase_count ==
           len(phase_names)`` (G2); ``phase_obs_dim == 2 *
           phase_count`` (G3); ``alias_of`` chain has no missing target
           / no cycle (G4); every catalog family is declared in
           ``families_catalog.json`` (G5). G6 emits a soft warning when
           a families_catalog family has no gait command entry -- the
           omission is legitimate for non-legged families (manipulator
           / wheeled / generic) and does not block boot.
       12. init_pose_presets catalog cross-family integrity (delegated
           to ``init_pose_presets.validate()``): every non-alias entry
           has unique preset names (I1) with 3-tuple float base_pos
           (I2); ``alias_of`` chain has no missing target / no cycle
           (I3); every catalog family is declared in
           ``families_catalog.json`` (I4); a user-overlay preset that
           smuggles ``joint_pos_by_ir`` into the family layer is
           rejected (I5 -- the family layer is name + base_pos by
           design; per-SKU joint angles go to
           ``robots_canonical.json[sku].init_pose_presets``). I6 emits
           a soft warning when a families_catalog family has no init
           pose preset entry -- the omission is legitimate for
           families without canonical poses and does not block boot.

        Plus SKU-side family checks:
        C4. Every SKU must declare a non-empty ``families`` field (factory
            SKUs raise; user-overlay SKUs warn so a half-finished Dump does
            not brick startup — same precedent as Rule 3).
        C5. Every family string declared on a SKU must exist in
            ``families_catalog.json`` (factory raise; user warn).

        Raises :class:`RegistryValidationError` on rule 1 / 3 / 4 / 5 / 6 /
        7 / C4 / C5 failure (rule 2 logs a warning and continues).
        """
        if not cls._loaded:
            raise RuntimeError("RegistryHub.load_all() 必须先调用")

        # Note: the validation loop body reassigns ``families`` to the
        # per-SKU list, so we must alias the module import — otherwise the
        # name would shadow halfway through the loop.
        from . import ir, robots
        from . import families as families_module

        _FORMATS = ("MJCF", "USD", "URDF")
        problems: list[str] = []

        for sku in robots.list_skus():
            entry = robots.get_robot(sku)
            if entry is None:
                continue
            name = entry.get("name", "")
            families = list(entry.get("families", []) or [])
            # User-overlay entries (set via Dump MJCF/USD, manual edits,
            # robots_custom.json) are works-in-progress: the user is
            # filling in IR roles via the canvas BodyMappingTable. We
            # log empty / unknown ir_role hits as warnings rather than
            # rejecting the whole boot — otherwise a half-finished Dump
            # would brick the app on next startup.
            is_user_layer = False
            try:
                is_user_layer = bool(robots.is_user_extension(sku))
            except Exception:
                pass

            # Rule 1: legacy ``joints`` field is forbidden post-migration.
            if "joints" in entry:
                problems.append(
                    f"  robot={sku}({name}) still carries the legacy flat "
                    f"'joints' field; run bootstrap/migrate_canonical_to_per_format.py "
                    f"to finish the per-format migration"
                )
                continue

            joints_pf = entry.get("joints_per_format")
            bodies_pf = entry.get("bodies_per_format")
            if not isinstance(joints_pf, dict) or not isinstance(bodies_pf, dict):
                problems.append(
                    f"  robot={sku}({name}) missing joints_per_format / "
                    f"bodies_per_format field"
                )
                continue

            # Rule 2: warn (don't error) when every format is null. This is
            # a legal "registered but not trainable" state — e.g. catalog
            # entries for robots whose joint table hasn't been filled in yet.
            # The training launcher surfaces the error at use-time, not here.
            any_declared = False
            for fmt in _FORMATS:
                if isinstance(joints_pf.get(fmt), dict) or isinstance(bodies_pf.get(fmt), dict):
                    any_declared = True
                    break
            if not any_declared:
                log_warning(
                    f"[registers] robot={sku}({name}) joints_per_format / "
                    f"bodies_per_format are all null - no format declared; "
                    f"open the Robot Asset card and click Dump MJCF/USD "
                    f"before training"
                )
                continue

            # Rule C4: SKU must declare ≥1 family. Fail-loud rather than
            # silent skip. User-overlay entries get a warning instead of
            # a hard problem, mirroring Rule 3's is_user_layer treatment
            # — a Dump MJCF/USD wizard that has not yet had family
            # assigned must not brick boot.
            if not families:
                if is_user_layer:
                    log_warning(
                        f"[registers] robot={sku}({name}) has empty families "
                        f"field - declare at least one family from "
                        f"families_catalog.json via the Robot Asset card"
                    )
                    continue
                problems.append(
                    f"  robot={sku}({name}) has empty families field; every "
                    f"registered SKU must declare at least one family from "
                    f"families_catalog.json"
                )
                continue

            # Rule C5: every SKU family must be in the catalog.
            catalog_fams = set(families_module.list_families())
            unknown_fams = [f for f in families if f not in catalog_fams]
            if unknown_fams:
                for fam in unknown_fams:
                    if is_user_layer:
                        log_warning(
                            f"[registers] robot={sku}({name}) declares "
                            f"family={fam!r} which is not in "
                            f"families_catalog.json - add the family to the "
                            f"catalog or correct the SKU declaration"
                        )
                    else:
                        problems.append(
                            f"  robot={sku}({name}) declares family={fam!r} "
                            f"which is not in families_catalog.json - add "
                            f"the family to the catalog or correct the SKU "
                            f"declaration"
                        )
                if not is_user_layer:
                    # Skip Rule 3 (IR role validation) — list_roles_for_families
                    # would silently return an empty role set for an unknown
                    # family and every ir_role would then fail "not in the IR
                    # canonical", drowning out the real C5 message.
                    continue

            valid_role_ids = {r["id"] for r in ir.list_roles_for_families(*families)}

            # Rule 3: every declared joint / body has a non-empty ir_role that
            # lives in the IR canonical for this robot's families.
            for fmt in _FORMATS:
                for kind, block_holder in (("joint", joints_pf), ("body", bodies_pf)):
                    block = block_holder.get(fmt)
                    if not isinstance(block, dict):
                        continue
                    for uid, spec in block.items():
                        ir_role = (spec or {}).get("ir_role", "")
                        role_str = str(ir_role).strip() if ir_role is not None else ""
                        if not role_str:
                            if is_user_layer:
                                log_warning(
                                    f"[registers] robot={sku}({name}) "
                                    f"{fmt}.{kind}={uid} ir_role is empty - "
                                    f"open the canvas Robot node body_mapping "
                                    f"table to assign it before training"
                                )
                            else:
                                problems.append(
                                    f"  robot={sku}({name}) {fmt}.{kind}={uid} "
                                    f"ir_role is empty; every {kind} must explicitly "
                                    f"map to an IR canonical role for families={families}"
                                )
                            continue
                        # Sentinel: user explicitly marked this entry as
                        # "(Out of Scope)" via the IR-role assignment
                        # dialog. Treat as resolved — don't warn.
                        if role_str == robots.OUT_OF_SCOPE_IR_ROLE:
                            continue
                        if role_str not in valid_role_ids:
                            if is_user_layer:
                                log_warning(
                                    f"[registers] robot={sku}({name}) "
                                    f"{fmt}.{kind}={uid} ir_role={role_str!r} "
                                    f"is not in the IR canonical for "
                                    f"families={families} - re-assign in the "
                                    f"canvas Robot node body_mapping table"
                                )
                            else:
                                problems.append(
                                    f"  robot={sku}({name}) {fmt}.{kind}={uid} "
                                    f"ir_role={role_str!r} is not in the IR canonical "
                                    f"for families={families}"
                                )

        # Rule 4: inheritance integrity (model_family / variant_label / inherits_from).
        # Runs after Rules 1–3 because those operate on the merged view, so by
        # the time we reach here every variant already saw its parent's joints
        # for IR-role validation.
        for sku in robots.list_skus():
            entry = robots.get_robot(sku)
            raw = robots.get_raw_robot(sku) or {}
            if entry is None:
                continue
            parent_sku = raw.get("inherits_from")
            if not parent_sku:
                continue
            parent = robots.get_robot(parent_sku)
            parent_raw = robots.get_raw_robot(parent_sku) or {}
            # 4a parent must exist
            if parent is None:
                problems.append(
                    f"  robot={sku} inherits_from={parent_sku!r} but parent is "
                    f"not registered"
                )
                continue
            # 4b chains forbidden
            if parent_raw.get("inherits_from"):
                problems.append(
                    f"  robot={sku} inherits_from={parent_sku} which is itself "
                    f"a variant (inherits_from={parent_raw.get('inherits_from')!r}); "
                    f"inheritance chains are not allowed"
                )
                continue
            # 4c model_family must match parent's
            v_family = raw.get("model_family") or raw.get("model")
            p_family = parent_raw.get("model_family") or parent_raw.get("model")
            if v_family and p_family and str(v_family).lower() != str(p_family).lower():
                problems.append(
                    f"  robot={sku} model_family={v_family!r} does not match "
                    f"parent {parent_sku} model_family={p_family!r}"
                )
            # 4d warn (not error) when families diverge from parent
            v_fams = set(raw.get("families", []) or [])
            p_fams = set(parent_raw.get("families", []) or [])
            if v_fams and p_fams and v_fams != p_fams:
                log_warning(
                    f"[registers] robot={sku} families={sorted(v_fams)} diverges "
                    f"from parent {parent_sku} families={sorted(p_fams)}; IR-role "
                    f"validation runs against the variant's own families"
                )

        # Rule 5: pd_groups integrity — every group regex covers ≥1 IR role
        # for its family, no two groups overlap, every defined group has a
        # default (omega_n, zeta) pair. Delegates to pd_groups.validate().
        from . import pd_groups
        pd_problems = pd_groups.validate()
        if pd_problems:
            problems.extend(pd_problems)

        # Rule 7: families catalog self-consistency — every non-null
        # inherits_from target must be a declared family (C2) and the
        # inheritance graph must be acyclic (C3). Per-entry schema (C1)
        # raises at families.load() so by now those passed. Delegates to
        # families.validate(). The SKU-side C4/C5 are already enforced
        # inline in the Rule 3 loop above.
        family_problems = families_module.validate()
        if family_problems:
            problems.extend(family_problems)

        # Rule 8: symmetry catalog cross-family partition.
        # Delegates to symmetry.validate() — S1 (partition closure per
        # family IR slice), S3 (referenced role exists in IR slice), S4
        # (alias_of target exists + no cycles). Per-entry schema (S2/S5
        # / L*) raises at symmetry.load() so by now those passed.
        # Symmetry data is family-keyed; intentionally NOT cross-checked
        # against robots SKUs.
        from . import symmetry
        sym_problems = symmetry.validate()
        if sym_problems:
            problems.extend(sym_problems)

        # Rule 9: pd_calibration_canaries catalog cross-family integrity.
        # Delegates to pd_calibration_canaries.validate() — S1 (every
        # canary group id is a defined pd_groups group for the family),
        # S2 (every catalog family is in families_catalog.json), plus
        # alias_of target / cycle defence. Per-entry schema (L1-L5) and
        # the asymmetric-overlay subset rule were enforced at
        # pd_calibration_canaries.load() / merge_user_extensions(); not
        # re-checked here. Canary data is family-keyed, intentionally
        # NOT cross-checked against robots SKUs.
        from . import pd_calibration_canaries
        canary_problems = pd_calibration_canaries.validate()
        if canary_problems:
            problems.extend(canary_problems)

        # Rule 10: amp_obs_templates catalog cross-family integrity.
        # Delegates to amp_obs_templates.validate() — A1 (every term
        # name registered in application.training.amp.obs_terms),
        # A2 (every catalog family is in families_catalog.json), A3
        # (alias_of target exists + no cycles). A4 fires as
        # log_warning, not a problem-list entry, so the omission of a
        # family from the obs-templates catalog never blocks boot.
        # Per-entry schema (L1-L5) was enforced at amp_obs_templates.
        # load() / merge_user_extensions(); not re-checked here. Obs
        # templates are family-keyed, intentionally NOT cross-checked
        # against robots SKUs.
        from . import amp_obs_templates
        amp_obs_problems = amp_obs_templates.validate()
        if amp_obs_problems:
            problems.extend(amp_obs_problems)

        # Rule 11: gait_commands catalog cross-family integrity.
        # Delegates to gait_commands.validate() -- G1 (non-empty
        # class_name), G2 (phase_count == len(phase_names)), G3
        # (phase_obs_dim == 2 * phase_count), G4 (alias_of target exists
        # + no cycles), G5 (every catalog family is in
        # families_catalog.json). G6 fires as log_warning, not a
        # problem-list entry, so the omission of a family from the
        # gait_commands catalog never blocks boot (manipulator /
        # wheeled / generic are not gait-trainable by construction).
        # Per-entry schema (L1-L7) was enforced at gait_commands.load()
        # / merge_user_extensions(); not re-checked here. Gait commands
        # are family-keyed, intentionally NOT cross-checked against
        # robots SKUs.
        from . import gait_commands
        gait_problems = gait_commands.validate()
        if gait_problems:
            problems.extend(gait_problems)

        # Rule 12: init_pose_presets catalog cross-family integrity.
        # Delegates to init_pose_presets.validate() -- I1 (preset name
        # uniqueness), I2 (base_pos 3-tuple float), I3 (alias_of target
        # exists + no cycles), I4 (every catalog family is in
        # families_catalog.json), I5 (user-overlay safety belt: the
        # family layer is name + base_pos by design; per-SKU
        # joint_pos_by_ir goes to robots_canonical). I6 fires as
        # log_warning, not a problem-list entry, so a families_catalog
        # family with no init pose preset entry never blocks boot.
        # Per-entry schema (L1-L9) was enforced at
        # init_pose_presets.load() / merge_user_extensions(); not
        # re-checked here. Init pose presets are family-keyed,
        # intentionally NOT cross-checked against robots SKUs --
        # SKU-level joint angle validation happens at Rule 3
        # (joints[*].ir_role) for the per-format joints table.
        from . import init_pose_presets
        ipp_problems = init_pose_presets.validate()
        if ipp_problems:
            problems.extend(ipp_problems)

        # Rule 6: physics_per_format / body_inertials_per_format /
        # joint_frames_per_format cache integrity — every joint/body NAME in
        # these caches must exist in the corresponding ``joints_per_format`` /
        # ``bodies_per_format`` block (rule §11 schema completeness: a Dump
        # writes both halves in lockstep, so a drift between them is a bug
        # in the writer or a hand-edited overlay). Empty blocks are valid
        # (format-not-dumped or format-not-applicable).
        _DRIVE_NUMERIC_FIELDS = (
            "stiffness", "damping", "max_force",
        )
        for sku in robots.list_skus():
            entry = robots.get_robot(sku)
            if entry is None:
                continue
            name = entry.get("name", "")
            ppf = entry.get("physics_per_format", {}) or {}
            bipf = entry.get("body_inertials_per_format", {}) or {}
            jfpf = entry.get("joint_frames_per_format", {}) or {}
            joints_pf = entry.get("joints_per_format", {}) or {}
            bodies_pf = entry.get("bodies_per_format", {}) or {}
            for fmt in _FORMATS:
                # physics_per_format[fmt] keys must be a subset of
                # joints_per_format[fmt][*].name
                phys = ppf.get(fmt)
                if isinstance(phys, dict) and phys:
                    joint_names = {
                        str((spec or {}).get("name", "") or "")
                        for spec in (joints_pf.get(fmt) or {}).values()
                        if isinstance(spec, dict)
                    }
                    joint_names.discard("")
                    for jn, block in phys.items():
                        if jn not in joint_names:
                            problems.append(
                                f"  robot={sku}({name}) "
                                f"physics_per_format[{fmt!r}][{jn!r}] has no "
                                f"matching name in joints_per_format[{fmt!r}] — "
                                f"re-run Dump {fmt} to rebuild the cache "
                                f"consistently"
                            )
                            continue
                        if not isinstance(block, dict):
                            problems.append(
                                f"  robot={sku}({name}) "
                                f"physics_per_format[{fmt!r}][{jn!r}] is not "
                                f"a dict (got {type(block).__name__})"
                            )
                            continue
                        drive = block.get("drive") or {}
                        if not isinstance(drive, dict):
                            problems.append(
                                f"  robot={sku}({name}) "
                                f"physics_per_format[{fmt!r}][{jn!r}].drive "
                                f"is not a dict"
                            )
                            continue
                        for k in _DRIVE_NUMERIC_FIELDS:
                            v = drive.get(k)
                            if v is not None:
                                try:
                                    float(v)
                                except (TypeError, ValueError):
                                    problems.append(
                                        f"  robot={sku}({name}) "
                                        f"physics_per_format[{fmt!r}][{jn!r}]"
                                        f".drive.{k}={v!r} is not numeric"
                                    )
                # body_inertials_per_format[fmt] keys must be a subset of
                # bodies_per_format[fmt][*].name (same lockstep contract).
                inert = bipf.get(fmt)
                if isinstance(inert, dict) and inert:
                    body_names = {
                        str((spec or {}).get("name", "") or "")
                        for spec in (bodies_pf.get(fmt) or {}).values()
                        if isinstance(spec, dict)
                    }
                    body_names.discard("")
                    for bn in inert:
                        if bn not in body_names:
                            problems.append(
                                f"  robot={sku}({name}) "
                                f"body_inertials_per_format[{fmt!r}][{bn!r}] "
                                f"has no matching name in bodies_per_format"
                                f"[{fmt!r}] — re-run Dump {fmt}"
                            )
                frames = jfpf.get(fmt)
                if isinstance(frames, dict) and frames:
                    joint_names = {
                        str((spec or {}).get("name", "") or "")
                        for spec in (joints_pf.get(fmt) or {}).values()
                        if isinstance(spec, dict)
                    }
                    joint_names.discard("")
                    for jn in frames:
                        if jn not in joint_names:
                            problems.append(
                                f"  robot={sku}({name}) "
                                f"joint_frames_per_format[{fmt!r}][{jn!r}] "
                                f"has no matching name in joints_per_format"
                                f"[{fmt!r}] — re-run Dump {fmt}"
                            )

        if problems:
            msg = "[registers] validate failed:\n" + "\n".join(problems)
            log_error(msg)
            raise RegistryValidationError(msg)
        log_success("[registers] validate passed")

    @classmethod
    def reload(cls) -> None:
        """Drop every per-domain cache then re-run :meth:`load_all`.

        Each sub-registry keeps its own ``_state["loaded"]`` flag, so just
        flipping ``cls._loaded`` is not enough — the per-domain ``load()``
        short-circuits on its own cache. We reset both layers so the
        ``user_extensions`` overlay (e.g. ``robots_custom.json``) is re-read
        from disk and stale entries are dropped.
        """
        cls._loaded = False
        cls._summary.clear()
        for mod_name in cls.DOMAINS:
            try:
                mod = __import__(f"{__name__}.{mod_name}", fromlist=[mod_name])
            except Exception:
                continue
            mod_state = getattr(mod, "_state", None)
            if not isinstance(mod_state, dict):
                continue
            mod_state["loaded"] = False
            # Drop merged-in user overlays so prior persists/deletes do not
            # linger across reload boundaries.
            if "user_extensions" in mod_state:
                ue = mod_state["user_extensions"]
                if isinstance(ue, dict):
                    ue.clear()
                elif isinstance(ue, list):
                    ue.clear()
        cls.load_all()

    @classmethod
    def summary(cls) -> Dict[str, int]:
        """每个子注册表的条目数（调试用）/ Per-domain entry count for diagnostics."""
        return dict(cls._summary)

    @classmethod
    def is_loaded(cls) -> bool:
        return cls._loaded


__all__ = [
    "RegistryHub",
    "RegistryValidationError",
    "build_sku",
    "resolve_robot_sku",
    "resolve_joint_sku",
]
