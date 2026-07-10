# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""PolicySimulationCard — mission_panel 中卡片：策略仿真.

横向 1 个子区段：
  1. _SimConfigSection — Policy / Play Ground / Backend 三个下拉 + Run 按钮

数据源：
  - Policy 列表：``training_assets.get_training_assets().policies()``
    （扫 ``projects/<proj>/training/exported/<backend>/<policy>/``；
    项目切换时自动重扫）
  - Play Ground 列表：``registers.backends.list_scenes()``，空时回退 flat_ground
  - Backend 列表：``registers.backends.list_installed()``，过滤仿真后端白名单
    `{mujoco, sb3_mujoco, isaac_lab}` 且 ``available=True``
  - SKU：``robot_assets.selected_robot_sku()``
  - Run：``MujocoReviewTask(asset.path, sku, scene_id)``（mujoco）或
    ``IsaacSimReviewTask(...)``（isaac_lab）→ ``get_tasks_manager().submit(...)``。
    两套仿真后端均已接线；init pose override 在两端同样生效。

持久化：选中项写 ``user.ini[SimConfig]``，重启后还原。

颜色与字体严格走 ``Config.get_color`` / ``Config.get_font_size``，禁止 hex 字面量。
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor
from PyQt6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from unitport_sdk import (
    Config,
    I18nLabel,
    LaviComboBox,
    Task,
    get_tasks_manager,
    i18n_bind,
    log_info,
    log_warning,
    setButton,
    setComboBox,
    tr,
)

from application.service.robot_init_poses import InitPoseOverride
from application.service.runtime.policy.bundle_loader import BundleLoader
from application.service.runtime.simulation.mujoco.review_session import (
    MujocoReviewTask,
)
from application.service.runtime.simulation.mujoco.robot_review_session import (
    RobotReviewTask,
)
from application.service.signals import get_app_signals
from application.service.training_assets import (
    PolicyAsset,
    get_training_assets,
)
from application.ui.widgets.init_pose_subsection import InitPoseSubsection
from application.ui.widgets.sim_section import FormRow, SectionFrame
from registers import backends as backends_reg


def _read_bundle_sku(bundle_dir) -> str:
    """Extract ``robot.sku`` from a bundle's ``manifest.yaml``.

    Returns an empty string when the file is missing, unparseable, or has no
    ``robot.sku`` field — callers treat empty as "abort with warning".

    ``manifest.robot.sku`` is the **frozen canvas Robot Node value at
    training time** — the only valid robot for replaying this policy.
    Replay code paths must source SKU from here (or refuse the run),
    NOT from the *current* canvas Robot Node (which the user may have
    moved to a different robot since training). The single definition
    source is the canvas Robot Node; the bundle just preserves its
    training-time snapshot.

    Goes through :class:`BundleLoader` (fresh ``open()`` + ``yaml.safe_load``)
    rather than ``read_data`` so a re-exported manifest is always re-parsed.
    ``read_data`` caches by absolute path with no mtime check, which silently
    returned a previous export's SKU after the bundle was overwritten and
    produced inverted "trained for X / canvas is Y" warnings versus the
    fresh read PolicyRunner performs at load time.
    """
    manifest_path = bundle_dir / "manifest.yaml"
    if not manifest_path.exists():
        return ""
    try:
        raw = BundleLoader._parse_manifest(bundle_dir)
    except Exception as exc:                           # pragma: no cover
        log_warning(f"[mission.sim] manifest read failed: {exc}")
        return ""
    robot = raw.get("robot")
    if not isinstance(robot, dict):
        return ""
    return str(robot.get("sku") or "").strip()


def _read_bundle_commands(bundle_dir) -> "Optional[dict]":
    """Extract ``deploy_contract.commands`` from a bundle's manifest.

    Returns ``None`` when missing / empty / unreadable — the Controller
    panel then shows the "no contract" hint. Same fresh-parse rationale
    as :func:`_read_bundle_sku` (no read_data mtime-less cache)."""
    manifest_path = bundle_dir / "manifest.yaml"
    if not manifest_path.exists():
        return None
    try:
        raw = BundleLoader._parse_manifest(bundle_dir)
    except Exception as exc:                           # pragma: no cover
        log_warning(f"[mission.sim] manifest read failed: {exc}")
        return None
    contract = raw.get("deploy_contract")
    commands = contract.get("commands") if isinstance(contract, dict) else None
    return commands if isinstance(commands, dict) and commands else None


def _robot_display_name(sku: str) -> str:
    """Best-effort human label for a SKU, for error messages.

    Returns the registered display name when ``sku`` is known, or the SKU
    itself (suffixed with ``[unregistered]``) when not. Never raises — we
    use this purely to make the SoT-mismatch error message readable.
    """
    sku = (sku or "").strip()
    if not sku:
        return "<empty>"
    try:
        from registers.robots import get_robot_spec
        spec = get_robot_spec(sku)
        if spec is not None and getattr(spec, "name", None):
            return f"{spec.name} [{sku}]"
    except Exception:
        pass
    return f"{sku} [unregistered]"


# ---------------------------------------------------------------------------
# Sim Config section
# ---------------------------------------------------------------------------


class _SimConfigSection(SectionFrame):
    """3 下拉 (Policy / Play Ground / Backend) + Run 按钮."""

    # (asset, scene_id, engine_id) — emitted on Run click.
    run_clicked = pyqtSignal(object, str, str)
    # (asset, scene_id, engine_id, live_physics) — emitted on Review Robot click.
    # ``asset`` is the currently-selected PolicyAsset (used to resolve the
    # robot SKU from manifest.yaml, same path as Run); ``live_physics``
    # mirrors the "Preview Physic" checkbox.
    review_clicked = pyqtSignal(object, str, str, bool)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(
            "mission.sim.section.simconfig", "Sim Config", parent
        )

        # 训练完成后被高亮的 policy ((policy_id, backend_id))。
        # _refresh_policies() 会在重建条目时给匹配项设 ForegroundRole=safe_zone,
        # popup 行 + 闭合态显示文本一起变色(配套 SDK LaviComboBox 的 delegate
        # 与 _build_qss override)。project_changed 时清空。
        self._highlighted_policy_key: Optional[Tuple[str, str]] = None

        # Three empty combos; populate via _refresh_* helpers below.
        self._cb_policy: LaviComboBox = setComboBox(
            [], height=28, i18n=False, parent=self._body
        )
        self._cb_field: LaviComboBox = setComboBox(
            [], height=28, i18n=False, parent=self._body
        )
        self._cb_backend: LaviComboBox = setComboBox(
            [], height=28, i18n=False, parent=self._body
        )

        rows: List[FormRow] = [
            FormRow(
                "mission.sim.field.policy", "Policy",
                self._cb_policy, self._body,
            ),
            FormRow(
                "mission.sim.field.play_ground", "Play Ground",
                self._cb_field, self._body,
            ),
            FormRow(
                "mission.sim.field.backend", "Backend",
                self._cb_backend, self._body,
            ),
        ]
        for r in rows:
            self.body_layout().addWidget(r, 0)
        self._rows = rows

        # ---- Init pose override subsection ------------------------------
        # Inline editor between the dropdowns and the action row. Off by
        # default — when the user enables it, both Review Robot and Run
        # spawn the robot from the edited values instead of bundle
        # defaults. SKU plumbing (set_parent_robot_sku / policy change)
        # drives the joint form rebuild.
        self._parent_robot_sku: str = ""
        self._init_pose = InitPoseSubsection(self._body)
        self.body_layout().addWidget(self._init_pose, 0)

        # ---- Action row -------------------------------------------------
        # Layout:  [Preview Physic ☐]  [Review Robot]  ──stretch──  [Run]
        # "Preview Physic" controls whether RobotReviewTask steps physics
        # live during the asset preview; "Review Robot" launches the
        # robot-only viewer (no policy needed) using the currently
        # selected backend at the keyframe pose.
        self._cb_preview_physic = QCheckBox(self._body)
        self._cb_preview_physic.setObjectName("missionSimPreviewPhysic")
        self._cb_preview_physic.setCursor(Qt.CursorShape.PointingHandCursor)
        i18n_bind(
            self._cb_preview_physic, "setText",
            "mission.sim.field.preview_physic", "Preview Physic",
        )
        i18n_bind(
            self._cb_preview_physic, "setToolTip",
            "mission.sim.field.preview_physic.tip",
            "Step physics live during Review Robot (gravity-only, no actuators)",
        )
        self._cb_preview_physic.setChecked(
            bool(Config.get_value(
                "SimConfig", "preview_physic", False, value_type=bool,
            ))
        )

        self._btn_review = setButton(
            "mission.sim.btn.review_robot", 120, 30,
            kind="normal", spec="none", default="Review Robot",
            parent=self._body,
        )
        self._btn_run = setButton(
            "mission.sim.btn.run", 92, 30,
            kind="normal", spec="save", default="Run", parent=self._body,
        )
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.setSpacing(8)
        btn_row.addStretch(1)
        btn_row.addWidget(
            self._cb_preview_physic, 0, Qt.AlignmentFlag.AlignVCenter,
        )
        btn_row.addWidget(self._btn_review, 0, Qt.AlignmentFlag.AlignVCenter)
        btn_row.addWidget(self._btn_run, 0)
        self.body_layout().addLayout(btn_row)
        self.body_layout().addStretch(1)

        # Initial populate. Order matters only for persistence restoration —
        # all three reads pull from independent stores.
        self._refresh_policies()
        self._refresh_scenes()
        self._refresh_backends()
        self._refresh_run_enabled()

        # Wiring.
        self._cb_policy.currentIndexChanged.connect(self._on_policy_changed)
        self._cb_field.currentIndexChanged.connect(self._on_field_changed)
        self._cb_backend.currentIndexChanged.connect(self._on_backend_changed)
        self._cb_preview_physic.toggled.connect(self._on_preview_physic_toggled)
        self._btn_run.clicked.connect(self._on_run)
        self._btn_review.clicked.connect(self._on_review)

        # External signals. Robot SKU is read from the bundle manifest at Run
        # time, not from the global robot selection — sim is determined by the
        # policy, not by whichever robot is currently "selected" in the UI.
        sigs = get_app_signals()
        sigs.project_changed.connect(self._on_project_changed)
        # TrainingAssetsCache itself listens to training_run_finished and will
        # have re-scanned by the time this slot fires (singleton was wired at
        # first _refresh_policies() above, before us). Just re-pull.
        sigs.training_run_finished.connect(self._on_training_run_finished)

        # Now that the policy combo is populated AND signals are wired, push
        # the resolved SKU into the init pose subsection so the preset
        # dropdown lights up immediately instead of waiting for the user
        # to re-select the policy. The combo's currentIndexChanged was
        # silent during _refresh_policies (signals not connected yet).
        self._refresh_init_pose_sku()
        # Broadcast the initially-selected policy's command contract so the
        # Controller panel renders + routes its channel bindings right away,
        # without waiting for a live-review to start (currentIndexChanged was
        # silent during the pre-wiring _refresh_policies).
        self._broadcast_policy_contract()

    # ------------------------------------------------------------------
    # Populate helpers
    # ------------------------------------------------------------------
    def _refresh_policies(self) -> None:
        prev = self._cb_policy.currentData()
        prev_id = prev.policy_id if isinstance(prev, PolicyAsset) else (
            Config.get_value("SimConfig", "last_policy_id", "")
        )
        prev_backend = prev.backend_id if isinstance(prev, PolicyAsset) else (
            Config.get_value("SimConfig", "last_policy_backend", "")
        )

        self._cb_policy.blockSignals(True)
        self._cb_policy.clear()
        assets = get_training_assets().policies()
        for asset in assets:
            label = (
                f"{asset.policy_id} [{asset.backend_id}] "
                f"{asset.algorithm or '?'}"
            )
            self._cb_policy.addItem(label, asset)

        if not assets:
            self._cb_policy.setPlaceholderText(
                tr("mission.sim.policy.empty", "No exported policies")
            )
            self._cb_policy.setCurrentIndex(-1)
        else:
            idx = self._find_policy_index(str(prev_id), str(prev_backend))
            self._cb_policy.setCurrentIndex(idx if idx >= 0 else 0)
        self._cb_policy.blockSignals(False)

        # 重建后重新涂高亮 —— combo.clear() 会丢掉 ForegroundRole。
        self._apply_policy_highlight()

        self._refresh_run_enabled()

    def _refresh_scenes(self) -> None:
        prev = str(self._cb_field.currentData() or "")
        if not prev:
            prev = str(Config.get_value("SimConfig", "last_scene", ""))

        try:
            scenes = backends_reg.list_scenes() or []
        except Exception as exc:                       # pragma: no cover
            log_warning(f"[mission.sim] list_scenes failed: {exc}")
            scenes = []
        if not scenes:
            scenes = [{"id": "flat_ground", "display_name": "flat_ground"}]

        self._cb_field.blockSignals(True)
        self._cb_field.clear()
        for s in scenes:
            sid = str(s.get("id") or s.get("scene_id") or "")
            if not sid:
                continue
            label = str(s.get("display_name") or sid)
            self._cb_field.addItem(label, sid)

        idx = self._find_data_index(self._cb_field, prev)
        self._cb_field.setCurrentIndex(idx if idx >= 0 else 0)
        self._cb_field.blockSignals(False)

    def _refresh_backends(self) -> None:
        prev = str(self._cb_backend.currentData() or "")
        if not prev:
            prev = str(Config.get_value("SimConfig", "last_backend", ""))

        try:
            infos = backends_reg.list_sim_backends()
        except Exception as exc:                       # pragma: no cover
            log_warning(f"[mission.sim] list_sim_backends failed: {exc}")
            infos = []

        self._cb_backend.blockSignals(True)
        self._cb_backend.clear()
        for info in infos:
            self._cb_backend.addItem(info.display_name, info.id)

        if not infos:
            self._cb_backend.setPlaceholderText(
                tr("mission.sim.backend.empty", "No simulation backend")
            )
            self._cb_backend.setCurrentIndex(-1)
        else:
            idx = self._find_data_index(self._cb_backend, prev)
            # First entry is the registry-declared preferred default.
            self._cb_backend.setCurrentIndex(idx if idx >= 0 else 0)
        self._cb_backend.blockSignals(False)

        self._refresh_run_enabled()

    def _refresh_run_enabled(self) -> None:
        has_policy = isinstance(self._cb_policy.currentData(), PolicyAsset)
        has_field = bool(self._cb_field.currentData())
        has_backend = bool(self._cb_backend.currentData())
        ready = has_policy and has_field and has_backend
        self._btn_run.setEnabled(ready)
        # Review Robot reads the SKU from the policy bundle manifest too
        # (same resolution path as Run), so it shares the policy gate.
        self._btn_review.setEnabled(ready)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _find_policy_index(self, policy_id: str, backend_id: str) -> int:
        if not policy_id:
            return -1
        # Prefer exact (id, backend) match; fall back to id-only match.
        fallback = -1
        for i in range(self._cb_policy.count()):
            data = self._cb_policy.itemData(i)
            if not isinstance(data, PolicyAsset):
                continue
            if data.policy_id == policy_id:
                if backend_id and data.backend_id == backend_id:
                    return i
                if fallback < 0:
                    fallback = i
        return fallback

    @staticmethod
    def _find_data_index(cb: LaviComboBox, value: str) -> int:
        if not value:
            return -1
        for i in range(cb.count()):
            if str(cb.itemData(i) or "") == value:
                return i
        return -1

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------
    def _on_project_changed(self, _info) -> None:
        # TrainingAssetsCache is already subscribed to project_changed and will
        # have rebuilt by the time this slot fires (singleton is connected
        # before us at first get_training_assets() call). Just re-pull.
        # 切项目 → 上一项目的"最新训练 policy"高亮在新项目里无意义,清空。
        self._highlighted_policy_key = None
        self._refresh_policies()

    def _on_training_run_finished(self, _run_id: str, success: bool) -> None:
        # 训练成功完成 → 把 combo 切到刚刚导出的 policy(按 mtime 排序后的头部),
        # 并把它标记为高亮项,_refresh_policies() 重建 items 时会自动涂色。
        # 失败的 run 不动选择 / 高亮 —— 没有新 bundle 产出,任何切换都是噪声。
        # mtime 比对:如果本次成功但没真的导出新 bundle(例如 Export Node 未配),
        # 头部 mtime 不会前进,这种情况下也保持原状,避免把陈旧条目当成"新训练"。
        switched = False
        if success:
            prev_max_mtime = 0.0
            for i in range(self._cb_policy.count()):
                data = self._cb_policy.itemData(i)
                if isinstance(data, PolicyAsset) and data.mtime > prev_max_mtime:
                    prev_max_mtime = data.mtime
            assets = get_training_assets().policies()
            if assets and assets[0].mtime > prev_max_mtime:
                latest = assets[0]
                self._highlighted_policy_key = (
                    latest.policy_id, latest.backend_id,
                )
                switched = True
        self._refresh_policies()
        if switched and self._highlighted_policy_key is not None:
            idx = self._find_policy_index(
                self._highlighted_policy_key[0],
                self._highlighted_policy_key[1],
            )
            if idx >= 0:
                self._cb_policy.setCurrentIndex(idx)
        self._refresh_run_enabled()

    def _apply_policy_highlight(self) -> None:
        """Paint the highlighted policy row's foreground with ``safe_zone``.

        Driven by ``_highlighted_policy_key``; clears any prior coloring on
        every call so a stale highlight cannot linger after the key changes.
        Both popup row and closed-combo display text pick up the override
        via the SDK delegate / ``LaviComboBox._build_qss`` integration.
        """
        model = self._cb_policy.model()
        # 先把所有项的 ForegroundRole 重置为 NoBrush —— SDK delegate / _build_qss
        # 都以 brush.style() == NoBrush 作为"无 override"的判定,显式 reset 避免
        # 上一次的高亮在 key 变化后残留。
        empty_brush = QBrush()
        for i in range(self._cb_policy.count()):
            item = model.item(i) if hasattr(model, "item") else None
            if item is None:
                continue
            item.setForeground(empty_brush)
        key = self._highlighted_policy_key
        if key is None:
            self._cb_policy.refresh_style()
            return
        idx = self._find_policy_index(key[0], key[1])
        if idx < 0:
            self._cb_policy.refresh_style()
            return
        item = model.item(idx) if hasattr(model, "item") else None
        if item is None:
            self._cb_policy.refresh_style()
            return
        item.setForeground(QBrush(QColor(Config.get_color("safe_zone"))))
        # _build_qss 读 currentIndex 的 ForegroundRole,若高亮项正好是当前
        # committed 项,闭合态文本颜色立即生效。
        self._cb_policy.refresh_style()

    def _on_policy_changed(self, _idx: int) -> None:
        data = self._cb_policy.currentData()
        if isinstance(data, PolicyAsset):
            Config.set_value("SimConfig", "last_policy_id", data.policy_id)
            Config.set_value(
                "SimConfig", "last_policy_backend", data.backend_id,
            )
        self._refresh_run_enabled()
        self._refresh_init_pose_sku()
        self._broadcast_policy_contract()

    def _broadcast_policy_contract(self) -> None:
        """Emit ``policy_contract_changed`` for the currently-selected policy so
        the Controller panel renders/routes its command-channel bindings from the
        loaded bundle's ``deploy_contract.commands`` — decoupled from live-review.
        No policy selected → ``("", None)`` so the panel shows its unavailable state."""
        asset = self._cb_policy.currentData()
        sku, commands = "", None
        if isinstance(asset, PolicyAsset):
            try:
                sku = _read_bundle_sku(asset.path) or (self._robot_sku or "")
                commands = _read_bundle_commands(asset.path)
            except Exception as exc:                       # pragma: no cover
                log_warning(f"[mission.sim] contract broadcast read failed: {exc}")
                sku, commands = "", None
        try:
            from application.service.signals import get_app_signals
            get_app_signals().policy_contract_changed.emit(sku, commands)
        except Exception as exc:                            # pragma: no cover
            log_warning(f"[mission.sim] contract broadcast failed: {exc}")

    def _on_field_changed(self, _idx: int) -> None:
        sid = str(self._cb_field.currentData() or "")
        if sid:
            Config.set_value("SimConfig", "last_scene", sid)
        self._refresh_run_enabled()

    def _on_backend_changed(self, _idx: int) -> None:
        eid = str(self._cb_backend.currentData() or "")
        if eid:
            Config.set_value("SimConfig", "last_backend", eid)
        self._refresh_run_enabled()

    def _on_run(self) -> None:
        asset = self._cb_policy.currentData()
        scene_id = str(self._cb_field.currentData() or "")
        engine_id = str(self._cb_backend.currentData() or "")
        if not isinstance(asset, PolicyAsset):
            log_warning("[mission.sim] aborted — no policy selected")
            return
        if not scene_id or not engine_id:
            log_warning(
                "[mission.sim] aborted — scene or backend missing"
            )
            return
        self.run_clicked.emit(asset, scene_id, engine_id)

    def _on_review(self) -> None:
        asset = self._cb_policy.currentData()
        scene_id = str(self._cb_field.currentData() or "")
        engine_id = str(self._cb_backend.currentData() or "")
        if not isinstance(asset, PolicyAsset):
            log_warning("[mission.sim] review aborted — no policy selected")
            return
        if not scene_id or not engine_id:
            log_warning(
                "[mission.sim] review aborted — scene or backend missing"
            )
            return
        live = bool(self._cb_preview_physic.isChecked())
        self.review_clicked.emit(asset, scene_id, engine_id, live)

    def _on_preview_physic_toggled(self, checked: bool) -> None:
        Config.set_value("SimConfig", "preview_physic", bool(checked))

    # ------------------------------------------------------------------
    # Init pose override plumbing
    # ------------------------------------------------------------------
    def set_parent_robot_sku(self, sku: str) -> None:
        """Plumbed by the card whenever Robot Config's SKU changes.

        Robot Config wins over the bundle manifest fallback (matches the
        dispatch-time resolution order). Empty string falls back to the
        currently-selected policy's bundle SKU on the next refresh.
        """
        self._parent_robot_sku = str(sku or "")
        self._refresh_init_pose_sku()

    def _refresh_init_pose_sku(self) -> None:
        """Resolve the active SKU (parent > bundle manifest) and push to
        the init pose subsection so its joint form matches the robot.
        """
        sku = self._parent_robot_sku
        if not sku:
            asset = self._cb_policy.currentData()
            if isinstance(asset, PolicyAsset):
                sku = _read_bundle_sku(asset.path)
        self._init_pose.set_sku(str(sku or ""))

    def current_init_pose_override(self) -> Optional[InitPoseOverride]:
        """Snapshot of the init pose editor's current state.

        Returns ``None`` when the user has not enabled the override —
        callers should treat that as "use bundle defaults".
        """
        return self._init_pose.current_override()

    # ------------------------------------------------------------------
    # Theme
    # ------------------------------------------------------------------
    def apply_theme(self) -> None:
        super().apply_theme()
        for r in self._rows:
            r.apply_theme()
        for cb in (self._cb_policy, self._cb_field, self._cb_backend):
            refresher = getattr(cb, "refresh_style", None)
            if callable(refresher):
                refresher()
        self._style_preview_physic_checkbox()
        self._init_pose.apply_theme()

    def _style_preview_physic_checkbox(self) -> None:
        """Style the Preview Physic QCheckBox via theme slots (no hex)."""
        fg = Config.get_color("main_t1")
        border = Config.get_color("border_1")
        checked = Config.get_color("safe_zone")
        hover_bg = Config.get_color("hover_2")
        font_small = Config.get_font_size("size_small")
        self._cb_preview_physic.setStyleSheet(
            f"QCheckBox#missionSimPreviewPhysic {{ "
            f"color: {fg}; background-color: transparent; "
            f"font-size: {font_small}px; spacing: 6px; "
            f"padding: 2px 4px; border-radius: 3px; }}"
            f"QCheckBox#missionSimPreviewPhysic:hover {{ "
            f"background-color: {hover_bg}; }}"
            f"QCheckBox#missionSimPreviewPhysic::indicator {{ "
            f"width: 14px; height: 14px; "
            f"border: 1px solid {border}; border-radius: 3px; "
            f"background-color: transparent; }}"
            f"QCheckBox#missionSimPreviewPhysic::indicator:checked {{ "
            f"background-color: {checked}; border: 1px solid {checked}; "
            f"image: none; }}"
        )


# ---------------------------------------------------------------------------
# Card composer
# ---------------------------------------------------------------------------


class PolicySimulationCard(QFrame):
    """mission_panel 中卡片：策略仿真."""

    # (PolicyAsset, scene_id, engine_id) — emitted after a Task is submitted.
    sim_run_requested = pyqtSignal(object, str, str)
    # (PolicyAsset, scene_id, engine_id, live_physics) — emitted after a
    # RobotReviewTask is submitted via the "Review Robot" button.
    sim_review_requested = pyqtSignal(object, str, str, bool)

    _ACCENT_W = 4

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("missionSimCard")
        # SKU plumbed in from training_config_card.robot_sku_changed.
        # Run / Review Robot dispatch on this SKU (Robot Config is the
        # single source of truth for "which robot to simulate"). Falls
        # back to the policy bundle's manifest SKU only when the Robot
        # Config has no canvas / no robot node yet.
        self._robot_sku: str = ""
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(10)

        # Title bar with left accent stripe.
        title_host = QFrame(self)
        title_host.setObjectName("missionSimCardTitle")
        title_host.setFrameShape(QFrame.Shape.NoFrame)
        title_host.setFixedHeight(28)
        title_host.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        title_layout = QHBoxLayout(title_host)
        title_layout.setContentsMargins(8, 0, 8, 0)
        title_layout.setSpacing(8)
        self._title_label = I18nLabel(
            "mission.sim.title", default="Policy Simulation",
            parent=title_host,
        )
        self._title_label.setObjectName("missionSimCardTitleLabel")
        title_layout.addWidget(
            self._title_label, 0, Qt.AlignmentFlag.AlignVCenter
        )
        title_layout.addStretch(1)
        outer.addWidget(title_host, 0)

        body = QWidget(self)
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(8)

        self._sec_sim = _SimConfigSection(body)
        body_layout.addWidget(self._sec_sim, 1)
        outer.addWidget(body, 1)

        self._sec_sim.run_clicked.connect(self._dispatch_run)
        self._sec_sim.review_clicked.connect(self._dispatch_review)

        self.apply_theme()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------
    def set_robot_sku(self, sku: str) -> None:
        """Plumbed by MissionControlPanel from Robot Config's SKU changes.

        Empty string means "no robot configured" — dispatch will fall
        back to the policy bundle's manifest SKU so the buttons remain
        functional when no canvas / Robot Config exists yet.
        """
        self._robot_sku = str(sku or "")
        # Forward to the init pose subsection so its joint form rebuilds
        # against the new SKU's IR roles.
        self._sec_sim.set_parent_robot_sku(self._robot_sku)

    def apply_theme(self) -> None:
        bg_card = Config.get_color("bg_2")
        border = Config.get_color("border_1")
        accent = Config.get_color("safe_zone")
        title_color = Config.get_color("main_t1")
        font_normal = Config.get_font_size("size_normal")

        self.setStyleSheet(
            f"QFrame#missionSimCard {{ background-color: {bg_card}; "
            f"border: 1px solid {border}; border-radius: 8px; }}"
            f"QFrame#missionSimCardTitle {{ background: transparent; "
            f"border-left: {self._ACCENT_W}px solid {accent}; }}"
        )
        self._title_label.setStyleSheet(
            f"QLabel#missionSimCardTitleLabel {{ color: {title_color}; "
            f"font-size: {font_normal}px; font-weight: 600; "
            f"background: transparent; }}"
        )
        self._sec_sim.apply_theme()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------
    def _dispatch_run(
        self, asset: PolicyAsset, scene_id: str, engine_id: str,
    ) -> None:
        # SKU SoT for replay: ``bundle.manifest.robot.sku`` is the frozen
        # canvas-Robot-Node value from training time, and the policy
        # weights are bound to THAT robot's joint topology. The current
        # canvas Robot Node may have moved on (user is now working on a
        # different robot); using it would silently mis-map joints and
        # corrupt sim2sim. We use the bundle SKU and refuse the run
        # when the current canvas is bound to a different robot, so the
        # mismatch surfaces loudly rather than as a twitching simulation.
        bundle_sku = _read_bundle_sku(asset.path)
        if not bundle_sku:
            # No canvas + no bundle SKU = nothing to do.
            if not self._robot_sku:
                log_warning(
                    f"[mission.sim] aborted — bundle {asset.policy_id!r} "
                    f"has no robot.sku in manifest.yaml and no Robot "
                    f"Config is bound. Re-export the bundle from a "
                    f"project where the canvas Robot Node was configured."
                )
                return
            # Legacy bundle without manifest SKU: fall back to canvas SKU
            # as a last resort (PolicyRunner will validate compatibility).
            sku = self._robot_sku
        else:
            canvas_sku = (self._robot_sku or "").strip()
            if canvas_sku and canvas_sku != bundle_sku:
                log_warning(
                    f"[mission.sim] aborted — policy {asset.policy_id!r} "
                    f"was trained for robot {_robot_display_name(bundle_sku)}; "
                    f"the canvas Robot Config is currently bound to "
                    f"{_robot_display_name(canvas_sku)}. Bundles are "
                    f"SKU-locked: switch the canvas Robot Node back to "
                    f"the training robot to replay this policy, or pick "
                    f"a policy trained for the current robot."
                )
                return
            sku = bundle_sku
        if engine_id == "mujoco":
            override = self._sec_sim.current_init_pose_override()
            task: Task = MujocoReviewTask(
                asset.path, sku, scene_id,
                init_pose_override=override,
            )
            tid = get_tasks_manager().submit(task)
            log_info(
                f"[mission.sim] MujocoReviewTask submitted id={tid} "
                f"policy={asset.policy_id} scene={scene_id} sku={sku} "
                f"init_pose_override={'yes' if override is not None else 'no'}"
            )
            # Register the live-sim task so a system estop can cancel it (Slice 0).
            from application.service.runtime.session_controller import (
                get_session_controller,
            )
            get_session_controller().register_review_task(tid)
            # Broadcast the active policy's command contract so the
            # Controller panel can render contract-driven channel bindings
            # (per SKU) and install the matching axis routing for this
            # live-sim teleop session. None commands = legacy / no contract.
            try:
                from application.service.signals import get_app_signals
                get_app_signals().policy_review_started.emit(
                    sku, _read_bundle_commands(asset.path)
                )
            except Exception as exc:
                log_warning(f"[mission.sim] contract broadcast failed: {exc}")
            self.sim_run_requested.emit(asset, scene_id, engine_id)
        elif engine_id == "isaac_lab":
            # Bundle-only IsaacSim replay (CLAUDE.md §1.9): the subprocess
            # loads policy + deploy_contract from the bundle dir and builds
            # a minimal Kit scene from the SKU registry. Same shape as the
            # mujoco branch; init_pose_override is threaded through to the
            # launcher (parity with MujocoReviewTask).
            override = self._sec_sim.current_init_pose_override()
            try:
                from application.service.runtime.simulation.isaac_sim import (
                    IsaacSimReviewTask,
                )
            except Exception as exc:  # pragma: no cover - import guard
                log_warning(
                    f"[mission.sim] IsaacSimReviewTask import failed: {exc}"
                )
                return
            task = IsaacSimReviewTask(
                asset.path, sku, scene_id,
                init_pose_override=override,
            )
            tid = get_tasks_manager().submit(task)
            log_info(
                f"[mission.sim] IsaacSimReviewTask submitted id={tid} "
                f"policy={asset.policy_id} scene={scene_id} sku={sku} "
                f"init_pose_override={'yes' if override is not None else 'no'}"
            )
            # Register the live-sim task so a system estop can cancel it (Slice 0).
            from application.service.runtime.session_controller import (
                get_session_controller,
            )
            get_session_controller().register_review_task(tid)
            self.sim_run_requested.emit(asset, scene_id, engine_id)
        else:
            log_warning(
                f"[mission.sim] backend {engine_id} review not wired yet"
            )

    def _dispatch_review(
        self,
        asset: PolicyAsset,
        scene_id: str,
        engine_id: str,
        live_physics: bool,
    ) -> None:
        """Review-Robot dispatcher — robot-only viewer at the asset keyframe.

        SKU resolution mirrors :meth:`_dispatch_run`: ``manifest.robot.sku``
        is the frozen canvas-Robot-Node value at training time, and the
        only valid robot for this asset. We refuse when the current
        canvas Robot Config is bound to a different robot, rather than
        silently displaying the wrong robot.
        ``live_physics`` controls whether physics is stepped during the
        preview (gravity-only, ``ctrl=0``).
        """
        bundle_sku = _read_bundle_sku(asset.path)
        if not bundle_sku:
            if not self._robot_sku:
                log_warning(
                    f"[mission.sim] review aborted — bundle "
                    f"{asset.policy_id!r} has no robot.sku in "
                    f"manifest.yaml and no Robot Config is bound."
                )
                return
            sku = self._robot_sku
        else:
            canvas_sku = (self._robot_sku or "").strip()
            if canvas_sku and canvas_sku != bundle_sku:
                log_warning(
                    f"[mission.sim] review aborted — asset "
                    f"{asset.policy_id!r} ships robot "
                    f"{_robot_display_name(bundle_sku)}; the canvas "
                    f"Robot Config is currently bound to "
                    f"{_robot_display_name(canvas_sku)}. Switch the "
                    f"canvas Robot Node back to match the asset, or "
                    f"pick a different asset."
                )
                return
            sku = bundle_sku
        if engine_id == "mujoco":
            override = self._sec_sim.current_init_pose_override()
            task: Task = RobotReviewTask(
                sku, scene_id,
                live_physics=live_physics,
                init_pose_override=override,
            )
            tid = get_tasks_manager().submit(task)
            log_info(
                f"[mission.sim] RobotReviewTask submitted id={tid} "
                f"sku={sku} scene={scene_id} live_physics={live_physics} "
                f"init_pose_override={'yes' if override is not None else 'no'}"
            )
            from application.service.runtime.session_controller import (
                get_session_controller,
            )
            get_session_controller().register_review_task(tid)
            self.sim_review_requested.emit(
                asset, scene_id, engine_id, live_physics,
            )
        elif engine_id == "isaac_lab":
            # Pose-only IsaacSim viewer (no policy). ``live_physics`` is a
            # MuJoCo-only distinction (frozen viewer vs gravity step); the
            # IsaacSim pose review always holds the override pose under
            # implicit-PD with gravity on (≈ live_physics=True), so the flag
            # is intentionally not forwarded — not silently dropped.
            override = self._sec_sim.current_init_pose_override()
            try:
                from application.service.runtime.simulation.isaac_sim import (
                    IsaacSimPoseReviewTask,
                )
            except Exception as exc:  # pragma: no cover - import guard
                log_warning(
                    f"[mission.sim] IsaacSimPoseReviewTask import failed: {exc}"
                )
                return
            task = IsaacSimPoseReviewTask(
                sku, scene_id,
                init_pose_override=override,
            )
            tid = get_tasks_manager().submit(task)
            log_info(
                f"[mission.sim] IsaacSimPoseReviewTask submitted id={tid} "
                f"sku={sku} scene={scene_id} "
                f"init_pose_override={'yes' if override is not None else 'no'}"
            )
            from application.service.runtime.session_controller import (
                get_session_controller,
            )
            get_session_controller().register_review_task(tid)
            self.sim_review_requested.emit(
                asset, scene_id, engine_id, live_physics,
            )
        else:
            log_warning(
                f"[mission.sim] backend {engine_id} review not wired yet"
            )


__all__ = ["PolicySimulationCard"]
