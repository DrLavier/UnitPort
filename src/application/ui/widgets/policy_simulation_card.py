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
  - Run：``MujocoReviewTask(asset.path, sku, scene_id)`` →
    ``get_tasks_manager().submit(...)``（仅 mujoco 系；isaac_lab 占位 warning）

持久化：选中项写 ``user.ini[SimConfig]``，重启后还原。

颜色与字体严格走 ``Config.get_color`` / ``Config.get_font_size``，禁止 hex 字面量。
"""

from __future__ import annotations

from typing import List, Optional

from PyQt6.QtCore import Qt, pyqtSignal
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
    read_data,
    setButton,
    setComboBox,
    tr,
)

from application.service.robot_init_poses import InitPoseOverride
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
    The SKU-only contract (CLAUDE.md §1.6) makes manifest the single source
    of truth for the robot identity bound to a policy.
    """
    manifest_path = bundle_dir / "manifest.yaml"
    if not manifest_path.exists():
        return ""
    try:
        raw = read_data(manifest_path)
    except Exception as exc:                           # pragma: no cover
        log_warning(f"[mission.sim] manifest read failed: {exc}")
        return ""
    if not isinstance(raw, dict):
        return ""
    robot = raw.get("robot")
    if not isinstance(robot, dict):
        return ""
    return str(robot.get("sku") or "").strip()


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
        self._refresh_policies()

    def _on_training_run_finished(self, _run_id: str, _success: bool) -> None:
        self._refresh_policies()
        self._refresh_run_enabled()

    def _on_policy_changed(self, _idx: int) -> None:
        data = self._cb_policy.currentData()
        if isinstance(data, PolicyAsset):
            Config.set_value("SimConfig", "last_policy_id", data.policy_id)
            Config.set_value(
                "SimConfig", "last_policy_backend", data.backend_id,
            )
        self._refresh_run_enabled()
        self._refresh_init_pose_sku()

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
        # Robot Config is the single source of truth for which robot
        # to simulate; the bundle manifest is only a fallback for the
        # no-canvas-loaded case.
        sku = self._robot_sku or _read_bundle_sku(asset.path)
        if not sku:
            log_warning(
                f"[mission.sim] aborted — no robot selected in Robot Config "
                f"and bundle {asset.policy_id!r} has no robot.sku in manifest.yaml"
            )
            return
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
            self.sim_run_requested.emit(asset, scene_id, engine_id)
        else:
            # isaac_lab review: simulation/ runtime has no isaac_sim subpkg
            # yet (Stage 3+). Surface the gap clearly instead of failing
            # silently.
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

        SKU resolution mirrors :meth:`_dispatch_run`: Robot Config is the
        primary source of truth (plumbed in via :meth:`set_robot_sku`),
        with the policy bundle's manifest as a fallback for the
        no-canvas-loaded case. Submits a :class:`RobotReviewTask` that
        opens a passive MuJoCo viewer posed at the MJCF's keyframe.
        ``live_physics`` controls whether physics is stepped during the
        preview (gravity-only, ``ctrl=0``).
        """
        sku = self._robot_sku or _read_bundle_sku(asset.path)
        if not sku:
            log_warning(
                f"[mission.sim] review aborted — no robot selected in Robot "
                f"Config and bundle {asset.policy_id!r} has no robot.sku "
                f"in manifest.yaml"
            )
            return
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
            self.sim_review_requested.emit(
                asset, scene_id, engine_id, live_physics,
            )
        else:
            log_warning(
                f"[mission.sim] backend {engine_id} review not wired yet"
            )


__all__ = ["PolicySimulationCard"]
