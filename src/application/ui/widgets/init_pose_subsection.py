"""InitPoseSubsection — compact init pose override control for the Simulation card.

Card-side surface (intentionally tight):

* an enable checkbox (off → no override, ``current_override()`` returns None)
* a preset dropdown merging factory + user presets for the current SKU
* an action row: [Edit…] [Save As…] [Delete]

Base position + per-IR-role joint angles are edited in a modal dialog
(:class:`init_pose_editor_dialog.InitPoseEditorDialog`) opened by
Edit…. The subsection owns the canonical edited state — the dialog
just borrows it for the duration of a session, then writes results
back on OK.

All colors / fonts go through ``Config.get_color`` / ``Config.get_font_size``
(CLAUDE.md §1.5); all UI strings go through ``tr`` / ``i18n_bind`` /
``I18nLabel``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QSizePolicy,
    QWidget,
)

from unitport_sdk import (
    Config,
    I18nLabel,
    LaviComboBox,
    i18n_bind,
    log_info,
    log_warning,
    setButton,
    setComboBox,
    tr,
)

from application.service.robot_init_poses import (
    InitPoseOverride,
    InitPosePreset,
    SOURCE_FACTORY,
    SOURCE_USER,
    get_init_pose_service,
)
from application.ui.widgets.init_pose_editor_dialog import InitPoseEditorDialog
from application.ui.widgets.sim_section import SectionFrame

from registers import robots as _robots_registry


# Sentinel item shown when the user has hand-edited values that no longer
# match any saved preset. itemData stays None so we can tell it apart
# from real presets.
_CUSTOM_LABEL_KEY = "mission.sim.init_pose.custom"
_CUSTOM_LABEL_DEFAULT = "Custom (unsaved)"


class InitPoseSubsection(SectionFrame):
    """Compact init pose override control — see module docstring."""

    override_changed = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(
            "mission.sim.section.init_pose",
            "Init Pose Override",
            parent,
        )
        self._service = get_init_pose_service()
        self._current_sku: str = ""
        self._current_ir_roles: List[str] = []

        # Canonical edited state — the dialog reads from / writes to
        # these; current_override() snapshots them at dispatch time.
        # base_pos uses the IL stance default (z=0.40) until a preset
        # is applied; joint dict starts empty so to_bundle_order falls
        # back to the bundle's default_joint_pos.
        self._base_pos: Tuple[float, float, float] = (0.0, 0.0, 0.40)
        self._joint_pos_by_ir: Dict[str, float] = {}

        body = self.body_layout()

        # ── Enable checkbox ───────────────────────────────────────────
        self._cb_enable = QCheckBox(self.body_widget())
        self._cb_enable.setObjectName("missionSimInitPoseEnable")
        self._cb_enable.setCursor(Qt.CursorShape.PointingHandCursor)
        i18n_bind(
            self._cb_enable, "setText",
            "mission.sim.field.init_pose_enable",
            "Override init pose",
        )
        i18n_bind(
            self._cb_enable, "setToolTip",
            "mission.sim.field.init_pose_enable.tip",
            "Spawn the robot from the edited values instead of the bundle defaults",
        )
        body.addWidget(self._cb_enable, 0)

        # ── Preset selector row ──────────────────────────────────────
        preset_row = QWidget(self.body_widget())
        preset_layout = QHBoxLayout(preset_row)
        preset_layout.setContentsMargins(0, 0, 0, 0)
        preset_layout.setSpacing(6)
        preset_label = I18nLabel(
            "mission.sim.field.init_pose_preset", default="Preset",
            parent=preset_row,
        )
        preset_label.setObjectName("missionSimInitPoseLabel")
        preset_layout.addWidget(preset_label, 0)
        self._cb_preset: LaviComboBox = setComboBox(
            [], height=26, i18n=False, parent=preset_row,
        )
        self._cb_preset.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        preset_layout.addWidget(self._cb_preset, 1)
        body.addWidget(preset_row, 0)

        # ── Action row ───────────────────────────────────────────────
        action_row = QWidget(self.body_widget())
        action_layout = QHBoxLayout(action_row)
        action_layout.setContentsMargins(0, 0, 0, 0)
        action_layout.setSpacing(6)
        self._btn_edit = setButton(
            "mission.sim.btn.edit_pose", 92, 26,
            kind="normal", spec="none", default="Edit…",
            parent=action_row,
        )
        self._btn_save_as = setButton(
            "mission.sim.btn.save_pose", 96, 26,
            kind="normal", spec="save", default="Save As…",
            parent=action_row,
        )
        self._btn_delete = setButton(
            "mission.sim.btn.delete_pose", 76, 26,
            kind="normal", spec="delete", default="Delete",
            parent=action_row,
        )
        action_layout.addWidget(self._btn_edit, 0)
        action_layout.addWidget(self._btn_save_as, 0)
        action_layout.addWidget(self._btn_delete, 0)
        action_layout.addStretch(1)
        body.addWidget(action_row, 0)

        # ── Empty-state hint (shown when no SKU/IR roles available) ──
        self._empty_label = I18nLabel(
            "mission.sim.init_pose.empty",
            default="Select a policy with a known robot SKU to edit init pose.",
            parent=self.body_widget(),
        )
        self._empty_label.setObjectName("missionSimInitPoseEmpty")
        self._empty_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        body.addWidget(self._empty_label, 0)

        # ── Wiring ───────────────────────────────────────────────────
        self._cb_enable.toggled.connect(self._on_enable_toggled)
        self._cb_preset.currentIndexChanged.connect(self._on_preset_changed)
        self._btn_edit.clicked.connect(self._on_edit)
        self._btn_save_as.clicked.connect(self._on_save_as)
        self._btn_delete.clicked.connect(self._on_delete)
        self._service.presets_changed.connect(self._on_presets_changed)

        # Initial state: override off, controls greyed.
        self._cb_enable.setChecked(False)
        self._update_enabled_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_sku(self, sku: str) -> None:
        """Rebuild the IR-role list + preset dropdown for ``sku``."""
        new_sku = str(sku or "")
        if new_sku == self._current_sku:
            return
        self._current_sku = new_sku
        self._current_ir_roles = self._resolve_ir_roles_for_sku(new_sku)
        # Drop joint values that no longer exist on this SKU; keep the
        # base_pos so a SKU swap doesn't reset the user's z-height.
        if self._current_ir_roles:
            self._joint_pos_by_ir = {
                k: v for k, v in self._joint_pos_by_ir.items()
                if k in self._current_ir_roles
            }
        else:
            self._joint_pos_by_ir = {}
        self._refresh_preset_list()
        self._update_enabled_state()
        self._update_empty_label_visibility()

    def current_override(self) -> Optional[InitPoseOverride]:
        """Snapshot the current edited values as an ``InitPoseOverride``.

        Returns ``None`` when the enable checkbox is off — callers
        treat that as "use bundle defaults".
        """
        if not self._cb_enable.isChecked():
            return None
        if not self._current_ir_roles:
            log_warning(
                "[mission.sim] init pose override enabled but no IR roles "
                "available for SKU %r — skipping override" % self._current_sku
            )
            return None
        return InitPoseOverride(
            base_pos=tuple(float(v) for v in self._base_pos),  # type: ignore[arg-type]
            joint_pos_by_ir=dict(self._joint_pos_by_ir),
        )

    # ------------------------------------------------------------------
    # Theme
    # ------------------------------------------------------------------
    def apply_theme(self) -> None:
        super().apply_theme()
        fg_main = Config.get_color("main_t1")
        fg_sub = Config.get_color("sub_t2")
        border = Config.get_color("border_1")
        checked = Config.get_color("safe_zone")
        hover_bg = Config.get_color("hover_2")
        font_small = Config.get_font_size("size_small")

        self._cb_enable.setStyleSheet(
            f"QCheckBox#missionSimInitPoseEnable {{ "
            f"color: {fg_main}; background-color: transparent; "
            f"font-size: {font_small}px; spacing: 6px; "
            f"padding: 2px 4px; border-radius: 3px; }}"
            f"QCheckBox#missionSimInitPoseEnable:hover {{ "
            f"background-color: {hover_bg}; }}"
            f"QCheckBox#missionSimInitPoseEnable::indicator {{ "
            f"width: 14px; height: 14px; "
            f"border: 1px solid {border}; border-radius: 3px; "
            f"background-color: transparent; }}"
            f"QCheckBox#missionSimInitPoseEnable::indicator:checked {{ "
            f"background-color: {checked}; border: 1px solid {checked}; "
            f"image: none; }}"
        )
        label_qss = (
            f"QLabel#missionSimInitPoseLabel {{ color: {fg_sub}; "
            f"font-size: {font_small}px; background: transparent; }}"
        )
        for w in self.body_widget().findChildren(QLabel, "missionSimInitPoseLabel"):
            w.setStyleSheet(label_qss)
        empty_qss = (
            f"QLabel#missionSimInitPoseEmpty {{ color: {fg_sub}; "
            f"font-size: {font_small}px; background: transparent; "
            f"font-style: italic; }}"
        )
        self._empty_label.setStyleSheet(empty_qss)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _resolve_ir_roles_for_sku(self, sku: str) -> List[str]:
        """Return IR roles for ``sku`` in canonical registry order."""
        if not sku:
            return []
        entry = _robots_registry.get_robot(sku)
        if not entry:
            return []
        joints_dict = entry.get("joints", {}) or {}
        roles: List[str] = []
        for jspec in joints_dict.values():
            if not isinstance(jspec, dict):
                continue
            role = str(jspec.get("ir_role", "")).strip()
            if role and role not in roles:
                roles.append(role)
        return roles

    def _resolve_sku_display_name(self) -> str:
        if not self._current_sku:
            return ""
        entry = _robots_registry.get_robot(self._current_sku)
        if not entry:
            return ""
        return str(entry.get("name") or entry.get("model") or "")

    def _update_empty_label_visibility(self) -> None:
        self._empty_label.setVisible(not bool(self._current_ir_roles))

    # ------------------------------------------------------------------
    # Preset dropdown
    # ------------------------------------------------------------------
    def _refresh_preset_list(self) -> None:
        """Repopulate the preset dropdown for the current SKU."""
        self._cb_preset.blockSignals(True)
        self._cb_preset.clear()
        # Sentinel "Custom" item — data=None.
        self._cb_preset.addItem(
            tr(_CUSTOM_LABEL_KEY, _CUSTOM_LABEL_DEFAULT), None,
        )
        for preset in self._service.list_presets(self._current_sku):
            badge = "[F]" if preset.source == SOURCE_FACTORY else "[U]"
            label = f"{badge} {preset.name}"
            self._cb_preset.addItem(label, preset)
        self._cb_preset.setCurrentIndex(0)
        self._cb_preset.blockSignals(False)
        self._update_delete_enabled()

    def _on_preset_changed(self, _idx: int) -> None:
        data = self._cb_preset.currentData()
        if isinstance(data, InitPosePreset):
            self._apply_preset_to_state(data)
            self.override_changed.emit()
        self._update_delete_enabled()

    def _apply_preset_to_state(self, preset: InitPosePreset) -> None:
        """Copy preset values into the subsection's internal state."""
        self._base_pos = (
            float(preset.base_pos[0]),
            float(preset.base_pos[1]),
            float(preset.base_pos[2]),
        )
        # Replace wholesale — preset's joint_pos_by_ir is the new truth.
        self._joint_pos_by_ir = {
            str(k): float(v) for k, v in preset.joint_pos_by_ir.items()
        }

    def _mark_as_custom(self) -> None:
        """Switch the dropdown to the sentinel without re-firing changed."""
        self._cb_preset.blockSignals(True)
        self._cb_preset.setCurrentIndex(0)
        self._cb_preset.blockSignals(False)
        self._update_delete_enabled()

    # ------------------------------------------------------------------
    # Edit dialog
    # ------------------------------------------------------------------
    def _on_edit(self) -> None:
        if not self._current_ir_roles:
            log_warning(
                "[mission.sim] edit init pose aborted — no SKU/IR roles resolved"
            )
            return
        dlg = InitPoseEditorDialog(
            sku=self._current_sku,
            sku_display_name=self._resolve_sku_display_name(),
            ir_roles=self._current_ir_roles,
            base_pos=self._base_pos,
            joint_pos_by_ir=self._joint_pos_by_ir,
            parent=self.window(),
        )
        if dlg.exec() != dlg.DialogCode.Accepted:
            return
        new_base = (
            float(dlg.result_base_pos[0]),
            float(dlg.result_base_pos[1]),
            float(dlg.result_base_pos[2]),
        )
        new_joints = {
            str(k): float(v) for k, v in dlg.result_joint_pos_by_ir.items()
        }
        if new_base == self._base_pos and new_joints == self._joint_pos_by_ir:
            return  # nothing changed — keep preset selection intact
        self._base_pos = new_base
        self._joint_pos_by_ir = new_joints
        self._mark_as_custom()
        self.override_changed.emit()

    # ------------------------------------------------------------------
    # Save / Delete
    # ------------------------------------------------------------------
    def _on_save_as(self) -> None:
        if not self._current_sku:
            log_warning("[mission.sim] save preset aborted — no SKU resolved")
            return
        name, ok = QInputDialog.getText(
            self.window(),
            tr("mission.sim.dialog.save_pose.title", "Save init pose preset"),
            tr("mission.sim.dialog.save_pose.prompt", "Preset name:"),
        )
        if not ok:
            return
        name = str(name or "").strip()
        if not name:
            return
        preset = InitPosePreset(
            name=name,
            description=f"user-saved {self._current_sku}",
            base_pos=tuple(self._base_pos),  # type: ignore[arg-type]
            joint_pos_by_ir=dict(self._joint_pos_by_ir),
            source=SOURCE_USER,
        )
        if self._service.save_user_preset(self._current_sku, preset):
            log_info(
                f"[mission.sim] init pose preset saved: {name!r} "
                f"(sku={self._current_sku})"
            )
            self._select_preset_by_name(name)

    def _on_delete(self) -> None:
        data = self._cb_preset.currentData()
        if not isinstance(data, InitPosePreset):
            return
        if data.source != SOURCE_USER:
            log_warning(
                "[mission.sim] cannot delete factory preset %r" % data.name
            )
            return
        if self._service.delete_user_preset(self._current_sku, data.name):
            log_info(
                f"[mission.sim] init pose preset deleted: {data.name!r}"
            )

    def _on_presets_changed(self, sku: str) -> None:
        if sku != self._current_sku:
            return
        self._refresh_preset_list()

    def _select_preset_by_name(self, name: str) -> None:
        for i in range(self._cb_preset.count()):
            data = self._cb_preset.itemData(i)
            if isinstance(data, InitPosePreset) and data.name == name:
                self._cb_preset.setCurrentIndex(i)
                return

    def _update_delete_enabled(self) -> None:
        data = self._cb_preset.currentData()
        ready = (
            isinstance(data, InitPosePreset)
            and data.source == SOURCE_USER
            and self._cb_enable.isChecked()
        )
        self._btn_delete.setEnabled(ready)

    # ------------------------------------------------------------------
    # Enable-checkbox state plumbing
    # ------------------------------------------------------------------
    def _on_enable_toggled(self, _checked: bool) -> None:
        self._update_enabled_state()
        self.override_changed.emit()

    def _update_enabled_state(self) -> None:
        """Disable interactive controls when override is off OR no IR roles loaded."""
        enabled = self._cb_enable.isChecked() and bool(self._current_ir_roles)
        for w in (self._cb_preset, self._btn_edit, self._btn_save_as):
            w.setEnabled(enabled)
        if enabled:
            self._update_delete_enabled()
        else:
            self._btn_delete.setEnabled(False)


__all__ = ["InitPoseSubsection"]
