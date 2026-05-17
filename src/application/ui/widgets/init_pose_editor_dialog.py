"""InitPoseEditorDialog — modal editor for base position + joint angles.

Opened from the Simulation card's Init Pose subsection. Renders the
full per-IR-role joint form so the card itself stays compact — only
the enable toggle, preset dropdown, and action buttons live there.

Layout:
* Base position row (x / y / z spinboxes, captioned)
* Joint angle rows (per-leg for quadrupeds, flat for everything else)
* OK / Cancel button row

All colors / fonts go through ``Config.get_color`` / ``Config.get_font_size``
(CLAUDE.md §1.5); all UI strings go through ``tr`` / ``i18n_bind`` /
``I18nLabel``.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Tuple

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from unitport_sdk import (
    Config,
    I18nLabel,
    tr,
)


# Joint-angle spinbox range — wider than physical joint limits so users
# can experiment with poses MuJoCo will then clamp via joint limits at
# spawn time.
_JOINT_MIN_RAD = -3.5
_JOINT_MAX_RAD = 3.5
_JOINT_STEP = 0.05

# Base position spinbox range — covers fall-from-height up to 2 m and
# slightly-below-ground for niche debug cases.
_BASE_X_MIN, _BASE_X_MAX = -10.0, 10.0
_BASE_Y_MIN, _BASE_Y_MAX = -10.0, 10.0
_BASE_Z_MIN, _BASE_Z_MAX = -0.5, 5.0
_BASE_STEP = 0.01

# Quadruped leg suffixes — when every IR role matches "<part>_<leg>"
# with leg in this set, the joint editor groups rows by leg. Otherwise
# falls back to a flat one-row-per-joint list.
_QUADRUPED_LEGS = ("FL", "FR", "RL", "RR")


def _ir_role_leg(role: str) -> Optional[str]:
    if not isinstance(role, str) or "_" not in role:
        return None
    suffix = role.rsplit("_", 1)[-1]
    return suffix if suffix in _QUADRUPED_LEGS else None


def _ir_role_part(role: str) -> str:
    if not isinstance(role, str) or "_" not in role:
        return role
    return role.rsplit("_", 1)[0]


class InitPoseEditorDialog(QDialog):
    """Modal init pose editor.

    Construct with the active SKU's IR-role list plus the current
    values; the caller pulls the edited values via
    :attr:`result_base_pos` and :attr:`result_joint_pos_by_ir` after
    :meth:`exec` returns ``Accepted``.
    """

    def __init__(
        self,
        sku: str,
        sku_display_name: str,
        ir_roles: List[str],
        base_pos: Tuple[float, float, float],
        joint_pos_by_ir: Mapping[str, float],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._sku = str(sku)
        self._ir_roles = list(ir_roles)
        # Initial values used by reset_to_initial(); cached so a user
        # who taps Cancel mid-edit just discards changes.
        self._initial_base_pos = tuple(float(v) for v in base_pos)
        self._initial_joints: Dict[str, float] = {
            str(k): float(v) for k, v in joint_pos_by_ir.items()
        }
        # Editor widgets — populated by _build_layout.
        self._sb_base_x: Optional[QDoubleSpinBox] = None
        self._sb_base_y: Optional[QDoubleSpinBox] = None
        self._sb_base_z: Optional[QDoubleSpinBox] = None
        self._joint_inputs: Dict[str, QDoubleSpinBox] = {}
        # Results — written on accept().
        self.result_base_pos: Tuple[float, float, float] = self._initial_base_pos
        self.result_joint_pos_by_ir: Dict[str, float] = dict(self._initial_joints)

        title = tr(
            "mission.sim.init_pose.editor.title",
            "Init Pose Editor",
        )
        if sku_display_name:
            title = f"{title} — {sku_display_name}"
        self.setWindowTitle(title)
        self.setModal(True)
        self.setMinimumWidth(520)

        self._build_layout()
        self.apply_theme()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def accept(self) -> None:  # noqa: D401 - overrides QDialog.accept
        """Snapshot edited values into ``result_*`` then close as Accepted."""
        if self._sb_base_x is not None:
            self.result_base_pos = (
                float(self._sb_base_x.value()),
                float(self._sb_base_y.value()),
                float(self._sb_base_z.value()),
            )
        edited: Dict[str, float] = {}
        for role, sb in self._joint_inputs.items():
            edited[role] = float(sb.value())
        # Preserve any IR roles the dialog didn't render (e.g. when the
        # incoming dict had entries outside the current SKU's IR set).
        for role, val in self._initial_joints.items():
            if role not in edited:
                edited[role] = float(val)
        self.result_joint_pos_by_ir = edited
        super().accept()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def _build_layout(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 16, 16, 12)
        outer.setSpacing(10)

        # --- Base position section ---------------------------------------
        base_label = I18nLabel(
            "mission.sim.field.init_pose_base",
            default="Base position [m]",
            parent=self,
        )
        base_label.setObjectName("missionSimInitPoseDialogLabel")
        outer.addWidget(base_label, 0)

        base_row = QWidget(self)
        base_row_layout = QHBoxLayout(base_row)
        base_row_layout.setContentsMargins(0, 0, 0, 0)
        base_row_layout.setSpacing(8)
        self._sb_base_x = self._make_spinbox(
            _BASE_X_MIN, _BASE_X_MAX, _BASE_STEP,
            value=self._initial_base_pos[0],
        )
        self._sb_base_y = self._make_spinbox(
            _BASE_Y_MIN, _BASE_Y_MAX, _BASE_STEP,
            value=self._initial_base_pos[1],
        )
        self._sb_base_z = self._make_spinbox(
            _BASE_Z_MIN, _BASE_Z_MAX, _BASE_STEP,
            value=self._initial_base_pos[2],
        )
        for axis, sb in (("x", self._sb_base_x),
                         ("y", self._sb_base_y),
                         ("z", self._sb_base_z)):
            cell = QWidget(base_row)
            cell_layout = QVBoxLayout(cell)
            cell_layout.setContentsMargins(0, 0, 0, 0)
            cell_layout.setSpacing(2)
            cap = QLabel(axis, cell)
            cap.setObjectName("missionSimInitPoseDialogAxis")
            cap.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            cell_layout.addWidget(cap, 0)
            cell_layout.addWidget(sb, 0)
            base_row_layout.addWidget(cell, 1)
        outer.addWidget(base_row, 0)

        # --- Joint angles section ----------------------------------------
        joints_label = I18nLabel(
            "mission.sim.field.init_pose_joints",
            default="Joint angles [rad]",
            parent=self,
        )
        joints_label.setObjectName("missionSimInitPoseDialogLabel")
        outer.addWidget(joints_label, 0)

        # Scroll area keeps the dialog usable when a robot has many
        # joints (humanoids ~30+); quadruped 12 joints fits inline but
        # the cost of always-scrollable is one borderless frame.
        scroll = QScrollArea(self)
        scroll.setObjectName("missionSimInitPoseDialogScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        joints_host = QWidget(scroll)
        joints_layout = QVBoxLayout(joints_host)
        joints_layout.setContentsMargins(0, 0, 0, 0)
        joints_layout.setSpacing(4)
        self._build_joint_form(joints_host, joints_layout)
        scroll.setWidget(joints_host)
        outer.addWidget(scroll, 1)

        # --- Buttons -----------------------------------------------------
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons, 0)

    def _build_joint_form(
        self,
        host: QWidget,
        layout: QVBoxLayout,
    ) -> None:
        if not self._ir_roles:
            empty = I18nLabel(
                "mission.sim.init_pose.empty",
                default="No joints available for this SKU.",
                parent=host,
            )
            empty.setObjectName("missionSimInitPoseDialogEmpty")
            layout.addWidget(empty, 0)
            return

        legs: Dict[str, List[str]] = {}
        for role in self._ir_roles:
            leg = _ir_role_leg(role)
            if leg is not None:
                legs.setdefault(leg, []).append(role)
        is_quadruped = (
            sorted(legs.keys()) == sorted(_QUADRUPED_LEGS)
            and sum(len(v) for v in legs.values()) == len(self._ir_roles)
        )
        if is_quadruped:
            self._build_quadruped_form(host, layout, legs)
        else:
            self._build_flat_form(host, layout, self._ir_roles)

    def _build_quadruped_form(
        self,
        host: QWidget,
        layout: QVBoxLayout,
        legs: Dict[str, List[str]],
    ) -> None:
        for leg in _QUADRUPED_LEGS:
            roles = legs.get(leg, [])
            if not roles:
                continue
            roles_sorted = sorted(roles, key=_ir_role_part)
            row = QWidget(host)
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(8)
            leg_label = QLabel(leg, row)
            leg_label.setObjectName("missionSimInitPoseDialogLeg")
            leg_label.setFixedWidth(32)
            row_layout.addWidget(leg_label, 0, Qt.AlignmentFlag.AlignVCenter)
            for role in roles_sorted:
                cell = QWidget(row)
                cell_layout = QVBoxLayout(cell)
                cell_layout.setContentsMargins(0, 0, 0, 0)
                cell_layout.setSpacing(2)
                cap = QLabel(_ir_role_part(role), cell)
                cap.setObjectName("missionSimInitPoseDialogJointCap")
                cap.setAlignment(Qt.AlignmentFlag.AlignHCenter)
                sb = self._make_spinbox(
                    _JOINT_MIN_RAD, _JOINT_MAX_RAD, _JOINT_STEP,
                    value=float(self._initial_joints.get(role, 0.0)),
                )
                cell_layout.addWidget(cap, 0)
                cell_layout.addWidget(sb, 0)
                row_layout.addWidget(cell, 1)
                self._joint_inputs[role] = sb
            layout.addWidget(row, 0)

    def _build_flat_form(
        self,
        host: QWidget,
        layout: QVBoxLayout,
        roles: List[str],
    ) -> None:
        for role in roles:
            row = QWidget(host)
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(8)
            cap = QLabel(role, row)
            cap.setObjectName("missionSimInitPoseDialogJointCap")
            cap.setMinimumWidth(120)
            row_layout.addWidget(cap, 0, Qt.AlignmentFlag.AlignVCenter)
            sb = self._make_spinbox(
                _JOINT_MIN_RAD, _JOINT_MAX_RAD, _JOINT_STEP,
                value=float(self._initial_joints.get(role, 0.0)),
            )
            row_layout.addWidget(sb, 1)
            layout.addWidget(row, 0)
            self._joint_inputs[role] = sb

    def _make_spinbox(
        self, lo: float, hi: float, step: float, value: float = 0.0,
    ) -> QDoubleSpinBox:
        sb = QDoubleSpinBox(self)
        sb.setDecimals(3)
        sb.setRange(lo, hi)
        sb.setSingleStep(step)
        sb.setValue(value)
        sb.setKeyboardTracking(False)
        sb.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.UpDownArrows)
        sb.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        return sb

    # ------------------------------------------------------------------
    # Theme
    # ------------------------------------------------------------------
    def apply_theme(self) -> None:
        bg = Config.get_color("bg_2")
        fg_main = Config.get_color("main_t1")
        fg_sub = Config.get_color("sub_t2")
        border = Config.get_color("border_1")
        accent = Config.get_color("safe_zone")
        font_small = Config.get_font_size("size_small")
        font_normal = Config.get_font_size("size_normal")

        self.setStyleSheet(
            f"QDialog {{ background-color: {bg}; }}"
        )
        label_qss = (
            f"QLabel#missionSimInitPoseDialogLabel {{ color: {fg_main}; "
            f"font-size: {font_normal}px; font-weight: 600; "
            f"background: transparent; }}"
        )
        for w in self.findChildren(QLabel, "missionSimInitPoseDialogLabel"):
            w.setStyleSheet(label_qss)
        axis_qss = (
            f"QLabel#missionSimInitPoseDialogAxis {{ color: {fg_sub}; "
            f"font-size: {font_small}px; background: transparent; }}"
        )
        for w in self.findChildren(QLabel, "missionSimInitPoseDialogAxis"):
            w.setStyleSheet(axis_qss)
        leg_qss = (
            f"QLabel#missionSimInitPoseDialogLeg {{ color: {fg_main}; "
            f"font-size: {font_normal}px; font-weight: 600; "
            f"background: transparent; }}"
        )
        for w in self.findChildren(QLabel, "missionSimInitPoseDialogLeg"):
            w.setStyleSheet(leg_qss)
        cap_qss = (
            f"QLabel#missionSimInitPoseDialogJointCap {{ color: {fg_sub}; "
            f"font-size: {font_small}px; background: transparent; }}"
        )
        for w in self.findChildren(QLabel, "missionSimInitPoseDialogJointCap"):
            w.setStyleSheet(cap_qss)
        empty_qss = (
            f"QLabel#missionSimInitPoseDialogEmpty {{ color: {fg_sub}; "
            f"font-size: {font_small}px; font-style: italic; "
            f"background: transparent; }}"
        )
        for w in self.findChildren(QLabel, "missionSimInitPoseDialogEmpty"):
            w.setStyleSheet(empty_qss)
        spin_qss = (
            f"QDoubleSpinBox {{ color: {fg_main}; "
            f"background-color: transparent; border: 1px solid {border}; "
            f"border-radius: 3px; padding: 2px 4px; "
            f"font-size: {font_small}px; }}"
            f"QDoubleSpinBox:focus {{ border: 1px solid {accent}; }}"
        )
        for sb in self.findChildren(QDoubleSpinBox):
            sb.setStyleSheet(spin_qss)


__all__ = ["InitPoseEditorDialog"]
