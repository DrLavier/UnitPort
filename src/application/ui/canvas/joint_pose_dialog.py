# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""JointPoseEditorDialog — slider-driven init pose editor for ActorSettingNode.

Opened from ``JointPoseTableRow._handle_click`` (canvas) when the upstream
RobotNode's SKU + MJCF can be resolved. Each IR role gets a horizontal slider
whose physical range equals ``mj_model.jnt_range`` for the corresponding MJCF
joint (``jnt_limited=False`` falls back to ±π and shows an "unlimited" hint).
A ``QDoubleSpinBox`` is bound bidirectionally to the slider for precise typing.

When ``min < 0 < max`` we paint a single vertical 0 tick over the groove so the
user has a neutral-pose visual cue; ranges that don't contain 0 (e.g. Go2's
calf ``[-2.72, -0.84]``) intentionally skip the tick.

Bottom button row:
    [ ▶ Review ]                            [ Cancel ]  [ OK ]

Review (when ``on_review`` provided) snapshots the **current in-flight** slider
values and calls back; it never closes the dialog. OK writes the snapshot to
``result_joint_pos_by_ir`` (mirroring ``InitPoseEditorDialog``'s contract).

All colors / fonts go through ``Config.get_color`` / ``Config.get_font_size``
(CLAUDE.md §1.5); UI strings go through ``tr`` / ``I18nLabel``.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Mapping, Optional, Tuple

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QStyle,
    QStyleOptionSlider,
    QVBoxLayout,
    QWidget,
)

from unitport_sdk import Config, I18nLabel, tr


# Slider precision — 1000 units per radian gives 3 decimal places, matches the
# QDoubleSpinBox decimals (so cross-sync is lossless within the displayed
# precision).
_SLIDER_SCALE = 1000

# Fallback range for joints flagged ``jnt_limited == False`` in MJCF — wide
# enough for any plausible debug pose; MuJoCo still won't clamp because the
# joint is unlimited.
_UNLIMITED_LO = -math.pi
_UNLIMITED_HI = math.pi


# ----------------------------------------------------------------------
# Custom slider with optional 0-tick overlay
# ----------------------------------------------------------------------


class _JointSlider(QSlider):
    """Horizontal slider that paints a single vertical 0 tick over the groove
    when ``lo < 0 < hi`` (the chosen UX from the plan — user picked option A:
    range strictly matches MJCF, 0 is a visual tick only when in range).
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(Qt.Orientation.Horizontal, parent)
        self._lo_f: float = 0.0
        self._hi_f: float = 1.0
        self._zero_visible: bool = False

    def configure(self, lo: float, hi: float, value: float) -> None:
        self._lo_f = float(lo)
        self._hi_f = float(hi)
        # QSlider is int-only — scale by _SLIDER_SCALE.
        self.setMinimum(int(round(self._lo_f * _SLIDER_SCALE)))
        self.setMaximum(int(round(self._hi_f * _SLIDER_SCALE)))
        self.setSingleStep(max(1, int(round(0.01 * _SLIDER_SCALE))))
        self.setPageStep(max(1, int(round(0.1 * _SLIDER_SCALE))))
        self._zero_visible = (self._lo_f < 0.0 < self._hi_f)
        # Initial value (clamped to range).
        clamped = max(self._lo_f, min(self._hi_f, float(value)))
        self.setValue(int(round(clamped * _SLIDER_SCALE)))

    def float_value(self) -> float:
        return self.value() / _SLIDER_SCALE

    def set_float_value(self, v: float) -> None:
        clamped = max(self._lo_f, min(self._hi_f, float(v)))
        self.setValue(int(round(clamped * _SLIDER_SCALE)))

    def paintEvent(self, event) -> None:  # noqa: D401 - Qt override
        super().paintEvent(event)
        if not self._zero_visible:
            return
        span = self._hi_f - self._lo_f
        if span <= 0.0:
            return
        opt = QStyleOptionSlider()
        self.initStyleOption(opt)
        groove = self.style().subControlRect(
            QStyle.ComplexControl.CC_Slider,
            opt,
            QStyle.SubControl.SC_SliderGroove,
            self,
        )
        ratio = (0.0 - self._lo_f) / span
        x = groove.left() + int(round(ratio * groove.width()))
        painter = QPainter(self)
        try:
            tick_color = QColor(
                Config.get_color("highlight", "#F6D393")
            )
            pen = QPen(tick_color)
            pen.setWidth(2)
            painter.setPen(pen)
            painter.drawLine(
                x, groove.top() - 2, x, groove.bottom() + 2
            )
        finally:
            painter.end()


# ----------------------------------------------------------------------
# Dialog
# ----------------------------------------------------------------------


class JointPoseEditorDialog(QDialog):
    """Modal slider editor for ``{ir_role: radians}`` init pose dicts.

    Construct with the resolved SKU's IR roles (in qpos order) + per-role
    MJCF ranges. Pull edited values from :attr:`result_joint_pos_by_ir`
    after :meth:`exec` returns ``Accepted``.
    """

    def __init__(
        self,
        sku: str,
        sku_display_name: str,
        ir_roles_in_order: List[str],
        joint_ranges: Mapping[str, Optional[Tuple[float, float]]],
        initial_joints: Mapping[str, float],
        on_review: Optional[Callable[[Dict[str, float]], None]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._sku = str(sku)
        self._ir_roles: List[str] = [str(r) for r in ir_roles_in_order]
        self._ranges: Dict[str, Optional[Tuple[float, float]]] = {
            str(k): (None if v is None else (float(v[0]), float(v[1])))
            for k, v in (joint_ranges or {}).items()
        }
        self._initial: Dict[str, float] = {
            str(k): float(v) for k, v in (initial_joints or {}).items()
        }
        self._on_review = on_review

        # Per-row widgets, indexed by ir_role.
        self._sliders: Dict[str, _JointSlider] = {}
        self._spinboxes: Dict[str, QDoubleSpinBox] = {}

        # Result snapshot — written on accept().
        self.result_joint_pos_by_ir: Dict[str, float] = dict(self._initial)

        title = tr(
            "canvas.actor_setting.init_pose.dialog_title",
            "Init Pose Editor",
        )
        if sku_display_name:
            title = f"{title} — {sku_display_name}"
        self.setWindowTitle(title)
        self.setModal(True)
        self.setMinimumWidth(560)
        self.setMinimumHeight(380)

        self._build_layout()
        self.apply_theme()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def current_values(self) -> Dict[str, float]:
        """Snapshot current slider/spinbox values into a plain dict."""
        out: Dict[str, float] = {}
        for role, sb in self._spinboxes.items():
            out[role] = float(sb.value())
        # Preserve IR roles we didn't render (e.g. incoming dict had keys
        # outside the current SKU's joint set) — same forgiveness as
        # InitPoseEditorDialog.
        for role, val in self._initial.items():
            if role not in out:
                out[role] = float(val)
        return out

    def accept(self) -> None:  # noqa: D401 - QDialog override
        self.result_joint_pos_by_ir = self.current_values()
        super().accept()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_layout(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 16, 16, 12)
        outer.setSpacing(10)

        header = I18nLabel(
            "canvas.actor_setting.init_pose.joints_label",
            default="Joint angles [rad] — range from MJCF",
            parent=self,
        )
        header.setObjectName("jointPoseDialogLabel")
        outer.addWidget(header, 0)

        scroll = QScrollArea(self)
        scroll.setObjectName("jointPoseDialogScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        body = QWidget(scroll)
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(6)
        self._build_joint_rows(body, body_layout)
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)

        # Bottom button bar — Review on the left, OK/Cancel on the right.
        bar = QWidget(self)
        bar_layout = QHBoxLayout(bar)
        bar_layout.setContentsMargins(0, 0, 0, 0)
        bar_layout.setSpacing(8)

        if self._on_review is not None:
            self._review_btn = QPushButton(
                "▶ " + tr(
                    "canvas.actor_setting.init_pose.review_button",
                    "Review",
                ),
                bar,
            )
            self._review_btn.setObjectName("jointPoseDialogReviewBtn")
            self._review_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            self._review_btn.clicked.connect(self._fire_review)
            bar_layout.addWidget(self._review_btn, 0)

        bar_layout.addStretch(1)

        ok_cancel = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
            parent=bar,
        )
        ok_cancel.accepted.connect(self.accept)
        ok_cancel.rejected.connect(self.reject)
        bar_layout.addWidget(ok_cancel, 0)

        outer.addWidget(bar, 0)

    def _build_joint_rows(self, host: QWidget, layout: QVBoxLayout) -> None:
        if not self._ir_roles:
            empty = I18nLabel(
                "canvas.actor_setting.init_pose.empty",
                default="No joints available for this SKU.",
                parent=host,
            )
            empty.setObjectName("jointPoseDialogEmpty")
            layout.addWidget(empty, 0)
            return

        for role in self._ir_roles:
            rng = self._ranges.get(role)
            unlimited = rng is None
            if unlimited:
                lo, hi = _UNLIMITED_LO, _UNLIMITED_HI
            else:
                lo, hi = rng  # type: ignore[misc]

            initial_val = float(self._initial.get(role, 0.0))
            # If 0 is in range, default unset entries to 0 (visually neutral);
            # otherwise default to the midpoint of the range.
            if role not in self._initial:
                if lo <= 0.0 <= hi:
                    initial_val = 0.0
                else:
                    initial_val = (lo + hi) / 2.0
            # Clamp regardless — incoming dict may have stale values outside
            # the current robot's range.
            initial_val = max(lo, min(hi, initial_val))

            row = QWidget(host)
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(8)

            label = QLabel(role, row)
            label.setObjectName("jointPoseDialogJointLabel")
            label.setMinimumWidth(96)
            label.setAlignment(
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft
            )
            row_layout.addWidget(label, 0)

            slider = _JointSlider(row)
            slider.configure(lo, hi, initial_val)
            slider.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
            )
            slider.setMinimumWidth(220)
            row_layout.addWidget(slider, 1)

            sb = QDoubleSpinBox(row)
            sb.setDecimals(3)
            sb.setRange(float(lo), float(hi))
            sb.setSingleStep(0.05)
            sb.setValue(float(initial_val))
            sb.setKeyboardTracking(False)
            sb.setFixedWidth(96)
            sb.setSizePolicy(
                QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed
            )
            row_layout.addWidget(sb, 0)

            if unlimited:
                hint = I18nLabel(
                    "canvas.actor_setting.init_pose.unlimited_hint",
                    default="(unlimited)",
                    parent=row,
                )
                hint.setObjectName("jointPoseDialogHint")
                row_layout.addWidget(hint, 0)

            # Bidirectional sync between slider and spinbox. Block signals on
            # the receiving widget while pushing to avoid feedback loops.
            slider.valueChanged.connect(
                lambda _i, s=slider, b=sb: self._on_slider_changed(s, b)
            )
            sb.valueChanged.connect(
                lambda _v, s=slider, b=sb: self._on_spinbox_changed(s, b)
            )

            self._sliders[role] = slider
            self._spinboxes[role] = sb
            layout.addWidget(row, 0)

    # ------------------------------------------------------------------
    # Slider <-> spinbox cross-sync
    # ------------------------------------------------------------------

    @staticmethod
    def _on_slider_changed(slider: _JointSlider, sb: QDoubleSpinBox) -> None:
        new_val = slider.float_value()
        if abs(new_val - sb.value()) < 1.0 / _SLIDER_SCALE:
            return
        sb.blockSignals(True)
        try:
            sb.setValue(new_val)
        finally:
            sb.blockSignals(False)

    @staticmethod
    def _on_spinbox_changed(slider: _JointSlider, sb: QDoubleSpinBox) -> None:
        new_val = float(sb.value())
        if abs(new_val - slider.float_value()) < 1.0 / _SLIDER_SCALE:
            return
        slider.blockSignals(True)
        try:
            slider.set_float_value(new_val)
        finally:
            slider.blockSignals(False)

    # ------------------------------------------------------------------
    # Review wiring
    # ------------------------------------------------------------------

    def _fire_review(self) -> None:
        if self._on_review is None:
            return
        try:
            self._on_review(self.current_values())
        except Exception:
            # Reviews must never break the dialog. Caller is expected to
            # log internally; we swallow to keep the editor responsive.
            from unitport_sdk import log_warning
            log_warning(
                "[joint_pose_dialog] review callback raised — swallowed"
            )

    # ------------------------------------------------------------------
    # Theme
    # ------------------------------------------------------------------

    def apply_theme(self) -> None:
        bg = Config.get_color("bg_2")
        fg_main = Config.get_color("main_t1")
        fg_sub = Config.get_color("sub_t2")
        border = Config.get_color("border_1")
        accent = Config.get_color("safe_zone")
        track = Config.get_color("opacity_slider_track")
        handle = Config.get_color("highlight")
        review_bg = Config.get_color("canvas_node_review_launch_bg")
        review_hover = Config.get_color("canvas_node_review_launch_hover_bg")
        review_text = Config.get_color("main_t2")
        font_small = Config.get_font_size("size_small")
        font_normal = Config.get_font_size("size_normal")

        self.setStyleSheet(f"QDialog {{ background-color: {bg}; }}")

        for w in self.findChildren(QLabel, "jointPoseDialogLabel"):
            w.setStyleSheet(
                f"QLabel#jointPoseDialogLabel {{ color: {fg_main}; "
                f"font-size: {font_normal}px; font-weight: 600; "
                f"background: transparent; }}"
            )
        for w in self.findChildren(QLabel, "jointPoseDialogJointLabel"):
            w.setStyleSheet(
                f"QLabel#jointPoseDialogJointLabel {{ color: {fg_main}; "
                f"font-size: {font_small}px; background: transparent; }}"
            )
        for w in self.findChildren(QLabel, "jointPoseDialogHint"):
            w.setStyleSheet(
                f"QLabel#jointPoseDialogHint {{ color: {fg_sub}; "
                f"font-size: {font_small}px; font-style: italic; "
                f"background: transparent; }}"
            )
        for w in self.findChildren(QLabel, "jointPoseDialogEmpty"):
            w.setStyleSheet(
                f"QLabel#jointPoseDialogEmpty {{ color: {fg_sub}; "
                f"font-size: {font_small}px; font-style: italic; "
                f"background: transparent; }}"
            )

        spin_qss = (
            f"QDoubleSpinBox {{ color: {fg_main}; "
            f"background-color: transparent; border: 1px solid {border}; "
            f"border-radius: 3px; padding: 2px 4px; "
            f"font-size: {font_small}px; }}"
            f"QDoubleSpinBox:focus {{ border: 1px solid {accent}; }}"
        )
        for sb in self.findChildren(QDoubleSpinBox):
            sb.setStyleSheet(spin_qss)

        slider_qss = (
            f"QSlider::groove:horizontal {{ height: 6px; "
            f"background: {track}; border-radius: 3px; }}"
            f"QSlider::handle:horizontal {{ background: {handle}; "
            f"width: 14px; margin: -5px 0; border-radius: 7px; }}"
            f"QSlider::sub-page:horizontal {{ background: {accent}; "
            f"border-radius: 3px; }}"
        )
        for sl in self.findChildren(QSlider):
            sl.setStyleSheet(slider_qss)

        review_qss = (
            f"QPushButton#jointPoseDialogReviewBtn {{ "
            f"background-color: {review_bg}; color: {review_text}; "
            f"border: none; border-radius: 4px; "
            f"padding: 6px 16px; font-weight: 600; "
            f"font-size: {font_small}px; }}"
            f"QPushButton#jointPoseDialogReviewBtn:hover {{ "
            f"background-color: {review_hover}; }}"
        )
        for b in self.findChildren(QPushButton, "jointPoseDialogReviewBtn"):
            b.setStyleSheet(review_qss)


__all__ = ["JointPoseEditorDialog"]
