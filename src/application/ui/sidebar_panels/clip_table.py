# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""MotionClipTable — the clip listing embedded in an expanded Motion card.

Built for a STRICT fixed sidebar width (280 px). The list is **2-dimensional**:
each clip is a row that can be *expanded* to reveal the segments cut from it in
the Clip Motion Editor (a segment = an inclusive ``[lo, hi]`` frame sub-range).

    [ ▸  clip_name ·N        ]   ← clip header row (click name = open editor)
      └ walk_loop  [120–360]     ← segment sub-rows (shown when expanded;
      └ turn       [400–520]        click = open editor drilled to that range)

The disclosure triangle (left ~18 px) toggles the sub-rows; clicking anywhere
else on the header opens the whole-clip editor. Every label is a
width-collapsing ``_ElidedLabel`` (0-width minimumSizeHint → can never push the
row past the panel); horizontal scrolling is disabled, only vertical.
"""

from __future__ import annotations

from typing import List, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QCursor
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from unitport_sdk import Config, tr

# Reuse the width-eliding label from the Resources panel (same package).
from application.ui.sidebar_panels.resources_panel import _ElidedLabel

#: Left gutter (px) that holds the disclosure triangle on a clip header row.
_TOGGLE_W = 18


def _mini() -> int:
    return int(Config.get_font_size("size_mini"))


class _SegmentRow(QFrame):
    """One indented, clickable segment sub-row (name + inclusive range)."""

    def __init__(self, clip_ref: str, seg, on_click, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._ref = clip_ref
        self._seg = seg
        self._on_click = on_click
        self.setObjectName("segRow")
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setToolTip(tr("clip_table.segment_tip", "Open this segment"))
        self._apply_style(hover=False)

        rl = QHBoxLayout(self)
        rl.setContentsMargins(_TOGGLE_W + 6, 2, 6, 2)  # indent under the triangle
        rl.setSpacing(4)

        tag = f" · {seg.task_tag}" if getattr(seg, "task_tag", "") else ""
        text = f"↳ {seg.name}{tag}"
        name = _ElidedLabel(text)
        name.setToolTip(f"{seg.name}  [{seg.start_frame}–{seg.end_frame}]")
        name.setStyleSheet(
            f"color: {Config.get_color('sub_t1')};"
            f" background: transparent; font-size: {_mini()}px;"
        )
        name.setMinimumWidth(0)
        name.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        rl.addWidget(name, 1)

        rng = _ElidedLabel(f"[{seg.start_frame}–{seg.end_frame}]")
        rng.setStyleSheet(
            f"color: {Config.get_color('highlight')};"
            f" background: transparent; font-size: {_mini()}px;"
        )
        rng.setMinimumWidth(0)
        rng.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        rl.addWidget(rng, 0)

    def _apply_style(self, *, hover: bool) -> None:
        bg = Config.get_color("bg_3") if hover else "transparent"
        self.setStyleSheet(f"#segRow {{ background: {bg}; border-radius: 3px; }}")

    def enterEvent(self, e):  # type: ignore[override]
        self._apply_style(hover=True)
        super().enterEvent(e)

    def leaveEvent(self, e):  # type: ignore[override]
        self._apply_style(hover=False)
        super().leaveEvent(e)

    def mousePressEvent(self, e):  # type: ignore[override]
        if e.button() == Qt.MouseButton.LeftButton:
            self._on_click(self._ref, int(self._seg.start_frame), int(self._seg.end_frame))
            e.accept()
            return
        super().mousePressEvent(e)


class _ClipHeaderRow(QFrame):
    """A clip's header row: [ ▸/▾ toggle gutter ][ elided name ].

    A position-based hit test routes clicks: the left ``_TOGGLE_W`` gutter (when
    the clip has segments) toggles expand/collapse; anywhere else opens the
    whole-clip editor. Proper QFrame subclass (not instance-monkeypatched event
    handlers) so Qt reliably dispatches the overrides.
    """

    def __init__(self, clip, has_segs, on_toggle, on_open, parent=None) -> None:
        super().__init__(parent)
        self._has_segs = has_segs
        self._on_toggle = on_toggle
        self._on_open = on_open
        self._ref = clip.clip_ref
        self.setObjectName("clipRow")
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setToolTip(tr("clip_table.edit_tip", "Click to crop / label this clip"))
        self._apply_style(hover=False)

        rl = QHBoxLayout(self)
        rl.setContentsMargins(6, 3, 6, 3)
        rl.setSpacing(0)

        self.toggle = _ElidedLabel("▸" if has_segs else "")
        self.toggle.setFixedWidth(_TOGGLE_W)
        self.toggle.setStyleSheet(
            f"color: {Config.get_color('sub_t1')};"
            f" background: transparent; font-size: {_mini()}px;"
        )
        self.toggle.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        rl.addWidget(self.toggle, 0)

        label = f"{clip.name}  ·{clip.segment_count}" if has_segs else clip.name
        name = _ElidedLabel(label)
        name.setToolTip(self._ref)  # hover shows the full address
        name.setStyleSheet(
            f"color: {Config.get_color('main_t1')};"
            f" background: transparent; font-size: {_mini()}px;"
        )
        name.setMinimumWidth(0)
        name.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        rl.addWidget(name, 1)

    def _apply_style(self, *, hover: bool) -> None:
        bg = Config.get_color("row_1") if hover else "transparent"
        self.setStyleSheet(f"#clipRow {{ background: {bg}; border-radius: 3px; }}")

    def enterEvent(self, e):  # type: ignore[override]
        self._apply_style(hover=True)
        super().enterEvent(e)

    def leaveEvent(self, e):  # type: ignore[override]
        self._apply_style(hover=False)
        super().leaveEvent(e)

    def mousePressEvent(self, e):  # type: ignore[override]
        if e.button() == Qt.MouseButton.LeftButton:
            if self._has_segs and e.position().x() <= _TOGGLE_W:
                self._on_toggle()
            else:
                self._on_open(self._ref)
            e.accept()
            return
        super().mousePressEvent(e)


class _ClipItem(QWidget):
    """A clip header row plus a collapsible container of its segment sub-rows."""

    def __init__(
        self,
        clip,
        provider,
        on_open_clip,
        on_open_segment,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._clip = clip
        self._ref = clip.clip_ref
        self._provider = provider
        self._on_open_segment = on_open_segment
        self._has_segs = bool(getattr(clip, "segment_count", 0))
        self._expanded = False
        self._built = False

        col = QVBoxLayout(self)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(0)

        self._header = _ClipHeaderRow(
            clip, self._has_segs, self._toggle_expand, on_open_clip
        )
        col.addWidget(self._header)

        self._segs_box = QWidget(self)
        self._segs_box.setStyleSheet("background: transparent;")
        self._segs_layout = QVBoxLayout(self._segs_box)
        self._segs_layout.setContentsMargins(0, 0, 0, 2)
        self._segs_layout.setSpacing(1)
        self._segs_box.setVisible(False)
        col.addWidget(self._segs_box)

    # ------------------------------------------------------------------

    def _toggle_expand(self) -> None:
        self._expanded = not self._expanded
        self._header.toggle.setText("▾" if self._expanded else "▸")
        if self._expanded and not self._built:
            self._build_segments()
        self._segs_box.setVisible(self._expanded)

    def _build_segments(self) -> None:
        self._built = True
        try:
            segs = self._provider.list_segments(self._ref)
        except Exception:
            segs = []
        for seg in segs:
            self._segs_layout.addWidget(
                _SegmentRow(self._ref, seg, self._on_open_segment)
            )


class MotionClipTable(QWidget):
    """Compact fixed-width 2-D clip list; rows open the editor / expand segments."""

    def __init__(self, package_id: str, *, provider, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._package_id = package_id
        self._provider = provider
        # Live modeless editors, kept referenced so they aren't GC'd while open.
        self._open_editors: list = []

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        # KEY: no horizontal scroll — rows are clipped/elided to the width,
        # never widened past the sidebar.
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setMaximumHeight(260)
        scroll.setStyleSheet(
            f"QScrollArea {{ background: {Config.get_color('bg_2')};"
            f" border: 1px solid {Config.get_color('border_1')}; }}"
        )
        viewport = scroll.viewport()
        if viewport is not None:
            viewport.setStyleSheet("background: transparent;")

        self._host = QWidget()
        self._host.setStyleSheet("background: transparent;")
        self._rows_layout = QVBoxLayout(self._host)
        self._rows_layout.setContentsMargins(2, 2, 2, 2)
        self._rows_layout.setSpacing(1)
        scroll.setWidget(self._host)
        outer.addWidget(scroll)

        self._reload()

    # ------------------------------------------------------------------

    def _reload(self) -> None:
        while self._rows_layout.count():
            item = self._rows_layout.takeAt(0)
            w = item.widget() if item is not None else None
            if w is not None:
                w.deleteLater()

        for clip in self._provider.list_clips(self._package_id):
            self._rows_layout.addWidget(
                _ClipItem(clip, self._provider, self._open_editor, self._open_segment)
            )
        self._rows_layout.addStretch(1)

    def _open_segment(self, clip_ref: str, lo: int, hi: int) -> None:
        self._open_editor(clip_ref, initial_selection=(lo, hi))

    def _open_editor(self, clip_ref: str, *, initial_selection: Optional[tuple] = None) -> None:
        from application.ui.dialogs.clip_crop_label_dialog import ClipCropLabelDialog
        dlg = ClipCropLabelDialog(
            clip_ref, provider=self._provider, parent=self.window(),
            initial_selection=initial_selection,
        )
        # Modeless: do NOT exec() (that locks focus). show() lets the user keep
        # working in the main window — and open several clip editors at once.
        dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self._open_editors.append(dlg)

        def _on_finished(_result: int, d=dlg) -> None:
            # Segment counts may have changed — refresh the row annotations.
            self._reload()
            try:
                self._open_editors.remove(d)
            except ValueError:
                pass

        dlg.finished.connect(_on_finished)
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()


__all__ = ["MotionClipTable"]
