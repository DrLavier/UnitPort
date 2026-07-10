# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""ClipCropLabelDialog — the standalone **Clip Motion Editor** window.

Thin modeless ``QDialog`` wrapper around :class:`ClipMotionEditorWidget`
(the reusable editor surface). This is the Resources-panel entry point: a
motion clip opens in its own window with a render-robot picker in the top
row and a Close button at the bottom. The Training Motion Editor embeds the
*same* widget instead (fixed robot + Apply column), so the render / timeline
/ segment logic lives in exactly one place (§11).

The class name + constructor signature are kept stable for the single caller
in ``clip_table.py`` (``ClipCropLabelDialog(clip_ref, provider=…, parent=…,
initial_selection=…)``).
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDialog, QHBoxLayout, QVBoxLayout, QWidget

from unitport_sdk import setButton, tr

from application.ui.dialogs.clip_motion_editor_widget import ClipMotionEditorWidget


class ClipCropLabelDialog(QDialog):
    """Standalone modeless window hosting a :class:`ClipMotionEditorWidget`."""

    def __init__(
        self,
        clip_ref: str,
        *,
        provider,
        parent: Optional[QWidget] = None,
        initial_selection: Optional[tuple] = None,
    ) -> None:
        super().__init__(parent)
        # Modeless: never lock focus — the user must stay free to interact with
        # the main window (and other editors) while this one is open. The caller
        # opens us with show() (not exec()) and keeps a reference.
        self.setModal(False)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setWindowTitle(tr("clip_editor.title", "Clip Motion Editor"))
        self.resize(820, 760)
        self.setMinimumSize(680, 600)

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(8)

        self._editor = ClipMotionEditorWidget(
            provider=provider,
            parent=self,
            clip_ref=clip_ref,
            initial_selection=initial_selection,
            show_robot_picker=True,
            show_reference_select=False,
        )
        root.addWidget(self._editor, 1)

        # ── Close ──
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        close_btn = setButton(
            "clip_editor.close", 96, 28, kind="border", spec="none",
            default=tr("clip_editor.close", "Close"),
        )
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        root.addLayout(btn_row)

    # ------------------------------------------------------------------
    # Lifecycle — release the render worker's GL context on close.
    # ------------------------------------------------------------------

    def closeEvent(self, e) -> None:
        self._editor.teardown()
        super().closeEvent(e)

    def done(self, r: int) -> None:
        self._editor.teardown()
        super().done(r)


__all__ = ["ClipCropLabelDialog"]
