"""AddResourceDialog — gather download parameters for a new third-party asset.

Triggered from ``ResourcesPanel`` by the "Add resource…" button. The
dialog presents:

* **Asset kind** — Motion / Policy (drives the destination subtree and
  the HuggingFace ``repo_type`` default).
* **Source** — GitHub clone / GitHub Release / HuggingFace (drives the
  transport implementation; placeholders adapt accordingly).
* **URL / repo id** — required.
* **Revision** — optional branch/tag/commit/HF revision.
* **Subpath** — optional sparse-checkout / HF ``allow_patterns``.
* **Display name** — optional override for the card title.

On accept the dialog calls
:meth:`ResourceManager.queue_download` directly; the panel learns about
the new entry via the ``resource_added`` signal from ``AppSignals``.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from unitport_sdk import (
    Config,
    LaviComboBox,
    LaviLineEdit,
    i18n_bind,
    setButton,
    setComboBox,
    setLineEdit,
    tr,
)

from application.service.resources import (
    ResourceKind,
    ResourceSource,
    TransportKind,
    get_resource_manager,
)


def _ss() -> int:
    return int(Config.get_font_size("size_small"))


def _mini() -> int:
    return int(Config.get_font_size("size_mini"))


def _apply_mini_font(edit: "LaviLineEdit") -> "LaviLineEdit":
    """Re-skin a LaviLineEdit to ``size_mini``.

    LaviLineEdit defaults to ``size_normal`` (set in its ctor). Inputs in
    this dialog should read in the smaller mini tier so the form reads
    as dense form chrome rather than primary content — we override the
    pixel size on the existing QFont and re-apply.
    """
    f = QFont(edit.font())
    f.setPixelSize(_mini())
    edit.setFont(f)
    return edit


# Combo item lists. Each tuple is (i18n_key, English fallback / display value).
_KIND_ITEMS: List[Tuple[str, str]] = [
    ("resources.kind.motion", "Motion"),
    ("resources.kind.policy", "Policy"),
]

_TRANSPORT_ITEMS: List[Tuple[str, str]] = [
    ("resources.transport.github_clone", "GitHub"),
    ("resources.transport.github_release", "Release"),
    ("resources.transport.huggingface", "HuggingFace"),
]

_KIND_VALUES = (ResourceKind.MOTION, ResourceKind.POLICY)
_TRANSPORT_VALUES = (
    TransportKind.GITHUB_CLONE,
    TransportKind.GITHUB_RELEASE,
    TransportKind.HUGGINGFACE,
)


class AddResourceDialog(QDialog):
    """Modal dialog. ``exec()`` returns ``QDialog.DialogCode.Accepted`` on submit."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("addResourceDialog")
        self.setModal(True)
        self.setWindowFlags(
            Qt.WindowType.Dialog
            | Qt.WindowType.WindowTitleHint
            | Qt.WindowType.CustomizeWindowHint
        )
        self.setWindowTitle(tr("resources.dialog.title", "Add resource"))
        self.setFixedWidth(460)

        self._kind_combo: Optional[LaviComboBox] = None
        self._transport_combo: Optional[LaviComboBox] = None
        self._url_edit: Optional[LaviLineEdit] = None
        self._revision_edit: Optional[LaviLineEdit] = None
        self._subpath_edit: Optional[LaviLineEdit] = None
        self._name_edit: Optional[LaviLineEdit] = None

        self._init_ui()
        self._on_transport_changed(0)  # seed placeholders

    # ----- UI ----------------------------------------------------------------

    def _init_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 18)
        root.setSpacing(10)

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)

        row = 0
        grid.addWidget(self._make_label("resources.dialog.kind_label", "Asset kind"),
                       row, 0, Qt.AlignmentFlag.AlignRight)
        self._kind_combo = setComboBox(_KIND_ITEMS, height=28, parent=self)
        grid.addWidget(self._kind_combo, row, 1)

        row += 1
        grid.addWidget(self._make_label("resources.dialog.transport_label", "Source"),
                       row, 0, Qt.AlignmentFlag.AlignRight)
        self._transport_combo = setComboBox(_TRANSPORT_ITEMS, height=28, parent=self)
        self._transport_combo.currentIndexChanged.connect(self._on_transport_changed)
        grid.addWidget(self._transport_combo, row, 1)

        row += 1
        grid.addWidget(self._make_label("resources.dialog.url_label", "URL or repo id"),
                       row, 0, Qt.AlignmentFlag.AlignRight)
        self._url_edit = _apply_mini_font(setLineEdit(placeholder="", parent=self))
        grid.addWidget(self._url_edit, row, 1)

        row += 1
        grid.addWidget(self._make_label("resources.dialog.revision_label", "Revision (optional)"),
                       row, 0, Qt.AlignmentFlag.AlignRight)
        self._revision_edit = _apply_mini_font(setLineEdit(
            placeholder=tr(
                "resources.dialog.revision_placeholder",
                "branch / tag / commit / HF revision",
            ),
            parent=self,
        ))
        grid.addWidget(self._revision_edit, row, 1)

        row += 1
        grid.addWidget(self._make_label("resources.dialog.subpath_label", "Subpath (optional)"),
                       row, 0, Qt.AlignmentFlag.AlignRight)
        self._subpath_edit = _apply_mini_font(
            setLineEdit(placeholder="", parent=self)
        )
        grid.addWidget(self._subpath_edit, row, 1)

        row += 1
        grid.addWidget(self._make_label("resources.dialog.name_label", "Display name (optional)"),
                       row, 0, Qt.AlignmentFlag.AlignRight)
        self._name_edit = _apply_mini_font(setLineEdit(
            placeholder=tr(
                "resources.dialog.name_label", "Display name (optional)",
            ),
            parent=self,
        ))
        grid.addWidget(self._name_edit, row, 1)

        root.addLayout(grid)

        # ----- buttons -----
        # Order: [Download] [Cancel] — primary action on the left so the
        # affirmative CTA reads first in the user's scan direction.
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)

        ok_btn = setButton(
            "resources.dialog.ok", 110, 28,
            kind="normal", spec="save",
            default=tr("resources.dialog.ok", "Download"),
        )
        ok_btn.clicked.connect(self._on_submit)
        btn_row.addWidget(ok_btn)

        cancel_btn = setButton(
            "resources.dialog.cancel", 96, 28,
            kind="border", spec="none",
            default=tr("resources.dialog.cancel", "Cancel"),
        )
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        root.addLayout(btn_row)

    def _make_label(self, key: str, default: str) -> QLabel:
        lbl = QLabel(self)
        i18n_bind(lbl, "setText", key, default)
        lbl.setStyleSheet(
            f"color: {Config.get_color('main_t1')};"
            f" background: transparent; font-size: {_ss()}px;"
        )
        return lbl

    # ----- behaviour ---------------------------------------------------------

    def _on_transport_changed(self, index: int) -> None:
        """Update URL/subpath placeholders for the picked transport."""
        if self._url_edit is None or self._subpath_edit is None:
            return
        if 0 <= index < len(_TRANSPORT_VALUES):
            transport = _TRANSPORT_VALUES[index]
        else:
            transport = TransportKind.GITHUB_CLONE
        if transport == TransportKind.GITHUB_CLONE:
            self._url_edit.setPlaceholderText(tr(
                "resources.dialog.url_placeholder_github",
                "https://github.com/owner/repo.git",
            ))
            self._subpath_edit.setPlaceholderText(tr(
                "resources.dialog.subpath_placeholder_github",
                "e.g. datasets/mocap_motions",
            ))
        elif transport == TransportKind.GITHUB_RELEASE:
            self._url_edit.setPlaceholderText(tr(
                "resources.dialog.url_placeholder_release",
                "https://github.com/owner/repo/releases/tag/v1.0",
            ))
            self._subpath_edit.setPlaceholderText("")
        else:
            self._url_edit.setPlaceholderText(tr(
                "resources.dialog.url_placeholder_huggingface",
                "owner/name  (or full HF URL)",
            ))
            self._subpath_edit.setPlaceholderText(tr(
                "resources.dialog.subpath_placeholder_huggingface",
                "allow patterns, e.g. *.json,*.txt",
            ))

    def _on_submit(self) -> None:
        url = self._url_edit.text().strip() if self._url_edit is not None else ""
        if not url:
            QMessageBox.warning(
                self,
                tr("resources.dialog.title", "Add resource"),
                tr(
                    "resources.dialog.url_label",
                    "URL or repo id is required.",
                ),
            )
            return
        kind_idx = self._kind_combo.currentIndex() if self._kind_combo else 0
        transport_idx = (
            self._transport_combo.currentIndex() if self._transport_combo else 0
        )
        kind = _KIND_VALUES[max(0, min(kind_idx, len(_KIND_VALUES) - 1))]
        transport = _TRANSPORT_VALUES[
            max(0, min(transport_idx, len(_TRANSPORT_VALUES) - 1))
        ]
        revision = self._revision_edit.text().strip() if self._revision_edit else ""
        subpath = self._subpath_edit.text().strip() if self._subpath_edit else ""
        name = self._name_edit.text().strip() if self._name_edit else ""

        source = ResourceSource(url=url, revision=revision, subpath=subpath)
        try:
            get_resource_manager().queue_download(
                asset_kind=kind,
                transport=transport,
                name=name,
                source=source,
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                tr("resources.dialog.title", "Add resource"),
                f"{type(e).__name__}: {e}",
            )
            return
        self.accept()


__all__ = ["AddResourceDialog"]
