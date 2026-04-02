#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CheckpointImportDialog — modal dialog for importing checkpoint bundles.

Supports two modes:

* **Local** — browse for a directory or .zip archive containing manifest.yaml.
* **HuggingFace** — paste a HuggingFace URL; the dialog downloads and
  normalizes the checkpoint in a background thread before importing.

Signals
-------
import_confirmed(str)
    Emitted when the user confirms a local import (source path string).
hf_import_confirmed(str)
    Emitted when HuggingFace download + normalization succeed and the user
    clicks "Import" (normalized bundle directory path string).
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)


class CheckpointImportDialog(QDialog):
    """
    Modal dialog for importing a checkpoint bundle into ``custom_mods/training/checkpoints/``.

    Signals
    -------
    import_confirmed(str)
        Emitted when the user clicks "Import" in **Local** mode.
        Argument: resolved source path (directory or .zip).
    hf_import_confirmed(str)
        Emitted when the user clicks "Import" after a successful HuggingFace
        download.  Argument: normalized bundle directory path (temporary;
        caller must install via ``CheckpointRegistry`` and then clean up).
    """

    import_confirmed = Signal(str)      # local mode: src path string
    hf_import_confirmed = Signal(str)   # HF mode: normalized bundle dir string

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Add Checkpoint")
        self.setModal(True)
        self.setMinimumWidth(500)
        self.setObjectName("checkpointImportDialog")

        self._src_path: Optional[Path] = None
        self._preview_data: Optional[dict] = None
        self._mode: str = "local"           # "local" | "hf"

        # HF-specific state
        self._hf_bundle_path: Optional[Path] = None  # normalized temp bundle
        self._hf_thread = None                        # HFDownloadThread

        self._build_ui()
        self._update_state()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 16)
        root.setSpacing(14)

        root.addWidget(self._build_source_selector())

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        root.addWidget(sep)

        # Input stack: page 0 = local, page 1 = HF
        self._input_stack = QStackedWidget()
        self._input_stack.addWidget(self._build_local_input())   # index 0
        self._input_stack.addWidget(self._build_hf_input())      # index 1
        root.addWidget(self._input_stack)

        root.addWidget(self._build_preview_section())

        self.status_label = QLabel("")
        self.status_label.setObjectName("importStatusLabel")
        self.status_label.setWordWrap(True)
        self.status_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        root.addWidget(self.status_label)

        self.button_box = QDialogButtonBox()
        self.import_button = self.button_box.addButton(
            "Import", QDialogButtonBox.ButtonRole.AcceptRole
        )
        self.import_button.setObjectName("importConfirmButton")
        self.import_button.setEnabled(False)
        cancel_button = self.button_box.addButton(
            "Cancel", QDialogButtonBox.ButtonRole.RejectRole
        )
        cancel_button.setObjectName("importCancelButton")
        self.button_box.accepted.connect(self._on_import_clicked)
        self.button_box.rejected.connect(self.reject)
        root.addWidget(self.button_box)

    def _build_source_selector(self) -> QWidget:
        container = QWidget()
        container.setObjectName("importSourceSelector")
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        heading = QLabel("Source:")
        heading.setFixedWidth(60)
        layout.addWidget(heading)

        self.local_btn = QPushButton("📁  Local File / Folder")
        self.local_btn.setObjectName("importSourceButton")
        self.local_btn.setCheckable(True)
        self.local_btn.setChecked(True)
        self.local_btn.setCursor(Qt.PointingHandCursor)
        self.local_btn.clicked.connect(self._on_source_local)
        layout.addWidget(self.local_btn, 1)

        self.hf_btn = QPushButton("🌐  HuggingFace")
        self.hf_btn.setObjectName("importSourceButton")
        self.hf_btn.setCheckable(True)
        self.hf_btn.setCursor(Qt.PointingHandCursor)
        self.hf_btn.clicked.connect(self._on_source_hf)
        layout.addWidget(self.hf_btn, 1)

        return container

    # ---- Local input panel ----

    def _build_local_input(self) -> QWidget:
        container = QWidget()
        container.setObjectName("importPathRow")
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.path_input = QLineEdit()
        self.path_input.setObjectName("importPathInput")
        self.path_input.setPlaceholderText("Select a folder or .zip archive…")
        self.path_input.setReadOnly(True)
        self.path_input.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        layout.addWidget(self.path_input, 1)

        self.browse_btn = QPushButton("Browse…")
        self.browse_btn.setObjectName("importBrowseButton")
        self.browse_btn.setCursor(Qt.PointingHandCursor)
        self.browse_btn.setFixedWidth(80)
        self.browse_btn.clicked.connect(self._on_browse)
        layout.addWidget(self.browse_btn)

        return container

    # ---- HuggingFace input panel ----

    def _build_hf_input(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # URL row
        url_row = QWidget()
        url_layout = QHBoxLayout(url_row)
        url_layout.setContentsMargins(0, 0, 0, 0)
        url_layout.setSpacing(8)

        url_label = QLabel("URL:")
        url_label.setFixedWidth(36)
        url_layout.addWidget(url_label)

        self.hf_url_input = QLineEdit()
        self.hf_url_input.setObjectName("importHFUrlInput")
        self.hf_url_input.setPlaceholderText(
            "https://huggingface.co/<owner>/<repo>/tree/<revision>"
        )
        self.hf_url_input.textChanged.connect(self._on_hf_url_changed)
        url_layout.addWidget(self.hf_url_input, 1)

        layout.addWidget(url_row)

        # Repo preview label
        self.hf_repo_label = QLabel("")
        self.hf_repo_label.setObjectName("importHFRepoLabel")
        self.hf_repo_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.hf_repo_label)

        # Download controls row
        ctrl_row = QWidget()
        ctrl_layout = QHBoxLayout(ctrl_row)
        ctrl_layout.setContentsMargins(0, 0, 0, 0)
        ctrl_layout.setSpacing(8)

        self.hf_download_btn = QPushButton("⬇  Download")
        self.hf_download_btn.setObjectName("importHFDownloadButton")
        self.hf_download_btn.setCursor(Qt.PointingHandCursor)
        self.hf_download_btn.setEnabled(False)
        self.hf_download_btn.clicked.connect(self._on_hf_download)
        ctrl_layout.addWidget(self.hf_download_btn, 1)

        self.hf_cancel_btn = QPushButton("✕  Cancel Download")
        self.hf_cancel_btn.setObjectName("importHFCancelButton")
        self.hf_cancel_btn.setCursor(Qt.PointingHandCursor)
        self.hf_cancel_btn.setEnabled(False)
        self.hf_cancel_btn.clicked.connect(self._on_hf_cancel)
        ctrl_layout.addWidget(self.hf_cancel_btn, 1)

        layout.addWidget(ctrl_row)

        # Progress label
        self.hf_progress_label = QLabel("")
        self.hf_progress_label.setObjectName("importHFProgressLabel")
        self.hf_progress_label.setWordWrap(True)
        layout.addWidget(self.hf_progress_label)

        return container

    # ---- Preview section (shared) ----

    def _build_preview_section(self) -> QWidget:
        container = QFrame()
        container.setObjectName("importPreviewBox")
        container.setFrameShape(QFrame.Shape.StyledPanel)

        grid = QGridLayout(container)
        grid.setContentsMargins(12, 10, 12, 10)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(6)

        self._preview_labels: dict[str, QLabel] = {}

        # --- Basic fields (read-only labels) ---
        basic_fields = [
            ("policy_id", "Policy ID"),
            ("name", "Display Name"),
            ("version", "Version"),
            ("robot_brand", "Robot Brand"),
            ("obs_dim", "Obs Dim"),
            ("action_dim", "Action Dim"),
        ]
        for row, (key, caption) in enumerate(basic_fields):
            caption_lbl = QLabel(f"{caption}:")
            caption_lbl.setObjectName("importPreviewCaption")
            caption_lbl.setAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            grid.addWidget(caption_lbl, row, 0)

            value_lbl = QLabel("—")
            value_lbl.setObjectName("importPreviewValue")
            value_lbl.setAlignment(
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
            )
            value_lbl.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )
            grid.addWidget(value_lbl, row, 1)
            self._preview_labels[key] = value_lbl

        # --- SkillManifest v2 section (editable) ---
        separator_row = len(basic_fields)
        skill_heading = QLabel("Skill Manifest")
        skill_heading.setObjectName("importSkillHeading")
        skill_heading.setStyleSheet("font-weight: bold; margin-top: 6px;")
        grid.addWidget(skill_heading, separator_row, 0, 1, 2)

        self._skill_edits: dict[str, QWidget] = {}
        skill_row = separator_row + 1

        # Action Space Type (combo box)
        grid.addWidget(self._caption_label("Action Space:"), skill_row, 0)
        self._action_space_combo = QComboBox()
        self._action_space_combo.setObjectName("importSkillActionSpace")
        self._action_space_combo.addItems([
            "joint_position", "joint_torque", "cartesian_delta",
            "end_effector_6d", "hybrid",
        ])
        grid.addWidget(self._action_space_combo, skill_row, 1)
        self._skill_edits["action_space_type"] = self._action_space_combo
        skill_row += 1

        # Control Frequency
        grid.addWidget(self._caption_label("Ctrl Freq (Hz):"), skill_row, 0)
        self._freq_edit = QLineEdit("50.0")
        self._freq_edit.setObjectName("importSkillFreq")
        grid.addWidget(self._freq_edit, skill_row, 1)
        self._skill_edits["control_frequency_hz"] = self._freq_edit
        skill_row += 1

        # Robot Family (combo box)
        grid.addWidget(self._caption_label("Robot Family:"), skill_row, 0)
        self._robot_family_combo = QComboBox()
        self._robot_family_combo.setObjectName("importSkillRobotFamily")
        self._robot_family_combo.addItems([
            "quadruped", "biped", "manipulator", "wheeled",
        ])
        grid.addWidget(self._robot_family_combo, skill_row, 1)
        self._skill_edits["target_robot_family"] = self._robot_family_combo
        skill_row += 1

        # Precondition Posture (combo box)
        grid.addWidget(self._caption_label("Precondition:"), skill_row, 0)
        self._pre_posture_combo = QComboBox()
        self._pre_posture_combo.setObjectName("importSkillPrePosture")
        self._pre_posture_combo.addItems([
            "any", "standing", "sitting", "prone", "custom",
        ])
        grid.addWidget(self._pre_posture_combo, skill_row, 1)
        self._skill_edits["precondition_posture"] = self._pre_posture_combo
        skill_row += 1

        # Postcondition Posture (combo box)
        grid.addWidget(self._caption_label("Postcondition:"), skill_row, 0)
        self._post_posture_combo = QComboBox()
        self._post_posture_combo.setObjectName("importSkillPostPosture")
        self._post_posture_combo.addItems([
            "standing", "sitting", "prone", "any", "custom",
        ])
        grid.addWidget(self._post_posture_combo, skill_row, 1)
        self._skill_edits["postcondition_posture"] = self._post_posture_combo
        skill_row += 1

        # Inference Backend (read-only label)
        grid.addWidget(self._caption_label("Backend:"), skill_row, 0)
        backend_lbl = QLabel("—")
        backend_lbl.setObjectName("importPreviewValue")
        backend_lbl.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        grid.addWidget(backend_lbl, skill_row, 1)
        self._preview_labels["inference_backend"] = backend_lbl
        skill_row += 1

        # Source Type (read-only label)
        grid.addWidget(self._caption_label("Source Type:"), skill_row, 0)
        source_lbl = QLabel("—")
        source_lbl.setObjectName("importPreviewValue")
        source_lbl.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        grid.addWidget(source_lbl, skill_row, 1)
        self._preview_labels["source_type"] = source_lbl

        # Initially hide skill section
        self._skill_section_widgets = [skill_heading]
        for w in self._skill_edits.values():
            self._skill_section_widgets.append(w)

        return container

    @staticmethod
    def _caption_label(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("importPreviewCaption")
        lbl.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        return lbl

    # ------------------------------------------------------------------
    # Source mode switching
    # ------------------------------------------------------------------

    def _on_source_local(self) -> None:
        self._mode = "local"
        self.local_btn.setChecked(True)
        self.hf_btn.setChecked(False)
        self._input_stack.setCurrentIndex(0)
        self._clear_preview()
        self.status_label.setText("")
        self._src_path = None
        self._preview_data = None
        self._update_state()

    def _on_source_hf(self) -> None:
        self._mode = "hf"
        self.hf_btn.setChecked(True)
        self.local_btn.setChecked(False)
        self._input_stack.setCurrentIndex(1)
        self._clear_preview()
        self.status_label.setText("")
        self._hf_bundle_path = None
        self._preview_data = None
        self._update_state()

    # ------------------------------------------------------------------
    # Local mode interactions
    # ------------------------------------------------------------------

    def _on_browse(self) -> None:
        path_str = QFileDialog.getExistingDirectory(
            self,
            "Select checkpoint bundle folder",
            str(Path.home()),
            QFileDialog.Option.ShowDirsOnly,
        )
        if not path_str:
            path_str, _ = QFileDialog.getOpenFileName(
                self,
                "Select checkpoint bundle .zip",
                str(Path.home()),
                "ZIP archives (*.zip)",
            )
        if not path_str:
            return

        src = Path(path_str)
        self.path_input.setText(str(src))
        self._src_path = src
        self._load_preview(src)

    def _load_preview(self, src: Path) -> None:
        self._src_path = src
        self._preview_data = None
        self._clear_preview()
        self.status_label.setText("")
        self.import_button.setEnabled(False)

        try:
            from src.system.service.checkpoint_registry import CheckpointRegistry

            data = CheckpointRegistry.peek_manifest(src)
            self._preview_data = data
            self._populate_preview(data)
            self.status_label.setText("")
        except Exception as exc:  # noqa: BLE001
            self.status_label.setText(f"Error: {exc}")

        self._update_state()

    # ------------------------------------------------------------------
    # HuggingFace mode interactions
    # ------------------------------------------------------------------

    def _on_hf_url_changed(self, text: str) -> None:
        """Update the repo preview label as the user types."""
        text = text.strip()
        self._hf_bundle_path = None
        self._preview_data = None
        self._clear_preview()
        self.status_label.setText("")

        if not text:
            self.hf_repo_label.setText("")
            self.hf_download_btn.setEnabled(False)
            self._update_state()
            return

        try:
            from src.system.training.hf_downloader import parse_hf_url

            repo_id, revision = parse_hf_url(text)
            self.hf_repo_label.setText(f"  {repo_id}  @  {revision}")
            self.hf_download_btn.setEnabled(True)
        except Exception:
            self.hf_repo_label.setText("  (invalid URL)")
            self.hf_download_btn.setEnabled(False)

        self._update_state()

    def _on_hf_download(self) -> None:
        """Start the background download thread."""
        url = self.hf_url_input.text().strip()
        if not url:
            return

        # Clean up any previous temp bundle
        self._cleanup_hf_bundle()

        from src.system.core.hf_download_thread import HFDownloadThread

        self._hf_thread = HFDownloadThread(url, parent=self)
        self._hf_thread.progress.connect(self._on_hf_progress)
        self._hf_thread.finished.connect(self._on_hf_finished)
        self._hf_thread.failed.connect(self._on_hf_failed)
        self._hf_thread.start()

        # Update UI for in-progress state
        self.hf_download_btn.setEnabled(False)
        self.hf_cancel_btn.setEnabled(True)
        self.hf_url_input.setEnabled(False)
        self.hf_progress_label.setText("Starting download…")
        self.status_label.setText("")
        self.import_button.setEnabled(False)
        self._clear_preview()

    def _on_hf_cancel(self) -> None:
        """Cancel the running download thread."""
        if self._hf_thread is not None:
            self._hf_thread.cancel()
        self._reset_hf_controls()
        self.hf_progress_label.setText("Cancelled.")

    def _on_hf_progress(self, msg: str) -> None:
        self.hf_progress_label.setText(msg)

    def _on_hf_finished(self, bundle_path_str: str) -> None:
        """Download completed; load preview and enable Import."""
        self._hf_bundle_path = Path(bundle_path_str)
        self._reset_hf_controls()
        self.hf_progress_label.setText("Download complete.")

        try:
            from src.system.service.checkpoint_registry import CheckpointRegistry

            data = CheckpointRegistry.peek_manifest(self._hf_bundle_path)
            self._preview_data = data
            self._populate_preview(data)
            self.status_label.setText("")
        except Exception as exc:  # noqa: BLE001
            self.status_label.setText(f"Preview error: {exc}")

        self._update_state()

    def _on_hf_failed(self, error_msg: str) -> None:
        self._hf_bundle_path = None
        self._preview_data = None
        self._reset_hf_controls()
        self.hf_progress_label.setText("")
        self.status_label.setText(f"Error: {error_msg}")
        self._update_state()

    def _reset_hf_controls(self) -> None:
        self.hf_download_btn.setEnabled(True)
        self.hf_cancel_btn.setEnabled(False)
        self.hf_url_input.setEnabled(True)

    # ------------------------------------------------------------------
    # Import action
    # ------------------------------------------------------------------

    def _on_import_clicked(self) -> None:
        if self._mode == "local":
            if self._src_path is None:
                return
            self._apply_skill_edits(self._src_path)
            self.import_confirmed.emit(str(self._src_path))
            self.accept()
        elif self._mode == "hf":
            if self._hf_bundle_path is None:
                return
            self._apply_skill_edits(self._hf_bundle_path)
            self.hf_import_confirmed.emit(str(self._hf_bundle_path))
            self._hf_bundle_path = None   # ownership transferred to caller
            self.accept()

    # ------------------------------------------------------------------
    # Preview helpers
    # ------------------------------------------------------------------

    def _clear_preview(self) -> None:
        for lbl in self._preview_labels.values():
            lbl.setText("—")
        # Reset skill edits to defaults
        self._action_space_combo.setCurrentIndex(0)
        self._freq_edit.setText("50.0")
        self._robot_family_combo.setCurrentIndex(0)
        self._pre_posture_combo.setCurrentIndex(0)
        self._post_posture_combo.setCurrentIndex(0)

    def _populate_preview(self, data: dict) -> None:
        for key, lbl in self._preview_labels.items():
            val = data.get(key, "—")
            lbl.setText(str(val) if val not in ("", None) else "—")

        # Populate skill manifest editable fields
        if data.get("has_skill_manifest"):
            self._set_combo(self._action_space_combo, data.get("action_space_type", ""))
            freq = data.get("control_frequency_hz", "")
            self._freq_edit.setText(str(freq) if freq else "50.0")
            self._set_combo(self._robot_family_combo, data.get("target_robot_family", ""))
            self._set_combo(self._pre_posture_combo, data.get("precondition_posture", ""))
            self._set_combo(self._post_posture_combo, data.get("postcondition_posture", ""))

    @staticmethod
    def _set_combo(combo: QComboBox, value: str) -> None:
        """Set combo box to *value* if it exists, otherwise leave at index 0."""
        if not value:
            return
        idx = combo.findText(str(value))
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def _collect_skill_edits(self) -> dict:
        """Collect user-edited skill manifest fields from the UI widgets."""
        try:
            freq = float(self._freq_edit.text().strip())
        except (ValueError, AttributeError):
            freq = 50.0
        return {
            "action_space_type": self._action_space_combo.currentText(),
            "control_frequency_hz": freq,
            "target_robot_family": self._robot_family_combo.currentText(),
            "precondition": {
                "posture": self._pre_posture_combo.currentText(),
                "posture_tolerance_rad": 0.3,
                "velocity_max_mps": 1.0,
            },
            "postcondition": {
                "posture": self._post_posture_combo.currentText(),
                "posture_tolerance_rad": 0.3,
                "velocity_max_mps": 0.5,
            },
        }

    def _apply_skill_edits(self, bundle_path: Path) -> None:
        """Merge user-edited skill fields back into the bundle's manifest.yaml."""
        manifest_path = bundle_path / "manifest.yaml"
        if not manifest_path.exists():
            return
        try:
            import yaml

            with manifest_path.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}

            skill = data.get("skill")
            if not isinstance(skill, dict):
                return  # No skill section to update

            edits = self._collect_skill_edits()
            skill["action_space_type"] = edits["action_space_type"]
            skill["control_frequency_hz"] = edits["control_frequency_hz"]
            skill["target_robot_family"] = edits["target_robot_family"]
            skill["precondition"] = edits["precondition"]
            skill["postcondition"] = edits["postcondition"]

            with manifest_path.open("w", encoding="utf-8") as fh:
                yaml.dump(data, fh, default_flow_style=False, allow_unicode=True)
        except Exception:  # noqa: BLE001
            pass  # Best-effort; import proceeds even if edit write fails

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def _update_state(self) -> None:
        if self._mode == "local":
            ok = self._src_path is not None and self._preview_data is not None
        else:
            ok = self._hf_bundle_path is not None and self._preview_data is not None
        self.import_button.setEnabled(ok)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _cleanup_hf_bundle(self) -> None:
        """Remove any leftover temporary HF bundle directory."""
        if self._hf_bundle_path is not None and self._hf_bundle_path.exists():
            shutil.rmtree(self._hf_bundle_path.parent, ignore_errors=True)
            self._hf_bundle_path = None

    def closeEvent(self, event) -> None:
        """Cancel running thread and clean up temp bundle on close."""
        if self._hf_thread is not None and self._hf_thread.isRunning():
            self._hf_thread.cancel()
            self._hf_thread.wait(3000)
        self._cleanup_hf_bundle()
        super().closeEvent(event)

    def reject(self) -> None:
        """Cancel on dialog dismiss."""
        if self._hf_thread is not None and self._hf_thread.isRunning():
            self._hf_thread.cancel()
            self._hf_thread.wait(3000)
        self._cleanup_hf_bundle()
        super().reject()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def selected_path(self) -> Optional[Path]:
        """Return the validated local source path, or None if not set."""
        return self._src_path

    def preview_data(self) -> Optional[dict]:
        """Return the last successfully loaded preview dict, or None."""
        return self._preview_data
