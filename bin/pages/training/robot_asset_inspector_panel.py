#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RobotAssetInspectorPanel — read-only inspector for a registered RobotAsset.

Per AMP_design.yaml §5.1.robot_asset_inspector_panel:

- Joint list (canonical order + default joint position)
- Link list with foot link highlights
- If both usd_path and mjcf_path are present, show a side-by-side joint
  comparison table (green = same name, red = diff). The actual cross-check
  is deferred to the launcher subprocess (path b), but mjcf-side names are
  parsable in the main venv right now and we display them here.
- No 3D mesh rendering — that would require a mesh loader, which is out of
  scope for "receive + validate".

The panel only consumes a ``RobotAsset`` instance from
``src.system.training.robot_assets.registry`` — it does no parsing of its
own. This keeps the panel a pure view layer.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.system.training.robot_assets import RobotAsset


# ---------------------------------------------------------------------------
# Color palette (deliberately neutral; matches existing training panels)
# ---------------------------------------------------------------------------

_COLOR_OK = QColor("#1e7e34")        # green
_COLOR_DIFF = QColor("#c82333")      # red
_COLOR_FOOT = QColor("#856404")      # amber for foot link rows
_COLOR_WARN = QColor("#856404")


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------


class RobotAssetInspectorPanel(QWidget):
    """Read-only inspector. Set the asset via :meth:`set_asset`.

    Layout:

    +-------------------------------------------------------------+
    | Header line: asset_id · family · primary metadata source    |
    +-----------------+-------------------------------------------+
    | Joint table     |  Link tree (with foot highlight)          |
    | (name, default) |                                           |
    +-----------------+-------------------------------------------+
    | Source comparison tab (only when usd + mjcf both present)   |
    +-------------------------------------------------------------+
    | Warnings text area                                          |
    +-------------------------------------------------------------+
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._asset: Optional[RobotAsset] = None

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # ── Header ──
        self._header_label = QLabel("(no asset loaded)")
        self._header_label.setStyleSheet(
            "font-weight: 600; padding: 4px; "
            "background: rgba(0, 0, 0, 0.04); border-radius: 4px;"
        )
        root.addWidget(self._header_label)

        # ── Tabs ──
        self._tabs = QTabWidget()
        root.addWidget(self._tabs, 1)

        # Tab 1: Joints + Links splitter
        self._tab_main = QWidget()
        self._build_main_tab()
        self._tabs.addTab(self._tab_main, "Joints && Links")

        # Tab 2: Source comparison (created lazily; only shown when both
        # usd and mjcf are present)
        self._tab_compare = QWidget()
        self._build_compare_tab()
        self._tabs.addTab(self._tab_compare, "Source Comparison")
        self._tabs.setTabVisible(self._tabs.indexOf(self._tab_compare), False)

        # ── Warnings ──
        self._warnings = QTextEdit()
        self._warnings.setReadOnly(True)
        self._warnings.setFixedHeight(80)
        self._warnings.setPlaceholderText("(no warnings)")
        root.addWidget(self._warnings)

    # ------------------------------------------------------------------
    # Layout helpers
    # ------------------------------------------------------------------

    def _build_main_tab(self) -> None:
        layout = QHBoxLayout(self._tab_main)
        layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Horizontal)

        # Joint table
        self._joint_table = QTableWidget(0, 2)
        self._joint_table.setHorizontalHeaderLabels(["Joint", "Default Pos"])
        self._joint_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )
        self._joint_table.verticalHeader().setVisible(False)
        self._joint_table.setEditTriggers(QTableWidget.NoEditTriggers)
        splitter.addWidget(self._joint_table)

        # Link tree
        self._link_tree = QTreeWidget()
        self._link_tree.setHeaderLabel("Links (foot links highlighted)")
        splitter.addWidget(self._link_tree)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter)

    def _build_compare_tab(self) -> None:
        layout = QVBoxLayout(self._tab_compare)
        info = QLabel(
            "Joint name comparison between mjcf and usd sources. "
            "USD joint extraction is deferred to the launcher subprocess "
            "(see AMP_design.yaml §7 path b), so this view shows mjcf-side "
            "names only until training launches."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #666; font-size: 11px; padding: 4px;")
        layout.addWidget(info)

        self._compare_table = QTableWidget(0, 3)
        self._compare_table.setHorizontalHeaderLabels(
            ["Index", "MJCF joint", "USD joint (deferred)"]
        )
        self._compare_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )
        self._compare_table.verticalHeader().setVisible(False)
        self._compare_table.setEditTriggers(QTableWidget.NoEditTriggers)
        layout.addWidget(self._compare_table, 1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_asset(self, asset: Optional[RobotAsset]) -> None:
        """Render an asset (or clear the view if ``None``)."""
        self._asset = asset
        if asset is None:
            self._header_label.setText("(no asset loaded)")
            self._joint_table.setRowCount(0)
            self._link_tree.clear()
            self._compare_table.setRowCount(0)
            self._warnings.clear()
            self._tabs.setTabVisible(self._tabs.indexOf(self._tab_compare), False)
            return

        primary = asset.source_meta.get("primary", "?")
        self._header_label.setText(
            f"{asset.asset_id}  ·  family={asset.family}  ·  "
            f"dof={asset.dof_count}  ·  base_link={asset.base_link or '?'}  ·  "
            f"primary={primary}"
        )

        # Joints
        self._joint_table.setRowCount(len(asset.joint_names))
        for row, jname in enumerate(asset.joint_names):
            name_item = QTableWidgetItem(jname)
            default = asset.default_joint_pos.get(jname, 0.0)
            pos_item = QTableWidgetItem(f"{default:.4f}")
            self._joint_table.setItem(row, 0, name_item)
            self._joint_table.setItem(row, 1, pos_item)

        # Links
        self._link_tree.clear()
        foot_set = set(asset.foot_link_names)
        # Without a parent map (the dataclass doesn't store one), render a
        # flat list. The link tree from the parsers is internal — exposing
        # it via RobotAsset is a follow-up if a real tree view becomes useful.
        # We do however indicate the base link prominently as the root.
        if asset.base_link:
            base_item = QTreeWidgetItem([f"{asset.base_link} (base)"])
            self._link_tree.addTopLevelItem(base_item)
            base_item.setExpanded(True)
            for link_name in asset.foot_link_names:
                child = QTreeWidgetItem([link_name])
                child.setForeground(0, _COLOR_FOOT)
                base_item.addChild(child)

        # Comparison tab — visible iff both usd and mjcf are present
        both_sources = bool(asset.usd_path and asset.mjcf_path)
        self._tabs.setTabVisible(
            self._tabs.indexOf(self._tab_compare), both_sources
        )
        if both_sources:
            self._compare_table.setRowCount(len(asset.joint_names))
            for row, jname in enumerate(asset.joint_names):
                idx_item = QTableWidgetItem(str(row))
                mjcf_item = QTableWidgetItem(jname)
                # USD side is deferred — show placeholder
                usd_item = QTableWidgetItem("(verified at launcher)")
                usd_item.setForeground(_COLOR_WARN)
                self._compare_table.setItem(row, 0, idx_item)
                self._compare_table.setItem(row, 1, mjcf_item)
                self._compare_table.setItem(row, 2, usd_item)
        else:
            self._compare_table.setRowCount(0)

        # Warnings
        warnings = asset.source_meta.get("warnings", [])
        if warnings:
            self._warnings.setPlainText("\n".join(f"• {w}" for w in warnings))
        else:
            self._warnings.clear()


# ---------------------------------------------------------------------------
# Standalone entry for manual testing (python robot_asset_inspector_panel.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    import sys
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    panel = RobotAssetInspectorPanel()

    # Mock asset for visual smoke-test
    mock = RobotAsset(
        asset_id="wildcat-mock",
        family="quadruped",
        joint_names=[f"FL_hip_{i}" for i in range(3)] + [f"FR_hip_{i}" for i in range(3)],
        dof_count=6,
        foot_link_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
        default_joint_pos={f"FL_hip_{i}": 0.1 * i for i in range(3)},
        base_link="trunk",
        mass=12.5,
        source_meta={"primary": "mjcf", "warnings": ["mock data"]},
    )
    panel.set_asset(mock)
    panel.resize(720, 520)
    panel.show()
    sys.exit(app.exec())
