"""Robot asset registry — registers.robots + per-asset user state."""

from __future__ import annotations

from .service import (
    ASSET_KINDS,
    STATUS_LOCAL,
    STATUS_MISSING,
    STATUS_REMOTE,
    AssetRecord,
    AssetStatus,
    RobotAsset,
    RobotAssetService,
    get_robot_asset_service,
    selected_robot,
    selected_robot_sku,
    set_selected_robot,
)

__all__ = [
    "ASSET_KINDS",
    "STATUS_LOCAL",
    "STATUS_REMOTE",
    "STATUS_MISSING",
    "AssetRecord",
    "AssetStatus",
    "RobotAsset",
    "RobotAssetService",
    "get_robot_asset_service",
    "selected_robot",
    "selected_robot_sku",
    "set_selected_robot",
]
