#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Checkpoint Node — loads a policy bundle from CheckpointRegistry.

CheckpointNode.execute() resolves the policy_id parameter against
CheckpointRegistry, loads the bundle via BundleLoader, and outputs
a CheckpointBundle dataclass on the ``checkpoint`` port for downstream
nodes (e.g. BehaviorNode) to consume.
"""

from typing import Dict, Any
from .base_node import BaseNode


class CheckpointNode(BaseNode):
    """
    Checkpoint node — loads a deployed policy bundle at runtime.

    The user selects a bundle by policy_id (discovered by CheckpointRegistry
    from custom_checkpoints/).  On execute(), the registry is queried for the
    bundle path and BundleLoader validates and returns a CheckpointBundle.

    Outputs a CheckpointBundle dataclass on the ``checkpoint`` port.
    """

    def __init__(self, node_id: str):
        super().__init__(node_id, "checkpoint")
        self.inputs = {}
        self.outputs = {"checkpoint": None}
        self.parameters = {
            "policy_id": "",     # bundle name from CheckpointRegistry
            "format": "auto",    # auto | onnx | jit  (resolved from manifest)
        }

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        from system.policy.bundle_loader import BundleLoader
        from system.service.checkpoint_registry import CheckpointRegistry

        policy_id = self.get_parameter("policy_id", "")
        if not policy_id:
            return {
                "checkpoint": {
                    "status": "error",
                    "error": "policy_id parameter is not set",
                }
            }

        # Resolve bundle path from registry
        try:
            registry = CheckpointRegistry()
            bundle_path = registry.get_bundle_path(policy_id)
        except KeyError:
            return {
                "checkpoint": {
                    "status": "error",
                    "error": f"policy_id '{policy_id}' not found in CheckpointRegistry",
                }
            }

        # Load and validate the bundle
        try:
            bundle = BundleLoader().load(bundle_path)
        except Exception as exc:  # noqa: BLE001
            return {
                "checkpoint": {
                    "status": "error",
                    "error": str(exc),
                }
            }

        return {
            "checkpoint": {
                "bundle": bundle,
                "bundle_path": str(bundle_path),
                "policy_id": policy_id,
            }
        }

    def get_display_name(self) -> str:
        return "Checkpoint"

    def get_description(self) -> str:
        return "Load policy checkpoint bundle from CheckpointRegistry"
