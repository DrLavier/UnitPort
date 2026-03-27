#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# DEPRECATED — PolicyNode has been removed from the Mission Canvas.
# Policy execution is now handled entirely by BehaviorNode (behavior_node.py).
# This file is retained for import compatibility only; do NOT register this
# node type in nodes/__init__.py or nodes/sys_nodes/__init__.py.
"""
Policy Node - DEPRECATED
Use BehaviorNode instead.  PolicyNode parameters (network architecture,
inference config) belong to the training side, not the runtime canvas.
"""

from typing import Dict, Any
from .base_node import BaseNode


class PolicyNode(BaseNode):
    """
    Policy node — placeholder shell for running a policy episode.

    Accepts a checkpoint handle from CheckpointNode via the `checkpoint` input port.
    The checkpoint dict carries `policy_id` (and the loaded bundle in Circle 3+),
    so PolicyNode itself does NOT need a registry dropdown — it inherits the
    bundle selection from the upstream CheckpointNode.

    Parameters control HOW the episode runs (steps, render), not WHICH bundle.
    No inference occurs in Circle 1.
    """

    def __init__(self, node_id: str):
        super().__init__(node_id, "policy")
        self.inputs = {"checkpoint": None}
        self.outputs = {"episode_result": None}
        self.parameters = {
            "max_steps": 1000,
            "render": False,
        }

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        checkpoint = inputs.get("checkpoint") or {}
        return {
            "episode_result": {
                "status": "placeholder",
                "source": "policy",
                "policy_id": checkpoint.get("policy_id", ""),
                "max_steps": self.get_parameter("max_steps", 1000),
                "render": self.get_parameter("render", False),
                "checkpoint_received": bool(checkpoint),
            }
        }

    def get_display_name(self) -> str:
        return "Policy"

    def get_description(self) -> str:
        return "Run a policy episode against a loaded checkpoint (shell placeholder)"
