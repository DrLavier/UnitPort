#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base Node
All draggable functional nodes should inherit from this base class
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List


class BaseNode(ABC):
    """Node base class.

    Region taxonomy (P6, mission_design.yaml §3 + principle P8):
        Every node belongs to exactly one of three regions —
            ``initial_setup`` / ``logic_execution`` / ``output_handling``
        — declared via :meth:`get_region`. Region membership is used by
        :class:`src.system.compiler.region_validator.RegionValidator`
        to emit WARN diagnostics when a canvas violates the
        [setup → logic → output] flow taxonomy. Validation is SOFT
        (warnings only) per principle P8 — legacy canvases keep
        loading.
    """

    REGION_INITIAL_SETUP   = "initial_setup"
    REGION_LOGIC_EXECUTION = "logic_execution"
    REGION_OUTPUT_HANDLING = "output_handling"

    def __init__(self, node_id: str, node_type: str):
        """
        Initialize node

        Args:
            node_id: Unique node ID
            node_type: Node type
        """
        self.node_id = node_id
        self.node_type = node_type
        self.inputs = {}
        self.outputs = {}
        self.parameters = {}

    @abstractmethod
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute node logic

        Args:
            inputs: Input data dictionary

        Returns:
            Output data dictionary
        """
        pass

    @abstractmethod
    def get_display_name(self) -> str:
        """Get node display name"""
        pass

    @abstractmethod
    def get_description(self) -> str:
        """Get node description"""
        pass

    def get_input_ports(self) -> List[str]:
        """Get input port list"""
        return list(self.inputs.keys())

    def get_output_ports(self) -> List[str]:
        """Get output port list"""
        return list(self.outputs.keys())

    def set_parameter(self, key: str, value: Any):
        """Set parameter"""
        self.parameters[key] = value

    def get_parameter(self, key: str, default: Any = None) -> Any:
        """Get parameter"""
        return self.parameters.get(key, default)

    def to_code(self) -> str:
        """
        Convert node to code

        Returns:
            Generated code string
        """
        return f"# {self.get_display_name()}: {self.node_id}\n"

    def get_region(self) -> str:
        """Return the canvas region this node belongs to.

        Default is ``REGION_LOGIC_EXECUTION`` because the vast majority
        of nodes (script, comparison, action_execution, behavior, …)
        are mid-canvas logic. Region 1 / Region 3 nodes override this
        explicitly. See :class:`src.system.compiler.region_validator.RegionValidator`
        for the consumer.
        """
        return self.REGION_LOGIC_EXECUTION

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id='{self.node_id}', type='{self.node_type}')"
