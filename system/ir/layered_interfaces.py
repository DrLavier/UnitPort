"""Layered IR interfaces and stub implementations.

These interfaces define extension points for backend/frontend codex agents.
The methods are intentionally minimal and can return passthrough data at v0.1.
"""

from __future__ import annotations

from typing import Any, Dict, Protocol

from system.ir.layered_contracts import LayeredIRBundle


class LayeredIRSerializer(Protocol):
    """Serializer contract for layered IR bundles."""

    def to_dict(self, bundle: LayeredIRBundle) -> Dict[str, Any]:
        """Serialize bundle to dict."""
        ...

    def from_dict(self, payload: Dict[str, Any]) -> LayeredIRBundle:
        """Deserialize dict to bundle."""
        ...


class LayeredIRBridge(Protocol):
    """Bridge contract between WorkflowIR and layered bundle."""

    def from_workflow_ir(self, workflow_ir: Any) -> LayeredIRBundle:
        """Build layered bundle from existing WorkflowIR."""
        ...

    def apply_to_workflow_ir(self, workflow_ir: Any, bundle: LayeredIRBundle) -> Any:
        """Attach layered contract fields to existing WorkflowIR object."""
        ...


class DefaultLayeredIRSerializer:
    """Minimal working serializer for LayeredIRBundle round-trips.

    Subgraphs (including PackageMetadata) are serialized via their own
    to_dict()/from_dict() methods so the dict form is JSON-safe and
    backward-compatible.  Other contract fields use dataclasses.asdict()
    as they contain only primitive types.
    """

    def to_dict(self, bundle: LayeredIRBundle) -> Dict[str, Any]:
        from dataclasses import asdict as _asdict
        return {
            "mission":       _asdict(bundle.mission),
            "behavior":      _asdict(bundle.behavior),
            "commands":      [_asdict(c) for c in bundle.commands],
            "safety":        _asdict(bundle.safety),
            "observability": _asdict(bundle.observability),
            "execution":     _asdict(bundle.execution),
            # SubgraphIR has nested PackageMetadata — use its own to_dict().
            "subgraphs":     [sg.to_dict() for sg in bundle.subgraphs],
        }

    def from_dict(self, payload: Dict[str, Any]) -> LayeredIRBundle:
        """Reconstruct a LayeredIRBundle from a persisted dict.

        Backward-compatible: missing or malformed payload → empty bundle.
        Subgraphs (+ their PackageMetadata) are fully reconstructed;
        other contract fields are left at safe defaults.
        """
        if not isinstance(payload, dict):
            return LayeredIRBundle()
        from system.ir.layered_contracts import SubgraphIR
        subgraphs = [
            SubgraphIR.from_dict(sg)
            for sg in (payload.get("subgraphs") or [])
            if isinstance(sg, dict)
        ]
        bundle = LayeredIRBundle()
        bundle.subgraphs = subgraphs
        return bundle


class DefaultLayeredIRBridge:
    """Minimal working bridge between WorkflowIR and LayeredIRBundle.

    Circle 3: both methods now perform real (de)serialization so
    package metadata carried in SubgraphIR survives the
    save → load → compile round-trip inside WorkflowIR.layered_contracts.
    """

    _serializer = DefaultLayeredIRSerializer()

    def from_workflow_ir(self, workflow_ir: Any) -> LayeredIRBundle:
        """Reconstruct a LayeredIRBundle from an existing WorkflowIR.

        Reads ``workflow_ir.layered_contracts`` (dict) when present and
        delegates to the serializer for typed reconstruction.  Returns an
        empty bundle for old WorkflowIR objects that pre-date Circle 3.
        """
        contracts: Any = None
        if isinstance(workflow_ir, dict):
            contracts = workflow_ir.get("layered_contracts")
        elif hasattr(workflow_ir, "layered_contracts"):
            contracts = workflow_ir.layered_contracts
        if isinstance(contracts, dict):
            return self._serializer.from_dict(contracts)
        return LayeredIRBundle()

    def apply_to_workflow_ir(self, workflow_ir: Any, bundle: LayeredIRBundle) -> Any:
        """Attach a serialized LayeredIRBundle dict to a WorkflowIR object.

        Stores the JSON-safe dict form (not a raw dataclass) so the field
        survives WorkflowIR.to_dict()/from_dict() round-trips.  Legacy
        fields already on the WorkflowIR are left untouched.
        """
        serialized = self._serializer.to_dict(bundle)
        if isinstance(workflow_ir, dict):
            # Dict or dict subclass: store as a key.
            workflow_ir["layered_contracts"] = serialized
        elif hasattr(workflow_ir, "layered_contracts"):
            # WorkflowIR dataclass field: assign directly (Optional[Dict]).
            workflow_ir.layered_contracts = serialized
        elif hasattr(workflow_ir, "__dict__"):
            workflow_ir.__dict__["layered_contracts"] = serialized
        return workflow_ir

