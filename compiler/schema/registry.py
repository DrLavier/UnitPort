#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Schema Registry - singleton registry for all node schemas.
"""

import json
import importlib.util
from pathlib import Path
from typing import Dict, Optional, List

from compiler.schema.node_schema import NodeSchema


class SchemaRegistry:
    """Singleton registry for node schemas."""

    _schemas: Dict[str, NodeSchema] = {}
    _loaded: bool = False

    @classmethod
    def register(cls, schema: NodeSchema):
        """Register a node schema."""
        cls._schemas[schema.schema_id] = schema

    @classmethod
    def get(cls, schema_id: str) -> Optional[NodeSchema]:
        """Get a schema by its schema_id."""
        cls._ensure_loaded()
        return cls._schemas.get(schema_id)

    @classmethod
    def get_by_node_type(cls, node_type: str) -> Optional[NodeSchema]:
        """Get the first schema matching a node_type."""
        cls._ensure_loaded()
        for s in cls._schemas.values():
            if s.node_type == node_type:
                return s
        return None

    @classmethod
    def get_by_display_name(cls, display_name: str) -> Optional[NodeSchema]:
        """Get the first schema matching a display_name."""
        cls._ensure_loaded()
        for s in cls._schemas.values():
            if s.display_name == display_name:
                return s
        return None

    @classmethod
    def all_schemas(cls) -> Dict[str, NodeSchema]:
        """Get all registered schemas."""
        cls._ensure_loaded()
        return cls._schemas.copy()

    @classmethod
    def list_schema_ids(cls) -> List[str]:
        """List all registered schema IDs."""
        cls._ensure_loaded()
        return list(cls._schemas.keys())

    @classmethod
    def load_builtins(cls):
        """Load all builtin JSON schemas from compiler/schema/builtin/."""
        builtin_dir = Path(__file__).parent / "builtin"
        if not builtin_dir.exists():
            return

        for json_file in sorted(builtin_dir.glob("*.json")):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                schema = NodeSchema.from_dict(data)
                cls._schemas[schema.schema_id] = schema
            except Exception as e:
                print(f"Warning: Failed to load schema {json_file}: {e}")

        cls._sync_dynamic_action_execution_schema()
        cls._loaded = True

    @classmethod
    def _sync_dynamic_action_execution_schema(cls):
        """Synchronize action_execution parameter choices from the IR registry."""
        schema = cls._schemas.get("builtin.action_execution")
        if schema is None:
            return

        action_param = schema.get_parameter("action")
        if action_param is None:
            return

        registered_raw_actions = cls._get_registered_action_raw_actions()
        if not registered_raw_actions:
            return

        if action_param.constraints is None:
            from compiler.schema.node_schema import ParamConstraint
            action_param.constraints = ParamConstraint()

        action_param.constraints.choices = list(registered_raw_actions)
        if action_param.default not in action_param.constraints.choices:
            action_param.default = action_param.constraints.choices[0]

    @classmethod
    def _get_registered_action_raw_actions(cls) -> List[str]:
        """Load IR action raw_action values without importing the full system package."""
        module_path = (
            Path(__file__).resolve().parents[2]
            / "system"
            / "behavior"
            / "ir_action_registry.py"
        )
        if not module_path.exists():
            return []

        try:
            spec = importlib.util.spec_from_file_location(
                "_unitport_ir_action_registry",
                str(module_path),
            )
            if spec is None or spec.loader is None:
                return []
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            list_ir_actions = getattr(module, "list_ir_actions", None)
            if not callable(list_ir_actions):
                return []
            return [
                str(desc.raw_action or "").strip()
                for desc in list_ir_actions()
                if str(desc.raw_action or "").strip()
            ]
        except Exception:
            return []

    @classmethod
    def _ensure_loaded(cls):
        """Ensure builtins are loaded."""
        if not cls._loaded:
            cls.load_builtins()

    @classmethod
    def reset(cls):
        """Reset registry (for testing)."""
        cls._schemas.clear()
        cls._loaded = False
