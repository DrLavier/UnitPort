#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Schema Registry - singleton registry for all node schemas.
"""

import json
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

        cls._loaded = True

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
