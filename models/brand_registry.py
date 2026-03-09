#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Brand Registry — auto-discovers brand/model mappings from models/ subdirectories.

Each brand package (e.g. models/Unitree/__init__.py) should export:
    SUPPORTED_MODELS: list[str]   e.g. ["go2", "a1", "b1", "b2", "h1"]

Discovery reads SUPPORTED_MODELS via AST parsing (no module import) to avoid
circular-import and case-sensitivity issues on Windows.
"""

import ast
import os
from typing import Dict, List


class _BrandInfo:
    __slots__ = ("display_name", "models", "has_adapter")

    def __init__(self, display_name: str, models: List[str], has_adapter: bool):
        self.display_name = display_name
        self.models = models
        self.has_adapter = has_adapter


class BrandRegistry:
    """Singleton registry of brand -> model mappings.  Lazy-discovers on first access."""

    _instance: "BrandRegistry" = None
    _brands: Dict[str, _BrandInfo]
    _discovered: bool

    def __new__(cls):
        if cls._instance is None:
            inst = super().__new__(cls)
            inst._brands = {}
            inst._discovered = False
            cls._instance = inst
        return cls._instance

    # -- discovery ----------------------------------------------------------

    def _ensure_discovered(self):
        if not self._discovered:
            self.discover()

    def discover(self):
        """Scan models/ subdirectories and populate the registry."""
        models_dir = os.path.dirname(os.path.abspath(__file__))
        self._brands.clear()

        for entry in sorted(os.listdir(models_dir)):
            subdir = os.path.join(models_dir, entry)
            if not os.path.isdir(subdir):
                continue
            if entry.startswith("_") or entry.startswith("."):
                continue

            init_path = os.path.join(subdir, "__init__.py")
            has_init = os.path.isfile(init_path)

            supported_models: List[str] = []
            has_adapter = False

            if has_init:
                supported_models = self._parse_supported_models(init_path)
                has_adapter = bool(supported_models)

            if not supported_models:
                supported_models = ["(no models)"]
                has_adapter = False

            key = entry.lower()
            self._brands[key] = _BrandInfo(
                display_name=entry,
                models=supported_models,
                has_adapter=has_adapter,
            )

        self._discovered = True

    @staticmethod
    def _parse_supported_models(init_path: str) -> List[str]:
        """Extract SUPPORTED_MODELS list from __init__.py via AST (no import)."""
        try:
            with open(init_path, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=init_path)
        except Exception:
            return []

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "SUPPORTED_MODELS":
                        return _eval_list_literal(node.value)
        return []

    # -- public API ---------------------------------------------------------

    def get_brands(self) -> List[str]:
        """Return brand display names (title-case as on disk)."""
        self._ensure_discovered()
        return [info.display_name for info in self._brands.values()]

    def get_models(self, brand: str) -> List[str]:
        """Return model list for a brand (case-insensitive lookup)."""
        self._ensure_discovered()
        info = self._brands.get(brand.lower())
        return list(info.models) if info else []

    def get_brand_model_map(self) -> Dict[str, List[str]]:
        """Return {display_name: [models]} for all brands."""
        self._ensure_discovered()
        return {info.display_name: list(info.models) for info in self._brands.values()}

    def get_robot_brand_map(self) -> Dict[str, str]:
        """Return {model: brand_key} for all brands/models (flat reverse map)."""
        self._ensure_discovered()
        result: Dict[str, str] = {}
        for key, info in self._brands.items():
            for model in info.models:
                if model != "(no models)":
                    result[model] = key
        return result


def _eval_list_literal(node) -> List[str]:
    """Safely evaluate an AST node that should be a list of string literals."""
    if isinstance(node, ast.List):
        result = []
        for elt in node.elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                result.append(elt.value)
        return result
    return []
