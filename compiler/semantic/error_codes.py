#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Error code directory for the UnitPort compiler.

Error codes follow the pattern:
- E1xxx: Syntax / Lexer / Parser errors
- E2xxx: Semantic / Schema errors
- W3xxx: Warnings (non-fatal)
- I4xxx: Informational

Each code has a category, severity, and description template.
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class ErrorCodeEntry:
    """Error code definition."""
    code: str
    category: str
    severity: str  # "error", "warn", "info"
    template: str


# Complete error code registry
ERROR_CODES: Dict[str, ErrorCodeEntry] = {
    # ---------- E1xxx: Syntax / Lexer / Parser ----------
    "E1001": ErrorCodeEntry(
        "E1001", "syntax", "error",
        "Lexer error: {detail}",
    ),
    "E1002": ErrorCodeEntry(
        "E1002", "syntax", "error",
        "Parse error: {detail}",
    ),
    "E1003": ErrorCodeEntry(
        "E1003", "syntax", "warn",
        "Unsupported for-loop syntax: only 'for x in range(...)' is supported",
    ),
    "E1004": ErrorCodeEntry(
        "E1004", "syntax", "error",
        "Unexpected token: {token}",
    ),
    "E1005": ErrorCodeEntry(
        "E1005", "syntax", "error",
        "Indentation error: tabs are not allowed, use spaces",
    ),

    # ---------- E2xxx: Semantic / Schema ----------
    "E2001": ErrorCodeEntry(
        "E2001", "semantic", "warn",
        "No schema found for node type '{node_type}'",
    ),
    "E2002": ErrorCodeEntry(
        "E2002", "semantic", "error",
        "Missing required parameter '{param}' for node '{schema_id}'",
    ),
    "E2003": ErrorCodeEntry(
        "E2003", "semantic", "error",
        "Parameter '{param}' value out of range: {value} (expected {min}-{max})",
    ),
    "E2004": ErrorCodeEntry(
        "E2004", "semantic", "error",
        "Parameter '{param}' has invalid value: '{value}' (allowed: {choices})",
    ),
    "E2005": ErrorCodeEntry(
        "E2005", "semantic", "error",
        "Node '{schema_id}' is not compatible with robot '{robot_type}'",
    ),
    "E2006": ErrorCodeEntry(
        "E2006", "semantic", "error",
        "Dangling edge: target node '{node_id}' not found",
    ),
    "E2007": ErrorCodeEntry(
        "E2007", "semantic", "error",
        "Dangling edge: source node '{node_id}' not found",
    ),
    "E2008": ErrorCodeEntry(
        "E2008", "semantic", "error",
        "Type mismatch on parameter '{param}': expected {expected}, got {actual}",
    ),

    # ---------- W3xxx: Warnings ----------
    "W3001": ErrorCodeEntry(
        "W3001", "lowering", "warn",
        "Skipping edge with unmapped node ID: {from_id} -> {to_id}",
    ),
    "W3002": ErrorCodeEntry(
        "W3002", "lowering", "warn",
        "Unknown function call wrapped as opaque block: {func_name}",
    ),
    "W3003": ErrorCodeEntry(
        "W3003", "lowering", "warn",
        "Unknown node kind for canvas conversion: {kind}",
    ),
    "W3004": ErrorCodeEntry(
        "W3004", "lowering", "warn",
        "Opaque code block cannot be fully reconstructed on canvas",
    ),
    "W3005": ErrorCodeEntry(
        "W3005", "codegen", "warn",
        "Unknown node type in code generation: {schema_id}",
    ),

    # ---------- I4xxx: Informational ----------
    "I4001": ErrorCodeEntry(
        "I4001", "codegen", "info",
        "Code generated: {node_count} nodes, {edge_count} edges",
    ),
    "I4002": ErrorCodeEntry(
        "I4002", "lowering", "info",
        "AST lowered: {node_count} nodes, {edge_count} edges",
    ),
    "I4003": ErrorCodeEntry(
        "I4003", "lowering", "info",
        "IR to canvas: {node_count} nodes, {connection_count} connections",
    ),
    "I4004": ErrorCodeEntry(
        "I4004", "parser", "info",
        "Function definition captured: {func_name}",
    ),
}


def get_error_code(code: str) -> Optional[ErrorCodeEntry]:
    """Look up an error code entry."""
    return ERROR_CODES.get(code)


def get_all_codes() -> Dict[str, ErrorCodeEntry]:
    """Return all error codes."""
    return ERROR_CODES.copy()
