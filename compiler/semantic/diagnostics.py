#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified diagnostic system for the UnitPort compiler.
All errors, warnings, and info messages across both
Canvas and Code paths use this structure.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List


class DiagnosticLevel(Enum):
    ERROR = "error"
    WARNING = "warn"
    INFO = "info"


@dataclass
class DiagnosticLocation:
    """Location in code or on the canvas."""
    line: Optional[int] = None
    column: Optional[int] = None
    span: Optional[int] = None
    node_id: Optional[str] = None
    port: Optional[str] = None

    def to_dict(self) -> dict:
        d = {}
        if self.line is not None:
            d["line"] = self.line
        if self.column is not None:
            d["column"] = self.column
        if self.span is not None:
            d["span"] = self.span
        if self.node_id is not None:
            d["node_id"] = self.node_id
        if self.port is not None:
            d["port"] = self.port
        return d


@dataclass
class Diagnostic:
    """A single diagnostic message."""
    code: str
    level: DiagnosticLevel
    message: str
    location: Optional[DiagnosticLocation] = None
    suggestion: str = ""
    autofix: Optional[dict] = None

    def to_dict(self) -> dict:
        d = {
            "code": self.code,
            "level": self.level.value,
            "message": self.message,
        }
        if self.location:
            d["location"] = self.location.to_dict()
        if self.suggestion:
            d["suggestion"] = self.suggestion
        if self.autofix:
            d["autofix"] = self.autofix
        return d

    def __str__(self) -> str:
        loc = ""
        if self.location:
            if self.location.line is not None:
                loc = f" (line {self.location.line})"
            elif self.location.node_id is not None:
                loc = f" (node {self.location.node_id})"
        return f"[{self.code}] {self.level.value.upper()}{loc}: {self.message}"


def make_error(code: str, message: str, node_id: str = None, **kwargs) -> Diagnostic:
    """Helper to create an error diagnostic."""
    loc = DiagnosticLocation(node_id=node_id) if node_id else None
    return Diagnostic(code=code, level=DiagnosticLevel.ERROR,
                      message=message, location=loc, **kwargs)


def make_warning(code: str, message: str, node_id: str = None, **kwargs) -> Diagnostic:
    """Helper to create a warning diagnostic."""
    loc = DiagnosticLocation(node_id=node_id) if node_id else None
    return Diagnostic(code=code, level=DiagnosticLevel.WARNING,
                      message=message, location=loc, **kwargs)


def make_info(code: str, message: str, **kwargs) -> Diagnostic:
    """Helper to create an info diagnostic."""
    return Diagnostic(code=code, level=DiagnosticLevel.INFO,
                      message=message, **kwargs)
