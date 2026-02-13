#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Type system for the UnitPort compiler IR.
"""

from enum import Enum


class IRType(Enum):
    """Data types used in IR ports and parameters."""
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    STRING = "string"
    ANY = "any"
    VOID = "void"

    @classmethod
    def from_string(cls, s: str) -> 'IRType':
        """Parse a type string to IRType, defaulting to ANY."""
        try:
            return cls(s.lower())
        except ValueError:
            return cls.ANY


class PortDirection(Enum):
    """Port direction for node schemas."""
    INPUT = "input"
    OUTPUT = "output"
