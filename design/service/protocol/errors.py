"""Service-layer error types."""

from __future__ import annotations


class ServiceError(Exception):
    """Base exception for service adapter errors."""

    def __init__(self, message: str = "", code: str = "", adapter: str = ""):
        self.code = code
        self.adapter = adapter
        super().__init__(message)
