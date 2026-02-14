"""Shared error codes re-export.

The canonical error-code registry lives in ``compiler.semantic.error_codes``.
This module re-exports it so that the new layered packages can
import from ``shared.types`` without duplicating definitions.
"""

from compiler.semantic.error_codes import (  # noqa: F401
    ERROR_CODES,
    ErrorCodeEntry,
    get_all_codes,
    get_error_code,
)
