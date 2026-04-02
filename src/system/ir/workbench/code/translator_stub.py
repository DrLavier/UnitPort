"""IR workbench translator stub.

This module is intentionally minimal for v0.1 workspace bootstrap.
"""

from pathlib import Path
from typing import Dict, Any


def load_readable_ir(path: str) -> Dict[str, Any]:
    """Load a readable IR file.

    TODO(v0.2): implement YAML parsing and schema validation.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"IR file not found: {file_path}")
    return {"status": "stub", "path": str(file_path)}
