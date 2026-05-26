# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Scan src/ for i18n call sites, replay through I18n.tr() to register keys,
then dump localisation/{LANG}/{category}.txt for each target language via
I18n.export_template (merge mode: existing translations preserved, new keys
seeded with the English default from source)."""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from unitport_sdk import I18n, log_info, log_success, log_warning  # noqa: E402

KEY_DEFAULT_CALLS = {"tr", "I18nLabel", "I18nButton"}
TARGET_LANGS = ("EN", "FR", "ZH")


def _str_arg(node):
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _extract(call: ast.Call):
    func = call.func
    if isinstance(func, ast.Name):
        name = func.id
    elif isinstance(func, ast.Attribute):
        name = func.attr
    else:
        return None

    if name in KEY_DEFAULT_CALLS:
        key_idx, def_idx = 0, 1
    elif name == "i18n_bind":
        key_idx, def_idx = 2, 3
    else:
        return None

    if len(call.args) <= key_idx:
        return None
    key = _str_arg(call.args[key_idx])
    if key is None:
        return None
    default = _str_arg(call.args[def_idx]) if len(call.args) > def_idx else ""
    if not default:
        for kw in call.keywords:
            if kw.arg == "default":
                default = _str_arg(kw.value) or ""
                break
    return key, default or ""


def _scan_file(path: Path):
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (UnicodeDecodeError, OSError, SyntaxError) as exc:
        log_warning(f"[i18n-export] skip {path.relative_to(PROJECT_ROOT)}: {exc}")
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            pair = _extract(node)
            if pair is not None:
                yield pair


def main() -> int:
    log_info(f"[i18n-export] scanning {SRC_DIR}")
    collected: dict[str, str] = {}
    for f in SRC_DIR.rglob("*.py"):
        if "__pycache__" in f.parts or "unitport_sdk" in f.parts:
            continue
        for key, default in _scan_file(f):
            if key not in collected or (not collected[key] and default):
                collected[key] = default

    langs = tuple(sys.argv[1:]) or TARGET_LANGS
    log_info(f"[i18n-export] found {len(collected)} unique keys; targets={list(langs)}")
    for key, default in collected.items():
        I18n.tr(key, default)

    for lang in langs:
        out_dir = I18n.export_template(lang, merge=True)
        log_success(f"[i18n-export] wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
