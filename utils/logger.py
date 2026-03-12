#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def _candidate_log_dirs() -> list[Path]:
    project_root = Path(__file__).resolve().parent.parent
    candidates = [project_root / "logs"]

    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        candidates.append(Path(local_appdata) / "UnitPort" / "logs")

    candidates.append(Path.cwd() / "logs")
    return candidates


def _resolve_log_dir() -> Optional[Path]:
    for log_dir in _candidate_log_dirs():
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            probe = log_dir / ".write_test"
            probe.write_text("", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return log_dir
        except OSError:
            continue
    return None


def setup_logger(
    name: str = "Celebrimbor",
    level: int = logging.INFO,
    log_to_file: bool = True,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_to_file:
        log_dir = _resolve_log_dir()
        if log_dir is None:
            logger.warning("File logging disabled: no writable log directory found.")
        else:
            log_file = log_dir / f"celebrimbor_{datetime.now().strftime('%Y%m%d')}.log"
            try:
                file_handler = logging.FileHandler(log_file, encoding="utf-8")
                file_handler.setLevel(level)
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
            except OSError as exc:
                logger.warning("File logging disabled: %s", exc)

    return logger


def get_logger(name: str = "Celebrimbor") -> logging.Logger:
    return logging.getLogger(name)
