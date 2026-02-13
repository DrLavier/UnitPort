#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
路径处理工具
"""

from pathlib import Path
from typing import List, Optional


def get_project_root() -> Path:
    """
    获取项目根目录
    
    Returns:
        项目根目录路径
    """
    # 从当前文件位置向上查找项目根目录
    # utils/path_helper.py -> utils -> celebrimbor
    return Path(__file__).parent.parent.absolute()


def find_file_in_paths(filename: str, search_paths: List[Path]) -> Optional[Path]:
    """
    在多个路径中查找文件
    
    Args:
        filename: 文件名
        search_paths: 搜索路径列表
    
    Returns:
        找到的文件路径，如果未找到则返回None
    """
    for path in search_paths:
        file_path = path / filename
        if file_path.exists():
            return file_path
    
    return None


def ensure_dir(path: Path) -> Path:
    """
    确保目录存在，如果不存在则创建
    
    Args:
        path: 目录路径
    
    Returns:
        目录路径
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_files_recursive(directory: Path, pattern: str = "*") -> List[Path]:
    """
    递归列出目录下所有匹配的文件
    
    Args:
        directory: 目录路径
        pattern: 文件匹配模式（如 "*.xml"）
    
    Returns:
        文件路径列表
    """
    if not directory.exists():
        return []
    
    return list(directory.rglob(pattern))


def get_relative_path(target: Path, base: Path) -> Path:
    """
    获取相对路径
    
    Args:
        target: 目标路径
        base: 基准路径
    
    Returns:
        相对路径
    """
    try:
        return target.relative_to(base)
    except ValueError:
        # 如果无法计算相对路径，返回绝对路径
        return target
