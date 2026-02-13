#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据管理器
支持线程安全的INI和JSON文件读写
"""

import os
import json
import threading
import configparser
from pathlib import Path
from typing import Optional, Any


class DataManager:
    """
    全局数据管理器单例
    确保工程不同位置读取时基于同一版本，避免冲突和覆盖
    线程安全
    """
    
    _instance: Optional['DataManager'] = None
    _initialized: bool = False
    _lock = threading.RLock()  # 可重入锁
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance
    
    def __init__(self):
        if DataManager._initialized:
            return
        
        with DataManager._lock:
            if DataManager._initialized:
                return
            
            self._ini_cache: dict[str, configparser.ConfigParser] = {}
            self._json_cache: dict[str, dict] = {}
            self._file_locks: dict[str, threading.RLock] = {}
            
            DataManager._initialized = True
    
    def _get_file_lock(self, file_path: str) -> threading.RLock:
        """获取文件专用锁"""
        abs_path = str(Path(file_path).resolve())
        with DataManager._lock:
            if abs_path not in self._file_locks:
                self._file_locks[abs_path] = threading.RLock()
            return self._file_locks[abs_path]
    
    # ========================================================================
    # INI 文件操作
    # ========================================================================
    
    def load_ini(self, file_path: str, force_reload: bool = False) -> configparser.ConfigParser:
        """初始读取 INI 文件到缓存（线程安全）"""
        abs_path = str(Path(file_path).resolve())
        file_lock = self._get_file_lock(abs_path)
        
        with file_lock:
            if abs_path in self._ini_cache and not force_reload:
                return self._ini_cache[abs_path]
            
            config = configparser.ConfigParser()
            
            if os.path.exists(abs_path):
                config.read(abs_path, encoding='utf-8')
            
            self._ini_cache[abs_path] = config
            return config
    
    def read_ini(self, file_path: str) -> configparser.ConfigParser:
        """读取 INI 缓存（线程安全）"""
        abs_path = str(Path(file_path).resolve())
        file_lock = self._get_file_lock(abs_path)
        
        with file_lock:
            if abs_path not in self._ini_cache:
                return self.load_ini(file_path)
            return self._ini_cache[abs_path]
    
    def up_ini(self, file_path: str, section: str = None, key: str = None, value: Any = None, 
               data: dict = None) -> bool:
        """更新 INI 文件并重载缓存（线程安全）"""
        abs_path = str(Path(file_path).resolve())
        file_lock = self._get_file_lock(abs_path)
        
        with file_lock:
            try:
                config = self.read_ini(file_path)
                
                if section and key is not None:
                    if section not in config:
                        config.add_section(section)
                    config.set(section, key, str(value))
                
                if data:
                    for sec, items in data.items():
                        if sec not in config:
                            config.add_section(sec)
                        for k, v in items.items():
                            config.set(sec, k, str(v))
                
                os.makedirs(os.path.dirname(abs_path), exist_ok=True)
                
                with open(abs_path, 'w', encoding='utf-8') as f:
                    config.write(f)
                
                self._ini_cache[abs_path] = config
                return True
                
            except Exception as e:
                print(f"[DataManager] Error updating INI: {e}")
                return False
    
    def get_ini_value(self, file_path: str, section: str, key: str, 
                      fallback: Any = None, value_type: type = str) -> Any:
        """获取 INI 值（线程安全）"""
        abs_path = str(Path(file_path).resolve())
        file_lock = self._get_file_lock(abs_path)
        
        with file_lock:
            config = self.read_ini(file_path)
            
            try:
                if value_type == int:
                    return config.getint(section, key, fallback=fallback)
                elif value_type == float:
                    return config.getfloat(section, key, fallback=fallback)
                elif value_type == bool:
                    return config.getboolean(section, key, fallback=fallback)
                else:
                    return config.get(section, key, fallback=fallback)
            except (configparser.NoSectionError, configparser.NoOptionError):
                return fallback
    
    # ========================================================================
    # JSON 文件操作
    # ========================================================================
    
    def load_json(self, file_path: str, force_reload: bool = False) -> dict:
        """初始读取 JSON 文件到缓存（线程安全）"""
        abs_path = str(Path(file_path).resolve())
        file_lock = self._get_file_lock(abs_path)
        
        with file_lock:
            if abs_path in self._json_cache and not force_reload:
                return self._json_cache[abs_path]
            
            data = {}
            
            if os.path.exists(abs_path):
                try:
                    with open(abs_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                except json.JSONDecodeError as e:
                    print(f"[DataManager] JSON decode error: {e}")
            
            self._json_cache[abs_path] = data
            return data
    
    def read_json(self, file_path: str) -> dict:
        """读取 JSON 缓存（线程安全）"""
        abs_path = str(Path(file_path).resolve())
        file_lock = self._get_file_lock(abs_path)
        
        with file_lock:
            if abs_path not in self._json_cache:
                return self.load_json(file_path)
            return self._json_cache[abs_path]
    
    def up_json(self, file_path: str, data: dict = None, 
                key: str = None, value: Any = None, merge: bool = True) -> bool:
        """更新 JSON 文件并重载缓存（线程安全）"""
        abs_path = str(Path(file_path).resolve())
        file_lock = self._get_file_lock(abs_path)
        
        with file_lock:
            try:
                current_data = self.read_json(file_path) if merge else {}
                
                if key is not None:
                    current_data[key] = value
                
                if data:
                    if merge:
                        current_data.update(data)
                    else:
                        current_data = data
                
                os.makedirs(os.path.dirname(abs_path), exist_ok=True)
                
                with open(abs_path, 'w', encoding='utf-8') as f:
                    json.dump(current_data, f, ensure_ascii=False, indent=2)
                
                self._json_cache[abs_path] = current_data
                return True
                
            except Exception as e:
                print(f"[DataManager] Error updating JSON: {e}")
                return False
    
    # ========================================================================
    # 缓存管理
    # ========================================================================
    
    def clear_cache(self, file_path: str = None):
        """清除缓存"""
        with DataManager._lock:
            if file_path:
                abs_path = str(Path(file_path).resolve())
                self._ini_cache.pop(abs_path, None)
                self._json_cache.pop(abs_path, None)
            else:
                self._ini_cache.clear()
                self._json_cache.clear()
    
    def reload_all(self):
        """重新加载所有已缓存的文件"""
        with DataManager._lock:
            ini_paths = list(self._ini_cache.keys())
            json_paths = list(self._json_cache.keys())
            
            for path in ini_paths:
                self.load_ini(path, force_reload=True)
            
            for path in json_paths:
                self.load_json(path, force_reload=True)


# ============================================================================
# 全局实例和便捷函数
# ============================================================================

_data_manager: Optional[DataManager] = None


def get_data_manager() -> DataManager:
    """获取全局数据管理器实例"""
    global _data_manager
    if _data_manager is None:
        _data_manager = DataManager()
    return _data_manager


def load_data(file_path: str):
    """加载数据文件到缓存"""
    dm = get_data_manager()
    if file_path.endswith('.ini'):
        dm.load_ini(file_path)
    elif file_path.endswith('.json'):
        dm.load_json(file_path)


def read_data(file_path: str):
    """读取数据文件"""
    dm = get_data_manager()
    if file_path.endswith('.ini'):
        return dm.read_ini(file_path)
    elif file_path.endswith('.json'):
        return dm.read_json(file_path)


def up_data(file_path: str, **kwargs) -> bool:
    """更新数据文件"""
    dm = get_data_manager()
    if file_path.endswith('.ini'):
        return dm.up_ini(file_path, **kwargs)
    elif file_path.endswith('.json'):
        return dm.up_json(file_path, **kwargs)
    return False


def get_value(file_path: str, section: str, key: str, fallback: Any = None, value_type: type = str) -> Any:
    """获取配置值"""
    dm = get_data_manager()
    return dm.get_ini_value(file_path, section, key, fallback, value_type)
