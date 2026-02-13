#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置文件管理器
负责加载和管理 system.ini 和 user.ini
"""

import configparser
from pathlib import Path
from typing import Optional, Any


class ConfigManager:
    """配置管理器"""
    
    def __init__(self):
        """初始化配置管理器"""
        # 获取项目根目录
        self.project_root = Path(__file__).parent.parent.parent.absolute()
        
        # 配置文件路径
        self.config_dir = self.project_root / "config"
        self.system_config_path = self.config_dir / "system.ini"
        self.user_config_path = self.config_dir / "user.ini"
        
        # 创建配置解析器
        self.system_config = configparser.ConfigParser()
        self.user_config = configparser.ConfigParser()
        
        # 加载配置文件
        self._load_configs()
        
        # 更新项目根路径
        self._update_project_root()
    
    def _load_configs(self):
        """加载配置文件"""
        # 确保配置目录存在
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载系统配置
        if self.system_config_path.exists():
            self.system_config.read(self.system_config_path, encoding='utf-8')
        else:
            self._create_default_system_config()
        
        # 加载用户配置
        if self.user_config_path.exists():
            self.user_config.read(self.user_config_path, encoding='utf-8')
        else:
            self._create_default_user_config()
    
    def _update_project_root(self):
        """更新配置文件中的项目根路径"""
        if not self.system_config.has_section('PATH'):
            self.system_config.add_section('PATH')
        
        self.system_config.set('PATH', 'project_root', str(self.project_root))
        self.save_system_config()
    
    def _create_default_system_config(self):
        """创建默认系统配置"""
        # 如果配置文件不存在，使用代码中的默认值
        self.system_config['PATH'] = {
            'project_root': str(self.project_root),
            'models_root': './models',
            'unitree_sdk': './models/unitree/unitree_sdk2_python',
            'unitree_mujoco': './models/unitree/unitree_mujoco',
            'unitree_robots': './models/unitree/unitree_mujoco/unitree_robots',
            'log_dir': './logs'
        }
        
        self.system_config['SIMULATION'] = {
            'default_robot': 'go2',
            'available_robots': 'go2,a1,b1',
            'default_action': 'stand'
        }
        
        self.system_config['MUJOCO'] = {
            'gl_backend': 'glfw',
            'timestep': '0.002',
            'keep_window_time': '5.0'
        }
        
        self.system_config['UI'] = {
            'window_width': '1400',
            'window_height': '900',
            'graph_editor_width': '820',
            'code_editor_width': '460',
            'module_palette_width': '320'
        }
        
        self.system_config['NETWORK'] = {
            'websocket_port': '8765',
            'enable_remote': 'false'
        }
        
        self.system_config['DEBUG'] = {
            'debug_mode': 'false',
            'verbose_logging': 'true',
            'print_directory_structure': 'false'
        }
        
        self.save_system_config()
    
    def _create_default_user_config(self):
        """创建默认用户配置"""
        self.user_config['PREFERENCES'] = {
            'theme': 'dark',
            'editor_font_size': '10',
            'autosave_interval': '5',
            'show_grid': 'true',
            'enable_syntax_highlight': 'true'
        }
        
        self.user_config['RECENT'] = {
            'recent_projects': '',
            'last_robot_type': 'go2',
            'last_action': 'stand'
        }
        
        self.user_config['CUSTOM'] = {}
        
        self.save_user_config()
    
    def get(self, section: str, option: str, fallback: Any = None, 
            config_type: str = 'system') -> Any:
        """
        获取配置值
        
        Args:
            section: 配置节名称
            option: 配置项名称
            fallback: 默认值
            config_type: 配置类型 ('system' 或 'user')
        
        Returns:
            配置值
        """
        config = self.system_config if config_type == 'system' else self.user_config
        
        try:
            return config.get(section, option, fallback=fallback)
        except (configparser.NoSectionError, configparser.NoOptionError):
            return fallback
    
    def get_int(self, section: str, option: str, fallback: int = 0,
                config_type: str = 'system') -> int:
        """获取整数配置值"""
        config = self.system_config if config_type == 'system' else self.user_config
        try:
            return config.getint(section, option, fallback=fallback)
        except (configparser.NoSectionError, configparser.NoOptionError):
            return fallback
    
    def get_float(self, section: str, option: str, fallback: float = 0.0,
                  config_type: str = 'system') -> float:
        """获取浮点数配置值"""
        config = self.system_config if config_type == 'system' else self.user_config
        try:
            return config.getfloat(section, option, fallback=fallback)
        except (configparser.NoSectionError, configparser.NoOptionError):
            return fallback
    
    def get_bool(self, section: str, option: str, fallback: bool = False,
                 config_type: str = 'system') -> bool:
        """获取布尔配置值"""
        config = self.system_config if config_type == 'system' else self.user_config
        try:
            return config.getboolean(section, option, fallback=fallback)
        except (configparser.NoSectionError, configparser.NoOptionError):
            return fallback
    
    def set(self, section: str, option: str, value: Any, 
            config_type: str = 'system'):
        """
        设置配置值
        
        Args:
            section: 配置节名称
            option: 配置项名称
            value: 配置值
            config_type: 配置类型 ('system' 或 'user')
        """
        config = self.system_config if config_type == 'system' else self.user_config
        
        if not config.has_section(section):
            config.add_section(section)
        
        config.set(section, option, str(value))
    
    def save_system_config(self):
        """保存系统配置"""
        with open(self.system_config_path, 'w', encoding='utf-8') as f:
            self.system_config.write(f)
    
    def save_user_config(self):
        """保存用户配置"""
        with open(self.user_config_path, 'w', encoding='utf-8') as f:
            self.user_config.write(f)
    
    def get_path(self, path_key: str) -> Path:
        """
        获取路径配置（自动转换为绝对路径）
        
        Args:
            path_key: PATH节中的配置项名称
        
        Returns:
            Path对象
        """
        path_str = self.get('PATH', path_key, fallback='')
        
        if not path_str:
            return self.project_root
        
        path = Path(path_str)
        
        # 如果是相对路径，转换为绝对路径
        if not path.is_absolute():
            path = self.project_root / path
        
        return path
    
    def get_available_robots(self) -> list:
        """获取可用机器人列表"""
        robots_str = self.get('SIMULATION', 'available_robots', fallback='go2')
        return [r.strip() for r in robots_str.split(',')]
