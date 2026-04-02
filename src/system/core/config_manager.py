#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
?
?system.ini ?user.ini
"""

import configparser
from pathlib import Path
from typing import Optional, Any

from src.system.models.sdk_manager import SdkManager


class ConfigManager:
    """."""
    
    def __init__(self):
        """"""
        # ?
        self.project_root = Path(__file__).parent.parent.parent.absolute()
        
        # 
        self.config_dir = self.project_root / "config"
        self.system_config_path = self.config_dir / "system.ini"
        self.user_config_path = self.config_dir / "user.ini"
        
        # ?
        self.system_config = configparser.ConfigParser()
        self.user_config = configparser.ConfigParser()
        
        # 
        self._load_configs()
        self._ensure_default_paths()
        
        # ?
        self._update_project_root()

    def _default_path_values(self) -> dict:
        """Return default PATH entries relative to project root."""
        sdk_manager = SdkManager()
        unitree_sdk = sdk_manager.resolve_path(
            'unitree_sdk',
            self.project_root / 'models' / 'Unitree' / 'unitree_sdk2_python',
        )
        unitree_mujoco = sdk_manager.resolve_path(
            'unitree_mujoco',
            self.project_root / 'models' / 'Unitree' / 'unitree_mujoco',
        )
        unitree_robots = sdk_manager.resolve_path(
            'unitree_robots',
            self.project_root / 'models' / 'Unitree' / 'unitree_mujoco' / 'unitree_robots',
        )
        return {
            'project_root': str(self.project_root),
            'models_root': './models',
            'unitree_sdk': self._to_config_path(unitree_sdk),
            'unitree_mujoco': self._to_config_path(unitree_mujoco),
            'unitree_robots': self._to_config_path(unitree_robots),
            'log_dir': './logs',
        }

    def _ensure_default_paths(self):
        """Ensure PATH section has required keys so get_path never falls back to project root silently."""
        defaults = self._default_path_values()
        changed = False
        if not self.system_config.has_section('PATH'):
            self.system_config.add_section('PATH')
            changed = True
        for key, value in defaults.items():
            if not self.system_config.get('PATH', key, fallback='').strip():
                self.system_config.set('PATH', key, value)
                changed = True
        if changed:
            self.save_system_config()
    
    def _load_configs(self):
        """"""
        # 
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # 
        if self.system_config_path.exists():
            self.system_config.read(self.system_config_path, encoding='utf-8')
        else:
            self._create_default_system_config()
        
        # 
        if self.user_config_path.exists():
            self.user_config.read(self.user_config_path, encoding='utf-8')
        else:
            self._create_default_user_config()
    
    def _update_project_root(self):
        """."""
        if not self.system_config.has_section('PATH'):
            self.system_config.add_section('PATH')
        
        self.system_config.set('PATH', 'project_root', str(self.project_root))
        self.save_system_config()
    
    def _create_default_system_config(self):
        """"""
        # ?
        self.system_config['PATH'] = self._default_path_values()
        
        self.system_config['SIMULATION'] = {
            'default_robot': 'go2',
            'available_robots': 'go2,go2w,a1,b2,g1,h1,h1_2',
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
            'module_palette_width': '320',
            'icon_acc': 'bin/assets/icon/icon_acc.svg',
            'icon_acc_dark': 'bin/assets/icon/icon_acc_w.svg',
            'icon_prj': 'bin/assets/icon/icon_prj.svg',
            'icon_prj_dark': 'bin/assets/icon/icon_prj_w.svg',
            'icon_nod': 'bin/assets/icon/icon_nod.svg',
            'icon_nod_dark': 'bin/assets/icon/icon_nod_w.svg',
            'icon_theme_toggle': 'bin/assets/icon/icon_L&D_w.svg',
            'icon_theme_toggle_dark': 'bin/assets/icon/icon_L&D.svg',
            'icon_play': 'bin/assets/icon/icon_play.svg',
            'icon_pause': 'bin/assets/icon/icon_pause.svg',
            'icon_stop': 'bin/assets/icon/icon_stop.svg',
            'icon_reset': 'bin/assets/icon/icon_reset.svg',
            'icon_setting': 'bin/assets/icon/icon_setting.svg',
            'icon_move': 'bin/assets/icon/icon_move.svg'
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
        """"""
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
        ?
        
        Args:
            section: ?
            option: ?
            fallback: ?
            config_type:  ('system' ?'user')
        
        Returns:
            ?
        """
        config = self.system_config if config_type == 'system' else self.user_config
        
        try:
            return config.get(section, option, fallback=fallback)
        except (configparser.NoSectionError, configparser.NoOptionError):
            return fallback
    
    def get_int(self, section: str, option: str, fallback: int = 0,
                config_type: str = 'system') -> int:
        """."""
        config = self.system_config if config_type == 'system' else self.user_config
        try:
            return config.getint(section, option, fallback=fallback)
        except (configparser.NoSectionError, configparser.NoOptionError):
            return fallback
    
    def get_float(self, section: str, option: str, fallback: float = 0.0,
                  config_type: str = 'system') -> float:
        """."""
        config = self.system_config if config_type == 'system' else self.user_config
        try:
            return config.getfloat(section, option, fallback=fallback)
        except (configparser.NoSectionError, configparser.NoOptionError):
            return fallback
    
    def get_bool(self, section: str, option: str, fallback: bool = False,
                 config_type: str = 'system') -> bool:
        """."""
        config = self.system_config if config_type == 'system' else self.user_config
        try:
            return config.getboolean(section, option, fallback=fallback)
        except (configparser.NoSectionError, configparser.NoOptionError):
            return fallback
    
    def set(self, section: str, option: str, value: Any, 
            config_type: str = 'system'):
        """
        ?
        
        Args:
            section: ?
            option: ?
            value: ?
            config_type:  ('system' ?'user')
        """
        config = self.system_config if config_type == 'system' else self.user_config
        
        if not config.has_section(section):
            config.add_section(section)
        
        config.set(section, option, str(value))
    
    def save_system_config(self):
        """"""
        with open(self.system_config_path, 'w', encoding='utf-8') as f:
            self.system_config.write(f)
    
    def save_user_config(self):
        """"""
        with open(self.user_config_path, 'w', encoding='utf-8') as f:
            self.user_config.write(f)
    
    def get_path(self, path_key: str) -> Path:
        """
        ?
        
        Args:
            path_key: PATH
        
        Returns:
            Path
        """
        defaults = self._default_path_values()
        sdk_override = SdkManager().resolve_path(path_key)
        if sdk_override is not None:
            return sdk_override

        path_str = self.get('PATH', path_key, fallback=defaults.get(path_key, ''))

        if not path_str:
            return self.project_root
        
        path = Path(path_str)
        
        # ?
        if not path.is_absolute():
            path = self.project_root / path
        
        return path

    def _to_config_path(self, path: Path) -> str:
        """Persist project-local paths as relative strings when possible."""
        try:
            return f"./{path.relative_to(self.project_root).as_posix()}"
        except ValueError:
            return str(path)
    
    def get_available_robots(self) -> list:
        """."""
        robots_str = self.get('SIMULATION', 'available_robots', fallback='go2')
        return [r.strip() for r in robots_str.split(',')]



