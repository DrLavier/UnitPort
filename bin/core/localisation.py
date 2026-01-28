#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Localisation Module
Provides internationalization support for the application
"""

import json
import threading
from pathlib import Path
from typing import Optional, Dict, Any

from PySide6.QtCore import QObject, Signal


class LocalisationManager(QObject):
    """Localisation manager - singleton pattern"""

    language_changed = Signal(str)  # Emitted when language changes

    _instance: Optional["LocalisationManager"] = None
    _lock = threading.Lock()

    # Supported languages
    SUPPORTED_LANGUAGES = {
        "en": "English",
        # Future: "zh": "Chinese", "ja": "Japanese", etc.
    }

    DEFAULT_LANGUAGE = "en"

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return

        super().__init__()
        self._initialized = True
        self._current_language = self.DEFAULT_LANGUAGE
        self._translations: Dict[str, Any] = {}
        self._localisation_dir: Optional[Path] = None

        # Auto-detect localisation directory
        self._detect_localisation_dir()

        # Load default language
        self.load_language(self._current_language)

    def _detect_localisation_dir(self):
        """Detect localisation directory"""
        # Try to find from current file location
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent
        localisation_dir = project_root / "localisation"

        if localisation_dir.exists():
            self._localisation_dir = localisation_dir
        else:
            # Create if not exists
            localisation_dir.mkdir(parents=True, exist_ok=True)
            self._localisation_dir = localisation_dir

    def set_localisation_dir(self, path: str):
        """Set localisation directory path"""
        self._localisation_dir = Path(path)

    def load_language(self, lang_code: str) -> bool:
        """
        Load language file

        Args:
            lang_code: Language code (e.g., "en", "zh")

        Returns:
            True if successful, False otherwise
        """
        if not self._localisation_dir:
            return False

        lang_file = self._localisation_dir / f"{lang_code}.json"

        if not lang_file.exists():
            # Fallback to default language
            if lang_code != self.DEFAULT_LANGUAGE:
                return self.load_language(self.DEFAULT_LANGUAGE)
            return False

        try:
            with open(lang_file, "r", encoding="utf-8") as f:
                self._translations = json.load(f)
            self._current_language = lang_code
            self.language_changed.emit(lang_code)
            return True
        except (json.JSONDecodeError, IOError):
            return False

    def get(self, key: str, default: str = "", **kwargs) -> str:
        """
        Get translated text

        Args:
            key: Dot-separated key (e.g., "toolbar.new", "messages.error")
            default: Default value if key not found
            **kwargs: Format arguments for string interpolation

        Returns:
            Translated text
        """
        keys = key.split(".")
        value = self._translations

        try:
            for k in keys:
                value = value[k]

            if isinstance(value, str):
                # Support format strings like "Hello {name}"
                if kwargs:
                    return value.format(**kwargs)
                return value
            elif isinstance(value, list):
                return value  # Return list as-is (for features)
            else:
                return default
        except (KeyError, TypeError):
            return default

    def get_list(self, key: str, default: list = None) -> list:
        """
        Get translated list (e.g., features)

        Args:
            key: Dot-separated key
            default: Default value if key not found

        Returns:
            List of translated items
        """
        result = self.get(key, default=None)
        if isinstance(result, list):
            return result
        return default or []

    @property
    def current_language(self) -> str:
        """Get current language code"""
        return self._current_language

    @property
    def current_language_name(self) -> str:
        """Get current language display name"""
        return self.SUPPORTED_LANGUAGES.get(self._current_language, "Unknown")

    def get_available_languages(self) -> Dict[str, str]:
        """Get available languages (code -> name)"""
        available = {}
        if self._localisation_dir:
            for lang_code, lang_name in self.SUPPORTED_LANGUAGES.items():
                lang_file = self._localisation_dir / f"{lang_code}.json"
                if lang_file.exists():
                    available[lang_code] = lang_name
        return available


# Global instance getter
def get_localisation() -> LocalisationManager:
    """Get global localisation manager instance"""
    return LocalisationManager()


# Shorthand function for translation
def tr(key: str, default: str = "", **kwargs) -> str:
    """
    Translate text (shorthand for get_localisation().get())

    Args:
        key: Dot-separated key
        default: Default value if not found
        **kwargs: Format arguments

    Returns:
        Translated text
    """
    return get_localisation().get(key, default, **kwargs)


def tr_list(key: str, default: list = None) -> list:
    """
    Get translated list (shorthand)

    Args:
        key: Dot-separated key
        default: Default list if not found

    Returns:
        Translated list
    """
    return get_localisation().get_list(key, default)
