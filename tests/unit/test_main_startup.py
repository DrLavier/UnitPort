#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import unittest
from unittest.mock import patch

import main


class TestConfigureLinuxRuntimeEnv(unittest.TestCase):
    def test_linux_wayland_sets_qt_platform(self):
        with patch.dict(os.environ, {"WAYLAND_DISPLAY": "wayland-0"}, clear=True):
            with patch("main.platform.system", return_value="Linux"):
                main._configure_linux_runtime_env()
                self.assertEqual(os.environ["QT_QPA_PLATFORM"], "wayland")

    def test_linux_x11_sets_qt_platform(self):
        with patch.dict(os.environ, {"DISPLAY": ":0"}, clear=True):
            with patch("main.platform.system", return_value="Linux"):
                main._configure_linux_runtime_env()
                self.assertEqual(os.environ["QT_QPA_PLATFORM"], "xcb")

    def test_linux_headless_sets_offscreen(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch("main.platform.system", return_value="Linux"):
                main._configure_linux_runtime_env()
                self.assertEqual(os.environ["QT_QPA_PLATFORM"], "offscreen")

    def test_existing_qt_platform_is_preserved(self):
        with patch.dict(os.environ, {"QT_QPA_PLATFORM": "minimal"}, clear=True):
            with patch("main.platform.system", return_value="Linux"):
                main._configure_linux_runtime_env()
                self.assertEqual(os.environ["QT_QPA_PLATFORM"], "minimal")

    def test_windows_does_not_set_qt_platform(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch("main.platform.system", return_value="Windows"):
                main._configure_linux_runtime_env()
                self.assertNotIn("QT_QPA_PLATFORM", os.environ)


if __name__ == "__main__":
    unittest.main()
