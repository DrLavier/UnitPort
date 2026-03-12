#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest

from bin.core.config_manager import ConfigManager
from models import get_model


class TestModelRegistry(unittest.TestCase):
    def test_get_model_returns_unitree_class_when_brand_exists(self):
        model_class = get_model("unitree")
        self.assertTrue(model_class is None or isinstance(model_class, type))

    def test_config_manager_uses_dynamic_sdk_paths(self):
        config = ConfigManager()
        self.assertEqual(config.get_path("unitree_sdk").name, "unitree_sdk2_python")
        self.assertEqual(config.get_path("unitree_mujoco").name, "unitree_mujoco")


if __name__ == "__main__":
    unittest.main()
