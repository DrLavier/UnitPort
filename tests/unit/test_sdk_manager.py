#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import unittest
from pathlib import Path
from shutil import rmtree
from unittest.mock import Mock, patch

from models.sdk_manager import SdkManager, SdkProject, configure_cyclonedds_env


class TestSdkManager(unittest.TestCase):
    def setUp(self):
        self.manager = SdkManager()
        self.manager.load_registry(force_reload=True)
        self._temp_root = Path(__file__).resolve().parents[2] / ".tmp" / "test_sdk_manager"
        if self._temp_root.exists():
            rmtree(self._temp_root, ignore_errors=True)
        self._temp_root.mkdir(parents=True, exist_ok=True)
        self.manager.state_path = self._temp_root / "sdk_state.json"

    def tearDown(self):
        if self._temp_root.exists():
            rmtree(self._temp_root, ignore_errors=True)

    def test_registry_contains_expected_projects(self):
        projects = {project.name: project for project in self.manager.get_projects()}

        self.assertIn("unitree_sdk2_python", projects)
        self.assertIn("unitree_mujoco", projects)
        self.assertIn("spot-sdk", projects)

    def test_resolve_known_paths(self):
        unitree_sdk = self.manager.resolve_path("unitree_sdk")
        unitree_mujoco = self.manager.resolve_path("unitree_mujoco")
        unitree_robots = self.manager.resolve_path("unitree_robots")

        self.assertIsInstance(unitree_sdk, Path)
        self.assertEqual(unitree_sdk.name, "unitree_sdk2_python")
        self.assertEqual(unitree_mujoco.name, "unitree_mujoco")
        self.assertEqual(unitree_robots.name, "unitree_robots")

    def test_ensure_registered_sdks_clones_missing_project(self):
        fake_project = SdkProject(
            brand="FakeBrand",
            brand_dir=self._temp_root / "FakeBrand",
            name="fake_sdk",
            url="https://example.invalid/fake.git",
        )

        with patch.object(self.manager, "load_registry", return_value=[fake_project]):
            with patch.object(self.manager, "_clone_project") as clone_mock:
                ensured = self.manager.ensure_registered_sdks()

        clone_mock.assert_called_once_with(fake_project, progress=None)
        self.assertEqual(ensured, [fake_project.path])

    def test_ensure_registered_sdks_installs_requirements_after_clone(self):
        brand_dir = self._temp_root / "FakeBrand"
        project = SdkProject(
            brand="FakeBrand",
            brand_dir=brand_dir,
            name="fake_sdk",
            url="https://example.invalid/fake.git",
        )
        requirement_file = project.path / "requirements.txt"

        def clone_side_effect(_project, progress=None):
            requirement_file.parent.mkdir(parents=True, exist_ok=True)
            requirement_file.write_text("requests==2.0.0\n", encoding="utf-8")

        with patch.object(self.manager, "load_registry", return_value=[project]):
            with patch.object(self.manager, "_clone_project", side_effect=clone_side_effect):
                with patch("models.sdk_manager.subprocess.run") as run_mock:
                    run_mock.return_value = Mock(returncode=0, stdout="", stderr="")
                    self.manager.ensure_registered_sdks()

        run_mock.assert_called_once()
        command = run_mock.call_args[0][0]
        self.assertEqual(command[2:4], ["pip", "install"])
        self.assertEqual(command[4], "-r")
        self.assertEqual(command[5], str(requirement_file))

    def test_ensure_registered_sdks_skips_reinstall_when_manifest_unchanged(self):
        brand_dir = self._temp_root / "FakeBrand"
        project = SdkProject(
            brand="FakeBrand",
            brand_dir=brand_dir,
            name="fake_sdk",
            url="https://example.invalid/fake.git",
        )
        requirement_file = project.path / "requirements.txt"
        requirement_file.parent.mkdir(parents=True, exist_ok=True)
        requirement_file.write_text("requests==2.0.0\n", encoding="utf-8")

        with patch.object(self.manager, "load_registry", return_value=[project]):
            with patch("models.sdk_manager.subprocess.run") as run_mock:
                run_mock.return_value = Mock(returncode=0, stdout="", stderr="")
                self.manager.ensure_registered_sdks()
                self.manager.ensure_registered_sdks()

        run_mock.assert_called_once()


class TestConfigureCycloneDDSEnv(unittest.TestCase):
    def setUp(self):
        self._temp_root = Path(__file__).resolve().parents[2] / ".tmp" / "test_cyclonedds_env"
        if self._temp_root.exists():
            rmtree(self._temp_root, ignore_errors=True)
        (self._temp_root / "runtime" / "cyclonedds" / "bin").mkdir(parents=True, exist_ok=True)
        (self._temp_root / "runtime" / "cyclonedds" / "lib").mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        if self._temp_root.exists():
            rmtree(self._temp_root, ignore_errors=True)

    def test_linux_injects_path_and_ld_library_path(self):
        with patch.dict(os.environ, {"PATH": "base-path"}, clear=True):
            with patch("models.sdk_manager.platform.system", return_value="Linux"):
                result = configure_cyclonedds_env(self._temp_root)
                cdds_dir = self._temp_root / "runtime" / "cyclonedds"
                self.assertEqual(result, str(cdds_dir))
                self.assertEqual(os.environ["CYCLONEDDS_HOME"], str(cdds_dir))
                self.assertIn(str(cdds_dir / "bin"), os.environ["PATH"])
                self.assertIn(str(cdds_dir / "lib"), os.environ["LD_LIBRARY_PATH"])

    def test_windows_keeps_path_only_and_does_not_set_ld_library_path(self):
        with patch.dict(os.environ, {"PATH": "base-path"}, clear=True):
            with patch("models.sdk_manager.platform.system", return_value="Windows"):
                result = configure_cyclonedds_env(self._temp_root)
                cdds_dir = self._temp_root / "runtime" / "cyclonedds"
                self.assertEqual(result, str(cdds_dir))
                self.assertEqual(os.environ["CYCLONEDDS_HOME"], str(cdds_dir))
                self.assertIn(str(cdds_dir / "bin"), os.environ["PATH"])
                self.assertNotIn("LD_LIBRARY_PATH", os.environ)


if __name__ == "__main__":
    unittest.main()
