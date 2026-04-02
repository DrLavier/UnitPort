from __future__ import annotations

import json
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


class TrainingAssetImportError(Exception):
    """Raised when a training asset cannot be imported."""


@dataclass
class TrainingAssetEntry:
    asset_id: str
    asset_path: Path
    display_name: str = ""
    source_type: str = ""
    framework: str = ""
    algorithm: str = ""
    robot_type: str = ""
    is_valid: bool = True
    error: str = ""
    primary_checkpoint: str = ""
    obs_dim: int = 0
    action_dim: int = 0
    action_type: str = ""   # "torque" | "joint_position" | "velocity"

    def label(self) -> str:
        parts = [self.display_name or self.asset_id]
        if self.framework:
            parts.append(f"[{self.framework}]")
        if self.algorithm:
            parts.append(self.algorithm)
        return "  ".join(parts)


class TrainingAssetRegistry:
    """Registry for external trainable assets used only inside Training Ground."""

    STORAGE_SUBDIR = "custom_mods/training/assets"

    def __init__(self, root: Optional[Path] = None):
        self._explicit_root = root
        self._cache: List[TrainingAssetEntry] = []
        self._discovered = False

    @property
    def root(self) -> Path:
        """``custom_mods/`` lives at the repo root (parent of ``project_root``)."""
        if self._explicit_root is not None:
            return self._explicit_root
        try:
            from src.system.core.config_manager import ConfigManager
            return ConfigManager().get_path("project_root").parent / self.STORAGE_SUBDIR
        except Exception:
            return Path.cwd() / self.STORAGE_SUBDIR

    def discover(self) -> List[TrainingAssetEntry]:
        if not self.root.exists():
            self._cache = []
            self._discovered = True
            return []

        results: List[TrainingAssetEntry] = []
        for candidate in sorted(self.root.iterdir()):
            if not candidate.is_dir():
                continue
            results.append(self._parse_entry(candidate))
        self._cache = results
        self._discovered = True
        return results

    def refresh(self) -> None:
        self._discovered = False
        self.discover()

    def list_assets(self) -> List[TrainingAssetEntry]:
        if not self._discovered:
            self.discover()
        return list(self._cache)

    def get(self, asset_id: str) -> TrainingAssetEntry:
        if not self._discovered:
            self.discover()
        for entry in self._cache:
            if entry.asset_id == asset_id:
                return entry
        raise KeyError(asset_id)

    def import_local(self, src_path: Path) -> TrainingAssetEntry:
        src_path = Path(src_path)
        tmp_dir: Optional[Path] = None
        try:
            bundle_src = self._resolve_source(src_path)
            tmp_dir = bundle_src if bundle_src != src_path else None
            if not self._looks_like_training_asset(bundle_src):
                raise TrainingAssetImportError(
                    "Source does not look like a supported training asset package."
                )

            asset_id = self._asset_id_for(bundle_src.name)
            dest = self.root / asset_id
            if dest.exists():
                raise TrainingAssetImportError(
                    f"Training asset '{asset_id}' already exists at {dest}."
                )

            self.root.mkdir(parents=True, exist_ok=True)
            staging = self.root / f"{asset_id}__importing"
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
            shutil.copytree(bundle_src, staging)
            self._write_asset_manifest(staging, asset_id)
            self._write_default_source(staging, src_path)
            staging.rename(dest)
            self.refresh()
            return self.get(asset_id)
        finally:
            if tmp_dir is not None and tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)

    _HF_ACCEPTED_SOURCE_TYPES = frozenset({"huggingface", "huggingface_training_asset"})

    def import_hf_snapshot(self, bundle_path: Path) -> TrainingAssetEntry:
        bundle_path = Path(bundle_path)
        source_file = bundle_path / "source.json"
        if source_file.exists():
            try:
                data = json.loads(source_file.read_text(encoding="utf-8"))
                if data.get("type", "") not in self._HF_ACCEPTED_SOURCE_TYPES:
                    raise TrainingAssetImportError(
                        f"Expected huggingface source type, got '{data.get('type', '')}'."
                    )
            except TrainingAssetImportError:
                raise
            except Exception as exc:
                raise TrainingAssetImportError(f"Cannot read source.json: {exc}") from exc
        return self.import_local(bundle_path)

    @staticmethod
    def _asset_id_for(raw_name: str) -> str:
        return re.sub(r"[^a-z0-9_]", "_", str(raw_name or "").strip().lower()) or "training_asset"

    @staticmethod
    def _resolve_source(src_path: Path) -> Path:
        if not src_path.exists():
            raise TrainingAssetImportError(f"Source path does not exist: {src_path}")
        if src_path.is_dir():
            return src_path
        if src_path.suffix.lower() == ".zip":
            import zipfile
            if not zipfile.is_zipfile(src_path):
                raise TrainingAssetImportError(f"File is not a valid zip archive: {src_path}")
            tmp = Path(tempfile.mkdtemp(prefix="unitport_training_asset_"))
            with zipfile.ZipFile(src_path, "r") as zf:
                zf.extractall(tmp)
            children = [c for c in tmp.iterdir()]
            if len(children) == 1 and children[0].is_dir():
                return children[0]
            return tmp
        raise TrainingAssetImportError(
            f"Unsupported source type '{src_path.suffix}'. Please select a folder or a .zip archive."
        )

    @staticmethod
    def _looks_like_training_asset(bundle_src: Path) -> bool:
        patterns = ("best/best_model.zip", "best_model.zip", "checkpoints/*.zip", "*.zip")
        for pattern in patterns:
            for candidate in bundle_src.glob(pattern):
                if candidate.is_file():
                    return True
        return False

    def _parse_entry(self, asset_path: Path) -> TrainingAssetEntry:
        manifest = self._read_asset_manifest(asset_path)
        if manifest is None:
            manifest = self._infer_asset_manifest(asset_path, asset_path.name)
        return TrainingAssetEntry(
            asset_id=asset_path.name,
            asset_path=asset_path,
            display_name=str(manifest.get("display_name", "") or asset_path.name),
            source_type=str(manifest.get("source_type", "") or ""),
            framework=str(manifest.get("framework", "") or ""),
            algorithm=str(manifest.get("algorithm", "") or ""),
            robot_type=str(manifest.get("robot_type", "") or ""),
            is_valid=bool(manifest.get("is_valid", True)),
            error=str(manifest.get("error", "") or ""),
            primary_checkpoint=str(manifest.get("primary_checkpoint", "") or ""),
            obs_dim=int(manifest.get("obs_dim", 0) or 0),
            action_dim=int(manifest.get("action_dim", 0) or 0),
            action_type=str(manifest.get("action_type", "") or ""),
        )

    @staticmethod
    def _read_asset_manifest(asset_path: Path) -> Optional[dict]:
        manifest_path = asset_path / "asset_manifest.json"
        if not manifest_path.exists():
            return None
        try:
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _write_asset_manifest(self, asset_path: Path, asset_id: str) -> None:
        manifest = self._infer_asset_manifest(asset_path, asset_id)
        (asset_path / "asset_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @staticmethod
    def _write_default_source(asset_path: Path, src_path: Path) -> None:
        source_file = asset_path / "source.json"
        if source_file.exists():
            return
        source_file.write_text(
            json.dumps({"type": "local_training_asset", "src": str(Path(src_path).resolve())}, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _read_yaml_manifest(asset_path: Path) -> dict:
        """Read manifest.yaml if present; returns {} on any failure."""
        yaml_path = asset_path / "manifest.yaml"
        if not yaml_path.exists():
            return {}
        try:
            text = yaml_path.read_text(encoding="utf-8")
            # Minimal YAML scalar parser — avoids pyyaml dependency.
            # Only reads simple key: value and nested key:\n  subkey: value.
            result: dict = {}
            current_section: dict = {}
            current_key: str = ""
            for line in text.splitlines():
                if not line.strip() or line.strip().startswith("#"):
                    continue
                if line.startswith("  ") and current_key:
                    sub = line.strip()
                    if ":" in sub:
                        sk, _, sv = sub.partition(":")
                        current_section[sk.strip()] = sv.strip()
                else:
                    if ":" in line:
                        k, _, v = line.partition(":")
                        k = k.strip()
                        v = v.strip()
                        if v:
                            result[k] = v
                            current_key = ""
                            current_section = {}
                        else:
                            current_key = k
                            current_section = {}
                            result[k] = current_section
            return result
        except Exception:
            return {}

    @staticmethod
    def _infer_asset_manifest(asset_path: Path, asset_id: str) -> dict:
        readme_text = ""
        readme_path = asset_path / "README.md"
        if readme_path.exists():
            try:
                readme_text = readme_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                readme_text = ""
        rid = str(asset_id or "").lower()
        readme_lower = readme_text.lower()
        algorithm = ""
        for candidate in ("sac", "ppo", "td3"):
            if candidate in rid or candidate in readme_lower:
                algorithm = candidate.upper()
                break
        robot_type = "go2" if ("go2" in rid or "go2" in readme_lower) else ""
        primary = ""
        for pattern in ("best/best_model.zip", "best_model.zip", "checkpoints/*.zip", "*.zip"):
            matches = sorted(asset_path.glob(pattern))
            if matches:
                primary = str(matches[0].relative_to(asset_path)).replace("\\", "/")
                break
        source_info = {}
        source_path = asset_path / "source.json"
        if source_path.exists():
            try:
                source_info = json.loads(source_path.read_text(encoding="utf-8"))
            except Exception:
                source_info = {}
        source_type = str(source_info.get("type", "") or "")
        if "huggingface" in source_type:
            source_type = "huggingface"
        elif not source_type:
            source_type = "local"

        # Read obs/action dims from manifest.yaml if present
        yaml_data = TrainingAssetRegistry._read_yaml_manifest(asset_path)
        obs_section = yaml_data.get("observation_space", {})
        action_section = yaml_data.get("action_space", {})
        obs_dim = 0
        action_dim = 0
        action_type = ""
        if isinstance(obs_section, dict):
            try:
                obs_dim = int(obs_section.get("dim", 0) or 0)
            except (ValueError, TypeError):
                obs_dim = 0
        if isinstance(action_section, dict):
            try:
                action_dim = int(action_section.get("dim", 0) or 0)
            except (ValueError, TypeError):
                action_dim = 0
            action_type = str(action_section.get("type", "") or "")

        return {
            "asset_id": asset_id,
            "display_name": asset_id,
            "source_type": source_type,
            "framework": "sb3" if primary else "",
            "algorithm": algorithm,
            "robot_type": robot_type,
            "is_valid": bool(primary),
            "error": "" if primary else "No recoverable SB3 checkpoint found.",
            "primary_checkpoint": primary,
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "action_type": action_type,
        }
