#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HuggingFace checkpoint downloader and bundle normalizer.

Implements URL parsing, snapshot download (via huggingface_hub), and
normalization of the raw snapshot into a valid UnitPort checkpoint bundle.

Public API
----------
parse_hf_url(url)          -> (repo_id, revision)
download_hf_checkpoint(url, ...) -> Path   (normalized temp bundle dir)
normalize_hf_snapshot(...)  -> Path

Exceptions
----------
HFDownloadError   — all user-facing errors raised by this module
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Callable, Optional, Tuple

from system.policy.manifest_schema import ManifestValidationError, validate_manifest


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------

class HFDownloadError(Exception):
    """Raised when any stage of HF download or normalization fails."""


_GO2_JOINT_NAMES = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
]

_JOINT_NAMES_BY_ROBOT = {
    "go2": _GO2_JOINT_NAMES,
    "go1": _GO2_JOINT_NAMES,
    "b1": _GO2_JOINT_NAMES,
}


# ---------------------------------------------------------------------------
# URL parsing
# ---------------------------------------------------------------------------

# Matches:
#   https://huggingface.co/<owner>/<repo>
#   https://huggingface.co/<owner>/<repo>/tree/<revision>
# with optional trailing slash.
_HF_URL_RE = re.compile(
    r"^https://huggingface\.co/"
    r"(?P<owner>[^/]+)/(?P<repo>[^/]+)"
    r"(?:/tree/(?P<revision>[^/]+))?"
    r"/?$"
)


def parse_hf_url(url: str) -> Tuple[str, str]:
    """Parse a HuggingFace URL and return ``(repo_id, revision)``.

    Supported forms:

    * ``https://huggingface.co/<owner>/<repo>``
    * ``https://huggingface.co/<owner>/<repo>/tree/<revision>``

    Parameters
    ----------
    url:
        Raw URL string pasted by the user.

    Returns
    -------
    tuple[str, str]
        ``(repo_id, revision)`` where *repo_id* is ``"<owner>/<repo>"``
        and *revision* defaults to ``"main"`` when absent.

    Raises
    ------
    HFDownloadError
        When *url* does not match the expected HuggingFace pattern.
    """
    url = url.strip().rstrip("/")
    m = _HF_URL_RE.match(url)
    if not m:
        raise HFDownloadError(
            f"Invalid HuggingFace URL: '{url}'.\n"
            "Expected format:\n"
            "  https://huggingface.co/<owner>/<repo>\n"
            "  https://huggingface.co/<owner>/<repo>/tree/<revision>"
        )
    owner = m.group("owner")
    repo = m.group("repo")
    revision = m.group("revision") or "main"
    return f"{owner}/{repo}", revision


# ---------------------------------------------------------------------------
# Manifest inference
# ---------------------------------------------------------------------------

def _policy_id_from_repo(repo_id: str) -> str:
    """Derive a filesystem-safe policy_id from a repo_id."""
    repo_name = repo_id.split("/")[-1]          # e.g. "sac-unitree-go2-mujoco"
    return re.sub(r"[^a-z0-9_]", "_", repo_name.lower())


def _infer_manifest(snapshot_dir: Path, repo_id: str) -> dict:
    """Build a minimal manifest dict from snapshot directory contents.

    Heuristics used:

    * ``name`` — taken from the repository slug.
    * ``robot.brand`` / ``robot.type`` — inferred from keywords in *repo_id*.
    * ``observation_space.dim`` / ``action_space.dim`` — inferred for
      well-known robot types (e.g. Unitree Go2 SAC-MuJoCo uses 45/12).
    * ``runtime.format`` — first model-file extension found in the snapshot.
    """
    repo_name = repo_id.split("/")[-1]

    # Detect model files
    formats = []
    for suffix in ("onnx", "pt", "pth", "pkl", "zip", "safetensors"):
        if any(snapshot_dir.glob(f"*.{suffix}")):
            formats.append(suffix)

    manifest: dict = {
        "name": repo_name,
        "version": "0.1.0",
    }

    # Robot brand / type heuristics
    rid_lower = repo_id.lower()
    brand = ""
    robot_type = ""
    if "unitree" in rid_lower or "go2" in rid_lower or "b1" in rid_lower or "h1" in rid_lower:
        brand = "unitree"
        if "go2" in rid_lower:
            robot_type = "go2"
        elif "b1" in rid_lower:
            robot_type = "b1"
        elif "h1" in rid_lower:
            robot_type = "h1"
    elif "spot" in rid_lower:
        brand = "spot"
    elif "cyberdog" in rid_lower or "xiaomi" in rid_lower:
        brand = "cyberdog"

    if brand:
        robot_section: dict = {"brand": brand}
        if robot_type:
            robot_section["type"] = robot_type
        manifest["robot"] = robot_section

    # Observation / action dim heuristics for known configs
    if robot_type == "go2" and "mujoco" in rid_lower:
        manifest["observation_space"] = {"dim": 45}
        manifest["action_space"] = {"dim": 12}

    if formats:
        manifest["runtime"] = {"format": formats[0]}

    return manifest


def _find_supported_runtime_policy_file(snapshot_dir: Path) -> Tuple[str, str]:
    """Return ``(relative_path, format)`` for the first supported runtime file."""
    for filename, fmt in (("policy.onnx", "onnx"), ("policy_jit.pt", "jit")):
        exact = snapshot_dir / filename
        if exact.exists():
            return filename, fmt

    for pattern, fmt in (("*.onnx", "onnx"), ("policy_jit*.pt", "jit")):
        for candidate in sorted(snapshot_dir.rglob(pattern)):
            if candidate.is_file():
                rel = candidate.relative_to(snapshot_dir)
                return str(rel).replace("\\", "/"), fmt

    raise HFDownloadError(
        "HuggingFace repository does not contain a supported runtime policy file. "
        "UnitPort can only import runnable ONNX/TorchScript bundles "
        "(`policy.onnx` or `policy_jit.pt`). Repositories that only provide "
        "training artifacts such as Stable-Baselines3 `.zip` checkpoints "
        "cannot be executed directly."
    )


def _build_runtime_manifest(snapshot_dir: Path, repo_id: str) -> dict:
    """Build a runnable UnitPort manifest from a raw HF snapshot."""
    manifest = _infer_manifest(snapshot_dir, repo_id)
    policy_file, policy_format = _find_supported_runtime_policy_file(snapshot_dir)
    manifest["policy"] = {
        "file": policy_file,
        "format": policy_format,
    }

    robot = manifest.setdefault("robot", {})
    robot_type = str(robot.get("type", "") or "").lower()
    joint_names = _JOINT_NAMES_BY_ROBOT.get(robot_type)

    if robot_type == "go2" and "mujoco" in repo_id.lower():
        manifest["observation_space"] = {"dim": 45}
        manifest["action_space"] = {"dim": 12, "type": "joint_position"}
        manifest["runtime"] = {
            "control_frequency_hz": 50.0,
            "decimation": 4,
        }

    if joint_names:
        robot["num_joints"] = len(joint_names)
        robot["joint_names"] = list(joint_names)

    validate_manifest(manifest, bundle_path=snapshot_dir)
    return manifest


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_hf_snapshot(
    snapshot_dir: Path,
    repo_id: str,
    revision: str,
    url: str,
    *,
    policy_id: Optional[str] = None,
) -> Path:
    """Normalize a raw HuggingFace snapshot into a valid checkpoint bundle.

    Creates a fresh temporary directory, copies the snapshot contents into a
    named sub-directory (the bundle root), then:

    1. Validates ``manifest.yaml`` if it exists, or writes an inferred one.
    2. Writes ``source.json`` with HuggingFace provenance.

    Parameters
    ----------
    snapshot_dir:
        Path to the directory populated by ``snapshot_download()``.
    repo_id:
        Repository identifier (``"<owner>/<repo>"``).
    revision:
        Revision string (branch, tag, or commit hash).
    url:
        Original URL provided by the user (stored in ``source.json``).
    policy_id:
        Override for the bundle directory name.  Defaults to a sanitised
        version of the repository slug.

    Returns
    -------
    Path
        Path to the normalized bundle directory (inside a temporary parent).
        The caller must install it via ``CheckpointRegistry`` and then remove
        the parent temp directory.
    """
    if policy_id is None:
        policy_id = _policy_id_from_repo(repo_id)

    # Create isolated temp dir; bundle lives as a sub-directory
    bundle_parent = Path(tempfile.mkdtemp(prefix="unitport_hf_bundle_"))
    bundle_dir = bundle_parent / policy_id

    # Copy snapshot → bundle
    shutil.copytree(str(snapshot_dir), str(bundle_dir))

    # --- manifest.yaml ---
    manifest_path = bundle_dir / "manifest.yaml"
    if manifest_path.exists():
        _validate_existing_manifest(manifest_path)
    else:
        _write_inferred_manifest(bundle_dir, repo_id)

    # --- source.json ---
    source_meta = {
        "type": "huggingface",
        "repo_id": repo_id,
        "revision": revision,
        "url": url,
    }
    (bundle_dir / "source.json").write_text(
        json.dumps(source_meta, indent=2), encoding="utf-8"
    )

    return bundle_dir


def _validate_existing_manifest(manifest_path: Path) -> None:
    """Validate an existing manifest.yaml in the snapshot."""
    try:
        import yaml
        with manifest_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, dict):
            raise HFDownloadError("manifest.yaml must contain a YAML mapping.")
        validate_manifest(data, bundle_path=manifest_path.parent)
        policy_file = data.get("policy", {}).get("file", "")
        if not policy_file:
            raise HFDownloadError("manifest.yaml is missing required field 'policy.file'.")
        resolved = manifest_path.parent / str(policy_file)
        if not resolved.exists():
            raise HFDownloadError(
                f"manifest.yaml references missing policy file: {policy_file}"
            )
    except ManifestValidationError as exc:
        raise HFDownloadError(
            "manifest.yaml is not a runnable UnitPort checkpoint manifest: "
            f"{exc}"
        ) from exc
    except HFDownloadError:
        raise
    except ImportError:
        pass  # PyYAML not available — skip validation; registry will catch later
    except Exception as exc:
        raise HFDownloadError(f"Failed to parse manifest.yaml: {exc}") from exc


def _write_inferred_manifest(bundle_dir: Path, repo_id: str) -> None:
    """Write an inferred manifest.yaml into *bundle_dir*."""
    manifest_data = _build_runtime_manifest(bundle_dir, repo_id)
    manifest_path = bundle_dir / "manifest.yaml"
    try:
        import yaml
        with manifest_path.open("w", encoding="utf-8") as fh:
            yaml.dump(manifest_data, fh, default_flow_style=False, allow_unicode=True)
    except ImportError:
        # Fallback: write minimal YAML manually
        lines = []
        for k, v in manifest_data.items():
            if isinstance(v, dict):
                lines.append(f"{k}:\n")
                for sk, sv in v.items():
                    lines.append(f"  {sk}: {sv}\n")
            else:
                lines.append(f"{k}: {v}\n")
        manifest_path.write_text("".join(lines), encoding="utf-8")
    except Exception as exc:
        raise HFDownloadError(f"Failed to write manifest.yaml: {exc}") from exc


# ---------------------------------------------------------------------------
# Main download entry point
# ---------------------------------------------------------------------------

def download_hf_checkpoint(
    url: str,
    *,
    progress_callback: Optional[Callable[[str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Path:
    """Download a HuggingFace repository and produce a normalized bundle.

    This is the primary entry point called by the download worker thread.

    Parameters
    ----------
    url:
        Full HuggingFace URL (``https://huggingface.co/...``).
    progress_callback:
        Optional callable that receives a human-readable progress string.
    cancel_check:
        Optional callable that returns ``True`` when the operation should
        be aborted.  Checked at key checkpoints in the download flow.

    Returns
    -------
    Path
        Normalized bundle directory (temporary).  Install it using
        ``CheckpointRegistry.import_hf_bundle()`` and then delete
        its parent temp directory.

    Raises
    ------
    HFDownloadError
        On invalid URL, missing dependency, download failure, or cancellation.
    """
    # 1. Parse URL
    repo_id, revision = parse_hf_url(url)

    def _emit(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)

    def _cancelled() -> bool:
        return bool(cancel_check and cancel_check())

    _emit(f"Parsed: {repo_id}  @  {revision}")

    if _cancelled():
        raise HFDownloadError("Download cancelled.")

    # 2. Ensure huggingface_hub is available
    try:
        from huggingface_hub import snapshot_download  # noqa: F401
    except ImportError:
        raise HFDownloadError(
            "huggingface_hub is not installed.\n"
            "Run:  pip install huggingface_hub"
        )

    # 3. Download to a temporary staging directory
    tmp_snapshot = Path(tempfile.mkdtemp(prefix="unitport_hf_snapshot_"))
    bundle_dir: Optional[Path] = None

    try:
        _emit(f"Downloading {repo_id} @ {revision}…")

        from huggingface_hub import snapshot_download

        downloaded_path = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=str(tmp_snapshot),
        )

        if _cancelled():
            raise HFDownloadError("Download cancelled.")

        _emit("Download complete. Normalizing bundle…")

        # 4. Normalize snapshot into a proper bundle
        bundle_dir = normalize_hf_snapshot(
            Path(downloaded_path),
            repo_id=repo_id,
            revision=revision,
            url=url,
        )

        if _cancelled():
            # Clean up the normalized bundle we just created
            parent = bundle_dir.parent
            shutil.rmtree(parent, ignore_errors=True)
            raise HFDownloadError("Download cancelled.")

        _emit("Bundle ready.")
        return bundle_dir

    except HFDownloadError:
        if bundle_dir is not None:
            shutil.rmtree(bundle_dir.parent, ignore_errors=True)
        raise
    except Exception as exc:
        if bundle_dir is not None:
            shutil.rmtree(bundle_dir.parent, ignore_errors=True)
        raise HFDownloadError(f"Download failed: {exc}") from exc
    finally:
        # Always remove the raw snapshot staging dir
        shutil.rmtree(tmp_snapshot, ignore_errors=True)
