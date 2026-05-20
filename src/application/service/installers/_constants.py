"""Pinned manifest for the in-app Isaac Lab installer.

Isaac Sim 5.x is officially distributed through NVIDIA's PyPI index
(``https://pypi.nvidia.com``) since 4.2.0 (Sep 2024). The installer
therefore does NOT download a raw zip; it ``pip install``s
``isaacsim`` into either the project ``.venv311`` (internal mode) or a
freshly-created dedicated venv under ``<install_dir>/venv/`` (external
mode), then ``git clone``s Isaac Lab and runs its bundled
``isaaclab.bat -i`` / ``isaaclab.sh -i`` to finish the Python-side
wiring.

This is a deliberate simplification of the original plan (which
assumed a direct binary zip + SHA256 verify) — the pip channel:

* removes the need to pin a zip URL + SHA256 that NVIDIA might rotate;
* honours the user's PyTorch / CUDA pin already declared in
  requirements.txt (the dependency solver harmonises);
* leaves NVIDIA's signed wheels as the source of truth for SHA
  integrity (pip + the wheel trust chain handles it).

EULA text + version tracking is still pinned here so the EulaDialog
re-prompts when the upstream wording changes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IsaacSimRelease:
    """Pip-distributable Isaac Sim release the installer targets."""

    # Human-readable version label shown in the wizard hint and dialog
    # title. Display-only — pip dependency resolution honours
    # ``pip_spec`` below.
    version_label: str
    # pip spec passed to ``pip install``. Set to a bare package name to
    # always pull the latest available on the NVIDIA index, or pin
    # explicitly via "isaacsim==5.0.0".
    pip_spec: str
    # NVIDIA's package index host. Passed to pip as
    # ``--extra-index-url``; the default PyPI index is still consulted
    # for transitive deps that NVIDIA does not mirror.
    pypi_extra_index: str = "https://pypi.nvidia.com"


@dataclass(frozen=True)
class IsaacLabRelease:
    """Git revision of Isaac Lab to clone alongside Isaac Sim."""

    git_url: str
    # Refspec passed to ``git checkout``. ``"main"`` follows upstream;
    # a value like ``"v2.1.0"`` pins to a release tag. The installer
    # uses ``git fetch --depth=1`` against this refspec to keep the
    # clone shallow.
    branch_or_tag: str


@dataclass(frozen=True)
class EulaTextSpec:
    """One licence document shipped under installers/data/."""

    eula_id: str
    version: str
    file_name: str
    # SHA256 of the vendored file's bytes. Empty string disables the
    # check (e.g. when the text is UnitPort-authored, like the
    # silent-prompts disclosure, and has no upstream to drift from).
    # Mismatch is a warning, never a hard fail — the dialog surfaces a
    # banner pointing at ``upstream_url`` so the user can re-review.
    sha256: str
    upstream_url: str


# ---------------------------------------------------------------------------
# Pinned values
# ---------------------------------------------------------------------------

ISAAC_SIM_RELEASE = IsaacSimRelease(
    version_label="5.1.0.0",
    # ``[all,extscache]`` is NVIDIA's documented training-ready combo:
    # ``[all]`` pulls every Isaac Sim extension wheel (without it the
    # core package installs but ``isaacsim.simulation_app`` etc. error
    # at import time with "Extension not found", which makes Isaac Lab
    # unusable); ``[extscache]`` ships the pre-built C++ extension
    # cache so the first import doesn't try to compile half the kit
    # SDK on the user's machine.
    pip_spec="isaacsim[all,extscache]==5.1.0.0",
)


ISAAC_LAB_RELEASE = IsaacLabRelease(
    git_url="https://github.com/isaac-sim/IsaacLab.git",
    # ``v2.3.2`` is the latest stable Isaac Lab tag matching the Isaac
    # Sim 5.1.x release window (Nov 2025). Verified via
    # ``git ls-remote --tags`` against the upstream repo. The earlier
    # ``v0.54.3`` pin in this constant did not exist on the public
    # repo and made every fresh install fail at the ``git fetch tag``
    # step. Bump alongside ``ISAAC_SIM_RELEASE`` when NVIDIA cuts a
    # matched pair; the ``source/isaaclab*/pyproject.toml`` layout the
    # installer relies on has been stable since v1.0.0.
    branch_or_tag="v2.3.2",
)


EULA_TEXTS: tuple[EulaTextSpec, ...] = (
    EulaTextSpec(
        eula_id="nvola_5_1",
        version="2024-12",
        file_name="nvola.txt",
        sha256="",
        upstream_url=(
            "https://www.nvidia.com/en-us/agreements/cloud-services/omniverse-license/"
        ),
    ),
    EulaTextSpec(
        eula_id="isaaclab_bsd3",
        version="2024",
        file_name="bsd3_isaaclab.txt",
        sha256="",
        upstream_url=(
            "https://github.com/isaac-sim/IsaacLab/blob/main/LICENSE"
        ),
    ),
    EulaTextSpec(
        eula_id="silent_prompts_ack",
        version="1",
        file_name="silent_prompts.md",
        sha256="",   # UnitPort-authored disclosure; no upstream drift to detect
        upstream_url="",
    ),
)


# ---------------------------------------------------------------------------
# Disk-space / install constants
# ---------------------------------------------------------------------------

# Sum of: pip cache headroom (~3 GB) + isaacsim wheels (~6-8 GB) +
# Isaac Lab clone (~0.5 GB) + Isaac Lab Python-side install (~2 GB) +
# safety margin. Pip-based install is significantly leaner than the
# zip-based approach (which needed both a 15 GB zip and ~25 GB
# extracted simultaneously).
PREFLIGHT_FREE_BYTES_MIN = 20 * 1024 * 1024 * 1024

# Windows long-path cap. Hard system limit is 260; we leave headroom
# for Isaac Lab's deep extension paths.
PREFLIGHT_WIN_PATH_LEN_MAX = 200

# Env vars passed when running ``isaaclab.bat -i`` / ``isaaclab.sh -i``.
INSTALLER_ACCEPT_ENV: dict[str, str] = {
    "ACCEPT_EULA": "Y",
    "OMNI_KIT_ACCEPT_EULA": "YES",
    "NVIDIA_OMNIVERSE_LICENSE": "accepted",
    "PRIVACY_CONSENT": "Y",
}

# stdin auto-yes fallback for older interactive prompts.
INSTALLER_AUTO_YES_STDIN = b"y\ny\ny\ny\n"


__all__ = [
    "IsaacSimRelease",
    "IsaacLabRelease",
    "EulaTextSpec",
    "ISAAC_SIM_RELEASE",
    "ISAAC_LAB_RELEASE",
    "EULA_TEXTS",
    "PREFLIGHT_FREE_BYTES_MIN",
    "PREFLIGHT_WIN_PATH_LEN_MAX",
    "INSTALLER_ACCEPT_ENV",
    "INSTALLER_AUTO_YES_STDIN",
]
