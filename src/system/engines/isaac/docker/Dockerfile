# ==============================================================================
# UnitPort Isaac Lab Training Image
# ==============================================================================
# Thin extension of the official NVIDIA Isaac Lab image.
#
# The official image already contains:
#   Isaac Sim, IsaacLab, PyTorch, isaaclab.sh, CUDA runtime
#
# This layer adds only what UnitPort specifically needs:
#   1. rsl-rl-lib  (RSL-RL v5 for OnPolicyRunner)
#   2. h5py sitecustomize.py  (preload workaround for SimulationApp crash)
#   3. Critical pin locks  (numpy, h5py ABI match with Isaac Sim)
#
# Build:
#   docker build -t unitport/isaaclab:latest .
#
# Quick test:
#   docker run --rm --gpus all unitport/isaaclab:latest \
#       isaaclab.sh -p /workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/train.py \
#       --task Isaac-Velocity-Flat-Unitree-Go2-v0 --num_envs 64 --headless
# ==============================================================================

ARG BASE_IMAGE=nvcr.io/nvidia/isaac-lab:4.2.0
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-c"]
USER root

# Path to Isaac Lab's embedded Python (official image layout).
# If isaaclab.sh exists, use its _isaac_sim/python.sh; else fall back.
ENV ISAACLAB_PATH="${ISAACLAB_PATH:-/workspace/isaaclab}"

# Helper: PIP through Isaac Sim's Python so packages land in the right
# site-packages.  Official image may use either _isaac_sim/python.sh or
# a conda env — try both.
RUN ISAACSIM_PYTHON="" && \
    if [ -x "${ISAACLAB_PATH}/_isaac_sim/python.sh" ]; then \
        ISAACSIM_PYTHON="${ISAACLAB_PATH}/_isaac_sim/python.sh"; \
    elif command -v python3 &>/dev/null; then \
        ISAACSIM_PYTHON="python3"; \
    else \
        ISAACSIM_PYTHON="python"; \
    fi && \
    echo "ISAACSIM_PYTHON=${ISAACSIM_PYTHON}" && \
    \
    # ── 1. RSL-RL + extras (--no-deps to protect existing torch/numpy) ─── \
    ${ISAACSIM_PYTHON} -m pip install --no-deps \
        "rsl-rl-lib==5.0.1" \
        "onnxscript>=0.5" \
        tensordict \
        GitPython gitdb smmap \
        onnx_ir importlib_metadata orjson pyvers && \
    \
    # ── 2. Pin locks (isaacsim ABI-critical) ────────────────────────────── \
    ${ISAACSIM_PYTHON} -m pip install \
        "numpy==1.26.0" \
        "h5py==3.13.0" \
        "Pillow==11.3.0" \
        "typing_extensions==4.12.2" && \
    \
    # ── 3. h5py sitecustomize.py workaround ─────────────────────────────── \
    SITE_PKG=$(${ISAACSIM_PYTHON} -c 'import site; print(site.getsitepackages()[0])') && \
    printf '%s\n' \
        'try:' \
        '    import h5py  # noqa: F401' \
        'except ImportError:' \
        '    pass' \
    > "${SITE_PKG}/sitecustomize.py" && \
    \
    # ── 4. Remove quadprog (official IsaacLab workaround) ───────────────── \
    ${ISAACSIM_PYTHON} -m pip uninstall -y quadprog 2>/dev/null || true && \
    \
    echo "[unitport] Extension layer installed OK"

# Accept EULA by default.
ENV OMNI_KIT_ACCEPT_EULA=YES
ENV PYTHONUNBUFFERED=1

WORKDIR ${ISAACLAB_PATH}
