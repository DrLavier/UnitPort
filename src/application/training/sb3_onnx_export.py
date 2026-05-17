"""SB3 → ONNX export helpers.

Splits the ``_build_ppo_onnx_net`` / ``_build_sac_onnx_net`` / ``export_to_onnx``
trio out of DEMO ``bundle_exporter.py`` (lines 153-254) so Stage 7's
``BundleExporter.export_bundle`` can compose them with the Phase 3
``export_from_artifacts`` path.

Stage 7 supports PPO, SAC, TD3. AMP-PPO actor export piggy-backs on
``_build_ppo_onnx_net`` because the AMP actor shares SB3's
``mlp_extractor.policy_net`` topology — Stage 8 will revisit if AMP
diverges.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union


# ---------------------------------------------------------------------------
# Net builders — return a traceable nn.Module wrapping the actor head
# ---------------------------------------------------------------------------


def build_ppo_onnx_net(policy):
    """Traceable nn.Module for deterministic PPO inference.

    Topology: obs → features_extractor → mlp_extractor.policy_net → action_net
    """
    import torch.nn as nn

    class _PPONet(nn.Module):
        def __init__(self, p):
            super().__init__()
            self._fe = p.features_extractor
            self._mlp = p.mlp_extractor.policy_net
            self._anet = p.action_net

        def forward(self, obs):
            features = self._fe(obs)
            latent_pi = self._mlp(features)
            return self._anet(latent_pi)

    net = _PPONet(policy)
    net.eval()
    return net


def build_sac_onnx_net(policy):
    """Traceable nn.Module for deterministic SAC / TD3 inference.

    Topology: obs → actor.latent_pi → actor.mu → tanh
    """
    import torch
    import torch.nn as nn

    class _SACNet(nn.Module):
        def __init__(self, p):
            super().__init__()
            self._latent = p.actor.latent_pi
            self._mu = p.actor.mu

        def forward(self, obs):
            return torch.tanh(self._mu(self._latent(obs)))

    net = _SACNet(policy)
    net.eval()
    return net


def build_actor_net(model):
    """Dispatch on ``type(model).__name__`` (PPO / SAC / TD3)."""
    name = type(model).__name__.upper()
    if name == "PPO":
        return build_ppo_onnx_net(model.policy)
    if name in ("SAC", "TD3"):
        return build_sac_onnx_net(model.policy)
    raise ValueError(
        f"ONNX export not supported for algorithm {name!r}; "
        f"Stage 7 supports PPO / SAC / TD3"
    )


# ---------------------------------------------------------------------------
# File-level exporters
# ---------------------------------------------------------------------------


def export_to_onnx(model, obs_dim: int, out_path: Union[Path, str]) -> Path:
    """Export the actor to a deterministic ONNX policy file.

    Returns the resolved output path. Raises :exc:`ValueError` for
    unsupported algorithms.
    """
    import torch

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    net = build_actor_net(model)
    try:
        export_device = next(net.parameters()).device
    except StopIteration:
        export_device = torch.device("cpu")
    net = net.to(export_device)

    dummy = torch.zeros(1, int(obs_dim), dtype=torch.float32, device=export_device)
    with torch.no_grad():
        torch.onnx.export(
            net,
            dummy,
            str(out),
            input_names=["obs"],
            output_names=["action"],
            dynamic_axes={
                "obs": {0: "batch_size"},
                "action": {0: "batch_size"},
            },
            opset_version=17,
            do_constant_folding=True,
        )
    return out


def export_to_onnx_bytes(model, obs_dim: int) -> bytes:
    """Export the actor to ONNX in-memory; returns serialized model bytes.

    Useful when the bundle exporter wants to avoid a temp file round-trip.
    """
    import io
    import torch

    net = build_actor_net(model)
    try:
        export_device = next(net.parameters()).device
    except StopIteration:
        export_device = torch.device("cpu")
    net = net.to(export_device)

    dummy = torch.zeros(1, int(obs_dim), dtype=torch.float32, device=export_device)
    buf = io.BytesIO()
    with torch.no_grad():
        torch.onnx.export(
            net,
            dummy,
            buf,
            input_names=["obs"],
            output_names=["action"],
            dynamic_axes={
                "obs": {0: "batch_size"},
                "action": {0: "batch_size"},
            },
            opset_version=17,
            do_constant_folding=True,
        )
    return buf.getvalue()


def export_to_torchscript(model, obs_dim: int, out_path: Union[Path, str]) -> Path:
    """Export the actor as TorchScript (JIT trace)."""
    import torch

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    net = build_actor_net(model)
    try:
        export_device = next(net.parameters()).device
    except StopIteration:
        export_device = torch.device("cpu")
    net = net.to(export_device)

    dummy = torch.zeros(1, int(obs_dim), dtype=torch.float32, device=export_device)
    with torch.no_grad():
        traced = torch.jit.trace(net, dummy)
    traced.save(str(out))
    return out


__all__ = [
    "build_ppo_onnx_net",
    "build_sac_onnx_net",
    "build_actor_net",
    "export_to_onnx",
    "export_to_onnx_bytes",
    "export_to_torchscript",
]
