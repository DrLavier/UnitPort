"""Convert an Isaac Lab / RSL-RL checkpoint to a self-contained ``policy.onnx``.

Background
----------
RSL-RL's ``OnPolicyRunner.save()`` writes checkpoints as PyTorch zip-format
state dicts (``{actor_state_dict, critic_state_dict, optimizer_state_dict,
iter, infos}``). The Isaac Lab ``play.py`` script normally re-exports the
actor to ONNX via ``torch.onnx.export`` so the bundle can be replayed
without depending on Isaac Lab at inference time.

When that subprocess export step fails for any reason — wrong CWD, Kit
boot failure, modified project layout — the bundle silently ends up
with the raw ``model_*.pt`` and a manifest claiming ``inference_backend:
onnx``. ``onnxruntime`` then chokes on the protobuf parse and the user
sees ``[ONNXRuntimeError] : 7 : INVALID_PROTOBUF`` at replay time.

This module reproduces the conversion **in-process inside UnitPort's own
venv** (which has torch but not rsl_rl), so the import path no longer
depends on Isaac Lab being usable. The result is a small, self-contained
``policy.onnx`` file that the existing ONNXEngine loads natively.

Supported checkpoint formats
----------------------------
Two rsl_rl flavours are supported, auto-detected by inspecting the
top-level keys of the checkpoint dict:

1. **Standard ``OnPolicyRunner`` + ``MLPModel``** (Isaac Lab default)::

       {actor_state_dict, critic_state_dict, optimizer_state_dict, iter, infos}

   with actor keys ``mlp.0.weight``, ``mlp.0.bias``, ``mlp.2.weight`` …
   (odd indices are activations with no params).

2. **AMP ``AMPOnPolicyRunner`` + ``ActorCritic``** (AMP-PPO, vendored
   in ``src/system/training/amp/runners``)::

       {model_state_dict, optimizer_state_dict, discriminator_state_dict,
        amp_normalizer, iter, infos}

   where ``model_state_dict`` holds an ``ActorCritic`` whose actor keys
   are ``actor.0.weight``, ``actor.0.bias``, ``actor.2.weight`` …
   (same alternating Linear/Act layout, different prefix).

Both flavours reduce to a single ``nn.Sequential`` of ``Linear`` layers
separated by an activation, so after stripping the leading
``mlp.`` / ``actor.`` prefix we can load the same flat key set into a
freshly built MLP and export.

Source of truth for dims
------------------------
We infer ``obs_dim``, ``action_dim``, and ``hidden_dims`` **from the
saved tensor shapes**, not from ``params/agent.yaml``. Rationale: the
AMP-PPO path uses our vendored ``ActorCritic`` but Isaac Lab still
serializes an ``RslRlOnPolicyRunnerCfg`` dataclass that reflects its
default ``MLPModel`` config, producing an ``agent.yaml`` whose
``actor.hidden_dims`` do not match the runner that actually ran. The
tensors are always correct; the YAML is only consulted for the
activation function (with an ``"elu"`` fallback).

The ``obs_normalization`` warning is preserved — if a future config
turns it on, the runtime ``Normalizer`` must still be populated
separately from ``normalization_stats.json``.

Limitations
-----------
- ``obs_normalization=True`` would mean rsl_rl's ``EmpiricalNormalization``
  layer is part of the actor. We don't fold that into the exported ONNX —
  if a future bundle uses it, the runtime ``Normalizer`` (which loads
  ``normalization_stats.json``) needs to be populated separately. The
  Go2 IL preset has ``obs_normalization=False``, so this is fine for now.
  We log a warning if we encounter ``True``.
- Only deterministic inference is exported (the actor's mean output).
  The Gaussian std is exploration noise used at training time — replay
  always wants the deterministic mean, which is what the MLP's last
  Linear layer outputs directly.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MLP reconstruction
# ---------------------------------------------------------------------------

def _activation_module(name: str):
    import torch.nn as nn

    key = (name or "elu").strip().lower()
    table = {
        "elu":        nn.ELU,
        "relu":       nn.ReLU,
        "tanh":       nn.Tanh,
        "leaky_relu": nn.LeakyReLU,
        "selu":       nn.SELU,
        "gelu":       nn.GELU,
        "silu":       nn.SiLU,
        "swish":      nn.SiLU,
        "sigmoid":    nn.Sigmoid,
    }
    cls = table.get(key)
    if cls is None:
        raise ValueError(
            f"Unsupported activation '{name}'. Supported: {sorted(table)}"
        )
    return cls()


def _build_actor_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: List[int],
    activation: str,
):
    """Mirror rsl_rl ``MLPModel.mlp`` exactly: ``Linear, Act, Linear, Act, ...``.

    The resulting ``nn.Sequential`` has parameter keys ``0.weight, 0.bias,
    2.weight, 2.bias, ..., {2*N}.weight, {2*N}.bias`` — i.e. odd indices
    are activations with no params. This matches the layout the rsl_rl
    state dict expects after stripping the ``mlp.`` prefix.
    """
    import torch.nn as nn

    layers: List[nn.Module] = []
    prev = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        layers.append(_activation_module(activation))
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


def _extract_actor_mlp_state_dict(
    ckpt: Dict[str, Any],
) -> Tuple[Dict[str, Any], str]:
    """Locate the actor sub–state-dict inside a checkpoint and normalise its keys.

    Returns ``(stripped_state_dict, format_name)`` where the returned
    dict uses flat keys ``"0.weight"``, ``"0.bias"``, ``"2.weight"`` …
    ready to load into an ``nn.Sequential``. ``format_name`` is one of
    ``"rsl_rl_mlpmodel"`` or ``"amp_actor_critic"``, for logging.

    Raises ``ValueError`` if neither known format is detected.
    """
    # Flavour 1 — standard rsl_rl OnPolicyRunner + MLPModel.
    if isinstance(ckpt, dict) and "actor_state_dict" in ckpt:
        actor_sd = ckpt["actor_state_dict"]
        if not isinstance(actor_sd, dict):
            raise ValueError(
                f"'actor_state_dict' is not a dict (got {type(actor_sd).__name__})"
            )
        mlp_sd = {k: v for k, v in actor_sd.items() if k.startswith("mlp.")}
        if not mlp_sd:
            raise ValueError(
                "actor_state_dict has no 'mlp.*' keys — the MLPModel format "
                f"may have changed. Available keys: {list(actor_sd.keys())}"
            )
        stripped = {k[len("mlp."):]: v for k, v in mlp_sd.items()}
        return stripped, "rsl_rl_mlpmodel"

    # Flavour 2 — AMP AMPOnPolicyRunner + ActorCritic (vendored in-tree).
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model_sd = ckpt["model_state_dict"]
        if not isinstance(model_sd, dict):
            raise ValueError(
                f"'model_state_dict' is not a dict (got {type(model_sd).__name__})"
            )
        actor_sd = {k: v for k, v in model_sd.items() if k.startswith("actor.")}
        if not actor_sd:
            raise ValueError(
                "model_state_dict has no 'actor.*' keys — the ActorCritic "
                f"format may have changed. Available keys: {list(model_sd.keys())}"
            )
        stripped = {k[len("actor."):]: v for k, v in actor_sd.items()}
        return stripped, "amp_actor_critic"

    have = list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt).__name__
    raise ValueError(
        "Checkpoint is neither an rsl_rl OnPolicyRunner save "
        "(expected 'actor_state_dict') nor an AMP AMPOnPolicyRunner save "
        f"(expected 'model_state_dict'). Got: {have}"
    )


def _infer_dims_from_stripped_sd(
    stripped_sd: Dict[str, Any],
) -> Tuple[int, List[int], int]:
    """Infer ``(obs_dim, hidden_dims, action_dim)`` from a flat MLP state dict.

    The state dict is expected to use ``nn.Sequential`` indexing where
    Linears sit at even indices (0, 2, 4, …) and activations at odd
    indices (no params). We scan every ``{i}.weight`` key, sort by
    index, and reconstruct the layer sizes from the tensor shapes. The
    checkpoint tensors are the ground truth; callers should not pass
    hidden_dims from config because it may disagree (see module docstring).
    """
    linear_indices: List[int] = []
    for k in stripped_sd:
        if not k.endswith(".weight"):
            continue
        head = k.split(".", 1)[0]
        try:
            linear_indices.append(int(head))
        except ValueError:
            continue
    if not linear_indices:
        raise KeyError(
            "No '{i}.weight' Linear keys in stripped actor state dict — "
            f"got keys {list(stripped_sd.keys())}"
        )
    linear_indices = sorted(set(linear_indices))

    # Pull (out, in) shapes per Linear layer in order.
    shapes: List[Tuple[int, int]] = []
    for idx in linear_indices:
        w = stripped_sd[f"{idx}.weight"]
        if w.ndim != 2:
            raise ValueError(
                f"Expected 2-D Linear weight at index {idx}, got shape {tuple(w.shape)}"
            )
        shapes.append((int(w.shape[0]), int(w.shape[1])))

    # Consistency: out_dim[i] must equal in_dim[i+1].
    for i in range(len(shapes) - 1):
        if shapes[i][0] != shapes[i + 1][1]:
            raise ValueError(
                f"Linear layer shape mismatch between index {linear_indices[i]} "
                f"(out={shapes[i][0]}) and {linear_indices[i + 1]} "
                f"(in={shapes[i + 1][1]}) — state dict is inconsistent"
            )

    obs_dim = shapes[0][1]
    action_dim = shapes[-1][0]
    hidden_dims = [out for out, _ in shapes[:-1]]
    return obs_dim, hidden_dims, action_dim


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def export_rsl_rl_actor_to_onnx(
    checkpoint_path: Path,
    agent_yaml_path: Path,
    onnx_out_path: Path,
) -> Dict[str, Any]:
    """Convert an rsl_rl actor checkpoint to a self-contained ``policy.onnx``.

    Returns a small dict with metadata used by the importer:
    ``{obs_dim, action_dim, hidden_dims, activation}``.

    Raises
    ------
    FileNotFoundError
        If the checkpoint is missing.
    ValueError
        If the checkpoint is neither a standard rsl_rl ``OnPolicyRunner``
        save nor an AMP ``AMPOnPolicyRunner`` save, or if its tensor
        shapes are internally inconsistent.
    """
    checkpoint_path = Path(checkpoint_path)
    agent_yaml_path = Path(agent_yaml_path)
    onnx_out_path = Path(onnx_out_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    # ── 1. Read activation (and opt. obs_normalization) from agent.yaml ─
    # Hidden dims are intentionally NOT read from here — see module
    # docstring. The AMP-PPO path writes an agent.yaml reflecting Isaac
    # Lab's default MLPModel config, not the runner that actually ran,
    # so trusting it would produce a wrong-sized MLP.
    activation = "elu"
    obs_norm = False
    if agent_yaml_path.is_file():
        import yaml
        try:
            from src.system.skill.isaac_lab_manifest_parser import _IsaacLabEnvYamlLoader
            with agent_yaml_path.open("r", encoding="utf-8") as fh:
                agent = yaml.load(fh, Loader=_IsaacLabEnvYamlLoader)
        except Exception:
            with agent_yaml_path.open("r", encoding="utf-8") as fh:
                agent = yaml.safe_load(fh)
        if isinstance(agent, dict):
            actor_cfg = agent.get("actor") or {}
            activation = str(actor_cfg.get("activation") or "elu")
            obs_norm = bool(actor_cfg.get("obs_normalization", False))
    else:
        log.warning(
            "agent.yaml not found at %s — defaulting activation to 'elu'",
            agent_yaml_path,
        )

    if obs_norm:
        log.warning(
            "rsl_rl checkpoint at %s has obs_normalization=True; the exported "
            "ONNX does NOT include the normalization layer. UnitPort's runtime "
            "Normalizer must load the matching stats from normalization_stats.json.",
            checkpoint_path,
        )

    # ── 2. Load checkpoint and extract the actor MLP state dict ───────
    import torch

    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    try:
        sd_clean, fmt = _extract_actor_mlp_state_dict(ckpt)
    except ValueError as exc:
        raise ValueError(f"{checkpoint_path} {exc}") from exc

    # ── 3. Infer dims directly from tensor shapes ─────────────────────
    obs_dim, hidden_dims, action_dim = _infer_dims_from_stripped_sd(sd_clean)
    log.info(
        "Detected %s actor: obs=%d hidden=%s action=%d activation=%s",
        fmt, obs_dim, hidden_dims, action_dim, activation,
    )

    # ── 4. Build MLP and load weights ─────────────────────────────────
    mlp = _build_actor_mlp(obs_dim, action_dim, hidden_dims, activation)
    missing, unexpected = mlp.load_state_dict(sd_clean, strict=True)
    if missing or unexpected:
        # ``strict=True`` raises before this point if anything is off, but
        # leave the warning in case torch ever silently relaxes that.
        log.warning(
            "MLP load_state_dict reported missing=%s unexpected=%s — "
            "exported ONNX may be incomplete.",
            missing,
            unexpected,
        )
    mlp.eval()

    # ── 5. Export to ONNX ─────────────────────────────────────────────
    onnx_out_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, obs_dim, dtype=torch.float32)
    with torch.no_grad():
        torch.onnx.export(
            mlp,
            dummy,
            str(onnx_out_path),
            input_names=["obs"],
            output_names=["action"],
            dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
            opset_version=17,
            do_constant_folding=True,
        )

    log.info(
        "Exported rsl_rl actor → ONNX: %s (obs=%d action=%d hidden=%s activation=%s)",
        onnx_out_path,
        obs_dim,
        action_dim,
        hidden_dims,
        activation,
    )

    return {
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "hidden_dims": hidden_dims,
        "activation": activation,
    }
