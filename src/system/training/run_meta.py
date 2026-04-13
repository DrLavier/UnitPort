"""UnitPort training-run metadata file — single source of truth for
"was this run AMP, PPO, or something else?"

H2 in ``knowledge_base/amp_fix.yaml``. Per spec §4 the compiler should
emit ``algorithm_class: 'AMP_PPO'`` when the canvas used the AMP
trainer node, and per spec §5 the run directory should carry a
``run.yaml`` that downstream tooling can consume without having to
parse Isaac Lab's auto-generated ``params/agent.yaml`` (which lies
about the runner class for AMP runs — see the ONNX exporter incident).

Schema version 1 — the minimum fields the bundle exporter, policy
runner, and ONNX exporter need to dispatch AMP-aware paths:

    unitport_run_meta_version: 1
    algorithm_class: "AMP_PPO" | "PPO" | "SAC" | ...
    created_utc: ISO-8601 string
    amp:                       # present iff algorithm_class == "AMP_PPO"
      amp_reward_coef: float
      task_reward_lerp: float
      num_preload_transitions: int
      transition_dt: float
      num_amp_obs_history: int
      amp_obs_terms: list[str]
      amp_obs_dim: int
      dataset_files: list[{path, basename}]
    loss:                      # provenance
      discriminator_loss: str
      style_reward_formula: str

Phase 5 (H1 / B6) will extend this with ``spec_hash`` (sha256 of the
run.yaml or of the compiled env_cfg file) and per-dataset sha256s so
the bundle exporter can verify consistency at export time.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

RUN_META_FILENAME = "unitport_run_meta.yaml"
RUN_META_VERSION = 1
#: Key under which the self-excluding sha256 lives inside the meta dict.
SPEC_HASH_KEY = "spec_hash"


def read_run_meta(run_dir: Any) -> Optional[Dict[str, Any]]:
    """Load ``unitport_run_meta.yaml`` from a run directory.

    Returns the parsed dict, or ``None`` when the file is missing /
    unreadable / malformed. Callers should use ``detect_algorithm_class``
    rather than inspecting the dict directly unless they need AMP-specific
    fields.
    """
    path = Path(run_dir) / RUN_META_FILENAME
    if not path.is_file():
        return None
    try:
        import yaml
    except ImportError:
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def detect_algorithm_class(run_dir: Any, default: str = "PPO") -> str:
    """Return the algorithm class for *run_dir*.

    Resolution order:
      1. ``unitport_run_meta.yaml::algorithm_class`` (authoritative when present)
      2. Heuristic: presence of ``discriminator_state_dict`` in the newest
         ``model_*.pt`` checkpoint → ``AMP_PPO``
      3. Fall back to *default* (``PPO``).

    The heuristic exists so runs created before the ``run_meta`` file
    was introduced still get classified correctly. Every new AMP run
    should emit the meta file, so the heuristic is strictly a
    back-compat path.
    """
    meta = read_run_meta(run_dir)
    if meta is not None:
        algo = meta.get("algorithm_class")
        if isinstance(algo, str) and algo:
            return algo

    # Heuristic fallback: peek at the newest checkpoint for AMP-only
    # keys. We only do this for ``model_*.pt`` files so a stray
    # ``latest.pt`` symlink on Windows (which may not resolve) does
    # not break detection.
    run_path = Path(run_dir)
    try:
        ckpts = sorted(run_path.glob("model_*.pt"))
    except OSError:
        ckpts = []
    if ckpts:
        try:
            import torch  # local: optional dep for the heuristic path
            ckpt = torch.load(
                str(ckpts[-1]), map_location="cpu", weights_only=False
            )
            if isinstance(ckpt, dict) and "discriminator_state_dict" in ckpt:
                return "AMP_PPO"
        except Exception:
            pass
    return default


def write_run_meta(run_dir: Any, meta: Dict[str, Any]) -> Path:
    """Serialize *meta* to ``unitport_run_meta.yaml`` under *run_dir*.

    Overwrites any existing file. The caller is expected to supply a
    dict already populated per the schema — ``write_run_meta`` does
    not fill defaults.

    H1 (AMP fix plan): if *meta* does not already carry a ``spec_hash``
    field, one is computed via :func:`compute_spec_hash` and injected
    before serialization. The hash deliberately excludes itself so
    ``read_run_meta`` → ``compute_spec_hash`` → compare is a fixed
    point. This gives the bundle exporter a cheap tamper check: if
    anyone hand-edits the file after training finishes, the recomputed
    hash no longer matches and export can abort.
    """
    import yaml
    if SPEC_HASH_KEY not in meta:
        meta = {**meta, SPEC_HASH_KEY: compute_spec_hash(meta)}
    path = Path(run_dir) / RUN_META_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.dump(
            meta, fh, default_flow_style=False, sort_keys=False,
            allow_unicode=True,
        )
    return path


# ---------------------------------------------------------------------------
# H1 — spec_hash
# ---------------------------------------------------------------------------


def compute_spec_hash(meta: Dict[str, Any]) -> str:
    """Return the canonical sha256 of *meta*, ignoring any existing
    ``spec_hash`` field.

    The hash input is the sorted-keys YAML dump of the dict with
    ``spec_hash`` removed — so the hash is stable regardless of how
    the YAML was originally formatted (key order, indentation, …)
    and the function is a fixed point under re-hashing.
    """
    import yaml
    scrubbed = {k: v for k, v in meta.items() if k != SPEC_HASH_KEY}
    blob = yaml.dump(
        scrubbed, sort_keys=True, default_flow_style=False, allow_unicode=True
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def verify_spec_hash(meta: Dict[str, Any]) -> Tuple[bool, str]:
    """Check whether *meta*'s embedded ``spec_hash`` matches its content.

    Returns ``(ok, message)``. ``ok`` is ``True`` when the embedded hash
    matches the recomputed one. ``message`` carries a human-readable
    reason when ``ok`` is ``False``.

    Missing ``spec_hash`` is treated as a verification failure with a
    distinct message so callers can distinguish "file was truncated"
    from "file was edited".
    """
    recorded = meta.get(SPEC_HASH_KEY)
    if not isinstance(recorded, str) or not recorded:
        return False, f"no {SPEC_HASH_KEY!r} field present in run meta"
    expected = compute_spec_hash(meta)
    if recorded != expected:
        return False, (
            f"{SPEC_HASH_KEY} mismatch — recorded={recorded[:12]}… "
            f"expected={expected[:12]}…; meta file was edited after "
            f"training finished"
        )
    return True, "ok"


def sha256_file(path: Any, chunk: int = 1 << 20) -> str:
    """Stream-hash *path* with sha256, returning the hex digest.

    Used by the bundle exporter to fill ``dataset_files[*].sha256`` at
    publish time. 1 MiB chunks keep memory bounded on big motion files.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            buf = fh.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()
