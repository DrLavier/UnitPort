# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Self-contained custom-terrain asset in a deploy bundle (Step 4).

A trained policy that ran on a custom terrain must be reviewable /
re-runnable on that SAME terrain on any machine, so the heightfield rides
*inside* the bundle (CLAUDE.md §9 self-contained) — the consumer reads
only the bundle, never reaching back to ``USER_CONFIG_DIR`` or the
training run dir.

Producer (`build_terrain_bundle_payload`)
    Runs the cross-engine consistency gate (Step 3) **before any bytes are
    emitted** — same discipline as the PD calibration gate (§10): a
    terrain whose two engines disagree never reaches disk. Then serialises
    the canonical heightfield to a portable ``.npz`` and returns the
    bundle-relative path + bytes + a manifest fragment carrying provenance
    (sha256, size, vertical_scale, ordering). The caller drops the bytes
    into the bundle via ``BundleExporter``'s ``extra_files`` mechanism and
    the fragment under the manifest's ``terrain`` key.

Consumer (`load_terrain_from_bundle`)
    Resolves the asset strictly *within* the bundle dir (path-escape
    guarded — no ``../`` reach-back), loads it via the canonical npz
    loader, and re-verifies the sha256 against the manifest fragment so a
    tampered/edited bundle fails loud (§8).
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

from application.training.terrain.consistency import check_cross_engine_consistency
from application.training.terrain.contract import (
    TERRAIN_SCHEMA_VERSION,
    TerrainContract,
    TerrainContractError,
    validate_terrain_contract,
)
from application.training.terrain.isaaclab_lowering import DEFAULT_VERTICAL_SCALE
from application.training.terrain.loaders.npz import NpzHeightFieldLoader


class TerrainBundleError(RuntimeError):
    """Raised when a terrain asset cannot be packaged into / loaded from a
    bundle (consistency gate failure, missing file, digest mismatch,
    path escape)."""


#: Default bundle-relative location for the terrain asset.
DEFAULT_TERRAIN_REL = "terrain/terrain.npz"


def serialize_heightfield_npz_bytes(contract: TerrainContract) -> bytes:
    """Serialise the canonical heightfield to portable ``.npz`` bytes.

    Layout matches :class:`NpzHeightFieldLoader` (keys ``heights`` /
    ``size_x`` / ``size_y`` / ``array_ordering``) so the consumer reuses
    the same loader the importer used.
    """
    hf = contract.height_field
    buf = io.BytesIO()
    np.savez(
        buf,
        heights=np.ascontiguousarray(hf.heights, dtype=np.float32),
        size_x=np.float64(hf.size_x),
        size_y=np.float64(hf.size_y),
        array_ordering=np.asarray(hf.array_ordering),
    )
    return buf.getvalue()


def build_terrain_bundle_payload(
    contract: TerrainContract,
    *,
    vertical_scale: float = DEFAULT_VERTICAL_SCALE,
    rel_path: str = DEFAULT_TERRAIN_REL,
    run_consistency_gate: bool = True,
    consistency_tol: Optional[float] = None,
) -> Dict[str, Any]:
    """Package a terrain for a bundle. Returns ``{rel_path, npz_bytes,
    manifest_fragment}``.

    ``vertical_scale`` is recorded in the fragment so the IsaacLab
    consumer rebuilds the int16 grid identically to training. Raises
    :class:`application.training.terrain.consistency.TerrainConsistencyError`
    (before producing any bytes) when the two engines disagree.
    """
    validate_terrain_contract(contract)  # re-assert before shipping (§8)
    hf = contract.height_field

    if run_consistency_gate:
        # Fail-loud BEFORE emitting bytes — never write a bundle whose two
        # engines would train/review on different ground.
        check_cross_engine_consistency(
            hf, vertical_scale=vertical_scale, tol=consistency_tol,
            raise_on_fail=True,
        )

    npz_bytes = serialize_heightfield_npz_bytes(contract)
    rel_clean = str(rel_path).strip().lstrip("/\\")
    if not rel_clean:
        raise TerrainBundleError("build_terrain_bundle_payload: rel_path is empty")

    fragment = {
        "schema_version": TERRAIN_SCHEMA_VERSION,
        "file": rel_clean,
        "format": "heightfield_npz",
        "sha256": contract.source_info.sha256,
        "size_x": float(hf.size_x),
        "size_y": float(hf.size_y),
        "array_ordering": hf.array_ordering,
        "vertical_scale": float(vertical_scale),
        "n_rows": int(hf.n_rows),
        "n_cols": int(hf.n_cols),
    }
    return {
        "rel_path": rel_clean,
        "npz_bytes": npz_bytes,
        "manifest_fragment": fragment,
    }


def embed_custom_terrain(
    spec: Any,
    manifest_dict: Dict[str, Any],
) -> Optional[Dict[str, bytes]]:
    """Embed a custom heightfield into a bundle (shared IL + SB3 producer).

    When ``spec.scene.custom`` is enabled: load the canonical ``.npz``, run
    the cross-engine consistency gate (BEFORE any bytes — §8/§10), write the
    provenance fragment into ``manifest_dict["terrain"]``, and return
    ``{rel_path: npz_bytes}`` for the exporter's ``extra_files``. Returns
    ``None`` when there is no custom terrain. Fail-loud — never ships a
    custom-scene bundle without its terrain.
    """
    custom = getattr(getattr(spec, "scene", None), "custom", None)
    if custom is None or not bool(getattr(custom, "enabled", False)):
        return None
    src = str(getattr(custom, "source_path", "") or "").strip()
    if not src:
        raise TerrainBundleError(
            "embed_custom_terrain: scene_type='custom' but custom terrain "
            "source_path is empty — cannot embed terrain in bundle (§8)."
        )
    from application.training.terrain.loaders.npz import NpzHeightFieldLoader

    contract = NpzHeightFieldLoader().load(src)  # validates + sha256
    payload = build_terrain_bundle_payload(
        contract,
        vertical_scale=float(getattr(custom, "vertical_scale", 0.005) or 0.005),
    )
    manifest_dict["terrain"] = payload["manifest_fragment"]
    return {payload["rel_path"]: payload["npz_bytes"]}


def load_terrain_from_bundle(
    bundle_dir: Union[Path, str],
    fragment: Dict[str, Any],
) -> TerrainContract:
    """Load the bundle's terrain asset. Reads ONLY within ``bundle_dir``.

    Path-escape guarded (§9 — a portable bundle must not reach back into
    local-only state) and sha256-verified against ``fragment`` (§8).
    """
    if not isinstance(fragment, dict):
        raise TerrainBundleError(
            f"load_terrain_from_bundle: fragment must be a dict, got "
            f"{type(fragment).__name__!r}"
        )
    rel = str(fragment.get("file", "")).strip()
    if not rel:
        raise TerrainBundleError(
            "load_terrain_from_bundle: manifest terrain fragment has no 'file'."
        )

    root = Path(bundle_dir).resolve()
    target = (root / rel).resolve()
    # Reject any path that escapes the bundle root (../, absolute, symlink).
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise TerrainBundleError(
            f"load_terrain_from_bundle: terrain file {rel!r} escapes the "
            f"bundle dir {root} — refusing to reach outside the bundle (§9)."
        ) from exc
    if not target.is_file():
        raise TerrainBundleError(
            f"load_terrain_from_bundle: terrain asset not found in bundle: "
            f"{target}"
        )

    contract = NpzHeightFieldLoader().load(target)

    expected = str(fragment.get("sha256", "")).strip()
    if expected and contract.source_info.sha256 != expected:
        raise TerrainBundleError(
            f"load_terrain_from_bundle: terrain sha256 mismatch — manifest "
            f"{expected!r} != asset {contract.source_info.sha256!r}. The "
            f"bundle terrain was modified after export (§8)."
        )
    return contract


__all__ = [
    "TerrainBundleError",
    "DEFAULT_TERRAIN_REL",
    "serialize_heightfield_npz_bytes",
    "build_terrain_bundle_payload",
    "embed_custom_terrain",
    "load_terrain_from_bundle",
]
