# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Bridge: phase_2 MotionClip → AmpOnPolicyRunner ``amp_data`` contract.

AMPOnPolicyRunner expects its ``amp_data`` object to expose:

- ``.observation_dim`` (int) — size of the concatenated ``(s_t, s_{t+1})``
  input to the discriminator divided by 2.
- ``.feed_forward_generator(num_mini_batch, mini_batch_size)`` yielding
  tuples ``(expert_state, expert_next_state)`` of torch tensors on the
  runner's device.
- ``.preload_transitions(num_preload_transitions)`` — optional. The
  vendored ``amp_on_policy_runner`` calls this via the
  ``preload_transitions=True`` runner flag. We provide it as a no-op
  when transitions were already materialized at ``build_amp_data`` time
  (which is the common case), and as a real preload pass otherwise.

The bridge keeps motion data on CPU as float32 numpy until the
generator is called, then moves minibatches to the target device. This
is both memory-friendly for long mocap datasets and test-friendly
(tests can run on CPU without a CUDA toolchain).

Per AMP_design.yaml §4.amp_backend.data_provider.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from application.training.motion.clip import MotionClip
from application.training.motion.loaders import (
    detect_motion_format,
    get_loader,
    mirror_motion_clip,
)

#: DoF of the canonical quadruped AMP layout. Sagittal-mirror augmentation
#: (``mirror_motion_clip``) only has a 12-DoF FL/FR/RL/RR table today, so
#: it is applied only to clips at this DoF; humanoid clips skip it with a
#: loud one-time notice (biped sagittal mirror is a separate feature, and
#: a silent skip would hide the missing augmentation).
_QUADRUPED_MIRROR_DOF = 12


# ---------------------------------------------------------------------------
# MotionAmpData — the object AmpOnPolicyRunner consumes
# ---------------------------------------------------------------------------


@dataclass
class MotionAmpData:
    """AMP data source for the vendored ``AmpOnPolicyRunner``.

    Constructed from one or more ``MotionClip`` objects (typically by
    ``build_amp_data`` or ``build_amp_data_from_files``). The important
    invariants:

    - All clips must share the same ``amp_obs_dim`` (we reject the
      build at mixing time).
    - At least one clip must have ``has_amp_payload() == True``
      (tracking-only clips cannot produce discriminator transitions).

    ``term_names`` is the ordered list of AMP obs term names every
    ``sample_transitions`` call against the wrapped clips will request.
    It is required (no default) so the caller MUST plumb a concrete
    family-keyed term list resolved via
    :func:`registers.amp_obs_templates.get_obs_template`. Refusing to
    accept ``None`` here mirrors the same refusal at
    :meth:`MotionClip._sample_amp_obs_batch` and catches a missing
    plumb at the constructor instead of deep inside disc training.
    """

    clips: List[MotionClip]
    device: torch.device
    transition_dt: float
    #: Ordered AMP obs term names this data source emits. Tuple shape
    #: locks immutability — the term order is load-bearing for
    #: field-order alignment between motion and env sides.
    term_names: Tuple[str, ...] = field(default_factory=tuple)
    #: Clip-level sampling probabilities, derived from MotionClip.motion_weight.
    clip_weights: np.ndarray = field(default_factory=lambda: np.zeros(0))
    #: Cache of preloaded transitions; populated lazily when the
    #: vendored runner calls ``preload_transitions``.
    _preloaded: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    #: Dimension reported to the discriminator constructor.
    _obs_dim: int = 0
    #: Per-tag sub-buffers built by ``initialize_sub_buffers``. Keyed by
    #: int tag id (from ``tag_router.TagRegistry``). When populated,
    #: ``sample_tagged`` routes each row of the returned expert batch to
    #: the sub-buffer matching that row's tag id. When ``None``, sampling
    #: falls back to the legacy uniform path — preserving backward
    #: compat for code paths that don't set up a registry.
    sub_buffers: Optional[Dict[int, "MotionAmpData"]] = None
    #: Mirror of ``tag_router.TagRegistry.id_to_name`` for log messages.
    #: Not used for sampling.
    _tag_names: Tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.clips:
            raise ValueError("MotionAmpData requires at least one MotionClip")
        # Normalise term_names to a tuple so the immutability contract
        # holds regardless of what the caller passed in (list / tuple /
        # generator). An empty tuple is rejected — the caller MUST
        # provide the family-keyed term list explicitly.
        self.term_names = tuple(self.term_names) if self.term_names else tuple()
        if not self.term_names:
            raise ValueError(
                "MotionAmpData: term_names must be non-empty. The caller "
                "is responsible for resolving the AMP obs term list via "
                "registers.amp_obs_templates.get_obs_template(family).template_terms "
                "and threading it through build_amp_data / "
                "build_amp_data_from_files. Refusing to substitute a "
                "hardcoded quadruped default."
            )
        for tn in self.term_names:
            if not isinstance(tn, str) or not tn.strip():
                raise ValueError(
                    f"MotionAmpData: every term_names entry must be a "
                    f"non-empty string; got {tn!r} in {self.term_names!r}"
                )
        dims = {c.amp_obs_dim for c in self.clips}
        if len(dims) != 1:
            raise ValueError(
                f"MotionAmpData: clips have inconsistent amp_obs_dim: {dims}. "
                f"All clips must share the same AMP observation layout."
            )
        obs_dim = next(iter(dims))
        if obs_dim <= 0:
            raise ValueError(
                "MotionAmpData: clips do not carry an AMP payload "
                "(amp_obs_dim == 0). Use format_id='amp_legged_gym'."
            )
        self._obs_dim = int(obs_dim)

        weights = np.array(
            [max(1e-6, float(c.motion_weight)) for c in self.clips],
            dtype=np.float64,
        )
        weights /= weights.sum()
        self.clip_weights = weights

        if self.transition_dt <= 0:
            raise ValueError(
                f"MotionAmpData.transition_dt must be > 0, got {self.transition_dt}"
            )

    # ------------------------------------------------------------------
    # AmpOnPolicyRunner contract
    # ------------------------------------------------------------------

    @property
    def observation_dim(self) -> int:
        """Size of one ``s_t`` observation (NOT the concatenated pair)."""
        return self._obs_dim

    @property
    def num_motions(self) -> int:
        return len(self.clips)

    def preload_transitions(
        self,
        num_preload_transitions: int,
        *,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        """Materialize a large batch of transitions up-front.

        Mirrors ``AMPLoader.preload_transitions`` in the vendored
        ``motion_loader.py``. Useful for runs where the clip set is
        small enough to fit in memory and you want per-minibatch
        sampling to be a pure index lookup.
        """
        n = int(num_preload_transitions)
        if n <= 0:
            self._preloaded = None
            return
        s_np, s_next_np = self._sample_np(n, rng=rng)
        self._preloaded = (
            torch.as_tensor(s_np, dtype=torch.float32, device=self.device),
            torch.as_tensor(s_next_np, dtype=torch.float32, device=self.device),
        )

    def feed_forward_generator(
        self,
        num_mini_batch: int,
        mini_batch_size: int,
        *,
        rng: Optional[np.random.Generator] = None,
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Yield ``num_mini_batch`` minibatches of expert transitions.

        If ``preload_transitions`` was called earlier, samples come from
        the cached tensor (much faster, deterministic with a seeded rng).
        Otherwise each minibatch re-samples from the raw clips.
        """
        if rng is None:
            rng = np.random.default_rng()

        for _ in range(int(num_mini_batch)):
            if self._preloaded is not None:
                s_cache, s_next_cache = self._preloaded
                total = s_cache.shape[0]
                idxs = rng.integers(0, total, size=int(mini_batch_size))
                idxs_t = torch.as_tensor(idxs, dtype=torch.long, device=self.device)
                yield s_cache.index_select(0, idxs_t), s_next_cache.index_select(0, idxs_t)
            else:
                s_np, s_next_np = self._sample_np(int(mini_batch_size), rng=rng)
                yield (
                    torch.as_tensor(s_np, dtype=torch.float32, device=self.device),
                    torch.as_tensor(s_next_np, dtype=torch.float32, device=self.device),
                )

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Command-conditioned sampling (per-tag sub-buffers)
    # ------------------------------------------------------------------

    def initialize_sub_buffers(
        self,
        task_items: Sequence[Any],
        tag_registry: Any,
        *,
        num_preload_transitions: int = 0,
    ) -> None:
        """Partition ``self.clips`` by task and build per-tag sub-buffers.

        Each sub-buffer is itself a ``MotionAmpData`` containing only the
        clips bound to one task (via ``task_item.clip_ref`` or
        ``task_item.motion_tag`` — see
        :func:`application.training.amp.tag_router.partition_clips_by_task_item`).
        After this call, ``sample_tagged(tag_ids, n)`` routes rows to
        the matching sub-buffer; tags with no clips fall back to global
        sampling (and emit a one-time warning).

        If ``num_preload_transitions > 0``, each sub-buffer preloads
        that many transitions. Total memory ≈ N_tags × N_preload ×
        obs_dim × 2 × 4B (fp32). For 3 tags × 200K × 43 × 2 × 4B ≈
        200 MB — fine on any reasonable GPU.
        """
        from application.training.amp.tag_router import (
            partition_clips_by_task_item,
            UNKNOWN_TAG_ID,
        )

        groups = partition_clips_by_task_item(self.clips, task_items)
        if not groups:
            # No task binding worked — keep sub_buffers None so legacy
            # uniform sampling is used.
            print(
                "[UnitPort][AMP][WARN] initialize_sub_buffers: no "
                "task_item produced a non-empty clip group. Falling "
                "back to global uniform sampling. Check that "
                "task_items.clip_ref / motion_tag match loaded clips.",
                flush=True,
            )
            return

        sub: Dict[int, MotionAmpData] = {}
        for t_id_name, clip_list in groups.items():
            tag_id = int(tag_registry.get_id(t_id_name))
            if tag_id == UNKNOWN_TAG_ID:
                continue
            sub[tag_id] = build_amp_data(
                clip_list,
                transition_dt=self.transition_dt,
                term_names=self.term_names,
                device=self.device,
            )
            if num_preload_transitions > 0:
                sub[tag_id].preload_transitions(num_preload_transitions)

        self.sub_buffers = sub
        self._tag_names = tuple(tag_registry.id_to_name)

        summary = {
            tag_registry.id_to_name[tid]: len(sb.clips)
            for tid, sb in sub.items()
        }
        print(
            f"[UnitPort][AMP] initialized per-tag sub-buffers: {summary} "
            f"(preload={num_preload_transitions} per tag)",
            flush=True,
        )

    def sample_tagged(
        self,
        tag_ids: "torch.Tensor | np.ndarray",
        *,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample ``(state, next_state)`` rows aligned to ``tag_ids``.

        For each row ``i``, the returned expert transition is drawn from
        ``self.sub_buffers[tag_ids[i]]``. Rows whose tag_id is unknown
        (-1) OR whose sub-buffer is empty fall back to the global
        (all-clips) sampler.

        Returns two tensors, each shape ``(len(tag_ids), obs_dim)``, on
        ``self.device``. Output dtype float32.

        This replaces ``feed_forward_generator``'s per-minibatch sampling
        when the caller has per-transition tag info (which policy
        rollouts always do once the runner threads tags through).
        """
        from application.training.amp.tag_router import UNKNOWN_TAG_ID

        if rng is None:
            rng = np.random.default_rng()

        if isinstance(tag_ids, torch.Tensor):
            tag_np = tag_ids.detach().cpu().numpy().astype(np.int64)
        else:
            tag_np = np.asarray(tag_ids, dtype=np.int64)

        n = int(tag_np.shape[0])
        s_out = np.empty((n, self._obs_dim), dtype=np.float32)
        s_next_out = np.empty((n, self._obs_dim), dtype=np.float32)

        # Legacy path — no sub-buffers or caller passed all-unknown.
        if not self.sub_buffers:
            s_out, s_next_out = self._sample_np(n, rng=rng)
            return (
                torch.as_tensor(s_out, dtype=torch.float32, device=self.device),
                torch.as_tensor(s_next_out, dtype=torch.float32, device=self.device),
            )

        # Group row indices by tag so each sub-buffer is hit once.
        unique_tags, inverse = np.unique(tag_np, return_inverse=True)
        fallback_mask = np.zeros(n, dtype=bool)
        for t_idx, tag_id in enumerate(unique_tags):
            row_idxs = np.where(inverse == t_idx)[0]
            k = len(row_idxs)
            if k == 0:
                continue
            sub = self.sub_buffers.get(int(tag_id)) if tag_id != UNKNOWN_TAG_ID else None
            if sub is None:
                fallback_mask[row_idxs] = True
                continue
            s_np, s_next_np = sub._sample_np(k, rng=rng)
            s_out[row_idxs] = s_np
            s_next_out[row_idxs] = s_next_np

        # Fallback: rows with unknown tag or missing sub-buffer — pull
        # from the full pool rather than fail.
        fb_count = int(fallback_mask.sum())
        if fb_count > 0:
            s_fb, s_next_fb = self._sample_np(fb_count, rng=rng)
            s_out[fallback_mask] = s_fb
            s_next_out[fallback_mask] = s_next_fb

        return (
            torch.as_tensor(s_out, dtype=torch.float32, device=self.device),
            torch.as_tensor(s_next_out, dtype=torch.float32, device=self.device),
        )

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _sample_np(
        self, n: int, *, rng: Optional[np.random.Generator] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Draw ``n`` transitions across clips weighted by motion_weight.

        Each transition is sourced from a single clip (sampled with
        replacement proportionally to ``clip_weights``) and uses
        ``MotionClip.sample_transitions`` to draw ``(s_t, s_{t+dt})``.
        """
        if rng is None:
            rng = np.random.default_rng()

        clip_idxs = rng.choice(
            len(self.clips), size=n, replace=True, p=self.clip_weights
        )
        # Bucket by clip so each clip gets one vectorised sample_transitions call.
        s_out = np.empty((n, self._obs_dim), dtype=np.float32)
        s_next_out = np.empty((n, self._obs_dim), dtype=np.float32)

        for ci, clip in enumerate(self.clips):
            mask = clip_idxs == ci
            k = int(mask.sum())
            if k == 0:
                continue
            s_t, s_tp1 = clip.sample_transitions(
                k, self.transition_dt,
                term_names=self.term_names, rng=rng,
            )
            s_out[mask] = s_t
            s_next_out[mask] = s_tp1
        return s_out, s_next_out


# ---------------------------------------------------------------------------
# Public factories
# ---------------------------------------------------------------------------


def build_amp_data(
    clips: Sequence[MotionClip],
    *,
    transition_dt: float,
    term_names: Sequence[str],
    device: Union[str, torch.device] = "cpu",
) -> MotionAmpData:
    """Wrap an existing list of ``MotionClip`` objects into a ``MotionAmpData``.

    Use this when the caller has already loaded and validated clips (as
    in unit tests). The launcher typically uses
    :func:`build_amp_data_from_files` instead.

    ``term_names`` is keyword-only required and forwarded to the
    :class:`MotionAmpData` constructor, which carries it through every
    sampling call so the wrapped clips emit the canonical family-keyed
    AMP obs vector layout.
    """
    dev = torch.device(device)
    return MotionAmpData(
        clips=list(clips),
        device=dev,
        transition_dt=float(transition_dt),
        term_names=tuple(term_names),
    )


def build_amp_data_from_files(
    paths: Sequence[Union[Path, str]],
    *,
    transition_dt: float,
    robot_sku: str,
    robot_family: str,
    term_names: Sequence[str],
    device: Union[str, torch.device] = "cpu",
    format_id: str = "",
    ranges: Optional[Sequence[Optional[Tuple[int, int]]]] = None,
    task_filter: str = "",
    task_label_overrides: Optional[dict] = None,
    mirror_augment: bool = True,
) -> MotionAmpData:
    """Load + validate + wrap motion files in one call.

    Used by the launcher's AMP branch. The loader for each file is
    selected from its extension via :func:`detect_motion_format` (so a
    canvas may mix robot families — Go2 ``.txt`` and G1 ``.csv`` resolve
    to ``amp_legged_gym`` and ``lafan_unitree_g1`` respectively), unless
    an explicit ``format_id`` override is passed (applied to every file).
    A tracking-only format (``unitport_npy``, ``amp_obs_dim == 0``) is
    rejected downstream by :class:`MotionAmpData` — AMP needs a clip that
    carries a discriminator payload.

    Parameters
    ----------
    format_id:
        Optional loader override. Empty (default) → per-file detection
        by extension. Non-empty → that loader is used for all files
        (e.g. forcing ``amp_legged_gym`` for a degenerate-geometry
        fixture). The pre-2026-06 quadruped-only hard gate is gone:
        humanoid AMP formats (``lafan_unitree_g1``) are first-class.

    robot_sku, robot_family:
        Canonical robot SKU + primary family slug for the policy this
        AMP dataset is being prepared for. Forwarded to
        :meth:`AMPLegGymLoader.load` so every produced
        ``ReferenceMotionContract`` records the target robot. Both
        are keyword-only and required — the loader's contract
        (``__post_init__``) refuses empty strings; the caller must
        plumb explicit values from the subprocess CLI / canvas
        compile, never synthesise an empty placeholder.
    term_names:
        Ordered AMP obs term names the wrapped clips will emit. Resolved
        by the caller via
        :func:`registers.amp_obs_templates.get_obs_template(family).template_terms`
        (typically once at launcher boot, then threaded into both this
        data-builder and the env-side AMP obs extractor so the
        discriminator's expert and policy distributions share field
        order). Keyword-only required — there is no hardcoded
        ``DEFAULT_QUADRUPED_TERMS`` fallback at this boundary.
    task_filter:
        When non-empty, only clips whose ``task_tag`` matches this
        value are kept. Tags are resolved via the motion label system
        (manifest file + filename auto-inference). This prevents the
        discriminator from learning a mixed distribution of unrelated
        gaits — e.g. filtering to ``"trot"`` ensures only trot clips
        feed the discriminator.

        Comma-separated values are supported for multi-tag selection
        (e.g. ``"walk,trot"`` keeps both walk and trot clips).
    task_label_overrides:
        Optional ``{filename_stem: tag}`` dict from the canvas-side
        tag table editor.  Overrides both auto-inference and manifest
        labels for the specified clips.
    ranges:
        Optional per-file inclusive frame sub-range, positionally aligned
        to ``paths``. Each element is ``(lo, hi)`` → the loaded clip is
        sliced to that segment via :meth:`MotionClip.subrange` before
        tagging / validation / mirroring, or ``None`` → whole file. When
        provided its length must equal ``len(paths)`` (fail-loud, §8). This
        is how a Clip Motion Editor "segment" (a sub-range of a larger
        capture) reaches the discriminator without a separate clip file.
    """
    if not paths:
        raise ValueError("build_amp_data_from_files: at least one file is required")
    if ranges is not None and len(ranges) != len(paths):
        raise ValueError(
            f"build_amp_data_from_files: ranges has {len(ranges)} entries but "
            f"{len(paths)} files were given — must be positionally aligned."
        )

    from application.training.motion.labels import resolve_task_tags
    from application.training.motion.validator import validate_clip

    # Resolve task tags for all files (manifest + auto-inference)
    resolved_paths = [Path(p).resolve() for p in paths]
    tag_map = resolve_task_tags(resolved_paths)

    # Apply canvas-side label overrides on top of manifest + inference
    overrides = task_label_overrides or {}

    explicit_fmt = str(format_id or "").strip()
    clips: List[MotionClip] = []
    _mirror_skip_warned = False
    for _idx, p in enumerate(resolved_paths):
        # Per-file loader dispatch: explicit override wins, else detect
        # from the extension. The loader returns a ReferenceMotionContract;
        # AMP consumes only the wrapped MotionClip payload. target_sku /
        # target_family are required keyword-only — propagated from the
        # caller's explicit ``robot_sku`` / ``robot_family`` parameters.
        fid = explicit_fmt or detect_motion_format(p)
        loader = get_loader(fid)
        clip = loader.load(
            p, target_sku=robot_sku, target_family=robot_family,
        ).clip
        # Segment slice (before tag/validate/mirror so everything downstream
        # sees exactly the selected frames). Out-of-range fails loud (§8).
        seg = ranges[_idx] if ranges is not None else None
        if seg is not None:
            clip = clip.subrange(int(seg[0]), int(seg[1]))
        # Priority: canvas override > manifest > auto-inferred (from loader)
        canvas_tag = overrides.get(clip.name, "")
        manifest_tag = tag_map.get(str(p), "")
        if canvas_tag:
            clip.task_tag = canvas_tag
        elif manifest_tag:
            clip.task_tag = manifest_tag
        validate_clip(clip)
        clips.append(clip)
        if mirror_augment:
            if clip.dof == _QUADRUPED_MIRROR_DOF:
                mirrored = mirror_motion_clip(clip)
                validate_clip(mirrored)
                clips.append(mirrored)
            elif not _mirror_skip_warned:
                # Loud, once-per-build: mirror augmentation has only a
                # 12-DoF quadruped table. Skipping for other morphologies
                # is honest (the user sees it) — NOT a silent fallback.
                print(
                    f"[UnitPort][AMP] mirror augmentation skipped for "
                    f"{clip.dof}-DoF clips (sagittal mirror table is "
                    f"quadruped-only today; biped mirror is a separate "
                    f"feature). Training proceeds without mirrored copies.",
                    flush=True,
                )
                _mirror_skip_warned = True

    # Apply task filter
    filter_str = (task_filter or "").strip()
    if filter_str:
        allowed_tags = {t.strip().lower() for t in filter_str.split(",") if t.strip()}
        filtered = [c for c in clips if c.task_tag.lower() in allowed_tags]
        if not filtered:
            available_tags = sorted({c.task_tag for c in clips if c.task_tag})
            raise ValueError(
                f"build_amp_data_from_files: task_filter={filter_str!r} "
                f"matched 0 of {len(clips)} clips. "
                f"Available tags: {available_tags or ['(no tags inferred — add motion_labels.yaml)']}"
            )
        print(
            f"[UnitPort] Motion task filter '{filter_str}': "
            f"kept {len(filtered)}/{len(clips)} clips "
            f"(dropped: {[c.name for c in clips if c not in filtered]})",
            flush=True,
        )
        clips = filtered

    return build_amp_data(
        clips, transition_dt=transition_dt,
        term_names=term_names, device=device,
    )


# ---------------------------------------------------------------------------
# RSI (Reference-State Initialization) pool builder
# ---------------------------------------------------------------------------


_RSI_REQUIRED_FIELDS: Tuple[str, ...] = (
    "root_pos",
    "root_quat",
    "joint_pos",
    "joint_vel",
    "lin_vel",
    "ang_vel",
)


def build_rsi_pool(
    clips: Sequence[MotionClip],
    n_frames: int = 20000,
    *,
    rng: Optional[np.random.Generator] = None,
    require_full_state: bool = True,
    random_start_phase: bool = True,
) -> Dict[str, np.ndarray]:
    """Sample ``n_frames`` full-state snapshots across ``clips`` into a flat
    numpy pool consumable by :func:`application.training.amp.mdp_events.register_rsi_pool`.

    Each row carries the six fields needed to reset an Isaac Lab
    articulation back onto the expert manifold: root pose + velocity
    (world frame) and joint pos + vel in the clip's canonical IR-role
    order (quadruped FL/FR/RL/RR × hip/thigh/calf from the amp-legged-gym
    loader; humanoid 29-DoF IR order from the lafan_unitree_g1 loader).
    The RSI reset event maps these columns to physical joints via the IR
    table, so the pool is morphology-agnostic.

    Root quaternion is rolled from MotionClip's ``(x, y, z, w)`` to
    Isaac Lab's ``(w, x, y, z)`` at build time so the event fn can skip
    the conversion on the hot path.

    Parameters
    ----------
    clips:
        Motion clips to sample from. Must share a common joint count
        (all-12 for quadruped AMP today).
    n_frames:
        Pool size. 20k covers ~400K envs at 4% replacement per iter.
    rng:
        Optional ``np.random.Generator``. Omit for fresh seeding.
    require_full_state:
        When True (default), raise ``ValueError`` on any clip missing
        one of the six required arrays. When False, drop incomplete
        clips silently — raise only if zero clips remain.
    """
    if not clips:
        raise ValueError("build_rsi_pool: at least one clip is required")
    if n_frames <= 0:
        raise ValueError(f"build_rsi_pool: n_frames must be >= 1, got {n_frames}")

    clips_used: List[MotionClip] = []
    for clip in clips:
        missing = [f for f in _RSI_REQUIRED_FIELDS if getattr(clip, f, None) is None]
        if missing:
            if require_full_state:
                raise ValueError(
                    f"build_rsi_pool: clip {clip.name!r} missing required "
                    f"field(s) {missing} for RSI. Pass require_full_state=False "
                    f"to skip incomplete clips."
                )
            continue
        clips_used.append(clip)

    if not clips_used:
        raise ValueError(
            "build_rsi_pool: no clips have the full state required for RSI "
            "(root_pos, root_quat, joint_pos, joint_vel, lin_vel, ang_vel)."
        )

    dofs = {c.dof for c in clips_used}
    if len(dofs) != 1:
        raise ValueError(
            f"build_rsi_pool: clips disagree on DoF count: {dofs}. "
            f"RSI demands a uniform joint layout across all clips."
        )
    dof = next(iter(dofs))
    if dof <= 0:
        raise ValueError(
            f"build_rsi_pool: clips report dof={dof}; need a positive joint "
            f"count. The per-clip arrays below are sized from this value, so "
            f"it must reflect the real morphology (12 quadruped / 29 G1 / …)."
        )

    if rng is None:
        rng = np.random.default_rng()

    weights = np.array(
        [max(1e-6, float(c.motion_weight)) for c in clips_used],
        dtype=np.float64,
    )
    weights /= weights.sum()
    clip_idxs = rng.choice(len(clips_used), size=int(n_frames), replace=True, p=weights)

    root_pos_out = np.empty((n_frames, 3), dtype=np.float32)
    root_quat_xyzw_out = np.empty((n_frames, 4), dtype=np.float32)
    lin_vel_out = np.empty((n_frames, 3), dtype=np.float32)
    ang_vel_out = np.empty((n_frames, 3), dtype=np.float32)
    joint_pos_out = np.empty((n_frames, dof), dtype=np.float32)
    joint_vel_out = np.empty((n_frames, dof), dtype=np.float32)

    # Bucket sampled frame indices per clip so each clip only pays one
    # random-draw call, matching _sample_np's pattern.
    for ci, clip in enumerate(clips_used):
        mask = clip_idxs == ci
        k = int(mask.sum())
        if k == 0:
            continue
        upper = max(clip.duration_s - clip.frame_dt, 0.0)
        # training_motion.random_start_phase=False pins every sampled
        # snapshot to the clip's frame 0 (the static "rest" pose). True
        # (default) draws uniformly across the clip duration — the legacy
        # behaviour that AMP RSI was built for.
        if random_start_phase:
            times = rng.uniform(0.0, upper if upper > 0.0 else 1e-9, size=k)
        else:
            times = np.zeros(k, dtype=np.float64)
        row_idxs = np.where(mask)[0]
        for row, t in zip(row_idxs, times):
            frame = clip.frame_at(float(t))
            root_pos_out[row] = frame["root_pos"]
            root_quat_xyzw_out[row] = frame["root_quat"]
            lin_vel_out[row] = frame["lin_vel"]
            ang_vel_out[row] = frame["ang_vel"]
            joint_pos_out[row] = frame["joint_pos"]
            joint_vel_out[row] = frame["joint_vel"]

    # np.roll(q, shift=1, axis=-1) with input [x, y, z, w] returns
    # [w, x, y, z] — last element lands at position 0, everything else
    # shifts right. Verified: [0.1, 0.2, 0.3, 0.4] -> [0.4, 0.1, 0.2, 0.3].
    root_quat_wxyz_out = np.roll(root_quat_xyzw_out, 1, axis=-1).astype(np.float32)

    # RSI sanity filter. A corrupt mocap frame (NaN mid-clip, root z below
    # the contact radius, or mocap artifact producing >10 m/s root velocity)
    # written directly to the sim produces huge contact impulses and NaN
    # physics on step 0 — which then poisons the value function and the
    # policy gradient cascades into a ValueError at ~iter 230. We reject
    # such frames here and resample replacements from the known-good subset
    # so the pool size stays at n_frames.
    invalid = (
        np.isnan(root_pos_out).any(axis=1)
        | np.isnan(root_quat_wxyz_out).any(axis=1)
        | np.isnan(lin_vel_out).any(axis=1)
        | np.isnan(ang_vel_out).any(axis=1)
        | np.isnan(joint_pos_out).any(axis=1)
        | np.isnan(joint_vel_out).any(axis=1)
        | np.isinf(root_pos_out).any(axis=1)
        | np.isinf(lin_vel_out).any(axis=1)
        | np.isinf(ang_vel_out).any(axis=1)
        | np.isinf(joint_vel_out).any(axis=1)
        | (root_pos_out[:, 2] < 0.15)
        | (root_pos_out[:, 2] > 1.5)
        | (np.abs(lin_vel_out) > 10.0).any(axis=1)
        | (np.abs(ang_vel_out) > 20.0).any(axis=1)
        | (np.abs(joint_vel_out) > 30.0).any(axis=1)
    )
    n_invalid = int(invalid.sum())
    if n_invalid > 0:
        valid_mask = ~invalid
        n_valid = int(valid_mask.sum())
        if n_valid == 0:
            raise ValueError(
                "build_rsi_pool: every sampled frame failed the RSI sanity "
                "filter (NaN / z<0.15 / |vel|>limits). Motion clips look "
                "corrupt — inspect the source data before retraining."
            )
        valid_idx = np.where(valid_mask)[0]
        repl = rng.choice(valid_idx, size=n_invalid, replace=True)
        root_pos_out[invalid] = root_pos_out[repl]
        root_quat_wxyz_out[invalid] = root_quat_wxyz_out[repl]
        lin_vel_out[invalid] = lin_vel_out[repl]
        ang_vel_out[invalid] = ang_vel_out[repl]
        joint_pos_out[invalid] = joint_pos_out[repl]
        joint_vel_out[invalid] = joint_vel_out[repl]
        print(
            f"[UnitPort][AMP][RSI] filtered {n_invalid}/{n_frames} bad frames "
            f"(NaN / z out of [0.15, 1.5] / |vel| over limits) — replaced "
            f"with {n_invalid} resampled valid frames. "
            f"Pool root_z stats: min={float(root_pos_out[:,2].min()):.3f} "
            f"max={float(root_pos_out[:,2].max()):.3f} "
            f"mean={float(root_pos_out[:,2].mean()):.3f}",
            flush=True,
        )

    return {
        "root_pos": root_pos_out,
        "root_quat_wxyz": root_quat_wxyz_out,
        "root_lin_vel_w": lin_vel_out,
        "root_ang_vel_w": ang_vel_out,
        "joint_pos_canonical": joint_pos_out,
        "joint_vel_canonical": joint_vel_out,
    }
