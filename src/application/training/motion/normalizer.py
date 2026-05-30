# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Reference-motion axis / convention normaliser — interface scaffold.

A motion clip authored in one coordinate convention (e.g.
``amp_legged_gym``'s z-up / x-forward / y-lateral / world-frame
velocities) is the *source* of truth on disk. The training target
(an Isaac Lab or MuJoCo robot) may expect a different convention.
This module is the single point where that frame translation will
happen.

Today only the **identity** branch is implemented:

* :func:`normalize_motion_clip` accepts ``override=None`` (no
  alignment override declared by the user) or an
  :class:`~application.training.motion.alignment.AlignmentOverride`
  whose ``axis_convention`` already matches the defaults declared
  in :mod:`application.training.motion.alignment`
  (:data:`DEFAULT_ROOT_UP_AXIS` / :data:`DEFAULT_ROOT_FORWARD_AXIS` /
  :data:`DEFAULT_ROOT_LATERAL_AXIS` /
  :data:`DEFAULT_LINEAR_VELOCITY_FRAME` /
  :data:`DEFAULT_ANGULAR_VELOCITY_FRAME` + empty
  ``joint_sign_flips``). In that branch the function returns the
  **same MotionClip instance it was given** — ``result is clip``
  holds, no copy, no field rewrite. This is the byte-identity
  guarantee loaders rely on.

* A non-identity override raises
  :exc:`MotionNormalizerNotImplementedError` with a clear directive
  pointing back at the override file's diverging field. This makes
  the unimplemented behaviour loud (per the project's no-silent-
  fallback rule); it never silently lets a non-default override
  through as if it had been applied.

Public surface
--------------
:exc:`MotionNormalizerNotImplementedError`
    Subclass of :class:`NotImplementedError`. Raised for every
    non-identity convention until the rewrite engine lands.
:func:`is_identity_convention`
    Pure predicate: ``True`` when an :class:`AxisConvention`
    matches every default (the criterion the identity branch
    accepts).
:func:`normalize_motion_clip`
    Loader-side entry. Returns the same clip on identity; raises on
    everything else.

Why this interface ships before the rewrite engine
--------------------------------------------------
The loader subpackage (``motion/loaders/``) is the single place a
motion file becomes a :class:`MotionClip`. Wiring the normaliser
into every loader **now**, while it is still a pure identity, means
that when the rewrite engine lands later there is no second pass to
revisit every loader's ``load()`` call. The integration surface is
fixed today; only the transform body grows.
"""
from __future__ import annotations

from typing import Optional

from application.training.motion.alignment import (
    DEFAULT_ANGULAR_VELOCITY_FRAME,
    DEFAULT_LINEAR_VELOCITY_FRAME,
    DEFAULT_ROOT_FORWARD_AXIS,
    DEFAULT_ROOT_LATERAL_AXIS,
    DEFAULT_ROOT_UP_AXIS,
    AlignmentOverride,
    AxisConvention,
)
from application.training.motion.clip import MotionClip


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class MotionNormalizerNotImplementedError(NotImplementedError):
    """Raised when an :class:`AlignmentOverride` declares a convention
    the identity-only normaliser cannot satisfy.

    The message names (a) the override file's source format and target
    SKU (so the caller can locate the override on disk), (b) the
    specific field whose value diverges from the loader-default
    convention, and (c) a concrete directive ("edit the override
    file" / "remove the override and retry") — the same three-element
    error shape used by
    :mod:`application.training.motion.alignment`.
    """


# ---------------------------------------------------------------------------
# Pure predicate
# ---------------------------------------------------------------------------


def is_identity_convention(convention: AxisConvention) -> bool:
    """Return ``True`` iff ``convention`` matches every default field
    (no axis flip, no velocity-frame re-projection, no per-joint sign
    flips). Used internally by :func:`normalize_motion_clip` and
    exposed so callers can decide whether a more expensive rewrite
    path needs to be taken.
    """
    return (
        convention.root_up_axis == DEFAULT_ROOT_UP_AXIS
        and convention.root_forward_axis == DEFAULT_ROOT_FORWARD_AXIS
        and convention.root_lateral_axis == DEFAULT_ROOT_LATERAL_AXIS
        and convention.linear_velocity_frame == DEFAULT_LINEAR_VELOCITY_FRAME
        and convention.angular_velocity_frame == DEFAULT_ANGULAR_VELOCITY_FRAME
        and not convention.joint_sign_flips
    )


# ---------------------------------------------------------------------------
# Loader-side entry
# ---------------------------------------------------------------------------


def normalize_motion_clip(
    clip: MotionClip,
    override: Optional[AlignmentOverride] = None,
) -> MotionClip:
    """Translate a clip from its source convention into the loader-default
    convention.

    Parameters
    ----------
    clip
        The :class:`MotionClip` a concrete loader has just constructed.
    override
        Optional :class:`AlignmentOverride` resolved by the loader
        (see :func:`application.training.motion.alignment.load_alignment_override`).
        ``None`` is the common case — no override on disk for this
        ``(source_format, target_sku)`` pair.

    Returns
    -------
    MotionClip
        On the identity branch (override is ``None`` or its
        ``axis_convention`` matches every default), the **same object
        that was passed in**: ``result is clip`` holds. The loader's
        byte-identical-output guarantee for the
        no-override-on-disk case rests on this.

    Raises
    ------
    MotionNormalizerNotImplementedError
        When ``override`` is non-``None`` and its
        :class:`AxisConvention` diverges from any default. The
        message identifies the override's
        ``(source_format, target_sku)`` and the first divergent
        field; the caller is directed either to edit the override
        to match the default convention or to remove the file and
        retry. Until the rewrite engine lands, no other resolution
        path is offered (loud failure beats silent mis-translation).
    """
    if override is None:
        return clip

    convention = override.axis_convention
    if is_identity_convention(convention):
        return clip

    diverging_field, diverging_value, expected_value = _first_divergence(
        convention
    )
    raise MotionNormalizerNotImplementedError(
        f"alignment override for source_format="
        f"{override.source_format!r}, target_sku="
        f"{override.target_sku!r} declares a non-identity "
        f"axis_convention: {diverging_field}={diverging_value!r} "
        f"(loader default is {expected_value!r}). The convention "
        f"rewrite engine has not landed yet; the identity-only "
        f"branch cannot fulfil this override. Either edit the "
        f"override file to match the loader default on every "
        f"axis_convention field, or remove the override and "
        f"retry — non-identity translation must not silently "
        f"succeed."
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _first_divergence(convention: AxisConvention):
    """Return ``(field_name, actual_value, expected_default)`` for the
    first :class:`AxisConvention` field that does not match the
    loader defaults. Iteration order is fixed to keep error messages
    reproducible across runs.
    """
    if convention.root_up_axis != DEFAULT_ROOT_UP_AXIS:
        return ("root_up_axis", convention.root_up_axis, DEFAULT_ROOT_UP_AXIS)
    if convention.root_forward_axis != DEFAULT_ROOT_FORWARD_AXIS:
        return (
            "root_forward_axis",
            convention.root_forward_axis,
            DEFAULT_ROOT_FORWARD_AXIS,
        )
    if convention.root_lateral_axis != DEFAULT_ROOT_LATERAL_AXIS:
        return (
            "root_lateral_axis",
            convention.root_lateral_axis,
            DEFAULT_ROOT_LATERAL_AXIS,
        )
    if convention.linear_velocity_frame != DEFAULT_LINEAR_VELOCITY_FRAME:
        return (
            "linear_velocity_frame",
            convention.linear_velocity_frame,
            DEFAULT_LINEAR_VELOCITY_FRAME,
        )
    if convention.angular_velocity_frame != DEFAULT_ANGULAR_VELOCITY_FRAME:
        return (
            "angular_velocity_frame",
            convention.angular_velocity_frame,
            DEFAULT_ANGULAR_VELOCITY_FRAME,
        )
    if convention.joint_sign_flips:
        return ("joint_sign_flips", dict(convention.joint_sign_flips), {})
    # Defensive — should not reach here because the caller already
    # verified is_identity_convention is False. Raise a clear
    # programmer-error if it ever does.
    raise AssertionError(
        "_first_divergence called on an axis_convention that already "
        "satisfies is_identity_convention; this is a logic error in "
        "the normaliser, not user input."
    )


__all__ = [
    "MotionNormalizerNotImplementedError",
    "is_identity_convention",
    "normalize_motion_clip",
]
