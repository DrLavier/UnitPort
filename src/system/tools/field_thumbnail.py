"""field_thumbnail — offscreen MuJoCo render for MjFieldNode previews.

P6-T1 deliverable. Builds a small (configurable, default 256x256)
RGB preview of a field as composed against a default actor (go2),
returns it as a numpy array, and provides a tiny adapter that turns
the array into a Qt :class:`QImage` for the canvas to draw.

Decoupling
----------
The core renderer returns numpy arrays so the function is usable
from CLI / scripts / RobustnessMatrix without dragging Qt into the
import graph. The Qt adapter is a separate function that the canvas
calls. Tests can hit either.

Caching
-------
A small LRU cache keyed by ``(field_id, actor_id, size)`` keeps the
canvas snappy: re-opening the same MjFieldNode parameter editor
doesn't re-render. Cache eviction is FIFO (we don't expect more
than ~10 distinct fields in a session).
"""
from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional, Tuple

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

# Per-process cache: {(field_id, actor_id, height, width): np.ndarray}
_THUMBNAIL_CACHE: "OrderedDict[Tuple[str, str, int, int], Any]" = OrderedDict()
_CACHE_MAX = 16


def clear_cache() -> None:
    """Drop the in-process thumbnail cache."""
    _THUMBNAIL_CACHE.clear()


# ---------------------------------------------------------------------------
# Core render
# ---------------------------------------------------------------------------


def render_field_thumbnail(
    field_id: str,
    actor_id: str = "go2",
    size: Tuple[int, int] = (256, 256),
    project_root: Optional[Path] = None,
    settle_steps: int = 10,
) -> Any:
    """Build an offscreen RGB thumbnail of *field_id* composed against *actor_id*.

    Returns a uint8 numpy array of shape (height, width, 3). On any
    failure (mujoco missing, field not found, GL error) returns
    ``None`` so the canvas can fall back to a parameter-only display.

    The robot is settled for ``settle_steps`` frames so feet sit on
    the ground at render time — otherwise the thumbnail would show
    the keyframe pose floating mid-air.
    """
    height, width = int(size[0]), int(size[1])
    cache_key = (field_id, actor_id, height, width)

    cached = _THUMBNAIL_CACHE.get(cache_key)
    if cached is not None:
        # LRU touch — move to end.
        _THUMBNAIL_CACHE.move_to_end(cache_key)
        return cached

    try:
        import mujoco
        import numpy as np
    except ImportError as exc:
        log.warning("render_field_thumbnail: mujoco/numpy unavailable: %s", exc)
        return None

    try:
        from src.system.runtime.simulation.mujoco import (
            MjActor,
            MjSimEnv,
            load_field,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("render_field_thumbnail: runtime import failed: %s", exc)
        return None

    try:
        actor = MjActor.from_menagerie(actor_id)
        field = load_field(field_id, project_root=project_root)
        env = MjSimEnv.build(actor, field)
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "render_field_thumbnail: setup failed for field=%s actor=%s: %s",
            field_id, actor_id, exc,
        )
        return None

    try:
        env.reset()
        for _ in range(max(0, int(settle_steps))):
            env.sim_step()
    except Exception as exc:  # noqa: BLE001
        log.warning("render_field_thumbnail: settle failed: %s", exc)

    try:
        renderer = mujoco.Renderer(env.mj_model, height=height, width=width)
        try:
            renderer.update_scene(env.mj_data)
            pixels = renderer.render()
        finally:
            try:
                renderer.close()
            except Exception:
                pass
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "render_field_thumbnail: render failed for field=%s: %s",
            field_id, exc,
        )
        return None

    # Defensive copy so the cached buffer isn't tied to the renderer's
    # internal lifetime.
    try:
        result = np.array(pixels, copy=True, dtype=np.uint8)
    except Exception:
        result = pixels

    _THUMBNAIL_CACHE[cache_key] = result
    if len(_THUMBNAIL_CACHE) > _CACHE_MAX:
        # FIFO eviction (popitem(last=False) drops the oldest entry).
        _THUMBNAIL_CACHE.popitem(last=False)
    return result


# ---------------------------------------------------------------------------
# Qt adapter
# ---------------------------------------------------------------------------


def field_thumbnail_qimage(
    field_id: str,
    actor_id: str = "go2",
    size: Tuple[int, int] = (256, 256),
    project_root: Optional[Path] = None,
) -> Any:
    """Return a :class:`PySide6.QtGui.QImage` of the field thumbnail.

    Returns ``None`` when the renderer fails or when Qt is unavailable.
    The QImage owns its own pixel buffer (we copy from the numpy
    source) so callers can store it without keeping the array alive.
    """
    pixels = render_field_thumbnail(
        field_id=field_id,
        actor_id=actor_id,
        size=size,
        project_root=project_root,
    )
    if pixels is None:
        return None

    try:
        from PySide6.QtGui import QImage
    except ImportError:
        return None

    height, width = int(pixels.shape[0]), int(pixels.shape[1])
    # MuJoCo's Renderer.render() returns RGB (no alpha) in HWC order.
    bytes_per_line = width * 3
    try:
        # bytes() makes Qt own a copy of the buffer.
        image = QImage(
            bytes(pixels.tobytes()),
            width,
            height,
            bytes_per_line,
            QImage.Format_RGB888,
        )
        # .copy() detaches from the local buffer so it survives GC.
        return image.copy()
    except Exception as exc:  # noqa: BLE001
        log.warning("field_thumbnail_qimage: QImage construction failed: %s", exc)
        return None
