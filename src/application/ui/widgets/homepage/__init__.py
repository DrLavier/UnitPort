"""Homepage — four-card empty-state landing page.

Replaces the legacy ``NewCanvasForm`` as the picker child of ``_MainPanel``
when no canvas is loaded. The four cards are:

* :class:`UserCard`           — auth state (signed-in / signed-out CTAs)
* :class:`NewsCard`            — community news (static placeholder)
* :class:`HomepageLocalFiles`  — multi-user canvas-file table
* :class:`HomepageCreateCanvas`— project name + template grid + Create CTA

See :class:`HomepagePage` for the container and signal contract.
"""

from __future__ import annotations

from .page import HomepagePage

__all__ = ["HomepagePage"]
