"""Shared colour palette helpers for the pyqtgraph GUIs.

Colormap list, reflection arc colours, and the small lookup helpers used by both
the main viewer and the manual reflection editor live here so the two stay in
sync (same colormaps, same arc colours).
"""

from __future__ import annotations

import pyqtgraph as pg
from PyQt5.QtGui import QColor

COLORMAPS = [
    "inferno", "viridis", "plasma", "magma", "cividis",
    "hot", "coolwarm", "gray", "jet", "turbo",
]

ARC_COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6",
]


def _get_cmap(name):
    """pyqtgraph ColorMap by name, falling back through matplotlib."""
    try:
        return pg.colormap.get(name)
    except Exception:
        try:
            return pg.colormap.get(name, source="matplotlib")
        except Exception:
            return pg.colormap.get("viridis")


def _hex_rgb(hex_color):
    c = QColor(hex_color)
    return c.red(), c.green(), c.blue()


# XRF element colours — a distinct palette (warm/element-y hues) so the XRF
# fluorescence layers read separately from the reflection outline palette. Shared
# by the Shape/Verify spectrum panel and the Device View XRF underlay so an
# element is the same colour everywhere.
ELEMENT_PALETTE = [
    "#ff5555", "#ffb000", "#ffee33", "#33dd55", "#33dddd",
    "#5599ff", "#b060ff", "#ff66cc", "#cc8844", "#88cc44",
]


def element_colors(elements):
    """Deterministic ``{element_name: hex}`` map over the element palette."""
    return {el: ELEMENT_PALETTE[i % len(ELEMENT_PALETTE)]
            for i, el in enumerate(elements)}


def hex_to_rgba(hex_color, alpha=255):
    """``(r, g, b, alpha)`` tuple from a hex colour string."""
    r, g, b = _hex_rgb(hex_color)
    return (r, g, b, int(alpha))
