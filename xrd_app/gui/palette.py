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
