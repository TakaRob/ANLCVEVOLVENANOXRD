"""Territorial Map tab — the skew-free cell-model device map + shape inspector.

Wraps :func:`xrd_app.gui.territory_map.build_window`. Renders the variable-
footprint territories (true (X, Y) polygons) of the reference binning and lets
you inspect the kept shapes linked across physical neighbors. When the reference
hasn't been built for the scan, the tab shows a "Build territorial reference"
button that runs the chain in-app and swaps in the map when done.
"""

from __future__ import annotations

from PyQt5.QtWidgets import QVBoxLayout, QWidget

from ..gui import territory_map
from ._embed import placeholder

TAB_META = {
    "title": "Territory Map",
    "order": 56,
    "takes_bin_size": False,
    "scan_dependent": True,
    "general": (
        "Source-of-truth (cell-model) device map: frames are binned by true "
        "(X, Y) stage positions into irregular territories — immune to the "
        "serpentine backlash that skews the N×N grid — and drawn to-scale as "
        "polygons coloured by frame count, cell area, or a shape's per-territory "
        "intensity. Use it as the reference to optimize the fast skew fix against. "
        "If the reference isn't built for this scan, a “Build territorial "
        "reference” button runs it in-app (with an (i/n) progress status)."
    ),
}


class TerritoryTab(QWidget):
    """Embed the territorial map, or the in-tab builder when it's not built yet.

    Rebuilds itself when the builder reports a successful build (``on_built``),
    so the map swaps in without a manual tab reload.
    """

    def __init__(self, project_root, scan=None):
        super().__init__()
        self._project_root = project_root
        self._scan = scan
        self._win = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self._rebuild()

    def _rebuild(self):
        if self._win is not None:
            self._win.setParent(None)
            self._win.deleteLater()
            self._win = None
        try:
            self._win = territory_map.build_window(
                self._project_root, scan=self._scan, on_built=self._rebuild)
        except Exception as e:
            self._win = placeholder("Could not load the Territory Map.",
                                    f"{type(e).__name__}: {e}")
        self.layout().addWidget(self._win)
        self._embedded_window = self._win   # keep alive; used by header sync


def make_tab(project_root=".", scan=None, bin_size=3):
    return TerritoryTab(project_root, scan=scan)


if __name__ == "__main__":
    from ._standalone import run_standalone
    run_standalone(make_tab, TAB_META["title"])
