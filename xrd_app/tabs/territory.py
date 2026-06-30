"""Territorial Map tab — the skew-free cell-model device map + shape inspector.

Wraps :func:`xrd_app.gui.territory_map.build_window`. Renders the variable-
footprint territories (true (X, Y) polygons) of the reference binning and lets
you inspect the kept shapes linked across physical neighbors. Shows a build-
instructions placeholder until ``xrd-app territory-grid`` (+ bin/peaks/shapes
``--variant territory``) has been run for the scan.
"""

from __future__ import annotations

from ..gui import territory_map

TAB_META = {
    "title": "Territory Map",
    "order": 55,
    "takes_bin_size": False,
    "scan_dependent": True,
    "general": (
        "Source-of-truth (cell-model) device map: frames are binned by true "
        "(X, Y) stage positions into irregular territories — immune to the "
        "serpentine backlash that skews the N×N grid — and drawn to-scale as "
        "polygons coloured by frame count, cell area, or a shape's per-territory "
        "intensity. Use it as the reference to optimize the fast skew fix against."
    ),
}


def make_tab(project_root=".", scan=None, bin_size=3):
    return territory_map.build_window(project_root, scan=scan, bin_size=bin_size)


if __name__ == "__main__":
    from ._standalone import run_standalone
    run_standalone(make_tab, TAB_META["title"])
