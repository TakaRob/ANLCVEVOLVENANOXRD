"""Orientation Map tab — wraps the orientation/azimuthal cluster map."""

from __future__ import annotations

from ..gui import orientation as orientation_gui
from ._embed import BinnedTab

TAB_META = {
    "title": "Orientation Map",
    "order": 60,
    "takes_bin_size": True,
    "scan_dependent": True,
    "general": (
        "Adaptive sector chart on detector geometry. Finds natural clusters of "
        "features along each Bragg ring via Gaussian KDE with valley-finding, and "
        "paints each ring with a density gradient; hover for azimuthal / Δ2θ "
        "histograms."
    ),
}


def make_tab(project_root=".", scan=None, bin_size=3):
    return BinnedTab(orientation_gui.build_window, project_root, scan=scan, bin_size=bin_size)


if __name__ == "__main__":
    from ._standalone import run_standalone
    run_standalone(make_tab, TAB_META["title"])
