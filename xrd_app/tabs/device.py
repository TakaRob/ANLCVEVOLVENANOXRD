"""Device View tab — wraps the device feature map."""

from __future__ import annotations

from ..gui import device_map
from ._embed import LineageCatalogTab

TAB_META = {
    "title": "Device View",
    "order": 50,
    "takes_bin_size": True,
    "scan_dependent": True,
    "general": (
        "Spatial maps across the full bin grid with switchable metrics: integrated "
        "intensity, 2θ deviation (Δ2θ from the reference Bragg angle), chi "
        "orientation, azimuthal breadth (χ FWHM), and radial breadth (Δ2θ FWHM). "
        "Chi-range filter with interactive controls."
    ),
}


def make_tab(project_root=".", scan=None, bin_size=3):
    return LineageCatalogTab(device_map.build_window, project_root, scan=scan, bin_size=bin_size)


if __name__ == "__main__":
    from ._standalone import run_standalone
    run_standalone(make_tab, TAB_META["title"])
