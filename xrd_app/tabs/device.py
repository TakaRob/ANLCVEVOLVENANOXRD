"""Device View tab — wraps the device feature map."""

from __future__ import annotations

from ..gui import device_map
from ._embed import embed_window

TAB_META = {
    "title": "Device View",
    "order": 50,
    "takes_bin_size": True,
    "scan_dependent": True,
    "general": (
        "Spatial maps across the full bin grid with switchable metrics: integrated "
        "intensity, lattice strain (Δ2θ), chi orientation, rocking width, and strain "
        "breadth. Chi-range filter with interactive controls."
    ),
}


def make_tab(project_root=".", scan=None, bin_size=3):
    return embed_window(device_map.build_window(project_root, scan=scan, bin_size=bin_size))


if __name__ == "__main__":
    from ._standalone import run_standalone
    run_standalone(make_tab, TAB_META["title"])
