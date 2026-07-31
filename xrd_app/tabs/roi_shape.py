"""ROI > Shape tab: manually constrain peak finding, then build normal shapes."""

from __future__ import annotations

from ..gui.roi_shape import build_window
from ._embed import embed_window

TAB_META = {
    "title": "ROI > Shape",
    "order": 45,
    "takes_bin_size": True,
    "scan_dependent": True,
    "general": (
        "Select one reflection feature on the fully summed detector image (or a "
        "single spatial-bin image). Each detector rectangle becomes one pending "
        "manual feature; its total detector counts are measured across every "
        "spatial bin to make the heatmap. Saved outputs stay in a dedicated manual "
        "ROI catalog and are not connected to Shape/Verify."
    ),
}


def make_tab(project_root=".", scan=None, bin_size=3):
    return embed_window(build_window(project_root, scan=scan, bin_size=bin_size,
                                     embedded=True))


if __name__ == "__main__":
    from ._standalone import run_standalone
    run_standalone(make_tab, TAB_META["title"])
