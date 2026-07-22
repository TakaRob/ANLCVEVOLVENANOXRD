"""HD Device View tab — 1×1 intensity beneath the N×N feature map."""

from __future__ import annotations

from ..gui import hd_device_map
from ._embed import BinnedTab

TAB_META = {
    "title": "HD Device View",
    "order": 55,
    "takes_bin_size": True,
    "scan_dependent": True,
    "general": (
        "The binned feature map's outlines over a true 1×1 heatmap: raw intensity "
        "at each feature's detector peak, sampled per unbinned pixel — the real "
        "scan with all its holes and 9× finer detail, so nearly-overlapping "
        "reflections separate. Switch to a real (x, y) stage-position scatter to "
        "see the actual scan geometry. If the HD map hasn't been built for this "
        "bin yet, the tab shows a “Build HD device map” button that runs it "
        "in-app (with an (i/n) progress status) and swaps in the view when done."
    ),
}


def make_tab(project_root=".", scan=None, bin_size=3):
    return BinnedTab(hd_device_map.build_window, project_root, scan=scan,
                     bin_size=bin_size)


if __name__ == "__main__":
    from ._standalone import run_standalone
    run_standalone(make_tab, TAB_META["title"])
