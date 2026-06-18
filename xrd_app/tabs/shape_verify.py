"""Shape/Verification tab — wraps the spatial feature viewer."""

from __future__ import annotations

from ..gui import viewer
from ._embed import embed_with_toolbar

TAB_META = {
    "title": "Shape/Verify",
    "order": 40,
    "takes_bin_size": True,
    "scan_dependent": True,
    "general": (
        "Review kept (gaussian-like) and filtered shapes from the shape-finding "
        "stage. A 'shape' is a peak that persists across neighboring bins with a "
        "gaussian-like spatial profile; metrics include rocking-curve FWHM, strain "
        "breadth, and chi angle."
    ),
}


def make_tab(project_root=".", scan=None, bin_size=3):
    win = viewer.build_window(project_root, scan=scan, bin_size=bin_size)

    def save_algo():
        from .save_algorithm_dialog import SaveAlgorithmDialog
        SaveAlgorithmDialog(project_root, bin_size=bin_size, kind="shape", parent=win).exec_()

    return embed_with_toolbar(win, [("Save Algorithm…", save_algo)])


if __name__ == "__main__":
    from ._standalone import run_standalone
    run_standalone(make_tab, TAB_META["title"])
