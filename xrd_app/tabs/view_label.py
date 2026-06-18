"""View/Label tab — wraps the interactive labeling GUI."""

from __future__ import annotations

from ..gui import labeling
from ._embed import embed_with_toolbar

TAB_META = {
    "title": "View/Label",
    "order": 30,
    "takes_bin_size": False,
    "scan_dependent": True,
    "general": (
        "Interactive per-bin viewer and Bragg-peak labeler. Navigate bins, toggle "
        "noise reduction, overlay a detector algorithm, adjust sensitivity, and "
        "annotate peaks. Human labels are stored per bin and can be marked "
        "'verified' for use as CVEvolve ground truth."
    ),
}


def make_tab(project_root=".", scan=None, bin_size=3):
    win = labeling.build_window(project_root, scan=scan, bin_size=bin_size)

    def save_algo():
        from .save_algorithm_dialog import SaveAlgorithmDialog
        SaveAlgorithmDialog(project_root, bin_size=bin_size, kind="peak", parent=win).exec_()

    return embed_with_toolbar(win, [("Save Algorithm…", save_algo)])


if __name__ == "__main__":
    from ._standalone import run_standalone
    run_standalone(make_tab, TAB_META["title"])
