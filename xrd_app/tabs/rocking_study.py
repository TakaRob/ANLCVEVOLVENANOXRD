"""Rocking Study tab — run the θ-series pipeline and browse its results.

Thin wrapper over :class:`xrd_app.gui.rocking_view.RockingStudyView`. Study-level
(not per-scan): a study selector drives the Run controls and the rocking-curve /
combined-device / prediction-report viewers. All logic lives in ``core/studies``
and the ``xrd-app run-study`` command it shells out to.
"""

from __future__ import annotations

from ..gui.rocking_view import RockingStudyView

TAB_META = {
    "title": "Rocking Study",
    "order": 65,
    "takes_bin_size": True,
    "scan_dependent": False,
    "general": (
        "Cross-scan θ-series (rocking) analysis. Run the whole pipeline "
        "(aggregate → track → rocking → predict → combined-device, optionally "
        "qspace → rsm) from the Run tab; it registers each result set in "
        "studies.json so it appears in the study selectors here and in "
        "Reciprocal Space. Browse per-grain rocking curves (θ_Bragg, FWHM "
        "mosaicity), the fused-over-θ combined device map, and the prediction "
        "report (recall / precision / repeatability floor)."
    ),
}


def make_tab(project_root=".", scan=None, bin_size=3):
    return RockingStudyView(project_root, scan=scan, bin_size=bin_size)


if __name__ == "__main__":
    from ._standalone import run_standalone
    run_standalone(make_tab, TAB_META["title"])
