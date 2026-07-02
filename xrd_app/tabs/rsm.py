"""Reciprocal Space (RSM) tab — 2D projections + 3D volume + per-grain cloud.

Thin wrapper over :class:`xrd_app.gui.rsm_view.RSMView`; all data prep lives in
``core/rsm.py`` and ``core/studies.py``. A study selector picks which analysis to
show (any dir with ``rsm.npz`` / ``qspace/*_features_q.csv``); the view toggles
between 2D max-projections and a 3D volume render (adjustable resolution).
"""

from __future__ import annotations

from ..gui.rsm_view import RSMView

TAB_META = {
    "title": "Reciprocal Space",
    "order": 70,
    "takes_bin_size": False,
    "scan_dependent": False,
    "general": (
        "Fused reciprocal-space map of the θ series. Pick a study, then view "
        "either 2D max-intensity projections of the 3D RSM (rsm.npz) on true "
        "q-axes, or the full 3D volume (adjustable resolution + opacity). Both "
        "overlay the per-grain feature cloud (qspace/*_features_q.csv) colored by "
        "reflection, θ, or intensity. The 3D view also draws reflection rings "
        "(concentric |Q| shells the cloud sits on) and optional per-scan rings. "
        "Radial |Q| = strain; transverse spread = tilt. Build the data with "
        "`xrd-app qspace` then `xrd-app rsm` (or `xrd-app run-study --with-rsm`). "
        "3D needs PyOpenGL (`xrd-app[gl]`)."
    ),
}


def make_tab(project_root=".", scan=None, bin_size=3):
    return RSMView(project_root, scan=scan, bin_size=bin_size)


if __name__ == "__main__":
    from ._standalone import run_standalone
    run_standalone(make_tab, TAB_META["title"])
