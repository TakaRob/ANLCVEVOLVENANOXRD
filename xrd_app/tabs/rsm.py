"""Reciprocal Space (RSM) tab — 3D-map projections + per-grain feature cloud.

Thin wrapper over :class:`xrd_app.gui.rsm_view.RSMView`; all data prep lives in
``core/rsm.py``. Reads the project's ``Study/rsm.npz`` (from ``xrd-app rsm``) and
``Study/qspace/*_features_q.csv`` (from ``xrd-app qspace``).
"""

from __future__ import annotations

from ..gui.rsm_view import RSMView

TAB_META = {
    "title": "Reciprocal Space",
    "order": 70,
    "takes_bin_size": False,
    "scan_dependent": False,
    "general": (
        "Fused reciprocal-space map of the θ series. Shows max-intensity "
        "projections of the 3D RSM (Study/rsm.npz) on true q-axes, overlaid with "
        "the per-grain feature cloud (Study/qspace/*_features_q.csv) colored by "
        "reflection, θ, or intensity. Radial |Q| = strain; transverse spread = "
        "tilt. Build the data with `xrd-app qspace` then `xrd-app rsm`."
    ),
}


def make_tab(project_root=".", scan=None, bin_size=3):
    return RSMView(project_root, scan=scan, bin_size=bin_size)


if __name__ == "__main__":
    from ._standalone import run_standalone
    run_standalone(make_tab, TAB_META["title"])
