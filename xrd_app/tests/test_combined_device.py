"""Headless tests for the combined device-view data layer (core/combined_device.py)."""
from __future__ import annotations

import numpy as np

from xrd_app.core import combined_device as cd


def _row(scan, reflection, r, c, integrated, intensity=None):
    return {"scan": scan, "reflection": reflection, "row": r, "col": c,
            "integrated": integrated, "intensity": intensity if intensity is not None else integrated}


def test_fuses_max_and_argmax_theta():
    rows = [
        _row("Scan_0213", "(001)", 2, 3, 50.0),    # θ=0.0
        _row("Scan_0203", "(001)", 2, 3, 200.0),   # θ=20.5  ← strongest here
        _row("Scan_0210", "(001)", 2, 3, 120.0),   # θ=4.0
        _row("Scan_0210", "(002)", 0, 0, 80.0),    # θ=4.0 other band/bin
    ]
    c = cd.build_combined(rows, log=lambda *_: None)
    assert (c["n_rows"], c["n_cols"]) == (3, 4)
    assert c["max_intensity"][2, 3] == 200.0
    assert c["argmax_theta"][2, 3] == 20.5         # θ of the strongest
    assert c["n_theta_present"][2, 3] == 3         # lit at 3 distinct θ
    # (002) is a separate layer
    assert set(c["reflections"]) == {"(001)", "(002)"}
    k2 = c["reflections"].index("(002)")
    assert c["layer_intensity"][k2, 0, 0] == 80.0
    assert c["layer_argmax_theta"][k2, 0, 0] == 4.0


def test_intensity_key_switch():
    rows = [_row("Scan_0210", "(001)", 1, 1, integrated=500.0, intensity=30.0)]
    ci = cd.build_combined(rows, intensity_key="intensity", log=lambda *_: None)
    assert ci["max_intensity"][1, 1] == 30.0
    cg = cd.build_combined(rows, intensity_key="integrated", log=lambda *_: None)
    assert cg["max_intensity"][1, 1] == 500.0


def test_unknown_scan_skipped():
    rows = [_row("Scan_9999", "(001)", 1, 1, 100.0),
            _row("Scan_0210", "(001)", 1, 1, 60.0)]
    c = cd.build_combined(rows, log=lambda *_: None)
    assert c["max_intensity"][1, 1] == 60.0        # 9999 dropped (no θ)
    assert c["n_theta_present"][1, 1] == 1


def test_track_overlay_and_summary():
    rows = [_row("Scan_0210", "(001)", 1, 1, 100.0)]
    tracks = [{"track_id": 0, "reflection": "(001)", "centroid_row": 1.0,
               "centroid_col": 1.0, "theta_at_max_I": 4.0, "is_recurrent": True}]
    c = cd.build_combined(rows, tracks=tracks, log=lambda *_: None)
    assert c["tracks"]["track_id"] == [0]
    s = cd.summary(c)
    assert s["n_tracks"] == 1 and s["n_recurrent_tracks"] == 1
    assert s["bins_with_signal"] == 1


def test_save_npz_roundtrip(tmp_path):
    rows = [_row("Scan_0203", "(001)", 0, 0, 200.0),
            _row("Scan_0210", "(002)", 1, 1, 90.0)]
    tracks = [{"track_id": 5, "reflection": "(001)", "centroid_row": 0.0,
               "centroid_col": 0.0, "theta_at_max_I": 20.5, "is_recurrent": True}]
    c = cd.build_combined(rows, tracks=tracks, log=lambda *_: None)
    p = cd.save_npz(tmp_path / "combined_device.npz", c)
    assert p.exists()
    z = np.load(p, allow_pickle=False)
    assert int(z["n_rows"]) == 2 and int(z["n_cols"]) == 2
    assert z["max_intensity"][0, 0] == 200.0
    assert list(z["reflections"]) == ["(001)", "(002)"]
    assert int(z["track_id"][0]) == 5
    assert bool(z["track_is_recurrent"][0]) is True
