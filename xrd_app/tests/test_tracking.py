"""Headless tests for cross-θ track linking (core/tracking.py)."""
from __future__ import annotations

from xrd_app.core import tracking


def _feat(scan, reflection, row, col, chi, intensity, fid=0):
    return {
        "scan": scan, "reflection": reflection, "ref_tth": 7.514,
        "center_row": row, "center_col": col, "chi_deg": chi,
        "peak_intensity": intensity, "sum_integrated": intensity * 5,
        "detector_x": 100, "detector_y": 200, "n_bins": 4,
        "chi_fwhm": 0.2, "tth_fwhm": 0.05, "feature_id": fid,
    }


def test_recurrent_grain_links_across_theta():
    # Same grain (same reflection, ~same bin) lit up at 3 adjacent θ, drifting χ.
    features = [
        _feat("Scan_0210", "(001)", 10.0, 20.0, 30.0, 100),  # θ=4.0
        _feat("Scan_0211", "(001)", 10.4, 20.2, 31.0, 200),  # θ=3.5 (peak)
        _feat("Scan_0212", "(001)", 9.7, 19.8, 32.0, 120),   # θ=3.0
    ]
    tracks = tracking.build_tracks(features, match_tol=2.0, min_theta=2, log=lambda *_: None)
    assert len(tracks) == 1
    t = tracks[0]
    assert t["reflection"] == "(001)"
    assert t["n_theta"] == 3
    assert t["is_recurrent"] is True
    # Peak intensity is at θ=3.5 (the strongest member).
    assert t["theta_at_max_I"] == 3.5
    # χ drifts smoothly 30→31→32: span 2, max step 1.
    assert t["chi_span"] == 2.0
    assert t["chi_max_step"] == 1.0
    thetas, ints = tracking.intensity_curve(t)
    assert thetas == [3.0, 3.5, 4.0]


def test_far_apart_features_do_not_link():
    features = [
        _feat("Scan_0210", "(001)", 5.0, 5.0, 10.0, 100),
        _feat("Scan_0211", "(001)", 40.0, 40.0, 10.0, 100),  # far away
    ]
    tracks = tracking.build_tracks(features, match_tol=2.0, log=lambda *_: None)
    assert len(tracks) == 2
    assert all(t["n_theta"] == 1 for t in tracks)
    assert all(t["is_recurrent"] is False for t in tracks)


def test_different_reflections_never_merge():
    features = [
        _feat("Scan_0210", "(001)", 10.0, 20.0, 30.0, 100),
        _feat("Scan_0211", "(002)", 10.0, 20.0, 30.0, 100),  # same spot, other band
    ]
    tracks = tracking.build_tracks(features, match_tol=5.0, log=lambda *_: None)
    assert len(tracks) == 2
    assert {t["reflection"] for t in tracks} == {"(001)", "(002)"}


def test_unknown_scan_is_dropped():
    features = [
        _feat("Scan_9999", "(001)", 10.0, 20.0, 30.0, 100),  # not in θ table
        _feat("Scan_0210", "(001)", 10.0, 20.0, 30.0, 100),
    ]
    tracks = tracking.build_tracks(features, log=lambda *_: None)
    assert len(tracks) == 1
    assert tracks[0]["n_members"] == 1


def test_same_scan_collapsed_in_intensity_curve():
    # Two over-segmented shapes from the same scan at one θ → one curve point (max).
    features = [
        _feat("Scan_0210", "(001)", 10.0, 20.0, 30.0, 100, fid=0),
        _feat("Scan_0210", "(001)", 10.5, 20.5, 30.0, 150, fid=1),
        _feat("Scan_0211", "(001)", 10.2, 20.2, 31.0, 120, fid=2),
    ]
    tracks = tracking.build_tracks(features, match_tol=3.0, log=lambda *_: None)
    assert len(tracks) == 1
    t = tracks[0]
    assert t["n_members"] == 3
    assert t["n_theta"] == 2          # two distinct scans
    thetas, ints = tracking.intensity_curve(t)
    assert thetas == [3.5, 4.0]
    # θ=4.0 (Scan_0210) collapses to max integrated intensity 150*5=750.
    assert ints[thetas.index(4.0)] == 750.0


def test_unwrap_deg_handles_wraparound():
    # −179 → +179 is a +2° step the short way, not −358.
    assert tracking.unwrap_deg([-179.0, 179.0]) == [-179.0, -181.0]
    # smooth drift across the ±180 boundary stays small-step
    uw = tracking.unwrap_deg([170.0, 178.0, -176.0, -170.0])
    steps = [abs(b - a) for a, b in zip(uw, uw[1:])]
    assert max(steps) <= 10.0


def test_chi_span_is_wrap_aware():
    # A grain near the ±180 seam: raw span would be ~358°, wrapped span ~4°.
    feats = [
        _feat("Scan_0210", "(001)", 10.0, 20.0, 179.0, 100),   # θ=4.0
        _feat("Scan_0211", "(001)", 10.1, 20.1, -179.0, 100),  # θ=3.5
    ]
    tracks = tracking.build_tracks(feats, match_tol=2.0, log=lambda *_: None)
    assert len(tracks) == 1
    assert tracks[0]["chi_span"] <= 5.0          # not ~358
    assert tracks[0]["chi_max_step"] <= 5.0


def test_theta_table_has_eleven_usable_scans():
    usable = [s for s in tracking.THETA_BY_SCAN if s != "Scan_0206"]
    assert len(usable) == 11
    assert tracking.theta_of("Scan_0203") == tracking.theta_of("Scan_0214") == 20.5
