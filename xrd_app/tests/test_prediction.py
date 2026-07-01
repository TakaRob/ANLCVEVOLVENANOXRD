"""Headless tests for predicted-vs-observed scoring (core/prediction.py)."""
from __future__ import annotations

from xrd_app.core import prediction, tracking


def _f(scan, reflection, row, col):
    return {"scan": scan, "reflection": reflection,
            "center_row": row, "center_col": col,
            "peak_intensity": 100.0, "sum_integrated": 500.0, "chi_deg": 30.0}


def test_repeatability_matches_same_orientation():
    # 203 and 214 (θ=20.5): two shapes each, one pair coincides, one offset within tol.
    features = [
        _f("Scan_0203", "(001)", 10.0, 20.0),
        _f("Scan_0203", "(002)", 30.0, 40.0),
        _f("Scan_0214", "(001)", 10.5, 20.4),   # matches 203's (001)
        _f("Scan_0214", "(002)", 33.0, 44.0),   # too far → no match
    ]
    rep = prediction.repeatability(features, "Scan_0203", "Scan_0214", match_tol=2.0)
    assert rep["n_a"] == 2 and rep["n_b"] == 2
    assert rep["matched"] == 1
    assert rep["reproducibility"] == round(2 * 1 / 4, 4)


def test_repeatability_respects_reflection_band():
    features = [
        _f("Scan_0203", "(001)", 10.0, 20.0),
        _f("Scan_0214", "(002)", 10.0, 20.0),   # same spot, different band → no match
    ]
    rep = prediction.repeatability(features, "Scan_0203", "Scan_0214")
    assert rep["matched"] == 0


def test_predicted_vs_observed_counts_gaps():
    # One recurrent track present at θ=5.0 and 4.0 but missing θ=4.5 (a gap).
    members = [
        {"scan": "Scan_0208", "theta": 5.0, "intensity": 100},
        {"scan": "Scan_0210", "theta": 4.0, "intensity": 90},
    ]
    track = {"track_id": 0, "reflection": "(001)", "is_recurrent": True,
             "theta_min": 4.0, "theta_max": 5.0, "chi_max_step": 1.0, "members": members}
    singleton = {"track_id": 1, "reflection": "(002)", "is_recurrent": False,
                 "theta_min": 3.0, "theta_max": 3.0, "chi_max_step": 0.0,
                 "members": [{"scan": "Scan_0212", "theta": 3.0, "intensity": 50}]}
    sampled = [3.0, 4.0, 4.5, 5.0]
    pvo = prediction.predicted_vs_observed([track, singleton], sampled)
    assert pvo["tp"] == 2          # θ=4.0 and 5.0 present
    assert pvo["fn"] == 1          # θ=4.5 in window but missing
    assert pvo["fp"] == 1          # the singleton at θ=3.0
    assert pvo["recall"] == round(2 / 3, 4)
    assert pvo["precision"] == round(2 / 3, 4)
    assert len(pvo["gaps"]) == 1 and pvo["gaps"][0]["theta"] == 4.5


def test_degradation_check_flags_film_loss():
    # 203: many perovskite + ITO; 214: perovskite collapses, ITO holds.
    features = (
        [_f("Scan_0203", "(001)", i, 0) for i in range(10)] +
        [_f("Scan_0203", "ITO", i, 5) for i in range(4)] +
        [_f("Scan_0214", "(001)", 0, 0)] +              # 10 → 1
        [_f("Scan_0214", "ITO", i, 5) for i in range(4)]  # 4 → 4
    )
    d = prediction.degradation_check(features, "Scan_0203", "Scan_0214")
    assert d["film_a"] == 10 and d["film_b"] == 1
    assert d["substrate_a"] == 4 and d["substrate_b"] == 4
    assert d["film_retention"] == 0.1 and d["substrate_retention"] == 1.0
    assert "degradation" in d["note"]


def test_rocking_summary_tolerates_csv_strings():
    # Simulate CSV reads: blanks are "" and numbers may be strings.
    rows = [
        {"status": "fit", "fwhm": "1.2", "theta_bragg": "4.5", "r_squared": "0.99",
         "microstrain": "-0.001", "strain_breadth_2th": "0.05", "chi_tilt_rate": "2.0"},
        {"status": "too_sparse", "fwhm": "", "theta_bragg": "", "r_squared": "",
         "microstrain": "0.0005", "strain_breadth_2th": "", "chi_tilt_rate": ""},
    ]
    summ = prediction.rocking_summary(rows)
    assert summ["n_fit"] == 1
    assert summ["fwhm"]["median"] == 1.2
    assert summ["microstrain"] is not None       # spans both rows, blanks dropped


def test_build_report_and_markdown_smoke():
    features = [
        _f("Scan_0208", "(001)", 10.0, 20.0),
        _f("Scan_0210", "(001)", 10.2, 20.1),
        _f("Scan_0203", "(002)", 5.0, 5.0),
        _f("Scan_0214", "(002)", 5.1, 5.2),
    ]
    tracks = tracking.build_tracks(features, match_tol=2.0, log=lambda *_: None)
    report = prediction.build_report(tracks, features, match_tol=2.0)
    assert "predicted_vs_observed" in report
    assert "repeatability_floor" in report
    md = prediction.to_markdown(report)
    assert "Prediction Report" in md
    assert "Repeatability floor" in md
    assert isinstance(report["verdict"], str) and report["verdict"]
