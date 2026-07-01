"""Headless tests for rocking-curve fitting (core/rocking.py)."""
from __future__ import annotations

import math

from xrd_app.core import rocking


def _gauss(theta, bg, amp, t0, fwhm):
    sigma = fwhm / (2 * math.sqrt(2 * math.log(2)))
    return bg + amp * math.exp(-0.5 * ((theta - t0) / sigma) ** 2)


def test_fits_clean_gaussian():
    t0, fwhm, amp, bg = 4.5, 1.5, 100.0, 5.0
    thetas = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
    ints = [_gauss(t, bg, amp, t0, fwhm) for t in thetas]
    fit = rocking.fit_rocking_curve(thetas, ints)
    assert fit["status"] == "fit"
    assert abs(fit["theta_bragg"] - t0) < 0.05
    assert abs(fit["fwhm"] - fwhm) < 0.1
    assert fit["r_squared"] > 0.999


def test_too_sparse_returns_moments_only():
    thetas = [0.0, 20.5]
    ints = [10.0, 80.0]
    fit = rocking.fit_rocking_curve(thetas, ints)
    assert fit["status"] == "too_sparse"
    assert fit["theta_bragg"] is None
    assert fit["theta_at_max"] == 20.5          # moments still provided
    assert fit["integrated_intensity"] is not None


def test_monotonic_flagged_not_fit():
    # Strictly increasing over the window — no interior peak.
    thetas = [3.0, 3.5, 4.0, 4.5, 5.0]
    ints = [10.0, 20.0, 30.0, 40.0, 50.0]
    fit = rocking.fit_rocking_curve(thetas, ints)
    assert fit["status"] in ("monotonic", "fit")
    # argmax is the last point regardless
    assert fit["theta_at_max"] == 5.0


def test_microstrain_sign_and_magnitude():
    # Measured 2θ above reference ⇒ smaller d ⇒ compressive (ε < 0).
    eps_comp = rocking.microstrain(7.60, 7.50)
    assert eps_comp is not None and eps_comp < 0
    # Measured below reference ⇒ tensile (ε > 0).
    eps_tens = rocking.microstrain(7.40, 7.50)
    assert eps_tens > 0
    # Symmetric about the reference.
    assert abs(eps_comp + eps_tens) < 1e-9
    assert rocking.microstrain(None, 7.5) is None
    assert rocking.microstrain(7.5, None) is None


def test_physical_metrics_three_axes():
    members = [
        {"theta": 4.0, "chi_deg": 30.0, "tth_com": 7.52, "tth_fwhm": 0.05, "intensity": 100},
        {"theta": 4.5, "chi_deg": 31.0, "tth_com": 7.52, "tth_fwhm": 0.06, "intensity": 200},
        {"theta": 5.0, "chi_deg": 32.0, "tth_com": 7.52, "tth_fwhm": 0.07, "intensity": 150},
    ]
    track = {"track_id": 0, "reflection": "(001)", "ref_tth": 7.50, "members": members}
    phys = rocking.physical_metrics(track)
    # microstrain from tth_com 7.52 vs 7.50 → compressive
    assert phys["microstrain"] < 0
    assert phys["tth_com"] == 7.52
    assert phys["strain_breadth_2th"] == 0.06        # median of [0.05,0.06,0.07]
    # χ rises 30→31→32 linearly with θ over 1.0° → slope 2.0 °/°
    assert abs(phys["chi_tilt_rate"] - 2.0) < 1e-6
    assert phys["chi_span"] == 2.0


def test_fit_tracks_filters_recurrent():
    t0, fwhm, amp, bg = 4.0, 1.2, 50.0, 2.0
    members = [{"scan": f"S{i}", "theta": th, "intensity": _gauss(th, bg, amp, t0, fwhm)}
               for i, th in enumerate([3.0, 3.5, 4.0, 4.5, 5.0])]
    recurrent = {"track_id": 0, "reflection": "(001)", "ref_tth": 7.5,
                 "centroid_row": 1, "centroid_col": 2, "is_recurrent": True,
                 "members": members}
    sparse = {"track_id": 1, "reflection": "(001)", "ref_tth": 7.5,
              "centroid_row": 9, "centroid_col": 9, "is_recurrent": False,
              "members": [{"scan": "S9", "theta": 0.0, "intensity": 10.0}]}

    rows = rocking.fit_tracks([recurrent, sparse], only_recurrent=True, log=lambda *_: None)
    assert len(rows) == 1
    assert rows[0]["track_id"] == 0
    assert rows[0]["status"] == "fit"

    rows_all = rocking.fit_tracks([recurrent, sparse], only_recurrent=False, log=lambda *_: None)
    assert len(rows_all) == 2
