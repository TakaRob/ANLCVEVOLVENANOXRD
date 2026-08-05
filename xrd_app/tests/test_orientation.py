"""Headless tests for orientation geometry, weighting, clustering, and density."""

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter1d

from xrd_app.core import orientation


def _feature(chi, area=1, intensity=1, rocking=0, strain=0):
    return {
        "chi_deg": chi,
        "n_bins": area,
        "peak_intensity": intensity,
        "chi_fwhm": rocking,
        "tth_fwhm": strain,
    }


def test_beam_center_recovery_and_chi_map_use_row_col_convention():
    rows, cols = np.mgrid[0:80, 0:70]
    expected_center = (95.0, 270.0)
    tth = 0.02 * np.hypot(rows - expected_center[0], cols - expected_center[1])
    recovered = orientation.estimate_beam_center(tth)
    assert recovered == pytest.approx(expected_center, abs=0.1)

    chi = orientation.compute_chi_map((3, 3), (1, 1))
    assert chi[1, 2] == pytest.approx(0)
    assert chi[2, 1] == pytest.approx(90)
    assert abs(chi[1, 0]) == pytest.approx(180)
    assert chi[0, 1] == pytest.approx(-90)


def test_feature_weight_modes_preserve_orientation_map_formulas():
    feature = _feature(0, area=4, intensity=5, rocking=2, strain=3)
    assert orientation.feature_weight(feature, "count") == 1
    assert orientation.feature_weight(feature, "area") == 4
    assert orientation.feature_weight(feature, "intensity") == 5
    assert orientation.feature_weight(feature, "bright_big") == 100
    assert orientation.feature_weight(feature, "unknown") == 1


def test_kde_cluster_percentages_follow_selected_weight():
    features = [
        _feature(-120, area=1), _feature(-118, area=1), _feature(-116, area=1),
        _feature(0, area=1.5), _feature(2, area=1.5), _feature(4, area=1.5),
        _feature(120, area=1), _feature(122, area=1), _feature(124, area=1),
    ]
    clusters, valleys = orientation.cluster_features_by_chi(
        features, bandwidth=3, weight_mode="area")

    assert len(clusters) == 2
    assert valleys
    assert sorted(cluster["n"] for cluster in clusters) == [3, 6]
    assert sorted(cluster["pct"] for cluster in clusters) == pytest.approx([42.9, 57.1])
    assert sum(cluster["weight"] for cluster in clusters) == pytest.approx(10.5)


def test_orientation_density_matches_weighted_wrapped_histogram():
    features = {"(001)": [_feature(-179, area=2), _feature(179, area=1)]}
    densities, maximum, vmin, vmax = orientation.orientation_densities(
        features, {"(001)"}, sigma=1, weight_mode="area")

    hist, _ = np.histogram([-179, 179], bins=np.arange(-180, 181, 1),
                           weights=[2, 1])
    expected = gaussian_filter1d(hist.astype(float), sigma=1, mode="wrap")
    assert np.allclose(densities["(001)"], expected)
    assert maximum == pytest.approx(expected.max())
    assert vmin >= 0
    assert vmax > vmin
    assert densities["(001)"][0] > 0
    assert densities["(001)"][-1] > 0
