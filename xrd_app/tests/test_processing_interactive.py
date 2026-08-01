"""Pure interactive-analysis behavior shared by labeling and feature viewer."""

import numpy as np
import pytest

from xrd_app.core.processing import (
    build_explore_feature, detect_peaks_on_image, expand_peak_spatially,
)


def _peak(x, y, snr, label="(001)", intensity=None):
    return {
        "x": x, "y": y, "snr": snr, "label": label,
        "cleaned_intensity": float(intensity if intensity is not None else snr),
    }


def test_one_image_detection_labels_nearest_reflection_and_ignores_edge():
    image = np.zeros((12, 14), dtype=float)
    image[4:6, 6:9] = 10
    image[0:2, 0:2] = 20
    tth = np.full(image.shape, 7.5)

    detected = detect_peaks_on_image(
        image, tth, [6.8, 7.5], ["PbI2", "(001)"],
        percentile=97, min_pixels=3, pad=1, ignore_edge=2)

    assert detected == {"(001)": [(3, 7, 5, 10, 7, 4)]}


def test_spatial_expansion_tracks_local_peak_and_selects_highest_snr():
    seed = _peak(10, 10, 5, intensity=12)
    bins = {
        "0_1": [_peak(12, 10, 4), _peak(11, 10, 8), _peak(10, 10, 99, "(002)")],
        "0_2": [_peak(15, 10, 7)],
        "1_2": [_peak(19, 10, 9)],
    }

    members = expand_peak_spatially("0_0", seed, bins.get, link_tolerance=5)

    assert [member[0] for member in members] == ["0_0", "0_1", "0_2", "1_2"]
    assert members[1][4] == 11
    assert members[-1][4] == 19


def test_spatial_expansion_can_be_cancelled_without_partial_result():
    seed = _peak(1, 1, 5)
    assert expand_peak_spatially(
        "0_0", seed, lambda _key: [], cancelled=lambda: True) is None


def test_explore_feature_schema_and_characterization_match_viewer_behavior():
    seed = _peak(10, 10, 5, intensity=10)
    second = _peak(12, 12, 8, intensity=30)
    members = [
        ("0_0", 0, 0, 0, 10, 10, seed),
        ("0_1", 0, 0, 1, 12, 12, second),
    ]

    feature = build_explore_feature(members, seed, beam_center=(11, 11))

    assert feature["reflection"] == "(001)"
    assert feature["detector_x"] == 11
    assert feature["detector_y"] == 11
    assert feature["peak_intensity"] == 30
    assert feature["mean_snr"] == pytest.approx(6.5)
    assert feature["n_bins"] == 2
    assert feature["center_bin"] == "0_1"
    assert feature["chi_deg"] == 0
    assert list(feature["intensity_profile"]) == ["0_0", "0_1"]
