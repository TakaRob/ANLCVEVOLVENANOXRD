"""Headless tests for detector ROI spatial maps."""
from __future__ import annotations

import numpy as np
import pytest

from xrd_app.core import roi_map


class _FakeSource:
    def __init__(self):
        self.images = {
            "0_0": np.arange(36, dtype=float).reshape(6, 6),
            "0_1": np.full((6, 6), 10.0),
            "1_0": None,
        }
        self.requests = []

    def keys(self):
        return list(self.images)

    def region(self, key, y0, y1, x0, x1):
        self.requests.append((key, y0, y1, x0, x1))
        image = self.images[key]
        return None if image is None else image[y0:y1, x0:x1]


def test_sample_roi_uses_detector_xy_and_leaves_missing_bins_empty():
    source = _FakeSource()
    result = roi_map.sample_roi(
        source, (2, 1, 5, 3), metric="integrated",
        grid_mapping={"n_bin_rows": 2, "n_bin_cols": 2,
                      "bins": {"0_0": [0], "0_1": [1, 2], "1_0": [3]}},
    )

    assert source.requests[0] == ("0_0", 1, 3, 2, 5)
    patch = source.images["0_0"][1:3, 2:5]
    assert result["profile"]["0_0"]["intensity"] == patch.max()
    assert result["profile"]["0_0"]["integrated"] == patch.sum()
    assert result["profile"]["0_0"]["mean"] == patch.mean()
    assert result["profile"]["0_1"]["n_frames"] == 2
    assert "1_0" not in result["profile"]
    assert result["center_bin"] == "0_0"

    grid = roi_map.grid_array(result)
    assert grid.shape == (2, 2)
    assert np.isnan(grid[1, 0])
    assert np.isnan(grid[1, 1])


def test_sample_roi_can_normalize_each_bin_by_frame_count():
    result = roi_map.sample_roi(
        _FakeSource(), (0, 0, 2, 2), metric="integrated",
        normalize_frames=True,
        grid_mapping={"bins": {"0_0": [0], "0_1": [1, 2]}},
    )

    assert result["profile"]["0_1"]["integrated"] == 20.0
    assert result["normalize_frames"] is True


def test_to_shape_feature_creates_one_complete_manual_feature():
    sampled = roi_map.sample_roi(
        _FakeSource(), (2, 1, 5, 3), metric="integrated",
        grid_mapping={"n_bin_rows": 2, "n_bin_cols": 2,
                      "bins": {"0_0": [0], "0_1": [1, 2]}},
    )
    feature = roi_map.to_shape_feature(sampled, "(001)", feature_id=7)

    assert feature["feature_id"] == 7
    assert feature["reflection"] == "(001)"
    assert feature["manual_roi"] == {"x0": 2, "y0": 1, "x1": 5, "y1": 3}
    assert feature["n_bins"] == 2
    assert feature["spatial_extent"] == ["0_0", "0_1"]
    assert set(feature["intensity_profile"]) == {"0_0", "0_1"}
    assert "manual fixed detector ROI" in feature["reason"]


def test_auto_roi_snaps_to_and_bounds_gaussian_peak():
    yy, xx = np.mgrid[:80, :90]
    image = 5.0 + 100.0 * np.exp(-((xx - 52) ** 2 / (2 * 4 ** 2) +
                                  (yy - 31) ** 2 / (2 * 6 ** 2)))

    bounds = roi_map.auto_roi_from_click(image, 47, 35)

    x0, y0, x1, y1 = bounds
    assert x0 < 52 < x1
    assert y0 < 31 < y1
    assert 10 <= x1 - x0 <= 25
    assert 15 <= y1 - y0 <= 35


def test_auto_roi_ignores_brighter_disconnected_neighbor():
    yy, xx = np.mgrid[:70, :80]
    clicked = 80.0 * np.exp(-((xx - 25) ** 2 + (yy - 30) ** 2) / (2 * 3 ** 2))
    neighbor = 120.0 * np.exp(-((xx - 44) ** 2 + (yy - 30) ** 2) / (2 * 3 ** 2))
    bounds = roi_map.auto_roi_from_click(clicked + neighbor, 25, 30, search_radius=8)

    x0, y0, x1, y1 = bounds
    assert x0 < 25 < x1
    assert not (x0 <= 44 < x1)


def test_fixed_roi_matches_known_spatial_maximum():
    class Source:
        def keys(self):
            return ["0_0", "0_1", "1_0"]

        def region(self, key, y0, y1, x0, x1):
            levels = {"0_0": 2.0, "0_1": 11.0, "1_0": 4.0}
            return np.full((y1 - y0, x1 - x0), levels[key])

    sampled = roi_map.sample_roi(
        Source(), (20, 30, 25, 36), metric="integrated",
        grid_mapping={"n_bin_rows": 2, "n_bin_cols": 2},
    )
    feature = roi_map.to_shape_feature(sampled, "(011)")

    assert sampled["center_bin"] == "0_1"
    assert feature["center_bin"] == "0_1"
    assert feature["intensity_profile"]["0_1"]["integrated"] == 330.0


def test_roi_validation_and_metric_validation():
    assert roi_map.normalize_roi((5, 4, 2, 1)) == (2, 1, 5, 4)
    with pytest.raises(ValueError):
        roi_map.normalize_roi((1, 1, 1, 4))
    with pytest.raises(ValueError):
        roi_map.sample_roi(_FakeSource(), (0, 0, 2, 2), metric="unknown")
