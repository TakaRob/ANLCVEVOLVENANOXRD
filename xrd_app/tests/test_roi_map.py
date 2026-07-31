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


def test_roi_validation_and_metric_validation():
    assert roi_map.normalize_roi((5, 4, 2, 1)) == (2, 1, 5, 4)
    with pytest.raises(ValueError):
        roi_map.normalize_roi((1, 1, 1, 4))
    with pytest.raises(ValueError):
        roi_map.sample_roi(_FakeSource(), (0, 0, 2, 2), metric="unknown")
