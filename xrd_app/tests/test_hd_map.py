"""Headless tests for the HD (1×1) intensity sampler (core/hd_map.py)."""
from __future__ import annotations

import numpy as np

from xrd_app.core import hd_map, io


class _FakeSource:
    """1×1 source with a bright spot at (x=12, y=10); some cells missing (holes)."""

    def __init__(self, present):
        self._present = set(present)

    def image(self, key):
        if key not in self._present:
            return None
        img = np.zeros((20, 20), dtype=np.float32)
        img[10, 12] = 100.0
        img[9, 11] = 50.0
        return img

    def region(self, key, y0, y1, x0, x1):
        img = self.image(key)
        if img is None:
            return None
        return img[max(0, y0):y1, max(0, x0):x1]

    def close(self):
        pass


def _feature(fid=1, bin_key="0_0", ref="PbI2"):
    return {"feature_id": fid, "reflection": ref, "chi_deg": 30.0,
            "ref_tth": 12.5, "detector_x": 12, "detector_y": 10,
            "spatial_extent": [bin_key]}


def test_subbin_footprint_and_window_sampling():
    feat = _feature()
    cells = io.subbin_keys("0_0", 3)          # nine 1×1 cells
    present = [c for c in cells if c != "2_2"]  # drop one → a hole
    res = hd_map.sample_hd_intensity([feat], _FakeSource(present), bin_size=3, win=4)
    prof = res[0]["hd_profile"]
    assert len(prof) == 8                      # 9 cells minus the hole
    assert "2_2" not in prof
    # window max/sum around the detector peak
    assert prof["0_0"]["intensity"] == 100.0
    assert prof["0_0"]["integrated"] == 150.0
    # carried-through feature metadata
    assert res[0]["reflection"] == "PbI2"
    assert res[0]["ref_tth"] == 12.5


def test_cell_xy_attached_when_positions_given():
    feat = _feature()
    cells = io.subbin_keys("0_0", 3)
    cell_xy = {c: (float(i) * 0.5, float(i) * 0.25) for i, c in enumerate(cells)}
    res = hd_map.sample_hd_intensity([feat], _FakeSource(cells), bin_size=3,
                                     win=4, cell_xy=cell_xy)
    e = res[0]["hd_profile"]["0_1"]
    assert e["x"] == cell_xy["0_1"][0]
    assert e["y"] == cell_xy["0_1"][1]
    s = hd_map.summarize(res)
    assert s["n_features"] == 1
    assert s["n_cells"] == 9
    assert s["n_cells_with_position"] == 9


def test_max_cells_cap():
    feat = _feature()
    cells = io.subbin_keys("0_0", 3)
    res = hd_map.sample_hd_intensity([feat], _FakeSource(cells), bin_size=3,
                                     win=4, max_cells_per_feature=4)
    assert len(res[0]["hd_profile"]) == 4


def test_scan_trajectory_grid_order():
    # 2×2 raster acquired serpentine: (0,0)(0,1)(1,1)(1,0) in frame order.
    gm = {"n_total_frames": 4,
          "bins": {"0_0": [0], "0_1": [1], "1_1": [2], "1_0": [3]}}
    t = hd_map.scan_trajectory(gm)
    # grid path is [col, row] in acquisition order
    assert t["grid"] == [[0, 0], [1, 0], [1, 1], [0, 1]]
    assert t["xy"] is None                     # no positions CSV


def _oriented_feature(x_from_row, y_from_col):
    profile = {}
    for row in range(3):
        for col in range(4):
            profile[f"{row}_{col}"] = {
                "x": float(x_from_row(row)),
                "y": float(y_from_col(col)),
            }
    return {"hd_profile": profile}


def test_infer_xy_orientation_handles_negative_row_x_scan():
    feature = _oriented_feature(lambda row: 10 - row, lambda col: 20 + col)

    orient = hd_map.infer_xy_orientation([feature])

    assert orient == {"horizontal": "y", "vertical": "x",
                      "invert_x": False, "invert_y": False}


def test_infer_xy_orientation_handles_positive_row_x_scan():
    feature = _oriented_feature(lambda row: 10 + row, lambda col: 20 - col)

    orient = hd_map.infer_xy_orientation([feature])

    assert orient == {"horizontal": "y", "vertical": "x",
                      "invert_x": True, "invert_y": True}
