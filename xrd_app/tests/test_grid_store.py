"""Compact HDF5 grid-mapping persistence."""
from __future__ import annotations

import h5py

from xrd_app.core import io


def test_grid_mapping_hdf5_round_trip_uses_offsets(tmp_path):
    path = tmp_path / "grid_mapping_3x3.h5"
    mapping = {
        "bin_size": 3, "coordinate_source": "positions_xy",
        "n_rows": 4, "n_cols": 5, "n_bin_rows": 2, "n_bin_cols": 2,
        "n_total_frames": 5, "xrd_files": ["a.h5", "b.h5"],
        "frame_map": [[0, 0], [0, 1], [1, 0], [1, 1], [1, 2]],
        "bins": {"0_0": [0, 1, 2], "0_1": [3, 4]},
    }

    io.save_grid_mapping(path, mapping)

    assert io.load_grid_mapping(path) == mapping
    with h5py.File(path) as handle:
        assert handle["bin_offsets"][:].tolist() == [0, 3, 5]
        assert handle["bin_frames"][:].tolist() == [0, 1, 2, 3, 4]


def test_grid_mapping_hdf5_round_trip_preserves_territories(tmp_path):
    path = tmp_path / "grid_mapping_1x1_territory.h5"
    mapping = {
        "bin_size": 1, "variant": "territory", "xrd_files": [],
        "frame_map": [], "bins": {"0_0": [4]},
        "territories": {"0_0": {
            "centroid_xy": [1.0, 2.0], "centroid_rc": [3.0, 4.0],
            "area": 0.5, "count": 1, "neighbors": [],
            "polygon": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        }},
    }

    io.save_grid_mapping(path, mapping)

    assert io.load_grid_mapping(path) == mapping
