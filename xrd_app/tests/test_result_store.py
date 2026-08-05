"""Round-trip tests for typed numerical HDF5 catalogs."""
from __future__ import annotations

import h5py

from xrd_app.core import result_store


def test_peak_catalog_round_trip_uses_columnar_datasets(tmp_path):
    path = tmp_path / "detector_peaks_3x3.h5"
    data = {
        "bin_size": 3, "scan": "Scan_0203", "lineage": {"stage": "peaks"},
        "peaks_by_bin": {
            "0_1": [{"x": 12, "y": 8, "label": "(001)", "snr": 4.5,
                     "cleaned_intensity": 22.0, "custom": "kept"}],
        },
    }

    result_store.save(path, data)

    with h5py.File(path) as handle:
        assert handle.attrs["format"] == result_store.FORMAT
        assert handle["peaks/x"].dtype.kind == "f"
        assert "peaks_by_bin" not in handle.attrs["metadata_json"]
    assert result_store.load(path) == data


def test_hd_map_round_trip_flattens_cells(tmp_path):
    path = tmp_path / "gaussian_hdmap_3x3.h5"
    data = {
        "kind": "hd_map", "n_bin_rows_1x1": 20, "n_bin_cols_1x1": 30,
        "features": [{
            "feature_id": 4, "reflection": "(011)",
            "hd_profile": {
                "2_3": {"intensity": 8.0, "integrated": 21.0,
                        "x": 1.25, "y": 2.5},
            },
        }],
    }

    result_store.save(path, data)

    assert result_store.load(path) == data
    with h5py.File(path) as handle:
        assert handle["hd_features/cell_feature"].shape == (1,)


def test_combined_catalog_round_trip_flattens_points(tmp_path):
    path = tmp_path / "combined_1x1.h5"
    data = {
        "n_features": 1,
        "by_bin": {"0_0": {"(001)": [[2, 3], [4.5, 6.5]]}},
        "features": [],
    }

    result_store.save(path, data)

    assert result_store.load(path) == data
    with h5py.File(path) as handle:
        assert handle["combined_points/x"].shape == (2,)


def test_shape_catalog_round_trip_flattens_profiles(tmp_path):
    path = tmp_path / "gaussian_shapes_3x3.h5"
    feature = {
        "feature_id": 1, "reflection": "(001)", "detector_x": 10,
        "spatial_extent": ["0_0", "0_1"], "center_bin": "0_1",
        "intensity_profile": {
            "0_0": {"intensity": 2.0, "integrated": 5.0, "det_x": 10,
                    "det_y": 11, "sample_crop_fill": True},
            "0_1": {"intensity": 8.0, "integrated": 20.0, "det_x": 12,
                    "det_y": 13, "tth": 9.2},
        },
    }
    data = {"bin_size": 3, "kept": [feature], "filtered": [], "n_kept": 1}

    result_store.save(path, data)
    loaded = result_store.load(path)

    assert loaded == data
    with h5py.File(path) as handle:
        assert handle["kept/profile_feature"].shape == (2,)
        assert handle["kept/feature_json"].shape == (1,)
