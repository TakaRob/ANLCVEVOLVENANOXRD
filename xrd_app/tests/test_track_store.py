"""Typed HDF5 persistence for rocking-study tracks."""
from __future__ import annotations

import h5py

from xrd_app.core import result_store


def test_track_round_trip_flattens_members(tmp_path):
    path = tmp_path / "tracks.h5"
    data = {
        "bin_size": 3, "n_tracks": 1,
        "tracks": [{
            "track_id": 7, "reflection": "(001)", "n_members": 1,
            "is_recurrent": True,
            "members": [{
                "scan": "Scan_0203", "theta": 4.0, "center_row": 2.0,
                "center_col": 3.0, "chi_deg": None, "tth_com": 9.1,
                "peak_intensity": 10.0, "sum_integrated": 20.0,
                "intensity": 20.0, "detector_x": 100, "detector_y": 200,
                "tth_fwhm": 0.1, "chi_fwhm": 0.2, "feature_id": 5,
            }],
        }],
    }

    result_store.save(path, data)

    assert result_store.load(path) == data
    with h5py.File(path) as handle:
        assert handle["tracks/member_track"].shape == (1,)
        assert '"tracks":' not in handle.attrs["metadata_json"]
