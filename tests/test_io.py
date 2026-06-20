import pytest
import h5py
import numpy as np
from xrd_app.core.io import generate_grid_mapping, build_bins

def test_data_prep_pipeline(synthetic_raw_scan, tmp_path):
    grid_out = tmp_path / "test_mapping.json"
    
    # 1. Test grid generation using the synthesized raster fallback
    mapping = generate_grid_mapping(
        xrd_dir=synthetic_raw_scan,
        pos_csv=None,
        bin_size=3,
        scan_number=203,
        output=grid_out,
        n_cols=5 # 10 frames / 5 cols = 2 rows
    )
    
    assert mapping["n_total_frames"] == 10
    assert grid_out.exists()
    
    # 2. Test bin building using the generated mapping
    binned_out = tmp_path / "test_bins.h5"
    build_bins(
        grid_mapping=grid_out,
        output=binned_out,
        bin_size=3,
        compression="none" # Speeds up tests
    )
    
    assert binned_out.exists()
    with h5py.File(binned_out, "r") as f:
        assert f.attrs["n_bins"] > 0
        # Check that the summation arithmetic worked
        assert "0_0" in f.keys()
        assert f["0_0"].shape == (5, 5)
