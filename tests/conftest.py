import pytest
import h5py
import numpy as np
import tifffile
import json

@pytest.fixture
def synthetic_workspace(tmp_path):
    """Creates a temporary folder with all required metadata and binned data to test Phase 1 & 2."""
    
    # 1. Generate Synthetic 2-theta map (TIFF)
    tth_path = tmp_path / "tth.tiff"
    # Create a 100x100 dummy map (much smaller than the real 1062x1028)
    dummy_tth = np.full((100, 100), 13.0)
    tifffile.imwrite(tth_path, dummy_tth.astype(np.float64))
    
    # 2. Generate Synthetic Binned HDF5 (e.g., 2x2 grid)
    bins_h5_path = tmp_path / "bins_3x3.h5"
    with h5py.File(bins_h5_path, "w") as f:
        f.attrs["bin_size"] = 3
        f.attrs["detector_shape"] = [100, 100]
        # Create fake imagery for 4 bins. Inject an artificial "peak" at (50, 50)
        for r in range(2):
            for c in range(2):
                img = np.random.rand(100, 100).astype(np.float32)
                img[48:52, 48:52] = 100.0 # bright spot
                f.create_dataset(f"{r}_{c}", data=img)
                
    # 3. Generate Grid Mapping JSON
    grid_path = tmp_path / "grid_mapping.json"
    grid_data = {
        "n_bin_rows": 2,
        "n_bin_cols": 2,
        "bin_size": 3,
        "bins": {"0_0": [0], "0_1": [1], "1_0": [2], "1_1": [3]},
        "frame_map": [[0, 0], [0, 1], [0, 2], [0, 3]],
        "xrd_files": ["dummy_file.h5"]
    }
    with open(grid_path, "w") as f:
        json.dump(grid_data, f)
        
    return {
        "root": tmp_path,
        "tth": tth_path,
        "bins_h5": bins_h5_path,
        "grid": grid_path
    }

@pytest.fixture
def synthetic_raw_scan(tmp_path):
    """Generates a tiny raw scan directory simulating the beamline format."""
    scan_dir = tmp_path / "Scan_0203"
    scan_dir.mkdir()
    
    h5_path = scan_dir / "scan_0203_001.h5"
    with h5py.File(h5_path, "w") as f:
        # Create 10 frames of 5x5 pixels (extremely fast to process)
        data = np.ones((10, 5, 5), dtype=np.uint32)
        f.create_dataset("entry/data/data", data=data)
        
    return scan_dir
