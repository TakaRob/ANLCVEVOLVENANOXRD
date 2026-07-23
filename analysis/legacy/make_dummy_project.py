#!/usr/bin/env python3
import os
import json
import h5py
import numpy as np
import tifffile
from pathlib import Path
from xrd_app.core.io import atomic_write_json
from xrd_app.core.processing import (
    run_peaks,
    run_shapes,
    write_feature_catalog,
    write_peak_table,
)

def main():
    print("Setting up dummy project...")
    root = Path("dummy_project")
    root.mkdir(exist_ok=True)
    (root / "Metadata").mkdir(parents=True, exist_ok=True)
    (root / "Binned" / "Scan_0203").mkdir(parents=True, exist_ok=True)
    (root / "Labels" / "Scan_0203").mkdir(parents=True, exist_ok=True)

    # 1. Write config.json
    config = {
        "project_name": "dummy_project",
        "created_at": "2026-06-20T02:00:00",
        "active_scan": "203",
        "active_bin_size": 3,
        "data_sources": {
            "tth_map": "Metadata/tth.tiff",
            "reflections": "Metadata/reflections.py",
            "detector_script": "xrd_app/PeakAlgorithms/5x5_tophat_band_adaptive_snr.py"
        }
    }
    atomic_write_json(root / "config.json", config)

    # 2. Write dummy reflections.py
    reflections_content = """# Reflections calibration
degs = [6.81319, 7.51422, 10.61748, 13.00831, 15.01266, 16.07224]
deg_labels = ['PbI2', '(001)', '(011)', '(111)', '(002)', 'ITO']
"""
    with open(root / "Metadata" / "reflections.py", "w") as f:
        f.write(reflections_content)

    # 3. Write dummy tth.tiff
    tth_path = root / "Metadata" / "tth.tiff"
    dummy_tth = np.full((100, 100), 13.0)
    tifffile.imwrite(tth_path, dummy_tth.astype(np.float64))

    # 4. Write dummy binned H5 file
    bins_h5_path = root / "Binned" / "Scan_0203" / "xrd_3x3_bins.h5"
    with h5py.File(bins_h5_path, "w") as f:
        f.attrs["bin_size"] = 3
        f.attrs["detector_shape"] = [100, 100]
        f.attrs["n_bin_rows"] = 2
        f.attrs["n_bin_cols"] = 2
        f.attrs["n_bins"] = 4
        # Inject peaks in some of the bins
        for r in range(2):
            for c in range(2):
                img = np.random.rand(100, 100).astype(np.float32)
                # Inject a bright spot representing a peak
                img[48:52, 48:52] = 100.0
                f.create_dataset(f"{r}_{c}", data=img)

    # 5. Write grid mapping JSON
    grid_path = root / "Metadata" / "Scan_0203_grid_mapping.json"
    grid_data = {
        "bin_size": 3,
        "n_rows": 6,
        "n_cols": 6,
        "n_bin_rows": 2,
        "n_bin_cols": 2,
        "n_total_frames": 36,
        "n_bins": 4,
        "h5_dataset": "entry/data/data",
        "xrd_files": ["dummy_file.h5"],
        "bins": {"0_0": [0], "0_1": [1], "1_0": [2], "1_1": [3]},
        "frame_map": [[0, 0], [0, 1], [0, 2], [0, 3]],
    }
    atomic_write_json(grid_path, grid_data)

    # 6. Run Peak finding
    print("Running peak finding stage...")
    peaks = run_peaks(
        bins_h5=bins_h5_path,
        tth_path=tth_path,
        detector_path="xrd_app/PeakAlgorithms/5x5_tophat_band_adaptive_snr.py",
        reflections_path=root / "Metadata" / "reflections.py",
        snr_threshold=3.0,
        log=print
    )

    # Save peaks.json
    peaks_out_path = root / "Labels" / "Scan_0203" / "5x5_tophat_band_adaptive_snr_peaks.json"
    atomic_write_json(peaks_out_path, peaks)

    # 7. Run Shape finding
    print("Running shape finding stage...")
    shapes = run_shapes(
        peaks=peaks,
        tth_path=tth_path,
        grid_mapping=grid_path,
        reflections_path=root / "Metadata" / "reflections.py",
        bin_size=3,
        log=print
    )

    # Save shape result catalogs
    shapes_out_path = root / "Labels" / "Scan_0203" / "5x5_tophat_band_adaptive_snr_shapes.json"
    shapes["scan"] = "Scan_0203"
    shapes["shape_algo"] = "default"
    shapes["peak_source"] = "5x5_tophat_band_adaptive_snr"
    shapes["lineage"] = {}
    atomic_write_json(shapes_out_path, shapes)

    # Legacy formats for GUI compatibility shims
    ldir = root / "Labels" / "Scan_0203"
    suffix = "3x3"
    write_feature_catalog(shapes["kept"], ldir / f"feature_catalog_{suffix}.json", print)
    write_peak_table(shapes["kept"], ldir / f"kept_peaks_{suffix}.csv", "kept peaks", print)
    write_peak_table(shapes["filtered"], ldir / f"filtered_peaks_{suffix}.csv", "filtered peaks", print)

    print("\nDummy project successfully generated!")
    print("You can run the GUI on this project by typing:")
    print("    xrd-app gui --project-root dummy_project")

if __name__ == "__main__":
    main()
