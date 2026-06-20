import os
import json
import csv
import h5py
import numpy as np
import tifffile
from pathlib import Path

def create_mock_workspace(root_dir="MockWorkspace", scan_name="Scan_0001", bin_size=3):
    root = Path(root_dir)
    scan_dir = root / "Results" / scan_name
    meta_dir = root / "Metadata" / scan_name
    binned_dir = root / "Binned" / scan_name
    assets_dir = root / "assets"
    algo_dir = root / "PeakAlgorithms"

    # 1. Create directory structure
    for d in [scan_dir, meta_dir, binned_dir, assets_dir, algo_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"Creating mock workspace in: {root.absolute()}")

    # 2. config.yaml (structured cleanly for ProjectConfig nested get resolution)
    config_data = f"""\
name: MockWorkspace
scan:
  number: 1
  name: {scan_name}
paths:
  labels_dir: "Results"
  metadata_dir: "Metadata"
  binned_dir: "Binned"
data_sources:
  tth_map: "assets/tth.tiff"
  reflections: "assets/reflections.py"
  detector_script: "PeakAlgorithms/baseline.py"
  position_csv: "Metadata/Scan_0001/positions.csv"
"""
    (root / "config.yaml").write_text(config_data)

    # 3. Dummy Detector Script
    dummy_detector = """\
def radial_median_subtract(image, tth_data): return image
def fast_tophat(cleaned, size): return cleaned
def build_tth_band_masks(*args, **kwargs): return {"Phase_A": None}
def detect_in_band(*args, **kwargs): return []
def precompute_tth(tth_map): return None
"""
    (algo_dir / "baseline.py").write_text(dummy_detector)

    # 4. Reflections & 2-Theta Map (1062 x 1028)
    (assets_dir / "reflections.py").write_text('degs = [10.0, 15.0]\ndeg_labels = ["Phase_A", "Phase_B"]')
    
    y, x = np.mgrid[0:1062, 0:1028]
    tth_map = np.sqrt((x - 500)**2 + (y - 500)**2) * 0.05
    tifffile.imwrite(assets_dir / "tth.tiff", tth_map.astype(np.float64))

    # 5. Grid Mapping JSON (10x10 bin grid)
    grid_mapping = {
        "bin_size": bin_size,
        "n_rows": 30, "n_cols": 30,
        "n_bin_rows": 10, "n_bin_cols": 10,
        "n_bins": 100
    }
    with open(meta_dir / f"grid_mapping_{bin_size}x{bin_size}.json", "w") as f:
        json.dump(grid_mapping, f, indent=2)

    # 5.5 dummy positions.csv
    csv_path_pos = meta_dir / "positions.csv"
    with open(csv_path_pos, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["X_Position", "Y_Position"])
        for i in range(100):
            writer.writerow([float(i % 10), float(i // 10)])

    # 6. Binned HDF5 Data (with a synthetic "hotspot" peak)
    h5_path = binned_dir / f"xrd_{bin_size}x{bin_size}_bins.h5"
    with h5py.File(h5_path, "w") as f:
        f.attrs["bin_size"] = bin_size
        f.attrs["detector_shape"] = [1062, 1028]
        f.attrs["n_bin_rows"] = 10
        f.attrs["n_bin_cols"] = 10
        f.attrs["n_bins"] = 100
        
        for r in range(10):
            for c in range(10):
                bin_key = f"{r}_{c}"
                # Generate base noise
                img = np.random.rand(1062, 1028).astype(np.float32) * 5.0
                
                # Inject a bright peak into a specific 3x3 cluster of bins (center is 5_5)
                if 4 <= r <= 6 and 4 <= c <= 6:
                    intensity_multiplier = 100.0 if (r==5 and c==5) else 30.0
                    # Place peak at detector coordinate (600, 400)
                    img[390:410, 590:610] += np.random.rand(20, 20) * intensity_multiplier
                
                f.create_dataset(bin_key, data=img, compression="gzip")

    # 7. Kept Features (feature_catalog.json)
    kept_features = [{
        "feature_id": 1,
        "reflection": "Phase_A",
        "detector_x": 600,
        "detector_y": 400,
        "peak_intensity": 105.2,
        "mean_snr": 12.4,
        "n_bins": 9,
        "spatial_extent": [f"{r}_{c}" for r in range(4, 7) for c in range(4, 7)],
        "center_bin": "5_5",
        "center_row": 5,
        "center_col": 5,
        "reason": "Gaussian-like: 85% monotonic, 9 bins",
        "intensity_profile": {
            f"{r}_{c}": {"intensity": 100.0 if (r==5 and c==5) else 30.0, "integrated": 150.0}
            for r in range(4, 7) for c in range(4, 7)
        }
    }]
    with open(scan_dir / f"feature_catalog_{bin_size}x{bin_size}.json", "w") as f:
        json.dump(kept_features, f, indent=2)

    # 8. Filtered Features (filtered_peaks.csv)
    csv_path = scan_dir / f"filtered_peaks_{bin_size}x{bin_size}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "feature_id", "reflection", "bin_key", "center_row", "center_col",
            "detector_x", "detector_y", "peak_intensity", "mean_snr",
            "n_bins", "spatial_extent", "reason"
        ])
        writer.writerow([
            2, "Phase_B", "2_2", 2, 2, 800, 200, 15.0, 2.1, 
            1, "2_2", "isolated: single-bin detection"
        ])

    print("Mock workspace created successfully! You can now launch the GUI.")

if __name__ == "__main__":
    create_mock_workspace()
