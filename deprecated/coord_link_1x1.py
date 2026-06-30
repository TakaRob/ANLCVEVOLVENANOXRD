"""Gridless coordinate-linked 1×1 shapes (reusing already-detected peaks).

The serpentine/backlash skew is a *grid* artefact: snapping frames to (row, col)
turns a sub-step position error into a wrong-column catastrophe. This skips the
grid entirely — it keeps each 1×1 frame at its true (X, Y) and links peaks across
**physical neighbors** (Delaunay), exactly like the territorial truth but at
per-frame (target-size 1) resolution. No deskew needed: there is no grid to skew.

Reuses the existing 1×1 peaks (no re-detection). Each default-grid 1×1 bin is
treated as one cell, RE-KEYED by nothing — its existing "r_c" key is kept only as
a unique label; adjacency comes solely from true (X, Y), so the (skewed) key
never drives linking. Writes a standard shapes catalog comparable to the truth.

    python3 coord_link_1x1.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from xrd_app.config import DataManager
from xrd_app.core import io, processing, territory as T

# ── CONFIG ─────────────────────────────────────────────────────────
PROJECT_ROOT = "TakaTest/TakaProject"
SCAN = None
PEAKS = "5x5_tophat_band_adaptive_snr_peaks_1x1.json"   # existing 1×1 peaks to reuse
LINK_TOL = 5
OUT_SHAPES = "coord_shapes_1x1.json"
OUT_MAPPING_VARIANT = "coord"                            # grid_mapping_1x1_coord.json


def main():
    dm = DataManager(PROJECT_ROOT, scan=SCAN)
    grid1 = dm.grid_mapping(bin_size=1)
    pos = dm.position_csv()
    peaks_path = dm.labels_dir() / PEAKS
    for label, p in [("1×1 grid mapping", grid1), ("positions", pos),
                     ("1×1 peaks", peaks_path)]:
        if not Path(p).exists():
            sys.exit(f"Missing {label}: {p}")

    gm = io.load_grid_mapping(grid1)
    bins = gm["bins"]                       # {"r_c": [frame_idx, ...]}
    n_total = gm.get("n_total_frames") or len(gm.get("frame_map", []))
    fx, fy = io.load_positions_xy(pos, n_total)
    fx, fy = io._interp_nan(fx), io._interp_nan(fy)

    # One point per 1×1 cell = mean true (X, Y) of its frame(s).
    keys = list(bins.keys())
    pts = np.array([[float(np.mean(fx[bins[k]])), float(np.mean(fy[bins[k]]))]
                    for k in keys], dtype=np.float64)
    step = T._median_step(pts)
    x_min, y_min = float(pts[:, 0].min()), float(pts[:, 1].min())
    print(f"{len(keys)} cells · step {step:.4g}")

    # Physical neighbor graph over the cell centroids (true X,Y) — no grid adjacency.
    print("Building (X, Y) Delaunay neighbor graph ...")
    adj = T._delaunay_adjacency(pts)
    territories = {}
    for i, k in enumerate(keys):
        cx, cy = pts[i]
        territories[k] = {
            "centroid_xy": [round(float(cx), 4), round(float(cy), 4)],
            "centroid_rc": [round((cy - y_min) / step, 3),
                            round((cx - x_min) / step, 3)],
            "count": len(bins[k]),
            "neighbors": [keys[j] for j in sorted(adj[i])],
        }

    coord_gm = dict(gm)
    coord_gm["coordinate_source"] = "coord_xy_1x1"
    coord_gm["target_size"] = 1
    coord_gm["step"] = step
    coord_gm["territories"] = territories

    map_out = dm.grid_mapping(bin_size=1, variant=OUT_MAPPING_VARIANT)
    io.atomic_write_json(map_out, coord_gm, indent=None)
    print(f"Wrote coordinate mapping -> {map_out}")

    with open(peaks_path) as f:
        peaks_data = json.load(f)

    print("Linking peaks across physical neighbors (territory algo) ...")
    result = processing.run_shapes(
        peaks=peaks_data, tth_path=dm.tth_map(), grid_mapping=coord_gm,
        reflections_path=dm.reflections(), bin_size=1, link_tolerance=LINK_TOL,
        shape_path=dm.shape_script("territory"), log=print)
    result["scan"] = dm.scan_name
    result["shape_algo"] = "territory_coord1x1"

    out = dm.labels_dir() / OUT_SHAPES
    io.atomic_write_json(out, result, indent=None)
    print(f"\nDone: {result['n_kept']} shapes kept, "
          f"{result['n_filtered']} filtered -> {out}")


if __name__ == "__main__":
    main()
