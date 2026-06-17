#!/usr/bin/env python3
"""
Generate grid_mapping.json for any spatial bin size.

Reads the raw scan H5 files and position CSV to compute which detector
frames belong to each spatial bin.  Output is a JSON file with the same
structure used by CVEvolve and prebuild_bins.py.

Usage:
    python generate_grid_mapping.py --bin-size 3 --output grid_mapping_3x3.json
    python generate_grid_mapping.py --bin-size 5  # defaults to grid_mapping_5x5.json
"""

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import argrelextrema


H5_DATASET = "entry/data/data"


def load_xrd_metadata(xrd_dir, scan_number=203):
    xrd_files = sorted(Path(xrd_dir).glob(f"scan_{scan_number:04d}_*.h5"))
    xrd_files = [f for f in xrd_files if f.stat().st_size > 0]
    frame_map = []
    for fi, fp in enumerate(xrd_files):
        with h5py.File(fp, "r") as f:
            n_frames = f[H5_DATASET].shape[0]
        for j in range(n_frames):
            frame_map.append([fi, j])
    return [str(f) for f in xrd_files], frame_map, len(frame_map)


def load_positions(csv_path, n_total):
    x = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            x.append(float(row["X_Position"]))
    frame_x = np.full(n_total, np.nan)
    n = min(len(x), n_total)
    frame_x[:n] = x[:n]
    return frame_x


def build_scan_grid(frame_x, n_total, kernel=20, order=50):
    valid = ~np.isnan(frame_x)
    x = frame_x.copy()
    if np.any(~valid):
        x[~valid] = np.interp(np.where(~valid)[0], np.where(valid)[0], frame_x[valid])
    x_smooth = np.convolve(x, np.ones(kernel) / kernel, mode="same")
    x_max = argrelextrema(x_smooth, np.greater, order=order)[0]
    x_min = argrelextrema(x_smooth, np.less, order=order)[0]
    turns = np.sort(np.concatenate([x_max, x_min]))
    starts = np.concatenate([[0], turns])
    ends = np.concatenate([turns, [n_total]])
    row = np.zeros(n_total, dtype=int)
    col = np.zeros(n_total, dtype=int)
    for i in range(len(starts)):
        s, e = int(starts[i]), int(ends[i])
        row[s:e] = i
        c = np.arange(e - s)
        if i % 2 == 1:
            c = c[::-1]
        col[s:e] = c
    return row, col, int(row.max()) + 1, int(col.max()) + 1


def build_bin_mapping(n_rows, n_cols, bin_size, grid_to_frames):
    n_br = (n_rows + bin_size - 1) // bin_size
    n_bc = (n_cols + bin_size - 1) // bin_size
    mapping = {}
    for br in range(n_br):
        for bc in range(n_bc):
            frames = []
            for dr in range(bin_size):
                for dc in range(bin_size):
                    r, c = br * bin_size + dr, bc * bin_size + dc
                    if r < n_rows and c < n_cols:
                        frames.extend(grid_to_frames.get((r, c), []))
            if frames:
                mapping[f"{br}_{bc}"] = frames
    return mapping, n_br, n_bc


def main():
    parser = argparse.ArgumentParser(description="Generate grid_mapping.json for any bin size")
    parser.add_argument("--bin-size", type=int, default=5)
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: grid_mapping_{N}x{N}.json)")
    parser.add_argument("--project-root", type=str, default=None,
                        help="Path to project root (parent of raw_scans/)")
    args = parser.parse_args()

    if args.output is None:
        args.output = f"grid_mapping_{args.bin_size}x{args.bin_size}.json"

    if args.project_root is None:
        args.project_root = str(Path(__file__).resolve().parent.parent)
    project_root = Path(args.project_root)

    xrd_dir = project_root / "raw_scans" / "Scan_0203" / "XRD"
    pos_csv = project_root / "203 other" / "Scan_0203_position.csv"

    print(f"Loading scan metadata from {xrd_dir} ...")
    xrd_files, frame_map, n_total = load_xrd_metadata(xrd_dir)
    print(f"  {n_total} frames across {len(xrd_files)} H5 files")

    print("Computing scan grid from positions ...")
    frame_x = load_positions(pos_csv, n_total)
    grid_row, grid_col, n_rows, n_cols = build_scan_grid(frame_x, n_total)
    print(f"  Scan grid: {n_rows} rows x {n_cols} cols")

    grid_to_frames = {}
    for gi in range(n_total):
        key = (int(grid_row[gi]), int(grid_col[gi]))
        grid_to_frames.setdefault(key, []).append(gi)

    print(f"Building bin mapping with bin_size={args.bin_size} ...")
    bins, n_bin_rows, n_bin_cols = build_bin_mapping(
        n_rows, n_cols, args.bin_size, grid_to_frames)
    print(f"  {len(bins)} bins ({n_bin_rows} x {n_bin_cols})")

    result = {
        "bin_size": args.bin_size,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "n_bin_rows": n_bin_rows,
        "n_bin_cols": n_bin_cols,
        "n_total_frames": n_total,
        "n_bins": len(bins),
        "h5_dataset": H5_DATASET,
        "xrd_files": xrd_files,
        "bins": bins,
        "frame_map": frame_map,
    }

    with open(args.output, "w") as f:
        json.dump(result, f)

    size_kb = Path(args.output).stat().st_size / 1024
    print(f"Wrote {args.output} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
