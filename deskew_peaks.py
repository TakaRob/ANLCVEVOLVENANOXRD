#!/usr/bin/env python3
"""
deskew_peaks.py — position-CSV deskew pre-filter for the shape stage.

A small, **toggleable** in-between step that sits between `xrd-app peaks` and
`xrd-app shapes`. It re-grids the per-bin peaks onto a *faithful* scan lattice using
the **true stage positions** from the scan position CSV, then writes a standard
peaks JSON + grid_mapping that you feed straight into the unchanged `xrd-app shapes`.

Why
---
The peak grid `(row, col)` is reconstructed from the position trace by a serpentine
turn-finder; alternate rows are scanned in opposite directions, so stage backlash
misregisters the *column* index between adjacent rows. A single Bragg feature then
fragments into horizontal slices (worst at 1x1). This filter bypasses the
reconstruction: it assigns each bin's `(row, col)` directly from its mean true
(X, Y) — surefire, because the CSV records where the beam actually was.

On / off
--------
ON : run this script, then point `shapes` at the *_deskew_* outputs it writes.
OFF: don't run it; `shapes` uses the original peaks as usual.

Scope
-----
**Intended for the finest grid (1x1 / raw)**, where the registration error is fully
present. At 2x2+ binning has already averaged the backlash out, so the grid is
already faithful and re-gridding only adds rounding noise — leave the filter OFF for
binned levels. (Validated on Scan_0203: 1x1 horizontal-slice fraction 53% -> 34%,
matching a direct true-position link; 2x2/3x3 not improved.)

Usage
-----
    python deskew_peaks.py --root <project> --scan Scan_0203 --bin-size 1 \\
        --peak-algo 5x5_tophat_band_adaptive_snr
    # then, exactly as printed at the end:
    xrd-app shapes --root <project> --scan Scan_0203 --bin-size 1 \\
        --peak-algo 5x5_tophat_band_adaptive_snr_deskew \\
        --grid-mapping <project>/Metadata/Scan_0203/grid_mapping_deskew_1x1.json \\
        --algorithm gaussian
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def _load_positions(csv_path):
    X, Y = [], []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            X.append(float(r["X_Position"])); Y.append(float(r["Y_Position"]))
    return np.array(X), np.array(Y)


def regrid(peaks_by_bin, grid_bins, X, Y, n_rows, n_cols, log=print):
    """Re-key bins onto an n_rows x n_cols lattice from true (X, Y).

    Auto-detects which true axis is the row vs col axis (and its sign) by correlating
    the original indices with the true positions, so it works regardless of the
    scan's fast/slow-axis convention. Returns (new_peaks_by_bin, new_grid_mapping).
    """
    binXY = {}
    for bk, frames in grid_bins.items():
        idx = [fr for fr in frames if fr < len(X)]
        if idx:
            binXY[bk] = (float(np.mean(X[idx])), float(np.mean(Y[idx])))
    if not binXY:
        raise SystemExit("No bins could be matched to positions — check the CSV / grid mapping.")

    ks = list(binXY)
    R = np.array([int(k.split("_")[0]) for k in ks])
    C = np.array([int(k.split("_")[1]) for k in ks])
    BX = np.array([binXY[k][0] for k in ks])
    BY = np.array([binXY[k][1] for k in ks])

    row_is_X = abs(np.corrcoef(R, BX)[0, 1]) >= abs(np.corrcoef(R, BY)[0, 1])
    rowv, colv = (BX, BY) if row_is_X else (BY, BX)
    sr = np.sign(np.corrcoef(R, rowv)[0, 1]) or 1.0
    sc = np.sign(np.corrcoef(C, colv)[0, 1]) or 1.0
    log(f"  axis map: row <- {'X' if row_is_X else 'Y'} (sign {int(sr):+d}), "
        f"col <- {'Y' if row_is_X else 'X'} (sign {int(sc):+d})")

    def scale(v, s, n):
        v = s * v
        lo, hi = np.percentile(v, 0.2), np.percentile(v, 99.8)
        return np.clip(np.round((v - lo) / (hi - lo) * (n - 1)), 0, n - 1).astype(int)

    newR, newC = scale(rowv, sr, n_rows), scale(colv, sc, n_cols)
    remap = {k: (int(newR[i]), int(newC[i])) for i, k in enumerate(ks)}

    new_pbb, new_bins = defaultdict(list), defaultdict(list)
    for bk, ps in peaks_by_bin.items():
        if bk in remap:
            nr, nc = remap[bk]; new_pbb[f"{nr}_{nc}"].extend(ps)
    for bk, frames in grid_bins.items():
        if bk in remap:
            nr, nc = remap[bk]; new_bins[f"{nr}_{nc}"].extend(frames)
    collisions = len(remap) - len(new_bins)
    log(f"  remapped {len(remap)} bins -> {len(new_bins)} cells "
        f"({collisions} merged by collision)")
    return dict(new_pbb), {"n_bin_rows": n_rows, "n_bin_cols": n_cols, "bins": dict(new_bins)}


def main():
    ap = argparse.ArgumentParser(description="Deskew peaks by true CSV positions, before 'xrd-app shapes'.")
    ap.add_argument("--root", default=".", help="Project root")
    ap.add_argument("--scan", default=None, help="Scan name/number (defaults to config scan)")
    ap.add_argument("--bin-size", type=int, default=1, help="Bin size (intended: 1)")
    ap.add_argument("--peak-algo", required=True, help="Peak set name in Labels/<scan>/ (the detector stem)")
    ap.add_argument("--positions", default=None, help="Override position CSV path")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.root).resolve()))
    sys.path.insert(0, str(Path.cwd()))
    from xrd_app.config import DataManager
    dm = DataManager(args.root, scan=args.scan)
    bs = args.bin_size

    peaks_path = dm.peaks_json(args.peak_algo, bs)
    grid_path = dm.grid_mapping(bin_size=bs)
    pos_path = Path(args.positions) if args.positions else dm.position_csv()
    for label, p in [("peaks", peaks_path), ("grid mapping", grid_path), ("positions CSV", pos_path)]:
        if not p or not Path(p).exists():
            raise SystemExit(f"Error: {label} not found: {p}")

    if bs != 1:
        print(f"NOTE: bin-size {bs} — binning already deskews; this filter is meant for 1x1. "
              "Continuing anyway, but expect little/no benefit.")

    peaks = json.load(open(peaks_path))
    pbb = peaks.get("peaks_by_bin", peaks)
    gm = json.load(open(grid_path))
    n_rows, n_cols = gm["n_bin_rows"], gm["n_bin_cols"]
    X, Y = _load_positions(pos_path)
    print(f"Deskewing {sum(len(v) for v in pbb.values())} peaks across {len(pbb)} bins "
          f"on a {n_rows}x{n_cols} grid using {pos_path.name} ...")

    new_pbb, new_grid = regrid(pbb, gm["bins"], X, Y, n_rows, n_cols)
    # carry over frame_map / xrd_files so the deskewed grid is a complete mapping
    for k in ("xrd_files", "frame_map", "bin_size", "scan"):
        if k in gm:
            new_grid[k] = gm[k]

    out_algo = f"{args.peak_algo}_deskew"
    out_peaks = dm.peaks_json(out_algo, bs)
    out_grid = dm.grid_mapping(bin_size=bs).with_name(f"grid_mapping_deskew_{bs}x{bs}.json")

    peaks_out = dict(peaks)
    peaks_out["peaks_by_bin"] = new_pbb
    peaks_out["n_peaks"] = sum(len(v) for v in new_pbb.values())
    peaks_out["n_bins_with_peaks"] = len(new_pbb)
    peaks_out["algorithm"] = out_algo
    peaks_out["deskew"] = {"source": "position_csv", "csv": str(pos_path)}

    out_peaks.parent.mkdir(parents=True, exist_ok=True)
    out_grid.parent.mkdir(parents=True, exist_ok=True)
    json.dump(peaks_out, open(out_peaks, "w"))
    json.dump(new_grid, open(out_grid, "w"))
    print(f"\nWrote deskewed peaks -> {out_peaks}")
    print(f"Wrote deskewed grid  -> {out_grid}")
    print("\nNow run shapes on the deskewed data:")
    print(f"  xrd-app shapes --root {args.root} --scan {dm.scan_name} --bin-size {bs} \\\n"
          f"      --peak-algo {out_algo} --grid-mapping {out_grid} --algorithm gaussian")


if __name__ == "__main__":
    main()
