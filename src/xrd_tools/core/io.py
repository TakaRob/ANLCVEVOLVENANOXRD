"""
Data loading and preparation for xrd-tools.

Covers the three input formats used by the pipeline:
  - Raw per-frame detector scans (HDF5)
  - The 2-theta map (TIFF)
  - The grid mapping + reflections metadata (JSON / Python module)

It also owns the two "prepare" steps that turn raw frames into the inputs the
detector consumes:
  - :func:`generate_grid_mapping` — assign raw frames to a spatial bin grid
  - :func:`build_bins` — sum each bin's frames into a single binned HDF5 file

Both are de-hardcoded ports of the original ``generate_grid_mapping.py`` and
``prebuild_bins.py`` scripts.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import time
from pathlib import Path
from typing import Callable, Optional, Union

import h5py
import numpy as np

H5_DATASET = "entry/data/data"
DETECTOR_SHAPE = (1062, 1028)


# ─────────────────────────────────────────────────────────────────────
# Generic loaders
# ─────────────────────────────────────────────────────────────────────
def load_module(path: Union[str, Path]):
    """Dynamically import a .py file as a module (detector / reflections).

    The file's own directory is placed on ``sys.path`` during import so a
    detector can import sibling modules (e.g. ``noise_reduction_algorithms``).
    """
    import sys
    path = Path(path)
    parent = str(path.resolve().parent)
    added = parent not in sys.path
    if added:
        sys.path.insert(0, parent)
    try:
        spec = importlib.util.spec_from_file_location(path.stem, str(path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        if added:
            try:
                sys.path.remove(parent)
            except ValueError:
                pass


def load_tth_map(path: Union[str, Path]) -> np.ndarray:
    """Load the 2-theta map TIFF as float64."""
    import tifffile
    return tifffile.imread(str(path)).astype(np.float64)


def load_reflections(path: Union[str, Path]):
    """Load a reflections.py module; return (degs, deg_labels)."""
    mod = load_module(path)
    return mod.degs, mod.deg_labels


def load_grid_mapping(grid_mapping: Union[str, Path, dict]) -> dict:
    """Accept a path or an already-loaded dict; return the grid-mapping dict."""
    if isinstance(grid_mapping, dict):
        return grid_mapping
    with open(grid_mapping) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────
# Grid mapping generation (port of generate_grid_mapping.py)
# ─────────────────────────────────────────────────────────────────────
def load_xrd_metadata(xrd_dir: Union[str, Path], scan_number: int = 203):
    """List raw scan H5 files and build a flat frame index."""
    from scipy.signal import argrelextrema  # noqa: F401  (kept import locality clear)
    xrd_files = sorted(Path(xrd_dir).glob(f"scan_{scan_number:04d}_*.h5"))
    xrd_files = [f for f in xrd_files if f.stat().st_size > 0]
    frame_map = []
    for fi, fp in enumerate(xrd_files):
        with h5py.File(fp, "r") as f:
            n_frames = f[H5_DATASET].shape[0]
        for j in range(n_frames):
            frame_map.append([fi, j])
    return [str(f) for f in xrd_files], frame_map, len(frame_map)


def load_positions(csv_path: Union[str, Path], n_total: int) -> np.ndarray:
    """Read X positions from the scan position CSV, padded/truncated to n_total."""
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
    """Infer (row, col) for each frame from the serpentine position trace."""
    from scipy.signal import argrelextrema
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


def build_regular_grid(n_total, n_cols):
    """Synthesize a serpentine raster grid without positions.

    Assigns frames row-major into ``n_cols`` columns, reversing every other row
    (boustrophedon), matching how a regular step raster is collected. Use when
    no position CSV exists and the scan is a uniform grid.
    """
    n_cols = int(n_cols)
    n_rows = (n_total + n_cols - 1) // n_cols
    row = np.zeros(n_total, dtype=int)
    col = np.zeros(n_total, dtype=int)
    for i in range(n_total):
        r = i // n_cols
        c = i % n_cols
        if r % 2 == 1:
            c = n_cols - 1 - c
        row[i] = r
        col[i] = c
    return row, col, n_rows, n_cols


def build_bin_mapping(n_rows, n_cols, bin_size, grid_to_frames):
    """Group the per-pixel grid into bin_size x bin_size spatial bins."""
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


def generate_grid_mapping(
    xrd_dir: Union[str, Path],
    pos_csv: Optional[Union[str, Path]],
    bin_size: int,
    scan_number: int = 203,
    output: Optional[Union[str, Path]] = None,
    n_cols: Optional[int] = None,
    log: Callable[[str], None] = print,
) -> dict:
    """Build the grid-mapping dict (and optionally write it to ``output``).

    If ``pos_csv`` is provided and exists, the serpentine grid is inferred from
    the position trace. Otherwise, if ``n_cols`` is given, a regular serpentine
    grid is synthesized (the no-positions fallback for uniform step rasters).
    """
    log(f"Loading scan metadata from {xrd_dir} ...")
    xrd_files, frame_map, n_total = load_xrd_metadata(xrd_dir, scan_number)
    log(f"  {n_total} frames across {len(xrd_files)} H5 files")

    have_positions = pos_csv is not None and Path(pos_csv).exists()
    if have_positions:
        log("Computing scan grid from positions ...")
        frame_x = load_positions(pos_csv, n_total)
        grid_row, grid_col, n_rows, n_cols = build_scan_grid(frame_x, n_total)
        log(f"  Scan grid: {n_rows} rows x {n_cols} cols")
    elif n_cols:
        log(f"No position CSV; synthesizing a regular {n_cols}-column serpentine grid ...")
        log("  (assumes a uniform raster; provide positions for irregular scans)")
        grid_row, grid_col, n_rows, n_cols = build_regular_grid(n_total, n_cols)
        log(f"  Scan grid: {n_rows} rows x {n_cols} cols")
    else:
        raise FileNotFoundError(
            f"No position CSV found ({pos_csv}) and no --shape/n_cols given. "
            "Provide a position CSV or specify the raster shape to synthesize one."
        )

    grid_to_frames = {}
    for gi in range(n_total):
        key = (int(grid_row[gi]), int(grid_col[gi]))
        grid_to_frames.setdefault(key, []).append(gi)

    log(f"Building bin mapping with bin_size={bin_size} ...")
    bins, n_bin_rows, n_bin_cols = build_bin_mapping(
        n_rows, n_cols, bin_size, grid_to_frames)
    log(f"  {len(bins)} bins ({n_bin_rows} x {n_bin_cols})")

    result = {
        "bin_size": bin_size,
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

    if output is not None:
        with open(output, "w") as f:
            json.dump(result, f)
        size_kb = Path(output).stat().st_size / 1024
        log(f"Wrote {output} ({size_kb:.0f} KB)")

    return result


# ─────────────────────────────────────────────────────────────────────
# Binning (port of prebuild_bins.py)
# ─────────────────────────────────────────────────────────────────────
def get_compression_kwargs(compression: str):
    if compression == "gzip":
        return {"compression": "gzip", "compression_opts": 4}
    elif compression == "lz4":
        import hdf5plugin
        return hdf5plugin.LZ4()
    elif compression == "none":
        return {}
    else:
        raise ValueError(f"Unknown compression: {compression}")


def build_bins(
    grid_mapping: Union[str, Path, dict],
    output: Union[str, Path],
    bin_size: Optional[int] = None,
    compression: str = "gzip",
    log: Callable[[str], None] = print,
) -> Path:
    """Sum each bin's raw frames into a single binned HDF5 file.

    Output structure: one float32 dataset per bin keyed ``"row_col"``, with
    ``bin_size``, ``n_bin_rows``, ``n_bin_cols``, ``n_bins`` and
    ``detector_shape`` stored as file attributes.
    """
    gm = load_grid_mapping(grid_mapping)
    bin_size = bin_size or gm["bin_size"]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    bins = gm["bins"]
    frame_map = gm["frame_map"]
    xrd_files = gm["xrd_files"]
    h5_dataset = gm.get("h5_dataset", H5_DATASET)
    n_bins = len(bins)

    comp_kwargs = get_compression_kwargs(compression)
    log(f"Building {n_bins} bin images ({bin_size}x{bin_size}) -> {output}")
    log(f"  Compression: {compression}")

    h5_handles: dict = {}
    out = h5py.File(str(output), "w")
    out.attrs["bin_size"] = bin_size
    out.attrs["n_bin_rows"] = gm["n_bin_rows"]
    out.attrs["n_bin_cols"] = gm["n_bin_cols"]
    out.attrs["n_bins"] = n_bins
    out.attrs["detector_shape"] = list(DETECTOR_SHAPE)

    t0 = time.time()
    try:
        for i, (bin_key, frame_indices) in enumerate(sorted(bins.items())):
            by_file: dict = {}
            for gi in frame_indices:
                fi, fj = frame_map[gi]
                by_file.setdefault(fi, []).append(fj)

            summed = None
            for fi, frame_list in by_file.items():
                if fi not in h5_handles:
                    h5_handles[fi] = h5py.File(xrd_files[fi], "r")
                ds = h5_handles[fi][h5_dataset]
                for fj in frame_list:
                    frame = ds[fj].astype(np.float64)
                    summed = frame if summed is None else summed + frame

            if summed is not None:
                summed[summed < 0] = 0
                summed[summed > 1e9] = 0
                out.create_dataset(bin_key, data=summed.astype(np.float32), **comp_kwargs)

            if (i + 1) % 100 == 0 or (i + 1) == n_bins:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (n_bins - i - 1) / rate if rate > 0 else 0
                log(f"  [{i+1}/{n_bins}] {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining")
    finally:
        for fh in h5_handles.values():
            fh.close()
        out.close()

    size_mb = os.path.getsize(output) / 1024 / 1024
    log(f"Done! {output}: {size_mb:.0f} MB ({size_mb/1024:.1f} GB)")
    return output
