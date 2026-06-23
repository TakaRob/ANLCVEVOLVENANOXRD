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
# Small utilities
# ─────────────────────────────────────────────────────────────────────
def atomic_write_json(path, data, indent: int = 2) -> Path:
    """Write JSON to a temp file then atomically rename.

    Prevents a concurrent reader (e.g. a viewer tab) from seeing a half-written
    catalog while a job is writing it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=indent)
    os.replace(tmp, path)
    return path


def is_wsl() -> bool:
    try:
        with open("/proc/version") as f:
            return "microsoft" in f.read().lower()
    except Exception:
        return False


def slow_mount_warning(path) -> Optional[str]:
    """Warn if ``path`` resolves onto a slow Windows mount under WSL."""
    p = str(Path(path).resolve())
    if is_wsl() and p.startswith("/mnt/"):
        return (f"{path} is on a Windows mount — binned-HDF5 IO will be slow under "
                "WSL. Consider a native-WSL path or a fast mount for Binned/.")
    return None


# ─────────────────────────────────────────────────────────────────────
# Generic loaders
# ─────────────────────────────────────────────────────────────────────
def load_module(path: Union[str, Path]):
    """Dynamically import a .py file as a module (detector / reflections).

    The file's own directory is placed on ``sys.path`` during import so a
    detector can import sibling modules. The shared ``NoiseReduction/`` library
    (``xrd_app/NoiseReduction``) is also added, so any detector — wherever it
    lives in the flattened algorithm library — can still
    ``from noise_reduction_algorithms import ...``.
    """
    import sys
    path = Path(path)
    # Shared library dirs so a detector can import sibling/library modules no
    # matter where it lives: its own dir, the flat PeakAlgorithms/ root (so an
    # automated algo in its own sub-folder can import a base detector), and the
    # NoiseReduction/ library.
    pkg = Path(__file__).resolve().parent.parent
    dirs = [str(path.resolve().parent)]
    for extra in (pkg / "PeakAlgorithms", pkg / "CombinedAlgorithms",
                  pkg / "NoiseReduction"):
        if extra.is_dir():
            dirs.append(str(extra))
    added = [d for d in dirs if d not in sys.path]
    for d in added:
        sys.path.insert(0, d)
    try:
        spec = importlib.util.spec_from_file_location(path.stem, str(path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        for d in added:
            try:
                sys.path.remove(d)
            except ValueError:
                pass


def load_tth_map(path: Union[str, Path]) -> np.ndarray:
    """Load the 2-theta map TIFF as float64."""
    import tifffile
    return tifffile.imread(str(path)).astype(np.float64)


# ─────────────────────────────────────────────────────────────────────
# Scan discovery + validation (Setup / scan-detect)
# ─────────────────────────────────────────────────────────────────────
import re as _re

_H5_EXTS = (".h5", ".hdf5")


def _h5_files(d: Path) -> list:
    """Sorted, non-empty HDF5 files directly inside ``d``."""
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir()
                  if p.is_file() and p.suffix.lower() in _H5_EXTS and p.stat().st_size > 0)


def _scan_name_from(files: list, scan_dir: Path) -> str:
    """Canonical Scan_NNNN name from the file names, else the dir name."""
    for fp in files:
        m = _re.search(r"scan[_-]?(\d+)", fp.stem, _re.IGNORECASE)
        if m:
            return f"Scan_{int(m.group(1)):04d}"
    m = _re.search(r"scan[_-]?(\d+)", scan_dir.name, _re.IGNORECASE)
    if m:
        return f"Scan_{int(m.group(1)):04d}"
    return scan_dir.name


def detect_frame_shape(scan_dir: Union[str, Path]) -> Optional[list]:
    """Read ONE frame's (H, W) from a scan dir. Never hard-codes the shape."""
    scan_dir = Path(scan_dir)
    frames_dir = scan_dir / "XRD" if (scan_dir / "XRD").is_dir() else scan_dir
    files = _h5_files(frames_dir)
    if not files:
        return None
    with h5py.File(files[0], "r") as f:
        if H5_DATASET not in f:
            return None
        return list(f[H5_DATASET].shape[-2:])


def _summarize_scan(files: list, deep: bool = False) -> tuple:
    """Return (n_frames, frame_shape, frames_estimated, warnings).

    Fast mode (``deep=False``, the default) opens only the FIRST file to read the
    frame shape + per-file frame count, then estimates the total as
    ``frames_in_first × n_files`` — essential on slow WSL/OneDrive mounts where
    opening every header is prohibitively slow. ``deep=True`` opens every file to
    sum exact frame counts and catch corrupt files / inconsistent shapes.
    """
    warnings, shapes, counts = [], set(), []
    total, frame_shape, estimated = 0, None, False

    probe = files if deep else files[:1]
    for fp in probe:
        try:
            with h5py.File(fp, "r") as f:
                if H5_DATASET not in f:
                    warnings.append(f"{fp.name}: no '{H5_DATASET}' dataset")
                    continue
                ds = f[H5_DATASET]
                total += int(ds.shape[0])
                counts.append(int(ds.shape[0]))
                shapes.add(tuple(int(x) for x in ds.shape[-2:]))
                if frame_shape is None:
                    frame_shape = list(ds.shape[-2:])
        except Exception as e:  # corrupt / unreadable file
            warnings.append(f"{fp.name}: unreadable ({e})")

    if not deep and counts:
        total = counts[0] * len(files)   # estimate from the first file
        estimated = True
    if deep and len(shapes) > 1:
        warnings.append(f"inconsistent frame shapes: {sorted(shapes)}")
    if deep and counts and len(set(counts)) > 1:
        warnings.append(f"varying frames/file: min={min(counts)} max={max(counts)}")
    return total, frame_shape, estimated, warnings


def scan_info(scan_dir: Union[str, Path], deep: bool = False) -> dict:
    """Summarize one scan directory.

    Returns ``{name, dir, frames_dir, n_files, n_frames, frames_estimated,
    shape, warnings}``. Honors a ``XRD/`` subdirectory if present. ``deep=False``
    is fast (samples the first file); see :func:`_summarize_scan`.
    """
    scan_dir = Path(scan_dir)
    frames_dir = scan_dir / "XRD" if (scan_dir / "XRD").is_dir() else scan_dir
    files = _h5_files(frames_dir)
    n_frames, shape, estimated, warnings = _summarize_scan(files, deep=deep)
    if not files:
        warnings = ["no HDF5 frames found"]
    return {
        "name": _scan_name_from(files, scan_dir),
        "dir": str(scan_dir.resolve()),
        "frames_dir": str(frames_dir.resolve()),
        "n_files": len(files),
        "n_frames": n_frames,
        "frames_estimated": estimated,
        "shape": shape,
        "warnings": warnings,
    }


def discover_scans(path: Union[str, Path], deep: bool = False) -> list:
    """Discover scans from a selection.

    - A single ``.h5/.hdf5`` file  → its parent directory (one scan).
    - A directory of frames        → one scan (that directory).
    - A directory of ``Scan_*/``   → every scan subdirectory.

    ``deep=False`` (default) samples the first file per scan for speed.
    """
    path = Path(path)
    if path.is_file():
        return [scan_info(path.parent, deep=deep)]
    if _h5_files(path) or (path / "XRD").is_dir():
        return [scan_info(path, deep=deep)]
    out = []
    for sub in sorted(path.iterdir()):
        if sub.is_dir() and (_h5_files(sub) or (sub / "XRD").is_dir()):
            out.append(scan_info(sub, deep=deep))
    return out


def validate_scan(info: dict, expected_shape: Optional[list] = None) -> list:
    """Return a list of problem strings for a scan (empty = OK)."""
    problems = list(info.get("warnings", []))
    if info.get("n_files", 0) == 0:
        problems.append("no HDF5 files")
    if info.get("n_frames", 0) == 0:
        problems.append("no readable frames")
    if expected_shape and info.get("shape") and list(info["shape"]) != list(expected_shape):
        problems.append(f"frame shape {info['shape']} != project {list(expected_shape)}")
    return problems


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


def sum_binned_image(h5_path: Union[str, Path], max_bins: Optional[int] = None,
                     progress: Optional[Callable[[int, int], None]] = None) -> np.ndarray:
    """Sum every bin in a binned HDF5 into a single 'fully binned' image."""
    with h5py.File(str(h5_path), "r") as f:
        keys = sorted(f.keys(), key=lambda k: (int(k.split("_")[0]), int(k.split("_")[1])))
        if max_bins:
            keys = keys[:max_bins]
        acc = None
        n = len(keys)
        for i, k in enumerate(keys):
            a = np.clip(f[k][:].astype(np.float64), 0, 1e9)
            acc = a if acc is None else acc + a
            if progress is not None:
                progress(i + 1, n)
    return acc if acc is not None else np.zeros((1, 1))


def radial_profile(image: np.ndarray, tth_map: np.ndarray, n_bins: int = 600):
    """Azimuthally-averaged intensity vs 2θ. Returns (centers_deg, mean_intensity)."""
    flat_t = tth_map.ravel()
    flat_i = image.ravel()
    edges = np.linspace(float(flat_t.min()), float(flat_t.max()), n_bins + 1)
    sum_i, _ = np.histogram(flat_t, bins=edges, weights=flat_i)
    cnt, _ = np.histogram(flat_t, bins=edges)
    profile = sum_i / np.maximum(cnt, 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, profile


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
        # Atomic write so a ctrl+C / crash can't leave a truncated grid mapping
        # (which `build_bins` later reads) behind.
        atomic_write_json(output, result, indent=None)
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

    # Write to a temporary file and atomically rename it onto `output` only
    # after a fully successful build. A ctrl+C / crash mid-build then leaves the
    # (discarded) .tmp file corrupt instead of the real output, so any previous
    # good bins file survives and the GUI never reads a half-written file.
    tmp = output.with_name(output.name + ".tmp")
    h5_handles: dict = {}
    out = h5py.File(str(tmp), "w")
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
        out.close()
        out = None
        # Atomic publish: rename the completed temp file onto the real path.
        os.replace(tmp, output)
    finally:
        for fh in h5_handles.values():
            fh.close()
        if out is not None:
            # Build was interrupted (ctrl+C, exception): drop the partial file.
            out.close()
            try:
                tmp.unlink()
            except OSError:
                pass

    size_mb = os.path.getsize(output) / 1024 / 1024
    log(f"Done! {output}: {size_mb:.0f} MB ({size_mb/1024:.1f} GB)")
    return output


# ─────────────────────────────────────────────────────────────────────
# Bin image source — per-bin summed images from a built h5 OR raw frames
# ─────────────────────────────────────────────────────────────────────
# Shown by the GUIs when they read pixel images straight from raw frames
# (no prebuilt xrd_NxN_bins.h5). Raw is correct but slow — re-binning every
# frame on demand — so the GUIs surface this and gate it behind a second press.
RAW_FALLBACK_NOTE = "raw frames (no binning) — slower; build bins in Programs to speed up"


def _bin_sort_key(k: str):
    a, b = k.split("_")
    return (int(a), int(b))


def sum_raw_frames(xrd_files, frame_map, frame_indices) -> Optional[np.ndarray]:
    """Sum the raw detector frames at ``frame_indices`` into one image.

    ``frame_map[gi]`` is ``[file_index, frame_index]`` into ``xrd_files`` (the
    grid-mapping format). Frames are grouped by file so each H5 is opened once.
    Port of the labeling tool's ``load_and_sum_frames``.
    """
    summed = None
    by_file = {}
    for gi in frame_indices:
        fi, fj = frame_map[gi]
        by_file.setdefault(fi, []).append(fj)
    for fi, frame_list in by_file.items():
        with h5py.File(xrd_files[fi], "r") as f:
            ds = f[H5_DATASET]
            for fj in sorted(frame_list):
                frame = ds[fj].astype(np.float64)
                summed = frame if summed is None else summed + frame
    if summed is not None:
        summed[summed < 0] = 0
        summed[summed > 1e9] = 0
    return summed


class BinImageSource:
    """Per-bin summed images from either a built h5 or raw frames.

    Common interface used by the image GUIs so they don't care which backing
    store provides the pixels. Use :func:`open_bin_source` to construct one.
    """

    is_raw = False

    def keys(self) -> list:
        raise NotImplementedError

    def image(self, key: str) -> Optional[np.ndarray]:
        raise NotImplementedError

    def sum_all(self, max_bins: Optional[int] = None,
                progress: Optional[Callable[[int, int], None]] = None) -> np.ndarray:
        raise NotImplementedError

    def close(self):
        pass

    # Mapping-style access so callers written against an h5py.File handle
    # (``key in h5`` / ``h5[key][:]``) work unchanged against either backend.
    def __contains__(self, key) -> bool:
        raise NotImplementedError

    def __getitem__(self, key):
        img = self.image(key)
        if img is None:
            raise KeyError(key)
        return img


class _H5Source(BinImageSource):
    """Read per-bin images from a prebuilt ``xrd_NxN_bins.h5``."""

    is_raw = False

    def __init__(self, h5_path):
        self._path = str(h5_path)
        self._f = h5py.File(self._path, "r")

    def keys(self) -> list:
        return sorted(self._f.keys(), key=_bin_sort_key)

    def __contains__(self, key) -> bool:
        return key in self._f

    def image(self, key: str) -> Optional[np.ndarray]:
        if key not in self._f:
            return None
        return np.clip(self._f[key][:].astype(np.float64), 0, 1e9)

    def sum_all(self, max_bins=None, progress=None) -> np.ndarray:
        keys = self.keys()
        if max_bins:
            keys = keys[:max_bins]
        acc = None
        n = len(keys)
        for i, k in enumerate(keys):
            a = np.clip(self._f[k][:].astype(np.float64), 0, 1e9)
            acc = a if acc is None else acc + a
            if progress is not None:
                progress(i + 1, n)
        return acc if acc is not None else np.zeros((1, 1))

    def close(self):
        try:
            self._f.close()
        except Exception:
            pass


class _RawSource(BinImageSource):
    """Bin raw frames on demand — used when no binned h5 exists.

    Bins are resolved from a saved ``grid_mapping_NxN.json`` when present,
    otherwise built in-memory from the scan positions (or a synthesized raster).
    """

    is_raw = True

    def __init__(self, dm, bin_size, scan=None, n_cols=None):
        self.bin_size = bin_size
        gm_path = dm.grid_mapping(bin_size=bin_size, scan=scan)
        if gm_path and Path(gm_path).exists():
            with open(gm_path) as f:
                gm = json.load(f)
            self._bins = gm["bins"]
            self._xrd_files = gm["xrd_files"]
            self._frame_map = gm["frame_map"]
        else:
            xrd_dir = dm.xrd_frames_dir(scan=scan)
            if not xrd_dir or not Path(xrd_dir).exists():
                raise FileNotFoundError(
                    f"No binned file and no raw frames directory ({xrd_dir}) to "
                    "fall back to. Link the raw scan in Setup or build bins.")
            scan_no = dm.scan_number(scan) or 203
            self._xrd_files, self._frame_map, n_total = load_xrd_metadata(
                xrd_dir, scan_number=scan_no)
            pos = dm.position_csv(scan=scan)
            if pos and Path(pos).exists():
                frame_x = load_positions(pos, n_total)
                grid_row, grid_col, n_rows, n_cols2 = build_scan_grid(frame_x, n_total)
            elif n_cols:
                grid_row, grid_col, n_rows, n_cols2 = build_regular_grid(n_total, n_cols)
            else:
                raise FileNotFoundError(
                    "No grid mapping, no position CSV, and no raster shape — "
                    "cannot assign raw frames to bins. Run 'xrd-app grid' or link positions.")
            grid_to_frames = {}
            for gi in range(n_total):
                grid_to_frames.setdefault(
                    (int(grid_row[gi]), int(grid_col[gi])), []).append(gi)
            self._bins, _, _ = build_bin_mapping(
                n_rows, n_cols2, bin_size, grid_to_frames)
        self._cache = {}

    def keys(self) -> list:
        return sorted(self._bins.keys(), key=_bin_sort_key)

    def __contains__(self, key) -> bool:
        return key in self._bins

    def image(self, key: str) -> Optional[np.ndarray]:
        if key not in self._bins:
            return None
        if key in self._cache:
            return self._cache[key]
        img = sum_raw_frames(self._xrd_files, self._frame_map, self._bins[key])
        # Bounded cache so panning across many bins doesn't grow without limit.
        if len(self._cache) < 64 and img is not None:
            self._cache[key] = img
        return img

    def sum_all(self, max_bins=None, progress=None) -> np.ndarray:
        keys = self.keys()
        if max_bins:
            keys = keys[:max_bins]
        acc = None
        n = len(keys)
        for i, k in enumerate(keys):
            a = self.image(k)
            if a is not None:
                acc = a if acc is None else acc + a
            if progress is not None:
                progress(i + 1, n)
        return acc if acc is not None else np.zeros((1, 1))


def open_bin_source(dm, bin_size, scan=None, n_cols=None) -> BinImageSource:
    """Open the best per-bin image source for a scan + bin size.

    Uses the prebuilt ``xrd_NxN_bins.h5`` when it exists (fast); otherwise falls
    back to summing raw frames on demand (``is_raw`` True, slower). 1×1 has no
    binned file by convention and always uses the raw source.
    """
    if bin_size != 1:
        h5 = dm.bins_h5(bin_size, scan=scan)
        if h5 and os.path.exists(h5):
            return _H5Source(h5)
    return _RawSource(dm, bin_size, scan=scan, n_cols=n_cols)
