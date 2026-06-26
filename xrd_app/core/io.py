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
def scan_h5_files(xrd_dir: Union[str, Path], scan_number: int) -> list:
    """Sorted, non-empty ``scan_NNNN_*.h5`` frame files for ``scan_number``.

    Matched case-insensitively and tolerant of (non-)zero-padding so it works on
    case-sensitive beamline mounts and across export naming (``scan_0203_*.h5`` /
    ``Scan_203_*.h5``). The trailing ``_`` after the number prevents ``203`` from
    matching ``2030``.
    """
    d = Path(xrd_dir)
    if not d.is_dir():
        return []
    pat = _re.compile(rf"scan[_-]?0*{scan_number}_.*\.h5$", _re.IGNORECASE)
    files = [p for p in d.iterdir()
             if p.is_file() and pat.match(p.name) and p.stat().st_size > 0]
    return sorted(files, key=lambda p: p.name.lower())


def has_raw_frames(xrd_dir: Union[str, Path], scan_number: int) -> bool:
    """True if ``xrd_dir`` holds at least one non-empty frame file for the scan.

    Cheap completeness probe (no HDF5 open) for skipping incomplete scans — many
    ``Scan_NNNN/`` dirs on the beamline mount have no ``XRD/`` files yet.
    """
    return len(scan_h5_files(xrd_dir, scan_number)) > 0


def load_xrd_metadata(xrd_dir: Union[str, Path], scan_number: int = 203):
    """List raw scan H5 files and build a flat frame index."""
    xrd_files = scan_h5_files(xrd_dir, scan_number)
    frame_map = []
    for fi, fp in enumerate(xrd_files):
        with h5py.File(fp, "r") as f:
            n_frames = f[H5_DATASET].shape[0]
        for j in range(n_frames):
            frame_map.append([fi, j])
    return [str(f) for f in xrd_files], frame_map, len(frame_map)


# A recreated (file-per-row) position CSV carries this as its first line, so we
# can tell a synthetic reconstruction apart from a real SOCKETSERVER export.
RECREATED_CSV_MARKER = "# xrd-app coordinate_source=file_per_row"


def _uncommented(fh):
    """Yield CSV lines, dropping any leading ``#`` comment/marker lines."""
    for line in fh:
        if not line.lstrip().startswith("#"):
            yield line


def is_recreated_csv(csv_path: Union[str, Path]) -> bool:
    """True if ``csv_path`` is a synthetic file-per-row CSV we wrote (has the marker)."""
    try:
        with open(csv_path) as f:
            return f.readline().lstrip().startswith(RECREATED_CSV_MARKER)
    except OSError:
        return False


def load_positions(csv_path: Union[str, Path], n_total: int) -> np.ndarray:
    """Read X positions from the scan position CSV, padded/truncated to n_total."""
    x = []
    with open(csv_path) as f:
        reader = csv.DictReader(_uncommented(f))
        for row in reader:
            x.append(float(row["X_Position"]))
    frame_x = np.full(n_total, np.nan)
    n = min(len(x), n_total)
    frame_x[:n] = x[:n]
    return frame_x


def load_positions_xy(csv_path: Union[str, Path], n_total: int):
    """Read X *and* Y positions from the scan position CSV → (frame_x, frame_y).

    Companion to :func:`load_positions` (X-only, kept for back-compat). Both
    arrays are padded/truncated to ``n_total``. A ``frame_y`` that is all-NaN
    means the CSV has no ``Y_Position`` column — callers should then fall back to
    the serpentine (X-only) grid.
    """
    x, y = [], []
    with open(csv_path) as f:
        reader = csv.DictReader(_uncommented(f))
        has_y = bool(reader.fieldnames) and "Y_Position" in reader.fieldnames
        for row in reader:
            x.append(float(row["X_Position"]))
            if has_y:
                y.append(float(row["Y_Position"]))
    frame_x = np.full(n_total, np.nan)
    frame_y = np.full(n_total, np.nan)
    nx = min(len(x), n_total)
    frame_x[:nx] = x[:nx]
    if has_y:
        ny = min(len(y), n_total)
        frame_y[:ny] = y[:ny]
    return frame_x, frame_y


def _interp_nan(v: np.ndarray) -> np.ndarray:
    """Fill NaNs by linear interpolation over index (no-op if none / all NaN)."""
    v = np.asarray(v, dtype=float).copy()
    bad = np.isnan(v)
    if bad.any() and (~bad).any():
        v[bad] = np.interp(np.where(bad)[0], np.where(~bad)[0], v[~bad])
    return v


def _scale_to_index(v, sign, n) -> np.ndarray:
    """Snap a continuous position axis onto ``n`` integer lattice indices.

    Robust to outliers ("whisker" scans) via a 0.2/99.8-percentile span rather
    than raw min/max, so a single stray frame doesn't compress the rest of the
    lattice. Ported from ``deskew_peaks.py::regrid``'s inner ``scale``.
    """
    v = sign * np.asarray(v, dtype=float)
    lo, hi = np.percentile(v, 0.2), np.percentile(v, 99.8)
    if hi <= lo:
        return np.zeros(len(v), dtype=int)
    idx = np.round((v - lo) / (hi - lo) * (n - 1))
    return np.clip(idx, 0, n - 1).astype(int)


def _corr(a, b) -> float:
    """Pearson r, but 0 for a constant input (avoids NaN in axis detection)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def assign_grid_from_positions(frame_x, frame_y, frame_map=None,
                               log: Callable[[str], None] = print):
    """Per-frame ``(row, col)`` for a scan, using true stage positions (X, Y).

    For a clean one-file-per-row raster (``frame_map`` provided) the grid comes
    straight from the acquisition layout — **rows = HDF5 file index, columns =
    within-file rank (the *commanded* fast-axis position)** — and the real (X, Y)
    are used *only* to orient the axes to the historical device-map convention.
    Columns are deliberately **not** re-snapped to the encoder readout: the
    even/odd-row position divergence on these scans is serpentine *backlash* (an
    encoder artefact at the same commanded position), so "correcting" it would
    scatter a feature's rows across columns and fragment it. Aligning by commanded
    rank keeps features intact, gives exact dimensions, and never merges frames.

    When ``frame_map`` is absent or the scan is *not* one-file-per-row (fly-scans,
    multi-row files, irregular rasters), it falls back to snapping both axes onto a
    serpentine turn-counted lattice from the positions (the de-skew used for scans
    where file index ≠ scan row).

    Returns ``(grid_row, grid_col, n_rows, n_cols)``.
    """
    n_total = len(frame_x)
    x = _interp_nan(frame_x)
    y = _interp_nan(frame_y)

    file_per_row = frame_map is not None and is_file_per_row(frame_map)[0]
    if not file_per_row:
        # Irregular / non-file-per-row scan: snap both axes onto a turn-counted
        # lattice (best effort when file index ≠ scan row).
        ref_row, ref_col, n_rows, n_cols = build_scan_grid(frame_x, n_total)
        row_is_X = abs(_corr(ref_row, x)) >= abs(_corr(ref_row, y))
        rowv, colv = (x, y) if row_is_X else (y, x)
        sr = np.sign(_corr(ref_row, rowv)) or 1.0
        sc = np.sign(_corr(ref_col, colv)) or 1.0
        log(f"  de-skew (turn-counted, non-file-per-row): row <- "
            f"{'X' if row_is_X else 'Y'} (sign {int(sr):+d}), "
            f"col <- {'Y' if row_is_X else 'X'} (sign {int(sc):+d})")
        grid_row = _scale_to_index(rowv, sr, n_rows)
        grid_col = _scale_to_index(colv, sc, n_cols)
        return grid_row, grid_col, int(grid_row.max()) + 1, int(grid_col.max()) + 1

    # File-per-row: rows & columns straight from the layout (commanded position).
    ref_row, ref_col, n_rows, n_cols = build_grid_from_frame_map(
        frame_map, log=lambda *a: None)
    row_is_X = abs(_corr(ref_row, x)) >= abs(_corr(ref_row, y))
    rowv, colv = (x, y) if row_is_X else (y, x)
    grid_row = ref_row.astype(int)
    grid_col = ref_col.astype(int)

    # Orient to the historical device-map convention. The file/serpentine
    # acquisition direction (ref_col) can point opposite to the physical axis the
    # previous coordinate system used, which would mirror the device map. Anchor
    # row & column direction to the serpentine reconstruction's signs (what the
    # legacy positions_xy grid used) — correlation *sign* is robust even though
    # that lattice's size is not — so the device map keeps its orientation.
    serp_row, serp_col, _, _ = build_scan_grid(frame_x, n_total)
    if _corr(grid_row, rowv) * _corr(serp_row, rowv) < 0:
        grid_row = grid_row.max() - grid_row
    if _corr(grid_col, colv) * _corr(serp_col, colv) < 0:
        grid_col = grid_col.max() - grid_col
    grid_row -= grid_row.min()
    grid_col -= grid_col.min()
    n_rows = int(grid_row.max()) + 1
    n_cols = int(grid_col.max()) + 1
    log(f"  file-per-row grid (commanded-position columns, position-oriented): "
        f"{n_rows} x {n_cols} (col <- {'Y' if row_is_X else 'X'})")
    return grid_row, grid_col, n_rows, n_cols


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


def file_row_layout(frame_map):
    """Frames-per-file from a grid-mapping ``frame_map`` (list of [file_idx, frame_idx]).

    Returns ``(n_files, counts)`` where ``counts[fi]`` is the number of frames in
    file ``fi``. The acquisition writes one HDF5 file per scan row, so this is the
    raw material for :func:`build_grid_from_frame_map`.
    """
    counts: dict = {}
    for fi, _fj in frame_map:
        counts[fi] = counts.get(fi, 0) + 1
    n_files = (max(counts) + 1) if counts else 0
    return n_files, [counts.get(fi, 0) for fi in range(n_files)]


def is_file_per_row(frame_map, min_uniform_frac: float = 0.9):
    """Detect the one-file-per-scan-row layout from ``frame_map``.

    True when ≥ ``min_uniform_frac`` of files share the modal frame count (one
    short/partial row at the end is fine). Returns ``(ok, n_files, mode_count)``.
    A non-uniform result means we cannot trust file index = scan row, and the
    caller should fall back (regular raster / explicit shape) rather than emit a
    ragged grid.
    """
    n_files, counts = file_row_layout(frame_map)
    if n_files < 2:
        return False, n_files, (counts[0] if counts else 0)
    vals, freq = np.unique(np.array(counts), return_counts=True)
    mode_count = int(vals[int(np.argmax(freq))])
    uniform = float(np.mean(np.array(counts) == mode_count))
    return uniform >= min_uniform_frac, n_files, mode_count


def build_grid_from_frame_map(frame_map, serpentine: bool = True,
                              log: Callable[[str], None] = print):
    """Assign (row, col) straight from the one-file-per-row acquisition layout.

    ``row`` = the frame's HDF5 file index (each file is one scan row); ``col`` =
    the frame's position within that file, reversed on alternate rows for the
    serpentine raster. This needs **no position CSV and no TETRAMM/SOCKETSERVER
    stream** — the file/frame structure already encodes the lattice, and on this
    beamline's scans it reconstructs a cleaner grid than the position-trace turn
    counter (exact dimensions, one cell per frame, no re-quantization merges).

    Returns ``(grid_row, grid_col, n_rows, n_cols)``.
    """
    n_files, counts = file_row_layout(frame_map)
    n_total = len(frame_map)
    grid_row = np.zeros(n_total, dtype=int)
    grid_col = np.zeros(n_total, dtype=int)
    # frame_map is in global frame order, grouped by file; walk it tracking the
    # running within-file index so we don't assume a fixed frames-per-file.
    seen: dict = {}
    for gi, (fi, _fj) in enumerate(frame_map):
        j = seen.get(fi, 0)
        seen[fi] = j + 1
        c = (counts[fi] - 1 - j) if (serpentine and fi % 2 == 1) else j
        grid_row[gi] = fi
        grid_col[gi] = c
    n_rows = int(grid_row.max()) + 1
    n_cols = int(grid_col.max()) + 1
    log(f"  file-per-row grid: {n_rows} rows x {n_cols} cols "
        f"({n_total} frames, serpentine={serpentine})")
    return grid_row, grid_col, n_rows, n_cols


def recreate_positions_csv(
    xrd_dir: Union[str, Path],
    output: Union[str, Path],
    scan_number: int = 203,
    step_x: float = 1.0,
    step_y: float = 1.0,
    serpentine: bool = True,
    log: Callable[[str], None] = print,
) -> dict:
    """Reconstruct a per-frame position CSV from the one-file-per-row layout.

    For scans whose SOCKETSERVER-derived ``*_position.csv`` is missing. Assigns
    each frame a ``(row, col)`` from the file/frame structure
    (:func:`build_grid_from_frame_map`) and writes the standard
    ``Trigger,X_Position,Y_Position`` CSV so the rest of the pipeline (and the
    external ptycho preprocessor) has positions instead of zero-padding.

    Orientation matches the real scans: **X = slow per-row axis**, **Y = fast
    within-row sweep**. Steps are nominal µm (the lattice is exact; only the
    absolute scale is synthetic — pass ``step_x``/``step_y`` if you know them).
    The file is tagged with :data:`RECREATED_CSV_MARKER` so loaders can tell it
    apart from a real export. Returns ``{path, n_rows, n_cols, n_total}``.
    """
    xrd_files, frame_map, n_total = load_xrd_metadata(xrd_dir, scan_number)
    ok, n_files, mode_count = is_file_per_row(frame_map)
    if not ok:
        raise ValueError(
            f"Cannot recreate positions: frames-per-file is non-uniform "
            f"({n_files} files, modal {mode_count}/file) — this scan is not a "
            "clean one-file-per-row raster, and TETRAMM/SOCKETSERVER auto-extraction "
            "is not available. Provide a position CSV or pass --shape ROWSxCOLS.")
    grid_row, grid_col, n_rows, n_cols = build_grid_from_frame_map(
        frame_map, serpentine=serpentine, log=log)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    with open(tmp, "w", newline="") as f:
        f.write(RECREATED_CSV_MARKER +
                f" n_rows={n_rows} n_cols={n_cols} step_x={step_x} step_y={step_y}\n")
        w = csv.writer(f)
        w.writerow(["Trigger", "X_Position", "Y_Position"])
        for gi in range(n_total):
            w.writerow([gi + 1, grid_row[gi] * step_x, grid_col[gi] * step_y])
    os.replace(tmp, output)
    log(f"Recreated positions ({n_total} frames, {n_rows}x{n_cols}) -> {output}")
    return {"path": output, "n_rows": n_rows, "n_cols": n_cols, "n_total": n_total}


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


def subbin_keys(bin_key, bin_size):
    """1×1 sub-bin keys covered by a binned ``bin_key`` ('R_C') at ``bin_size``.

    The inverse of :func:`build_bin_mapping`: a binned bin (R, C) aggregates the
    raw grid cells (bin_size·R + dr, bin_size·C + dc) for dr, dc in [0, bin_size),
    and since 1×1 bins are raw grid cells, those cells *are* the 1×1 bin keys.
    Used by the viewer's "View 1×1" mode to map a binned feature's footprint to
    its high-definition (unbinned) bins. ``bin_size == 1`` returns ``[bin_key]``.
    """
    r, c = (int(p) for p in bin_key.split("_"))
    return [f"{bin_size * r + dr}_{bin_size * c + dc}"
            for dr in range(bin_size) for dc in range(bin_size)]


def generate_grid_mapping(
    xrd_dir: Union[str, Path],
    pos_csv: Optional[Union[str, Path]],
    bin_size: int,
    scan_number: int = 203,
    output: Optional[Union[str, Path]] = None,
    n_cols: Optional[int] = None,
    deskew: bool = True,
    deskew_method: str = "commanded",
    log: Callable[[str], None] = print,
) -> dict:
    """Build the grid-mapping dict (and optionally write it to ``output``).

    Coordinate source, in priority order:
      - ``file_per_row`` (default for real positions on a clean one-file-per-row
        raster): rows = file index, columns = within-file rank (commanded
        position), oriented by the real (X, Y). The canonical system.
      - ``positions_xy``: turn-counted position snap — used for *irregular* scans
        (file index ≠ scan row) when a real position CSV is available.
      - ``serpentine``: legacy X-only grid by turn-counting (``--rawgrid`` bypass,
        or the CSV has no ``Y_Position``).
      - ``synthetic``: a regular serpentine raster from ``n_cols`` (last resort).

    ``deskew_method`` selects the column assignment for file-per-row scans:
    ``"commanded"`` (default, align by rank) or ``"perrow_offset"`` (DEPRECATED —
    the per-row encoder offset / "triangle" method, kept only for comparison; see
    :mod:`xrd_app.core.deskew_legacy`).

    A CSV we recreated ourselves (tagged with :data:`RECREATED_CSV_MARKER`) is not
    treated as real positions — it routes to ``file_per_row``. The chosen source is
    recorded in the output JSON as ``coordinate_source``.
    """
    log(f"Loading scan metadata from {xrd_dir} ...")
    xrd_files, frame_map, n_total = load_xrd_metadata(xrd_dir, scan_number)
    log(f"  {n_total} frames across {len(xrd_files)} H5 files")

    have_positions = (pos_csv is not None and Path(pos_csv).exists()
                      and not is_recreated_csv(pos_csv))
    if have_positions and deskew:
        frame_x, frame_y = load_positions_xy(pos_csv, n_total)
        if np.isfinite(frame_y).any():
            fpr = is_file_per_row(frame_map)[0]
            if fpr and deskew_method == "perrow_offset":
                from .deskew_legacy import assign_grid_perrow_offset
                log("Computing scan grid (DEPRECATED perrow_offset method) ...")
                grid_row, grid_col, n_rows, n_cols = assign_grid_perrow_offset(
                    frame_x, frame_y, frame_map, log=log)
                coordinate_source = "perrow_offset_deprecated"
            else:
                log("Computing scan grid from layout + true (X, Y) positions ...")
                grid_row, grid_col, n_rows, n_cols = assign_grid_from_positions(
                    frame_x, frame_y, frame_map=frame_map, log=log)
                # Clean file-per-row raster → commanded-position columns
                # (file_per_row); irregular scans → turn-counted snap (positions_xy).
                coordinate_source = "file_per_row" if fpr else "positions_xy"
        else:
            log("Position CSV has no Y_Position column; falling back to "
                "serpentine (X-only) grid ...")
            grid_row, grid_col, n_rows, n_cols = build_scan_grid(frame_x, n_total)
            coordinate_source = "serpentine"
        log(f"  Scan grid: {n_rows} rows x {n_cols} cols")
    elif have_positions:
        log("Computing serpentine scan grid from X positions (--rawgrid) ...")
        frame_x = load_positions(pos_csv, n_total)
        grid_row, grid_col, n_rows, n_cols = build_scan_grid(frame_x, n_total)
        coordinate_source = "serpentine"
        log(f"  Scan grid: {n_rows} rows x {n_cols} cols")
    else:
        # No real position CSV. Prefer the one-file-per-row layout (cleaner and
        # exact for these scans); fall back to a synthesized raster only if the
        # frame counts aren't uniform enough to trust file index = scan row.
        fpr_ok, n_files, mode_count = is_file_per_row(frame_map)
        if fpr_ok:
            log("No position CSV; reconstructing grid from the one-file-per-row "
                "layout ...")
            grid_row, grid_col, n_rows, n_cols = build_grid_from_frame_map(
                frame_map, log=log)
            coordinate_source = "file_per_row"
        elif n_cols:
            log(f"No position CSV; synthesizing a regular {n_cols}-column serpentine grid ...")
            log("  (assumes a uniform raster; provide positions for irregular scans)")
            grid_row, grid_col, n_rows, n_cols = build_regular_grid(n_total, n_cols)
            coordinate_source = "synthetic"
            log(f"  Scan grid: {n_rows} rows x {n_cols} cols")
        else:
            raise FileNotFoundError(
                f"No position CSV ({pos_csv}) and the scan is not a clean "
                f"one-file-per-row raster ({n_files} files, modal {mode_count} "
                "frames/file). Provide a position CSV or pass --shape ROWSxCOLS."
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
        "coordinate_source": coordinate_source,
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
            has_real_pos = pos and Path(pos).exists() and not is_recreated_csv(pos)
            if has_real_pos:
                # De-skew from true (X, Y) when Y is present (matches what
                # 'xrd-app grid' writes); else fall back to serpentine X-only.
                frame_x, frame_y = load_positions_xy(pos, n_total)
                if np.isfinite(frame_y).any():
                    grid_row, grid_col, n_rows, n_cols2 = assign_grid_from_positions(
                        frame_x, frame_y, frame_map=self._frame_map)
                else:
                    grid_row, grid_col, n_rows, n_cols2 = build_scan_grid(frame_x, n_total)
            elif is_file_per_row(self._frame_map)[0]:
                # No real CSV: reconstruct from the one-file-per-row layout (matches
                # 'xrd-app grid' with no positions).
                grid_row, grid_col, n_rows, n_cols2 = build_grid_from_frame_map(
                    self._frame_map, log=lambda *a: None)
            elif n_cols:
                grid_row, grid_col, n_rows, n_cols2 = build_regular_grid(n_total, n_cols)
            else:
                raise FileNotFoundError(
                    "No grid mapping, no usable position CSV, and the scan is not a "
                    "clean one-file-per-row raster — cannot assign raw frames to bins. "
                    "Run 'xrd-app grid' or 'xrd-app recreate-positions'.")
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
