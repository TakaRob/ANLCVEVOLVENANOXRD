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
import hashlib
import importlib.util
import json
import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Optional, Union

import h5py
import numpy as np

H5_DATASET = "entry/data/data"
ARCHIVE_FRAMES = "frames"
ARCHIVE_METADATA = "metadata"
ARCHIVE_FORMAT = "xrd-app-unbinned-archive"
ARCHIVE_VERSION = 1
# Lozano-style per-frame stage positions: a small HDF5 (``Scan_NNNN.h5``) with an
# ``entry/data/Position`` group holding already-reduced ``X_Position``/``Y_Position``
# (µm) arrays — one value per frame. Distinct from the raw SOCKETSERVER stream
# (``H5_DATASET``, 24 encoder cols), which ``core.positions`` reduces separately.
H5_POSITION_GROUP = "entry/data/Position"
_H5_SUFFIXES = (".h5", ".hdf5")


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
    """Dynamically import a detector or algorithm Python module.

    The file's own directory and packaged algorithm libraries are placed on
    ``sys.path`` during import so detectors can import sibling/base modules.
    """
    import sys
    path = Path(path).resolve()
    # Use path and current file state for identity. This prevents equal stems in
    # different libraries, or a rewritten candidate in a long-running process,
    # from sharing import/pickle identity.
    source = path.read_bytes()
    identity = str(path).encode() + b"\0" + source
    module_name = f"_xrd_dynamic_{path.stem}_{hashlib.sha256(identity).hexdigest()[:16]}"
    # Shared library dirs let detectors import sibling/library modules no matter
    # where they live, including a base detector from an automated sub-folder.
    pkg = Path(__file__).resolve().parent.parent
    dirs = [str(path.parent)]
    if path.name == "detector.py":
        dirs.append(str(path.parent.parent))
    for extra in (pkg / "PeakAlgorithms", pkg / "CombinedAlgorithms"):
        if extra.is_dir():
            dirs.append(str(extra))
    added = [d for d in dirs if d not in sys.path]
    for d in added:
        sys.path.insert(0, d)
    try:
        spec = importlib.util.spec_from_file_location(module_name, str(path))
        mod = importlib.util.module_from_spec(spec)
        # Execute the bytes used for identity directly. SourceFileLoader's pyc
        # cache is timestamp/size based and can return stale code after a rapid
        # same-size rewrite (common during candidate evolution).
        exec(compile(source, str(path), "exec"), mod.__dict__)
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
    """Load canonical reflection JSON; return angles and labels."""
    from .reflections import read_json

    path = Path(path)
    if path.suffix.lower() != ".json":
        raise ValueError(f"Reflection source must be JSON: {path}")
    reflections = read_json(path)
    return ([float(r["two_theta"]) for r in reflections],
            [str(r["name"]) for r in reflections])


def save_grid_mapping(path, mapping: dict) -> Path:
    """Atomically persist a grid mapping as compact HDF5 arrays."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    metadata = {key: value for key, value in mapping.items()
                if key not in ("bins", "frame_map", "xrd_files", "territories")}
    bins = mapping.get("bins") or {}
    keys = list(bins)
    offsets = [0]
    members = []
    for key in keys:
        members.extend(int(value) for value in bins[key])
        offsets.append(len(members))
    strings = h5py.string_dtype("utf-8")
    kwargs = {"compression": "gzip", "compression_opts": 1, "shuffle": True}
    try:
        with h5py.File(tmp, "w") as handle:
            handle.attrs["format"] = "xrd-app-grid"
            handle.attrs["schema_version"] = 1
            handle.attrs["metadata_json"] = json.dumps(metadata, separators=(",", ":"))
            handle.create_dataset("xrd_files", data=np.asarray(
                [str(value) for value in mapping.get("xrd_files", [])], dtype=object),
                dtype=strings, compression="gzip", compression_opts=1)
            frame_map = np.asarray(mapping.get("frame_map", []), dtype=np.int64).reshape(-1, 2)
            handle.create_dataset("frame_map", data=frame_map,
                                  **(kwargs if frame_map.size else {}))
            handle.create_dataset("bin_keys", data=np.asarray(keys, dtype=object),
                                  dtype=strings, compression="gzip", compression_opts=1)
            handle.create_dataset("bin_offsets", data=np.asarray(offsets, dtype=np.int64),
                                  **kwargs)
            member_array = np.asarray(members, dtype=np.int64)
            handle.create_dataset("bin_frames", data=member_array,
                                  **(kwargs if member_array.size else {}))
            territories = mapping.get("territories") or {}
            if territories:
                handle.create_dataset("territory_keys", data=np.asarray(
                    list(territories), dtype=object), dtype=strings,
                    compression="gzip", compression_opts=1)
                handle.create_dataset("territory_json", data=np.asarray([
                    json.dumps(territories[key], separators=(",", ":"))
                    for key in territories
                ], dtype=object), dtype=strings, compression="gzip", compression_opts=1)
        os.replace(tmp, path)
    except Exception:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    return path


def load_grid_mapping(grid_mapping: Union[str, Path, dict]) -> dict:
    """Accept an HDF5 path or an already-loaded grid mapping."""
    if isinstance(grid_mapping, dict):
        return grid_mapping
    path = Path(grid_mapping)
    with h5py.File(path, "r") as handle:
        if str(handle.attrs.get("format", "")) != "xrd-app-grid":
            raise ValueError(f"Unsupported grid mapping HDF5: {path}")
        result = json.loads(str(handle.attrs.get("metadata_json", "{}")))
        result["xrd_files"] = [value.decode() if isinstance(value, bytes) else str(value)
                               for value in handle["xrd_files"][:]]
        result["frame_map"] = handle["frame_map"][:].astype(int).tolist()
        keys = [value.decode() if isinstance(value, bytes) else str(value)
                for value in handle["bin_keys"][:]]
        offsets = handle["bin_offsets"][:]
        frames = handle["bin_frames"][:]
        result["bins"] = {
            key: frames[int(offsets[index]):int(offsets[index + 1])].astype(int).tolist()
            for index, key in enumerate(keys)
        }
        if "territory_keys" in handle:
            territory_keys = [value.decode() if isinstance(value, bytes) else str(value)
                              for value in handle["territory_keys"][:]]
            result["territories"] = {
                key: json.loads(value.decode() if isinstance(value, bytes) else str(value))
                for key, value in zip(territory_keys, handle["territory_json"][:])
            }
        return result


def validate_grid_mapping_bin_size(grid_mapping, expected_bin_size) -> dict:
    """Load a grid mapping and reject a known bin-size mismatch."""
    gm = load_grid_mapping(grid_mapping)
    actual = gm.get("bin_size")
    if actual is None and not isinstance(grid_mapping, dict):
        match = _re.search(r"grid_mapping_(\d+)x(\d+)", Path(grid_mapping).stem)
        actual = int(match.group(1)) if match and match.group(1) == match.group(2) else None
    if actual is not None and int(actual) != int(expected_bin_size):
        raise ValueError(
            f"Grid mapping bin size is {actual}x{actual}, not requested "
            f"{expected_bin_size}x{expected_bin_size}: {grid_mapping}")
    return gm


def _grid_variant(grid_mapping) -> Optional[str]:
    if isinstance(grid_mapping, dict):
        value = grid_mapping.get("variant")
        return str(value) if value else None
    try:
        value = load_grid_mapping(grid_mapping).get("variant")
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        value = None
    if value:
        return str(value)
    match = _re.match(r"grid_mapping_\d+x\d+_(.+)$", Path(grid_mapping).stem)
    return match.group(1) if match else None


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


def archive_positions(archive: Union[str, Path]):
    """Return embedded per-frame ``(x, y)`` arrays."""
    with h5py.File(archive, "r") as f:
        meta = f[ARCHIVE_METADATA]
        return (np.asarray(meta["x"][()], dtype=np.float64),
                np.asarray(meta["y"][()], dtype=np.float64))


def archive_has_real_positions(archive: Union[str, Path]) -> bool:
    try:
        with h5py.File(archive, "r") as f:
            return bool(f.attrs.get("positions_real", False))
    except OSError:
        return False


def archive_metadata(archive: Union[str, Path]):
    """Return ``(source_files, frame_map, n_frames)`` without reading pixels."""
    with h5py.File(archive, "r") as f:
        if f.attrs.get("format") != ARCHIVE_FORMAT or ARCHIVE_FRAMES not in f:
            raise ValueError(f"Not an {ARCHIVE_FORMAT}: {archive}")
        meta = f[ARCHIVE_METADATA]
        source_files = [
            s.decode("utf-8") if isinstance(s, bytes) else str(s)
            for s in meta["source_files"][()]
        ]
        file_index = meta["source_file_index"][()]
        local_index = meta["source_frame_index"][()]
        frame_map = np.column_stack([file_index, local_index]).astype(int).tolist()
        return source_files, frame_map, int(f[ARCHIVE_FRAMES].shape[0])


def load_xrd_metadata(xrd_dir: Union[str, Path], scan_number: int = 203,
                      archive: Optional[Union[str, Path]] = None):
    """Build the acquisition frame index from an archive or loose raw files."""
    if archive is not None and Path(archive).is_file():
        return archive_metadata(archive)
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
    """True if ``csv_path`` is a synthetic file-per-row CSV we wrote (has the marker).

    An HDF5 position file is never a recreated CSV (and reading it as text would
    raise), so short-circuit to ``False`` for ``.h5``/``.hdf5``.
    """
    if Path(csv_path).suffix.lower() in _H5_SUFFIXES:
        return False
    try:
        with open(csv_path) as f:
            return f.readline().lstrip().startswith(RECREATED_CSV_MARKER)
    except OSError:
        return False


def _read_positions_h5(h5_path: Union[str, Path]):
    """Read ``(x, y)`` µm arrays from a Lozano-style position HDF5.

    Layout: ``entry/data/Position/{X_Position,Y_Position}`` — one already-reduced
    value per frame (contrast the raw SOCKETSERVER stream, which
    :mod:`core.positions` reduces). Returns two 1-D float arrays.
    """
    with h5py.File(h5_path, "r") as f:
        grp = f[H5_POSITION_GROUP]
        x = np.asarray(grp["X_Position"][()], dtype=float).ravel()
        y = np.asarray(grp["Y_Position"][()], dtype=float).ravel()
    return x, y


def load_positions(path: Union[str, Path], n_total: int) -> np.ndarray:
    """Read X positions from the scan position file, padded/truncated to n_total.

    Accepts either the ``Trigger,X_Position,Y_Position`` CSV or a Lozano-style
    position HDF5 (``.h5``/``.hdf5``, see :func:`_read_positions_h5`).
    """
    if Path(path).suffix.lower() in _H5_SUFFIXES:
        x, _ = _read_positions_h5(path)
    else:
        x = []
        with open(path) as f:
            reader = csv.DictReader(_uncommented(f))
            for row in reader:
                x.append(float(row["X_Position"]))
    frame_x = np.full(n_total, np.nan)
    n = min(len(x), n_total)
    frame_x[:n] = x[:n]
    return frame_x


def load_positions_xy(path: Union[str, Path], n_total: int):
    """Read X *and* Y positions from the scan position file → (frame_x, frame_y).

    Accepts either the ``Trigger,X_Position,Y_Position`` CSV or a Lozano-style
    position HDF5 (``.h5``/``.hdf5``, ``entry/data/Position`` group). Companion to
    :func:`load_positions` (X-only, kept for back-compat). Both arrays are
    padded/truncated to ``n_total``. A ``frame_y`` that is all-NaN means the
    source has no Y column — callers should then fall back to the serpentine
    (X-only) grid.
    """
    if Path(path).suffix.lower() in _H5_SUFFIXES:
        x, y = _read_positions_h5(path)
        has_y = True
    else:
        x, y = [], []
        with open(path) as f:
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
                               force_xy: bool = False,
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

    ``force_xy=True`` takes that same both-axes turn-counted path even for a clean
    one-file-per-row scan — used by the ``positions_xy`` grid mode, where file
    index is *not* a trustworthy spatial row (e.g. sweeps that pinch to a point
    mid-scan) so rows must come from true X, not the file layout.

    Returns ``(grid_row, grid_col, n_rows, n_cols)``.
    """
    n_total = len(frame_x)
    x = _interp_nan(frame_x)
    y = _interp_nan(frame_y)

    file_per_row = (not force_xy) and frame_map is not None and is_file_per_row(frame_map)[0]
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
    # positions_xy grid used) — correlation *sign* is robust even though
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


def assign_grid_coordinate_faithful(frame_x, frame_y, frame_map,
                                    column_mode: str = "square",
                                    log: Callable[[str], None] = print):
    """Per-frame ``(row, col)`` on a de-skewed lattice from true (X, Y).

    The de-skewed coordinate system for clean one-file-per-row rasters. Rows stay
    the exact acquisition layout (file index — the slow axis is already faithful:
    ``corr(row, X) ≈ 1``), but **columns are snapped to the true fast-axis stage
    position** so serpentine stage backlash no longer puts the same physical
    position at a different column on alternate rows (the ``file_per_row`` comb / V).

    ``column_mode`` sizes the column count:

    - ``"square"`` (default): ``n_cols`` from the physical X/Y span so device-map
      pixels are ~square and the image is to-scale. Best for *viewing*.
    - ``"native"``: ``n_cols`` = the native frames-per-row count, so each true-Y
      sample keeps ~one cell (minimal collisions / empty cells). Best for
      *detection/recall* — it de-skews without losing the per-frame sampling that
      the under-sampled square lattice merges away.

    Returns ``(grid_row, grid_col, n_rows, n_cols)``.
    """
    n_total = len(frame_x)
    x = _interp_nan(frame_x)
    y = _interp_nan(frame_y)

    # Rows straight from the one-file-per-row layout (exact, no re-snap).
    ref_row, _ref_col, n_rows, _ = build_grid_from_frame_map(
        frame_map, log=lambda *a: None)
    grid_row = ref_row.astype(int)

    # Which physical axis sweeps within a row (the fast / column axis)?
    row_is_X = abs(_corr(ref_row, x)) >= abs(_corr(ref_row, y))
    rowv, colv = (x, y) if row_is_X else (y, x)

    def _span(v):
        lo, hi = np.percentile(v, 0.2), np.percentile(v, 99.8)
        return max(hi - lo, 1e-9)
    row_span, col_span = _span(rowv), _span(colv)
    if column_mode == "native":
        # Native fast-axis density: one column per acquired frame-per-row, so the
        # true-Y snap keeps ~one frame per cell (no detection loss from merges).
        _n_files, counts = file_row_layout(frame_map)
        n_cols = int(np.median([c for c in counts if c > 0])) if counts else n_rows
    else:
        # Square pixels: column step ≈ row step in physical units (to-scale view).
        n_cols = max(int(round(1 + (n_rows - 1) * col_span / row_span)), 1)
    n_cols = max(n_cols, 1)

    # Orient to the historical device-map convention via the serpentine
    # reconstruction's correlation *sign* (its lattice size is irrelevant here),
    # so the faithful map keeps the same orientation as the file-per-row default.
    serp_row, serp_col, _, _ = build_scan_grid(frame_x, n_total)
    sc = np.sign(_corr(serp_col, colv)) or 1.0
    grid_col = _scale_to_index(colv, sc, n_cols)
    if _corr(grid_row, rowv) * _corr(serp_row, rowv) < 0:
        grid_row = grid_row.max() - grid_row
    grid_row = grid_row - grid_row.min()
    n_rows = int(grid_row.max()) + 1
    log(f"  coordinate-faithful grid ({column_mode}, true-position columns): "
        f"{n_rows} x {n_cols} (col <- {'Y' if row_is_X else 'X'}, "
        f"aspect {col_span / row_span:.3f})")
    return grid_row, grid_col, n_rows, n_cols


def build_scan_grid(frame_x, n_total, kernel=20, order=50):
    """Infer (row, col) for each frame from the serpentine position trace."""
    from scipy.signal import argrelextrema
    if n_total == 0 or len(frame_x) == 0:
        raise ValueError(
            "No scan frames to build a grid from (0 frames). The raw XRD "
            "frames could not be found — check that the raw data source is "
            "mounted/reachable and the scan number is correct.")
    valid = ~np.isnan(frame_x)
    x = frame_x.copy()
    if not np.any(valid):
        x[:] = np.arange(n_total, dtype=float)
    elif np.any(~valid):
        x[~valid] = np.interp(np.where(~valid)[0], np.where(valid)[0], frame_x[valid])
    kernel = max(1, min(kernel, len(x)))
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


# coordinate_source (as recorded in a grid mapping) → the deskew_method that
# reproduces it. Used to build an aligned finer/coarser grid from the *same*
# per-frame lattice: because build_bin_mapping coarsens by floor-division, a 1×1
# grid is a clean N× refinement of an N×N grid only when both use the same
# deskew. Mirror the source catalog's coordinate_source rather than the "auto"
# default (which picks positions_xy at 1×1 — a different lattice than a
# positions_faithful N×N — and would misalign the sub-bin keys).
_SOURCE_TO_DESKEW = {
    "positions_faithful": "faithful",
    "positions_faithful_native": "faithful_native",
    "positions_xy": "positions_xy",
    "file_per_row": "commanded",
}


def deskew_method_for_source(coordinate_source: Optional[str]) -> str:
    """deskew_method that rebuilds a grid matching ``coordinate_source``.

    Defaults to ``"faithful"`` (the dominant real-position case) for unknown or
    non-real sources (e.g. ``serpentine``/``synthetic`` — those carry no real
    positions anyway, so the caller's real-position step is a no-op regardless).
    """
    return _SOURCE_TO_DESKEW.get(coordinate_source or "", "faithful")


def generate_grid_mapping(
    xrd_dir: Union[str, Path],
    pos_csv: Optional[Union[str, Path]],
    bin_size: int,
    scan_number: int = 203,
    output: Optional[Union[str, Path]] = None,
    n_cols: Optional[int] = None,
    deskew: bool = True,
    deskew_method: str = "auto",
    log: Callable[[str], None] = print,
    archive: Optional[Union[str, Path]] = None,
) -> dict:
    """Build the grid-mapping dict (and optionally write it to ``output``).

    Coordinate source, in priority order:
      - ``file_per_row`` (default for real positions on a clean one-file-per-row
        raster): rows = file index, columns = within-file rank (commanded
        position), oriented by the real (X, Y). The canonical system.
      - ``positions_xy``: turn-counted position snap — used for *irregular* scans
        (file index ≠ scan row) when a real position CSV is available.
      - ``serpentine``: X-only grid by turn-counting when the position source has
        no ``Y_Position``.
      - ``synthetic``: a regular serpentine raster from ``n_cols`` (last resort).

    ``deskew_method`` selects how frames map to the lattice. ``"auto"`` (default)
    picks ``"positions_xy"`` at 1×1 (both axes snapped to true (X, Y) — skew-free,
    avoids the file-index-row bowtie) and ``"faithful"`` at coarser bins. Explicit
    options: ``"positions_xy"``, ``"faithful"``/``"faithful_native"`` (file-index
    rows, true-Y columns), or ``"commanded"`` (align columns by rank).

    A CSV we recreated ourselves (tagged with :data:`RECREATED_CSV_MARKER`) is not
    treated as real positions — it routes to ``file_per_row``. The chosen source is
    recorded in the output JSON as ``coordinate_source``.
    """
    log(f"Loading scan metadata from {xrd_dir} ...")
    xrd_files, frame_map, n_total = load_xrd_metadata(
        xrd_dir, scan_number, archive=archive)
    log(f"  {n_total} frames across {len(xrd_files)} H5 files")

    # "auto" (the default): true-(X, Y) both-axes lattice at 1×1 — where the
    # file-index-row `faithful` grid produces the skew "X"/bowtie — and the
    # feature-preserving `faithful` grid at coarser bins (where it's already
    # clean). Pass an explicit method to override.
    if deskew_method in (None, "auto"):
        deskew_method = "positions_xy" if bin_size == 1 else "faithful"

    have_position_file = (pos_csv is not None and Path(pos_csv).exists()
                          and not is_recreated_csv(pos_csv))
    have_archive_positions = bool(
        archive and archive_has_real_positions(archive))
    have_positions = have_position_file or have_archive_positions
    if have_positions and deskew:
        if have_position_file:
            frame_x, frame_y = load_positions_xy(pos_csv, n_total)
        else:
            frame_x, frame_y = archive_positions(archive)
        if np.isfinite(frame_y).any():
            fpr = is_file_per_row(frame_map)[0]
            if deskew_method == "positions_xy":
                log("Computing scan grid from true (X, Y) positions "
                    "(positions_xy, both axes) ...")
                grid_row, grid_col, n_rows, n_cols = assign_grid_from_positions(
                    frame_x, frame_y, frame_map=frame_map, force_xy=True, log=log)
                coordinate_source = "positions_xy"
            elif fpr and deskew_method in ("faithful", "faithful_native"):
                col_mode = "native" if deskew_method == "faithful_native" else "square"
                log(f"Computing coordinate-faithful scan grid ({col_mode}, "
                    "true-position columns) ...")
                grid_row, grid_col, n_rows, n_cols = assign_grid_coordinate_faithful(
                    frame_x, frame_y, frame_map, column_mode=col_mode, log=log)
                coordinate_source = ("positions_faithful_native"
                                     if col_mode == "native" else "positions_faithful")
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
        if have_position_file:
            frame_x = load_positions(pos_csv, n_total)
        else:
            frame_x, _ = archive_positions(archive)
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
        # Positions provenance — whether this grid was built from a REAL (X, Y)
        # coordinate CSV (vs the file-per-row layout / a synthetic raster). `bin`
        # requires positions_real to be true, so a layout-reconstructed or
        # synthesized grid can never be silently binned.
        "positions_csv": str(pos_csv) if pos_csv is not None else None,
        "positions_real": bool(have_positions),
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
        save_grid_mapping(output, result)
        size_kb = Path(output).stat().st_size / 1024
        log(f"Wrote {output} ({size_kb:.0f} KB)")

    return result


# ─────────────────────────────────────────────────────────────────────
# Binning (port of prebuild_bins.py)
# ─────────────────────────────────────────────────────────────────────
def get_compression_kwargs(compression: str, bitshuffle: bool = False):
    if compression == "zstd":
        try:
            import hdf5plugin
            if bitshuffle:
                return dict(hdf5plugin.Bitshuffle(cname="zstd", clevel=3)), "bitshuffle+zstd3"
            return {**hdf5plugin.Zstd(clevel=3), "shuffle": True}, "zstd3+shuffle"
        except (ImportError, AttributeError):
            return {"compression": "gzip", "compression_opts": 4, "shuffle": True}, "gzip4+shuffle fallback"
    elif compression == "gzip":
        return {"compression": "gzip", "compression_opts": 4, "shuffle": True}, "gzip4+shuffle"
    elif compression == "lz4":
        try:
            import hdf5plugin
            return hdf5plugin.LZ4(), "lz4"
        except (ImportError, AttributeError):
            return {"compression": "gzip", "compression_opts": 4, "shuffle": True}, "gzip4+shuffle fallback"
    elif compression == "none":
        return {}, "none"
    else:
        raise ValueError(f"Unknown compression: {compression}")


def build_unbinned_archive(
    xrd_dir: Union[str, Path],
    output: Union[str, Path],
    scan_number: int,
    positions: Optional[Union[str, Path]] = None,
    compression: str = "zstd",
    log: Callable[[str], None] = print,
) -> Path:
    """Create a lossless, grid-neutral archive with one detector frame per chunk."""
    xrd_files = scan_h5_files(xrd_dir, scan_number)
    if not xrd_files:
        raise FileNotFoundError(
            f"No raw XRD files for scan {scan_number} in {xrd_dir}")

    counts, shape, dtype = [], None, None
    for fp in xrd_files:
        with h5py.File(fp, "r") as f:
            if H5_DATASET not in f:
                raise KeyError(f"{fp}: no {H5_DATASET}")
            ds = f[H5_DATASET]
            if ds.ndim != 3:
                raise ValueError(f"{fp}: expected 3-D detector data, got {ds.shape}")
            current_shape = tuple(int(v) for v in ds.shape[1:])
            if shape is None:
                shape, dtype = current_shape, ds.dtype
            elif current_shape != shape or ds.dtype != dtype:
                raise ValueError(
                    f"Inconsistent raw detector data in {fp}: "
                    f"shape={current_shape}, dtype={ds.dtype}; expected {shape}, {dtype}")
            counts.append(int(ds.shape[0]))

    n_frames = sum(counts)
    frame_x = frame_y = None
    positions_real = False
    if positions and Path(positions).exists():
        frame_x, frame_y = load_positions_xy(positions, n_frames)
        positions_real = not is_recreated_csv(positions)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    comp_kwargs, compression_label = get_compression_kwargs(
        compression, bitshuffle=True)
    log(f"Archiving {n_frames} unbinned frames -> {output}")
    log(f"  Chunks: 1 x {shape[0]} x {shape[1]}; compression: {compression_label}")

    out = None
    try:
        out = h5py.File(tmp, "w")
        out.attrs["format"] = ARCHIVE_FORMAT
        out.attrs["format_version"] = ARCHIVE_VERSION
        out.attrs["scan_number"] = int(scan_number)
        out.attrs["n_frames"] = n_frames
        out.attrs["detector_shape"] = shape
        out.attrs["source_dtype"] = np.dtype(dtype).str
        out.attrs["source_dataset"] = H5_DATASET
        out.attrs["compression"] = compression_label
        out.attrs["positions_real"] = positions_real
        if positions:
            out.attrs["positions_source"] = str(positions)
        frames = out.create_dataset(
            ARCHIVE_FRAMES, shape=(n_frames, *shape), dtype=dtype,
            chunks=(1, *shape), **comp_kwargs)
        meta = out.create_group(ARCHIVE_METADATA)
        string_dtype = h5py.string_dtype(encoding="utf-8")
        meta.create_dataset("source_files", data=[str(p) for p in xrd_files],
                            dtype=string_dtype)
        file_idx = np.repeat(np.arange(len(counts), dtype=np.int32), counts)
        local_idx = np.concatenate(
            [np.arange(n, dtype=np.int32) for n in counts])
        meta.create_dataset("source_file_index", data=file_idx)
        meta.create_dataset("source_frame_index", data=local_idx)
        meta.create_dataset(
            "x", data=(frame_x if frame_x is not None else
                       np.full(n_frames, np.nan, dtype=np.float64)))
        meta.create_dataset(
            "y", data=(frame_y if frame_y is not None else
                       np.full(n_frames, np.nan, dtype=np.float64)))

        offset = 0
        t0 = time.time()
        for fi, (fp, count) in enumerate(zip(xrd_files, counts)):
            with h5py.File(fp, "r") as src:
                ds = src[H5_DATASET]
                for j in range(count):
                    frames[offset + j] = ds[j]
            offset += count
            elapsed = time.time() - t0
            rate = offset / elapsed if elapsed else 0.0
            eta = (n_frames - offset) / rate if rate else 0.0
            log(f"  [{fi + 1}/{len(xrd_files)} files, {offset}/{n_frames} frames] "
                f"{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining")
        out.flush()
        out.close()
        out = None
        os.replace(tmp, output)
    finally:
        if out is not None:
            out.close()
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass

    size_mb = output.stat().st_size / 1024 / 1024
    log(f"Done! {output}: {size_mb:.0f} MB ({size_mb / 1024:.1f} GB)")
    return output


class FrameStore:
    """Acquisition-indexed detector frames from an archive or loose raw files."""

    def __init__(self, archive=None, xrd_files=None, frame_map=None):
        self._archive = None
        self._handles = OrderedDict()
        self._xrd_files = list(xrd_files or [])
        self._frame_map = list(frame_map or [])
        if archive is not None and Path(archive).is_file():
            try:
                import hdf5plugin  # noqa: F401
            except ImportError:
                pass
            self._archive = h5py.File(archive, "r")
            if self._archive.attrs.get("format") != ARCHIVE_FORMAT:
                self._archive.close()
                self._archive = None
                raise ValueError(f"Not an {ARCHIVE_FORMAT}: {archive}")
            self._frames = self._archive[ARCHIVE_FRAMES]

    @property
    def is_archive(self):
        return self._archive is not None

    def _dataset(self, global_index):
        if self._archive is not None:
            return self._frames, int(global_index)
        fi, local = self._frame_map[int(global_index)]
        if fi not in self._handles:
            if len(self._handles) >= 32:
                _, old = self._handles.popitem(last=False)
                old.close()
            self._handles[fi] = h5py.File(self._xrd_files[fi], "r")
        else:
            self._handles.move_to_end(fi)
        return self._handles[fi][H5_DATASET], int(local)

    def frame(self, global_index):
        ds, index = self._dataset(global_index)
        return ds[index]

    def region(self, global_index, y0, y1, x0, x1):
        ds, index = self._dataset(global_index)
        return ds[index, max(0, y0):y1, max(0, x0):x1]

    def close(self):
        if self._archive is not None:
            self._archive.close()
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()


def build_bins(
    grid_mapping: Union[str, Path, dict],
    output: Union[str, Path],
    bin_size: Optional[int] = None,
    compression: str = "zstd",
    log: Callable[[str], None] = print,
    archive: Optional[Union[str, Path]] = None,
    normalize_frames: bool = False,
) -> Path:
    """Aggregate each bin's raw frames into a single binned HDF5 file.

    Frames are summed by default. With ``normalize_frames=True``, each sum is
    divided by its contributing frame count to produce a per-frame mean, which
    removes intensity artifacts from unequal true-position cell occupancy.
    Output structure: one float32 dataset per bin keyed ``"row_col"``, with
    aggregation provenance and grid dimensions stored as file attributes.
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

    comp_kwargs, compression_label = get_compression_kwargs(compression)
    aggregation = "mean_per_frame" if normalize_frames else "sum"
    log(f"Building {n_bins} bin images ({bin_size}x{bin_size}) -> {output}")
    log(f"  Aggregation: {aggregation}")
    log(f"  Compression: {compression_label}")

    # Write to a temporary file and atomically rename it onto `output` only
    # after a fully successful build. A ctrl+C / crash mid-build then leaves the
    # (discarded) .tmp file corrupt instead of the real output, so any previous
    # good bins file survives and the GUI never reads a half-written file.
    tmp = output.with_name(output.name + ".tmp")
    frame_store = FrameStore(archive=archive, xrd_files=xrd_files,
                             frame_map=frame_map)
    out = h5py.File(str(tmp), "w")
    out.attrs["bin_size"] = bin_size
    out.attrs["n_bin_rows"] = gm["n_bin_rows"]
    out.attrs["n_bin_cols"] = gm["n_bin_cols"]
    out.attrs["n_bins"] = n_bins
    out.attrs["aggregation"] = aggregation
    if normalize_frames:
        out.attrs["normalized_by"] = "contributing_frame_count"
    if frame_store.is_archive:
        out.attrs["pixel_source"] = "xrd_unbinned_archive.h5"
    first_frame = next((gi for indices in bins.values() for gi in indices), None)
    if first_frame is not None:
        out.attrs["detector_shape"] = list(frame_store.frame(first_frame).shape)

    t0 = time.time()
    try:
        for i, (bin_key, frame_indices) in enumerate(sorted(bins.items())):
            summed = None
            for gi in frame_indices:
                frame = frame_store.frame(gi).astype(np.float64)
                summed = frame if summed is None else summed + frame

            if summed is not None:
                if normalize_frames:
                    summed /= len(frame_indices)
                np.clip(summed, 0, 1e9, out=summed)
                dataset = out.create_dataset(
                    bin_key, data=summed.astype(np.float32), **comp_kwargs)
                dataset.attrs["n_frames"] = len(frame_indices)

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
        frame_store.close()
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


_BIN_KEY_RE = _re.compile(r"^(\d+)_(\d+)$")


def _int_metadata(value, label, source):
    """Return an integer metadata value with a source-specific error."""
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"Invalid {label} metadata {value!r} in {source}") from None


def _validate_binned_h5(h5, h5_path, bin_size, grid_mapping, variant=None):
    """Validate a prebuilt bins file against the grid that gives keys meaning."""
    gm = validate_grid_mapping_bin_size(grid_mapping, bin_size)
    bins = gm.get("bins")
    if not isinstance(bins, dict):
        raise ValueError(f"Grid mapping has no valid 'bins' mapping: {grid_mapping}")

    territory = variant == "territory" or gm.get("coordinate_source") == "territory_xy"
    dimensions = {}
    for name in ("n_bin_rows", "n_bin_cols"):
        if name in gm:
            dimensions[name] = _int_metadata(gm[name], name, grid_mapping)
            if dimensions[name] < 0:
                raise ValueError(f"Grid mapping {name} must be non-negative: {grid_mapping}")

    mapping_keys = set()
    required_keys = set()
    for key, frame_indices in bins.items():
        match = _BIN_KEY_RE.fullmatch(str(key))
        if match is None:
            raise ValueError(f"Invalid bin key {key!r} in grid mapping {grid_mapping}")
        row, col = map(int, match.groups())
        if not territory:
            if "n_bin_rows" in dimensions and row >= dimensions["n_bin_rows"]:
                raise ValueError(
                    f"Grid mapping bin key {key!r} is outside n_bin_rows="
                    f"{dimensions['n_bin_rows']}: {grid_mapping}")
            if "n_bin_cols" in dimensions and col >= dimensions["n_bin_cols"]:
                raise ValueError(
                    f"Grid mapping bin key {key!r} is outside n_bin_cols="
                    f"{dimensions['n_bin_cols']}: {grid_mapping}")
        mapping_keys.add(str(key))
        if frame_indices:
            required_keys.add(str(key))

    if "bin_size" in h5.attrs:
        actual = _int_metadata(h5.attrs["bin_size"], "bin_size", h5_path)
        if actual != int(bin_size):
            raise ValueError(
                f"Binned HDF5 bin_size is {actual}x{actual}, not requested "
                f"{bin_size}x{bin_size}; file is stale or mismatched: {h5_path}")
    for name, expected in dimensions.items():
        if name in h5.attrs:
            actual = _int_metadata(h5.attrs[name], name, h5_path)
            if actual != expected:
                raise ValueError(
                    f"Binned HDF5 {name}={actual} does not match grid mapping "
                    f"{name}={expected}; file is stale or mismatched: {h5_path}")

    detector_shape = None
    if "detector_shape" in h5.attrs:
        try:
            detector_shape = tuple(int(v) for v in h5.attrs["detector_shape"])
        except (TypeError, ValueError, OverflowError):
            raise ValueError(
                f"Invalid detector_shape metadata in binned HDF5: {h5_path}") from None
        if len(detector_shape) != 2 or any(v <= 0 for v in detector_shape):
            raise ValueError(
                f"Invalid detector_shape metadata {detector_shape!r} in binned HDF5: "
                f"{h5_path}")

    dataset_keys = set()
    observed_shape = detector_shape
    for key, obj in h5.items():
        if _BIN_KEY_RE.fullmatch(key) is None:
            if isinstance(obj, h5py.Dataset):
                raise ValueError(
                    f"Binned HDF5 dataset key {key!r} is not a row_col bin key: "
                    f"{h5_path}")
            continue  # Groups may hold unrelated metadata.
        if not isinstance(obj, h5py.Dataset):
            raise ValueError(f"Binned HDF5 bin key {key!r} is not a dataset: {h5_path}")
        if key not in mapping_keys:
            raise ValueError(
                f"Binned HDF5 bin key {key!r} is not present in grid mapping; "
                f"file is stale or mismatched: {h5_path}")
        if obj.ndim != 2:
            raise ValueError(
                f"Binned HDF5 dataset {key!r} must be 2-D, got {obj.ndim}-D: {h5_path}")
        if observed_shape is None:
            observed_shape = obj.shape
        elif tuple(obj.shape) != tuple(observed_shape):
            raise ValueError(
                f"Binned HDF5 dataset {key!r} has detector shape {obj.shape}, "
                f"expected {tuple(observed_shape)}: {h5_path}")
        dataset_keys.add(key)

    missing = required_keys - dataset_keys
    if missing:
        sample = ", ".join(sorted(missing, key=_bin_sort_key)[:5])
        raise ValueError(
            f"Binned HDF5 is missing {len(missing)} populated grid-mapping bin(s) "
            f"({sample}); file is stale or incomplete: {h5_path}")
    return sorted(dataset_keys, key=_bin_sort_key)


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
        np.clip(summed, 0, 1e9, out=summed)
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

    def region(self, key: str, y0: int, y1: int, x0: int,
               x1: int) -> Optional[np.ndarray]:
        """A detector-space window ``[y0:y1, x0:x1]`` of a bin's image.

        Default: read the full image and crop. Backends that can read a slice
        directly (e.g. an HDF5 dataset) override this to avoid materializing the
        whole frame — used by the HD-map sampler, which only needs a small window
        around each feature's detector peak. Bounds are clamped to ``>= 0``;
        upper bounds past the frame edge are fine (slicing clamps them).
        """
        img = self.image(key)
        if img is None:
            return None
        return img[max(0, y0):y1, max(0, x0):x1]

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

    def __init__(self, h5_path, bin_size=None, grid_mapping=None, variant=None):
        self._path = str(h5_path)
        self._f = h5py.File(self._path, "r")
        self.aggregation = str(self._f.attrs.get("aggregation", "sum"))
        try:
            if grid_mapping is not None:
                self._keys = _validate_binned_h5(
                    self._f, self._path, bin_size, grid_mapping, variant=variant)
            else:
                self._keys = sorted(
                    (key for key, obj in self._f.items()
                     if _BIN_KEY_RE.fullmatch(key) and isinstance(obj, h5py.Dataset)),
                    key=_bin_sort_key)
        except Exception:
            self._f.close()
            raise

    def keys(self) -> list:
        return list(self._keys)

    def __contains__(self, key) -> bool:
        return key in self._keys

    def image(self, key: str) -> Optional[np.ndarray]:
        if key not in self._keys:
            return None
        return np.clip(self._f[key][:].astype(np.float64), 0, 1e9)

    def region(self, key, y0, y1, x0, x1):
        if key not in self._keys:
            return None
        # h5py reads only the requested slice from disk (clamps stop past edge).
        sub = self._f[key][max(0, y0):y1, max(0, x0):x1]
        return np.clip(sub.astype(np.float64), 0, 1e9)

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


class _ArchiveSource(BinImageSource):
    """Map spatial cells onto acquisition frames in the unbinned archive."""

    is_raw = False
    is_archive = True
    source_kind = "archive"

    def __init__(self, archive, grid_mapping):
        gm = load_grid_mapping(grid_mapping)
        self._bins = gm["bins"]
        self._store = FrameStore(archive=archive)
        self._cache = OrderedDict()

    def keys(self) -> list:
        return sorted(self._bins.keys(), key=_bin_sort_key)

    def __contains__(self, key) -> bool:
        return key in self._bins

    def _sum(self, key, region=None):
        if key not in self._bins:
            return None
        summed = None
        for gi in self._bins[key]:
            if region is None:
                frame = self._store.frame(gi)
            else:
                frame = self._store.region(gi, *region)
            frame = frame.astype(np.float64)
            summed = frame if summed is None else summed + frame
        if summed is not None:
            np.clip(summed, 0, 1e9, out=summed)
        return summed

    def image(self, key: str) -> Optional[np.ndarray]:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        image = self._sum(key)
        if image is not None:
            self._cache[key] = image
            if len(self._cache) > 8:
                self._cache.popitem(last=False)
        return image

    def region(self, key, y0, y1, x0, x1):
        return self._sum(key, (y0, y1, x0, x1))

    def sum_all(self, max_bins=None, progress=None) -> np.ndarray:
        keys = self.keys()
        if max_bins:
            keys = keys[:max_bins]
        acc = None
        for i, key in enumerate(keys):
            image = self.image(key)
            if image is not None:
                acc = image if acc is None else acc + image
            if progress is not None:
                progress(i + 1, len(keys))
        return acc if acc is not None else np.zeros((1, 1))

    def close(self):
        self._cache.clear()
        self._store.close()


class _RawSource(BinImageSource):
    """Bin raw frames on demand — used when no binned h5 exists.

    Bins are resolved from a saved ``grid_mapping_NxN.h5`` when present,
    otherwise built in-memory from the scan positions (or a synthesized raster).
    """

    is_raw = True

    def __init__(self, dm, bin_size, scan=None, n_cols=None, grid_mapping=None):
        self.bin_size = bin_size
        gm_path = grid_mapping or dm.grid_mapping(bin_size=bin_size, scan=scan)
        if gm_path and Path(gm_path).exists():
            gm = validate_grid_mapping_bin_size(gm_path, bin_size)
            self._bins = gm["bins"]
            self._xrd_files = gm["xrd_files"]
            self._frame_map = gm["frame_map"]
        else:
            xrd_dir = dm.xrd_frames_dir(scan=scan)
            if not xrd_dir or not Path(xrd_dir).exists():
                raise FileNotFoundError(
                    f"No binned file and no raw frames directory ({xrd_dir}) to "
                    "fall back to. Link the raw scan in Setup or build bins.")
            scan_no = dm.scan_number(scan)
            if scan_no is None:
                raise ValueError(
                    "Cannot read raw frames: no scan selected and the project "
                    "has no global scan.number. Pass an explicit scan.")
            self._xrd_files, self._frame_map, n_total = load_xrd_metadata(
                xrd_dir, scan_number=scan_no)
            if n_total == 0:
                raise FileNotFoundError(
                    f"No raw XRD frames found for scan {scan_no} in {xrd_dir}. "
                    "There is no binned file for this size and the raw frames "
                    "could not be read — the raw data source (network share) is "
                    "likely not mounted/reachable, or the scan number is wrong. "
                    "Connect the raw source or build the binned file.")
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
                    "Run 'xrd-app grid' (optionally with --shape ROWSxCOLS).")
            grid_to_frames = {}
            for gi in range(n_total):
                grid_to_frames.setdefault(
                    (int(grid_row[gi]), int(grid_col[gi])), []).append(gi)
            self._bins, _, _ = build_bin_mapping(
                n_rows, n_cols2, bin_size, grid_to_frames)
        self._cache = OrderedDict()

    def keys(self) -> list:
        return sorted(self._bins.keys(), key=_bin_sort_key)

    def __contains__(self, key) -> bool:
        return key in self._bins

    def image(self, key: str) -> Optional[np.ndarray]:
        if key not in self._bins:
            return None
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        img = sum_raw_frames(self._xrd_files, self._frame_map, self._bins[key])
        if img is not None:
            self._cache[key] = img
            if len(self._cache) > 8:
                self._cache.popitem(last=False)
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

    def close(self):
        self._cache.clear()


def open_bin_source(dm, bin_size, scan=None, n_cols=None, grid_mapping=None,
                    variant=None) -> BinImageSource:
    """Open the best per-bin image source for a scan + bin size.

    Uses the prebuilt ``xrd_NxN_bins.h5`` when it exists (fast); otherwise falls
    back to summing raw frames on demand (``is_raw`` True, slower). This includes
    1×1: a built ``xrd_1x1_bins.h5`` (one frame per bin) is used when present, so
    a project stays fully browsable after its raw frames have been deleted to
    save disk. Only when no binned file exists do we need the raw frames.
    ``grid_mapping`` overrides which grid the raw source bins against (so a
    feature catalog built on a non-default grid maps its bins correctly).
    ``variant`` selects a non-default bins file (e.g. ``"territory"`` →
    ``xrd_1x1_bins_territory.h5``, keyed by ``"<tid>_0"``), so a territorial
    catalog loads its own per-territory frames rather than the plain grid's.
    """
    gm = grid_mapping or dm.grid_mapping(bin_size=bin_size, scan=scan,
                                         variant=variant)
    grid_variant = _grid_variant(gm)
    if grid_mapping and variant and variant != grid_variant:
        raise ValueError(
            f"Grid mapping variant {grid_variant!r} does not match requested "
            f"variant {variant!r}: {gm}")
    if grid_mapping and grid_variant:
        variant = grid_variant
    h5 = dm.binned_h5(bin_size, scan=scan, variant=variant)
    archive = dm.unbinned_archive_h5(scan=scan)
    if gm and Path(gm).exists():
        validate_grid_mapping_bin_size(gm, bin_size)
    # The ordinary 1x1 bins file may contain collision sums and float conversion.
    # Prefer the lossless archive plus mapping there; explicit variants (notably
    # territory) and coarser prebuilt bins retain their exact-file precedence.
    if bin_size == 1 and not variant and archive.exists() and gm and Path(gm).exists():
        return _ArchiveSource(archive, gm)
    if h5 and os.path.exists(h5):
        return _H5Source(h5, bin_size=bin_size,
                         grid_mapping=gm if gm and Path(gm).exists() else None,
                         variant=variant)
    if archive.exists() and gm and Path(gm).exists():
        return _ArchiveSource(archive, gm)
    return _RawSource(dm, bin_size, scan=scan, n_cols=n_cols, grid_mapping=gm)
