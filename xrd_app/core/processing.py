"""
Spatial feature analysis and peak detection for binned nano-XRD data.

Faithful, de-hardcoded port of ``analysis/spatial_feature_analysis.py``:

  Phase 1  per-bin detection      (run a detector module over every bin)
  Phase 2  spatial linking        (Union-Find across neighboring bins)
  Phase 3  Gaussian filtering     (keep clusters with a clear bright center)
  Phase 4  output                 (shapes JSON via the CLI + kept/filtered CSVs)

The detector itself is supplied as an external module (``detector_script``)
exposing ``radial_median_subtract``, ``fast_tophat``, ``build_tth_band_masks``,
``detect_in_band`` and ``precompute_tth`` — this is what CVEvolve evolves.
"""

from __future__ import annotations

import csv
import json
import os
import time
import multiprocessing as mp
from pathlib import Path
from typing import Callable, Union

import h5py
import numpy as np

from . import io

DEFAULT_SNR = 4.0
DEFAULT_LINK_TOLERANCE = 5  # pixels — max distance to consider same peak across bins
DEFAULT_MAX_PEAKS_PER_BIN = 25


# ── Phase 1: per-bin detection ─────────────────────────────────────
def detect_peaks_with_intensity(image, tth_map, degs, deg_labels, tth_data, det,
                                snr_threshold=DEFAULT_SNR,
                                max_peaks=DEFAULT_MAX_PEAKS_PER_BIN):
    """Run the detector pipeline but preserve intensity/SNR per peak."""
    cleaned = det.radial_median_subtract(image, tth_data)
    tophat = det.fast_tophat(cleaned, size=15)
    bands = det.build_tth_band_masks(tth_map, degs, deg_labels, tth_tolerance=0.4)

    all_peaks = []
    for label, band_mask in bands.items():
        peaks = det.detect_in_band(
            tophat, cleaned, band_mask, label,
            snr_threshold=snr_threshold, min_pixels=3, max_pixels=150,
            min_compactness=0.12, ignore_edge=3
        )
        all_peaks.extend(peaks)

    if not all_peaks:
        return [], cleaned

    all_peaks.sort(key=lambda p: p['snr'], reverse=True)
    kept = []
    for peak in all_peaks:
        is_dup = any(
            np.sqrt((peak['y'] - e['y'])**2 + (peak['x'] - e['x'])**2) < 15
            for e in kept
        )
        if not is_dup:
            kept.append(peak)

    if max_peaks and len(kept) > max_peaks:
        kept = kept[:max_peaks]

    return kept, cleaned


# Worker-process state for parallel per-bin detection. The parent sets _DET_CTX
# before the Pool is created so forked children inherit the (read-only) detector,
# maps and params; each worker opens its own HDF5 handle via the initializer.
_DET_CTX: dict = {}
_DET_STATE: dict = {}


def _detect_worker_init():
    _DET_STATE["h5"] = h5py.File(_DET_CTX["h5_path"], "r")


def _detect_one_bin(bk):
    """Detect peaks in a single bin — the exact body of the serial loop, so the
    serial and parallel paths produce identical results. Returns (bk, peaks)."""
    c = _DET_CTX
    image = np.clip(_DET_STATE["h5"][bk][:].astype(np.float64), 0, 1e9)
    peaks, cleaned = detect_peaks_with_intensity(
        image, c["tth_map"], c["degs"], c["deg_labels"], c["tth_data"],
        c["det"], c["snr"])
    if peaks:
        for p in peaks:
            r = 3
            y0 = max(0, p['y'] - r); y1 = min(cleaned.shape[0], p['y'] + r + 1)
            x0 = max(0, p['x'] - r); x1 = min(cleaned.shape[1], p['x'] + r + 1)
            p['cleaned_intensity'] = float(np.max(cleaned[y0:y1, x0:x1]))
    return bk, peaks


def run_detection_all_bins(h5_path, tth_map, degs, deg_labels, det,
                           snr_threshold=DEFAULT_SNR, log: Callable[[str], None] = print,
                           progress: Callable[[int, int], None] = None,
                           n_workers: int = None):
    """Detect peaks in every bin and return {bin_key: [peak_dicts]}.

    Per-bin detection is CPU-bound and independent, so it runs across cores
    (``n_workers``, default ``cpu_count-1``). Results are consumed in sorted-key
    order, and serial and parallel both call :func:`_detect_one_bin`, so the
    output is identical to the old serial loop. Falls back to serial on 1 worker
    or if the pool can't start.

    ``progress(i, n)`` is called after each bin so callers (the CLI / GUI) can
    show an ``(i/n)`` count for long jobs.
    """
    tth_data = det.precompute_tth(tth_map)
    all_detections = {}

    with h5py.File(str(h5_path), "r") as h5f:
        bin_keys = sorted(h5f.keys(),
                          key=lambda k: (int(k.split("_")[0]), int(k.split("_")[1])))
    n_bins = len(bin_keys)
    nw = n_workers if n_workers is not None else max(1, (os.cpu_count() or 2) - 1)
    log(f"  Running detector on {n_bins} bins ({nw} worker{'s' if nw != 1 else ''})...")

    global _DET_CTX
    _DET_CTX = dict(det=det, tth_map=tth_map, degs=degs, deg_labels=deg_labels,
                    tth_data=tth_data, snr=snr_threshold, h5_path=str(h5_path))

    def _consume(it):
        for i, (bk, peaks) in enumerate(it):
            if peaks:
                all_detections[bk] = peaks
            if progress is not None:
                progress(i + 1, n_bins)

    used_parallel = False
    if nw > 1:
        try:
            ctxmp = mp.get_context("fork")
            with ctxmp.Pool(nw, initializer=_detect_worker_init) as pool:
                _consume(pool.imap(_detect_one_bin, bin_keys, chunksize=8))
            used_parallel = True
        except Exception as e:   # pragma: no cover — degrade to serial
            log(f"  parallel detection failed ({e}); serial fallback")
            all_detections.clear()
    if not used_parallel:
        with h5py.File(str(h5_path), "r") as h5f:
            _DET_STATE["h5"] = h5f
            try:
                _consume(_detect_one_bin(bk) for bk in bin_keys)
            finally:
                _DET_STATE.pop("h5", None)

    n_total_peaks = sum(len(v) for v in all_detections.values())
    log(f"  Detection complete: {n_total_peaks} peaks in {len(all_detections)} bins")
    return all_detections


# ── Shared characterization helpers (used by the GUI explore mode) ──
def _best_per_bin(members, tth_map=None, beam_center=None):
    """Build intensity_profile keeping the strongest peak per bin.

    Kept here (alongside :func:`estimate_beam_center`) as a shared utility the
    viewer uses when characterizing manually-explored features. The bundled
    shape algorithm (``ShapeAlgorithms/gaussian.py``) carries its own copy so it
    stays self-contained.
    """
    best = {}
    for m in members:
        bk = m[0]
        intensity = m[6]['cleaned_intensity']
        integrated = m[6].get('integrated_intensity', intensity)
        px, py = m[4], m[5]
        if bk not in best or intensity > best[bk]["intensity"]:
            entry = {
                "intensity": round(intensity, 1),
                "integrated": round(integrated, 1),
                "det_x": int(px),
                "det_y": int(py),
            }
            if tth_map is not None:
                entry["tth"] = round(float(tth_map[int(py), int(px)]), 5)
            if beam_center is not None:
                by, bx = beam_center
                entry["chi"] = round(float(np.degrees(np.arctan2(py - by, px - bx))), 2)
            best[bk] = entry
    return best


# ── Beam-center estimation (shared geometry utility) ───────────────
def estimate_beam_center(tth_map):
    """Estimate the beam center (y0, x0) by fitting tth ~ k * dist."""
    from scipy.optimize import minimize
    step = 10
    ys, xs = np.mgrid[0:tth_map.shape[0]:step, 0:tth_map.shape[1]:step]
    ts = tth_map[::step, ::step]

    def objective(params):
        y0, x0 = params
        dist = np.sqrt((ys - y0)**2 + (xs - x0)**2)
        dist = np.maximum(dist, 1e-6)
        k = np.sum(ts * dist) / np.sum(dist**2)
        return np.sum((ts - k * dist)**2)

    res = minimize(objective, [tth_map.shape[0] // 2, tth_map.shape[1] + 200],
                   method='Nelder-Mead')
    return res.x[0], res.x[1]


# ── Phase 4: output ────────────────────────────────────────────────
def write_peak_table(peaks, output_path, title, log: Callable[[str], None] = print):
    with open(output_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "feature_id", "reflection", "bin_key", "center_row", "center_col",
            "detector_x", "detector_y", "peak_intensity", "mean_snr",
            "n_bins", "spatial_extent", "reason"
        ])
        for i, f in enumerate(peaks):
            writer.writerow([
                f.get("feature_id", i + 1), f["reflection"], f["center_bin"],
                f["center_row"], f["center_col"], f["detector_x"], f["detector_y"],
                f"{f['peak_intensity']:.1f}", f"{f['mean_snr']:.1f}", f["n_bins"],
                " ".join(f["spatial_extent"]), f["reason"],
            ])
    log(f"  Wrote {len(peaks)} {title} to {output_path}")


# ── Serialization helpers ──────────────────────────────────────────
def _coerce(obj):
    """Recursively convert numpy scalars/arrays to JSON-native Python types."""
    if isinstance(obj, dict):
        return {k: _coerce(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_coerce(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return _coerce(obj.tolist())
    return obj


# ── Public stage 1: peak finding ───────────────────────────────────
def run_peaks(
    bins_h5: Union[str, Path],
    tth_path: Union[str, Path],
    detector_path: Union[str, Path],
    reflections_path: Union[str, Path],
    bin_size: int = 3,
    snr_threshold: float = DEFAULT_SNR,
    progress: Callable[[int, int], None] = None,
    log: Callable[[str], None] = print,
    n_workers: int = None,
) -> dict:
    """Phase 1 only: run a detector over every bin → per-bin peaks.

    Returns a JSON-serializable dict::

        {bin_size, detector, snr, n_peaks, n_bins_with_peaks,
         peaks_by_bin: {"<r>_<c>": [peak_dict, ...]}}

    Each ``peak_dict`` keeps the detector's fields (x, y, snr, label,
    cleaned_intensity, …) so it round-trips into :func:`run_shapes`.
    """
    det = io.load_module(detector_path)
    tth_map = io.load_tth_map(tth_path)
    degs, deg_labels = io.load_reflections(reflections_path)
    log(f"  Reflections: {deg_labels}")

    all_detections = run_detection_all_bins(
        bins_h5, tth_map, degs, deg_labels, det, snr_threshold, log, progress,
        n_workers=n_workers)

    peaks_by_bin = {bk: _coerce(peaks) for bk, peaks in all_detections.items()}
    n_peaks = sum(len(v) for v in peaks_by_bin.values())
    return {
        "bin_size": bin_size,
        "detector": Path(detector_path).stem,
        "snr": snr_threshold,
        "n_peaks": n_peaks,
        "n_bins_with_peaks": len(peaks_by_bin),
        "peaks_by_bin": peaks_by_bin,
    }


# ── Public: combined (peak + shape in one per-frame pass) ──────────
def _fmt_duration(seconds: float) -> str:
    """Human-readable duration ("3h 12m", "8m 5s", "42s", "—" for unknown)."""
    if seconds == float("inf") or seconds != seconds:  # inf or NaN
        return "—"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def _features_from_by_bin(by_bin: dict) -> list:
    """Build the viewer-compatible feature list from a ``{bin: {label: pts}}`` map.

    Shared by both combined paths (per-center-bin loop and the global one-pass).
    Points are coerced to ints; these per-frame detections carry no per-bin
    intensities, so ``peak_intensity``/``mean_snr`` are null and
    ``intensity_profile`` is empty.
    """
    features, fid = [], 0
    for bk in sorted(by_bin, key=lambda k: (int(k.split("_")[0]), int(k.split("_")[1]))):
        row, col = int(bk.split("_")[0]), int(bk.split("_")[1])
        for label, pts in by_bin[bk].items():
            for x, y in pts:
                fid += 1
                features.append({
                    "feature_id": fid, "reflection": label,
                    "detector_x": int(round(x)), "detector_y": int(round(y)),
                    "center_bin": bk, "center_row": row, "center_col": col,
                    "n_bins": 1, "peak_intensity": None, "mean_snr": None,
                    "intensity_profile": {},
                })
    return features


def run_combined(
    detector_path: Union[str, Path],
    tth_path: Union[str, Path],
    reflections_path: Union[str, Path],
    bins_h5: Union[str, Path],
    grid_mapping: Union[str, Path, dict],
    progress: Callable[[int, int], None] = None,
    log: Callable[[str], None] = print,
    **params,
) -> dict:
    """Run a combined (per-frame) algorithm over every center bin.

    The detector exposes ``run_full_pipeline(center_bin, bins_h5, tth_map, degs,
    deg_labels, grid_mapping, **params)`` returning ``{label: [(x, y), ...]}`` —
    final spatially-validated points (peak + shape in one pass). Returns::

        {algorithm, bin_size, n_features, by_bin, features}

    ``features`` is a viewer-compatible feature list (points only — these
    detectors don't emit per-bin intensities, so ``peak_intensity``/``mean_snr``
    are null and ``intensity_profile`` is empty).
    """
    import json as _json
    import sys as _sys
    det = io.load_module(detector_path)
    tth_map = io.load_tth_map(tth_path)
    degs, deg_labels = io.load_reflections(reflections_path)
    if isinstance(grid_mapping, (str, Path)):
        with open(grid_mapping) as f:
            grid_mapping = _json.load(f)
    log(f"  Reflections: {deg_labels}")

    # Identity for output files: a sub-foldered detector.py is identified by its
    # folder name (e.g. "1x1_global_perframe_uf_voigt"), a flat algo by its stem.
    _dp = Path(detector_path)
    algo_name = _dp.parent.name if _dp.stem == "detector" else _dp.stem

    # Fast path: a detector exposing run_full_scan does detection + linking +
    # verification ONCE over the whole scan (reading each frame ~twice) instead
    # of the per-center-bin loop below that re-reads a (2r+1)^2 window per bin.
    if hasattr(det, "run_full_scan"):
        log("  Using global one-pass (run_full_scan): each frame read once, "
            "not per-center-bin.")
        t0 = time.time()
        # The detector parallelises detection with multiprocessing (fork); its
        # worker functions pickle by module name, so register the dynamically
        # loaded module under its name for the duration of the call.
        _prev_mod = _sys.modules.get(det.__name__)
        _sys.modules[det.__name__] = det
        try:
            by_bin = det.run_full_scan(
                str(bins_h5), tth_map, degs, deg_labels, grid_mapping,
                progress=progress, log=log, **params)
        finally:
            if _prev_mod is not None:
                _sys.modules[det.__name__] = _prev_mod
            else:
                _sys.modules.pop(det.__name__, None)
        features = _features_from_by_bin(by_bin)
        log(f"  Combined complete: {len(features)} features in {len(by_bin)} bins "
            f"({_fmt_duration(time.time() - t0)})")
        return {
            "algorithm": algo_name,
            "bin_size": 1,
            "n_features": len(features),
            "by_bin": by_bin,
            "features": features,
        }

    with h5py.File(str(bins_h5), "r") as h5f:
        bin_keys = sorted(h5f.keys(),
                          key=lambda k: (int(k.split("_")[0]), int(k.split("_")[1])))
    n_bins = len(bin_keys)
    radius = int(params.get("spatial_radius", 5))
    per_bin_frames = (2 * radius + 1) ** 2
    log(f"  Running combined pipeline on {n_bins} bins...")
    log(f"  (per-frame pass: each bin reads up to {per_bin_frames} neighbor "
        f"frames at spatial_radius={radius} — this is heavy; expect seconds per "
        f"bin. Progress + ETA below.)")

    by_bin, n_feat = {}, 0
    t_start = time.time()
    t_last = t_start
    for i, bk in enumerate(bin_keys):
        try:
            validated = det.run_full_pipeline(
                bk, str(bins_h5), tth_map, degs, deg_labels, grid_mapping, **params)
        except Exception as e:  # one bad bin shouldn't sink the whole run
            log(f"  [{bk}] failed: {e}")
            validated = {}
        clean = {}
        for label, pts in (validated or {}).items():
            ipts = [[int(round(x)), int(round(y))] for (x, y) in pts]
            if not ipts:
                continue
            clean[label] = ipts
            n_feat += len(ipts)
        if clean:
            by_bin[bk] = clean
        if progress is not None:
            progress(i + 1, n_bins)

        # Time-based heartbeat: combined per-frame bins can take several seconds
        # each, so the step-throttled PROGRESS line above can stay silent for a
        # long time on a large scan. Emit a human-readable rate + ETA after the
        # first bin and then every ~10 s so it's clear the run is alive.
        now = time.time()
        if i == 0 or now - t_last >= 10.0 or i + 1 == n_bins:
            t_last = now
            done = i + 1
            elapsed = now - t_start
            rate = done / elapsed if elapsed > 0 else 0.0   # bins/sec
            eta = (n_bins - done) / rate if rate > 0 else float("inf")
            log(f"  [combined] {done}/{n_bins} bins ({100 * done / n_bins:.1f}%)"
                f"  ·  {rate * 60:.1f} bins/min  ·  ETA {_fmt_duration(eta)}"
                f"  ·  {n_feat} features so far")

    features = _features_from_by_bin(by_bin)
    log(f"  Combined complete: {len(features)} features in {len(by_bin)} bins")
    return {
        "algorithm": algo_name,
        "bin_size": 1,
        "n_features": len(features),
        "by_bin": by_bin,
        "features": features,
    }


# ── Public stage 2: shape finding ──────────────────────────────────
def _default_shape_path() -> Path:
    """Bundled baseline shape algorithm, used when no ``shape_path`` is given."""
    return Path(__file__).resolve().parent.parent / "ShapeAlgorithms" / "gaussian.py"


def run_shapes(
    peaks: dict,
    tth_path: Union[str, Path],
    grid_mapping: Union[str, Path, dict],
    reflections_path: Union[str, Path],
    bin_size: int = 3,
    link_tolerance: int = DEFAULT_LINK_TOLERANCE,
    shape_path: Union[str, Path, None] = None,
    progress: Callable[[int, int], None] = None,
    log: Callable[[str], None] = print,
) -> dict:
    """Phase 2+3: link peaks across bins, keep gaussian-like clusters, characterize.

    The shape algorithm itself is supplied as an external module
    (``shape_path``, default :func:`_default_shape_path`) exposing
    ``link_peaks`` and ``characterize_features`` — this is the swappable part,
    mirroring how :func:`run_peaks` loads a detector module.

    ``peaks`` is a :func:`run_peaks` result OR a loaded ``*_peaks.json`` — both
    carry ``peaks_by_bin``. A "shape" is a kept (gaussian-like) feature. Returns
    ``{bin_size, n_kept, n_filtered, kept: [...], filtered: [...]}``.
    """
    all_detections = peaks["peaks_by_bin"] if isinstance(peaks, dict) and \
        "peaks_by_bin" in peaks else peaks

    shape = io.load_module(shape_path or _default_shape_path())
    tth_map = io.load_tth_map(tth_path)
    degs, deg_labels = io.load_reflections(reflections_path)
    gm = io.load_grid_mapping(grid_mapping)
    n_rows, n_cols = gm["n_bin_rows"], gm["n_bin_cols"]
    log(f"  Grid: {n_rows} x {n_cols} = {n_rows * n_cols} bins")

    territories = gm.get("territories")
    if territories is not None:
        # Territorial / cell-model mapping: link across physical (X, Y) neighbors
        # instead of the N×N 8-neighborhood. The shape algorithm must expose the
        # neighbor-graph link_peaks signature (ShapeAlgorithms/territory.py).
        neighbors = {k: info.get("neighbors", []) for k, info in territories.items()}
        if hasattr(shape, "set_centroids"):
            shape.set_centroids({k: info.get("centroid_rc") for k, info in territories.items()})
        log(f"  Linking peaks across {len(neighbors)} territories (Union-Find)...")
        features = shape.link_peaks(all_detections, neighbors, link_tolerance)
    else:
        log("  Linking peaks across bins (Union-Find)...")
        features = shape.link_peaks(all_detections, n_rows, n_cols, link_tolerance)
    n_single = sum(1 for f in features if len(set(m[0] for m in f)) == 1)
    log(f"  {len(features)} raw features: {len(features) - n_single} multi-bin, "
        f"{n_single} single-bin")
    if progress is not None:
        progress(1, 2)

    beam_center = estimate_beam_center(tth_map)
    ref_tth_map = {lbl: round(d, 5) for d, lbl in zip(degs, deg_labels)}

    log("  Filtering by gaussian profile + characterizing shapes...")
    kept, filtered = shape.characterize_features(
        features, beam_center=beam_center, tth_map=tth_map, ref_tth_map=ref_tth_map)
    kept.sort(key=lambda f: (f["center_row"], f["center_col"]))
    filtered.sort(key=lambda f: (f["center_row"], f["center_col"]))
    for i, f in enumerate(kept):
        f["feature_id"] = i + 1
    if progress is not None:
        progress(2, 2)

    log(f"  Kept (shapes): {len(kept)}   Filtered: {len(filtered)}")
    return {
        "bin_size": bin_size,
        "link_tolerance": link_tolerance,
        "n_kept": len(kept),
        "n_filtered": len(filtered),
        "kept": _coerce(kept),
        "filtered": _coerce(filtered),
    }
