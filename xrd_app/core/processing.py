"""
Spatial feature analysis and peak detection for binned nano-XRD data.

Faithful, de-hardcoded port of ``analysis/spatial_feature_analysis.py``:

  Phase 1  per-bin detection      (run a detector module over every bin)
  Phase 2  spatial linking        (Union-Find across neighboring bins)
  Phase 3  Gaussian filtering     (keep clusters with a clear bright center)
  Phase 4  output                 (feature_catalog.json + kept/filtered CSVs)

The detector itself is supplied as an external module (``detector_script``)
exposing ``radial_median_subtract``, ``fast_tophat``, ``build_tth_band_masks``,
``detect_in_band`` and ``precompute_tth`` — this is what CVEvolve evolves.
"""

from __future__ import annotations

import csv
import json
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


def run_detection_all_bins(h5_path, tth_map, degs, deg_labels, det,
                           snr_threshold=DEFAULT_SNR, log: Callable[[str], None] = print,
                           progress: Callable[[int, int], None] = None):
    """Detect peaks in every bin and return {bin_key: [peak_dicts]}.

    ``progress(i, n)`` is called after each bin so callers (the CLI / GUI) can
    show an ``(i/n)`` count for long jobs.
    """
    tth_data = det.precompute_tth(tth_map)
    all_detections = {}
    n_total_peaks = 0

    with h5py.File(str(h5_path), "r") as h5f:
        bin_keys = sorted(h5f.keys(),
                          key=lambda k: (int(k.split("_")[0]), int(k.split("_")[1])))
        n_bins = len(bin_keys)
        log(f"  Running detector on {n_bins} bins...")

        for i, bk in enumerate(bin_keys):
            image = np.clip(h5f[bk][:].astype(np.float64), 0, 1e9)
            peaks, cleaned = detect_peaks_with_intensity(
                image, tth_map, degs, deg_labels, tth_data, det, snr_threshold)

            if peaks:
                # Also measure the cleaned-image intensity at peak position
                for p in peaks:
                    r = 3
                    y0 = max(0, p['y'] - r)
                    y1 = min(cleaned.shape[0], p['y'] + r + 1)
                    x0 = max(0, p['x'] - r)
                    x1 = min(cleaned.shape[1], p['x'] + r + 1)
                    p['cleaned_intensity'] = float(np.max(cleaned[y0:y1, x0:x1]))

                all_detections[bk] = peaks
                n_total_peaks += len(peaks)

            if progress is not None:
                progress(i + 1, n_bins)

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
def write_feature_catalog(kept, output_path, log: Callable[[str], None] = print):
    for i, f in enumerate(kept):
        f["feature_id"] = i + 1
    io.atomic_write_json(output_path, kept)   # atomic: viewers never see a partial file
    log(f"  Wrote {len(kept)} kept features to {output_path}")


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
        bins_h5, tth_map, degs, deg_labels, det, snr_threshold, log, progress)

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
    det = io.load_module(detector_path)
    tth_map = io.load_tth_map(tth_path)
    degs, deg_labels = io.load_reflections(reflections_path)
    if isinstance(grid_mapping, (str, Path)):
        with open(grid_mapping) as f:
            grid_mapping = _json.load(f)
    log(f"  Reflections: {deg_labels}")

    with h5py.File(str(bins_h5), "r") as h5f:
        bin_keys = sorted(h5f.keys(),
                          key=lambda k: (int(k.split("_")[0]), int(k.split("_")[1])))
    n_bins = len(bin_keys)
    log(f"  Running combined pipeline on {n_bins} bins...")

    by_bin, features, fid = {}, [], 0
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
            row, col = int(bk.split("_")[0]), int(bk.split("_")[1])
            for x, y in ipts:
                fid += 1
                features.append({
                    "feature_id": fid, "reflection": label,
                    "detector_x": x, "detector_y": y,
                    "center_bin": bk, "center_row": row, "center_col": col,
                    "n_bins": 1, "peak_intensity": None, "mean_snr": None,
                    "intensity_profile": {},
                })
        if clean:
            by_bin[bk] = clean
        if progress is not None:
            progress(i + 1, n_bins)

    log(f"  Combined complete: {len(features)} features in {len(by_bin)} bins")
    return {
        "algorithm": Path(detector_path).stem,
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
