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
from collections import defaultdict
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
                           snr_threshold=DEFAULT_SNR, log: Callable[[str], None] = print):
    """Detect peaks in every bin and return {bin_key: [peak_dicts]}."""
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

    log(f"  Detection complete: {n_total_peaks} peaks in {len(all_detections)} bins")
    return all_detections


# ── Phase 2: spatial linking ───────────────────────────────────────
def link_peaks(all_detections, n_rows, n_cols, link_tolerance=DEFAULT_LINK_TOLERANCE):
    """Link same-peak detections across neighboring bins via Union-Find.

    Two detections in adjacent bins are linked if they fall within
    ``link_tolerance`` pixels at the same detector position. Returns a list
    of features, each a list of member nodes
    ``(bin_key, peak_index, row, col, x, y, peak_dict)``.
    """
    nodes = []
    for bk, peaks in all_detections.items():
        r, c = int(bk.split("_")[0]), int(bk.split("_")[1])
        for pi, p in enumerate(peaks):
            nodes.append((bk, pi, r, c, p['x'], p['y'], p))

    if not nodes:
        return []

    parent = list(range(len(nodes)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    spatial = defaultdict(list)
    for idx, (bk, pi, r, c, x, y, p) in enumerate(nodes):
        spatial[(r, c)].append(idx)

    for idx, (bk, pi, r, c, x, y, p) in enumerate(nodes):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if (nr, nc) not in spatial:
                    continue
                for nidx in spatial[(nr, nc)]:
                    nx, ny = nodes[nidx][4], nodes[nidx][5]
                    if np.sqrt((x - nx)**2 + (y - ny)**2) <= link_tolerance:
                        union(idx, nidx)

    components = defaultdict(list)
    for idx in range(len(nodes)):
        components[find(idx)].append(idx)

    return [[nodes[i] for i in member_indices] for member_indices in components.values()]


# ── Phase 3: characterize and filter ───────────────────────────────
def check_gaussian_profile(members):
    """Return (is_gaussian, reason) based on the cross-bin intensity profile."""
    if len(members) < 2:
        return False, "isolated: single-bin detection"

    intensities = [m[6]['cleaned_intensity'] for m in members]
    positions = [(m[2], m[3]) for m in members]

    imax = np.argmax(intensities)
    center_r, center_c = positions[imax]
    peak_intensity = intensities[imax]

    if peak_intensity <= 0:
        return False, "non-positive peak intensity"

    i_arr = np.array(intensities)
    cv = np.std(i_arr) / np.mean(i_arr) if np.mean(i_arr) > 0 else 0
    if cv < 0.05:
        return False, f"flat profile: CV={cv:.3f} (no clear center)"

    distances = [np.sqrt((r - center_r)**2 + (c - center_c)**2) for r, c in positions]

    n_closer_brighter = 0
    n_comparisons = 0
    for i in range(len(members)):
        for j in range(i + 1, len(members)):
            if abs(distances[i] - distances[j]) < 0.1:
                continue
            n_comparisons += 1
            if (distances[i] < distances[j] and intensities[i] > intensities[j]) or \
               (distances[i] > distances[j] and intensities[i] < intensities[j]):
                n_closer_brighter += 1

    if n_comparisons == 0:
        return True, "small cluster, no directional check possible"

    monotonic_fraction = n_closer_brighter / n_comparisons
    if monotonic_fraction < 0.4:
        return False, (f"non-Gaussian: only {monotonic_fraction:.0%} of pairs "
                       f"follow distance-intensity trend")

    return True, f"Gaussian-like: {monotonic_fraction:.0%} monotonic, {len(members)} bins"


def _best_per_bin(members, tth_map=None, beam_center=None):
    """Build intensity_profile keeping the strongest peak per bin."""
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


def characterize_features(features, beam_center=None, tth_map=None, ref_tth_map=None):
    """Classify each feature as kept or filtered, with reason and metrics."""
    kept, filtered = [], []

    for members in features:
        bins_in_feature = set(m[0] for m in members)
        is_gaussian, reason = check_gaussian_profile(members)

        intensities = [m[6]['cleaned_intensity'] for m in members]
        snrs = [m[6]['snr'] for m in members]
        xs = [m[4] for m in members]
        ys = [m[5] for m in members]
        rows = [m[2] for m in members]
        cols = [m[3] for m in members]

        imax = np.argmax(intensities)
        reflection = members[imax][6]['label']
        det_x = int(np.mean(xs))
        det_y = int(np.mean(ys))

        feature_info = {
            "reflection": reflection,
            "detector_x": det_x,
            "detector_y": det_y,
            "peak_intensity": float(max(intensities)),
            "mean_snr": float(np.mean(snrs)),
            "n_bins": len(bins_in_feature),
            "spatial_extent": sorted(bins_in_feature),
            "center_bin": members[imax][0],
            "center_row": rows[imax],
            "center_col": cols[imax],
            "intensity_profile": _best_per_bin(members, tth_map, beam_center),
            "reason": reason,
        }

        if ref_tth_map is not None and reflection in ref_tth_map:
            feature_info["ref_tth"] = ref_tth_map[reflection]

        profile = feature_info["intensity_profile"]
        bin_chis, bin_tths, bin_weights = [], [], []
        for entry in profile.values():
            if not isinstance(entry, dict):
                continue
            w = entry.get("integrated", entry.get("intensity", 0))
            if w <= 0:
                continue
            if "chi" in entry:
                bin_chis.append(entry["chi"])
            if "tth" in entry:
                bin_tths.append(entry["tth"])
            bin_weights.append(w)

        if len(bin_chis) >= 3 and len(bin_weights) == len(bin_chis):
            wa = np.array(bin_weights)
            ca = np.array(bin_chis)
            wn = wa / wa.sum()
            mu = np.dot(wn, ca)
            var = np.dot(wn, (ca - mu) ** 2)
            feature_info["rocking_fwhm"] = round(2.3548 * np.sqrt(var), 4)

        if len(bin_tths) >= 3 and len(bin_weights) == len(bin_tths):
            ref_val = ref_tth_map.get(reflection) if ref_tth_map else None
            if ref_val is not None:
                wa = np.array(bin_weights)
                ta = np.array(bin_tths) - ref_val
                wn = wa / wa.sum()
                mu = np.dot(wn, ta)
                var = np.dot(wn, (ta - mu) ** 2)
                feature_info["strain_breadth"] = round(2.3548 * np.sqrt(var), 4)

        if beam_center is not None:
            by, bx = beam_center
            chi = float(np.degrees(np.arctan2(det_y - by, det_x - bx)))
            feature_info["chi_deg"] = round(chi, 1)

        (kept if is_gaussian else filtered).append(feature_info)

    return kept, filtered


# ── Phase 4: output ────────────────────────────────────────────────
def write_feature_catalog(kept, output_path, log: Callable[[str], None] = print):
    for i, f in enumerate(kept):
        f["feature_id"] = i + 1
    with open(output_path, "w") as fh:
        json.dump(kept, fh, indent=2)
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


# ── Orchestrator ───────────────────────────────────────────────────
def run_analysis(
    bins_h5: Union[str, Path],
    tth_path: Union[str, Path],
    detector_path: Union[str, Path],
    reflections_path: Union[str, Path],
    grid_mapping: Union[str, Path, dict],
    output_dir: Union[str, Path],
    bin_size: int = 3,
    snr_threshold: float = DEFAULT_SNR,
    link_tolerance: int = DEFAULT_LINK_TOLERANCE,
    log: Callable[[str], None] = print,
) -> dict:
    """Run the full detect -> link -> filter -> write pipeline.

    Returns a summary dict with kept/filtered counts and output file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{bin_size}x{bin_size}"

    log("Loading detector module and shared data...")
    det = io.load_module(detector_path)
    tth_map = io.load_tth_map(tth_path)
    degs, deg_labels = io.load_reflections(reflections_path)
    log(f"  Reflections: {deg_labels}")

    gm = io.load_grid_mapping(grid_mapping)
    n_rows, n_cols = gm["n_bin_rows"], gm["n_bin_cols"]
    log(f"  Grid: {n_rows} x {n_cols} = {n_rows * n_cols} bins")

    log("\n--- Phase 1: Per-bin detection ---")
    all_detections = run_detection_all_bins(
        bins_h5, tth_map, degs, deg_labels, det, snr_threshold, log)

    log("\n--- Phase 2: Linking peaks across bins ---")
    features = link_peaks(all_detections, n_rows, n_cols, link_tolerance)
    n_single = sum(1 for f in features if len(set(m[0] for m in f)) == 1)
    n_multi = len(features) - n_single
    log(f"  Found {len(features)} raw features: {n_multi} multi-bin, {n_single} single-bin")

    beam_center = estimate_beam_center(tth_map)
    log(f"  Beam center (y, x): ({beam_center[0]:.1f}, {beam_center[1]:.1f})")

    ref_tth_map = {lbl: round(d, 5) for d, lbl in zip(degs, deg_labels)}

    log("\n--- Phase 3: Gaussian profile filtering ---")
    kept, filtered = characterize_features(
        features, beam_center=beam_center, tth_map=tth_map, ref_tth_map=ref_tth_map)

    kept.sort(key=lambda f: (f["center_row"], f["center_col"]))
    filtered.sort(key=lambda f: (f["center_row"], f["center_col"]))
    log(f"  Kept:     {len(kept)} features (Gaussian-like spatial profile)")
    log(f"  Filtered: {len(filtered)} features")

    log("\n--- Phase 4: Writing output ---")
    catalog_path = output_dir / f"feature_catalog_{suffix}.json"
    kept_csv = output_dir / f"kept_peaks_{suffix}.csv"
    filtered_csv = output_dir / f"filtered_peaks_{suffix}.csv"
    write_feature_catalog(kept, catalog_path, log)
    write_peak_table(kept, kept_csv, "kept peaks", log)
    write_peak_table(filtered, filtered_csv, "filtered peaks", log)

    return {
        "n_kept": len(kept),
        "n_filtered": len(filtered),
        "feature_catalog": str(catalog_path),
        "kept_csv": str(kept_csv),
        "filtered_csv": str(filtered_csv),
    }
