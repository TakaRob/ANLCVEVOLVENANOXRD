"""
Spatial feature analysis for 3x3 binned nano-XRD data.

Runs the per-bin detector on all bins, then groups detections across the
spatial grid by matching detector (x,y) positions. Peaks that appear in
clusters with Gaussian-like intensity profiles across spatial bins are kept;
isolated single-bin detections are filtered as likely noise.

Outputs:
  - feature_catalog_3x3.json:  kept features with spatial extent + intensity
  - filtered_peaks_3x3.csv:    table of filtered-out peaks for manual review
  - kept_peaks_3x3.csv:        table of kept peaks for manual review

Usage:
    python3 -u analysis/spatial_feature_analysis.py
"""

import csv
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np

BASE = Path(__file__).resolve().parent.parent
BINS_H5 = Path("/home/takaji/xrd_3x3_bins.h5")
HOLDOUT_DIR = BASE / "cvevolve_3x3" / "holdout_data"
DETECTOR_PATH = BASE / "cvevolve_3x3" / "test_data" / "tophat_band_adaptive_snr.py"
OUTPUT_DIR = BASE / "results" / "scan203"

BIN_SIZE = 3
LINK_TOLERANCE = 5  # pixels — max distance to consider same peak across bins


def load_module(path):
    spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def detect_peaks_with_intensity(image, tth_map, degs, deg_labels, tth_data, det):
    """Run the detector pipeline but preserve intensity/SNR per peak."""
    cleaned = det.radial_median_subtract(image, tth_data)
    tophat = det.fast_tophat(cleaned, size=15)
    bands = det.build_tth_band_masks(tth_map, degs, deg_labels, tth_tolerance=0.4)

    all_peaks = []
    for label, band_mask in bands.items():
        peaks = det.detect_in_band(
            tophat, cleaned, band_mask, label,
            snr_threshold=4.0, min_pixels=3, max_pixels=150,
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

    if len(kept) > 25:
        kept = kept[:25]

    return kept, cleaned


# ── Phase 1: Run detector on all bins ──────────────────────────────

def run_detection_all_bins(h5_path, tth_map, degs, deg_labels, det):
    """Detect peaks in every bin and return {bin_key: [peak_dicts]}."""
    tth_data = det.precompute_tth(tth_map)
    all_detections = {}
    n_total_peaks = 0

    with h5py.File(str(h5_path), "r") as h5f:
        bin_keys = sorted(h5f.keys(), key=lambda k: (int(k.split("_")[0]), int(k.split("_")[1])))
        n_bins = len(bin_keys)
        print(f"  Running detector on {n_bins} bins...", flush=True)

        for i, bk in enumerate(bin_keys):
            if (i + 1) % 50 == 0 or i == 0 or i == n_bins - 1:
                pct = (i + 1) / n_bins
                bar = "█" * int(pct * 30) + "░" * (30 - int(pct * 30))
                print(f"\r  [{bar}] {i+1}/{n_bins} ({100*pct:.0f}%) peaks={n_total_peaks}    ", end="", flush=True)

            image = np.clip(h5f[bk][:].astype(np.float64), 0, 1e9)
            peaks, cleaned = detect_peaks_with_intensity(image, tth_map, degs, deg_labels, tth_data, det)

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

    print(f"\n  Detection complete: {n_total_peaks} peaks in {len(all_detections)} bins")
    return all_detections


# ── Phase 2: Link peaks across spatial bins ────────────────────────

def link_peaks(all_detections, n_rows, n_cols):
    """Build features by linking same-peak detections across neighboring bins.

    Uses Union-Find on (bin_key, peak_index) nodes. Two detections in
    adjacent bins are linked if they're within LINK_TOLERANCE pixels at
    the same detector position.
    """
    # Flatten all detections into a list with IDs
    nodes = []  # (bin_key, peak_index, row, col, x, y, peak_dict)
    for bk, peaks in all_detections.items():
        r, c = int(bk.split("_")[0]), int(bk.split("_")[1])
        for pi, p in enumerate(peaks):
            nodes.append((bk, pi, r, c, p['x'], p['y'], p))

    if not nodes:
        return []

    # Union-Find
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

    # Build spatial index: (row, col) -> list of node indices
    spatial = defaultdict(list)
    for idx, (bk, pi, r, c, x, y, p) in enumerate(nodes):
        spatial[(r, c)].append(idx)

    # Link nodes in adjacent bins with matching detector positions
    for idx, (bk, pi, r, c, x, y, p) in enumerate(nodes):
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if (nr, nc) not in spatial:
                    continue
                for nidx in spatial[(nr, nc)]:
                    nx, ny = nodes[nidx][4], nodes[nidx][5]
                    if np.sqrt((x - nx)**2 + (y - ny)**2) <= LINK_TOLERANCE:
                        union(idx, nidx)

    # Group by component
    components = defaultdict(list)
    for idx in range(len(nodes)):
        components[find(idx)].append(idx)

    features = []
    for comp_idx, member_indices in components.items():
        members = [nodes[i] for i in member_indices]
        features.append(members)

    return features


# ── Phase 3: Characterize and filter features ──────────────────────

def check_gaussian_profile(members):
    """Check if intensity profile across bins is Gaussian-like.

    Returns (is_gaussian, reason).

    A Gaussian-like profile means:
    - There's a clear center (highest intensity bin)
    - Intensity generally decreases with distance from center
    - Not all bins have the same intensity (which would indicate artifact)
    """
    if len(members) < 2:
        return False, "isolated: single-bin detection"

    intensities = [m[6]['cleaned_intensity'] for m in members]
    positions = [(m[2], m[3]) for m in members]  # (row, col) in grid

    imax = np.argmax(intensities)
    center_r, center_c = positions[imax]
    peak_intensity = intensities[imax]

    if peak_intensity <= 0:
        return False, "non-positive peak intensity"

    # Check intensity variation — if all similar, suspicious
    i_arr = np.array(intensities)
    cv = np.std(i_arr) / np.mean(i_arr) if np.mean(i_arr) > 0 else 0
    if cv < 0.05:
        return False, f"flat profile: CV={cv:.3f} (no clear center)"

    # For each member, compute distance from center and check trend
    distances = [np.sqrt((r - center_r)**2 + (c - center_c)**2) for r, c in positions]

    # Bin by distance and check monotonic decrease
    n_closer_brighter = 0
    n_farther_dimmer = 0
    n_comparisons = 0
    for i in range(len(members)):
        for j in range(i + 1, len(members)):
            if abs(distances[i] - distances[j]) < 0.1:
                continue
            n_comparisons += 1
            if distances[i] < distances[j] and intensities[i] > intensities[j]:
                n_closer_brighter += 1
                n_farther_dimmer += 1
            elif distances[i] > distances[j] and intensities[i] < intensities[j]:
                n_closer_brighter += 1
                n_farther_dimmer += 1

    if n_comparisons == 0:
        return True, "small cluster, no directional check possible"

    monotonic_fraction = n_closer_brighter / n_comparisons
    if monotonic_fraction < 0.4:
        return False, f"non-Gaussian: only {monotonic_fraction:.0%} of pairs follow distance-intensity trend"

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
                entry["chi"] = round(float(np.degrees(
                    np.arctan2(py - by, px - bx))), 2)
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


def characterize_features(features, beam_center=None, tth_map=None,
                          ref_tth_map=None):
    """Classify each feature as kept or filtered, with reason."""
    kept = []
    filtered = []

    for members in features:
        bins_in_feature = set()
        for bk, pi, r, c, x, y, p in members:
            bins_in_feature.add(bk)

        is_gaussian, reason = check_gaussian_profile(members)

        # Compute feature summary
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

        if is_gaussian:
            kept.append(feature_info)
        else:
            filtered.append(feature_info)

    return kept, filtered


# ── Phase 4: Output ────────────────────────────────────────────────

def write_feature_catalog(kept, output_path):
    for i, f in enumerate(kept):
        f["feature_id"] = i + 1
    with open(output_path, "w") as fh:
        json.dump(kept, fh, indent=2)
    print(f"  Wrote {len(kept)} kept features to {output_path}")


def write_peak_table(peaks, output_path, title):
    with open(output_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "feature_id", "reflection", "bin_key", "center_row", "center_col",
            "detector_x", "detector_y", "peak_intensity", "mean_snr",
            "n_bins", "spatial_extent", "reason"
        ])
        for i, f in enumerate(peaks):
            writer.writerow([
                f.get("feature_id", i + 1),
                f["reflection"],
                f["center_bin"],
                f["center_row"],
                f["center_col"],
                f["detector_x"],
                f["detector_y"],
                f"{f['peak_intensity']:.1f}",
                f"{f['mean_snr']:.1f}",
                f["n_bins"],
                " ".join(f["spatial_extent"]),
                f["reason"],
            ])
    print(f"  Wrote {len(peaks)} {title} to {output_path}")


def main():
    print("=" * 60)
    print("  Spatial Feature Analysis — 3x3 bins")
    print("=" * 60)

    # Load shared data
    print("\nLoading detector module and shared data...", flush=True)
    det = load_module(DETECTOR_PATH)
    tth_map = __import__("tifffile").imread(str(HOLDOUT_DIR / "tth.tiff"))
    ref = load_module(HOLDOUT_DIR / "reflections.py")
    degs, deg_labels = ref.degs, ref.deg_labels
    print(f"  Reflections: {deg_labels}", flush=True)

    with open(HOLDOUT_DIR / "grid_mapping.json") as f:
        gm = json.load(f)
    n_rows, n_cols = gm["n_bin_rows"], gm["n_bin_cols"]
    print(f"  Grid: {n_rows} x {n_cols} = {n_rows * n_cols} bins")

    # Phase 1: Detect
    print("\n--- Phase 1: Per-bin detection ---", flush=True)
    all_detections = run_detection_all_bins(BINS_H5, tth_map, degs, deg_labels, det)

    # Phase 2: Link
    print("\n--- Phase 2: Linking peaks across bins ---", flush=True)
    features = link_peaks(all_detections, n_rows, n_cols)
    n_single = sum(1 for f in features if len(set(m[0] for m in f)) == 1)
    n_multi = len(features) - n_single
    print(f"  Found {len(features)} raw features: {n_multi} multi-bin, {n_single} single-bin")

    # Estimate beam center from tth map for azimuthal angle (chi)
    beam_center = estimate_beam_center(tth_map)
    print(f"  Beam center (y, x): ({beam_center[0]:.1f}, {beam_center[1]:.1f})")

    # Build reference 2-theta lookup: label → degrees
    ref_tth_map = {}
    for d, lbl in zip(degs, deg_labels):
        ref_tth_map[lbl] = round(d, 5)

    # Phase 3: Filter
    print("\n--- Phase 3: Gaussian profile filtering ---", flush=True)
    kept, filtered = characterize_features(
        features, beam_center=beam_center, tth_map=tth_map,
        ref_tth_map=ref_tth_map)

    # Sort by raster scan path (row-major: top-to-bottom, left-to-right)
    kept.sort(key=lambda f: (f["center_row"], f["center_col"]))
    filtered.sort(key=lambda f: (f["center_row"], f["center_col"]))

    print(f"  Kept:     {len(kept)} features (Gaussian-like spatial profile)")
    print(f"  Filtered: {len(filtered)} features")

    # Breakdown of filtered reasons
    reason_counts = defaultdict(int)
    for f in filtered:
        key = f["reason"].split(":")[0]
        reason_counts[key] += 1
    for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
        print(f"    {reason}: {count}")

    # Phase 4: Output
    print("\n--- Phase 4: Writing output ---", flush=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    write_feature_catalog(kept, OUTPUT_DIR / "feature_catalog_3x3.json")
    write_peak_table(filtered, OUTPUT_DIR / "filtered_peaks_3x3.csv", "filtered peaks")
    write_peak_table(kept, OUTPUT_DIR / "kept_peaks_3x3.csv", "kept peaks")

    # Summary stats
    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    per_ref_kept = defaultdict(int)
    per_ref_filt = defaultdict(int)
    for f in kept:
        per_ref_kept[f["reflection"]] += 1
    for f in filtered:
        per_ref_filt[f["reflection"]] += 1

    all_refs = sorted(set(list(per_ref_kept.keys()) + list(per_ref_filt.keys())))
    print(f"\n  {'Reflection':>12}  {'Kept':>6}  {'Filtered':>8}  {'Total':>6}")
    print(f"  {'-'*12}  {'-'*6}  {'-'*8}  {'-'*6}")
    for ref in all_refs:
        k = per_ref_kept.get(ref, 0)
        f = per_ref_filt.get(ref, 0)
        print(f"  {ref:>12}  {k:>6}  {f:>8}  {k+f:>6}")
    total_k = len(kept)
    total_f = len(filtered)
    print(f"  {'TOTAL':>12}  {total_k:>6}  {total_f:>8}  {total_k+total_f:>6}")

    print(f"\n  Output files:")
    print(f"    {OUTPUT_DIR / 'feature_catalog_3x3.json'}")
    print(f"    {OUTPUT_DIR / 'kept_peaks_3x3.csv'}")
    print(f"    {OUTPUT_DIR / 'filtered_peaks_3x3.csv'}")
    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
