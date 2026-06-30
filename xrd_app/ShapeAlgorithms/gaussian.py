"""
Baseline shape-finding algorithm for binned nano-XRD peak sets.

This is the Phase-2 (cross-bin) stage that runs *after* per-bin peak detection
('xrd-app peaks'). Given the per-bin peaks, it:

  Phase 2  spatial linking    Union-Find: peaks within ``link_tolerance`` px at
                              the same detector position in neighboring spatial
                              bins are merged into one feature.
  Phase 3  gaussian filtering keep clusters whose cross-bin intensity profile
                              has a clear bright center (a real Bragg "shape"),
                              drop flat / non-monotonic clusters.

A "shape" is a kept (gaussian-like) feature. This file is self-contained and
swappable — drop a new ``ShapeAlgorithms/<name>.py`` exposing ``link_peaks`` and
``characterize_features`` (and register it in ``catalog.json``) to evolve the
shape stage, exactly like the detectors in ``PeakAlgorithms/``.

Usage (standalone)
------------------
    python gaussian.py \
        --peaks Labels/Scan_0203/baseline_peaks_3x3.json \
        --two-theta tth.tiff \
        --grid-mapping Metadata/Scan_0203/grid_mapping_3x3.json \
        --reflections reflections.py \
        --output shapes_catalog_3x3.json
"""

import argparse
import importlib.util
import json
from collections import defaultdict

import numpy as np
import tifffile


DEFAULT_LINK_TOLERANCE = 5  # pixels — max distance to consider same peak across bins


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
            feature_info["chi_fwhm"] = round(2.3548 * np.sqrt(var), 4)

        if len(bin_tths) >= 3 and len(bin_weights) == len(bin_tths):
            ref_val = ref_tth_map.get(reflection) if ref_tth_map else None
            if ref_val is not None:
                wa = np.array(bin_weights)
                ta = np.array(bin_tths) - ref_val
                wn = wa / wa.sum()
                mu = np.dot(wn, ta)
                var = np.dot(wn, (ta - mu) ** 2)
                # Radial breadth: FWHM of Δ2θ across the feature's bins. (Was
                # named "strain_breadth"; renamed for honesty — it is not a
                # calibrated strain. See TERMINOLOGY.md §3.3.)
                feature_info["tth_fwhm"] = round(2.3548 * np.sqrt(var), 4)

        if beam_center is not None:
            by, bx = beam_center
            chi = float(np.degrees(np.arctan2(det_y - by, det_x - bx)))
            feature_info["chi_deg"] = round(chi, 1)

        (kept if is_gaussian else filtered).append(feature_info)

    return kept, filtered


# ── Convenience entry point ────────────────────────────────────────
def run_shape_pipeline(peaks_by_bin, n_rows, n_cols, tth_map, degs, deg_labels,
                       link_tolerance=DEFAULT_LINK_TOLERANCE):
    """Link peaks then keep gaussian-like shapes. Returns (kept, filtered).

    ``peaks_by_bin`` maps ``"<row>_<col>"`` to a list of peak dicts (each with
    ``x``, ``y``, ``snr``, ``label``, ``cleaned_intensity`` — the output of the
    peak stage). ``kept``/``filtered`` are sorted by (center_row, center_col)
    and ``kept`` features carry a 1-based ``feature_id``.
    """
    features = link_peaks(peaks_by_bin, n_rows, n_cols, link_tolerance)
    beam_center = estimate_beam_center(tth_map)
    ref_tth_map = {lbl: round(d, 5) for d, lbl in zip(degs, deg_labels)}
    kept, filtered = characterize_features(
        features, beam_center=beam_center, tth_map=tth_map, ref_tth_map=ref_tth_map)
    kept.sort(key=lambda f: (f["center_row"], f["center_col"]))
    filtered.sort(key=lambda f: (f["center_row"], f["center_col"]))
    for i, f in enumerate(kept):
        f["feature_id"] = i + 1
    return kept, filtered


# ── Serialization helper (numpy → JSON-native) ─────────────────────
def _coerce(obj):
    if isinstance(obj, dict):
        return {k: _coerce(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_coerce(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return _coerce(obj.tolist())
    return obj


# ── Standalone CLI ─────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Baseline shape finder (link + gaussian filter)")
    parser.add_argument("--peaks", required=True, help="Saved *_peaks.json from 'xrd-app peaks'")
    parser.add_argument("--two-theta", dest="two_theta", default="tth.tiff")
    parser.add_argument("--grid-mapping", required=True, help="grid_mapping_*.json")
    parser.add_argument("--reflections", default="reflections.py")
    parser.add_argument("--output", default="shapes_catalog.json")
    parser.add_argument("--link-tolerance", type=int, default=DEFAULT_LINK_TOLERANCE)
    args = parser.parse_args()

    with open(args.peaks) as f:
        peaks_data = json.load(f)
    peaks_by_bin = peaks_data.get("peaks_by_bin", peaks_data)

    with open(args.grid_mapping) as f:
        gm = json.load(f)
    n_rows, n_cols = gm["n_bin_rows"], gm["n_bin_cols"]

    spec = importlib.util.spec_from_file_location("reflections", args.reflections)
    ref_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref_mod)
    degs, deg_labels = ref_mod.degs, ref_mod.deg_labels

    tth_map = tifffile.imread(args.two_theta).astype(np.float64)

    kept, filtered = run_shape_pipeline(
        peaks_by_bin, n_rows, n_cols, tth_map, degs, deg_labels,
        link_tolerance=args.link_tolerance)

    with open(args.output, "w") as f:
        json.dump(_coerce(kept), f, indent=2)

    print(f"Kept {len(kept)} shapes, filtered {len(filtered)} -> {args.output}")


if __name__ == "__main__":
    main()
