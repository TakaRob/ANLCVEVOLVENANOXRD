"""
Baseline: per-frame Bragg peak detection + spatial linking + Voigt profile filtering.

Full pipeline for 1x1 (un-binned) XRD data:
1. Detect candidate peaks in each single-exposure frame
2. Link detections at matching detector positions across neighboring spatial bins
3. Fit Voigt profile to intensity vs. spatial distance; keep features that fit well
4. Output spatially-validated peaks only

Usage:
    python baseline.py \
        --center-bin 30_50 \
        --bins-h5 /home/takaji/xrd_1x1_bins.h5 \
        --two-theta tth.tiff \
        --reflections reflections.py \
        --grid-mapping grid_mapping.json \
        --output detections.csv \
        --spatial-radius 5
"""

import argparse
import csv
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import scipy.ndimage as ndi


BINS_H5_DEFAULT = "/home/takaji/xrd_1x1_bins.h5"


# ── Stage 1: Per-frame peak detection ──────────────────────────────

def precompute_tth(tth_map, bin_width=0.05):
    tth_min, tth_max = float(tth_map.min()), float(tth_map.max())
    edges = np.arange(tth_min, tth_max + bin_width, bin_width)
    centers = 0.5 * (edges[:-1] + edges[1:])
    n_bins = len(centers)
    flat = tth_map.ravel()
    indices = np.digitize(flat, edges) - 1
    indices = np.clip(indices, 0, n_bins - 1).astype(np.int32)
    counts = np.bincount(indices, minlength=n_bins)[:n_bins]
    order = np.argsort(indices)
    sorted_idx = indices[order]
    boundaries = np.searchsorted(sorted_idx, np.arange(n_bins + 1))
    return {
        'edges': edges, 'centers': centers, 'n_bins': n_bins,
        'indices': indices, 'counts': counts, 'order': order,
        'boundaries': boundaries, 'valid_mask': counts > 50
    }


def radial_median_subtract(image, tth_data):
    img_flat = image.ravel()
    sorted_vals = img_flat[tth_data['order']]
    n = tth_data['n_bins']
    boundaries = tth_data['boundaries']
    median_profile = np.zeros(n)
    for i in range(n):
        lo, hi = int(boundaries[i]), int(boundaries[i + 1])
        if hi > lo:
            median_profile[i] = np.median(sorted_vals[lo:hi])
    if n > 5:
        median_profile_smooth = ndi.uniform_filter1d(median_profile, size=min(15, n))
    else:
        median_profile_smooth = median_profile.copy()
    bg = median_profile_smooth[tth_data['indices']].reshape(image.shape)
    return image - bg


def fast_tophat(image, size=11):
    eroded = ndi.minimum_filter(image, size=size)
    opened = ndi.maximum_filter(eroded, size=size)
    return image - opened


def build_tth_band_masks(tth_map, degs, deg_labels, tth_tolerance=0.5):
    bands = {}
    for label, deg_val in zip(deg_labels, degs):
        mask = np.abs(tth_map - deg_val) <= tth_tolerance
        if label in bands:
            bands[label] = bands[label] | mask
        else:
            bands[label] = mask
    return bands


def detect_in_band(tophat_image, cleaned_image, band_mask, label,
                   snr_threshold=3.0, min_pixels=2, max_pixels=200,
                   min_compactness=0.10, ignore_edge=3):
    rows, cols = tophat_image.shape
    edge_mask = np.ones((rows, cols), dtype=bool)
    if ignore_edge > 0:
        edge_mask[:ignore_edge, :] = False
        edge_mask[-ignore_edge:, :] = False
        edge_mask[:, :ignore_edge] = False
        edge_mask[:, -ignore_edge:] = False
    valid_mask = band_mask & edge_mask
    band_vals = tophat_image[valid_mask]
    if len(band_vals) < 100:
        return []
    med = np.median(band_vals)
    mad = np.median(np.abs(band_vals - med))
    sigma_est = mad * 1.4826 if mad > 0 else np.std(band_vals)
    if sigma_est <= 0:
        return []
    threshold = med + snr_threshold * sigma_est
    hotspot = np.zeros((rows, cols), dtype=bool)
    hotspot[valid_mask] = tophat_image[valid_mask] > threshold
    cc, n_comp = ndi.label(hotspot)
    if n_comp == 0:
        return []
    # Gather all labeled pixels once and group them by component via a single
    # sort, instead of scanning the whole image with np.where(cc == comp_id)
    # for every component (which is O(image x n_components) and was the dominant
    # cost — hundreds of components per band under aggressive thresholds).
    flat_idx = np.flatnonzero(cc)
    comp_of_pixel = cc.ravel()[flat_idx]
    ys_flat = flat_idx // cols
    xs_flat = flat_idx % cols
    sort_order = np.argsort(comp_of_pixel, kind="stable")
    ys_flat = ys_flat[sort_order]
    xs_flat = xs_flat[sort_order]
    comp_sorted = comp_of_pixel[sort_order]
    seg_starts = np.searchsorted(comp_sorted, np.arange(1, n_comp + 2))
    peaks = []
    for comp_id in range(1, n_comp + 1):
        lo = seg_starts[comp_id - 1]
        hi = seg_starts[comp_id]
        npix = hi - lo
        if npix < min_pixels or npix > max_pixels:
            continue
        ys = ys_flat[lo:hi]
        xs = xs_flat[lo:hi]
        h = ys.max() - ys.min() + 1
        w = xs.max() - xs.min() + 1
        if h == 0 or w == 0:
            continue
        aspect = min(h, w) / max(h, w)
        fill = npix / (h * w)
        compactness = aspect * fill
        if compactness < min_compactness:
            continue
        weights = np.maximum(cleaned_image[ys, xs], 0)
        weight_sum = weights.sum()
        if weight_sum > 0:
            cx = int(round(np.sum(xs * weights) / weight_sum))
            cy = int(round(np.sum(ys * weights) / weight_sum))
        else:
            cx = int(np.mean(xs))
            cy = int(np.mean(ys))
        peak_val = np.max(tophat_image[ys, xs])
        snr = (peak_val - med) / sigma_est if sigma_est > 0 else 0
        peaks.append({
            'x': cx, 'y': cy, 'label': label,
            'npix': npix, 'compactness': compactness,
            'snr': snr, 'peak_val': peak_val,
            'intensity': float(weight_sum),
        })
    return peaks


def detect_peaks_single_frame(image, tth_map, degs, deg_labels, tth_data,
                              tophat_size=13, tth_tolerance=0.5,
                              snr_threshold=3.0, max_detections=30,
                              bands=None):
    cleaned = radial_median_subtract(image, tth_data)
    tophat = fast_tophat(cleaned, size=tophat_size)
    # Band masks depend only on the (fixed) tth_map, so they are computed once
    # in run_full_pipeline and reused for every frame. Fall back to building
    # them here if a caller does not supply them.
    if bands is None:
        bands = build_tth_band_masks(tth_map, degs, deg_labels,
                                     tth_tolerance=tth_tolerance)
    all_peaks = []
    for label, band_mask in bands.items():
        peaks = detect_in_band(tophat, cleaned, band_mask, label,
                               snr_threshold=snr_threshold)
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
    if len(kept) > max_detections:
        kept = kept[:max_detections]
    for p in kept:
        r = 3
        y0 = max(0, p['y'] - r)
        y1 = min(cleaned.shape[0], p['y'] + r + 1)
        x0 = max(0, p['x'] - r)
        x1 = min(cleaned.shape[1], p['x'] + r + 1)
        p['cleaned_intensity'] = float(np.max(cleaned[y0:y1, x0:x1]))
    return kept, cleaned


# ── Stage 2: Spatial linking across frames ────────────────────────

def get_neighbor_bins(center_row, center_col, n_rows, n_cols, radius=5):
    neighbors = []
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            r, c = center_row + dr, center_col + dc
            if 0 <= r < n_rows and 0 <= c < n_cols:
                neighbors.append((r, c))
    return neighbors


def link_peaks_spatial(all_detections, link_tolerance=5):
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
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
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

    features = []
    for member_indices in components.values():
        members = [nodes[i] for i in member_indices]
        features.append(members)
    return features


# ── Stage 3: Voigt profile shape verification ─────────────────────

def voigt_profile(r, amplitude, sigma, gamma):
    """Approximate Voigt profile using the pseudo-Voigt approximation.

    V(r) ≈ amplitude * [eta * L(r, gamma) + (1 - eta) * G(r, sigma)]

    where eta is a mixing parameter that depends on the relative widths.
    This captures both Gaussian (strain broadening) and Lorentzian (size
    broadening) contributions to the spatial intensity profile.
    """
    fg = 2.0 * sigma * np.sqrt(2.0 * np.log(2.0))
    fl = 2.0 * gamma
    f = (fg**5 + 2.69269 * fg**4 * fl + 2.42843 * fg**3 * fl**2
         + 4.47163 * fg**2 * fl**3 + 0.07842 * fg * fl**4 + fl**5) ** 0.2
    if f < 1e-10:
        return np.zeros_like(r)
    eta = 1.36603 * (fl / f) - 0.47719 * (fl / f)**2 + 0.11116 * (fl / f)**3
    eta = np.clip(eta, 0, 1)
    gaussian = np.exp(-r**2 / (2.0 * sigma**2 + 1e-10))
    lorentzian = 1.0 / (1.0 + (r / (gamma + 1e-10))**2)
    return amplitude * (eta * lorentzian + (1 - eta) * gaussian)


def fit_voigt_to_feature(members):
    """Fit a Voigt profile to the spatial intensity distribution of a feature.

    Returns (is_good_fit, fit_params, reason).
    """
    if len(members) < 3:
        if len(members) == 2:
            return True, {}, "small-cluster: 2 bins, accepted"
        return False, {}, "isolated: single-bin detection"

    intensities = np.array([m[6]['cleaned_intensity'] for m in members])
    positions = np.array([(m[2], m[3]) for m in members], dtype=float)

    if np.max(intensities) <= 0:
        return False, {}, "non-positive peak intensity"

    imax = np.argmax(intensities)
    center = positions[imax]
    distances = np.sqrt(np.sum((positions - center)**2, axis=1))

    norm_intensities = intensities / (np.max(intensities) + 1e-10)

    cv = np.std(norm_intensities) / (np.mean(norm_intensities) + 1e-10)
    if cv < 0.05:
        return False, {}, f"flat profile: CV={cv:.3f}"

    from scipy.optimize import curve_fit

    def voigt_1d(r, amp, sigma, gamma):
        return voigt_profile(r, amp, sigma, gamma)

    try:
        r_max = np.max(distances) + 1
        p0 = [1.0, r_max / 3.0, r_max / 3.0]
        bounds = ([0.5, 0.1, 0.01], [1.5, r_max * 2, r_max * 2])
        popt, pcov = curve_fit(voigt_1d, distances, norm_intensities,
                               p0=p0, bounds=bounds, maxfev=500)
        fitted = voigt_1d(distances, *popt)
        residuals = norm_intensities - fitted
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((norm_intensities - np.mean(norm_intensities))**2)
        r_squared = 1 - ss_res / (ss_tot + 1e-10)

        params = {
            'amplitude': float(popt[0]),
            'sigma': float(popt[1]),
            'gamma': float(popt[2]),
            'r_squared': float(r_squared),
            'n_bins': len(members),
        }

        if r_squared > 0.3:
            return True, params, f"Voigt fit: R²={r_squared:.3f}, σ={popt[1]:.2f}, γ={popt[2]:.2f}"
        else:
            return False, params, f"poor Voigt fit: R²={r_squared:.3f}"
    except (RuntimeError, ValueError) as e:
        n_closer = 0
        n_total = 0
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                if abs(distances[i] - distances[j]) < 0.1:
                    continue
                n_total += 1
                if (distances[i] < distances[j]) == (intensities[i] > intensities[j]):
                    n_closer += 1
        if n_total > 0 and n_closer / n_total > 0.5:
            return True, {}, f"monotonic falloff: {n_closer}/{n_total} pairs"
        return False, {}, f"Voigt fit failed, no monotonic trend"


# ── Stage 4: Full pipeline ────────────────────────────────────────

def block_reduce_mean(arr, factor):
    """Down-sample a 2D array by averaging non-overlapping factor x factor blocks."""
    if factor <= 1:
        return arr
    h, w = arr.shape
    h2, w2 = (h // factor) * factor, (w // factor) * factor
    arr = arr[:h2, :w2]
    return arr.reshape(h2 // factor, factor, w2 // factor, factor).mean(axis=(1, 3))


def run_full_pipeline(center_bin, bins_h5_path, tth_map, degs, deg_labels,
                      grid_mapping, spatial_radius=5, link_tolerance=5,
                      min_feature_bins=2, downsample=1):
    """Run detection + linking + Voigt filtering around a center bin.

    `downsample` > 1 is a DEVELOPMENT-MODE speed knob: each frame and the tth
    map are block-averaged by that factor before detection (e.g. 2 -> ~4x fewer
    pixels, ~4x faster filters). Detector pixel parameters are scaled to the
    reduced grid and output peak coordinates are rescaled back to full
    resolution so the 40-pixel match tolerance still applies in original space.
    Use downsample=1 for final scoring.
    """
    downsample = max(1, int(downsample))
    if downsample > 1:
        tth_map = block_reduce_mean(tth_map, downsample)
    tth_data = precompute_tth(tth_map)
    # Detector-pixel parameters scaled to the (possibly) reduced grid.
    tophat_size = max(3, int(round(13 / downsample)))
    eff_link_tol = max(1.0, link_tolerance / downsample)
    # Band masks depend only on tth_map — build once, reuse for every frame.
    bands = build_tth_band_masks(tth_map, degs, deg_labels, tth_tolerance=0.5)

    center_row, center_col = int(center_bin.split("_")[0]), int(center_bin.split("_")[1])
    n_rows = grid_mapping['n_bin_rows']
    n_cols = grid_mapping['n_bin_cols']

    neighbor_positions = get_neighbor_bins(center_row, center_col,
                                           n_rows, n_cols, radius=spatial_radius)
    neighbor_keys = [f"{r}_{c}" for r, c in neighbor_positions]

    all_detections = {}
    with h5py.File(bins_h5_path, "r") as h5f:
        for bk in neighbor_keys:
            if bk not in h5f:
                continue
            image = np.clip(h5f[bk][:].astype(np.float64), 0, 1e9)
            if downsample > 1:
                image = block_reduce_mean(image, downsample)
            peaks, _ = detect_peaks_single_frame(
                image, tth_map, degs, deg_labels, tth_data,
                tophat_size=tophat_size, bands=bands)
            if peaks:
                all_detections[bk] = peaks

    features = link_peaks_spatial(all_detections, link_tolerance=eff_link_tol)

    validated_peaks = {}
    for members in features:
        bins_in_feature = set(m[0] for m in members)
        if center_bin not in bins_in_feature:
            continue
        if len(bins_in_feature) < min_feature_bins:
            continue
        is_good, params, reason = fit_voigt_to_feature(members)
        if not is_good:
            continue
        center_members = [m for m in members if m[0] == center_bin]
        if not center_members:
            continue
        best = max(center_members, key=lambda m: m[6]['snr'])
        label = best[6]['label']
        # Rescale detector coordinates back to full resolution when downsampled
        # (block center offset keeps the mapping unbiased).
        bx, by = best[4], best[5]
        if downsample > 1:
            bx = bx * downsample + downsample // 2
            by = by * downsample + downsample // 2
        validated_peaks.setdefault(label, []).append([bx, by])

    return validated_peaks


# ── Evaluation ────────────────────────────────────────────────────

def compute_f1(detections, labels, tolerance=40):
    det_pts = [p for pts in detections.values() for p in pts]
    gt_pts = [p for pts in labels.values() for p in pts]
    if not gt_pts and not det_pts:
        return 1.0
    if not gt_pts or not det_pts:
        return 0.0
    det_arr = np.array(det_pts, dtype=float)
    gt_arr = np.array(gt_pts, dtype=float)
    matched_gt = set()
    for dp in det_arr:
        dists = np.sqrt(np.sum((gt_arr - dp) ** 2, axis=1))
        nearest = int(np.argmin(dists))
        if dists[nearest] <= tolerance and nearest not in matched_gt:
            matched_gt.add(nearest)
    tp = len(matched_gt)
    fp = len(det_pts) - tp
    fn = len(gt_pts) - tp
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def main():
    parser = argparse.ArgumentParser(
        description="Baseline: per-frame detection + spatial linking + Voigt filtering")
    parser.add_argument("--center-bin", required=True, help="Center bin key, e.g. '30_50'")
    parser.add_argument("--bins-h5", default=BINS_H5_DEFAULT)
    parser.add_argument("--two-theta", default="tth.tiff")
    parser.add_argument("--reflections", default="reflections.py")
    parser.add_argument("--grid-mapping", default="grid_mapping.json")
    parser.add_argument("--output", default="detections.csv")
    parser.add_argument("--labels", default=None)
    parser.add_argument("--spatial-radius", type=int, default=5)
    parser.add_argument("--link-tolerance", type=float, default=5.0)
    parser.add_argument("--min-feature-bins", type=int, default=2)
    parser.add_argument("--downsample", type=int, default=1,
                        help="Dev-mode pixel downsample factor (1 = full resolution)")
    args = parser.parse_args()

    spec = importlib.util.spec_from_file_location("reflections", args.reflections)
    ref_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref_mod)
    degs = ref_mod.degs
    deg_labels = ref_mod.deg_labels

    import tifffile
    tth_map = tifffile.imread(args.two_theta)

    with open(args.grid_mapping) as f:
        grid_mapping = json.load(f)

    validated = run_full_pipeline(
        args.center_bin, args.bins_h5, tth_map, degs, deg_labels,
        grid_mapping,
        spatial_radius=args.spatial_radius,
        link_tolerance=args.link_tolerance,
        min_feature_bins=args.min_feature_bins,
        downsample=args.downsample,
    )

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["reflection", "x", "y"])
        for label, pts in validated.items():
            for x, y in pts:
                writer.writerow([label, x, y])

    total = sum(len(pts) for pts in validated.values())
    print(f"Detected {total} spatially-validated peaks at center bin {args.center_bin}")

    if args.labels:
        with open(args.labels) as f:
            labels = json.load(f)
        labels = {k: v for k, v in labels.items() if not k.startswith("__")}
        f1 = compute_f1(validated, labels)
        print(f"F1 score: {f1:.4f}")


if __name__ == "__main__":
    main()
