"""
2-theta Band + Fast Top-Hat + Adaptive Thresholding Bragg Peak Detector

Multi-stage detector for Bragg peaks in spatially-binned XRD images.

Pipeline:
1. Robust non-parametric radial median background subtraction
2. Fast white top-hat via min-max filters to extract compact bright features
3. 2-theta band masking: restrict search to narrow bands around known reflections
4. Per-band adaptive thresholding using MAD-based statistics
5. Connected component analysis with size + compactness filtering
6. Cross-band duplicate suppression

Usage:
    python detect.py \
        --bin-key 3_7 \
        --bins-h5 /home/takaji/xrd_5x5_bins.h5 \
        --two-theta tth.tiff \
        --reflections reflections.py \
        --output detections.csv
"""

import argparse
import csv
import json
import importlib.util

import h5py
import numpy as np
import scipy.ndimage as ndi
import tifffile


BINS_H5_DEFAULT = "/home/takaji/xrd_5x5_bins.h5"


# ── Radial background subtraction ───────────────────────────────────

def precompute_tth(tth_map, bin_width=0.05):
    """Pre-compute 2-theta binning data for fast radial operations."""
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
    """Non-parametric radial background subtraction using median profile."""
    img_flat = image.ravel()
    sorted_vals = img_flat[tth_data['order']]
    n = tth_data['n_bins']
    boundaries = tth_data['boundaries']

    median_profile = np.zeros(n)
    for i in range(n):
        lo, hi = int(boundaries[i]), int(boundaries[i + 1])
        if hi > lo:
            median_profile[i] = np.median(sorted_vals[lo:hi])

    # Smooth with uniform filter to avoid subtracting peak signal
    if n > 5:
        median_profile_smooth = ndi.uniform_filter1d(median_profile, size=min(15, n))
    else:
        median_profile_smooth = median_profile.copy()

    bg = median_profile_smooth[tth_data['indices']].reshape(image.shape)
    return image - bg


# ── Fast morphological top-hat ───────────────────────────────────────

def fast_tophat(image, size=11):
    """Fast white top-hat using min/max filters with a square kernel.

    Top-hat = image - opening(image)
    opening = dilation(erosion(image)) = max_filter(min_filter(image))
    
    This is ~20x faster than grey_opening with a disk footprint.
    """
    eroded = ndi.minimum_filter(image, size=size)
    opened = ndi.maximum_filter(eroded, size=size)
    return image - opened


# ── 2-theta band masks ──────────────────────────────────────────────

def build_tth_band_masks(tth_map, degs, deg_labels, tth_tolerance=0.4):
    """Pre-compute boolean masks for each reflection's 2-theta band."""
    bands = {}
    for label, deg_val in zip(deg_labels, degs):
        mask = np.abs(tth_map - deg_val) <= tth_tolerance
        if label in bands:
            bands[label] = bands[label] | mask
        else:
            bands[label] = mask
    return bands


# ── Per-band adaptive detection ──────────────────────────────────────

def detect_in_band(tophat_image, cleaned_image, band_mask, label,
                   snr_threshold=3.5, min_pixels=3, max_pixels=150,
                   min_compactness=0.12, ignore_edge=3):
    """Detect peaks in a single 2-theta band using adaptive MAD threshold."""
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

    # Vectorized component analysis
    comp_sizes = np.bincount(cc.ravel(), minlength=n_comp + 1)

    peaks = []
    for comp_id in range(1, n_comp + 1):
        npix = comp_sizes[comp_id]
        if npix < min_pixels or npix > max_pixels:
            continue

        ys, xs = np.where(cc == comp_id)

        # Compactness: aspect_ratio * fill_ratio
        h = ys.max() - ys.min() + 1
        w = xs.max() - xs.min() + 1
        if h == 0 or w == 0:
            continue
        aspect = min(h, w) / max(h, w)
        fill = npix / (h * w)
        compactness = aspect * fill

        if compactness < min_compactness:
            continue

        # Intensity-weighted centroid using cleaned image
        weights = np.maximum(cleaned_image[ys, xs], 0)
        weight_sum = weights.sum()
        if weight_sum > 0:
            cx = int(round(np.sum(xs * weights) / weight_sum))
            cy = int(round(np.sum(ys * weights) / weight_sum))
        else:
            cx = int(np.mean(xs))
            cy = int(np.mean(ys))

        # Peak SNR from tophat band stats
        peak_val = np.max(tophat_image[ys, xs])
        snr = (peak_val - med) / sigma_est if sigma_est > 0 else 0

        peaks.append({
            'x': cx, 'y': cy, 'label': label,
            'npix': npix, 'compactness': compactness,
            'snr': snr, 'peak_val': peak_val
        })

    return peaks


# ── Full pipeline ────────────────────────────────────────────────────

def detect_peaks(image, tth_map, degs, deg_labels, tth_data,
                 tophat_size=15,
                 tth_tolerance=0.4,
                 snr_threshold=4.0,
                 min_pixels=3,
                 max_pixels=150,
                 min_compactness=0.12,
                 dup_distance=15,
                 ignore_edge=3,
                 max_detections=25):
    """Full multi-stage Bragg peak detection pipeline."""

    # Stage 1: Radial background subtraction
    cleaned = radial_median_subtract(image, tth_data)

    # Stage 2: Fast top-hat for compact feature extraction
    tophat = fast_tophat(cleaned, size=tophat_size)

    # Stage 3: Build 2-theta band masks
    bands = build_tth_band_masks(tth_map, degs, deg_labels,
                                  tth_tolerance=tth_tolerance)

    # Stage 4: Detect peaks in each band independently
    all_peaks = []
    for label, band_mask in bands.items():
        peaks = detect_in_band(
            tophat, cleaned, band_mask, label,
            snr_threshold=snr_threshold,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            min_compactness=min_compactness,
            ignore_edge=ignore_edge
        )
        all_peaks.extend(peaks)

    if not all_peaks:
        return {}

    # Stage 5: Duplicate suppression across bands (keep highest SNR)
    all_peaks.sort(key=lambda p: p['snr'], reverse=True)
    kept = []
    for peak in all_peaks:
        is_dup = False
        for existing in kept:
            dist = np.sqrt((peak['y'] - existing['y'])**2 +
                          (peak['x'] - existing['x'])**2)
            if dist < dup_distance:
                is_dup = True
                break
        if not is_dup:
            kept.append(peak)

    # Stage 6: Limit total detections
    if len(kept) > max_detections:
        kept = kept[:max_detections]

    # Convert to output format
    results = {}
    for peak in kept:
        label = peak['label']
        results.setdefault(label, []).append([peak['x'], peak['y']])

    return results


# ── Evaluation ───────────────────────────────────────────────────────

def compute_f1(detections, labels, tolerance=40):
    """Compute F1 score with pixel-distance matching tolerance."""
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


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="2-theta Band + Fast Top-Hat + Adaptive Threshold Bragg Peak Detector")
    parser.add_argument("--bin-key", required=True, help="Bin key, e.g. '3_7'")
    parser.add_argument("--bins-h5", default=BINS_H5_DEFAULT,
                        help="Pre-built bin images HDF5")
    parser.add_argument("--two-theta", dest="two_theta", default="tth.tiff")
    parser.add_argument("--reflections", default="reflections.py")
    parser.add_argument("--output", default="detections.csv")
    parser.add_argument("--labels", default=None,
                        help="Ground truth JSON for evaluation")
    # Tunable parameters
    parser.add_argument("--tophat-size", type=int, default=15)
    parser.add_argument("--tth-tolerance", type=float, default=0.4)
    parser.add_argument("--snr-threshold", type=float, default=4.0)
    parser.add_argument("--min-pixels", type=int, default=3)
    parser.add_argument("--max-pixels", type=int, default=150)
    parser.add_argument("--min-compactness", type=float, default=0.12)
    parser.add_argument("--dup-distance", type=float, default=15)
    parser.add_argument("--ignore-edge", type=int, default=3)
    parser.add_argument("--max-detections", type=int, default=25)
    args = parser.parse_args()

    # Load reflections
    spec = importlib.util.spec_from_file_location("reflections", args.reflections)
    ref_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref_mod)
    degs = ref_mod.degs
    deg_labels = ref_mod.deg_labels

    # Load data
    with h5py.File(args.bins_h5, "r") as f:
        if args.bin_key not in f:
            raise ValueError(f"Bin '{args.bin_key}' not found in {args.bins_h5}")
        image = f[args.bin_key][:].astype(np.float64)

    tth_map = tifffile.imread(args.two_theta)
    tth_data = precompute_tth(tth_map)

    detections = detect_peaks(
        image, tth_map, degs, deg_labels, tth_data,
        tophat_size=args.tophat_size,
        tth_tolerance=args.tth_tolerance,
        snr_threshold=args.snr_threshold,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        min_compactness=args.min_compactness,
        dup_distance=args.dup_distance,
        ignore_edge=args.ignore_edge,
        max_detections=args.max_detections
    )

    # Write output CSV
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["reflection", "x", "y"])
        for label, pts in detections.items():
            for x, y in pts:
                writer.writerow([label, x, y])

    total = sum(len(pts) for pts in detections.values())
    print(f"Detected {total} peaks in bin {args.bin_key}")

    if args.labels:
        with open(args.labels) as f:
            labels = json.load(f)
        f1 = compute_f1(detections, labels)
        print(f"F1 score: {f1:.4f}")


if __name__ == "__main__":
    main()
