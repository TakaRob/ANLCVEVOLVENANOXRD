"""
Tuned: Multi-Scale Top-Hat + Local-Contrast Rescue Bragg Peak Detector

Tuned version of r001-c670e9ab (tophat_band_adaptive_snr, F1=0.808).
Parent had very high precision (0.933) but low recall (0.672).

Key tuning changes (preserving core pipeline):
1. Multi-scale top-hat (sizes 9, 15, 21) with max-response fusion
   -> captures peaks at varying spatial scales (parent used only size=15)
2. Iterative sigma-clipped noise estimation for more robust per-band MAD
3. Lowered min_compactness 0.12->0.08, min_pixels 3->2, max_pixels 150->200
4. Added local annular SNR rescue: peaks below main threshold but with
   strong local contrast (tophat_peak >> annular_background on tophat image)
   are rescued. This specifically targets faint peaks missed by band-wide MAD.
5. max_detections 25->50, ignore_edge 3->5
6. dup_distance 15->12 for better separation of nearby peaks

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


BINS_H5_DEFAULT = "/home/takaji/xrd_4x4_bins.h5"


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


# ── Multi-scale fast morphological top-hat ───────────────────────────

def fast_tophat(image, size=11):
    """Fast white top-hat using min/max filters with a square kernel."""
    eroded = ndi.minimum_filter(image, size=size)
    opened = ndi.maximum_filter(eroded, size=size)
    return image - opened


def multi_scale_tophat(image, sizes=(9, 15, 21)):
    """Multi-scale top-hat: take max response across scales."""
    result = np.zeros_like(image)
    for s in sizes:
        th = fast_tophat(image, size=s)
        np.maximum(result, th, out=result)
    return result


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


# ── Local annular SNR computation ────────────────────────────────────

def compute_local_snr_batch(tophat_image, ys, xs, inner_r=4, outer_r=15):
    """Compute local SNR for multiple points on the tophat image.
    
    Local SNR = (peak_val - annular_median) / (annular_MAD * 1.4826)
    """
    H, W = tophat_image.shape
    local_snrs = np.zeros(len(ys))
    
    for i in range(len(ys)):
        y, x = ys[i], xs[i]
        y_lo = max(0, y - outer_r)
        y_hi = min(H, y + outer_r + 1)
        x_lo = max(0, x - outer_r)
        x_hi = min(W, x + outer_r + 1)

        patch = tophat_image[y_lo:y_hi, x_lo:x_hi]
        cy_local = y - y_lo
        cx_local = x - x_lo

        # Build annular mask
        yy, xx = np.mgrid[0:patch.shape[0], 0:patch.shape[1]]
        dist = np.sqrt((yy - cy_local)**2 + (xx - cx_local)**2)
        annular = (dist >= inner_r) & (dist <= outer_r)

        ring_vals = patch[annular]
        if len(ring_vals) < 10:
            local_snrs[i] = 0.0
            continue

        bg_med = np.median(ring_vals)
        bg_mad = np.median(np.abs(ring_vals - bg_med))
        bg_sigma = max(bg_mad * 1.4826, 0.01)

        peak_val = tophat_image[y, x]
        local_snrs[i] = (peak_val - bg_med) / bg_sigma
    
    return local_snrs


# ── Per-band adaptive detection with rescue ──────────────────────────

def detect_in_band(tophat_image, cleaned_image, band_mask, label,
                   snr_threshold=4.8, min_pixels=2, max_pixels=200,
                   min_compactness=0.08, ignore_edge=5,
                   rescue_local_snr=99.0, rescue_band_snr=99.0,
                   min_peak_value=6.0):
    """Detect peaks in a single 2-theta band using adaptive MAD threshold.
    
    Two-tier detection:
    1. Primary: band SNR >= snr_threshold (confident)
    2. Rescue: band SNR >= rescue_band_snr AND local annular SNR >= rescue_local_snr
       (catches faint peaks with strong local contrast)
    """
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

    # Iterative sigma-clipped MAD estimation
    vals = band_vals.copy()
    for _ in range(3):
        med = np.median(vals)
        mad = np.median(np.abs(vals - med))
        sigma_est = mad * 1.4826 if mad > 0 else np.std(vals)
        if sigma_est <= 0:
            break
        clip_mask = vals < med + 5 * sigma_est
        if clip_mask.sum() < 50:
            break
        vals = vals[clip_mask]

    med = np.median(vals)
    mad = np.median(np.abs(vals - med))
    sigma_est = mad * 1.4826 if mad > 0 else np.std(vals)

    if sigma_est <= 0:
        return []

    # Use lower threshold for initial CC labeling to find all candidate regions
    effective_low = min(snr_threshold, rescue_band_snr)
    candidate_threshold = med + effective_low * sigma_est
    primary_threshold = med + snr_threshold * sigma_est

    hotspot = np.zeros((rows, cols), dtype=bool)
    hotspot[valid_mask] = tophat_image[valid_mask] > candidate_threshold

    cc, n_comp = ndi.label(hotspot)
    if n_comp == 0:
        return []

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

        # Peak value and band SNR
        peak_val = np.max(tophat_image[ys, xs])
        band_snr = (peak_val - med) / sigma_est if sigma_est > 0 else 0

        # Minimum absolute peak value to filter noise in very faint bins
        if peak_val < min_peak_value:
            continue

        # Check if peak passes primary threshold
        is_primary = peak_val > primary_threshold
        
        # If not primary, check rescue criteria
        is_rescue = False
        if not is_primary and band_snr >= rescue_band_snr:
            # Need to compute local SNR on tophat
            peak_idx = np.argmax(tophat_image[ys, xs])
            py, px = ys[peak_idx], xs[peak_idx]
            local_snrs = compute_local_snr_batch(tophat_image, [py], [px])
            if local_snrs[0] >= rescue_local_snr:
                is_rescue = True

        if not is_primary and not is_rescue:
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

        peaks.append({
            'x': cx, 'y': cy, 'label': label,
            'npix': npix, 'compactness': compactness,
            'snr': band_snr, 'peak_val': peak_val,
            'is_rescue': is_rescue,
        })

    return peaks


# ── Full pipeline ────────────────────────────────────────────────────

def detect_peaks(image, tth_map, degs, deg_labels, tth_data=None,
                 tophat_sizes=(9, 15, 21),
                 tth_tolerance=0.30,
                 snr_threshold=4.8,
                 rescue_band_snr=99.0,
                 rescue_local_snr=99.0,
                 min_pixels=2,
                 max_pixels=200,
                 min_compactness=0.08,
                 dup_distance=12,
                 ignore_edge=5,
                 max_detections=50,
                 min_peak_value=6.0):
    """Full multi-stage Bragg peak detection pipeline."""

    # Empty frame check
    if image.max() < 5.0:
        return {}

    if tth_data is None:
        tth_data = precompute_tth(tth_map)

    # Stage 1: Radial background subtraction
    cleaned = radial_median_subtract(image, tth_data)

    # Stage 2: Multi-scale fast top-hat for compact feature extraction
    tophat = multi_scale_tophat(cleaned, sizes=tophat_sizes)

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
            ignore_edge=ignore_edge,
            rescue_local_snr=rescue_local_snr,
            rescue_band_snr=rescue_band_snr,
            min_peak_value=min_peak_value,
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
        return 1.0, 0, 0, 0
    if not gt_pts:
        return 0.0, 0, len(det_pts), 0
    if not det_pts:
        return 0.0, 0, 0, len(gt_pts)

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
        return 0.0, tp, fp, fn
    return 2 * prec * rec / (prec + rec), tp, fp, fn


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Tuned: Multi-Scale Top-Hat + Local-Contrast Rescue Bragg Peak Detector")
    parser.add_argument("--bin-key", required=True, help="Bin key, e.g. '3_7'")
    parser.add_argument("--bins-h5", default=BINS_H5_DEFAULT,
                        help="Pre-built bin images HDF5")
    parser.add_argument("--two-theta", dest="two_theta", default="tth.tiff")
    parser.add_argument("--reflections", default="reflections.py")
    parser.add_argument("--output", default="detections.csv")
    parser.add_argument("--labels", default=None,
                        help="Ground truth JSON for evaluation")
    # Tunable parameters with tuned defaults
    parser.add_argument("--tophat-sizes", type=int, nargs="+", default=[9, 15, 21])
    parser.add_argument("--tth-tolerance", type=float, default=0.30)
    parser.add_argument("--snr-threshold", type=float, default=4.8)
    parser.add_argument("--rescue-band-snr", type=float, default=99.0,
                        help="Rescue band SNR threshold (99=disabled)")
    parser.add_argument("--rescue-local-snr", type=float, default=99.0,
                        help="Rescue local annular SNR threshold (99=disabled)")
    parser.add_argument("--min-pixels", type=int, default=2)
    parser.add_argument("--max-pixels", type=int, default=200)
    parser.add_argument("--min-compactness", type=float, default=0.08)
    parser.add_argument("--dup-distance", type=float, default=12)
    parser.add_argument("--ignore-edge", type=int, default=5)
    parser.add_argument("--max-detections", type=int, default=50)
    parser.add_argument("--min-peak-value", type=float, default=6.0)
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
        tophat_sizes=tuple(args.tophat_sizes),
        tth_tolerance=args.tth_tolerance,
        snr_threshold=args.snr_threshold,
        rescue_band_snr=args.rescue_band_snr,
        rescue_local_snr=args.rescue_local_snr,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        min_compactness=args.min_compactness,
        dup_distance=args.dup_distance,
        ignore_edge=args.ignore_edge,
        max_detections=args.max_detections,
        min_peak_value=args.min_peak_value,
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
        f1, tp, fp, fn = compute_f1(detections, labels)
        print(f"F1 score: {f1:.4f} (TP={tp} FP={fp} FN={fn})")


if __name__ == "__main__":
    main()
