#!/usr/bin/env python3
"""
Bragg Peak Detector v3-tuned: Multi-Scale Top-Hat + Dual SNR
Tuned from r001-8a641fb0 (tophat_dual_snr_detector, F1=0.775)

Key tuning changes from parent:
1. Multi-scale top-hat: sizes [7, 11, 15] (from single 11) to capture varying peak sizes
2. Sigma-clipped MAD noise estimation: excludes bright peaks from noise floor
3. Lowered thresholds: snr 7.0->5.8, local_snr 3.5->3.0, min_intensity_sigma 2.5->2.0
4. min_pixels 3->4: rejects single-pixel noise speckles that pass SNR thresholds
5. min_separation 15->8: resolves closely-spaced peaks that were merged
6. local_window 41->31: tighter local neighborhood for more sensitive local SNR
7. max_pixels 150->200: allows slightly larger peak components
"""

import argparse
import csv
import json
import importlib.util
import sys
from pathlib import Path

import h5py
import numpy as np
import scipy.ndimage as ndi
from scipy.ndimage import uniform_filter1d
from scipy.ndimage import grey_dilation, grey_erosion, uniform_filter
import tifffile

BINS_H5_DEFAULT = "/home/takaji/xrd_4x4_bins.h5"


# ── Data loading ──────────────────────────────────────────────────────

def load_bin_image(bin_key, bins_h5_path=BINS_H5_DEFAULT):
    with h5py.File(bins_h5_path, "r") as f:
        if bin_key not in f:
            raise ValueError(f"Bin '{bin_key}' not found in {bins_h5_path}")
        return f[bin_key][:].astype(np.float64)


# ── 2-theta utilities ────────────────────────────────────────────────

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


def make_tth_band_mask(tth_map, degs, band_half_width=0.35):
    mask = np.zeros(tth_map.shape, dtype=bool)
    for d in degs:
        mask |= (np.abs(tth_map - d) <= band_half_width)
    return mask


# ── Noise reduction ──────────────────────────────────────────────────

def radial_median_background(image, tth_data):
    img_flat = image.ravel()
    sorted_vals = img_flat[tth_data['order']]
    n = tth_data['n_bins']
    boundaries = tth_data['boundaries']
    profile = np.zeros(n)
    for i in range(n):
        lo, hi = int(boundaries[i]), int(boundaries[i + 1])
        if hi > lo:
            profile[i] = np.median(sorted_vals[lo:hi])
    profile_smooth = uniform_filter1d(profile, size=5)
    bg = profile_smooth[tth_data['indices']].reshape(image.shape)
    return bg, profile_smooth


def compute_annular_stats_sigmaclip(image, tth_data, n_iters=2, clip_sigma=3.0):
    """Per-annulus MAD with sigma clipping to exclude bright peak pixels."""
    img_flat = image.ravel()
    sorted_vals = img_flat[tth_data['order']]
    n = tth_data['n_bins']
    boundaries = tth_data['boundaries']
    mad_profile = np.zeros(n)
    for i in range(n):
        lo, hi = int(boundaries[i]), int(boundaries[i + 1])
        if hi > lo:
            vals = sorted_vals[lo:hi]
            for _ in range(n_iters):
                med = np.median(vals)
                mad = np.median(np.abs(vals - med))
                if mad < 1e-10:
                    break
                threshold = med + clip_sigma * mad * 1.4826
                vals = vals[vals <= threshold]
                if len(vals) < 10:
                    break
            mad_profile[i] = np.median(np.abs(vals - np.median(vals)))
    mad_smooth = uniform_filter1d(mad_profile, size=5)
    mad_smooth = np.maximum(mad_smooth, 0.01)
    return mad_smooth


# ── Morphological top-hat ────────────────────────────────────────────

def white_tophat(image, size=15):
    opened = grey_dilation(grey_erosion(image, size=(size, size)), size=(size, size))
    tophat = image - opened
    tophat[tophat < 0] = 0
    return tophat


def multiscale_tophat(image, sizes=(7, 11, 15)):
    result = np.zeros_like(image)
    for s in sizes:
        th = white_tophat(image, size=s)
        result = np.maximum(result, th)
    return result


# ── Detector gap detection ───────────────────────────────────────────

def detect_gap_mask(image, gap_margin=5):
    row_means = np.mean(image, axis=1)
    gap_rows = np.where(row_means < 0.5)[0]
    mask = np.ones(image.shape, dtype=bool)
    for r in gap_rows:
        r0 = max(0, r - gap_margin)
        r1 = min(image.shape[0], r + gap_margin + 1)
        mask[r0:r1, :] = False
    return mask


# ── Peak detection pipeline ─────────────────────────────────────────

def detect_peaks(image, tth_map, tth_data, degs, deg_labels,
                 band_half_width=0.35,
                 tophat_sizes=(7, 11, 15),
                 snr_threshold=5.8,
                 local_snr_threshold=3.0,
                 local_window=31,
                 min_pixels=4,
                 max_pixels=200,
                 min_separation=8,
                 min_intensity_sigma=2.0,
                 ignore_edge=5,
                 max_aspect=6):
    """
    Peak detection: radial median bg subtraction -> multi-scale tophat ->
    dual SNR thresholding (annular + local) -> connected components ->
    shape filtering -> NMS -> reflection assignment.
    """
    H, W = image.shape

    # Step 1: Background subtraction
    bg, bg_profile = radial_median_background(image, tth_data)
    cleaned = image - bg

    # Step 2: Compute global image statistics for adaptive floor
    finite_cleaned = cleaned[np.isfinite(cleaned)]
    global_median = np.median(finite_cleaned)
    global_mad = np.median(np.abs(finite_cleaned - global_median))
    if global_mad < 0.01:
        global_mad = np.std(finite_cleaned) * 0.6745

    # Minimum absolute intensity floor
    min_abs_intensity = global_median + min_intensity_sigma * global_mad * 1.4826

    # Step 3: Multi-scale morphological top-hat on cleaned image
    tophat = multiscale_tophat(cleaned, sizes=tophat_sizes)

    # Step 4: Sigma-clipped MAD for per-annulus noise estimation
    mad_profile = compute_annular_stats_sigmaclip(cleaned, tth_data)
    mad_map = mad_profile[tth_data['indices']].reshape(H, W)

    # Step 5: Annular SNR map
    annular_snr = tophat / mad_map

    # Step 6: Local spatial SNR
    local_mean = uniform_filter(cleaned, size=local_window)
    local_sq_mean = uniform_filter(cleaned**2, size=local_window)
    local_std = np.sqrt(np.maximum(local_sq_mean - local_mean**2, 0))
    local_std = np.maximum(local_std, 0.01)
    local_snr = (cleaned - local_mean) / local_std

    # Step 7: Create masks
    band_mask = make_tth_band_mask(tth_map, degs, band_half_width=band_half_width)
    gap_mask = detect_gap_mask(image, gap_margin=max(tophat_sizes) + 2)

    edge_mask = np.ones((H, W), dtype=bool)
    edge_mask[:ignore_edge, :] = False
    edge_mask[-ignore_edge:, :] = False
    edge_mask[:, :ignore_edge] = False
    edge_mask[:, -ignore_edge:] = False

    valid_mask = band_mask & gap_mask & edge_mask

    # Step 8: Combined threshold - dual SNR AND
    peak_mask = (
        (annular_snr >= snr_threshold) &
        (local_snr >= local_snr_threshold) &
        (cleaned >= min_abs_intensity) &
        valid_mask
    )

    # Step 9: Connected components with vectorized size filtering
    cc, n_comp = ndi.label(peak_mask)
    if n_comp == 0:
        return {}

    sizes = np.bincount(cc.ravel(), minlength=n_comp + 1)
    valid_ids = np.where((sizes[1:] >= min_pixels) & (sizes[1:] <= max_pixels))[0] + 1
    if len(valid_ids) == 0:
        return {}

    slices = ndi.find_objects(cc)
    degs_arr = np.array(degs)

    candidates = []
    for comp_id in valid_ids:
        sl = slices[comp_id - 1]
        if sl is None:
            continue

        patch_cc = cc[sl]
        mask = (patch_cc == comp_id)
        n_pix = int(sizes[comp_id])

        local_ys, local_xs = np.where(mask)
        ys = local_ys + sl[0].start
        xs = local_xs + sl[1].start

        # Weighted centroid
        weights = np.maximum(cleaned[ys, xs], 0)
        if weights.sum() > 0:
            cx = np.average(xs, weights=weights)
            cy = np.average(ys, weights=weights)
        else:
            cx, cy = float(np.mean(xs)), float(np.mean(ys))

        cy_int = int(np.clip(round(cy), 0, H-1))
        cx_int = int(np.clip(round(cx), 0, W-1))

        max_snr = float(np.max(annular_snr[ys, xs]))

        # Shape filtering
        bbox_h = sl[0].stop - sl[0].start
        bbox_w = sl[1].stop - sl[1].start
        aspect = max(bbox_h, bbox_w) / (min(bbox_h, bbox_w) + 1e-6)
        if aspect > max_aspect and n_pix > 15:
            continue
        if n_pix > 20:
            fill = n_pix / (bbox_h * bbox_w)
            if fill < 0.08:
                continue

        # Assign reflection
        tth_val = tth_map[cy_int, cx_int]
        dists = np.abs(degs_arr - tth_val)
        best_idx = np.argmin(dists)
        if dists[best_idx] > band_half_width:
            continue
        label = deg_labels[best_idx % len(deg_labels)]

        candidates.append({
            'cx': cx, 'cy': cy,
            'cx_int': cx_int, 'cy_int': cy_int,
            'max_snr': max_snr,
            'label': label,
        })

    # Step 10: Non-maximum suppression
    candidates.sort(key=lambda c: -c['max_snr'])
    kept = []
    sep_sq = min_separation * min_separation
    for cand in candidates:
        too_close = False
        for k in kept:
            dx = cand['cx'] - k['cx']
            dy = cand['cy'] - k['cy']
            if dx*dx + dy*dy < sep_sq:
                too_close = True
                break
        if not too_close:
            kept.append(cand)

    # Step 11: Build output
    results = {}
    for c in kept:
        results.setdefault(c['label'], []).append([int(round(c['cx'])), int(round(c['cy']))])

    return results


# ── Evaluation ───────────────────────────────────────────────────────

def compute_f1(detections, labels, tolerance=40):
    det_pts = []
    for pts in detections.values():
        det_pts.extend(pts)
    gt_pts = []
    for pts in labels.values():
        gt_pts.extend(pts)

    if not gt_pts and not det_pts:
        return 1.0, 0, 0, 0
    if not gt_pts:
        return 0.0, 0, len(det_pts), 0
    if not det_pts:
        return 0.0, 0, 0, len(gt_pts)

    det_arr = np.array(det_pts, dtype=float)
    gt_arr = np.array(gt_pts, dtype=float)

    matched_gt = set()
    for di, dp in enumerate(det_arr):
        dists = np.sqrt(np.sum((gt_arr - dp) ** 2, axis=1))
        order = np.argsort(dists)
        for gi in order:
            if dists[gi] > tolerance:
                break
            if gi not in matched_gt:
                matched_gt.add(gi)
                break

    tp = len(matched_gt)
    fp = len(det_pts) - tp
    fn = len(gt_pts) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0, tp, fp, fn
    return 2 * precision * recall / (precision + recall), tp, fp, fn


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Tuned Multi-Scale Top-Hat + Dual SNR Bragg peak detector")
    parser.add_argument("--bin-key", required=True)
    parser.add_argument("--bins-h5", default=BINS_H5_DEFAULT)
    parser.add_argument("--two-theta", dest="two_theta", default="tth.tiff")
    parser.add_argument("--reflections", default="reflections.py")
    parser.add_argument("--output", default="detections.csv")
    parser.add_argument("--labels", default=None)
    # Tunable parameters with optimal defaults
    parser.add_argument("--band-half-width", type=float, default=0.35)
    parser.add_argument("--snr-threshold", type=float, default=5.8)
    parser.add_argument("--local-snr-threshold", type=float, default=3.0)
    parser.add_argument("--local-window", type=int, default=31)
    parser.add_argument("--tophat-sizes", type=str, default="7,11,15")
    parser.add_argument("--min-pixels", type=int, default=4)
    parser.add_argument("--max-pixels", type=int, default=200)
    parser.add_argument("--min-separation", type=float, default=8.0)
    parser.add_argument("--min-intensity-sigma", type=float, default=2.0)
    args = parser.parse_args()

    tophat_sizes = tuple(int(x) for x in args.tophat_sizes.split(","))

    # Load reflections
    spec = importlib.util.spec_from_file_location("reflections", args.reflections)
    ref_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref_mod)
    degs = ref_mod.degs
    deg_labels = ref_mod.deg_labels

    # Load data
    image = load_bin_image(args.bin_key, args.bins_h5)
    tth_map = tifffile.imread(args.two_theta)

    # Precompute 2-theta binning
    tth_data = precompute_tth(tth_map)

    # Detect peaks
    detections = detect_peaks(
        image, tth_map, tth_data, degs, deg_labels,
        band_half_width=args.band_half_width,
        snr_threshold=args.snr_threshold,
        local_snr_threshold=args.local_snr_threshold,
        local_window=args.local_window,
        tophat_sizes=tophat_sizes,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        min_separation=args.min_separation,
        min_intensity_sigma=args.min_intensity_sigma,
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
