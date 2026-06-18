#!/usr/bin/env python3
"""
Bragg Peak Detector v3: Morphological Top-Hat + Annular Local SNR + 2-Theta Bands

Key improvements over baseline (global percentile thresholding):
1. Non-parametric radial median background subtraction
2. Morphological white top-hat to extract compact bright features
3. 2-theta band masking: only search near known reflection arcs
4. Dual-threshold SNR: annular MAD-based SNR + local spatial SNR
5. Detector gap masking to avoid edge artifacts
6. Adaptive minimum intensity floor based on global image statistics
7. Connected component analysis with strict size/compactness filtering
8. Non-maximum suppression with intensity-weighted centroids
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
from scipy.ndimage import uniform_filter1d, median_filter
from scipy.ndimage import grey_dilation, grey_erosion, uniform_filter
import tifffile


BINS_H5_DEFAULT = "/home/takaji/xrd_5x5_bins.h5"


# ── Data loading ──────────────────────────────────────────────────────

def load_bin_image(bin_key, bins_h5_path=BINS_H5_DEFAULT):
    with h5py.File(bins_h5_path, "r") as f:
        if bin_key not in f:
            raise ValueError(f"Bin '{bin_key}' not found in {bins_h5_path}")
        return f[bin_key][:].astype(np.float64)


# ── 2-theta utilities ────────────────────────────────────────────────

def precompute_tth(tth_map, bin_width=0.05):
    """Pre-compute 2-theta bin edges, centers, and per-pixel bin indices."""
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


def make_tth_band_mask(tth_map, degs, band_half_width=0.25):
    """Create a mask that is True only within band_half_width degrees of any reflection."""
    mask = np.zeros(tth_map.shape, dtype=bool)
    for d in degs:
        mask |= (np.abs(tth_map - d) <= band_half_width)
    return mask


# ── Noise reduction ──────────────────────────────────────────────────

def radial_median_background(image, tth_data):
    """Non-parametric radial background: per-annulus median mapped back to 2D."""
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


def compute_annular_stats(image, tth_data):
    """Compute per-annulus MAD and IQR for noise estimation."""
    img_flat = image.ravel()
    sorted_vals = img_flat[tth_data['order']]
    n = tth_data['n_bins']
    boundaries = tth_data['boundaries']
    
    mad_profile = np.zeros(n)
    for i in range(n):
        lo, hi = int(boundaries[i]), int(boundaries[i + 1])
        if hi > lo:
            vals = sorted_vals[lo:hi]
            med = np.median(vals)
            mad_profile[i] = np.median(np.abs(vals - med))
    
    mad_smooth = uniform_filter1d(mad_profile, size=5)
    mad_smooth = np.maximum(mad_smooth, 0.01)
    return mad_smooth


# ── Morphological top-hat ────────────────────────────────────────────

def white_tophat(image, size=15):
    """White top-hat: extracts bright features smaller than structuring element.
    Uses a circular-ish structuring element via footprint."""
    # Use square for speed - works well for compact spots
    opened = grey_dilation(grey_erosion(image, size=(size, size)), size=(size, size))
    tophat = image - opened
    tophat[tophat < 0] = 0
    return tophat


# ── Detector gap detection ───────────────────────────────────────────

def detect_gap_mask(image, gap_margin=5):
    """Detect the horizontal detector gap (rows of zeros) and mask nearby rows."""
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
                 tophat_size=11,
                 snr_threshold=7.0,
                 local_snr_threshold=3.5,
                 local_window=41,
                 min_pixels=3,
                 max_pixels=150,
                 min_separation=15,
                 min_intensity_sigma=2.5,
                 ignore_edge=5):
    """
    Full peak detection pipeline.
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
        global_mad = np.std(finite_cleaned) * 0.6745  # fallback
    
    # Minimum absolute intensity floor
    min_abs_intensity = global_median + min_intensity_sigma * global_mad * 1.4826
    
    # Step 3: Morphological top-hat on cleaned image
    tophat = white_tophat(cleaned, size=tophat_size)
    
    # Step 4: Compute annular MAD for noise estimation (on cleaned image)
    mad_profile = compute_annular_stats(cleaned, tth_data)
    mad_map = mad_profile[tth_data['indices']].reshape(H, W)
    
    # Step 5: Annular SNR map
    annular_snr = tophat / mad_map
    
    # Step 6: Local spatial SNR - compare each pixel to its local neighborhood
    local_mean = uniform_filter(cleaned, size=local_window)
    local_sq_mean = uniform_filter(cleaned**2, size=local_window)
    local_std = np.sqrt(np.maximum(local_sq_mean - local_mean**2, 0))
    local_std = np.maximum(local_std, 0.01)
    local_snr = (cleaned - local_mean) / local_std
    
    # Step 7: Create masks
    band_mask = make_tth_band_mask(tth_map, degs, band_half_width=band_half_width)
    gap_mask = detect_gap_mask(image, gap_margin=tophat_size + 2)
    
    edge_mask = np.ones((H, W), dtype=bool)
    edge_mask[:ignore_edge, :] = False
    edge_mask[-ignore_edge:, :] = False
    edge_mask[:, :ignore_edge] = False
    edge_mask[:, -ignore_edge:] = False
    
    # Step 8: Combined threshold
    # A pixel is a peak candidate if:
    # - It's in a 2-theta band
    # - It's not at edge or in detector gap
    # - Its annular SNR exceeds threshold
    # - Its local spatial SNR exceeds threshold
    # - Its cleaned value exceeds the absolute intensity floor
    peak_mask = (
        (annular_snr >= snr_threshold) &
        (local_snr >= local_snr_threshold) &
        (cleaned >= min_abs_intensity) &
        band_mask &
        gap_mask &
        edge_mask
    )
    
    # Step 9: Connected components
    cc, n_comp = ndi.label(peak_mask)
    if n_comp == 0:
        return {}
    
    # Step 10: Filter components by size and compute centroids
    candidates = []
    for comp_id in range(1, n_comp + 1):
        ys, xs = np.where(cc == comp_id)
        n_pix = len(ys)
        if n_pix < min_pixels or n_pix > max_pixels:
            continue
        
        # Weighted centroid using cleaned values
        weights = np.maximum(cleaned[ys, xs], 0)
        if weights.sum() > 0:
            cx = np.average(xs, weights=weights)
            cy = np.average(ys, weights=weights)
        else:
            cx, cy = np.mean(xs), np.mean(ys)
        
        cy_int = int(np.clip(round(cy), 0, H-1))
        cx_int = int(np.clip(round(cx), 0, W-1))
        
        # Peak metrics
        max_snr = np.max(annular_snr[ys, xs])
        peak_intensity = np.max(cleaned[ys, xs])
        
        # Compactness: reject very elongated features
        bbox_h = ys.max() - ys.min() + 1
        bbox_w = xs.max() - xs.min() + 1
        aspect = max(bbox_h, bbox_w) / (min(bbox_h, bbox_w) + 1e-6)
        if aspect > 6 and n_pix > 15:
            continue
        
        candidates.append({
            'cx': cx, 'cy': cy,
            'cx_int': cx_int, 'cy_int': cy_int,
            'n_pix': n_pix,
            'max_snr': max_snr,
            'peak_intensity': peak_intensity,
            'aspect': aspect,
        })
    
    # Step 11: Non-maximum suppression (keep strongest within min_separation)
    candidates.sort(key=lambda c: -c['max_snr'])
    
    kept = []
    for cand in candidates:
        too_close = False
        for k in kept:
            dist = np.sqrt((cand['cy'] - k['cy']) ** 2 + (cand['cx'] - k['cx']) ** 2)
            if dist < min_separation:
                too_close = True
                break
        if not too_close:
            kept.append(cand)
    
    # Step 12: Assign reflections based on 2-theta
    results = {}
    for cand in kept:
        cx_int, cy_int = cand['cx_int'], cand['cy_int']
        tth_val = tth_map[cy_int, cx_int]
        best_dist = float("inf")
        label = "unknown"
        for lab, d in zip(deg_labels, degs):
            dist = abs(tth_val - d)
            if dist < best_dist:
                best_dist = dist
                label = lab
        if best_dist <= band_half_width:
            results.setdefault(label, []).append([int(round(cand['cx'])), int(round(cand['cy']))])
    
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
        return 1.0
    if not gt_pts or not det_pts:
        return 0.0

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
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Morphological top-hat + annular SNR Bragg peak detector")
    parser.add_argument("--bin-key", required=True, help="Bin key, e.g. '3_7'")
    parser.add_argument("--bins-h5", default=BINS_H5_DEFAULT,
                        help="Pre-built bin images HDF5")
    parser.add_argument("--two-theta", dest="two_theta", default="tth.tiff")
    parser.add_argument("--reflections", default="reflections.py")
    parser.add_argument("--output", default="detections.csv")
    parser.add_argument("--labels", default=None,
                        help="Ground truth JSON for evaluation")
    # Tunable parameters with best defaults
    parser.add_argument("--band-half-width", type=float, default=0.35)
    parser.add_argument("--snr-threshold", type=float, default=7.0)
    parser.add_argument("--local-snr-threshold", type=float, default=3.5)
    parser.add_argument("--local-window", type=int, default=41)
    parser.add_argument("--tophat-size", type=int, default=11)
    parser.add_argument("--min-pixels", type=int, default=3)
    parser.add_argument("--max-pixels", type=int, default=150)
    parser.add_argument("--min-separation", type=float, default=15.0)
    parser.add_argument("--min-intensity-sigma", type=float, default=2.5)
    args = parser.parse_args()

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
        tophat_size=args.tophat_size,
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

    # Evaluate if labels provided
    if args.labels:
        with open(args.labels) as f:
            labels = json.load(f)
        f1 = compute_f1(detections, labels)
        print(f"F1 score: {f1:.4f}")


if __name__ == "__main__":
    main()
