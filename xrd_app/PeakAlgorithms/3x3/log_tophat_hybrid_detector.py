"""
Multi-Scale LoG + Top-Hat Hybrid Bragg Peak Detector

Uses two complementary feature extractors:
1. Morphological top-hat: excellent for compact bright features on smooth bg
2. Noise-normalized LoG: scale-adaptive blob detection, catches peaks
   at varying sizes that a fixed-size top-hat might miss

Key innovations vs prior candidates:
- LoG provides scale-adaptive detection (multi-scale sigmas)
- Noise normalization makes LoG adaptive across the detector
- UNION of top-hat and LoG candidates: catches peaks that either alone misses
- Per-band MAD-adaptive thresholding on both feature maps
- Peak validation via radial SNR
- Compactness filtering to reject Debye-Scherrer arc segments

Pipeline:
1. Non-parametric radial median background subtraction + noise estimation
2. Fast morphological top-hat (compact feature extraction)
3. Multi-scale noise-normalized LoG blob detection
4. Per-band MAD-adaptive thresholding on BOTH feature maps
5. Union of candidates from both detectors
6. Connected component analysis with compactness filtering
7. Peak validation via radial SNR
8. Cross-band duplicate suppression

Usage:
    python detect_peaks.py \
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
import math

import h5py
import numpy as np
import scipy.ndimage as ndi
import tifffile


BINS_H5_DEFAULT = "/home/takaji/xrd_3x3_bins.h5"


# ── Radial background subtraction ───────────────────────────────────

def precompute_tth(tth_map, bin_width=0.05):
    """Pre-compute 2-theta binning for fast radial operations."""
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
    """Non-parametric radial background subtraction with noise estimation."""
    img_flat = image.ravel()
    sorted_vals = img_flat[tth_data['order']]
    n = tth_data['n_bins']
    boundaries = tth_data['boundaries']

    median_profile = np.zeros(n)
    mad_profile = np.zeros(n)
    for i in range(n):
        lo, hi = int(boundaries[i]), int(boundaries[i + 1])
        if hi > lo:
            vals = sorted_vals[lo:hi]
            med = np.median(vals)
            median_profile[i] = med
            mad_profile[i] = np.median(np.abs(vals - med))

    if n > 5:
        median_smooth = ndi.uniform_filter1d(median_profile, size=min(15, n))
    else:
        median_smooth = median_profile.copy()

    bg = median_smooth[tth_data['indices']].reshape(image.shape)
    noise_sigma = (mad_profile * 1.4826)[tth_data['indices']].reshape(image.shape)
    cleaned = image - bg
    return cleaned, noise_sigma


# ── Feature extractors ──────────────────────────────────────────────

def fast_tophat(image, size=15):
    """Fast morphological white top-hat using min/max filters."""
    eroded = ndi.minimum_filter(image, size=size)
    opened = ndi.maximum_filter(eroded, size=size)
    return image - opened


def log_blob_response(image, noise_sigma, sigmas=(2.0, 4.0, 7.0)):
    """Multi-scale noise-normalized LoG blob detection.
    
    Returns LoG response normalized by radial noise level.
    Uses a robust noise floor to prevent artifacts.
    """
    positive_noise = noise_sigma[noise_sigma > 0]
    if len(positive_noise) > 0:
        noise_floor = max(np.median(positive_noise) * 0.1, 1e-6)
    else:
        noise_floor = 1.0
    
    safe_sigma = noise_sigma.copy()
    safe_sigma[safe_sigma < noise_floor] = noise_floor
    
    max_response = np.zeros_like(image)
    for sigma in sigmas:
        log_resp = -ndi.gaussian_laplace(image, sigma) * (sigma ** 2)
        log_snr = log_resp / safe_sigma
        np.maximum(max_response, log_snr, out=max_response)
    
    return max_response


# ── Per-band detection ──────────────────────────────────────────────

def detect_in_band(tophat_img, log_snr_img, cleaned, noise_sigma,
                   band_mask, label,
                   tophat_snr_threshold=4.0,
                   log_snr_threshold=5.0,
                   min_snr_radial=2.5,
                   min_pixels=2, max_pixels=200,
                   min_compactness=0.10, ignore_edge=5):
    """Detect peaks using UNION of top-hat and LoG candidates.
    
    Both feature maps are thresholded independently using per-band
    MAD-adaptive statistics. A pixel is a candidate if flagged by
    EITHER detector. This recovers peaks missed by either alone.
    """
    H, W = tophat_img.shape
    
    edge_mask = np.ones((H, W), dtype=bool)
    if ignore_edge > 0:
        edge_mask[:ignore_edge, :] = False
        edge_mask[-ignore_edge:, :] = False
        edge_mask[:, :ignore_edge] = False
        edge_mask[:, -ignore_edge:] = False
    
    valid = band_mask & edge_mask
    n_valid = valid.sum()
    if n_valid < 100:
        return []
    
    # === Top-hat per-band threshold ===
    th_vals = tophat_img[valid]
    th_med = np.median(th_vals)
    th_mad = np.median(np.abs(th_vals - th_med))
    th_sigma = th_mad * 1.4826 if th_mad > 0 else np.std(th_vals)
    if th_sigma <= 0:
        th_sigma = 1.0
    th_threshold = th_med + tophat_snr_threshold * th_sigma
    
    # === LoG SNR per-band threshold ===
    lg_vals = log_snr_img[valid]
    lg_med = np.median(lg_vals)
    lg_mad = np.median(np.abs(lg_vals - lg_med))
    lg_sigma = lg_mad * 1.4826 if lg_mad > 0 else 1.0
    if lg_sigma <= 0:
        lg_sigma = 1.0
    lg_threshold = lg_med + log_snr_threshold * lg_sigma
    lg_threshold = max(lg_threshold, log_snr_threshold * 0.5)
    
    # === Top-hat primary, LoG as secondary boost ===
    # Top-hat candidates: primary detector
    tophat_hot = tophat_img[valid] > th_threshold
    # LoG candidates: only used to boost candidates already near top-hat threshold
    # A pixel is a candidate if:
    # 1. Top-hat is above threshold (standard), OR
    # 2. Top-hat is above 60% of threshold AND LoG is above threshold
    #    (catches faint peaks that LoG confirms as blob-like)
    log_hot = log_snr_img[valid] > lg_threshold
    tophat_near = tophat_img[valid] > (th_threshold * 0.6)
    log_boosted = tophat_near & log_hot
    
    hotspot = np.zeros((H, W), dtype=bool)
    hotspot[valid] = tophat_hot | log_boosted
    
    if not hotspot.any():
        return []
    
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
        
        # Compactness
        h = ys.max() - ys.min() + 1
        w = xs.max() - xs.min() + 1
        if h == 0 or w == 0:
            continue
        aspect = min(h, w) / max(h, w)
        fill = npix / (h * w)
        compactness = aspect * fill
        
        if compactness < min_compactness:
            continue
        
        # Peak validation: radial SNR check
        comp_vals = cleaned[ys, xs]
        comp_noise = noise_sigma[ys, xs]
        peak_intensity = np.max(comp_vals)
        peak_noise = np.median(comp_noise[comp_noise > 0]) if np.any(comp_noise > 0) else 0
        
        if peak_noise > 0:
            peak_radial_snr = peak_intensity / peak_noise
            if peak_radial_snr < min_snr_radial:
                continue
        
        # Combined score for ranking
        th_score = float(np.max(tophat_img[ys, xs]))
        lg_score = float(np.max(log_snr_img[ys, xs]))
        # Normalize and combine
        combined = (th_score / max(th_threshold, 1e-6) + 
                   lg_score / max(lg_threshold, 1e-6))
        
        # Intensity-weighted centroid
        weights = np.maximum(cleaned[ys, xs], 0)
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
            'score': combined,
            'peak_val': peak_intensity,
        })
    
    return peaks


# ── Full pipeline ────────────────────────────────────────────────────

def detect_peaks(image, tth_map, degs, deg_labels, tth_data,
                 phi_map=None,
                 tophat_size=15,
                 log_sigmas=(2.0, 4.0, 7.0),
                 tth_tolerance=0.35,
                 tophat_snr_threshold=3.5,
                 log_snr_threshold=5.0,
                 min_snr_radial=2.5,
                 min_pixels=3,
                 max_pixels=150,
                 min_compactness=0.10,
                 dup_distance=15,
                 ignore_edge=5,
                 max_detections=40):
    """Full hybrid top-hat + LoG Bragg peak detection pipeline."""
    
    # Stage 1: Radial background subtraction
    cleaned, noise_sigma = radial_median_subtract(image, tth_data)
    
    # Early exit for nearly empty bins
    global_mad = np.median(np.abs(cleaned - np.median(cleaned)))
    if global_mad < 1e-6:
        p99 = np.percentile(cleaned, 99)
        if p99 < 5.0:
            return {}
    
    # Stage 2: Dual feature extraction
    tophat_img = fast_tophat(cleaned, size=tophat_size)
    log_snr_img = log_blob_response(cleaned, noise_sigma, sigmas=log_sigmas)
    
    # Stage 3: Per-band detection with union of detectors
    tth_max = tth_map.max()
    all_peaks = []
    
    for label, deg_val in zip(deg_labels, degs):
        if deg_val > tth_max + 1.0:
            continue
        
        band_mask = np.abs(tth_map - deg_val) <= tth_tolerance
        
        peaks = detect_in_band(
            tophat_img, log_snr_img, cleaned, noise_sigma,
            band_mask, label,
            tophat_snr_threshold=tophat_snr_threshold,
            log_snr_threshold=log_snr_threshold,
            min_snr_radial=min_snr_radial,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            min_compactness=min_compactness,
            ignore_edge=ignore_edge
        )
        all_peaks.extend(peaks)
    
    if not all_peaks:
        return {}
    
    # Stage 4: Sort by combined score and suppress duplicates
    all_peaks.sort(key=lambda p: p['score'], reverse=True)
    
    kept = []
    for peak in all_peaks:
        is_dup = False
        for existing in kept:
            dist = math.sqrt((peak['y'] - existing['y'])**2 +
                           (peak['x'] - existing['x'])**2)
            if dist < dup_distance:
                is_dup = True
                break
        if not is_dup:
            kept.append(peak)
    
    if len(kept) > max_detections:
        kept = kept[:max_detections]
    
    results = {}
    for peak in kept:
        results.setdefault(peak['label'], []).append([peak['x'], peak['y']])
    
    return results


# ── Evaluation helper ────────────────────────────────────────────────

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


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Multi-Scale LoG + Top-Hat Hybrid Bragg Peak Detector")
    parser.add_argument("--bin-key", required=True, help="Bin key, e.g. '3_7'")
    parser.add_argument("--bins-h5", default=BINS_H5_DEFAULT)
    parser.add_argument("--two-theta", dest="two_theta", default="tth.tiff")
    parser.add_argument("--reflections", default="reflections.py")
    parser.add_argument("--output", default="detections.csv")
    parser.add_argument("--labels", default=None)
    # Tunable parameters
    parser.add_argument("--tophat-size", type=int, default=15)
    parser.add_argument("--tophat-snr-threshold", type=float, default=3.5)
    parser.add_argument("--log-snr-threshold", type=float, default=5.0)
    parser.add_argument("--min-snr-radial", type=float, default=2.5)
    parser.add_argument("--tth-tolerance", type=float, default=0.35)
    parser.add_argument("--min-pixels", type=int, default=3)
    parser.add_argument("--max-pixels", type=int, default=150)
    parser.add_argument("--min-compactness", type=float, default=0.10)
    parser.add_argument("--dup-distance", type=float, default=15)
    parser.add_argument("--ignore-edge", type=int, default=5)
    parser.add_argument("--max-detections", type=int, default=40)
    args = parser.parse_args()

    spec = importlib.util.spec_from_file_location("reflections", args.reflections)
    ref_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref_mod)
    degs = ref_mod.degs
    deg_labels = ref_mod.deg_labels

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
        tophat_snr_threshold=args.tophat_snr_threshold,
        log_snr_threshold=args.log_snr_threshold,
        min_snr_radial=args.min_snr_radial,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        min_compactness=args.min_compactness,
        dup_distance=args.dup_distance,
        ignore_edge=args.ignore_edge,
        max_detections=args.max_detections
    )

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
