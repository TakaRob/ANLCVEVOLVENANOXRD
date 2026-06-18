"""
Baseline Bragg peak detector for 5x5 binned XRD images.

Loads a bin image from the pre-built HDF5 file, applies radial background
subtraction (noise reduction), then detects bright spots via global
percentile thresholding and connected-component labeling.  Each detected
peak centroid is assigned to the nearest 2-theta reflection arc.

Usage
-----
    python baseline.py \
        --bin-key 0_0 \
        --two-theta tth.tiff \
        --reflections reflections.py \
        --output detections.csv

If --labels is provided (a JSON file keyed by reflection name, values are
[[x,y], ...] lists), the script computes and prints the F1 score using a
40-pixel matching tolerance.
"""

import argparse
import csv
import json
import importlib.util
from pathlib import Path

import h5py
import numpy as np
import scipy.ndimage as ndi
import tifffile

from noise_reduction_algorithms import (
    compute_tth_binning,
    compute_radial_profile,
    fit_all_models,
    build_background_image,
    subtract_background,
)


BINS_H5_DEFAULT = "/home/takaji/xrd_3x3_bins.h5"


# ── Data loading ──────────────────────────────────────────────────────

def load_bin_image(bin_key, bins_h5_path=BINS_H5_DEFAULT):
    """Load a pre-computed bin image from the HDF5 file."""
    with h5py.File(bins_h5_path, "r") as f:
        if bin_key not in f:
            raise ValueError(f"Bin '{bin_key}' not found in {bins_h5_path}")
        return f[bin_key][:].astype(np.float64)


# ── Noise reduction ──────────────────────────────────────────────────

def reduce_noise(image, tth_map, algorithm="gaussian", strength=1.0):
    """Apply radial background subtraction using a fitted noise model."""
    edges, centers, n_bins, indices, counts = compute_tth_binning(tth_map)
    valid_mask = counts > 50
    profile = compute_radial_profile(image, indices, n_bins)
    fits = fit_all_models(centers, profile, valid_mask, edges[0], edges[-1])

    if algorithm not in fits:
        return image.copy()

    fit = fits[algorithm]
    bg = build_background_image(tth_map, centers, fit["profile"], indices)
    return subtract_background(image, bg, strength=strength)


# ── Peak detection ───────────────────────────────────────────────────

def find_peaks(image, tth_map, degs, deg_labels,
               percentile=97.0, min_pixels=3, pad=10, ignore_edge=2):
    """Detect bright peaks via global thresholding.

    Returns a dict mapping reflection labels to lists of (x, y) centroids.
    """
    finite = image[np.isfinite(image)]
    if len(finite) == 0:
        return {}

    thr = np.percentile(finite, percentile)
    hotspot = image >= thr

    if ignore_edge > 0:
        hotspot[:ignore_edge, :] = False
        hotspot[-ignore_edge:, :] = False
        hotspot[:, :ignore_edge] = False
        hotspot[:, -ignore_edge:] = False

    cc, n_comp = ndi.label(hotspot)
    peaks = []
    for comp_id in range(1, n_comp + 1):
        ys, xs = np.where(cc == comp_id)
        if len(ys) < min_pixels:
            continue
        cx, cy = int(np.mean(xs)), int(np.mean(ys))
        peaks.append((cx, cy))

    # Assign each peak to the nearest 2-theta reflection
    results = {}
    for cx, cy in peaks:
        if 0 <= cy < tth_map.shape[0] and 0 <= cx < tth_map.shape[1]:
            tth_val = tth_map[cy, cx]
            best_dist = float("inf")
            label = "unknown"
            for lab, d in zip(deg_labels, degs):
                dist = abs(tth_val - d)
                if dist < best_dist:
                    best_dist = dist
                    label = lab
            results.setdefault(label, []).append([cx, cy])
        else:
            results.setdefault("unknown", []).append([cx, cy])

    return results


# ── Evaluation ───────────────────────────────────────────────────────

def compute_f1(detections, labels, tolerance=40):
    """Compute F1 score across all reflections with a pixel-distance tolerance."""
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
    matched_det = set()
    for di, dp in enumerate(det_arr):
        dists = np.sqrt(np.sum((gt_arr - dp) ** 2, axis=1))
        nearest = int(np.argmin(dists))
        if dists[nearest] <= tolerance and nearest not in matched_gt:
            matched_gt.add(nearest)
            matched_det.add(di)

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
    parser = argparse.ArgumentParser(description="Baseline peak detector for 5x5 bins")
    parser.add_argument("--bin-key", required=True, help="Bin key, e.g. '3_7'")
    parser.add_argument("--bins-h5", default=BINS_H5_DEFAULT, help="Pre-built bin images HDF5")
    parser.add_argument("--two-theta", dest="two_theta", default="tth.tiff")
    parser.add_argument("--reflections", default="reflections.py")
    parser.add_argument("--output", default="detections.csv")
    parser.add_argument("--labels", default=None, help="Ground truth JSON for evaluation")
    parser.add_argument("--noise-algorithm", default="gaussian",
                        choices=["gaussian", "split_gaussian", "skewed_gaussian", "fourier", "none"])
    parser.add_argument("--percentile", type=float, default=97.0)
    args = parser.parse_args()

    # Load reflections
    spec = importlib.util.spec_from_file_location("reflections", args.reflections)
    ref_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref_mod)
    degs = ref_mod.degs
    deg_labels = ref_mod.deg_labels

    # Load image
    image = load_bin_image(args.bin_key, args.bins_h5)
    tth_map = tifffile.imread(args.two_theta)

    # Noise reduction
    if args.noise_algorithm != "none":
        image = reduce_noise(image, tth_map, algorithm=args.noise_algorithm)

    # Detect peaks
    detections = find_peaks(image, tth_map, degs, deg_labels,
                            percentile=args.percentile)

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
