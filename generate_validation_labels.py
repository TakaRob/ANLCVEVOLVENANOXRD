"""
Transfer 5x5 holdout annotations to 3x3 and 4x4 bin grids.

For each annotated 5x5 bin, finds spatially overlapping bins at the target
resolution and verifies each peak is present by checking local signal-to-noise
at the known peak location. No full detection pipeline needed — we already
know WHERE the peaks are, we just need to confirm they're visible at lower SNR.

Two verification methods (peak confirmed if EITHER passes):
  1. Local SNR: background-subtracted intensity in a 5px radius around the
     peak location exceeds 3× the local MAD noise estimate.
  2. Percentile check: the peak pixel is above the 99th percentile of the
     background-subtracted image (the baseline approach).
"""

import json
import importlib.util
from collections import defaultdict
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401 — registers LZ4 filter for 3x3/4x4 HDF5 files
import numpy as np
import tifffile

BASE = Path(__file__).resolve().parent

BINS_H5 = {
    3: Path("/home/takaji/xrd_3x3_bins.h5"),
    4: Path("/home/takaji/xrd_4x4_bins.h5"),
    5: Path("/home/takaji/xrd_5x5_bins.h5"),
}

HOLDOUT_5x5 = BASE / "cvevolve_5x5" / "holdout_data"
ANNOTATIONS_DIR = BASE / "results" / "scan203"
HOLDOUT_DIRS = {
    3: BASE / "cvevolve_3x3" / "holdout_data",
    4: BASE / "cvevolve_4x4" / "holdout_data",
}

PEAK_RADIUS = 5
SNR_THRESHOLD = 3.0
ANNULUS_INNER = 10
ANNULUS_OUTER = 20


def load_module(path):
    spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_reflections():
    ref = load_module(HOLDOUT_5x5 / "reflections.py")
    return ref.degs, ref.deg_labels


# ── Fast background subtraction (radial median) ────────────────────

def radial_median_subtract(image, tth_map, bin_width=0.05):
    """Vectorized radial median background subtraction."""
    tth_min, tth_max = float(tth_map.min()), float(tth_map.max())
    edges = np.arange(tth_min, tth_max + bin_width, bin_width)
    n_bins = len(edges) - 1
    flat_tth = tth_map.ravel()
    flat_img = image.ravel()
    indices = np.clip(np.digitize(flat_tth, edges) - 1, 0, n_bins - 1)

    bg = np.zeros_like(flat_img)
    for b in range(n_bins):
        mask = indices == b
        if mask.sum() > 50:
            bg[mask] = np.median(flat_img[mask])

    return image - bg.reshape(image.shape)


# ── Local SNR check at a known peak location ───────────────────────

def check_peak_present(cleaned, px, py, threshold_99):
    """Check if a peak is visible at (px, py) using local SNR and percentile."""
    h, w = cleaned.shape
    if py < 3 or py >= h - 3 or px < 3 or px >= w - 3:
        return False, "edge"

    # Peak value: max in a small window around the annotated position
    r = PEAK_RADIUS
    y0, y1 = max(0, py - r), min(h, py + r + 1)
    x0, x1 = max(0, px - r), min(w, px + r + 1)
    peak_val = np.max(cleaned[y0:y1, x0:x1])

    # Method 1: local SNR using annular background
    ri, ro = ANNULUS_INNER, ANNULUS_OUTER
    yy, xx = np.ogrid[max(0, py - ro):min(h, py + ro + 1),
                       max(0, px - ro):min(w, px + ro + 1)]
    cy = py - max(0, py - ro)
    cx = px - max(0, px - ro)
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    annulus_mask = (dist >= ri) & (dist <= ro)
    if annulus_mask.sum() > 20:
        annulus_region = cleaned[max(0, py - ro):min(h, py + ro + 1),
                                 max(0, px - ro):min(w, px + ro + 1)]
        bg_vals = annulus_region[annulus_mask]
        bg_median = np.median(bg_vals)
        bg_mad = np.median(np.abs(bg_vals - bg_median)) * 1.4826
        if bg_mad > 0:
            local_snr = (peak_val - bg_median) / bg_mad
            if local_snr >= SNR_THRESHOLD:
                return True, "snr"

    # Method 2: above 99th percentile of the full image
    if peak_val >= threshold_99:
        return True, "pct99"

    return False, "below"


# ── Spatial mapping ─────────────────────────────────────────────────

def find_overlapping_bins(br5, bc5, target_bin_size, gm5, gm_target):
    sr_start, sr_end = br5 * 5, br5 * 5 + 4
    sc_start, sc_end = bc5 * 5, bc5 * 5 + 4

    bs = target_bin_size
    br_start, br_end = sr_start // bs, sr_end // bs
    bc_start, bc_end = sc_start // bs, sc_end // bs

    frames_5 = set(gm5["bins"].get(f"{br5}_{bc5}", []))
    target_bins = gm_target["bins"]

    candidates = []
    for br in range(br_start, br_end + 1):
        for bc in range(bc_start, bc_end + 1):
            key = f"{br}_{bc}"
            if key in target_bins:
                overlap = len(frames_5 & set(target_bins[key]))
                candidates.append((key, overlap))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates


def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# ── Main ────────────────────────────────────────────────────────────

def generate_labels_for_bin_size(target_bin_size, ann5, gm5, tth_map, degs, deg_labels):
    holdout_dir = HOLDOUT_DIRS[target_bin_size]
    with open(holdout_dir / "grid_mapping.json") as f:
        gm_target = json.load(f)

    h5_path = str(BINS_H5[target_bin_size])

    annotations_out = {}
    empty_bins_out = []

    stats = {
        "total_5x5_bins": 0,
        "total_peaks": 0,
        "confirmed_peaks": 0,
        "dropped_peaks": 0,
        "per_reflection": defaultdict(lambda: {"total": 0, "confirmed": 0}),
        "confirmed_by_snr": 0,
        "confirmed_by_pct99": 0,
    }

    # Cache cleaned images + thresholds to avoid reprocessing
    cleaned_cache = {}

    def get_cleaned(bin_key, h5f):
        if bin_key in cleaned_cache:
            return cleaned_cache[bin_key]
        if bin_key not in h5f:
            cleaned_cache[bin_key] = None, None
            return None, None
        image = np.clip(h5f[bin_key][:].astype(np.float64), 0, 1e9)
        cleaned = radial_median_subtract(image, tth_map)
        finite = cleaned[np.isfinite(cleaned)]
        thr99 = np.percentile(finite, 99.0) if len(finite) > 0 else np.inf
        cleaned_cache[bin_key] = cleaned, thr99
        return cleaned, thr99

    print(f"\n{'='*60}")
    print(f"  Generating {target_bin_size}x{target_bin_size} holdout labels")
    print(f"{'='*60}")

    n_total = len(ann5)

    with h5py.File(h5_path, "r") as h5f:
        for i, (bk5, peaks5) in enumerate(ann5.items()):
            pct = (i + 1) / n_total
            bar = "█" * int(pct * 30) + "░" * (30 - int(pct * 30))
            print(f"\r  [{bar}] {i+1}/{n_total} ({100*pct:.0f}%) bin {bk5}    ", end="", flush=True)

            stats["total_5x5_bins"] += 1
            br5, bc5 = map(int, bk5.split("_"))

            peak_entries = {r: pts for r, pts in peaks5.items() if r != "__reviewed__"}

            if not peak_entries:
                overlapping = find_overlapping_bins(br5, bc5, target_bin_size, gm5, gm_target)
                for target_key, _ in overlapping:
                    if target_key not in annotations_out and target_key not in empty_bins_out:
                        empty_bins_out.append(target_key)
                continue

            overlapping = find_overlapping_bins(br5, bc5, target_bin_size, gm5, gm_target)
            if not overlapping:
                continue

            for reflection, points in peak_entries.items():
                for px, py in points:
                    stats["total_peaks"] += 1
                    stats["per_reflection"][reflection]["total"] += 1

                    confirmed = False
                    confirmed_in_key = None

                    for target_key, overlap_count in overlapping:
                        cleaned, thr99 = get_cleaned(target_key, h5f)
                        if cleaned is None:
                            continue

                        present, method = check_peak_present(cleaned, px, py, thr99)
                        if present:
                            confirmed = True
                            confirmed_in_key = target_key
                            if method == "snr":
                                stats["confirmed_by_snr"] += 1
                            else:
                                stats["confirmed_by_pct99"] += 1
                            break

                    if confirmed:
                        stats["confirmed_peaks"] += 1
                        stats["per_reflection"][reflection]["confirmed"] += 1
                        annotations_out.setdefault(confirmed_in_key, {})
                        annotations_out[confirmed_in_key].setdefault(reflection, []).append([px, py])
                    else:
                        stats["dropped_peaks"] += 1

    print()

    # Deduplicate points within each bin
    for bk, refs in annotations_out.items():
        for ref, pts in refs.items():
            unique = []
            for p in pts:
                if not any(distance(p, u) < 10 for u in unique):
                    unique.append(p)
            refs[ref] = unique

    empty_bins_out = [b for b in empty_bins_out if b not in annotations_out]

    total_points = sum(len(pts) for refs in annotations_out.values() for pts in refs.values())

    print(f"\n  Results for {target_bin_size}x{target_bin_size}:")
    print(f"    5x5 source bins:         {stats['total_5x5_bins']}")
    print(f"    Total peaks to transfer: {stats['total_peaks']}")
    print(f"    Confirmed peaks:         {stats['confirmed_peaks']} ({100*stats['confirmed_peaks']/max(1,stats['total_peaks']):.1f}%)")
    print(f"    Dropped (too faint):     {stats['dropped_peaks']} ({100*stats['dropped_peaks']/max(1,stats['total_peaks']):.1f}%)")
    print(f"    Confirmed by local SNR:  {stats['confirmed_by_snr']}")
    print(f"    Confirmed by 99th pct:   {stats['confirmed_by_pct99']}")
    print(f"    Target bins annotated:   {len(annotations_out)} (with {total_points} total points)")
    print(f"    Target bins empty:       {len(empty_bins_out)}")

    print(f"\n    Per-reflection transfer rates:")
    for ref in sorted(stats["per_reflection"]):
        s = stats["per_reflection"][ref]
        rate = 100 * s["confirmed"] / max(1, s["total"])
        print(f"      {ref:>8}: {s['confirmed']:>3}/{s['total']:>3} ({rate:.0f}%)")

    return annotations_out, empty_bins_out, stats


def main():
    print("Loading shared data...")
    tth_map = tifffile.imread(str(HOLDOUT_5x5 / "tth.tiff"))
    degs, deg_labels = load_reflections()

    with open(HOLDOUT_5x5 / "bin_annotations.json") as f:
        ann5 = json.load(f)
    with open(HOLDOUT_5x5 / "grid_mapping.json") as f:
        gm5 = json.load(f)

    print(f"Loaded {len(ann5)} annotated 5x5 holdout bins", flush=True)
    print(f"Reflections: {deg_labels}", flush=True)

    for target_bs in [3, 4]:
        annotations, empty_bins, stats = generate_labels_for_bin_size(
            target_bs, ann5, gm5, tth_map, degs, deg_labels
        )

        # Write to the labeling tool's annotation file for manual review
        ann_path = ANNOTATIONS_DIR / f"bin_annotations_{target_bs}x{target_bs}.json"
        ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
        with open(ann_path, "w") as f:
            json.dump(annotations, f, indent=2)
        print(f"\n    Wrote {ann_path}")
        print(f"    Review in labeling tool, then use 'Update Holdout Set' button.")

    print(f"\n{'='*60}")
    print("  Done! Holdout labels ready for 3x3 and 4x4 CVEvolve runs.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
