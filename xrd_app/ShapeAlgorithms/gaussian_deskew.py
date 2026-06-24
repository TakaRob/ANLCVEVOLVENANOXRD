"""
Skew-aware shape finder: serpentine-deskew cross-bin linking + gaussian filtering.

Drop-in variant of ``gaussian.py`` (same Phase-2/3 stage, same interface) that
fixes the **serpentine-raster registration error** which fragments features at
fine binning — most visibly at 1×1, where a single physical Bragg feature breaks
into *horizontal slices* (it spans scan columns within a row but not across rows).

Why the baseline slices it
--------------------------
``gaussian.link_peaks`` joins detections in 8-connected ``(row, col)`` bins only
when their detector ``(x, y)`` also agree within ``link_tolerance`` px. A scan is a
serpentine raster — alternate rows are collected in opposite directions, so any
stage backlash offsets the *column registration* between adjacent rows. The same
grain (same detector pixel) then appears one row away at a **shifted column**,
outside the ±1 window, and the vertical link is silently dropped → horizontal
slicing. The size of that shift varies scan to scan (different backlash), so it
must be measured per scan, not hard-coded.

What this module does
---------------------
1. **Estimate the per-row column skew from the data itself.** For each adjacent
   row pair it finds *unambiguous* 1:1 detector-position matches (a detector cell
   holding exactly one detection in each row, agreeing within ``link_tolerance``)
   and takes the median column offset — the systematic backlash shift for that row
   pair. Unambiguous-only matching avoids the bias from widespread reflections
   where many same-orientation grains create spurious cross-row matches.
2. **De-skew** by accumulating those shifts into a per-row correction and linking
   on the corrected column, so the grain's detections in adjacent rows line up.
3. **Adaptive residual window.** When a real skew is detected (median |shift| ≥ 1)
   it widens the *cross-row* column reach by one to mop up residual sub-step
   misalignment; when no skew is present (well-registered binned grids) it stays at
   ±1, so this module is **neutral on already-clean scans** and only acts where
   there is skew to correct.

Everything downstream (gaussian-profile filtering, characterization, the metrics
``chi_fwhm`` / ``tth_fwhm``) is **reused unchanged** from ``gaussian.py`` — only
the linking step differs. What deskew cannot fix is missing detections (SNR
dropouts): those need binning or a lower threshold, not linking.

Register in ``catalog.json`` and select with
``xrd-app shapes --algorithm gaussian_deskew``.
"""

import argparse
import importlib.util
import json
import os
from collections import defaultdict

import numpy as np
import tifffile

DEFAULT_LINK_TOLERANCE = 5   # px — same meaning as gaussian.py
_MAX_SHIFT = 15              # px-cols — search range for the per-row backlash shift
_MIN_PAIRS = 5              # min unambiguous matches to trust a row-pair shift


# Reuse the (unchanged) gaussian Phase-3 characterization so this variant differs
# only in linking. Loaded from the sibling file so it works whether imported as a
# package module or exec'd standalone by io.load_module.
def _load_gaussian_base():
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        "_gaussian_base", os.path.join(here, "gaussian.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_BASE = _load_gaussian_base()
check_gaussian_profile = _BASE.check_gaussian_profile
characterize_features = _BASE.characterize_features          # reused verbatim
estimate_beam_center = _BASE.estimate_beam_center
_coerce = _BASE._coerce


# ── Phase 2: skew-aware spatial linking ─────────────────────────────
def estimate_row_shifts(by_cell, n_rows, link_tolerance):
    """Per adjacent-row pair, the systematic column offset of the SAME grain.

    ``by_cell`` maps ``(row, x//tol, y//tol)`` → list of ``(col, x, y)``. Uses only
    unambiguous 1:1 detector-cell matches (one detection in each row) so widespread
    reflections don't bias the estimate. Returns ``{row: shift_to_next_row}``.
    """
    tol2 = link_tolerance * link_tolerance
    shifts = {}
    cells_by_row = defaultdict(list)
    for (r, cx, cy), lst in by_cell.items():
        cells_by_row[r].append((cx, cy, lst))
    for r in range(n_rows - 1):
        diffs = []
        for (cx, cy, lst) in cells_by_row.get(r, []):
            if len(lst) != 1:
                continue
            nb = by_cell.get((r + 1, cx, cy))
            if not nb or len(nb) != 1:
                continue
            (c1, x1, y1) = lst[0]
            (c2, x2, y2) = nb[0]
            if (x1 - x2) ** 2 + (y1 - y2) ** 2 <= tol2 and abs(c2 - c1) <= _MAX_SHIFT:
                diffs.append(c2 - c1)
        shifts[r] = int(np.median(diffs)) if len(diffs) >= _MIN_PAIRS else 0
    return shifts


def link_peaks(all_detections, n_rows, n_cols, link_tolerance=DEFAULT_LINK_TOLERANCE):
    """Link same-peak detections across bins, correcting per-scan serpentine skew.

    Same signature and return shape as ``gaussian.link_peaks`` (a list of features,
    each a list of member nodes ``(bin_key, peak_index, row, col, x, y, peak_dict)``)
    so it is a drop-in for the shape stage.
    """
    nodes = []
    by_cell = defaultdict(list)
    for bk, peaks in all_detections.items():
        r, c = int(bk.split("_")[0]), int(bk.split("_")[1])
        for pi, p in enumerate(peaks):
            x, y = p["x"], p["y"]
            by_cell[(r, int(x) // link_tolerance, int(y) // link_tolerance)].append((c, x, y))
            nodes.append((bk, pi, r, c, x, y, p))
    if not nodes:
        return []

    # 1) estimate + accumulate the per-row backlash shift
    shifts = estimate_row_shifts(by_cell, n_rows, link_tolerance)
    cum = np.zeros(max(n_rows, 1), dtype=int)
    for r in range(1, len(cum)):
        cum[r] = cum[r - 1] + shifts.get(r - 1, 0)

    # 2) adaptive cross-row residual window: widen only when a real skew exists
    nz = [abs(v) for v in shifts.values() if v != 0]
    skew = np.median(nz) if len(nz) >= max(3, 0.05 * n_rows) else 0
    xwin = 1 if skew < 1 else (2 if skew < 3 else 3)

    # 3) index by (row, corrected_col, detector-cell) and union neighbours
    idx = defaultdict(list)
    ccol = []
    for ni, (bk, pi, r, c, x, y, p) in enumerate(nodes):
        cc = c - int(cum[r]) if r < len(cum) else c
        ccol.append(cc)
        idx[(r, cc, int(x) // link_tolerance, int(y) // link_tolerance)].append(ni)

    parent = list(range(len(nodes)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    tol2 = link_tolerance * link_tolerance
    for ni, (bk, pi, r, c, x, y, p) in enumerate(nodes):
        cc = ccol[ni]
        cxx, cyy = int(x) // link_tolerance, int(y) // link_tolerance
        for dr in (-1, 0, 1):
            reach = xwin if dr != 0 else 1   # wider column reach only across rows
            for dcc in range(-reach, reach + 1):
                if dr == 0 and dcc == 0:
                    continue
                for ex in (-1, 0, 1):
                    for ey in (-1, 0, 1):
                        for nj in idx.get((r + dr, cc + dcc, cxx + ex, cyy + ey), []):
                            nx, ny = nodes[nj][4], nodes[nj][5]
                            if (x - nx) ** 2 + (y - ny) ** 2 <= tol2:
                                union(ni, nj)

    components = defaultdict(list)
    for ni in range(len(nodes)):
        components[find(ni)].append(ni)
    return [[nodes[i] for i in members] for members in components.values()]


# ── Convenience entry point (mirrors gaussian.run_shape_pipeline) ───
def run_shape_pipeline(peaks_by_bin, n_rows, n_cols, tth_map, degs, deg_labels,
                       link_tolerance=DEFAULT_LINK_TOLERANCE):
    """Skew-aware link, then keep gaussian-like shapes. Returns (kept, filtered)."""
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


# ── Standalone CLI ─────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Skew-aware shape finder (deskew link + gaussian filter)")
    ap.add_argument("--peaks", required=True, help="Saved *_peaks.json from 'xrd-app peaks'")
    ap.add_argument("--two-theta", dest="two_theta", default="tth.tiff")
    ap.add_argument("--grid-mapping", required=True, help="grid_mapping_*.json")
    ap.add_argument("--reflections", default="reflections.py")
    ap.add_argument("--output", default="feature_catalog.json")
    ap.add_argument("--link-tolerance", type=int, default=DEFAULT_LINK_TOLERANCE)
    args = ap.parse_args()

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
