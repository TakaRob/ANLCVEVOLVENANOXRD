# Tomorrow

## 1. PRIMARY — De-skew at the source (redefine the coordinate system)
See **`TASK_deskew_coordinates.md`** for the full spec. In short: assign per-frame
`(row, col)` from true `(X, Y)` in `core/io.py` grid generation (promote
`deskew_peaks.py::regrid()` into `assign_grid_from_positions`), so binning, linking,
device maps, and aggregate all inherit the de-skewed coordinates with no downstream
edits. Target: 1×1 slice 53%→34% from plain `grid→peaks→shapes`, binned unchanged.
This retires the two band-aids below once it lands:
  - `deskew_peaks.py` (repo-root position-CSV pre-filter, 1×1-only).
  - `xrd_app/ShapeAlgorithms/gaussian_deskew.py` (data-driven shape variant).

GUI follow-on (after the coordinate fix): the de-skew is then automatic, so the GUI just
needs to use the corrected grid mappings — no special toggle. (If you ship before the
coordinate fix, the interim path is a `gaussian_deskew` dropdown entry + a 1×1-only
"Deskew (stage positions)" checkbox that shells out to the pre-filter.)

## 2. Look back over the binning benefits report
Re-read `binning_benefits_report.py` end to end now that the data + findings settled.

- Re-run it cell-by-cell against TakaProject (all 5 bin sizes are built/cached).
- Sanity-check the conclusions still read correctly:
  - counts (peaks & features) **decrease** with bin size; SNR **rises** at real detections
    (Sec E2); 1×1 over-segments into horizontal slices (Sec D3); D4 deskew comparison.
  - Confirm the loader still reads `gaussian_shapes_*.json["kept"]` (canonical) and that the
    1×1 catalog is the standard gaussian (1825), not a deskewed leftover.
- Decide the headline: sweet spot ≈ **3×3** (SNR plateau, no fragmentation, resolution kept).
- Possible polish: turn D4 into a before/after **1×1 device-map** A/B (run the pre-filter,
  show slices merging) for a stronger visual.

Context + numbers are in memory: `project_binning-findings.md`.
