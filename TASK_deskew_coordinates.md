# TASK: De-skew at the source — define the scan coordinate system from true (X, Y)

**Goal:** make the per-frame `(row, col)` scan-grid assignment come from the **true
stage positions** (both `X_Position` and `Y_Position`) snapped to a regular lattice,
computed **once** in grid generation — so binning, peak bin-keys, cross-bin linking,
device maps, and the aggregate CSV all inherit a faithful, de-skewed coordinate system
with **no changes to those downstream modules**. This promotes the `deskew_peaks.py`
prototype from an after-the-fact band-aid into the canonical coordinate definition.

## Why (the insight that motivates this)
Today the grid is reconstructed serpentine-style from **X only** (`load_positions`
reads just `X_Position`; `build_scan_grid` counts turns). The serpentine reversal +
stage backlash misregister the *column* axis between adjacent rows → a single feature
fragments into horizontal slices (1×1 single-row fraction 53%; `corr(col, Y)=0.585`).

De-skewing **after** detection (the `deskew_peaks.py` pre-filter / `gaussian_deskew`
variant) only re-quantizes an already-discrete grid: it helps 1×1 (broken grid → real
fix, 53%→34%) but *hurts* binned levels (re-rounding an already-correct lattice injects
noise: 2×2 310→422 kept, 3×3 213→328). The correct place is **before binning**: fix the
frame-level grid once from true positions, then bin on the corrected grid. Then 1×1 is
native-correct AND binned stays exact (no re-quantization), and both band-aids retire.

## Single source of truth to change — `xrd_app/core/io.py`
1. **Read Y too.** Add `load_positions_xy(csv_path, n_total) -> (frame_x, frame_y)`
   reading both `X_Position` and `Y_Position` (keep `load_positions` for back-compat).
2. **New assignment** `assign_grid_from_positions(frame_x, frame_y) -> (grid_row,
   grid_col, n_rows, n_cols)`: snap each frame to a regular lattice from true (X, Y).
   Port the validated logic from `deskew_peaks.py::regrid()`:
   - auto-detect which physical axis is row vs col (and sign) so it adapts per scan;
   - snap by median step (robust to the "whisker"/irregular scans), not by serpentine
     turn-counting; collisions (oversampled frames → same cell) merge naturally.
3. **Wire it in** `generate_grid_mapping(...)`: when Y positions are available, use
   `assign_grid_from_positions`; otherwise fall back to the existing `build_scan_grid`
   (no-Y) / `build_regular_grid` (no-CSV) paths. Record the choice in the output JSON,
   e.g. `"coordinate_source": "positions_xy" | "serpentine" | "synthetic"`.
4. **Leave `build_bin_mapping` unchanged** — it bins on whatever `(row, col)` it is
   given, so it now groups physically-adjacent frames automatically. This is the whole
   point: the fix lives in coordinate assignment, not in binning or linking.

## Axis-convention decision (must be explicit)
`TERMINOLOGY.md` says **Row = scan y, Col = scan x**, but the *current* serpentine grid
has `row ↔ X` (`corr(row0, X) = −1.0`). Choosing positions-based assignment forces a
decision: either (a) adopt the documented convention (`col ← X_Position`, `row ←
Y_Position`) — which transposes vs today — or (b) preserve the current orientation.
Pick one, document it in `TERMINOLOGY.md`, and make the device-map / viewer axis labels
(`gui/device_map.py`, `gui/viewer.py`) match. Decide by what keeps the device map
physically upright; verify against a known feature's real sample location.

## Downstream — verify unaffected (no code changes expected)
These all consume `(row, col)` / `n_bin_rows,n_bin_cols` from the grid mapping and should
inherit the new coordinates transparently. Re-run and visually confirm, don't edit:
`core/processing.py` (run_peaks/run_shapes), `ShapeAlgorithms/gaussian.py` (bin-key
linking), `gui/device_map.py` (`build_device_grids`), `gui/viewer.py`, `gui/labeling.py`,
`core/holdout.py`, `aggregate.py`, `binning_benefits_report.py`, `scan_203_skew_map.py`,
`core/io.py::_RawSource`/`open_bin_source`.

## Migration / back-compat
- Renaming nothing on disk; only the *values* of `(row, col)` change. **Existing
  `grid_mapping_*.json`, `*_peaks_*.json`, feature catalogs, and bin HDF5s must be
  regenerated** (`xrd-app make-bins` / `grid` / `peaks` / `shapes` per scan) — their
  bin keys are now on the old grid. Loaders should tolerate a missing
  `coordinate_source` key (treat as legacy/serpentine).
- Keep `build_scan_grid` (X-only) as the documented fallback when a scan has no
  `Y_Position` column.

## Acceptance criteria (concrete, from this session's experiments)
- **1×1**: with plain `xrd-app grid → peaks → shapes --algorithm gaussian` (no
  pre-filter, no `gaussian_deskew`), single-row slice fraction drops **53% → ~34%** and
  kept ≈ 1655 (slices merged). The win comes purely from the new coordinates.
- **2×2 / 3×3**: within noise of today's standard (**NOT worse** — must not show the
  re-quantization regression): kept ≈ 310 / 213, slice ≈ 29% / 34%.
- **Physics check** (`CLAUDE.md`): detected peaks still fall inside the expected 2θ
  bands; device maps render upright with correct, convention-matched axis labels.
- `coordinate_source: positions_xy` present in regenerated grid mappings.

## Cleanup once this lands (retire the band-aids)
- `deskew_peaks.py` (repo root) → delete or convert to a thin no-op shim; its logic now
  lives in `assign_grid_from_positions`.
- `ShapeAlgorithms/gaussian_deskew.py` + its `catalog.json` entry → remove, OR keep only
  as the fallback for scans with no position CSV (note that in its docstring).
- Update `binning_benefits_report.py` D4 (deskew comparison) to reflect that deskew is now
  built into the coordinates, not an optional step.

## Risks / edge cases
- Irregular scans (the Scan_0203 "whisker"): snapping must place outliers at their true
  cell without compressing the rest — use robust per-axis step (median NN gap), not raw
  min/max range. Validate on Scan_0203.
- Collisions at 1×1 (~3.3k of 25k frames share a cell): expected and fine (merge frames
  into the cell). Confirm n_bins shrinks ~25,170 → ~21,800 and downstream handles it.
- Anisotropic X/Y step (X-step ≪ Y-step here) → estimate row-step and col-step
  independently.
