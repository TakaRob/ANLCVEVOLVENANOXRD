# Tomorrow

## 1. Wire deskewing into the GUI
Make the deskew available as a real option in the app, not just CLI/scripts.

- Two implementations exist today:
  - **`deskew_peaks.py`** (repo root) — surefire position-CSV pre-filter. Re-grids peaks
    by true (X,Y) before `shapes`. Best for **1×1** (slice fraction 53%→34%). 1×1-only.
  - **`xrd_app/ShapeAlgorithms/gaussian_deskew.py`** — data-driven shape variant (no CSV,
    neutral on binned). Already selectable via `xrd-app shapes --algorithm gaussian_deskew`.
- GUI work:
  - The shape-algorithm dropdown should already list `gaussian_deskew` (it reads
    `ShapeAlgorithms/catalog.json`). **Verify it appears and runs** from the Shape/Programs tab.
  - Add a **"Deskew (use stage positions)" toggle** for the 1×1 case that runs the
    `deskew_peaks.py` logic before shapes. Per the "CLI is the engine" rule, first promote
    that script to a CLI command (e.g. `xrd-app deskew-peaks --scan … --bin-size 1
    --peak-algo …`) by moving its `regrid()` into `core/`, then wire the checkbox to call it.
  - Gray out / warn on the toggle for bin-size != 1 (binning already deskews; re-gridding a
    clean grid adds noise — see findings).

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
