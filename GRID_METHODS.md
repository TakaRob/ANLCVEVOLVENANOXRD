# Scan-grid coordinate methods (nano-XRD, Scan_0203)

How each raw frame gets a `(row, col)` scan-grid position, the methods we tried,
what each produces, and what it means for binned levels. All numbers below are
from **Scan_0203** (the reference scan).

---

## 1. What we start from (raw)

Per scan, on the beamline mount (`/mnt/z/.../Raw/Scan_NNNN/`):

| Stream | What it is | Use for the grid |
|---|---|---|
| `XRD/scan_NNNN_*.h5` | The detector frames. **One HDF5 file per scan row.** 0203 = 151 files × 167 frames (last file 120) = **25,170 frames**. | The file/frame layout *is* the commanded raster. |
| `Processed/SOCKETSERVER/Scan_NNNN_position.csv` | Per-frame stage encoder `Trigger, X_Position, Y_Position` (µm). 0203: X 434–446, Y 760–771. | True per-frame position (what the encoder *read*). |
| `TETRAMM1/` | Picoammeter / slow-stage meta. Only ~166 coarse samples per scan. | **Too coarse — cannot position frames.** Not used. |

**Axis convention (0203):** the slow axis is **X** (one value per row), the fast
sweep axis is **Y** (167 steps within a row). So `row ↔ X`, `col ↔ Y`.

**The core problem.** The encoder Y of the two serpentine directions (even vs odd
rows) **diverges**, growing across the scan to **±33 columns** by the ends
(even-row mean-Y drifts down, odd-row up). This is **stage backlash** — an encoder
artifact at the *same commanded position* — not real sample geometry. How each
method handles (or mishandles) this is what separates them.

---

## 2. The methods at a glance

| Method | `coordinate_source` | rows | cols | 0203 grid | merges* | Status |
|---|---|---|---|---|---|---|
| Serpentine (X-only) | `serpentine` | turn-count of X trace | within-segment index | 156×221 | 0 | legacy / `--rawgrid` |
| Global-scale de-skew | `positions_xy` (old) | scale of X | scale of Y (clipped) | 156×221 | ~3,358 | **superseded** |
| Per-row offset ("triangle") | `perrow_offset_deprecated` | file index | rank + encoder offset | 151×235 | 0 | **DEPRECATED** |
| **File-per-row (current)** | `file_per_row` | file index | within-file **rank** | **151×167** | **0** | **default** |
| Synthetic | `synthetic` | `i // n_cols` | boustrophedon | from `--shape` | 0 | no-CSV fallback |

\* "merges" = frames forced to share a cell (lost spatial resolution).

---

## 3. Each method, from raw

### 3.1 Serpentine, X-only (`build_scan_grid`) — *legacy*

**How:** ignore Y; smooth the X trace and count turning points to segment rows,
number columns within each segment (reverse alternate rows).

**Output (0203):** 156×221. `corr(row, X) = −1.0` (good), but **`corr(col, Y) =
0.585`** — columns are badly misregistered between adjacent rows. The "whisker"
(one over-long row) and encoder noise create *spurious* turns, inflating the row
count (156 vs the true 151) and the column count (221 vs 167).

**Device map:** a single Bragg feature **fragments into horizontal slices**; the
1×1 single-row-feature fraction is ~53%.

**Verdict:** the original skew source. Kept only as the `--rawgrid` bypass and as a
sign reference for orienting the newer methods.

---

### 3.2 Global-scale de-skew (old `positions_xy`) — *superseded*

**How:** snap each frame's real `(X, Y)` onto a 156×221 lattice (sized by the
serpentine turn-counter) via percentile min→max scaling, with edge clipping.

**Output (0203):** 156×221, but **~3,358 of 25,170 frames merge** into shared
cells (≈13% resolution loss). `corr(col, Y) = 1.0` *by construction* (col is
derived from Y).

**Device map:** looks de-skewed (features re-merge), **but** the percentile
clipping piles all outlier-Y frames onto the **edge column (col 220)** — e.g. the
ubiquitous substrate (002) spot at detector (250,414) appears as one inflated
"hot" blob (353 bins, I≈6065) at the right edge. That hot edge peak is largely a
**clipping artifact**, not a real localized grain.

**Verdict:** corrects the slice fragmentation but at the cost of merges *and* a
false edge peak. Superseded. Preserved on disk as `*_preHybrid156x221`.

---

### 3.3 Per-row offset, the "triangle" (`perrow_offset`) — *DEPRECATED*

**How:** rows from the file layout (exact); columns = within-file rank **plus a
rigid per-row integer offset derived from the real encoder Y**, intended to slide
a physically-offset row to its true columns. Lives in `core/deskew_legacy.py`.

**Output (0203):** 151×235, **0 merges**, `corr(col, Y) = 0.99`.

**Device map:** a **parallelogram with triangular edges** — because it "corrects"
the ±33-col even/odd-row divergence. But that divergence is **backlash**, so the
correction throws a feature's *adjacent* rows tens of columns apart (median
|Δcol| between adjacent rows ≈ **44 px**, far beyond the 5 px link tolerance),
which **fragments** real features (e.g. (002) goes from one feature to ~8). It
also shipped with a column-orientation **reflection bug** (mirror + apparent
peak relocation).

**Verdict:** over-corrects backlash → fragmentation + triangular distortion.
**Deprecated.** Reproduce for comparison only:
`xrd-app grid --deskew-method perrow_offset ...`. Preserved as `*_perrowOffset151x235`.

---

### 3.4 File-per-row (`file_per_row`) — *current default*

**How:** the acquisition layout already *is* the commanded raster, so:
`row = HDF5 file index`, `col = within-file rank` (the position the stage was
*told* to go to), serpentine-aware. Real `(X, Y)`, when present, are used **only
to orient the axes** (anchored to the serpentine sign so the device map isn't
mirrored) — **columns are not re-snapped to the encoder**, which is what avoids
amplifying backlash.

**Output (0203):** **151×167** (the true raster shape), **0 merges**, one cell per
frame. `corr(col, Y) = −0.92` — and the residual 0.08 *is* the uncorrected
backlash, left in deliberately (correcting it would fragment, see 3.3).

**Device map:** compact and rectangular; features stay intact (no slice
fragmentation, no edge pile, no triangle). Needs no positions to work — if the CSV
is missing it is reconstructed from the file layout (`xrd-app recreate-positions`).

**Verdict:** aligns by commanded position → the cleanest, most faithful grid.
**Use this.**

---

## 4. Side-by-side (Scan_0203, the (002) hot region)

| | Serpentine | Global-scale (old) | Per-row offset | **File-per-row** |
|---|---|---|---|---|
| grid (1×1) | 156×221 | 156×221 | 151×235 | **151×167** |
| frames merged | 0 | ~3,358 | 0 | **0** |
| `corr(col,Y)` | 0.585 (skewed) | 1.0 (clipped) | 0.99 | −0.92 (backlash left in) |
| feature integrity | slices | merged + edge pile | **fragments** | **intact** |
| backlash handling | none | hidden by clipping | over-corrected | left as-is (correct) |
| artifact | horizontal slices | false hot edge col | triangle + reflection | none |

---

## 5. Implications for binned sizes (2×2 … 5×5)

A binned frame sums an N×N block of the grid. The grid *dimensions* and *which
frames get summed together* therefore depend on the method — but the differences
shrink with N:

- **The coordinate choice matters most at 1×1.** Slice fragmentation (serpentine),
  the edge-clip pile (global-scale), and backlash fragmentation (per-row offset)
  are all 1×1 phenomena. At 2×2 and up, summing N² neighbouring frames **averages
  the ±sub-cell backlash out**, so the grid is already effectively faithful and the
  methods nearly converge. (This is also why the old binned study "looked fine"
  even on a flawed 1×1 grid.)

- **Binned grid dimensions** (0203), `n_bin_rows × n_bin_cols`:

  | bin | file_per_row | per-row offset | global-scale (old) |
  |---|---|---|---|
  | 1×1 | 151×167 | 151×235 | 156×221 |
  | 2×2 | 76×84 | 76×118 | 78×111 |
  | 3×3 | 51×56 | 51×79 | 52×74 |
  | 4×4 | 38×42 | 38×59 | 39×56 |
  | 5×5 | 31×34 | 31×47 | 32×45 |

  `file_per_row` is the **narrowest** (no spurious column inflation), so its binned
  maps are **denser** (fewer empty cells) and a fixed physical region maps to fewer,
  fuller bins — a cleaner device map at every N.

- **Correct grouping at edges.** Binning on the *commanded* grid sums
  physically-adjacent frames. On the skewed/offset grids, the N×N blocks straddle
  the wrong neighbours near the edges and the divergence band — a small but real
  error that the wrong-lattice binning bakes in.

- **Re-binning is required to switch methods.** Because the binned *grouping*
  differs (`col // N` differs between lattices), you cannot relabel an existing
  binned HDF5 from one method to another — the bins must be rebuilt from raw. (1×1
  is the exception: one frame per cell, so only the labels change.)

- **Recommendation for the sweep:** generate every bin size on `file_per_row`. The
  sweet-spot analysis (recall vs resolution; ≈3×3) is unaffected by the coordinate
  fix — the fix changes *where* features sit, not the binning-SNR tradeoff — but
  the maps are cleanest and the 1×1 baseline is finally trustworthy.

---

## 6. Provenance / reproducing each

- Current default: `xrd-app grid --scan Scan_0203 --bin-size N` → `file_per_row`.
- Triangle (comparison): `xrd-app grid --bin-size N --deskew-method perrow_offset`.
- Serpentine: `xrd-app grid --bin-size N --rawgrid`.
- On-disk comparison sets (TakaProject/Labels & Binned): `*_preHybrid156x221`
  (global-scale), `*_perrowOffset151x235` (triangle), `*_rawgrid` (serpentine);
  standard names = `file_per_row`.

See `TERMINOLOGY.md` (§Coordinates) and `core/io.py::assign_grid_from_positions`
+ `core/deskew_legacy.py`.
