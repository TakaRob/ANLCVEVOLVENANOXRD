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

## 2. Current methods

| `--deskew-method` | `coordinate_source` | Assignment | Use |
|---|---|---|---|
| `positions_xy` | `positions_xy` | Snap both axes from measured `(X,Y)` without clipping to a commanded raster. | Default at 1×1 and for irregular scans that still need a rectangular grid. |
| `faithful` | `positions_faithful` | Exact file-index rows; measured fast-axis columns on an approximately square display lattice. | Default at N≥2 for clean file-per-row scans. |
| `faithful_native` | `positions_faithful_native` | Exact file-index rows; measured fast-axis columns at native frame density. | Detection/recall work where native sampling matters more than square display pixels. |
| `commanded` | `file_per_row` | File index and within-file rank, serpentine-aware. | Commanded-raster comparison or when backlash makes encoder columns misleading. |

Use `xrd-app territory-grid` instead of forcing a rectangle when cells must follow
irregular physical neighborhoods. It partitions true `(X,Y)` positions and stores
physical adjacency for shape linking.

## 3. Why these methods

The acquisition layout gives exact row membership, while measured `(X,Y)` gives
physical placement. `faithful` and `faithful_native` combine those strengths for
file-per-row scans. `positions_xy` uses both measured axes and avoids clipping
outlying positions onto an edge cell, which is essential at 1×1 and for irregular
layouts. `commanded` deliberately uses within-row rank when the desired reference
is where the stage was told to go.

The backlash observation remains scientifically important: on Scan_0203 the
encoder Y traces for opposite serpentine directions diverge by up to ±33 columns
at nominally identical commanded positions. Treating that divergence as a rigid
row translation can throw adjacent parts of one grain tens of columns apart and
fragment a feature. The current methods therefore either use measured coordinates
as a complete lattice (`positions_xy`), preserve exact rows while choosing a
controlled fast-axis lattice (`faithful*`), or use commanded rank (`commanded`).

Binning sums N×N cells after assignment, so changing methods changes which frames
are summed and requires rebuilding the binned HDF5. The choice matters most at
1×1; coarser bins average much of the sub-cell backlash while trading spatial
resolution for SNR.

## 4. Usage

```bash
xrd-app grid --scan Scan_0203 --bin-size 1                 # auto: positions_xy
xrd-app grid --scan Scan_0203 --bin-size 3                 # auto: faithful
xrd-app grid --scan Scan_0203 --bin-size 3 --deskew-method faithful_native
xrd-app grid --scan Scan_0203 --bin-size 3 --deskew-method commanded
xrd-app territory-grid --scan Scan_0203 --target-size 9    # irregular cells
```

Each command writes `Metadata/<scan>/grid_mapping_NxN[_variant].h5`. See
`TERMINOLOGY.md` (§Coordinates) and `core/io.py` for the assignment logic.
