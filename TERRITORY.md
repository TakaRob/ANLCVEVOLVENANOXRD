# TERRITORY.md — the territorial (cell-model) source of truth

The **territorial model** is a skew-free *reference* binning used to optimize the
fast 1×1 post-processing (skew / backlash fixing) against. It is not the
production path — it is the ruler the production path is measured with.

Pairs with `CLAUDE.md` (science), `xrd_app/CLAUDE.md` (app architecture), and
`ANALYSIS_PLAYBOOK.md` (driving the CLI from chat).

## Why it exists

The N×N grid is built by *reconstructing* a serpentine lattice from the scan
(`xrd_app/core/io.assign_grid_from_positions`). On these scans the even/odd rows
diverge by stage **backlash**, so the reconstruction injects a horizontal /
angular **skew**: a single grain can land a row away at a shifted column, its
vertical link gets dropped, and the feature fragments into horizontal slices.

Any "ground truth" built on that grid inherits the same skew. The territorial
model sidesteps it: it bins by **true (X, Y) stage positions** (from the position
CSV), which never pass through the serpentine reconstruction. So its shapes are
the skew-free reference the fast 1×1 catalog should converge toward once its skew
fix is correct.

## What it does (the model)

- Frames are points at their true `(X, Y)`. A **Delaunay** graph gives physical
  neighbors.
- Greedy **region-growing**: a seed frame absorbs its nearest unassigned
  neighbors until the **territory** reaches a target frame count. This is a clean
  **partition** — every frame is in exactly one territory.
- Variable frames-per-territory is the point: SNR scales with frame count, so
  denser regions get tighter territories (finer resolution), sparser regions get
  more averaging — **adaptive SNR**, unlike a rigid grid.
- Each territory carries a **centroid, area, frame count, polygon footprint, and
  physical neighbor list**. Shapes are linked across those *physical* neighbors
  (not a fixed N×N 8-neighborhood); detection + characterization are the same as
  the baseline (a territory is just an irregular bin).

Code: `xrd_app/core/territory.py` (partition),
`xrd_app/ShapeAlgorithms/territory.py` (linker),
`xrd_app/gui/territory_map.py` + `tabs/territory.py` (to-scale polygon viewer).

## Requirements & caveats

- **Needs a real position CSV** with `X_Position`/`Y_Position`. It *errors* rather
  than falling back to a recreated lattice — a truth built on the recreated grid
  would carry the very skew it is meant to measure. Scans with only
  SOCKETSERVER-derived / recreated CSVs can't be used as truth.
- **Resolution parity.** Bigger `--target-size` = fatter territories = higher SNR
  but coarser resolution. Keep it near the 1×1 feature scale (start ~9; use ~4
  for near-1×1) so the truth doesn't merge features the 1×1 grid would resolve
  separately. When in doubt, sweep `4 → 9 → 16` and confirm conclusions hold.
- **Nominal `bin_size` is 1×1**; the `territories` block (not the bin size) is
  what makes it territorial. All artifacts use the `territory` variant tag, so
  they sit alongside the grid ones.

## The 1×1 production path is gridless coordinate linking

The skew is a *grid* artefact, so the production 1×1 path **doesn't build a grid
for linking**. `xrd-app shapes --bin-size 1` now **defaults** to gridless
coordinate linking: it keeps the standard 1×1 detection/peaks but links peaks
across true (X, Y) physical neighbors (Delaunay) instead of grid adjacency. No
deskew step is needed — there is no grid to skew.

```bash
xrd-app grid   --bin-size 1                 # standard 1×1 grid + peaks (unchanged)
xrd-app bin    --bin-size 1
xrd-app peaks  --bin-size 1
xrd-app shapes --bin-size 1                  # gridless coordinate linking (DEFAULT) → *_coord shapes
xrd-app shapes --bin-size 1 --grid-link      # force the OLD grid linking (for comparison)
```

Binned sizes (≥2×2) keep grid linking by default (backlash is averaged out
there). Coordinate mode needs the real position CSV; without one it degrades to
grid linking with a note. The grid-repair deskew options
(`grid --deskew-method faithful/perrow_offset`, `shapes --algorithm
gaussian_deskew`) remain in the CLI for comparison but are no longer the default
— see `deprecated/README.md` for why.

## Build the truth (per scan, once)

```bash
xrd-app territory-grid --target-size 9                                   # true-(X,Y) partition
xrd-app bin    --bin-size 1 --variant territory                         # sum frames per territory
xrd-app peaks  --bin-size 1 --variant territory                        # per-territory detection
xrd-app shapes --bin-size 1 --variant territory --algorithm territory  # link over physical neighbors
```

Artifacts written (the `_territory` tag keeps them beside the grid results):

```
Metadata/<scan>/grid_mapping_1x1_territory.json   # territories: centroid/area/count/neighbors/polygon
Binned/<scan>/xrd_1x1_bins_territory.h5
Labels/<scan>/<det>_peaks_1x1_territory.json
Labels/<scan>/territory_shapes_1x1_territory.json # ← the source-of-truth catalog
```

## Compare the fast 1×1 catalog to the truth

`compare_to_truth.py` (repo root) is a standalone script (not a CLI command —
run it rarely). Edit its **CONFIG** cell (`PROJECT_ROOT`, `SCAN`, `FAST_SHAPES`),
then `python3 compare_to_truth.py`.

It projects **both** catalogs into a common true-(X,Y) + detector-pixel space at
full 1×1 resolution (this is how resolution is "matched"), matches greedily by
reflection + detector proximity, and prints per fast catalog:

| Field | Meaning |
|---|---|
| `recall` | fraction of truth shapes the fast catalog recovered — **the primary metric** (recall-first, per `CLAUDE.md`) |
| `prec` | fraction of fast shapes that matched a truth shape |
| `F2` | recall-weighted F-score, `5PR/(4P+R)` — secondary metric |
| `offset(X,Y)` / `|·|` | mean `fast − truth` position of matched pairs — the **residual skew** still present; drive it toward 0 |

Put several catalogs in `FAST_SHAPES` (e.g. current vs skew-fixed) to get one row
each, side by side.

## Optimization loop (for the skew-fix work)

1. Build the truth once (above).
2. Generate the 1×1 shapes **without** and **with** the skew fix; list both in
   `FAST_SHAPES`; record baseline recall / F2 / |offset|.
3. Change **one** thing in the skew fix → regenerate 1×1 shapes → re-run the
   comparison.
4. Keep the change only if **recall and F2 rise AND |offset| holds or drops**,
   with no 2-θ band regressions (physics gate). Stop after 1–2 accepted wins.

**Don't tune the ruler to flatter the result:** `core/territory.py`, the
territorial linker, and `compare_to_truth.py`'s matching logic define the truth —
leave them alone while optimizing the fast path. Use the same scan and the same
`--target-size` for every comparison, and confirm the final number at full
settings, not a dev subset.
