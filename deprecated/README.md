# deprecated/

Superseded code, kept for reference. Nothing here is on the active path.

## Why

The serpentine/backlash **skew is a grid artefact** — snapping a frame to a
(row, col) lattice turns a sub-step position error into a wrong-column split that
fragments grains. The fix is to **not build a grid** at 1×1: detect per frame and
link peaks by true (X, Y) physical neighbors. That is now the **default** for
`xrd-app shapes --bin-size 1` (gridless coordinate linking; pass `--grid-link`
to force the old grid linking). See `TERRITORY.md`.

## Contents

- `coord_link_1x1.py` — the original one-off proof that coordinate linking
  reproduces the best grid-repair result while eliminating the skew by
  construction. **Superseded by** `xrd-app shapes --coordinate` (now the 1×1
  default), which reuses the standard peaks and only changes the linking stage.

## Still live in the CLI (NOT deprecated)

The grid-repair deskew options remain available in the CLI for comparison — they
are intentionally kept, just no longer the default:

- `xrd-app grid --deskew-method faithful` (square-pixel re-grid, true-Y columns)
  and `--deskew-method perrow_offset` (legacy "triangle", in
  `xrd_app/core/deskew_legacy.py`).
- `xrd-app shapes --algorithm gaussian_deskew` — the linker-stage column-shift
  fix (`xrd_app/ShapeAlgorithms/gaussian_deskew.py`). Note: a linker fix cannot
  move a shape's `center_bin`, so it cannot actually de-skew positions; kept only
  for the record.

The experimental deskew **data** catalogs (`*_perrowOffset*`, `*_preHybrid*`,
`*_faithful*`, `*_deskew*` peaks/shapes in `Labels/`) were left in place.
