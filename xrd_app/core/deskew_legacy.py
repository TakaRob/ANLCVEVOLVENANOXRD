"""DEPRECATED de-skew variants — kept only for comparison, not for production use.

These were earlier attempts at the scan-coordinate assignment. They are retained
so their output (e.g. the "triangle" device maps) can be reproduced and compared
against the current default (``io.assign_grid_from_positions`` → ``file_per_row``,
which aligns columns by *commanded* position).

    Do not use these for new analysis.

Reproduce a legacy grid for comparison:

    xrd-app grid --scan Scan_0203 --bin-size 1 --deskew-method perrow_offset \\
        --output .../grid_mapping_1x1_perrow_offset.json

----------------------------------------------------------------------
perrow_offset (the "triangle" method)
----------------------------------------------------------------------
Rows come from the one-file-per-row layout; columns are the within-file rank
shifted by a *rigid per-row integer offset* derived from the real fast-axis
encoder position. The intent was to slide a physically-offset row to its true
columns (de-skew) with zero within-row merges.

Why it's deprecated: on Scan_0203 the even/odd-row position divergence it
"corrects" is serpentine **backlash** (an encoder artefact at the same commanded
position), not real geometry. Applying the offset throws a feature's consecutive
rows tens of columns apart (median |Δcol| between adjacent rows ~44 px » the 5 px
link tolerance), which **fragments** real features and renders the scan as a
parallelogram with triangular edges. Aligning by commanded rank
(``file_per_row``) avoids this. See ``[[project_xrd-coordinate-system]]``.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from . import io


def assign_grid_perrow_offset(frame_x, frame_y, frame_map,
                              log: Callable[[str], None] = print):
    """DEPRECATED. File-per-row rows + rigid per-row integer column offset from
    real (X, Y). Kept for comparison only — see module docstring. Returns
    ``(grid_row, grid_col, n_rows, n_cols)``."""
    n_total = len(frame_x)
    x = io._interp_nan(frame_x)
    y = io._interp_nan(frame_y)

    ref_row, ref_col, n_rows, n_cols = io.build_grid_from_frame_map(
        frame_map, log=lambda *a: None)
    row_is_X = abs(io._corr(ref_row, x)) >= abs(io._corr(ref_row, y))
    rowv, colv = (x, y) if row_is_X else (y, x)
    sc = np.sign(io._corr(ref_col, colv)) or 1.0
    scv = sc * colv

    # Column step = median within-row nearest-neighbour spacing.
    within = [np.diff(np.sort(scv[ref_row == r]))
              for r in np.unique(ref_row) if (ref_row == r).sum() > 1]
    cstep = float(np.median(np.concatenate(within))) if within else 1.0
    if cstep <= 0:
        cstep = 1.0
    gcoord = (scv - scv.min()) / cstep

    grid_col = ref_col.astype(int).copy()
    for r in np.unique(ref_row):
        m = ref_row == r
        grid_col[m] = ref_col[m] + int(round(float(np.median(gcoord[m] - ref_col[m]))))
    grid_row = ref_row.astype(int)

    # Orient to the serpentine reconstruction's signs (historical device-map
    # orientation), so this differs from the current default only in the per-row
    # offset, not in mirroring.
    serp_row, serp_col, _, _ = io.build_scan_grid(frame_x, n_total)
    if io._corr(grid_row, rowv) * io._corr(serp_row, rowv) < 0:
        grid_row = grid_row.max() - grid_row
    if io._corr(grid_col, colv) * io._corr(serp_col, colv) < 0:
        grid_col = grid_col.max() - grid_col
    grid_row -= grid_row.min()
    grid_col -= grid_col.min()
    n_rows = int(grid_row.max()) + 1
    n_cols = int(grid_col.max()) + 1
    log(f"  [DEPRECATED perrow_offset] file-per-row rows × per-row-offset columns "
        f"-> {n_rows} x {n_cols}")
    return grid_row, grid_col, n_rows, n_cols
