"""High-def (1×1) intensity sampled beneath a binned feature map.

The binned Device View shows one coarse value per N×N bin. This module drops to
true 1×1 resolution *inside* each already-found feature: for every N×N bin in a
feature's footprint it expands to the nine (``bin_size²``) raw 1×1 cells
(:func:`io.subbin_keys`) and reads that single raw frame, taking the max / sum in
a small window around the feature's *detector peak* — i.e. the simple 1×1
intensity at the detected Bragg peak at each spatial pixel.

The result is the 1×1 analog of a feature's ``intensity_profile`` (a
``hd_profile`` keyed by 1×1 cell), optionally carrying each cell's true stage
position ``(x, y)`` so a GUI can plot in real-position space instead of the grid.

Pure module — no PyQt, no click. This is the whole-device generalization of the
per-feature raw sampling in ``gui/viewer.py`` (``_RawIntensityWorker``), moved to
``core`` so the CLI (``xrd-app hd-device-map``) and the GUI share one code path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np

from . import io


def build_cell_xy(grid_mapping: dict, positions_csv=None, archive=None) -> dict:
    """``{cell_key: (x, y)}`` real stage position for each 1×1 cell.

    ``grid_mapping`` is the loaded 1×1 grid-mapping dict (``bins`` maps a cell
    ``"r_c"`` → its raw frame indices; ``n_total_frames`` sizes the position
    arrays). Positions come from ``positions_csv`` via
    :func:`io.load_positions_xy`; each cell's ``(x, y)`` is the mean over its
    frames. Cells with no finite position are dropped. Returns ``{}`` when the CSV
    is missing or carries no usable (real) positions — the caller then omits the
    real-position layer.
    """
    bins = grid_mapping.get("bins") or {}
    have_positions = positions_csv and Path(positions_csv).exists()
    have_archive_positions = archive and io.archive_has_real_positions(archive)
    if not bins or not (have_positions or have_archive_positions):
        return {}
    n_total = grid_mapping.get("n_total_frames")
    if not n_total:
        n_total = 1 + max((max(v) for v in bins.values() if v), default=-1)
    if n_total <= 0:
        return {}
    if have_positions:
        frame_x, frame_y = io.load_positions_xy(positions_csv, n_total)
    else:
        frame_x, frame_y = io.archive_positions(archive)
    if not (np.isfinite(frame_x).any() and np.isfinite(frame_y).any()):
        return {}  # X-only / no real positions → no scatter layer

    out = {}
    for cell, frames in bins.items():
        idx = [i for i in frames if 0 <= i < n_total]
        if not idx:
            continue
        xs, ys = frame_x[idx], frame_y[idx]
        if not (np.isfinite(xs).any() and np.isfinite(ys).any()):
            continue
        out[cell] = (float(np.nanmean(xs)), float(np.nanmean(ys)))
    return out


def scan_trajectory(grid_mapping: dict, positions_csv=None, archive=None) -> dict:
    """Acquisition-order scan path through the 1×1 cells.

    The beam visits one frame per 1×1 cell in global frame order (the serpentine
    raster); connecting those cells in order traces the physical scan trajectory,
    which threads through the pixels inside each feature. Returns::

        {"grid": [[col, row], ...],   # plot-space (x=col, y=row), grid mode
         "xy":   [[x, y], ...] | None}  # real stage positions, xy mode (None if
                                        # no real position CSV)

    Both lists are in acquisition order. ``grid_mapping`` is the loaded 1×1
    grid-mapping dict (``bins`` = cell → frame indices).
    """
    bins = grid_mapping.get("bins") or {}
    if not bins:
        return {"grid": [], "xy": None}
    n_total = grid_mapping.get("n_total_frames")
    if not n_total:
        n_total = 1 + max((max(v) for v in bins.values() if v), default=-1)

    frame_rc = [None] * max(n_total, 0)
    for cell, frames in bins.items():
        r, c = _parse_cell_key(cell)
        for gi in frames:
            if 0 <= gi < n_total:
                frame_rc[gi] = (r, c)
    grid_path = [[c, r] for rc in frame_rc if rc is not None for (r, c) in (rc,)]

    xy_path = None
    if positions_csv and Path(positions_csv).exists():
        frame_x, frame_y = io.load_positions_xy(positions_csv, n_total)
    elif archive and io.archive_has_real_positions(archive):
        frame_x, frame_y = io.archive_positions(archive)
    else:
        frame_x = frame_y = np.array([])
    if np.isfinite(frame_x).any() and np.isfinite(frame_y).any():
        xy_path = [[float(frame_x[gi]), float(frame_y[gi])]
                   for gi in range(min(n_total, len(frame_x)))
                   if np.isfinite(frame_x[gi]) and np.isfinite(frame_y[gi])]
    return {"grid": grid_path, "xy": xy_path}


def _parse_cell_key(key):
    r, c = key.split("_")
    return int(r), int(c)


def _footprint_cells(feat: dict, bin_size: int) -> list:
    """Ordered, de-duplicated 1×1 cells under a binned feature's footprint."""
    keys = feat.get("spatial_extent") or list(feat.get("intensity_profile", {}))
    seen, cells = set(), []
    for bk in keys:
        for s in io.subbin_keys(bk, bin_size):
            if s not in seen:
                seen.add(s)
                cells.append(s)
    return cells


def sample_hd_intensity(
    features,
    source: "io.BinImageSource",
    bin_size: int,
    win: int = 4,
    cell_xy: Optional[dict] = None,
    max_cells_per_feature: Optional[int] = None,
    progress: Optional[Callable[[int, int], None]] = None,
    log: Callable[[str], None] = print,
) -> list:
    """Sample raw 1×1 intensity at each feature's detector peak.

    ``features`` are binned feature dicts (from a shapes/feature catalog);
    ``source`` is a **1×1** :class:`io.BinImageSource` (``open_bin_source(dm, 1,
    scan)``). For each feature, every 1×1 cell under its footprint is read and the
    ``win`` half-window around ``(detector_x, detector_y)`` is reduced to
    ``intensity`` (max) and ``integrated`` (sum). Missing frames are simply absent
    from ``hd_profile`` (they render as holes).

    ``cell_xy`` (from :func:`build_cell_xy`) attaches each cell's real ``(x, y)``.
    Returns a list of HD feature dicts::

        {feature_id, reflection, chi_deg, ref_tth, detector_x, detector_y,
         hd_profile: {"<r>_<c>": {intensity, integrated, [x, y]}, ...}}
    """
    cell_xy = cell_xy or {}
    out = []
    n = len(features)
    for fi, feat in enumerate(features):
        try:
            det_x = int(round(float(feat["detector_x"])))
            det_y = int(round(float(feat["detector_y"])))
        except (KeyError, TypeError, ValueError):
            if progress is not None:
                progress(fi + 1, n)
            continue

        cells = _footprint_cells(feat, bin_size)
        if max_cells_per_feature and len(cells) > max_cells_per_feature:
            log(f"  feature {feat.get('feature_id')}: capping "
                f"{len(cells)} cells → {max_cells_per_feature}")
            cells = cells[:max_cells_per_feature]

        hd = {}
        for cell in cells:
            # Read only the window around the detector peak (h5 slices from disk).
            patch = source.region(cell, det_y - win, det_y + win + 1,
                                  det_x - win, det_x + win + 1)
            if patch is None or patch.size == 0:
                continue  # missing / incomplete frame → hole
            entry = {"intensity": float(patch.max()),
                     "integrated": float(patch.sum())}
            xy = cell_xy.get(cell)
            if xy is not None:
                entry["x"], entry["y"] = xy
            hd[cell] = entry

        out.append({
            "feature_id": feat.get("feature_id"),
            "reflection": feat.get("reflection", "unknown"),
            "chi_deg": feat.get("chi_deg"),
            "ref_tth": feat.get("ref_tth"),
            "detector_x": det_x,
            "detector_y": det_y,
            "hd_profile": hd,
        })
        if progress is not None:
            progress(fi + 1, n)
    return out


def summarize(hd_features) -> dict:
    """Counts for the CLI physics-check line: features, sampled cells, empties."""
    n_cells = sum(len(f["hd_profile"]) for f in hd_features)
    n_empty = sum(1 for f in hd_features if not f["hd_profile"])
    n_pos = sum(1 for f in hd_features for e in f["hd_profile"].values() if "x" in e)
    return {
        "n_features": len(hd_features),
        "n_cells": n_cells,
        "n_features_empty": n_empty,
        "n_cells_with_position": n_pos,
    }
