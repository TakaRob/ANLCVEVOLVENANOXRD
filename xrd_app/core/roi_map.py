"""Detector ROI intensity mapped across a scan's spatial bins.

This is the headless engine for the ROI > Shape view. A fixed rectangular
region in detector coordinates is reduced independently for every spatial bin.
The exact ROI and all basic reductions are retained so a saved map can be
replotted with a different metric without rereading detector frames.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np


METRICS = ("integrated", "intensity", "mean")


def normalize_roi(roi) -> tuple[int, int, int, int]:
    """Return ordered integer detector bounds as ``(x0, y0, x1, y1)``."""
    if len(roi) != 4:
        raise ValueError("ROI must contain x0, y0, x1, y1")
    x0, y0, x1, y1 = (int(round(float(v))) for v in roi)
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    x0, y0 = max(0, x0), max(0, y0)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("ROI must have positive width and height")
    return x0, y0, x1, y1


def _parse_key(key):
    try:
        row, col = key.split("_", 1)
        return int(row), int(col)
    except (AttributeError, TypeError, ValueError):
        return None


def sample_roi(
    source,
    roi,
    *,
    grid_mapping: Optional[dict] = None,
    metric: str = "integrated",
    normalize_frames: bool = False,
    progress: Optional[Callable[[int, int], None]] = None,
) -> dict:
    """Reduce one detector rectangle over every spatial bin in ``source``.

    Detector coordinates are supplied as ``(x0, y0, x1, y1)`` and read from the
    source as ``[y0:y1, x0:x1]``. Missing bins remain absent from ``profile`` so
    they render as holes rather than false zero intensity. When
    ``normalize_frames`` is true, each metric is divided by the number of raw
    frames represented by that spatial bin.
    """
    if metric not in METRICS:
        raise ValueError(f"Unknown metric {metric!r}; choose from {METRICS}")
    x0, y0, x1, y1 = normalize_roi(roi)
    mapping_bins = (grid_mapping or {}).get("bins") or {}
    keys = list(source.keys())
    profile = {}

    for i, key in enumerate(keys):
        patch = source.region(key, y0, y1, x0, x1)
        if patch is not None and patch.size:
            values = np.asarray(patch, dtype=np.float64)
            n_frames = len(mapping_bins.get(key) or []) or 1
            scale = float(n_frames) if normalize_frames else 1.0
            profile[key] = {
                "intensity": float(np.nanmax(values)) / scale,
                "integrated": float(np.nansum(values)) / scale,
                "mean": float(np.nanmean(values)) / scale,
                "n_pixels": int(values.size),
                "n_frames": int(n_frames),
            }
        if progress is not None:
            progress(i + 1, len(keys))

    parsed = [(key, _parse_key(key)) for key in profile]
    parsed = [(key, rc) for key, rc in parsed if rc is not None]
    n_rows = int((grid_mapping or {}).get("n_bin_rows") or
                 (grid_mapping or {}).get("n_rows") or
                 (max((rc[0] for _, rc in parsed), default=-1) + 1))
    n_cols = int((grid_mapping or {}).get("n_bin_cols") or
                 (grid_mapping or {}).get("n_cols") or
                 (max((rc[1] for _, rc in parsed), default=-1) + 1))
    center_bin = max(profile, key=lambda key: profile[key][metric]) if profile else None

    return {
        "roi": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
        "metric": metric,
        "normalize_frames": bool(normalize_frames),
        "n_bin_rows": n_rows,
        "n_bin_cols": n_cols,
        "center_bin": center_bin,
        "profile": profile,
    }


def grid_array(result: dict, metric: Optional[str] = None) -> np.ndarray:
    """Convert a sampled result to a row-major array with NaN for missing bins."""
    metric = metric or result.get("metric", "integrated")
    if metric not in METRICS:
        raise ValueError(f"Unknown metric {metric!r}; choose from {METRICS}")
    grid = np.full((int(result.get("n_bin_rows", 0)),
                    int(result.get("n_bin_cols", 0))), np.nan, dtype=float)
    for key, entry in (result.get("profile") or {}).items():
        rc = _parse_key(key)
        if rc is None:
            continue
        row, col = rc
        if 0 <= row < grid.shape[0] and 0 <= col < grid.shape[1]:
            grid[row, col] = float(entry[metric])
    return grid
