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


def auto_roi_from_click(image, x: int, y: int, *, search_radius: int = 25,
                        fit_radius: int = 18, sigma_extent: float = 2.0):
    """Snap to a local maximum and bound its connected Gaussian-like footprint.

    Background is estimated from the fit-window border. The component threshold
    is ``exp(-sigma_extent² / 2)`` of the background-subtracted peak, corresponding
    to the requested Gaussian sigma extent (2σ by default).
    """
    from scipy import ndimage

    image = np.asarray(image, dtype=float)
    if image.ndim != 2:
        raise ValueError("Detector image must be two-dimensional")
    h, w = image.shape
    x, y = int(x), int(y)
    x0, x1 = max(0, x - search_radius), min(w, x + search_radius + 1)
    y0, y1 = max(0, y - search_radius), min(h, y + search_radius + 1)
    search = image[y0:y1, x0:x1]
    if not search.size or not np.isfinite(search).any():
        return None
    local_y, local_x = np.unravel_index(np.nanargmax(search), search.shape)
    peak_x, peak_y = x0 + int(local_x), y0 + int(local_y)

    fx0, fx1 = max(0, peak_x - fit_radius), min(w, peak_x + fit_radius + 1)
    fy0, fy1 = max(0, peak_y - fit_radius), min(h, peak_y + fit_radius + 1)
    patch = image[fy0:fy1, fx0:fx1]
    border = np.concatenate((patch[0], patch[-1], patch[:, 0], patch[:, -1]))
    background = float(np.nanmedian(border))
    signal = np.clip(patch - background, 0, None)
    peak = float(signal[peak_y - fy0, peak_x - fx0])
    if not np.isfinite(peak) or peak <= 0:
        half = 5
        return (max(0, peak_x - half), max(0, peak_y - half),
                min(w, peak_x + half + 1), min(h, peak_y + half + 1))

    threshold = peak * np.exp(-(float(sigma_extent) ** 2) / 2.0)
    mask = signal >= threshold
    labels, _ = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    label = labels[peak_y - fy0, peak_x - fx0]
    component = labels == label if label else mask
    ys, xs = np.nonzero(component)
    if not len(xs):
        return None
    margin = 2
    return (max(0, fx0 + int(xs.min()) - margin),
            max(0, fy0 + int(ys.min()) - margin),
            min(w, fx0 + int(xs.max()) + margin + 1),
            min(h, fy0 + int(ys.max()) + margin + 1))


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


def to_shape_feature(result: dict, reflection: str = "manual ROI", *,
                     feature_id: int = 1, tth_map=None, beam_center=None) -> dict:
    """Represent one fixed detector ROI as a manual ROI feature.

    Every sampled spatial bin is retained in ``intensity_profile``. This is not a
    Gaussian-verified linked peak; it belongs only to the dedicated ROI catalog.
    """
    profile = result.get("profile") or {}
    if not profile:
        raise ValueError("ROI map has no sampled spatial bins")
    metric = result.get("metric", "integrated")
    center_bin = result.get("center_bin") or max(
        profile, key=lambda key: profile[key][metric])
    try:
        center_row, center_col = (int(v) for v in center_bin.split("_", 1))
    except (AttributeError, ValueError) as exc:
        raise ValueError(f"Invalid center bin {center_bin!r}") from exc
    roi = result["roi"]
    det_x = int(round((roi["x0"] + roi["x1"] - 1) / 2))
    det_y = int(round((roi["y0"] + roi["y1"] - 1) / 2))

    intensity_profile = {}
    for key, entry in profile.items():
        item = {
            "intensity": round(float(entry["intensity"]), 1),
            "integrated": round(float(entry["integrated"]), 1),
            "mean": round(float(entry["mean"]), 3),
            "det_x": det_x,
            "det_y": det_y,
        }
        if tth_map is not None and 0 <= det_y < tth_map.shape[0] and 0 <= det_x < tth_map.shape[1]:
            item["tth"] = round(float(tth_map[det_y, det_x]), 5)
        if beam_center is not None:
            by, bx = beam_center
            item["chi"] = round(float(np.degrees(np.arctan2(det_y - by, det_x - bx))), 2)
        intensity_profile[key] = item

    center_entry = profile[center_bin]
    feature = {
        "feature_id": int(feature_id),
        "reflection": reflection,
        "detector_x": det_x,
        "detector_y": det_y,
        "peak_intensity": float(center_entry["intensity"]),
        "mean_snr": None,
        "n_bins": len(profile),
        "spatial_extent": sorted(profile, key=lambda key: _parse_key(key) or (0, 0)),
        "center_bin": center_bin,
        "center_row": center_row,
        "center_col": center_col,
        "intensity_profile": intensity_profile,
        "reason": "manual fixed detector ROI integrated across all spatial bins",
        "manual_roi": dict(roi),
        "roi_metric": metric,
    }
    if tth_map is not None and 0 <= det_y < tth_map.shape[0] and 0 <= det_x < tth_map.shape[1]:
        feature["ref_tth"] = round(float(tth_map[det_y, det_x]), 5)
    if beam_center is not None:
        by, bx = beam_center
        feature["chi_deg"] = round(float(np.degrees(np.arctan2(det_y - by, det_x - bx))), 1)
    return feature


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
