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


def normalize_sample_crop(crop, n_rows: int, n_cols: int) -> tuple[int, int, int, int]:
    """Clamp half-open sample-space bounds to the active binned grid."""
    x0, y0, x1, y1 = normalize_roi(crop)
    x0, x1 = min(x0, n_cols), min(x1, n_cols)
    y0, y1 = min(y0, n_rows), min(y1, n_rows)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Sample crop does not overlap the spatial grid")
    return x0, y0, x1, y1


def raw_crop_to_bins(crop, bin_size: int, n_rows: int, n_cols: int):
    """Convert half-open raw (1x1) sample bounds to overlapping binned cells."""
    x0, y0, x1, y1 = normalize_roi(crop)
    size = max(1, int(bin_size))
    return normalize_sample_crop(
        (x0 // size, y0 // size, (x1 + size - 1) // size, (y1 + size - 1) // size),
        n_rows, n_cols,
    )


def apply_sample_crop(grid, crop):
    """Return a same-size grid with finite values outside ``crop`` set to zero."""
    values = np.array(grid, dtype=float, copy=True)
    if crop is None:
        return values
    x0, y0, x1, y1 = normalize_sample_crop(crop, *values.shape)
    outside = np.ones(values.shape, dtype=bool)
    outside[y0:y1, x0:x1] = False
    values[outside] = 0.0
    return values


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


def _reduction(values, n_frames, normalize_frames):
    values = np.asarray(values, dtype=np.float64)
    scale = float(n_frames) if normalize_frames else 1.0
    return {
        "intensity": float(np.nanmax(values)) / scale,
        "integrated": float(np.nansum(values)) / scale,
        "mean": float(np.nanmean(values)) / scale,
        "n_pixels": int(values.size),
        "n_frames": int(n_frames),
    }


def _result(roi, profile, grid_mapping, metric, normalize_frames, **extra):
    parsed = [(key, _parse_key(key)) for key in profile]
    parsed = [(key, rc) for key, rc in parsed if rc is not None]
    n_rows = int((grid_mapping or {}).get("n_bin_rows") or
                 (grid_mapping or {}).get("n_rows") or
                 (max((rc[0] for _, rc in parsed), default=-1) + 1))
    n_cols = int((grid_mapping or {}).get("n_bin_cols") or
                 (grid_mapping or {}).get("n_cols") or
                 (max((rc[1] for _, rc in parsed), default=-1) + 1))
    center_bin = max(profile, key=lambda key: profile[key][metric]) if profile else None
    x0, y0, x1, y1 = roi
    return {
        "roi": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
        "metric": metric,
        "normalize_frames": bool(normalize_frames),
        "n_bin_rows": n_rows,
        "n_bin_cols": n_cols,
        "center_bin": center_bin,
        "profile": profile,
        **extra,
    }


def _roi_groups(rois):
    """Group nearby ROIs without making sparse unions much larger than their data."""
    groups = []
    for roi_index, roi in enumerate(rois):
        best = None
        roi_area = (roi[2] - roi[0]) * (roi[3] - roi[1])
        for group in groups:
            x0 = min(group["bounds"][0], roi[0]); y0 = min(group["bounds"][1], roi[1])
            x1 = max(group["bounds"][2], roi[2]); y1 = max(group["bounds"][3], roi[3])
            union_area = (x1 - x0) * (y1 - y0)
            if union_area <= 4 * (group["area"] + roi_area):
                best = (group, (x0, y0, x1, y1))
                break
        if best is None:
            groups.append({"bounds": roi, "area": roi_area,
                           "members": [(roi_index, roi)]})
        else:
            group, bounds = best
            group["bounds"] = bounds
            group["area"] += roi_area
            group["members"].append((roi_index, roi))
    return groups


def _apply_profile_crops(profiles, crops, parsed, mapping_bins):
    for profile, crop in zip(profiles, crops):
        if crop is None:
            continue
        x0, y0, x1, y1 = crop
        for key, (row, col) in parsed.items():
            if y0 <= row < y1 and x0 <= col < x1:
                continue
            profile[key] = {
                "intensity": 0.0, "integrated": 0.0, "mean": 0.0,
                "n_pixels": 0, "n_frames": len(mapping_bins.get(key) or []) or 1,
                "sample_crop_fill": True,
            }


def _sample_keys(source, keys, rois, mapping_bins, normalize_frames, profiles,
                 progress=None, progress_offset=0, progress_total=None):
    groups = _roi_groups(rois)
    normalize_source = (normalize_frames and
                        getattr(source, "aggregation", "sum") != "mean_per_frame")
    total = progress_total or len(keys)
    for index, key in enumerate(keys):
        for group in groups:
            ux0, uy0, ux1, uy1 = group["bounds"]
            union = source.region(key, uy0, uy1, ux0, ux1)
            if union is None or not union.size:
                continue
            for roi_index, (x0, y0, x1, y1) in group["members"]:
                patch = union[y0 - uy0:y1 - uy0, x0 - ux0:x1 - ux0]
                if patch.size:
                    n_frames = len(mapping_bins.get(key) or []) or 1
                    profiles[roi_index][key] = _reduction(
                        patch, n_frames, normalize_source)
        if progress is not None:
            progress(progress_offset + index + 1, total)


def sample_rois(
    source,
    rois,
    *,
    grid_mapping: Optional[dict] = None,
    metric: str = "integrated",
    normalize_frames: bool = False,
    sample_crop=None,
    sample_crops=None,
    fast: bool = False,
    stride: int = 3,
    progress: Optional[Callable[[int, int], None]] = None,
    log: Callable[[str], None] = print,
) -> list[dict]:
    """Reduce multiple detector ROIs in one spatial-bin pass.

    Each bin's union detector bounding box is read once and all ROIs are sliced
    from that in-memory patch. ``fast`` is an approximate preview: sample a
    guarded spatial stride, identify bright coarse cells, then densely reread
    their neighborhoods. Unsampled cells receive the coarse median floor and are
    marked ``coarse_fill``. Exact analysis remains the default.
    """
    if metric not in METRICS:
        raise ValueError(f"Unknown metric {metric!r}; choose from {METRICS}")
    normalized = [normalize_roi(roi) for roi in rois]
    if not normalized:
        return []
    keys = list(source.keys())
    if not keys:
        raise ValueError("ROI source has no spatial bins")
    parsed = {key: _parse_key(key) for key in keys}
    invalid = [key for key, rc in parsed.items() if rc is None]
    if invalid:
        raise ValueError(f"ROI source has invalid spatial bin key {invalid[0]!r}; expected ROW_COL")
    mapping_bins = (grid_mapping or {}).get("bins") or {}
    if mapping_bins and not set(keys).intersection(mapping_bins):
        raise ValueError("ROI source spatial bins do not overlap the grid mapping")
    if sample_crops is not None and sample_crop is not None:
        raise ValueError("Use sample_crop or sample_crops, not both")
    requested_crops = sample_crops if sample_crops is not None else [sample_crop] * len(normalized)
    if len(requested_crops) != len(normalized):
        raise ValueError("sample_crops must contain one entry per detector ROI")
    parsed_values = [rc for rc in parsed.values() if rc is not None]
    n_rows = int((grid_mapping or {}).get("n_bin_rows") or
                 (max((rc[0] for rc in parsed_values), default=-1) + 1))
    n_cols = int((grid_mapping or {}).get("n_bin_cols") or
                 (max((rc[1] for rc in parsed_values), default=-1) + 1))
    crops = [None if crop is None else normalize_sample_crop(crop, n_rows, n_cols)
             for crop in requested_crops]
    profiles = [{} for _ in normalized]
    stride = max(2, int(stride))
    regular = list(keys)

    if not fast or stride <= 1 or len(regular) < stride * stride * 2:
        if fast:
            log("[roi-map] fast preview guard: grid too small/irregular; using exact sweep")
        _sample_keys(source, keys, normalized, mapping_bins, normalize_frames,
                     profiles, progress=progress)
        if any(not profile for profile in profiles):
            raise ValueError("ROI does not overlap readable detector data")
        _apply_profile_crops(profiles, crops, parsed, mapping_bins)
        return [_result(roi, profile, grid_mapping, metric, normalize_frames,
                        approximate=False, sample_crop=crop)
                for roi, profile, crop in zip(normalized, profiles, crops)]

    coarse = [key for key in regular
              if parsed[key][0] % stride == 0 and parsed[key][1] % stride == 0]
    if not coarse:
        log("[roi-map] fast preview guard: no stride samples; using exact sweep")
        _sample_keys(source, keys, normalized, mapping_bins, normalize_frames,
                     profiles, progress=progress)
        if any(not profile for profile in profiles):
            raise ValueError("ROI does not overlap readable detector data")
        _apply_profile_crops(profiles, crops, parsed, mapping_bins)
        return [_result(roi, profile, grid_mapping, metric, normalize_frames,
                        approximate=False, sample_crop=crop)
                for roi, profile, crop in zip(normalized, profiles, crops)]

    log(f"[roi-map] FAST PREVIEW: stride={stride}, coarse {len(coarse)}/{len(keys)} bins; "
        "small features between sampled cells may be missed")
    _sample_keys(source, coarse, normalized, mapping_bins, normalize_frames, profiles,
                 progress=progress)
    if any(not profile for profile in profiles):
        raise ValueError("ROI does not overlap readable detector data")
    refine = set()
    for profile in profiles:
        values = np.asarray([entry[metric] for entry in profile.values()], dtype=float)
        median = float(np.nanmedian(values))
        mad = float(np.nanmedian(np.abs(values - median))) * 1.4826
        threshold = max(median + 3.0 * mad,
                        median + 0.15 * (float(np.nanmax(values)) - median))
        seeds = [key for key, entry in profile.items() if entry[metric] > threshold]
        if not seeds:
            seeds = [max(profile, key=lambda key: profile[key][metric])]
        for seed in seeds:
            row, col = parsed[seed]
            margin = stride + 1
            for key in regular:
                rc = parsed[key]
                if abs(rc[0] - row) <= margin and abs(rc[1] - col) <= margin:
                    refine.add(key)
    refine.difference_update(coarse)
    total_reads = len(coarse) + len(refine)
    log(f"[roi-map] FAST PREVIEW: densely refining {len(refine)} bins around coarse signal")
    _sample_keys(source, sorted(refine, key=lambda key: parsed[key]), normalized,
                 mapping_bins, normalize_frames, profiles, progress=progress,
                 progress_offset=len(coarse), progress_total=max(total_reads, 1))

    for profile in profiles:
        floors = {
            name: float(np.nanmedian([entry[name] for entry in profile.values()]))
            for name in METRICS
        }
        for key in keys:
            if key not in profile:
                n_frames = len(mapping_bins.get(key) or []) or 1
                profile[key] = {**floors, "n_pixels": 0,
                                "n_frames": int(n_frames), "coarse_fill": True}
    _apply_profile_crops(profiles, crops, parsed, mapping_bins)
    return [_result(roi, profile, grid_mapping, metric, normalize_frames,
                    approximate=True, stride=stride, sample_crop=crop,
                    sampled_bins=total_reads, total_bins=len(keys))
            for roi, profile, crop in zip(normalized, profiles, crops)]


def sample_roi(source, roi, **kwargs) -> dict:
    """Backward-compatible one-ROI wrapper over :func:`sample_rois`."""
    return sample_rois(source, [roi], **kwargs)[0]


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
        "n_bin_rows": int(result.get("n_bin_rows", 0)),
        "n_bin_cols": int(result.get("n_bin_cols", 0)),
        "intensity_profile": intensity_profile,
        "reason": "manual fixed detector ROI integrated across all spatial bins",
        "manual_roi": dict(roi),
        "roi_metric": metric,
    }
    if result.get("sample_crop") is not None:
        x0, y0, x1, y1 = result["sample_crop"]
        feature["sample_crop"] = {"x0": x0, "y0": y0, "x1": x1, "y1": y1}
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
