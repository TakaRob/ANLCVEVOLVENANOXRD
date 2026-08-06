"""Framework-independent data reductions used by device-map renderers."""

from __future__ import annotations

import numpy as np


REFLECTION_PALETTE = [
    "#4363d8", "#e6194b", "#3cb44b", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#9a6324",
    "#800000", "#008080", "#fabebe", "#aaffc3",
]

PER_FEATURE_METRICS = {
    "chi_breadth": "chi_fwhm",
    "tth_breadth": "tth_fwhm",
    "chi": "chi_deg",
}


def _extract_metric(entry, feature, metric):
    if not isinstance(entry, dict):
        return float(entry) if metric == "intensity" else None
    if metric == "intensity":
        return entry.get("integrated", entry.get("intensity", 0))
    if metric == "tth_dev":
        tth = entry.get("tth")
        ref_tth = feature.get("ref_tth")
        return tth - ref_tth if tth is not None and ref_tth is not None else None
    return None


def build_device_grids(features, n_rows, n_cols, metric="intensity"):
    """Build one regular device grid per reflection.

    This is the shared numerical engine behind the interactive Device View and
    report export. At cells occupied by multiple features, intensity-like
    metrics retain the maximum value; 2-theta deviation retains the largest
    absolute deviation.
    """
    reflections = sorted({feature["reflection"] for feature in features})
    grids = {}
    feature_key = PER_FEATURE_METRICS.get(metric)
    for reflection in reflections:
        grid = np.full((n_rows, n_cols), np.nan)
        for feature in features:
            if feature.get("reflection") != reflection:
                continue
            profile = feature.get("intensity_profile", {})
            for bin_key, entry in profile.items():
                try:
                    row, col = (int(value) for value in bin_key.split("_", 1))
                except (AttributeError, ValueError):
                    continue
                if not (0 <= row < n_rows and 0 <= col < n_cols):
                    continue
                value = feature.get(feature_key) if feature_key else _extract_metric(
                    entry, feature, metric)
                if value is None:
                    continue
                if feature_key:
                    grid[row, col] = value
                elif metric == "tth_dev":
                    if np.isnan(grid[row, col]) or abs(value) > abs(grid[row, col]):
                        grid[row, col] = value
                else:
                    grid[row, col] = np.nanmax([grid[row, col], value])
        grids[reflection] = grid
    return grids


def feature_masks(features, n_rows, n_cols):
    """Return merged occupied-cell masks per reflection, as Device View does."""
    masks = {
        reflection: np.zeros((n_rows, n_cols), dtype=bool)
        for reflection in sorted({feature["reflection"] for feature in features})
    }
    for feature in features:
        mask = masks[feature["reflection"]]
        for bin_key in feature.get("intensity_profile", {}):
            try:
                row, col = (int(value) for value in bin_key.split("_", 1))
            except (AttributeError, ValueError):
                continue
            if 0 <= row < n_rows and 0 <= col < n_cols:
                mask[row, col] = True
    return masks


def territory_intensities(features, reflection=None, predicate=None):
    """Return maximum peak intensity per irregular territory."""
    values = {}
    for feature in features:
        if reflection is not None and feature.get("reflection") != reflection:
            continue
        if predicate is not None and not predicate(feature):
            continue
        for key, entry in (feature.get("intensity_profile") or {}).items():
            if isinstance(entry, dict):
                value = entry.get("intensity", 0)
                values[key] = max(values.get(key, 0), value)
    return values
