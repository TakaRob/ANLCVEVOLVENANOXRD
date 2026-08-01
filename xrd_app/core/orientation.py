"""Pure orientation analysis for detector features.

This module owns beam-relative azimuths, feature weighting, circular KDE
clustering, and one-degree orientation densities. Rendering remains in the GUI.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from .processing import estimate_beam_center


def compute_chi_map(shape, beam_center):
    """Return detector azimuth chi in degrees for an image ``shape``."""
    by, bx = beam_center
    ys, xs = np.mgrid[0:shape[0], 0:shape[1]]
    return np.degrees(np.arctan2(ys - by, xs - bx))


def feature_area(feature):
    return float(feature.get("n_bins") or
                 len(feature.get("intensity_profile") or {}) or 1)


def feature_spread(feature):
    """Combined rocking/strain spread, floored to one for weighting."""
    spread = (float(feature.get("rocking_fwhm") or 0.0) +
              float(feature.get("strain_breadth") or 0.0))
    return spread if spread > 0 else 1.0


def feature_weight(feature, mode="count"):
    """Return a non-negative count, area, intensity, or bright-big weight."""
    if mode == "area":
        return max(feature_area(feature), 0.0)
    if mode == "intensity":
        return max(float(feature.get("peak_intensity") or 0.0), 0.0)
    if mode == "bright_big":
        intensity = max(float(feature.get("peak_intensity") or 0.0), 0.0)
        return intensity * feature_spread(feature) * max(feature_area(feature), 1.0)
    return 1.0


def _make_cluster(features, chi_values, mode, total=None):
    cluster_weight = float(sum(feature_weight(f, mode) for f in features))
    total = total if total else cluster_weight
    chi_min, chi_max = min(chi_values), max(chi_values)
    if chi_max - chi_min > 180:
        shifted = [chi + 360 if chi < 0 else chi for chi in chi_values]
        shifted_min, shifted_max = min(shifted), max(shifted)
        center = (shifted_min + shifted_max) / 2
        if center > 180:
            center -= 360
        span = shifted_max - shifted_min
        margin = max(3.0, span * 0.12)
        lo, hi = shifted_min - margin, shifted_max + margin
        chi_lo = lo if lo <= 180 else lo - 360
        chi_hi = hi if hi <= 180 else hi - 360
        wraps = True
    else:
        center = (chi_min + chi_max) / 2
        span = chi_max - chi_min
        margin = max(3.0, span * 0.12)
        chi_lo, chi_hi = chi_min - margin, chi_max + margin
        wraps = False
    return {
        "chi_center": round(center, 1), "chi_span": round(span, 1),
        "chi_lo": chi_lo, "chi_hi": chi_hi, "wraps": wraps,
        "pct": round(100.0 * cluster_weight / total, 1) if total else 0.0,
        "features": features, "n": len(features),
        "weight": round(cluster_weight, 1),
    }


def cluster_features_by_chi(features, bandwidth=5.0, weight_mode="count"):
    """Split features at valleys in a weighted circular Gaussian KDE."""
    items = [(f.get("chi_deg"), f) for f in features
             if f.get("chi_deg") is not None]
    if not items:
        return [], []
    items.sort(key=lambda item: item[0])
    if len(items) < 3:
        return [_make_cluster([item[1] for item in items],
                              [item[0] for item in items], weight_mode)], []

    chis = np.array([item[0] for item in items])
    weights = np.array([feature_weight(item[1], weight_mode) for item in items],
                       dtype=float)
    total_weight = float(weights.sum())
    grid = np.linspace(-180, 179, 360)
    kde = np.zeros(360)
    for chi, weight in zip(chis, weights):
        diff = (grid - chi + 180) % 360 - 180
        kde += weight * np.exp(-0.5 * (diff / bandwidth) ** 2)

    pad = max(4, int(bandwidth * 2))
    extended = np.concatenate([kde[-pad:], kde, kde[:pad]])
    valley_idx, _ = find_peaks(
        -extended, distance=max(4, int(bandwidth * 1.5)),
        prominence=0.3 * kde.max())
    valley_idx -= pad
    valley_idx = valley_idx[(valley_idx >= 0) & (valley_idx < 360)]
    if len(valley_idx) < 2 or kde.max() == 0:
        return [_make_cluster([item[1] for item in items], chis.tolist(),
                              weight_mode)], []

    valley_angles = grid[valley_idx]
    normalized_valleys = np.sort((valley_angles + 180) % 360)
    groups = defaultdict(list)
    for chi, feature in items:
        normalized_chi = (chi + 180) % 360
        idx = int(np.searchsorted(normalized_valleys, normalized_chi,
                                  side="right")) % len(normalized_valleys)
        groups[idx].append((chi, feature))

    clusters = []
    for idx in sorted(groups):
        group = groups[idx]
        cluster = _make_cluster([item[1] for item in group],
                                [item[0] for item in group], weight_mode,
                                total_weight)
        cluster["chi_lo"] = float(normalized_valleys[idx - 1] - 180)
        cluster["chi_hi"] = float(normalized_valleys[idx] - 180)
        cluster["wraps"] = normalized_valleys[idx - 1] > normalized_valleys[idx]
        clusters.append(cluster)
    return clusters, valley_angles.tolist()


def orientation_densities(features_by_ref, active_refs, sigma=3.0,
                          low_pct=0.0, high_pct=100.0,
                          weight_mode="count"):
    """Return smoothed one-degree chi densities and their contrast range."""
    densities = {}
    global_max = 0.0
    for reflection in active_refs:
        pairs = [(f["chi_deg"], feature_weight(f, weight_mode))
                 for f in features_by_ref.get(reflection, [])
                 if f.get("chi_deg") is not None]
        if not pairs:
            continue
        chis = np.array([pair[0] for pair in pairs])
        weights = np.array([pair[1] for pair in pairs], dtype=float)
        hist, _ = np.histogram(chis, bins=np.arange(-180, 181, 1),
                               weights=weights)
        density = gaussian_filter1d(hist.astype(float), sigma=sigma, mode="wrap")
        densities[reflection] = density
        global_max = max(global_max, float(density.max()))

    if global_max == 0:
        return densities, 0.0, 0.0, 1.0
    pool = np.concatenate(list(densities.values()))
    pool = pool[pool > 0]
    if pool.size:
        if high_pct <= low_pct:
            high_pct = min(100.0, low_pct + 1.0)
        vmin = float(np.percentile(pool, low_pct))
        vmax = float(np.percentile(pool, high_pct))
    else:
        vmin, vmax = 0.0, global_max
    if vmax <= vmin:
        vmax = vmin + 1e-9
    return densities, global_max, vmin, vmax
