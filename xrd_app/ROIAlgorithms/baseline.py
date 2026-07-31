"""Baseline fully-summed-image ROI detector.

Contract for ROI > Shape and CVEvolve candidates:
``detect_rois(image, sensitivity=4.0, min_distance=12, max_rois=200)`` returns
``[{"roi": (x0, y0, x1, y1), "score": float}, ...]``.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def detect_rois(image, sensitivity=4.0, min_distance=12, max_rois=200):
    image = np.asarray(image, dtype=float)
    # Remove broad detector background without requiring a tth calibration.
    # Gaussian smoothing is fast enough for an interactive full-detector button;
    # a 31x31 median filter is prohibitively slow on 1062x1028 images.
    background = ndimage.gaussian_filter(image, sigma=10.0)
    cleaned = np.clip(image - background, 0, None)
    finite = cleaned[np.isfinite(cleaned)]
    if not finite.size:
        return []
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median))) * 1.4826
    threshold = median + float(sensitivity) * max(mad, 1e-9)
    radius = max(1, int(min_distance))
    maxima = cleaned == ndimage.maximum_filter(cleaned, size=2 * radius + 1)
    ys, xs = np.nonzero(maxima & (cleaned > threshold))
    order = np.argsort(cleaned[ys, xs])[::-1][:int(max_rois)]

    from xrd_app.core.roi_map import auto_roi_from_click
    out, seen = [], set()
    for index in order:
        roi = auto_roi_from_click(image, int(xs[index]), int(ys[index]), search_radius=radius)
        if roi is None or roi in seen:
            continue
        seen.add(roi)
        out.append({"roi": roi, "score": float(cleaned[ys[index], xs[index]])})
    return out
