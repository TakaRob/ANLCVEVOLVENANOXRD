"""Tuned fully-summed-image ROI detector for the 2026-2 Wieghold scans.

Selected on Scan_0029-0034 and 0037-0039 using mean per-scan F2. Scan_0035 and
Scan_0036 were held out during tuning. The detector uses a difference of
Gaussians to retain compact/broad Bragg spots while removing detector background.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def detect_rois(image, sensitivity=12.0, min_distance=30, max_rois=50):
    image = np.asarray(image, dtype=float)
    response = np.clip(
        ndimage.gaussian_filter(image, sigma=2.0) -
        ndimage.gaussian_filter(image, sigma=45.0),
        0, None)
    finite = response[np.isfinite(response)]
    positive = finite[finite > 0]
    if not positive.size:
        return []
    median = float(np.median(positive))
    mad = float(np.median(np.abs(positive - median))) * 1.4826
    threshold = max(median + float(sensitivity) * max(mad, 1e-9),
                    0.005 * float(np.nanmax(response)))
    radius = max(1, int(min_distance))
    maxima = response == ndimage.maximum_filter(response, size=2 * radius + 1)
    ys, xs = np.nonzero(maxima & (response > threshold))
    order = np.argsort(response[ys, xs])[::-1][:int(max_rois)]

    from xrd_app.core.roi_map import auto_roi_from_click
    out, seen = [], set()
    for index in order:
        roi = auto_roi_from_click(
            image, int(xs[index]), int(ys[index]), search_radius=radius,
            fit_radius=24, sigma_extent=2.0)
        if roi is None or roi in seen:
            continue
        seen.add(roi)
        out.append({"roi": roi, "score": float(response[ys[index], xs[index]])})
    return out
