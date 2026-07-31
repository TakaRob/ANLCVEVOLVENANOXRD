"""Image-only learned peak detector for Wieghold fully summed XRD images.

A compact CNN-like filter bank: five Difference-of-Gaussian channels generate
candidate peaks, four local-contrast channels describe peak morphology, and a
logistic classifier trained from manual ROI centers scores candidates. No
2-theta map, reflection bands, or detector coordinates are used.

Model selection used leave-one-scan-out F1 over Scan_0029..0039:
mean F1=0.8341, precision=0.9463, recall=0.7680 (25 px matching tolerance).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

_SCALES = ((1, 4), (2, 8), (3, 15), (5, 30), (8, 45))
_MEAN = np.array([
    369028979.53871864, 363186754.33541816, 87347794.88671194,
    47.37683709802708, 5.0218248119925155, 0.885126469070879,
    1.1554604734090654, 1.3437680207251106, 1.4345419157331971,
])
_SCALE = np.array([
    486981432.1852397, 521224496.61218286, 350425065.7454567,
    119.67317460800989, 16.33359148552477, 0.6466569753256526,
    0.61038486647957, 0.6207311715517125, 0.7394328691284578,
])
_WEIGHTS = np.array([
    -5.240733376045919, 0.6365299768284921, 0.1266633865067956,
    -0.11241323634373161, 0.1632655588716248, 0.34766908203042807,
    -0.4779818473429938, -0.27595587846033776, -1.5022654866548464,
    3.0961203644662336,
])


def _robust_z(values):
    finite = values[np.isfinite(values)]
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median))) * 1.4826
    return (values - median) / max(mad, 1e-9)


def detect_rois(image, sensitivity=0.5, min_distance=4, max_rois=15):
    image = np.asarray(image, dtype=float)
    log_image = np.log1p(np.clip(image, 0, None))
    channels = []
    for inner, outer in _SCALES:
        response = np.clip(
            ndimage.gaussian_filter(log_image, inner) -
            ndimage.gaussian_filter(log_image, outer), 0, None)
        channels.append(_robust_z(response))
    stack = np.stack(channels)
    combined = stack.max(axis=0)
    distance = max(1, int(min_distance))
    maxima = combined == ndimage.maximum_filter(combined, size=2 * distance + 1)
    ys, xs = np.nonzero(maxima & (combined > 0.5))
    order = np.argsort(combined[ys, xs])[::-1][:800]
    xs, ys = xs[order], ys[order]
    features = [stack[:, ys, xs].T]
    for radius in (2, 4, 8, 16):
        mean = ndimage.uniform_filter(log_image, size=2 * radius + 1)
        square_mean = ndimage.uniform_filter(log_image * log_image, size=2 * radius + 1)
        std = np.sqrt(np.maximum(square_mean - mean * mean, 1e-8))
        features.append(((log_image[ys, xs] - mean[ys, xs]) / std[ys, xs])[:, None])
    X = (np.hstack(features) - _MEAN) / _SCALE
    logits = _WEIGHTS[0] + X @ _WEIGHTS[1:]
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
    selected = np.argsort(probabilities)[::-1]
    selected = selected[probabilities[selected] >= float(sensitivity)][:int(max_rois)]

    from xrd_app.core.roi_map import auto_roi_from_click
    output = []
    for index in selected:
        roi = auto_roi_from_click(
            image, int(xs[index]), int(ys[index]), search_radius=10,
            fit_radius=24, sigma_extent=2.0)
        if roi is not None:
            output.append({"roi": roi, "score": float(probabilities[index])})
    return output
