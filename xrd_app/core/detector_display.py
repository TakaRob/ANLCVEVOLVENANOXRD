"""Shared display processing for high-dynamic-range XRD detector images."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from . import algorithms


def radial_median_subtract(image, tth_map, bin_width=0.05):
    """Subtract the smoothed median detector background in 2-theta rings.

    This is the non-parametric noise reduction used by Manual Reflections. It is
    a display transformation only; callers should retain the original image for
    quantitative ROI integration.
    """
    image = np.asarray(image, dtype=float)
    tth_map = np.asarray(tth_map, dtype=float)
    if image.shape != tth_map.shape:
        raise ValueError(
            f"Detector image shape {image.shape} does not match 2-theta map {tth_map.shape}")
    _edges, _centers, n_bins, indices, _counts = algorithms.compute_tth_binning(
        tth_map, bin_width=bin_width)
    flat = image.ravel()
    order = np.argsort(indices, kind="stable")
    sorted_indices = indices[order]
    boundaries = np.searchsorted(sorted_indices, np.arange(n_bins + 1))
    profile = np.zeros(n_bins, dtype=float)
    sorted_values = flat[order]
    for index in range(n_bins):
        start, stop = boundaries[index], boundaries[index + 1]
        if stop > start:
            profile[index] = np.nanmedian(sorted_values[start:stop])
    if n_bins > 5:
        profile = ndimage.uniform_filter1d(profile, size=min(15, n_bins))
    background = profile[indices].reshape(image.shape)
    return image - background


def prepare(image, *, tth_map=None, noise_reduction=False, log_scale=True):
    """Return a display image using Manual Reflections processing semantics."""
    display = np.asarray(image, dtype=float)
    if noise_reduction:
        if tth_map is None:
            raise ValueError("Noise reduction requires a 2-theta map")
        display = radial_median_subtract(display, tth_map)
    if log_scale:
        display = np.log1p(np.clip(display, 0, None))
    return display


def auto_levels(display, low=1.0, high=99.5):
    """Robust intensity window used by the Manual Reflections image view."""
    display = np.asarray(display)
    finite = display[np.isfinite(display)]
    if not finite.size:
        return 0.0, 1.0
    minimum = float(np.percentile(finite, low))
    maximum = float(np.percentile(finite, high))
    if maximum <= minimum:
        maximum = minimum + 1.0
    return minimum, maximum
