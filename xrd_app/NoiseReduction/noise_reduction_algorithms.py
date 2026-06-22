"""
Noise reduction algorithms for XRD detector images.

Four radial background models:
  1. Gaussian — symmetric bell curve
  2. Split Gaussian — asymmetric (different left/right widths)
  3. Skewed Gaussian — asymmetric via scipy.stats.skewnorm
  4. Fourier low-pass — FFT-based smoothing of empirical profile

Usage:
    from noise_reduction_algorithms import (
        compute_radial_profile, fit_all_models, build_background_image,
        subtract_background, ALGORITHM_NAMES,
    )
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import skewnorm


ALGORITHM_NAMES = [
    "gaussian",
    "split_gaussian",
    "skewed_gaussian",
    "fourier",
]

ALGORITHM_DISPLAY = {
    "gaussian": "Gaussian",
    "split_gaussian": "Split Gaussian",
    "skewed_gaussian": "Skewed Gaussian",
    "fourier": "Fourier Low-Pass",
}


# ---------------------------------------------------------------------------
# Model functions
# ---------------------------------------------------------------------------

def gaussian_model(x, amplitude, mu, sigma, offset):
    return amplitude * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + offset


def split_gaussian_model(x, amplitude, mu, sigma_left, sigma_right, offset):
    result = np.empty_like(x)
    left = x <= mu
    right = ~left
    result[left] = amplitude * np.exp(-0.5 * ((x[left] - mu) / sigma_left) ** 2)
    result[right] = amplitude * np.exp(-0.5 * ((x[right] - mu) / sigma_right) ** 2)
    return result + offset


def skewed_gaussian_model(x, amplitude, mu, sigma, skew, offset):
    z = (x - mu) / sigma
    pdf_vals = skewnorm.pdf(z, skew) / skewnorm.pdf(skewnorm.mean(skew), skew)
    return amplitude * pdf_vals + offset


def fourier_lowpass(profile, cutoff_fraction=0.05):
    n = len(profile)
    fft_vals = np.fft.rfft(profile)
    freqs = np.fft.rfftfreq(n)
    fft_vals[freqs > cutoff_fraction] = 0
    return np.fft.irfft(fft_vals, n=n)


# ---------------------------------------------------------------------------
# Radial profile computation
# ---------------------------------------------------------------------------

# A genuine 2θ-per-pixel map needs only a few thousand bins; far more means the
# linked map is not a 2θ map (e.g. a summed-intensity image), which would
# otherwise allocate a multi-million-element histogram and hang.
_MAX_TTH_BINS = 100_000


def compute_tth_binning(tth_map, bin_width=0.05):
    """Pre-compute 2-theta bin edges, centers, and per-pixel bin indices."""
    tth_min, tth_max = float(tth_map.min()), float(tth_map.max())
    n_bins = int(np.ceil((tth_max - tth_min) / bin_width)) if tth_max > tth_min else 1
    if n_bins > _MAX_TTH_BINS:
        raise ValueError(
            f"2θ map spans {tth_min:.1f}..{tth_max:.1f} → {n_bins} bins at "
            f"bin_width={bin_width}; the linked tth_map is probably not a 2θ map "
            f"(e.g. a summed intensity image). Re-link it with "
            f"`xrd-app link --tth <tth.tiff>`."
        )
    edges = np.arange(tth_min, tth_max + bin_width, bin_width)
    centers = 0.5 * (edges[:-1] + edges[1:])
    n_bins = len(centers)

    flat = tth_map.ravel()
    indices = np.digitize(flat, edges) - 1
    indices = np.clip(indices, 0, n_bins - 1)

    counts = np.bincount(indices, minlength=n_bins)

    return edges, centers, n_bins, indices, counts


def compute_radial_profile(image, bin_indices, n_tth_bins):
    """Median intensity vs 2-theta for a single image."""
    img_flat = image.ravel()
    profile = np.zeros(n_tth_bins)
    order = np.argsort(bin_indices, kind="stable")
    sorted_idx = bin_indices[order]
    boundaries = np.searchsorted(sorted_idx, np.arange(n_tth_bins + 1))
    for i in range(n_tth_bins):
        start, end = boundaries[i], boundaries[i + 1]
        if end > start:
            profile[i] = np.median(img_flat[order[start:end]])
    return profile


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------

def fit_all_models(tth_centers, radial_profile, valid_mask, tth_min, tth_max):
    """Fit all four noise models to a radial profile.

    Returns dict of {algorithm_name: {"profile": array, "params": dict}}.
    Only includes models that converged.
    """
    x_data = tth_centers
    y_data = radial_profile
    x_fit = x_data[valid_mask]
    y_fit = y_data[valid_mask]

    if len(x_fit) < 5:
        return {}

    peak_idx = np.argmax(y_fit)
    mu_init = x_fit[peak_idx]
    amp_init = y_fit[peak_idx] - np.percentile(y_fit, 5)
    sigma_init = 3.0
    offset_init = np.percentile(y_fit, 5)

    results = {}

    try:
        popt, _ = curve_fit(
            gaussian_model, x_fit, y_fit,
            p0=[amp_init, mu_init, sigma_init, offset_init],
            maxfev=10000,
        )
        results["gaussian"] = {
            "profile": gaussian_model(x_data, *popt),
            "params": dict(amplitude=popt[0], mu=popt[1], sigma=popt[2], offset=popt[3]),
        }
    except Exception:
        pass

    try:
        popt, _ = curve_fit(
            split_gaussian_model, x_fit, y_fit,
            p0=[amp_init, mu_init, sigma_init, sigma_init, offset_init],
            maxfev=10000,
        )
        results["split_gaussian"] = {
            "profile": split_gaussian_model(x_data, *popt),
            "params": dict(amplitude=popt[0], mu=popt[1],
                           sigma_left=popt[2], sigma_right=popt[3], offset=popt[4]),
        }
    except Exception:
        pass

    try:
        popt, _ = curve_fit(
            skewed_gaussian_model, x_fit, y_fit,
            p0=[amp_init, mu_init, sigma_init, 0.0, offset_init],
            maxfev=10000,
            bounds=([0, tth_min, 0.1, -20, 0], [np.inf, tth_max, 30, 20, np.inf]),
        )
        results["skewed_gaussian"] = {
            "profile": skewed_gaussian_model(x_data, *popt),
            "params": dict(amplitude=popt[0], mu=popt[1],
                           sigma=popt[2], skew=popt[3], offset=popt[4]),
        }
    except Exception:
        pass

    best_fourier = None
    best_rmse = np.inf
    for cutoff in [0.02, 0.03, 0.05, 0.07, 0.10]:
        fitted = fourier_lowpass(y_data, cutoff_fraction=cutoff)
        rmse = np.sqrt(np.mean((y_data[valid_mask] - fitted[valid_mask]) ** 2))
        if rmse < best_rmse:
            best_rmse = rmse
            best_fourier = {"profile": fitted, "cutoff": cutoff}
    if best_fourier is not None:
        results["fourier"] = {
            "profile": best_fourier["profile"],
            "params": dict(cutoff=best_fourier["cutoff"]),
        }

    return results


# ---------------------------------------------------------------------------
# Background construction & subtraction
# ---------------------------------------------------------------------------

def build_background_image(tth_map, tth_centers, radial_profile, bin_indices_2d,
                           tth_edges=None):
    """Map a 1D radial profile back to the 2D detector geometry."""
    if bin_indices_2d is None and tth_edges is not None:
        flat = tth_map.ravel()
        idx = np.digitize(flat, tth_edges) - 1
        idx = np.clip(idx, 0, len(radial_profile) - 1)
    else:
        idx = bin_indices_2d.ravel()
    return radial_profile[idx].reshape(tth_map.shape)


def subtract_background(image, bg_image, strength=1.0, shift=0.0):
    """Subtract scaled background.  strength in [0,1], shift is additive offset."""
    return image - strength * (bg_image + shift)


# ---------------------------------------------------------------------------
# Convenience: full pipeline for a single image
# ---------------------------------------------------------------------------

def reduce_noise(image, tth_map, tth_edges, tth_centers, n_tth_bins, bin_indices,
                 radial_counts, algorithm="gaussian", strength=1.0, shift=0.0):
    """One-shot noise reduction: fit model to image's radial profile, subtract.

    Returns (cleaned_image, background_image, fit_info_dict) or
    (image, None, None) if the requested model fails to converge.
    """
    valid_mask = radial_counts > 50
    profile = compute_radial_profile(image, bin_indices, n_tth_bins)
    fits = fit_all_models(tth_centers, profile, valid_mask,
                          tth_edges[0], tth_edges[-1])

    if algorithm not in fits:
        return image.copy(), None, None

    fit = fits[algorithm]
    bg = build_background_image(tth_map, tth_centers, fit["profile"], bin_indices)
    cleaned = subtract_background(image, bg, strength=strength, shift=shift)
    return cleaned, bg, fit
