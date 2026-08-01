"""Compatibility imports for legacy dynamically loaded detector modules.

The active implementation lives in :mod:`xrd_app.core.algorithms`. This module
remains importable as ``noise_reduction_algorithms`` because existing detector
scripts use that top-level name when loaded from arbitrary directories.
"""

from xrd_app.core.algorithms import (
    ALGORITHM_DISPLAY,
    ALGORITHM_NAMES,
    build_background_image,
    compute_radial_profile,
    compute_tth_binning,
    fit_all_models,
    fourier_lowpass,
    gaussian_model,
    reduce_noise,
    skewed_gaussian_model,
    split_gaussian_model,
    subtract_background,
)

__all__ = [
    "ALGORITHM_DISPLAY",
    "ALGORITHM_NAMES",
    "build_background_image",
    "compute_radial_profile",
    "compute_tth_binning",
    "fit_all_models",
    "fourier_lowpass",
    "gaussian_model",
    "reduce_noise",
    "skewed_gaussian_model",
    "split_gaussian_model",
    "subtract_background",
]
