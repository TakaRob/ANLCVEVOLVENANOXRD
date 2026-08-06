import numpy as np

from xrd_app.core import detector_display


def test_auto_levels_match_manual_reflections_percentiles():
    image = np.arange(1000, dtype=float).reshape(20, 50)

    low, high = detector_display.auto_levels(image)

    assert low == np.percentile(image, 1.0)
    assert high == np.percentile(image, 99.5)


def test_radial_median_subtraction_removes_ring_background():
    rows, cols = np.indices((80, 80))
    radius = np.hypot(rows - 40, cols - 40) * 0.1
    background = 20.0 + 30.0 * radius
    image = background.copy()
    image[20:23, 40:43] += 500.0

    cleaned = detector_display.radial_median_subtract(image, radius)

    quiet = np.ones(image.shape, dtype=bool)
    quiet[18:25, 38:45] = False
    assert abs(float(np.median(cleaned[quiet]))) < 2.0
    assert float(np.max(cleaned[20:23, 40:43])) > 450.0


def test_prepare_noise_and_log_keeps_nonnegative_display():
    rows, cols = np.indices((40, 40))
    radius = np.hypot(rows - 20, cols - 20) * 0.1
    image = 10.0 + radius
    image[10, 20] += 100.0

    display = detector_display.prepare(
        image, tth_map=radius, noise_reduction=True, log_scale=True)

    assert display.shape == image.shape
    assert np.isfinite(display).all()
    assert np.min(display) >= 0.0
