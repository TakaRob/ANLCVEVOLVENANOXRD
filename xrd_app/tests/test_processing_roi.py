"""ROI constraints for the standard peak detector pipeline."""
from __future__ import annotations

import numpy as np

from xrd_app.core import processing
from xrd_app.core.processing import detect_peaks_with_intensity


class _Detector:
    def radial_median_subtract(self, image, _tth_data):
        return image

    def fast_tophat(self, cleaned, size):
        assert size == 15
        return cleaned

    def build_tth_band_masks(self, tth_map, degs, labels, tth_tolerance):
        assert tth_tolerance == 0.4
        return {labels[0]: np.ones(tth_map.shape, dtype=bool)}

    def detect_in_band(self, _tophat, _cleaned, mask, label, **_kwargs):
        ys, xs = np.nonzero(mask)
        return [{"x": int(xs[0]), "y": int(ys[0]), "snr": 10.0,
                 "label": label}] if len(xs) else []


def test_default_worker_count_is_capped(monkeypatch):
    monkeypatch.setattr(processing.os, "cpu_count", lambda: 64)
    assert processing.default_worker_count() == 4
    monkeypatch.setattr(processing.os, "cpu_count", lambda: 2)
    assert processing.default_worker_count() == 1


def test_detector_roi_is_intersected_with_reflection_band():
    image = np.zeros((8, 9), dtype=float)
    peaks, _ = detect_peaks_with_intensity(
        image, image, [7.5], ["(001)"], None, _Detector(),
        detector_roi=(4, 3, 7, 6),
    )

    assert peaks == [{"x": 4, "y": 3, "snr": 10.0, "label": "(001)"}]
