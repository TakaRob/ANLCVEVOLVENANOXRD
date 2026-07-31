"""Summed-image ROI detector contract and CVEvolve export."""
from __future__ import annotations

import numpy as np

from xrd_app.core import roi_detection


def test_tuned_wieghold_detector_is_first_discovered_algorithm(tmp_path):
    class DM:
        cvevolve_dir = tmp_path

    algorithms = roi_detection.discover_algorithms(DM())
    assert [algorithm["file"].split("/")[-1] for algorithm in algorithms[:3]] == [
        "wieghold_peak_conservative.py",
        "wieghold_peak_balanced.py",
        "wieghold_peak_very_conservative.py",
    ]
    assert algorithms[0]["default_sensitivity"] == 0.65


def test_detect_intersects_candidates_with_detector_bounds(tmp_path):
    algorithm = tmp_path / "detector.py"
    algorithm.write_text(
        "def detect_rois(image, **params):\n"
        "    return [(-4, -3, 3, 2), (8, 7, 14, 12), "
        "(-9, 1, -2, 4), (12, 1, 15, 4)]\n"
    )

    candidates = roi_detection.detect(np.zeros((10, 12)), algorithm)

    assert [item["roi"] for item in candidates] == [(0, 0, 3, 2), (8, 7, 12, 10)]


def test_baseline_detects_bright_summed_image_spots():
    yy, xx = np.mgrid[:120, :140]
    image = (500 * np.exp(-((xx - 35) ** 2 + (yy - 45) ** 2) / (2 * 3 ** 2)) +
             350 * np.exp(-((xx - 100) ** 2 + (yy - 80) ** 2) / (2 * 4 ** 2)))

    candidates = roi_detection.detect(
        image, roi_detection.default_algorithm(), sensitivity=2.0,
        min_distance=10, max_rois=10)
    centers = [((r["roi"][0] + r["roi"][2]) / 2,
                (r["roi"][1] + r["roi"][3]) / 2) for r in candidates]

    assert any(abs(x - 35) < 5 and abs(y - 45) < 5 for x, y in centers)
    assert any(abs(x - 100) < 5 and abs(y - 80) < 5 for x, y in centers)
