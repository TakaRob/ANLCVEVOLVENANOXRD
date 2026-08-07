"""Whole-detector unlimited-width "(no reflections)" reflection set."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from xrd_app.core import io, reflections as refl_io

_BASELINE = (Path(__file__).resolve().parent.parent
             / "CombinedAlgorithms" / "1x1_baseline.py")


def test_copy_bundled_creates_identical_editable_file(tmp_path):
    destination = tmp_path / "Metadata" / "reflections.json"
    bundled = Path(refl_io.__file__).resolve().parent.parent / "assets" / "reflections.json"

    assert refl_io.copy_bundled(destination) == destination
    assert destination.read_bytes() == bundled.read_bytes()


def test_whole_frame_is_one_unlimited_reflection():
    refls = refl_io.whole_frame_reflections()
    assert refls == [{
        "name": refl_io.WHOLE_FRAME_LABEL,
        "two_theta": 0.0,
        "width": refl_io.WHOLE_FRAME_WIDTH,
    }]


def test_whole_frame_builds_one_full_band():
    baseline = io.load_module(_BASELINE)
    tth = np.linspace(2.0, 300.0, 64 * 64).reshape(64, 64)
    refls = refl_io.whole_frame_reflections()
    degs = [r["two_theta"] for r in refls]
    labels = [r["name"] for r in refls]
    bands = baseline.build_tth_band_masks(tth, degs, labels, tth_tolerance=0.4)
    assert list(bands) == [refl_io.WHOLE_FRAME_LABEL]
    assert bands[refl_io.WHOLE_FRAME_LABEL].all()


def test_whole_frame_save_roundtrip(tmp_path):
    refls = refl_io.whole_frame_reflections()
    path = refl_io.save(refls, tmp_path / "reflections.json")
    assert refl_io.read_json(path) == refls
