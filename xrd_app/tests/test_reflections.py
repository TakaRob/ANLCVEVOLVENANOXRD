"""Whole-detector "(no reflections)" reflection set.

A set whose entries all share one label must OR-merge (in the detector's
``build_tth_band_masks``) into a single band covering the whole detector — the
"no known Bragg reflections" workflow, expressed as an ordinary reflection set
with no special-case code.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from xrd_app.core import io, reflections as refl_io

_BASELINE = (Path(__file__).resolve().parent.parent
             / "CombinedAlgorithms" / "1x1_baseline.py")


def test_whole_frame_single_label():
    refls = refl_io.whole_frame_reflections()
    labels = {r["name"] for r in refls}
    assert labels == {refl_io.WHOLE_FRAME_LABEL}
    assert len(refls) > 1  # actually tiled, not a single entry


def test_whole_frame_spacing_below_tolerance():
    # tiles must be closer than the 0.4° detection tolerance so the merged band
    # is contiguous (no dead gaps between adjacent tiles)
    refls = refl_io.whole_frame_reflections(spacing=0.3)
    tths = sorted(r["two_theta"] for r in refls)
    gaps = np.diff(tths)
    assert gaps.max() <= 0.3 + 1e-9


def test_whole_frame_merges_to_one_full_band():
    baseline = io.load_module(_BASELINE)
    # a synthetic 2θ map spanning the tiled range
    tth = np.linspace(2.0, 30.0, 64 * 64).reshape(64, 64)
    refls = refl_io.whole_frame_reflections()
    degs = [r["two_theta"] for r in refls]
    labels = [r["name"] for r in refls]
    bands = baseline.build_tth_band_masks(tth, degs, labels, tth_tolerance=0.4)
    assert list(bands) == [refl_io.WHOLE_FRAME_LABEL]  # exactly one merged band
    assert bands[refl_io.WHOLE_FRAME_LABEL].all()      # covers the whole frame


def test_whole_frame_clamped_to_tth_map():
    tth = np.full((16, 16), 10.0)
    tth[0, 0] = 8.0
    tth[-1, -1] = 12.0
    refls = refl_io.whole_frame_reflections(tth, spacing=0.3, margin=1.0)
    tths = [r["two_theta"] for r in refls]
    assert min(tths) <= 8.0 and max(tths) >= 12.0        # spans observed range
    assert min(tths) >= 8.0 - 1.0 - 0.3                  # but stays near it (clamped)
    assert max(tths) <= 12.0 + 1.0 + 0.3


def test_whole_frame_save_roundtrip(tmp_path):
    refls = refl_io.whole_frame_reflections()
    refl_io.save(refls, tmp_path / "reflections.json", tmp_path / "reflections.py")
    mod = io.load_module(tmp_path / "reflections.py")
    assert set(mod.deg_labels) == {refl_io.WHOLE_FRAME_LABEL}
    assert len(mod.degs) == len(refls)
