"""Noise-reduction implementation and legacy import compatibility."""

import importlib
import sys
from pathlib import Path

import numpy as np

from xrd_app.core import algorithms, io


def test_legacy_noise_module_reexports_core_implementation(monkeypatch):
    legacy_dir = Path(algorithms.__file__).resolve().parent.parent / "NoiseReduction"
    monkeypatch.syspath_prepend(str(legacy_dir))
    sys.modules.pop("noise_reduction_algorithms", None)

    legacy = importlib.import_module("noise_reduction_algorithms")

    for name in legacy.__all__:
        assert getattr(legacy, name) is getattr(algorithms, name)


def test_dynamic_legacy_detector_import_uses_core_noise_implementation(tmp_path):
    detector = tmp_path / "detector.py"
    detector.write_text(
        "from noise_reduction_algorithms import compute_tth_binning\n"
        "IMPL_MODULE = compute_tth_binning.__module__\n"
    )

    loaded = io.load_module(detector)

    assert loaded.IMPL_MODULE == "xrd_app.core.algorithms"


def test_legacy_and_core_calls_have_numeric_parity(monkeypatch):
    legacy_dir = Path(algorithms.__file__).resolve().parent.parent / "NoiseReduction"
    monkeypatch.syspath_prepend(str(legacy_dir))
    legacy = importlib.import_module("noise_reduction_algorithms")
    tth = np.linspace(5.0, 20.0, 24).reshape(4, 6)

    core_result = algorithms.compute_tth_binning(tth, bin_width=0.5)
    legacy_result = legacy.compute_tth_binning(tth, bin_width=0.5)

    assert core_result[2] == legacy_result[2]
    for core_value, legacy_value in zip(core_result[:2] + core_result[3:],
                                        legacy_result[:2] + legacy_result[3:]):
        assert np.array_equal(core_value, legacy_value)
