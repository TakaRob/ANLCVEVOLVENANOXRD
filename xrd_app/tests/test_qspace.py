"""Headless tests for pixel → q-space mapping (core/qspace.py).

Geometry is synthesized from known parameters so every assertion is
deterministic and needs no external calibration files. The xrayutilities-backed
path is validated against an independent lab-frame calculation; those tests skip
cleanly if the optional extra is not installed.
"""
from __future__ import annotations

import numpy as np
import pytest

from xrd_app.core import qspace as qs


# A flat-detector geometry with the beam center OFF the detector (the real
# off-axis nano-XRD layout), so 2θ spans a radial band across the frame.
TRUE = qs.Geometry(beam_row=300.5, beam_col=-40.0, distance_m=0.33,
                   pixel_m=qs.DEFAULT_PIXEL_M, rms_deg=0.0, source="synthetic")
SHAPE = (120, 100)


def _synth_tth(geom: qs.Geometry, shape=SHAPE) -> np.ndarray:
    """2θ-per-pixel map for a flat detector with the given geometry (degrees)."""
    rr, cc = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float64)
    rho = geom.pixel_m * np.hypot(rr - geom.beam_row, cc - geom.beam_col)
    return np.degrees(np.arctan2(rho, geom.distance_m))


def test_wavelength_matches_constant():
    # 15 keV → ~0.8266 Å; xu path and hc/E fallback must agree.
    lam = qs.wavelength_angstrom(15000.0)
    assert abs(lam - 0.8266) < 1e-3
    assert abs(lam - qs._HC_EV_A / 15000.0) < 1e-3


def test_qmagnitude_matches_bragg_for_a_reflection():
    # A uniform 2θ map at a known reflection must give |Q| = 4π sin(θ)/λ.
    lam = qs.wavelength_angstrom(15000.0)
    tth = np.full((4, 4), 15.01266)  # (002)
    q = qs.qmagnitude_map(tth, lam)
    expected = 4 * np.pi * np.sin(np.radians(15.01266 / 2)) / lam
    assert np.allclose(q, expected)
    # d = 2π/|Q| should be physically sane for (002)
    d = 2 * np.pi / q.mean()
    assert 3.0 < d < 3.3


def test_recover_geometry_round_trips():
    tth = _synth_tth(TRUE)
    geom = qs.recover_geometry(tth, pixel_m=qs.DEFAULT_PIXEL_M)
    assert abs(geom.beam_row - TRUE.beam_row) < 0.5
    assert abs(geom.beam_col - TRUE.beam_col) < 0.5
    assert abs(geom.distance_m - TRUE.distance_m) < 1e-3
    assert geom.rms_deg < 1e-4  # exact flat model → ~zero residual


def test_labframe_qmag_matches_exact_tth():
    # The lab-frame 3D vectors must reproduce the exact tth-based |Q|.
    lam = qs.wavelength_angstrom(15000.0)
    tth = _synth_tth(TRUE)
    qx, qy, qz = qs.q_vectors_labframe(tth, TRUE, lam)
    qmag = np.sqrt(qx**2 + qy**2 + qz**2)
    assert np.allclose(qmag, qs.qmagnitude_map(tth, lam), atol=1e-9)


def test_annotate_features_indexes_qmap_at_detector_pixel():
    lam = qs.wavelength_angstrom(15000.0)
    tth = _synth_tth(TRUE)
    qx, qy, qz = qs.q_vectors_labframe(tth, TRUE, lam)
    # detector_y = row, detector_x = col (the Bragg-peak location on the detector)
    feats = [
        {"feature_id": 1, "detector_y": 10, "detector_x": 20},
        {"feature_id": 2, "detector_y": 80.4, "detector_x": 55.6},
        {"feature_id": 3, "detector_y": None, "detector_x": None},  # skipped
    ]
    tagged = qs.annotate_features(feats, qx, qy, qz)
    assert len(tagged) == 2
    f = tagged[0]
    assert f["qx"] == pytest.approx(qx[10, 20])
    assert f["q_mag"] == pytest.approx(np.sqrt(qx[10, 20]**2 + qy[10, 20]**2 + qz[10, 20]**2))


def test_annotated_qmag_matches_feature_two_theta():
    # Physics check: a feature's annotated |Q| must equal 4π·sin(θ)/λ for the
    # 2θ that the tth map reports at that feature's detector pixel.
    lam = qs.wavelength_angstrom(15000.0)
    tth = _synth_tth(TRUE)
    qx, qy, qz = qs.q_vectors_labframe(tth, TRUE, lam)
    dy, dx = 70, 33
    feat = {"detector_y": dy, "detector_x": dx}
    tagged = qs.annotate_features([feat], qx, qy, qz)[0]
    expected = 4 * np.pi * np.sin(np.radians(tth[dy, dx] / 2)) / lam
    assert tagged["q_mag"] == pytest.approx(expected, abs=1e-6)


# ----- xrayutilities-backed path (skips if the extra is absent) -----------
xu = pytest.importorskip("xrayutilities")


def test_xu_qmag_matches_labframe_and_tth():
    lam = qs.wavelength_angstrom(15000.0)
    tth = _synth_tth(TRUE)
    qx, qy, qz = qs.q_vectors(tth, TRUE, energy_ev=15000.0, theta_deg=0.0)
    qmag = np.sqrt(qx**2 + qy**2 + qz**2)
    # xu reproduces the exact tth-based |Q| ...
    assert np.allclose(qmag, qs.qmagnitude_map(tth, lam), atol=1e-6)
    # ... and the independent lab-frame |Q| to machine precision.
    lx, ly, lz = qs.q_vectors_labframe(tth, TRUE, lam)
    assert np.allclose(qmag, np.sqrt(lx**2 + ly**2 + lz**2), atol=1e-6)


def test_xu_qmag_invariant_under_theta():
    # Rotating the sample changes the direction sampled, not |Q| at a fixed pixel.
    tth = _synth_tth(TRUE)
    q0 = np.sqrt(sum(a**2 for a in qs.q_vectors(tth, TRUE, theta_deg=0.0)))
    q5 = np.sqrt(sum(a**2 for a in qs.q_vectors(tth, TRUE, theta_deg=5.0)))
    assert np.allclose(q0, q5, atol=1e-6)


# ----- .poni (pyFAI) path (skips if pyFAI is absent) ----------------------
pyfai = pytest.importorskip("pyFAI")


def _zero_tilt_poni(tmp_path, geom=TRUE, shape=SHAPE):
    """A pyFAI Geometry with zero tilt equal to `geom`, saved as a .poni."""
    from pyFAI.geometry import Geometry as PGeom
    from pyFAI.detectors import Detector
    det = Detector(pixel1=geom.pixel_m, pixel2=geom.pixel_m, max_shape=shape)
    ai = PGeom(dist=geom.distance_m, poni1=geom.beam_row * geom.pixel_m,
               poni2=geom.beam_col * geom.pixel_m, rot1=0, rot2=0, rot3=0,
               detector=det, wavelength=qs.wavelength_angstrom(15000.0) * 1e-10)
    p = tmp_path / "zero_tilt.poni"
    ai.save(str(p))
    return p


def test_poni_zero_tilt_matches_xu_frame(tmp_path):
    # A zero-tilt .poni must reproduce the xrayutilities flat path (same frame),
    # both at θ=0 and after the rocking rotation.
    poni = _zero_tilt_poni(tmp_path)
    tth = _synth_tth(TRUE)
    for th in (0.0, 12.5):
        px, py, pz = qs.q_vectors_from_poni(poni, energy_ev=15000.0, theta_deg=th)
        ux, uy, uz = qs.q_vectors(tth, TRUE, energy_ev=15000.0, theta_deg=th)
        # components agree to well under a milli-1/Å (limited by pyFAI vs the
        # analytic flat 2θ model, not by frame/rotation convention)
        assert np.allclose(px, ux, atol=5e-3)
        assert np.allclose(py, uy, atol=5e-3)
        assert np.allclose(pz, uz, atol=5e-3)


def test_poni_geometry_round_trips(tmp_path):
    poni = _zero_tilt_poni(tmp_path)
    geom = qs.geometry_from_poni(poni)
    assert geom.source == "poni"
    assert abs(geom.beam_row - TRUE.beam_row) < 1e-6
    assert abs(geom.beam_col - TRUE.beam_col) < 1e-6
    assert abs(geom.distance_m - TRUE.distance_m) < 1e-9


def test_poni_qmag_invariant_under_theta(tmp_path):
    poni = _zero_tilt_poni(tmp_path)
    q0 = np.sqrt(sum(a**2 for a in qs.q_vectors_from_poni(poni, theta_deg=0.0)))
    q7 = np.sqrt(sum(a**2 for a in qs.q_vectors_from_poni(poni, theta_deg=7.0)))
    assert np.allclose(q0, q7, atol=1e-9)
