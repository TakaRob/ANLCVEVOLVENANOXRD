"""Pixel → 3D reciprocal-space (q) mapping.

The rest of the pipeline works in 2θ-radial + χ-azimuth space (see
``geometry.py`` / ``rocking.py``). This module adds the missing third capability:
converting detector pixels (+ the sample θ of a rocking scan) into full 3D
scattering vectors ``Q = (qx, qy, qz)``, so a θ series can be assembled into a
reciprocal-space map and lattice **tilt** (transverse Q) separated from
**microstrain** (radial |Q|).

Two kinds of quantity, with different geometry needs:

* **|Q| per pixel is exact from the 2θ map alone**: ``|Q| = 4π·sin(θ)/λ``. No
  geometry fit is needed — this is the radial reciprocal coordinate (the strain
  axis), and it is what :func:`qmagnitude_map` returns.
* **The direction of Q** (splitting |Q| into qx,qy,qz) needs the beam center and
  sample-detector distance. With a ``.poni`` those are exact (incl. tilt); from a
  bare ``tth.tiff`` we recover a flat detector ``(beam_row, beam_col, distance)``
  by least squares (:func:`recover_geometry`) — no tilt term, so a few-tens-of-
  mdeg approximation (the reported ``rms_deg`` quantifies it).

``xrayutilities`` is the geometry engine for the 3D vectors (it carries the
goniometer circles + tilt and is the basis for a future silx-LUT 3D rebin). It is
an **optional** dependency (``pip install 'xrd-app[qspace]'``); the |Q| and
geometry-recovery paths are pure numpy/scipy and work without it. The
``xrayutilities`` area-detector conversion was validated against an independent
lab-frame calculation (:func:`q_vectors_labframe`) to machine precision — see
``notebooks/qspace_poc.py``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

# Beamline defaults (ISN 26-ID-C). Geometry itself is recovered from the 2θ map;
# only the photon energy and pixel pitch are constants here.
DEFAULT_ENERGY_EV = 15000.0   # 15 keV
DEFAULT_PIXEL_M = 75e-6       # 75 µm pixel

_HC_EV_A = 12398.419843320026  # h·c in eV·Å (λ[Å] = _HC / E[eV])


def _require_xrayutilities():
    try:
        import xrayutilities as xu  # noqa: F401
    except ImportError as e:  # pragma: no cover - depends on environment
        raise ImportError(
            "xrayutilities is required for 3D q-vector conversion. Install it "
            "with `pip install 'xrd-app[qspace]'` or `pip install xrayutilities`."
        ) from e
    return xu


# ─────────────────────────────────────────────────────────────────────
# wavelength + |Q| — exact, no geometry needed
# ─────────────────────────────────────────────────────────────────────
def wavelength_angstrom(energy_ev: float = DEFAULT_ENERGY_EV) -> float:
    """Photon wavelength (Å) from energy (eV). Uses xrayutilities if present,
    else the ``hc/E`` constant (the two agree to < 1e-4 Å)."""
    try:
        import xrayutilities as xu
        return float(xu.en2lam(float(energy_ev)))
    except ImportError:
        return _HC_EV_A / float(energy_ev)


def qmagnitude_map(tth_deg: np.ndarray, lam_A: float) -> np.ndarray:
    """|Q| (1/Å) per pixel from a 2θ-per-pixel map — exact, no geometry fit.

    ``|Q| = 4π·sin(θ)/λ`` with θ = 2θ/2. This is the radial reciprocal-space
    coordinate; comparing it to a reference reflection's |Q| gives microstrain.
    """
    tth = np.asarray(tth_deg, dtype=np.float64)
    return 4.0 * np.pi * np.sin(np.radians(tth) / 2.0) / float(lam_A)


# ─────────────────────────────────────────────────────────────────────
# geometry recovery (flat detector) from a 2θ map
# ─────────────────────────────────────────────────────────────────────
@dataclass
class Geometry:
    """Recovered (or supplied) flat-detector geometry.

    ``beam_row``/``beam_col`` are the direct-beam center in pixels (may fall
    *off* the detector for an off-axis nano-XRD layout); ``distance_m`` is the
    sample-detector distance; ``rms_deg`` is the 2θ fit residual (0 when exact).
    """
    beam_row: float
    beam_col: float
    distance_m: float
    pixel_m: float
    rms_deg: float
    source: str  # "tth-fit" or "poni"

    def as_dict(self) -> dict:
        return asdict(self)


def recover_geometry(tth_deg: np.ndarray, pixel_m: float = DEFAULT_PIXEL_M,
                     subsample: int = 4) -> Geometry:
    """Fit a flat detector (beam center + distance) to a 2θ map.

    Model: ``tan(2θ) = pixel_m·ρ / D`` with ``ρ = hypot(row−cy, col−cx)``. Three
    parameters, Levenberg-Marquardt on a subsampled grid; the RMS residual is
    reported over the full map. There is no tilt term — ``rms_deg`` is the price
    of the flat approximation (supply a ``.poni``-derived 2θ map for tilt).
    """
    from scipy.optimize import least_squares

    tth = np.asarray(tth_deg, dtype=np.float64)
    nrow, ncol = tth.shape
    rr, cc = np.mgrid[0:nrow, 0:ncol]
    sl = (slice(None, None, subsample), slice(None, None, subsample))
    rs = rr[sl].ravel().astype(np.float64)
    cs = cc[sl].ravel().astype(np.float64)
    ts = np.radians(tth[sl].ravel())

    def resid(p):
        cy, cx, dist = p
        rho = pixel_m * np.hypot(rs - cy, cs - cx)
        return np.degrees(np.arctan2(rho, dist)) - np.degrees(ts)

    r0, c0 = np.unravel_index(int(np.argmin(tth)), tth.shape)
    p0 = [float(r0), float(c0), 0.5]
    sol = least_squares(resid, p0, method="lm")
    cy, cx, dist = (float(v) for v in sol.x)

    rho_full = pixel_m * np.hypot(rr - cy, cc - cx)
    tth_fit = np.degrees(np.arctan2(rho_full, dist))
    rms = float(np.sqrt(np.mean((tth_fit - tth) ** 2)))
    return Geometry(cy, cx, abs(dist), pixel_m, rms, "tth-fit")


# ─────────────────────────────────────────────────────────────────────
# 3D Q vectors per pixel
# ─────────────────────────────────────────────────────────────────────
def q_vectors(tth_deg: np.ndarray, geom: Geometry,
              energy_ev: float = DEFAULT_ENERGY_EV,
              theta_deg: float = 0.0):
    """3D scattering vectors ``(qx, qy, qz)`` per pixel in the **sample** frame.

    Rotates by the sample rocking angle ``theta_deg`` so a fixed detector pixel
    maps to the reciprocal-space point actually sampled at that θ (|Q| stays
    invariant; only the direction in the sample frame rotates). Requires
    xrayutilities. Returns three detector-shaped arrays (1/Å).
    """
    xu = _require_xrayutilities()
    nrow, ncol = np.asarray(tth_deg).shape
    # sample circle z+, detector circle z+, primary beam along +x.
    qconv = xu.QConversion(["z+"], ["z+"], [1, 0, 0])
    hxrd = xu.HXRD([1, 0, 0], [0, 0, 1], en=float(energy_ev), qconv=qconv)
    hxrd.Ang2Q.init_area(
        "z-", "y+",
        cch1=geom.beam_row, cch2=geom.beam_col,
        Nch1=nrow, Nch2=ncol,
        pwidth1=geom.pixel_m * 1e3, pwidth2=geom.pixel_m * 1e3,  # mm
        distance=geom.distance_m * 1e3,                          # mm
    )
    qx, qy, qz = hxrd.Ang2Q.area(float(theta_deg), 0.0)
    return np.asarray(qx), np.asarray(qy), np.asarray(qz)


def _q_from_dirs(dx, dy, dz, lam_A: float, theta_deg: float = 0.0):
    """Sample-frame Q from per-pixel lab **directions** (beam +x, z vertical).

    ``Q = k·(ŝ − x̂)`` with ``ŝ`` the scattered unit vector; then the sample
    rocking rotation about the vertical (z) axis by ``theta_deg``. The frame and
    rotation match :func:`q_vectors` (xrayutilities) to machine precision — see
    the cross-checks in ``tests/test_qspace.py`` — so the flat (xu) and .poni
    (pyFAI) paths produce Q in one common frame.
    """
    k = 2.0 * np.pi / float(lam_A)
    n = np.sqrt(dx * dx + dy * dy + dz * dz)
    qx = k * (dx / n) - k
    qy = k * (dy / n)
    qz = k * (dz / n)
    if theta_deg:
        t = np.radians(float(theta_deg))
        c, s = np.cos(t), np.sin(t)
        qx, qy = c * qx + s * qy, -s * qx + c * qy
    return qx, qy, qz


def geometry_from_poni(poni_path) -> Geometry:
    """Read a pyFAI ``.poni`` into a :class:`Geometry` (for reporting).

    The direct-beam center is ``poni / pixel`` (exact only at zero tilt; with
    tilt the true beam center shifts slightly, but this is display-only — the
    q-vectors themselves use pyFAI's full tilted geometry). ``rms_deg`` is 0
    (the geometry is measured, not fit) and ``source`` is ``"poni"``.
    """
    from . import geometry as _geom
    pyFAI = _geom._require_pyfai()
    ai = pyFAI.load(str(poni_path))
    pix1 = float(ai.pixel1)
    return Geometry(
        beam_row=float(ai.poni1) / pix1,
        beam_col=float(ai.poni2) / float(ai.pixel2),
        distance_m=float(ai.dist),
        pixel_m=pix1,
        rms_deg=0.0,
        source="poni",
    )


def q_vectors_from_poni(poni_path, energy_ev: float = DEFAULT_ENERGY_EV,
                        theta_deg: float = 0.0, shape=None):
    """Tilt-accurate 3D Q per pixel from a pyFAI ``.poni``.

    Unlike the flat :func:`recover_geometry` fit, a ``.poni`` carries the detector
    **tilt** (rot1/rot2/rot3), so per-pixel scattered directions — and hence the
    direction of Q — are exact. pyFAI supplies the per-pixel lab positions; the
    sample-frame assembly + rocking rotation reuse :func:`_q_from_dirs`, so the
    result shares the frame of :func:`q_vectors`. Requires pyFAI
    (``pip install 'xrd-app[poni]'``). Returns ``(qx, qy, qz)`` (1/Å).

    ``shape`` (rows, cols) is used only if the .poni's detector has no shape.
    """
    from . import geometry as _geom
    pyFAI = _geom._require_pyfai()
    ai = pyFAI.load(str(poni_path))
    if getattr(ai.detector, "shape", None) is None and shape is not None:
        ai.detector.shape = tuple(shape)
    # pyFAI lab frame: calc_pos_zyx -> (z=beam, y=slow/vertical, x=fast/horizontal).
    # Map into the q_vectors frame: beam=+x_lab <- z_pyfai; y_lab <- x_pyfai;
    # z_lab(vertical) <- -y_pyfai  (validated against xrayutilities at zero tilt).
    z, y, x = ai.calc_pos_zyx(corners=False)
    lam = wavelength_angstrom(energy_ev)
    return _q_from_dirs(z, x, -y, lam, theta_deg)


def q_vectors_labframe(tth_deg: np.ndarray, geom: Geometry, lam_A: float):
    """Reference ``(qx, qy, qz)`` at θ=0 by direct lab-frame geometry (no xu).

    Incident beam ``k_i`` along +x, ``|k| = 2π/λ``; each pixel sits at lab
    position ``(D, dy, dz)`` and ``Q = k_f − k_i``. Used to validate
    :func:`q_vectors` (their |Q| agree to machine precision). The sample-θ
    rotation is **not** applied here — this is the θ=0 reference only.
    """
    tth = np.asarray(tth_deg, dtype=np.float64)
    nrow, ncol = tth.shape
    rr, cc = np.mgrid[0:nrow, 0:ncol].astype(np.float64)
    k = 2.0 * np.pi / float(lam_A)
    dy = (cc - geom.beam_col) * geom.pixel_m
    dz = (rr - geom.beam_row) * geom.pixel_m
    dx = np.full_like(dy, geom.distance_m)
    norm = np.sqrt(dx * dx + dy * dy + dz * dz)
    qx = k * (dx / norm) - k
    qy = k * (dy / norm)
    qz = k * (dz / norm)
    return qx, qy, qz


# ─────────────────────────────────────────────────────────────────────
# annotate detected features with q-coordinates
# ─────────────────────────────────────────────────────────────────────
def annotate_features(features: Sequence[dict], qx: np.ndarray,
                      qy: np.ndarray, qz: np.ndarray,
                      row_key: str = "detector_y",
                      col_key: str = "detector_x") -> List[dict]:
    """Tag each feature with ``(qx, qy, qz, q_mag)`` at its **detector** pixel.

    A feature's reciprocal-space position is set by where its Bragg peak lands on
    the detector (``detector_y`` = row, ``detector_x`` = col) — *not* by its
    spatial scan-grid bin (``center_row``/``center_col``), which is where on the
    sample the grain sits. Features missing a detector location are skipped.
    """
    nrow, ncol = qx.shape
    qmag = np.sqrt(qx ** 2 + qy ** 2 + qz ** 2)
    out: List[dict] = []
    for f in features:
        r, c = f.get(row_key), f.get(col_key)
        if r is None or c is None:
            continue
        ri = int(np.clip(round(float(r)), 0, nrow - 1))
        ci = int(np.clip(round(float(c)), 0, ncol - 1))
        out.append({
            **f,
            "qx": float(qx[ri, ci]),
            "qy": float(qy[ri, ci]),
            "qz": float(qz[ri, ci]),
            "q_mag": float(qmag[ri, ci]),
        })
    return out


# ─────────────────────────────────────────────────────────────────────
# IO
# ─────────────────────────────────────────────────────────────────────
def save_qmap(path, qx: np.ndarray, qy: np.ndarray, qz: np.ndarray,
              geom: Geometry, meta: Optional[dict] = None) -> Path:
    """Write the per-pixel Q field + geometry/meta to a compressed ``.npz``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    qmag = np.sqrt(qx ** 2 + qy ** 2 + qz ** 2)
    payload = {
        "qx": qx, "qy": qy, "qz": qz, "q_mag": qmag,
        "beam_row": geom.beam_row, "beam_col": geom.beam_col,
        "distance_m": geom.distance_m, "pixel_m": geom.pixel_m,
        "rms_deg": geom.rms_deg, "geometry_source": geom.source,
    }
    for k, v in (meta or {}).items():
        payload[k] = v
    np.savez_compressed(path, **payload)
    return path


def summary(geom: Geometry, theta_deg: Optional[float], energy_ev: float,
            lam_A: float, qmag: np.ndarray, n_features: int = 0) -> dict:
    """A small JSON-able summary of one scan's q-map."""
    return {
        "energy_ev": float(energy_ev),
        "wavelength_A": float(lam_A),
        "theta_deg": (None if theta_deg is None else float(theta_deg)),
        "geometry": geom.as_dict(),
        "q_mag_min": float(np.min(qmag)),
        "q_mag_max": float(np.max(qmag)),
        "n_features_annotated": int(n_features),
    }
