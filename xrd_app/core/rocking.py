"""Per-track rocking curves: fit intensity(θ) to extract θ_Bragg and mosaicity.

For a single grain, diffracted intensity vs sample θ traces a peak centred at the
Bragg angle θ_Bragg, with a width (FWHM) set by the mosaic/rocking spread. This
module fits each cross-θ track (from :mod:`core.tracking`) to a Gaussian in θ:

    I(θ) = background + amplitude · exp(−½ ((θ − θ_Bragg) / σ)²),
    FWHM = 2·√(2 ln 2) · σ.

Sampling caveat (this study): θ is **clustered** — dense near 3–6°, sparse at
0° and 19.5–20.5°. A fit is only attempted with enough distinct θ; sparser
tracks get moment-based descriptors (argmax θ, intensity-weighted centroid θ)
and a ``too_sparse`` status so downstream code never trusts a 2-point "fit".

Pure module — no click, no Qt. scipy is imported lazily inside the fitter.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

_FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))

ROCKING_COLUMNS = [
    "track_id", "reflection", "ref_tth", "centroid_row", "centroid_col",
    "n_theta", "theta_min", "theta_max", "status",
    # axis 1 — disorder / mosaicity (θ-broadening)
    "theta_bragg", "fwhm", "amplitude", "background", "r_squared",
    # axis 2 — microstrain (radial 2θ shift)
    "tth_com", "microstrain", "strain_breadth_2th",
    # axis 3 — lattice tilt (χ azimuth drift)
    "chi_tilt_rate", "chi_span",
    "theta_at_max", "theta_centroid", "integrated_intensity", "max_intensity",
]


def gaussian(theta, background, amplitude, theta0, fwhm):
    """Rocking-curve model (vectorized over a numpy ``theta``)."""
    import numpy as np
    sigma = fwhm / _FWHM_PER_SIGMA
    return background + amplitude * np.exp(-0.5 * ((np.asarray(theta) - theta0) / sigma) ** 2)


def _moments(thetas, intensities):
    """Fit-free descriptors: argmax θ, intensity-weighted centroid, trapezoid area."""
    theta_at_max = thetas[max(range(len(thetas)), key=lambda i: intensities[i])]
    wsum = sum(intensities)
    centroid = (sum(t * i for t, i in zip(thetas, intensities)) / wsum) if wsum else None
    integrated = 0.0
    for (t0, i0), (t1, i1) in zip(zip(thetas, intensities), list(zip(thetas, intensities))[1:]):
        integrated += 0.5 * (i0 + i1) * (t1 - t0)
    return theta_at_max, centroid, integrated


def fit_rocking_curve(thetas, intensities, min_points: int = 4,
                      min_r2: float = 0.8) -> dict:
    """Fit one track's intensity(θ). Always returns moment descriptors; adds a
    Gaussian fit (θ_Bragg, FWHM, amplitude, background, R²) when θ is sampled
    densely enough, else ``status='too_sparse'``.

    ``thetas`` must be distinct and θ-sorted (use ``tracking.intensity_curve``).
    """
    n = len(thetas)
    out = {
        "n_theta": n,
        "theta_min": min(thetas) if thetas else None,
        "theta_max": max(thetas) if thetas else None,
        "max_intensity": round(max(intensities), 3) if intensities else 0.0,
        "theta_bragg": None, "fwhm": None, "amplitude": None,
        "background": None, "r_squared": None,
    }
    if n == 0:
        out["status"] = "empty"
        out["theta_at_max"] = out["theta_centroid"] = out["integrated_intensity"] = None
        return out

    theta_at_max, centroid, integrated = _moments(thetas, intensities)
    out["theta_at_max"] = theta_at_max
    out["theta_centroid"] = round(centroid, 4) if centroid is not None else None
    out["integrated_intensity"] = round(integrated, 3)

    if n < max(3, min_points):
        out["status"] = "too_sparse"
        return out

    try:
        import numpy as np
        from scipy.optimize import curve_fit

        x = np.asarray(thetas, float)
        y = np.asarray(intensities, float)
        span = float(x.max() - x.min()) or 1.0
        bg0 = float(y.min())
        amp0 = float(y.max() - y.min()) or 1.0
        p0 = [bg0, amp0, float(theta_at_max), span / 2.0]
        # bounds: bg ≥ 0, amp ≥ 0, θ0 within data ± half-span, FWHM in (0, 3·span]
        bounds = ([0.0, 0.0, x.min() - span, 1e-3],
                  [max(bg0, 1e-9) + amp0 * 3 + 1, amp0 * 5 + 1, x.max() + span, 3 * span])
        popt, _ = curve_fit(gaussian, x, y, p0=p0, bounds=bounds, maxfev=10000)
        bg, amp, theta0, fwhm = (float(v) for v in popt)

        resid = y - gaussian(x, *popt)
        ss_res = float(np.sum(resid ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2)) or 1e-12
        r2 = 1.0 - ss_res / ss_tot

        # A meaningful peak must sit inside (or at) the sampled θ range; a θ0 that
        # ran to a bound means the data is monotonic, not peaked here. A peaked
        # shape with poor R² (e.g. fitting a Gaussian across the clustered θ gaps)
        # is flagged 'poor_fit' so it doesn't pollute the FWHM/θ_Bragg stats.
        peaked = (x.min() - 1e-6) <= theta0 <= (x.max() + 1e-6)
        status = "fit" if (peaked and r2 >= min_r2) else ("poor_fit" if peaked else "monotonic")
        out.update({
            "status": status,
            "theta_bragg": round(theta0, 4),
            "fwhm": round(abs(fwhm), 4),
            "amplitude": round(amp, 3),
            "background": round(bg, 3),
            "r_squared": round(r2, 4),
        })
    except Exception as e:  # degrade to moments, never crash the batch
        out["status"] = f"fit_failed:{type(e).__name__}"
    return out


def microstrain(tth_com, ref_tth):
    """Radial microstrain ε = Δd/d from a measured 2θ vs the reference band.

    Differentiating Bragg's law: Δd/d = −cot(θ_B)·Δθ, with θ the *diffraction*
    half-angle (θ = 2θ/2). A measured 2θ above reference ⇒ smaller d ⇒
    compressive (ε < 0). Returns None if inputs are missing.
    """
    if tth_com is None or not ref_tth:
        return None
    theta_b = math.radians(ref_tth / 2.0)
    if math.tan(theta_b) == 0:
        return None
    dtheta = math.radians((tth_com - ref_tth) / 2.0)
    return -dtheta / math.tan(theta_b)


def _wmean(pairs):
    """Weighted mean of (value, weight) pairs, ignoring None values."""
    vw = [(v, w) for v, w in pairs if v is not None and w]
    s = sum(w for _, w in vw)
    return (sum(v * w for v, w in vw) / s) if s else None


def _slope(xs, ys):
    """Least-squares slope dy/dx, or None with < 2 distinct x."""
    pts = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len({x for x, _ in pts}) < 2:
        return None
    n = len(pts)
    sx = sum(x for x, _ in pts); sy = sum(y for _, y in pts)
    sxx = sum(x * x for x, _ in pts); sxy = sum(x * y for x, y in pts)
    denom = n * sxx - sx * sx
    return (n * sxy - sx * sy) / denom if denom else None


def physical_metrics(track: dict) -> dict:
    """The three θ-axis physical metrics for a track (strain + tilt).

    The mosaicity/disorder axis (rocking FWHM) is produced by the curve fit;
    this adds the *radial* (microstrain) and *azimuthal* (lattice tilt) axes
    described in the rocking-curve methodology.
    """
    members = track.get("members", [])
    ref_tth = track.get("ref_tth")
    # axis 2 — microstrain from the intensity-weighted measured 2θ COM
    tth_com = _wmean([(m.get("tth_com"), m.get("intensity")) for m in members])
    eps = microstrain(tth_com, ref_tth)
    strain_breadths = [m.get("tth_fwhm") for m in members if m.get("tth_fwhm") is not None]
    strain_breadth = (sorted(strain_breadths)[len(strain_breadths) // 2]
                      if strain_breadths else None)
    # axis 3 — lattice tilt: χ drift rate with θ, and total χ span. χ is
    # azimuthal, so unwrap (in θ order) before the slope/span — otherwise a
    # ±180° wrap reads as a spurious ~700°/° tilt. members arrive θ-sorted.
    from .tracking import unwrap_deg
    pairs = [(m.get("theta"), m.get("chi_deg")) for m in members
             if m.get("theta") is not None and m.get("chi_deg") is not None]
    pairs.sort(key=lambda p: p[0])
    thetas = [p[0] for p in pairs]
    chis = unwrap_deg([p[1] for p in pairs])
    tilt_rate = _slope(thetas, chis)
    chi_span = (max(chis) - min(chis)) if len(chis) >= 2 else 0.0
    return {
        "tth_com": round(tth_com, 5) if tth_com is not None else None,
        "microstrain": round(eps, 6) if eps is not None else None,
        "strain_breadth_2th": round(strain_breadth, 4) if strain_breadth is not None else None,
        "chi_tilt_rate": round(tilt_rate, 4) if tilt_rate is not None else None,
        "chi_span": round(chi_span, 3),
    }


def fit_tracks(tracks, min_points: int = 4, only_recurrent: bool = True,
               min_r2: float = 0.8, log: Callable[[str], None] = print) -> list:
    """Fit every track's rocking curve → one row per track (``ROCKING_COLUMNS``)."""
    from . import tracking
    rows, n_fit = [], 0
    for t in tracks:
        if only_recurrent and not t.get("is_recurrent"):
            continue
        thetas, intens = tracking.intensity_curve(t)
        fit = fit_rocking_curve(thetas, intens, min_points=min_points, min_r2=min_r2)
        phys = physical_metrics(t)
        if fit["status"] == "fit":
            n_fit += 1
        rows.append({
            "track_id": t.get("track_id"),
            "reflection": t.get("reflection"),
            "ref_tth": t.get("ref_tth"),
            "centroid_row": t.get("centroid_row"),
            "centroid_col": t.get("centroid_col"),
            **{k: fit.get(k) for k in (
                "n_theta", "theta_min", "theta_max", "status",
                "theta_bragg", "fwhm", "amplitude", "background", "r_squared",
                "theta_at_max", "theta_centroid", "integrated_intensity",
                "max_intensity")},
            **phys,
        })
    log(f"Rocking: {n_fit}/{len(rows)} tracks fit a peaked curve "
        f"(rest too sparse / monotonic over the clustered θ sampling)")
    return rows
