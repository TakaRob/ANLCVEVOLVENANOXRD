"""Voigt shape finder — Phase-2 linking + pseudo-Voigt profile verification.

Ports the Voigt verification from the per-frame ``CombinedAlgorithms`` family
onto the standard ``peaks → shapes`` pipeline, so it rides the same engine as
``gaussian`` / ``territory`` instead of a separate combined runner. It keeps a
linked feature when its **cross-bin intensity profile** (distance-from-centre vs
intensity) fits a pseudo-Voigt radial peak (R² test, with compact/monotonic
fallbacks for small clusters) — a stricter, shape-aware alternative to the
gaussian-profile monotonicity check. Unlike the old combined path it uses the
per-bin intensities already produced by ``peaks`` (no raw-frame re-probe), so it
emits full ``intensity_profile`` / metrics and populates the Device/Orientation
maps.

Linking is shared: this dispatches to the grid linker (binned sizes) or the
coordinate/neighbor linker (1×1, when ``run_shapes`` passes a neighbor map), so
``xrd-app shapes --algorithm voigt`` works at any bin size.

Select with ``xrd-app shapes --algorithm voigt``. Registered in ``catalog.json``.
"""

import numpy as np

# Dual import (see territory.py): bare names work under core.io.load_module
# (sibling dir on sys.path); dotted works as a package import (py_compile/tests).
try:  # noqa: F401  (names re-exported for run_shapes)
    import gaussian as _g
    import territory as _t
except ImportError:  # pragma: no cover
    from . import gaussian as _g
    from . import territory as _t

DEFAULT_LINK_TOLERANCE = _g.DEFAULT_LINK_TOLERANCE
estimate_beam_center = _g.estimate_beam_center


# ── Phase 2: linking (shared with gaussian / territory) ────────────
def set_centroids(centroid_rc):
    """Register territory centroids for coordinate linking (see territory)."""
    _t.set_centroids(centroid_rc)


def link_peaks(all_detections, a, b=None, link_tolerance=DEFAULT_LINK_TOLERANCE):
    """Link peaks, dispatching by call shape.

    ``run_shapes`` calls ``link_peaks(dets, neighbors, link_tol)`` for a
    coordinate/territorial mapping (``a`` is the neighbor dict, ``b`` the
    tolerance) and ``link_peaks(dets, n_rows, n_cols, link_tol)`` for a grid.
    """
    if isinstance(a, dict):                       # coordinate / neighbor linking
        tol = b if b is not None else link_tolerance
        return _t.link_peaks(all_detections, a, tol)
    return _g.link_peaks(all_detections, a, b, link_tolerance)   # grid linking


# ── Phase 3: pseudo-Voigt profile verification ─────────────────────
def pseudo_voigt(r, amplitude, sigma, gamma):
    """Pseudo-Voigt radial profile (Thompson–Cox–Hastings η). Port of the
    combined detector's ``pseudo_voigt`` (CombinedAlgorithms)."""
    fg = 2.0 * sigma * np.sqrt(2.0 * np.log(2.0))
    fl = 2.0 * gamma
    f = (fg**5 + 2.69269*fg**4*fl + 2.42843*fg**3*fl**2
         + 4.47163*fg**2*fl**3 + 0.07842*fg*fl**4 + fl**5) ** 0.2
    if f < 1e-10:
        return np.zeros_like(r)
    eta = 1.36603*(fl/f) - 0.47719*(fl/f)**2 + 0.11116*(fl/f)**3
    eta = np.clip(eta, 0, 1)
    return amplitude * (eta / (1.0 + (r/(gamma+1e-10))**2)
                        + (1 - eta) * np.exp(-r**2/(2.0*sigma**2+1e-10)))


def check_voigt_profile(members, r2_thresh=0.25, max_width=8.0,
                        monotonic_thresh=0.60, cv_flat_thresh=0.05):
    """Return (is_voigt, reason) from the feature's cross-bin radial profile.

    Builds distance-from-(intensity-weighted)-centre vs normalized intensity over
    the linked bins and fits a pseudo-Voigt; keeps on a good R² fit, with compact
    (tiny cluster) and monotonic fallbacks. Adapted from the combined detector's
    ``verify_voigt`` to use the per-bin ``cleaned_intensity`` the peak stage
    already produced (member tuple = (bin_key, pi, row, col, x, y, peak_dict)).
    """
    if len(members) < 2:
        return False, "isolated: single-bin detection"

    intensities = np.array([m[6]['cleaned_intensity'] for m in members], dtype=float)
    rows = np.array([m[2] for m in members], dtype=float)
    cols = np.array([m[3] for m in members], dtype=float)

    mx = float(intensities.max())
    if mx <= 0:
        return False, "non-positive peak intensity"
    norm = intensities / (mx + 1e-10)

    cv = float(norm.std() / (norm.mean() + 1e-10))
    if cv < cv_flat_thresh:
        return False, f"flat profile: CV={cv:.3f} (no clear center)"

    # Intensity-weighted centroid → radial distances.
    w = np.maximum(norm, 0.0)
    ws = float(w.sum())
    if ws > 0:
        pr, pc = float((rows*w).sum()/ws), float((cols*w).sum()/ws)
    else:
        imax = int(np.argmax(norm)); pr, pc = rows[imax], cols[imax]
    distances = np.sqrt((rows-pr)**2 + (cols-pc)**2)
    n = len(members)

    # Tiny clusters: accept if compact around the centre (mirrors verify_voigt).
    if n <= 2:
        return True, f"compact: {n} bins"

    from scipy.optimize import curve_fit

    def voff(r, a, s, g, o):
        return o + pseudo_voigt(r, a, s, g)

    try:
        rm = float(distances.max()) + 1.0
        fw = np.maximum(norm, 0.05)
        popt, _ = curve_fit(voff, distances, norm, p0=[1.0, rm/3, rm/3, 0.0],
                            bounds=([0.1, 0.1, 0.01, -0.3], [2.0, rm*2, rm*2, 0.7]),
                            sigma=1.0/fw, maxfev=600)
        fitted = voff(distances, *popt)
        ss_res = float(np.sum(((norm - fitted) * fw) ** 2))
        ss_tot = float(np.sum(((norm - norm.mean()) * fw) ** 2))
        r2 = 1.0 - ss_res / (ss_tot + 1e-10)

        if popt[1] > max_width and popt[2] > max_width:
            return False, f"too wide (R2={r2:.3f})"
        et = r2_thresh if n > 4 else max(0.10, r2_thresh - 0.10)
        if r2 > et:
            return True, f"Voigt-like: R2={r2:.3f}, {n} bins"
    except (RuntimeError, ValueError):
        r2 = float("nan")

    # Monotonic fallback (distance ↑ ⇒ intensity ↓), as in verify_voigt.
    nc = nt = 0
    for i in range(n):
        for j in range(i + 1, n):
            if abs(distances[i] - distances[j]) < 0.1:
                continue
            nt += 1
            if (distances[i] < distances[j]) == (norm[i] > norm[j]):
                nc += 1
    em = monotonic_thresh if n >= 5 else max(0.50, monotonic_thresh - 0.05)
    if nt > 3 and nc / nt > em:
        return True, f"Voigt (monotonic): {nc/nt:.0%}, {n} bins"
    r2s = f"{r2:.3f}" if r2 == r2 else "fit-failed"
    return False, f"non-Voigt: R2={r2s}"


# ── Phase 3: characterize (gaussian's body, Voigt keep-decision) ────
def characterize_features(features, beam_center=None, tth_map=None, ref_tth_map=None):
    """Classify each feature kept/filtered by the Voigt profile test, with the
    same metrics as ``gaussian.characterize_features`` (only the keep-decision
    differs)."""
    kept, filtered = [], []
    for members in features:
        bins_in_feature = set(m[0] for m in members)
        is_voigt, reason = check_voigt_profile(members)

        intensities = [m[6]['cleaned_intensity'] for m in members]
        snrs = [m[6]['snr'] for m in members]
        xs = [m[4] for m in members]
        ys = [m[5] for m in members]
        rows = [m[2] for m in members]
        cols = [m[3] for m in members]

        imax = int(np.argmax(intensities))
        reflection = members[imax][6]['label']
        det_x, det_y = int(np.mean(xs)), int(np.mean(ys))

        feature_info = {
            "reflection": reflection,
            "detector_x": det_x,
            "detector_y": det_y,
            "peak_intensity": float(max(intensities)),
            "mean_snr": float(np.mean(snrs)),
            "n_bins": len(bins_in_feature),
            "spatial_extent": sorted(bins_in_feature),
            "center_bin": members[imax][0],
            "center_row": rows[imax],
            "center_col": cols[imax],
            "intensity_profile": _g._best_per_bin(members, tth_map, beam_center),
            "reason": reason,
        }
        if ref_tth_map is not None and reflection in ref_tth_map:
            feature_info["ref_tth"] = ref_tth_map[reflection]

        profile = feature_info["intensity_profile"]
        bin_chis, bin_tths, bin_weights = [], [], []
        for entry in profile.values():
            if not isinstance(entry, dict):
                continue
            wt = entry.get("integrated", entry.get("intensity", 0))
            if wt <= 0:
                continue
            if "chi" in entry:
                bin_chis.append(entry["chi"])
            if "tth" in entry:
                bin_tths.append(entry["tth"])
            bin_weights.append(wt)

        if len(bin_chis) >= 3 and len(bin_weights) == len(bin_chis):
            wa = np.array(bin_weights); ca = np.array(bin_chis)
            wn = wa / wa.sum(); mu = np.dot(wn, ca)
            feature_info["chi_fwhm"] = round(2.3548 * np.sqrt(np.dot(wn, (ca-mu)**2)), 4)

        if len(bin_tths) >= 3 and len(bin_weights) == len(bin_tths):
            ref_val = ref_tth_map.get(reflection) if ref_tth_map else None
            if ref_val is not None:
                wa = np.array(bin_weights); ta = np.array(bin_tths) - ref_val
                wn = wa / wa.sum(); mu = np.dot(wn, ta)
                feature_info["tth_fwhm"] = round(2.3548 * np.sqrt(np.dot(wn, (ta-mu)**2)), 4)

        if beam_center is not None:
            by, bx = beam_center
            feature_info["chi_deg"] = round(
                float(np.degrees(np.arctan2(det_y - by, det_x - bx))), 1)

        (kept if is_voigt else filtered).append(feature_info)

    return kept, filtered
