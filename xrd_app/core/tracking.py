"""Link shapes across the θ sweep into grain *tracks* (the rocking analogue of
the spatial cross-bin Union-Find).

A single crystalline grain at sample position (X,Y) diffracts a given reflection
over a range of θ (its rocking width). Across the rocking series it therefore
appears as a shape at the *same de-skewed spatial bin*, in the *same reflection
band*, in several adjacent-θ scans, with χ (detector azimuth) drifting smoothly.
This module groups per-scan shapes into those tracks.

Registration note: for this study every scan shares an identical raster grid
(151×167 native; same at every θ — the spot is stable, ``samy`` fixed), so the
binned ``(center_row, center_col)`` is directly comparable across θ with no
resampling. Matching is therefore: same reflection band + spatial proximity
within ``match_tol`` bins. If a future series has stage drift, add a per-scan
(row, col) offset upstream (feature-based registration) before calling this.

Pure module — no click, no Qt. The CLI (`xrd-app track`) feeds it the feature
rows produced by :mod:`core.aggregate` and writes the JSON/CSV.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Callable, Iterable, Optional

# θ (sample rocking angle, degrees) per scan for the 5%_DI_Yes_GB series
# (logbook 10270, scans 203–214). Scan_0206 is incomplete and excluded.
# 203 and 214 are the same orientation (20.5°) — the repeatability check.
THETA_BY_SCAN = {
    "Scan_0203": 20.5, "Scan_0204": 20.0, "Scan_0205": 19.5,
    "Scan_0206": 6.0,  # incomplete — excluded from binning/analysis
    "Scan_0207": 5.5,  "Scan_0208": 5.0,  "Scan_0209": 4.5,
    "Scan_0210": 4.0,  "Scan_0211": 3.5,  "Scan_0212": 3.0,
    "Scan_0213": 0.0,  "Scan_0214": 20.5,
}

# Columns for the one-row-per-track summary CSV.
TRACK_COLUMNS = [
    "track_id", "reflection", "ref_tth",
    "centroid_row", "centroid_col", "pos_drift",
    "n_theta", "n_members", "theta_min", "theta_max", "theta_at_max_I",
    "chi_mean", "chi_span", "chi_max_step", "max_intensity", "is_recurrent",
]


def theta_of(scan: str, theta_by_scan: Optional[dict] = None):
    """θ (deg) for a scan name, or ``None`` if unknown."""
    return (theta_by_scan or THETA_BY_SCAN).get(scan)


def unwrap_deg(values):
    """Unwrap a sequence of angles (deg) so steps take the short way around the
    circle — χ is azimuthal (±180° wraps), so −179°→+179° is a 2° step, not 358°.
    """
    if not values:
        return []
    out = [values[0]]
    for v in values[1:]:
        prev = out[-1]
        out.append(prev + (((v - prev) + 180.0) % 360.0 - 180.0))
    return out


def _intensity(f: dict) -> float:
    """Rocking-curve signal for a feature: integrated intensity, else peak."""
    for k in ("sum_integrated", "peak_intensity", "mean_intensity"):
        v = f.get(k)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    return 0.0


def _prep(features: Iterable[dict], theta_by_scan: dict):
    """Normalize raw feature rows into the minimal fields tracking needs.

    Drops rows without a spatial bin or a known θ (e.g. an excluded scan).
    """
    out = []
    for f in features:
        row, col = f.get("center_row"), f.get("center_col")
        if row is None or col is None:
            continue
        scan = f.get("scan")
        theta = theta_of(scan, theta_by_scan)
        if theta is None:
            continue
        out.append({
            "scan": scan,
            "theta": float(theta),
            "reflection": f.get("reflection"),
            "ref_tth": f.get("ref_tth"),
            "center_row": float(row),
            "center_col": float(col),
            "chi_deg": _num(f.get("chi_deg")),
            "tth_com": _num(f.get("tth_com")),
            "peak_intensity": _num(f.get("peak_intensity")),
            "sum_integrated": _num(f.get("sum_integrated")),
            "intensity": _intensity(f),
            "detector_x": f.get("detector_x"),
            "detector_y": f.get("detector_y"),
            "n_bins": f.get("n_bins"),
            "chi_fwhm": _num(f.get("chi_fwhm")),
            "tth_fwhm": _num(f.get("tth_fwhm")),
            "feature_id": f.get("feature_id"),
        })
    return out


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def build_tracks(features: Iterable[dict], match_tol: float = 2.0,
                 min_theta: int = 2, theta_by_scan: Optional[dict] = None,
                 log: Callable[[str], None] = print) -> list:
    """Group features into cross-θ tracks.

    Greedy single-linkage within each reflection band: features are seeded
    strongest-first, and each is attached to the nearest existing track whose
    running spatial centroid is within ``match_tol`` bins; otherwise it starts a
    new track. Same grid across θ ⇒ ``(center_row, center_col)`` compares directly.

    Args:
        features:   feature rows (from :func:`core.aggregate.aggregate`).
        match_tol:  max centroid distance (in bins) to join a track.
        min_theta:  a track is only "recurrent" (H1) if seen at ≥ this many
                    distinct θ; sparser tracks are still emitted but flagged.
        theta_by_scan: scan→θ map (defaults to the 203–214 table).

    Returns a list of track dicts sorted by (reflection, descending max
    intensity), each with a θ-sorted ``members`` list.
    """
    theta_by_scan = theta_by_scan or THETA_BY_SCAN
    feats = _prep(features, theta_by_scan)

    by_ref = defaultdict(list)
    for f in feats:
        by_ref[f["reflection"]].append(f)

    tracks = []
    next_id = 0
    for ref in sorted(by_ref, key=lambda r: (r is None, r)):
        group = by_ref[ref]
        group.sort(key=lambda f: -(f["intensity"] or 0.0))  # strong seeds first
        clusters = []  # each: {row, col, n, members}
        for f in group:
            best, best_d = None, match_tol
            for c in clusters:
                d = math.hypot(f["center_row"] - c["row"], f["center_col"] - c["col"])
                if d <= best_d:
                    best, best_d = c, d
            if best is None:
                clusters.append({"row": f["center_row"], "col": f["center_col"],
                                 "n": 1, "members": [f]})
            else:
                best["members"].append(f)
                n = best["n"] + 1  # incremental centroid
                best["row"] += (f["center_row"] - best["row"]) / n
                best["col"] += (f["center_col"] - best["col"]) / n
                best["n"] = n
        for c in clusters:
            tracks.append(_summarize_track(next_id, ref, c["members"], min_theta))
            next_id += 1

    tracks.sort(key=lambda t: (str(t["reflection"]), -t["max_intensity"]))
    n_rec = sum(1 for t in tracks if t["is_recurrent"])
    log(f"Built {len(tracks)} tracks ({n_rec} recurrent, ≥{min_theta} θ) "
        f"from {len(feats)} features across {len({f['scan'] for f in feats})} scans")
    return tracks


def _summarize_track(track_id: int, reflection, members: list, min_theta: int) -> dict:
    members = sorted(members, key=lambda m: m["theta"])
    rows = [m["center_row"] for m in members]
    cols = [m["center_col"] for m in members]
    crow, ccol = _mean(rows), _mean(cols)
    # spatial spread: rms distance of members from centroid (bins)
    drift = math.sqrt(_mean([(r - crow) ** 2 + (c - ccol) ** 2
                             for r, c in zip(rows, cols)])) if members else 0.0

    thetas = [m["theta"] for m in members]
    intens = [m["intensity"] for m in members]
    i_max = max(range(len(members)), key=lambda i: intens[i]) if members else 0
    n_theta = len({m["scan"] for m in members})

    # χ is azimuthal — unwrap (in θ order) before span/step so wraparound at
    # ±180° doesn't masquerade as a huge tilt. members are already θ-sorted.
    chis_raw = [m["chi_deg"] for m in members if m["chi_deg"] is not None]
    chis = unwrap_deg(chis_raw)
    chi_mean = _mean(chis) if chis else None
    chi_span = (max(chis) - min(chis)) if len(chis) >= 2 else 0.0
    # largest χ jump between consecutive θ (a single grain should drift smoothly;
    # a big jump hints the track merged two grains).
    chi_steps = [abs(b - a) for a, b in zip(chis, chis[1:])] if len(chis) >= 2 else []
    chi_max_step = max(chi_steps) if chi_steps else 0.0

    ref_tths = [m["ref_tth"] for m in members if m["ref_tth"] is not None]
    return {
        "track_id": track_id,
        "reflection": reflection,
        "ref_tth": ref_tths[0] if ref_tths else None,
        "centroid_row": round(crow, 2), "centroid_col": round(ccol, 2),
        "pos_drift": round(drift, 3),
        "n_theta": n_theta, "n_members": len(members),
        "theta_min": min(thetas) if thetas else None,
        "theta_max": max(thetas) if thetas else None,
        "theta_at_max_I": members[i_max]["theta"] if members else None,
        "chi_mean": round(chi_mean, 2) if chi_mean is not None else None,
        "chi_span": round(chi_span, 2),
        "chi_max_step": round(chi_max_step, 2),
        "max_intensity": round(max(intens), 2) if intens else 0.0,
        "is_recurrent": n_theta >= min_theta,
        "members": [{
            "scan": m["scan"], "theta": m["theta"],
            "center_row": m["center_row"], "center_col": m["center_col"],
            "chi_deg": m["chi_deg"], "tth_com": m["tth_com"],
            "peak_intensity": m["peak_intensity"],
            "sum_integrated": m["sum_integrated"], "intensity": m["intensity"],
            "detector_x": m["detector_x"], "detector_y": m["detector_y"],
            "tth_fwhm": m["tth_fwhm"], "chi_fwhm": m["chi_fwhm"],
            "feature_id": m["feature_id"],
        } for m in members],
    }


def intensity_curve(track: dict):
    """Collapse a track's members to one (θ, intensity) point per scan.

    Multiple shapes from the same scan in one track (over-segmentation) are
    combined by taking the max intensity at that θ. Returns (thetas, intensities)
    sorted by θ — the input to a rocking-curve fit.
    """
    by_theta = {}
    for m in track["members"]:
        t = m["theta"]
        by_theta[t] = max(by_theta.get(t, 0.0), m["intensity"] or 0.0)
    items = sorted(by_theta.items())
    return [t for t, _ in items], [i for _, i in items]


def track_summary_rows(tracks: list) -> list:
    """One row per track (the ``members`` list dropped) for the summary CSV."""
    return [{k: t.get(k) for k in TRACK_COLUMNS} for t in tracks]


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0
