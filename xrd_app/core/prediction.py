"""Predict which shapes should appear across the θ sweep, and compare to observed.

The recurrent grain tracks (from :mod:`core.tracking`) are the *model* of real
features: a grain that diffracts a reflection at θ_i and θ_k should also light up
at the sampled θ in between (within its rocking window). This module turns that
into a falsifiable prediction and scores it.

Definitions, per sampled θ:
  * **predicted set** P(θ) — recurrent tracks whose θ-window [θ_min, θ_max]
    covers θ (the grain is expected to be diffracting there).
  * **observed set** O(θ) — track members actually detected at θ.
  * **TP** predicted and present · **FN** predicted but absent (a *gap* —
    the strongest evidence for/against H1) · **FP** present but not predicted
    by any recurrent track (singletons → the candidate-noise pool).

  recall   = ΣTP / Σ(TP+FN)   — "do the predicted shapes appear?"
  precision= ΣTP / Σ(TP+FP)   — "are detections predicted, or noise?"

Repeatability floor: scans 203 and 214 are the *same* orientation (θ=20.5°), so
matching their two shape catalogs directly gives the empirical detection
reproducibility — the ceiling any prediction recall can realistically reach.

χ smoothness: a single grain's χ drifts smoothly with θ; a large χ step inside a
track flags a probable two-grain merge.

Pure module — no click, no Qt. Emits a report dict and a Markdown rendering.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Optional

from . import tracking


# ── repeatability floor (203 vs 214) ──────────────────────────────────
def repeatability(features, scan_a: str, scan_b: str, match_tol: float = 2.0) -> dict:
    """Match two scans' shape catalogs (same orientation) → reproducibility.

    Greedy nearest-neighbour match within a reflection band and ``match_tol``
    bins. Returns counts and a symmetric reproducibility fraction
    2·matched / (n_a + n_b).
    """
    a = [f for f in features if f.get("scan") == scan_a]
    b = [f for f in features if f.get("scan") == scan_b]
    by_ref_b = defaultdict(list)
    for i, f in enumerate(b):
        by_ref_b[f.get("reflection")].append(i)

    used = set()
    matched = 0
    for fa in a:
        cands = by_ref_b.get(fa.get("reflection"), [])
        best, best_d = None, match_tol
        for j in cands:
            if j in used:
                continue
            fb = b[j]
            d = _dist(fa, fb)
            if d is not None and d <= best_d:
                best, best_d = j, d
        if best is not None:
            used.add(best)
            matched += 1
    n_a, n_b = len(a), len(b)
    denom = n_a + n_b
    return {
        "scan_a": scan_a, "scan_b": scan_b,
        "n_a": n_a, "n_b": n_b, "matched": matched,
        "reproducibility": round(2 * matched / denom, 4) if denom else None,
        "recall_a_in_b": round(matched / n_a, 4) if n_a else None,
    }


def degradation_check(features, scan_a: str, scan_b: str,
                      substrate=("ITO",)) -> dict:
    """Reflection-selective count comparison between two same-orientation scans.

    Beam-induced radiolysis degrades the (organic) perovskite phase while the
    inorganic substrate (ITO) is robust, so a drop in *film* reflection counts
    with a *stable substrate* count — at the same θ, same spot — is a dose/
    beam-damage signature rather than a registration or noise effect. Returns
    per-reflection counts plus film/substrate retention ratios.
    """
    sub = set(substrate)
    a = [f for f in features if f.get("scan") == scan_a]
    b = [f for f in features if f.get("scan") == scan_b]

    def counts(fs):
        c = {}
        for f in fs:
            c[f.get("reflection")] = c.get(f.get("reflection"), 0) + 1
        return c
    ca, cb = counts(a), counts(b)
    refls = sorted(set(ca) | set(cb), key=lambda r: (r is None, r))
    per_refl = {r: {"a": ca.get(r, 0), "b": cb.get(r, 0)} for r in refls}

    film_a = sum(v for r, v in ca.items() if r not in sub)
    film_b = sum(v for r, v in cb.items() if r not in sub)
    sub_a = sum(v for r, v in ca.items() if r in sub)
    sub_b = sum(v for r, v in cb.items() if r in sub)
    film_ret = round(film_b / film_a, 3) if film_a else None
    sub_ret = round(sub_b / sub_a, 3) if sub_a else None

    note = None
    if film_ret is not None and sub_ret is not None:
        if film_ret < 0.5 <= sub_ret * 1.0 and sub_ret >= 0.7:
            note = ("film reflections drop sharply while the substrate holds — "
                    "consistent with beam-induced perovskite degradation "
                    "(radiolysis) over the series' accumulated dose")
        else:
            note = ("no clear film-vs-substrate divergence — degradation not "
                    "evident at the reflection-count level")
    return {
        "scan_a": scan_a, "scan_b": scan_b, "substrate": list(sub),
        "per_reflection": per_refl,
        "film_a": film_a, "film_b": film_b, "film_retention": film_ret,
        "substrate_a": sub_a, "substrate_b": sub_b, "substrate_retention": sub_ret,
        "note": note,
    }


def _dist(fa, fb):
    try:
        return math.hypot(float(fa["center_row"]) - float(fb["center_row"]),
                          float(fa["center_col"]) - float(fb["center_col"]))
    except (KeyError, TypeError, ValueError):
        return None


# ── predicted-vs-observed over the sweep ───────────────────────────────
def predicted_vs_observed(tracks, sampled_thetas) -> dict:
    """Score recurrent-track predictions against detections at every sampled θ."""
    sampled = sorted(set(sampled_thetas))
    tp = fn = fp = 0
    gaps = []           # (track_id, θ) predicted-but-missing
    recurrent = [t for t in tracks if t.get("is_recurrent")]

    for t in recurrent:
        lo, hi = t["theta_min"], t["theta_max"]
        present = {m["theta"] for m in t["members"]}
        window = [th for th in sampled if lo - 1e-9 <= th <= hi + 1e-9]
        for th in window:
            if th in present:
                tp += 1
            else:
                fn += 1
                gaps.append({"track_id": t["track_id"], "reflection": t["reflection"],
                             "theta": th})
    # FP: detections from non-recurrent (singleton) tracks — not predicted.
    fp = sum(len({m["theta"] for m in t["members"]})
             for t in tracks if not t.get("is_recurrent"))

    recall = tp / (tp + fn) if (tp + fn) else None
    precision = tp / (tp + fp) if (tp + fp) else None
    return {
        "n_recurrent_tracks": len(recurrent),
        "n_singleton_tracks": len(tracks) - len(recurrent),
        "tp": tp, "fn": fn, "fp": fp,
        "recall": round(recall, 4) if recall is not None else None,
        "precision": round(precision, 4) if precision is not None else None,
        "f1": round(2 * precision * recall / (precision + recall), 4)
              if precision and recall else None,
        "gaps": gaps,
    }


def chi_smoothness(tracks, step_threshold: float = 10.0) -> dict:
    """χ-continuity stats over recurrent tracks (smooth ⇒ single grain)."""
    steps = [t["chi_max_step"] for t in tracks
             if t.get("is_recurrent") and t.get("chi_max_step") is not None]
    if not steps:
        return {"n": 0, "median_max_step": None, "frac_smooth": None,
                "step_threshold": step_threshold}
    steps_sorted = sorted(steps)
    median = steps_sorted[len(steps_sorted) // 2]
    frac_smooth = sum(1 for s in steps if s <= step_threshold) / len(steps)
    return {"n": len(steps), "median_max_step": round(median, 3),
            "frac_smooth": round(frac_smooth, 4), "step_threshold": step_threshold}


def rocking_summary(rocking_rows) -> dict:
    """Aggregate rocking-fit quality + θ_Bragg/FWHM ranges over fitted tracks."""
    if not rocking_rows:
        return {"n_total": 0, "n_fit": 0}
    fits = [r for r in rocking_rows if r.get("status") == "fit"]

    def _nums(group, key):
        # CSV reads leave blanks as "" — coerce and drop anything non-numeric.
        out = []
        for g in group:
            v = g.get(key)
            try:
                if v in ("", None):
                    continue
                out.append(float(v))
            except (TypeError, ValueError):
                continue
        return sorted(out)

    def _stat(key, group=fits, nd=3):
        vals = _nums(group, key)
        if not vals:
            return None
        return {"min": round(vals[0], nd), "median": round(vals[len(vals) // 2], nd),
                "max": round(vals[-1], nd)}
    # microstrain / tilt span the whole recurrent set, not just peaked fits
    rec = [r for r in rocking_rows if r.get("status") not in ("empty",)]
    r2s = _nums(fits, "r_squared")
    return {
        "n_total": len(rocking_rows), "n_fit": len(fits),
        "median_r2": round(r2s[len(r2s) // 2], 4) if r2s else None,
        "fwhm": _stat("fwhm"), "theta_bragg": _stat("theta_bragg"),
        "microstrain": _stat("microstrain", rec, nd=6),
        "strain_breadth_2th": _stat("strain_breadth_2th", rec, nd=4),
        "chi_tilt_rate": _stat("chi_tilt_rate", rec, nd=4),
    }


def build_report(tracks, features, theta_by_scan: Optional[dict] = None,
                 match_tol: float = 2.0, repeat_pair=("Scan_0203", "Scan_0214"),
                 rocking_rows=None) -> dict:
    """Assemble the full predicted-vs-observed report dict."""
    theta_by_scan = theta_by_scan or tracking.THETA_BY_SCAN
    scans_present = sorted({f.get("scan") for f in features if f.get("scan")})
    sampled_thetas = [theta_by_scan[s] for s in scans_present if s in theta_by_scan]

    per_scan = defaultdict(int)
    for f in features:
        per_scan[f.get("scan")] += 1

    pvo = predicted_vs_observed(tracks, sampled_thetas)
    rep = repeatability(features, repeat_pair[0], repeat_pair[1], match_tol=match_tol)
    dose = degradation_check(features, repeat_pair[0], repeat_pair[1])
    chi = chi_smoothness(tracks)
    rock = rocking_summary(rocking_rows)

    # Interpretation: recall vs the repeatability ceiling.
    floor = rep.get("reproducibility")
    verdict = _verdict(pvo.get("recall"), floor)
    return {
        "scans": scans_present,
        "n_scans": len(scans_present),
        "sampled_thetas": sorted(set(sampled_thetas)),
        "n_features": len(features),
        "per_scan_features": dict(sorted(per_scan.items())),
        "n_tracks": len(tracks),
        "match_tol_bins": match_tol,
        "predicted_vs_observed": pvo,
        "repeatability_floor": rep,
        "dose_check": dose,
        "chi_smoothness": chi,
        "rocking": rock,
        "verdict": verdict,
    }


def _verdict(recall, floor) -> str:
    if recall is None:
        return "insufficient data (no recurrent tracks)"
    if floor is None:
        return f"recall={recall:.2f} (no repeatability floor available)"
    if recall >= floor:
        return (f"recall {recall:.2f} meets/exceeds the 203-vs-214 repeatability "
                f"floor {floor:.2f} — predicted shapes appear as well as an identical "
                f"orientation reproduces. H1/H2 supported.")
    return (f"recall {recall:.2f} is below the repeatability floor {floor:.2f} — "
            f"predicted shapes appear less often than identical-orientation "
            f"reproducibility; gaps exceed the noise floor (tune --match-tol or "
            f"investigate stage drift).")


# ── Markdown rendering ─────────────────────────────────────────────────
def to_markdown(report: dict) -> str:
    pvo = report["predicted_vs_observed"]
    rep = report["repeatability_floor"]
    chi = report["chi_smoothness"]
    rock = report["rocking"]
    L = []
    L.append("# Rocking-Series Prediction Report — `5%_DI_Yes_GB` (scans 203–214)\n")
    L.append(f"_{report['n_scans']} scans, {report['n_features']} shapes, "
             f"{report['n_tracks']} tracks, match tol {report['match_tol_bins']} bins._\n")

    L.append("## Headline\n")
    L.append(f"> {report['verdict']}\n")

    L.append("## Predicted vs observed (across the θ sweep)\n")
    L.append("| metric | value |")
    L.append("|---|---|")
    L.append(f"| recurrent tracks (the predictor) | {pvo['n_recurrent_tracks']} |")
    L.append(f"| singleton tracks (candidate noise) | {pvo['n_singleton_tracks']} |")
    L.append(f"| TP (predicted & present) | {pvo['tp']} |")
    L.append(f"| FN (predicted, missing = gaps) | {pvo['fn']} |")
    L.append(f"| FP (present, not predicted) | {pvo['fp']} |")
    L.append(f"| **recall** (do predicted shapes appear?) | **{_fmt(pvo['recall'])}** |")
    L.append(f"| **precision** (are detections predicted?) | **{_fmt(pvo['precision'])}** |")
    L.append(f"| F1 | {_fmt(pvo['f1'])} |\n")

    L.append("## Repeatability floor — 203 vs 214 (same orientation, θ=20.5°)\n")
    L.append(f"- shapes in {rep['scan_a']}: {rep['n_a']}, in {rep['scan_b']}: {rep['n_b']}")
    L.append(f"- matched within {report['match_tol_bins']} bins: {rep['matched']}")
    L.append(f"- **reproducibility** = {_fmt(rep['reproducibility'])} "
             f"(per-shape spatial match — a detection noise floor; low at 3×3/SNR, "
             f"see the dose check below for the stronger reflection-level signal)\n")

    dose = report.get("dose_check")
    if dose:
        L.append("## Dose / beam-damage check — 203 vs 214 (reflection-selective)\n")
        L.append("| reflection | 203 | 214 |")
        L.append("|---|---|---|")
        for r, c in dose["per_reflection"].items():
            tag = " *(substrate)*" if r in dose["substrate"] else ""
            L.append(f"| {r}{tag} | {c['a']} | {c['b']} |")
        L.append("")
        L.append(f"- **film** reflections: {dose['film_a']} → {dose['film_b']} "
                 f"(retention {_fmt(dose['film_retention'])})")
        L.append(f"- **substrate** ({', '.join(dose['substrate'])}): "
                 f"{dose['substrate_a']} → {dose['substrate_b']} "
                 f"(retention {_fmt(dose['substrate_retention'])})")
        if dose.get("note"):
            L.append(f"\n> {dose['note']}\n")

    L.append("## χ(θ) smoothness (single-grain check)\n")
    if chi["n"]:
        L.append(f"- recurrent tracks with χ series: {chi['n']}")
        L.append(f"- median max χ step between adjacent θ: {_fmt(chi['median_max_step'])}°")
        L.append(f"- fraction smooth (≤{chi['step_threshold']}°): {_fmt(chi['frac_smooth'])}\n")
    else:
        L.append("- (no recurrent tracks with a χ series)\n")

    L.append("## Rocking-curve fits — the three θ-axis physics\n")
    if rock.get("n_total"):
        L.append(f"- tracks scored: {rock['n_total']}, peaked fits: {rock['n_fit']}, "
                 f"median R²: {_fmt(rock.get('median_r2'))}\n")
        L.append("| axis | metric | min | median | max |")
        L.append("|---|---|---|---|---|")
        if rock.get("fwhm"):
            fw = rock["fwhm"]
            L.append(f"| disorder/mosaicity | rocking FWHM (°) | {fw['min']} | {fw['median']} | {fw['max']} |")
        if rock.get("theta_bragg"):
            tb = rock["theta_bragg"]
            L.append(f"| — | θ_Bragg (°) | {tb['min']} | {tb['median']} | {tb['max']} |")
        if rock.get("microstrain"):
            ms = rock["microstrain"]
            L.append(f"| microstrain | ε = Δd/d | {ms['min']} | {ms['median']} | {ms['max']} |")
        if rock.get("strain_breadth_2th"):
            sb = rock["strain_breadth_2th"]
            L.append(f"| — | strain breadth, 2θ FWHM (°) | {sb['min']} | {sb['median']} | {sb['max']} |")
        if rock.get("chi_tilt_rate"):
            ct = rock["chi_tilt_rate"]
            L.append(f"| lattice tilt | dχ/dθ (°/°) | {ct['min']} | {ct['median']} | {ct['max']} |")
        L.append("")
        L.append("_Lower rocking FWHM ⇒ larger, better-oriented grains — the expected "
                 "signature of the 5% H₂O (DI_Yes) crystallization modulator. θ sampling "
                 "is clustered, so FWHM is most reliable for tracks living near θ≈3–6°._\n")
    else:
        L.append("- (run `xrd-app rocking`, then `xrd-app predict --rocking …` to include fits)\n")

    L.append("## Per-scan shape counts\n")
    L.append("| scan | θ (°) | shapes |")
    L.append("|---|---|---|")
    tb = report.get("_theta_by_scan", tracking.THETA_BY_SCAN)
    for scan, n in report["per_scan_features"].items():
        L.append(f"| {scan} | {tb.get(scan, '?')} | {n} |")
    L.append("")
    if pvo["gaps"]:
        L.append(f"\n_{len(pvo['gaps'])} predicted-but-missing (gap) detections; "
                 f"see prediction_report.json `predicted_vs_observed.gaps`._")
    return "\n".join(L)


def _fmt(v):
    return "—" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))
