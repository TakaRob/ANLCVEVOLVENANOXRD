"""Pole figures / inverse-pole-figure color keys from xrd-app catalogs.

WHAT THIS IS (read first — the physics decides what is possible)
================================================================
The micress/orix tutorial builds an *IPF map* by giving every grain a full
3-DOF crystal orientation (a quaternion) and coloring the sample-Z direction by
where it lands in the crystal fundamental zone. That workflow needs the full
orientation of each grain.

Our nano-XRD data does **not** carry full orientations. Each detected feature is
a single Bragg reflection: we know its crystal plane family (the ``reflection``
label, e.g. ``(001)``), its azimuth on the detector ring (``chi_deg``) and its
radial position (2θ). One reflection = one measured plane normal = at most 2 of
the 3 orientation DOF. You can only *solve* a grain's full orientation where two
**independent** reflection families land on the same spatial bin — and on real
data that essentially never happens (Scan_0203, 1x1 territory: 1 of 121 bins).

So the honest, data-faithful object is the **pole figure**, not the EBSD IPF map:

    Pole figure (PF)          fix a crystal plane (hkl), plot where its normal
                              points in the sample  -> texture. WE HAVE THIS.
    Inverse pole figure (IPF) fix a sample direction, plot which crystal axes
                              align with it, in crystal coords, IPF-colored.
                              Needs full orientation per grain -> WE DO NOT HAVE
                              it per grain, but we DO know each measured pole's
                              crystal direction exactly (it is the hkl), so we
                              can place + IPF-color the measured poles (Sec 4).

This module: (1) loads binned + territorial catalogs, (2) turns features into
measured poles, (3) plots a single pole figure, (4) draws the orix IPF color key
and places our measured reflections in the crystal fundamental zone.

Run:  python xrd_app/notebooks/inverse_pole_figures.py
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

# xrd-app core (pure, importable — no Qt)
from xrd_app.core import catalogs, qspace

# ── crystal directions for the perovskite reflection labels (pseudocubic) ──
# PbI2 / ITO are different phases and are excluded from the perovskite figures.
HKL = {
    "(001)": (0, 0, 1), "(002)": (0, 0, 2),
    "(011)": (0, 1, 1), "(111)": (1, 1, 1),
    "(012)": (0, 1, 2), "(112)": (1, 1, 2),
}


# ─────────────────────────────────────────────────────────────────────
# 1. LOAD DATA — binned and territorial catalogs
# ─────────────────────────────────────────────────────────────────────
def load_shapes(catalog_path):
    """Load one shapes/combined catalog -> list of kept feature dicts.

    Works for BOTH binned and territorial catalogs — they share the exact same
    feature schema (``core/catalogs.load_features_any``). What differs is only
    the *grid* the bins map to (a real row/col grid vs. territory ids); the
    per-feature fields we need for a pole figure are identical:

        reflection, ref_tth, chi_deg, detector_x, detector_y,
        tth_fwhm, chi_fwhm, peak_intensity, center_row, center_col,
        intensity_profile{bin -> {tth, chi, det_x, det_y, intensity}}
    """
    kept, _filtered = catalogs.load_features_any(str(catalog_path))
    return kept


def find_catalogs(project_root, scan, bin_size=3):
    """Discover the newest binned and territorial shapes catalogs for a scan.

    Returns (binned_path, territory_path); either may be None if absent.
    Binned = a plain grid catalog; territorial = the ``*_territory*`` variant.
    """
    from xrd_app.config import DataManager
    dm = DataManager(str(project_root))
    labels = Path(dm.labels_dir(scan))
    shapes = sorted(labels.glob("*shapes*.h5")) + sorted(labels.glob("*combined*.h5"))
    territory = next((p for p in shapes if "territory" in p.name), None)
    binned = next((p for p in shapes if "territory" not in p.name), None)
    return binned, territory


# ─────────────────────────────────────────────────────────────────────
# 2. MEASURED POLES — from features to plottable directions
# ─────────────────────────────────────────────────────────────────────
def poles_native(features, refs=None):
    """The zero-assumption pole figure coordinates: (2θ, χ) straight from the
    detector. radius = ref 2θ (deg), angle = chi_deg. Every value is measured;
    no geometry fit needed. Returns arrays (tth, chi_deg, reflection, weight)."""
    tth, chi, ref, wt = [], [], [], []
    for f in features:
        r = f.get("reflection")
        if refs is not None and r not in refs:
            continue
        c = f.get("chi_deg")
        if c is None:
            continue
        tth.append(float(f.get("ref_tth") or 0.0))
        chi.append(float(c))
        ref.append(r)
        wt.append(float(f.get("peak_intensity") or 1.0))
    return np.array(tth), np.array(chi), np.array(ref, dtype=object), np.array(wt)


def poles_qspace(features, tth_map, theta_deg=0.0, energy_ev=qspace.DEFAULT_ENERGY_EV):
    """Full 3D pole directions in the sample frame via core/qspace.

    Uses the exact machinery already in the app: recover a flat-detector geometry
    from the 2θ map, build the per-pixel Q field at the scan's rocking angle
    ``theta_deg``, then tag each feature at its Bragg pixel (detector_x/y).
    Returns the annotated features (each gains qx, qy, qz, q_mag) + the geometry.

    NOTE the frame (see qspace.py): beam along +x, z vertical, y horizontal-
    transverse. theta_deg only rotates the whole figure about z; pass the scan's
    real sample θ for absolute orientation, or 0 for a relative texture view.
    """
    geom = qspace.recover_geometry(tth_map)
    qx, qy, qz = qspace.q_vectors(tth_map, geom, energy_ev=energy_ev,
                                  theta_deg=theta_deg)
    annotated = qspace.annotate_features(features, qx, qy, qz)
    return annotated, geom


# ─────────────────────────────────────────────────────────────────────
# 3. SINGLE POLE FIGURE  (the "plot one" step)
# ─────────────────────────────────────────────────────────────────────
def plot_pole_figure(features, title="Pole figure", refs=None, ax=None):
    """One pole figure in native detector coords: polar plot, angle = χ,
    radius = 2θ, one color per reflection, marker size ∝ peak intensity.

    This is the correct single-scan texture plot for single-reflection data:
    tight χ clusters = strong in-plane texture; a full ring = random in-plane.
    """
    import matplotlib.pyplot as plt

    tth, chi, ref, wt = poles_native(features, refs=refs)
    if ax is None:
        _fig, ax = plt.subplots(figsize=(6, 6),
                                subplot_kw={"projection": "polar"})
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)

    order = [r for r in HKL if r in set(ref)] if refs is None else list(refs)
    cmap = plt.get_cmap("tab10")
    sizes = 20 + 180 * (wt / wt.max()) if len(wt) and wt.max() > 0 else 30
    for i, r in enumerate(order):
        m = ref == r
        if not m.any():
            continue
        s = sizes[m] if np.ndim(sizes) else sizes
        ax.scatter(np.radians(chi[m]), tth[m], s=s, alpha=0.7,
                   color=cmap(i % 10), edgecolors="k", linewidths=0.3, label=r)
    ax.set_rlabel_position(90)
    ax.set_title(title, pad=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=8,
              title="reflection\n(radius = 2θ°)")
    return ax


# ─────────────────────────────────────────────────────────────────────
# 4. orix — IPF of the measured poles in the crystal fundamental zone
# ─────────────────────────────────────────────────────────────────────
# NOTE: orix could not be installed in this env (its dep pycifrw needs a C
# compiler; Python 3.14 has no wheel yet).  `apt install gcc` — or a Python
# <=3.12 venv with a prebuilt wheel — makes the two functions below run.
#
# This is the honest analog of the micress tutorial's orix half.  The tutorial
# colors grain *orientations*; we have none, but we DO know each measured pole's
# crystal direction exactly (its hkl).  So we place every measured reflection in
# the crystal fundamental zone and IPF-color it by crystal axis — an inverse
# pole figure of the measured plane normals.
#
# Point group: halide perovskite (e.g. MAPbI3) is tetragonal at room temp, so we
# default to **4/mmm** — its fundamental sector splits the c-axis (00l) from the
# a/b directions, which is the physically correct triangle for these films.
# The (hkl) *numbers* in HKL stay the same (they are the reflection labels from
# reflections.py); only the symmetry used to fold directions + color them
# changes.  To reproduce the micress-style cubic triangle instead, pass
# point_group="m-3m" (pseudocubic Oh) — the (00l)/(0kl) labels index cleanly
# there and it matches the FCC example in the orix docs.

_POINT_GROUPS = {"4/mmm": "D4h", "m-3m": "Oh", "mmm": "D2h"}


def _perovskite_millers(features, point_group="4/mmm"):
    """(Miller directions, intensity weights, reflection counts) for the
    perovskite features that carry a known hkl.  PbI2/ITO are dropped."""
    from orix.crystal_map import Phase
    from orix.vector import Miller

    hkls, weights, counts = [], [], Counter()
    for f in features:
        d = HKL.get(f.get("reflection"))
        if d is None:
            continue
        hkls.append(d)
        weights.append(float(f.get("peak_intensity") or 1.0))
        counts[f.get("reflection")] += 1
    if not hkls:
        raise ValueError("no perovskite reflections with a known hkl")
    phase = Phase(point_group=point_group)
    m = Miller(uvw=np.asarray(hkls, dtype=float), phase=phase)
    return m, np.asarray(weights), counts


def plot_ipf_scatter(features, point_group="4/mmm", title="Measured poles (IPF)"):
    """Colored IPF scatter: every measured pole placed at its crystal direction
    in the fundamental sector, IPF-colored by crystal axis, sized by intensity.
    Requires orix. Returns the matplotlib Figure."""
    import matplotlib.pyplot as plt  # noqa: F401
    from orix.plot import DirectionColorKeyTSL

    m, weights, counts = _perovskite_millers(features, point_group)
    ckey = DirectionColorKeyTSL(m.phase.point_group)
    rgb = ckey.direction2color(m)

    sizes = 30 + 220 * (weights / weights.max() if weights.max() else 1.0)
    fig = m.scatter(hemisphere="upper", c=rgb, s=sizes, return_figure=True,
                    ec="k", lw=0.3)
    fig.axes[0].set_title(
        title + "\n" + ", ".join(f"{k}:{v}" for k, v in counts.most_common()))
    ckey.plot()  # the color key legend (fundamental sector) in its own figure
    return fig


def plot_ipf_density(features, point_group="4/mmm", vmax=None,
                     title="Pole density (IPF)"):
    """IPF *density* of the measured poles — the closest thing to an EBSD IPF for
    single-reflection data: how the measured plane normals concentrate in the
    crystal fundamental zone (texture strength). Requires orix."""
    import matplotlib.pyplot as plt

    m, _weights, counts = _perovskite_millers(features, point_group)
    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={
        "projection": "ipf", "symmetry": m.phase.point_group})
    ax.pole_density_function(m, vmin=0, vmax=vmax)
    ax.set_title(title + "\n" +
                 ", ".join(f"{k}:{v}" for k, v in counts.most_common()))
    return fig


# ─────────────────────────────────────────────────────────────────────
# demo / self-check
# ─────────────────────────────────────────────────────────────────────
def _demo():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import tifffile

    root = "/mnt/z/isn/2026-1/2026-1-Luo/ANLCVEVOLVENANOXRD/Scans179-226Perovskite"
    scan = "Scan_0203"
    binned, territory = find_catalogs(root, scan, bin_size=1)
    cat = territory or binned
    print(f"catalog: {cat}")
    feats = load_shapes(cat)
    print(f"features: {len(feats)}  reflections: "
          f"{Counter(f['reflection'] for f in feats).most_common()}")

    # (2) qspace poles (uses xrayutilities + the scan's 2θ map)
    tth_map = tifffile.imread(f"{root}/Metadata/tth.tiff")
    annotated, geom = poles_qspace(feats, tth_map, theta_deg=0.0)
    print(f"geometry: beam=({geom.beam_row:.0f},{geom.beam_col:.0f}) "
          f"D={geom.distance_m*1e3:.1f}mm rms={geom.rms_deg:.3f}°  "
          f"annotated={len(annotated)}")

    # (3) single pole figure
    plot_pole_figure(feats, title=f"{scan} pole figure (perovskite)",
                     refs=list(HKL))
    plt.savefig("/tmp/pole_figure.png", dpi=130, bbox_inches="tight")
    print("wrote /tmp/pole_figure.png")

    # (4) orix IPF of the measured poles (needs orix; see Section 4 note).
    # Halide perovskite is tetragonal -> 4/mmm; pass "m-3m" for the pseudocubic
    # (micress-style cubic) triangle instead.
    try:
        plot_ipf_scatter(feats, point_group="4/mmm")
        plt.savefig("/tmp/ipf_scatter.png", dpi=130, bbox_inches="tight")
        plot_ipf_density(feats, point_group="4/mmm")
        plt.savefig("/tmp/ipf_density.png", dpi=130, bbox_inches="tight")
        print("wrote /tmp/ipf_scatter.png, /tmp/ipf_density.png")
    except ImportError:
        print("orix not installed — skipping Section 4 (see the note there).")
    except Exception as e:  # orix API varies by version — report, don't crash
        print(f"orix IPF step: {type(e).__name__}: {e}")


if __name__ == "__main__":
    _demo()
