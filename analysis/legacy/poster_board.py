# %% [markdown]
# # High-Resolution Structural Mapping of Perovskite Thin Films
# ## Resolving (111) and (001) Microstructures via Automated Nano-XRD Analysis and a CLI-Based Computer Vision Pipeline
#
# **ISN 26-ID-C beamline · Advanced Photon Source, Argonne National Laboratory**
# 15 keV nano-focused beam · 75 µm CCD pixel · perovskite / halide thin films
#
# *We built the open **`xrd-app` CLI** to automate the analysis of large nano-XRD
# datasets — turning thousands of raw detector frames per scan into physics-checked
# grain-orientation and grain-size maps, reproducibly and without hand-picking.*
#
# ---
# This file is the **poster board**, written as Jupyter-style `# %%` cells (run in
# VS Code / Jupyter / Spyder, or `python poster_board.py`). Read it top to bottom:
# **Abstract → Motivation → Methods → Results → Algorithms → Conclusions & Next Steps.**
# Every figure is either loaded from disk (already computed) or regenerated here
# from the shape catalogs — no beamline data or GUI required.

# %%
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy.ndimage import gaussian_filter
except Exception:  # scipy always present in this env, but degrade rather than crash
    gaussian_filter = None

# Repo root (this file lives next to feature_summary_table.py). Fall back to cwd.
BASE = Path("/mnt/c/Users/trobson/OneDrive - Argonne National Laboratory/2026-1_Luo")
if not (BASE / "feature_summary_table.py").exists():
    BASE = Path.cwd()

# Scan trees that hold the shape catalogs (fast local WSL copies).
PROJECTS = {
    "179": "/home/takaji/179-201",
    "182": "/home/takaji/179-201",
    "203": "/home/takaji/rocking_203_214",
    "207": "/home/takaji/rocking_203_214",
    "215": "/home/takaji/215-226",
    "218": "/home/takaji/215-226",
}
BIN, ALGO = 3, "gaussian"

# ── master text size ────────────────────────────────────────────────
# One knob for every label, title, tick and annotation in this file. Raise
# FONT to enlarge all text together (relative sizes are preserved); FONT = 1.0
# is the original look. Every explicit fontsize below is multiplied by FONT,
# and the rcParams here set the scaled defaults for anything not set explicitly.
FONT = 1.75
plt.rcParams.update({k: v * FONT for k, v in {
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9, "figure.titlesize": 13,
}.items()})
# ────────────────────────────────────────────────────────────────────

print(f"BASE       : {BASE}")
print(f"figures dir: {BASE}")


def load_shapes(scan):
    """Return (catalog dict, total grid bins) for a scan at 3x3, or (None, 0)."""
    root = PROJECTS[scan]
    sp = Path(root) / f"Labels/Scan_0{scan}/{ALGO}_shapes_{BIN}x{BIN}.json"
    gp = Path(root) / f"Metadata/Scan_0{scan}/grid_mapping_{BIN}x{BIN}.json"
    if not sp.exists() or not gp.exists():
        return None, 0
    cat = json.load(open(sp))
    g = json.load(open(gp))
    return cat, g["n_bin_rows"] * g["n_bin_cols"], g["n_bin_rows"], g["n_bin_cols"]


# %% [markdown]
# ## Abstract
#
# Nano-focused X-ray diffraction (nano-XRD) at the APS 26-ID-C beamline resolves
# how crystal grains in perovskite/halide solar-cell films are **oriented** and
# **sized** across a sample surface — but a single scan is *thousands* of CCD
# frames, one per spatial position, and manual peak-picking does not scale. We
# built **`xrd-app`**, a CLI-first computer-vision pipeline that streams the raw
# HDF5, detects Bragg peaks frame-by-frame, and links them across space into
# physically real **grains ("shapes")**. Two design choices carry the accuracy:
# **spatial binning**, which raises per-position signal-to-noise ~linearly with bin
# size (signal ∝ N², shot noise ∝ N), and **contouring** — a Union-Find + Gaussian
# -profile filter that merges thousands of scattered detections into coherent
# grain outlines and rejects noise. Applied to six trusted focus scans across
# three surface treatments, the pipeline shows that a **grain-boundary (GB)
# passivation** preserves the preferred (001) texture (largest-grain 001:111 size
# ratio ≈ 3.4 with GB → ≈ 0.9 without), and that a **5 % Cl + DI** treatment
# consolidates (001) coverage (001:111 summed-area ratio ≈ 2.9 → ≈ 4.6). A single
# (001) grain is resolved spanning **1,122 binned positions — 15.7 % of the entire
# scan** — with a 0.76° azimuthal spread, demonstrating the map's fidelity.

# %% [markdown]
# ## Motivation
#
# Perovskite photovoltaic performance is governed by *microstructure*: which
# lattice planes face the surface (texture) and how large and continuous the grains
# are. Nano-XRD can measure both at ~100 nm resolution, but the raw output is
# unwieldy — a single Scan_0179 map is 84 × 85 = **7,140 spatial positions**, each
# a full detector image, and a rocking series multiplies that again. Hand-analysis
# is slow, irreproducible, and cannot keep pace with a beamtime that produces
# dozens of scans. We needed an **automated, scriptable, physics-checked** engine:
# one that runs headless over an entire project, records exactly what it did
# (MLflow-tracked), validates every detection against the known 2θ reflection bands
# (a peak off-band is a bug, not a discovery), and produces the *same* device maps
# whether driven from a GUI button or a batch job. That engine is `xrd-app`.

# %% [markdown]
# ## Methods
#
# **The CLI is the engine; the GUI is a face over it.** The whole analysis is a
# chain of `xrd-app` commands, each reading and writing the same JSON/HDF5 so any
# result is reproducible headless:
#
# ```
# init → scan-detect → link (2θ map + reflections + positions)
#      → grid → bin → peaks (Phase 1, per position)
#      → shapes (Phase 2, link across positions) → Device / Orientation Map
# ```
#
# **Binning** sums N×N neighbouring frames before detection, trading spatial
# resolution for signal-to-noise so weak Bragg peaks cross threshold (recall is the
# quantity we chase; CVEvolve optimises mean-F2). **Contouring** (the `peaks →
# shapes` step) links per-position detections into grains via Union-Find and keeps
# only those with a bright, monotonic Gaussian centre. Grains are then characterised
# by azimuth **χ** (orientation), radial breadth (Δ2θ), and footprint (grain size),
# and clustered by χ to build orientation maps. Six **trusted focus scans** —
# others degraded from beam damage — fall into three treatment groups (A/B/C) so
# the effect of grain-boundary passivation and Cl+DI can be isolated.

# %% [markdown]
# ## Results — orientation and treatment-group effects
#
# The figures below are produced by `feature_summary_table.py` from the **Gaussian
# shapes at 3×3 binning** for all six focus scans.
#
# - **χ area distribution** — where each reflection's grains point, area-weighted.
# - **Conclusion 1 (clean, B vs C)** — grain-boundary passivation suppresses (111)
#   relative to (001).
# - **Conclusion 2 (confounded, A vs B)** — 5 % Cl + DI consolidates (001) coverage.
#
# Each conclusion also gets an **overlaid half-violin companion**: grain footprint
# on the x-axis (log), the two compared groups drawn as one-sided violins,
# (001)-on-(001) and (111)-on-(111), so the shift of the *whole population* is
# visible rather than a single largest/mean point. Conclusion 1 overlays **B vs C**
# (GB off pushes the (111) upper tail out); Conclusion 2 overlays **A vs B** (5% Cl
# + DI shifts (001) out while (111) retreats).

# %%
# Display the already-computed result figures (skip any that are missing). Each PNG
# already carries its own title, so we don't add a caption here (avoids a double title).
_RESULT_FIGS = ["chi_hist.png",
                "conclusion1_GB.png", "conclusion1_GB_violin.png",
                "conclusion2_5%Cl.png", "conclusion2_5%Cl_violin.png"]
for fname in _RESULT_FIGS:
    fp = BASE / fname
    if not fp.exists():
        print(f"  (missing {fname} — run feature_summary_table.py to regenerate)")
        continue
    img = plt.imread(fp)
    h, w = img.shape[:2]
    fig, ax = plt.subplots(figsize=(min(15, w / 90), min(15, w / 90) * h / w))
    ax.imshow(img)
    ax.axis("off")
    plt.show()

# %% [markdown]
# ## Algorithm 1 — Binning: why summing frames *finds more grains*
#
# A binned position is the **sum of N×N neighbouring frames**. Coherent Bragg
# **signal grows ∝ N²** while random shot **noise grows only ∝ N**, so the per-
# position **SNR scales ∝ N** — weak peaks that sit *below* threshold at 1×1 cross
# it once binned, raising recall.
#
# Binning also **stabilises grain outlines**. The scan is a serpentine raster;
# stage backlash offsets the detector registration between adjacent rows, so at
# 1×1 a single grain fragments into disconnected **horizontal slices**. Summing
# N² frames washes out that offset. Snapping every frame to its true stage (X, Y)
# (`coordinate_source: file_per_row`) fixes it at the source: the fraction of
# multi-position grains collapsed to a single row drops **≈ 53 % → ≈ 34 %**. Net
# effect: **3×3 is the recall × resolution sweet spot** — enough SNR to recover
# weak grains, fine enough to still resolve the map.

# %%
# ── Binning explainer figure ────────────────────────────────────────────
# Real 3x3 (001) grain-size distribution for Scan 179, with the 1122-bin largest
# grain annotated — binning is what lifts these grains over the detection threshold.
cat179, tot179, nr179, nc179 = (load_shapes("179") + (0, 0))[:4] \
    if load_shapes("179")[0] else (None, 0, 0, 0)

fig, ax = plt.subplots(figsize=(8.5, 5))

if cat179 is not None:
    sizes = [f["n_bins"] for f in cat179["kept"] if f["reflection"] == "(001)"]
    ax.hist(np.clip(sizes, 1, 1200), bins=40, color="#4f86c6", edgecolor="white", alpha=0.9)
    ax.axvline(max(sizes), color="#c0392b", ls="--", lw=1.6)
    ax.text(max(sizes), ax.get_ylim()[1] * 0.85,
            f"  largest (001) grain\n  {max(sizes)} \n  = {100*max(sizes)/tot179:.1f}% of scan",
            color="#c0392b", fontsize=9.5 * FONT, va="top", ha="right")
    ax.set_yscale("log")
    ax.set_xlabel("grain footprint (binned positions)")
    ax.set_ylabel("number of (001) grains (log)")
    ax.set_title(f"Grain-size distribution — Scan 179 (001)\n"
                 f"{len(sizes)} grains on a {nr179}×{nc179} = {tot179}-position grid", fontsize=11 * FONT)
else:
    ax.text(0.5, 0.5, "Scan 179 catalog not found\n(needs local 179-201 tree)",
            ha="center", va="center", transform=ax.transAxes)
    ax.axis("off")

fig.suptitle("Algorithm 1 — Binning: more signal per position lifts grains over threshold", fontsize=12 * FONT)
fig.tight_layout()
fig.savefig(BASE / "binning_explainer.png", dpi=130, bbox_inches="tight")
plt.show()
print("wrote", (BASE / "binning_explainer.png").resolve())

# %% [markdown]
# ## Binning shifts the whole SNR distribution to the right
#
# The direct evidence that binning raises signal quality: the detected-peak **SNR
# distribution per bin size** (one detector, `5x5_tophat_band_adaptive_snr`, snr=4,
# run at every bin on Scan_0203 — only the binning changes). Log-x; dashed line =
# detection threshold (SNR = 4), grey band = the near-noise zone [4, 6). Top row is
# absolute counts (small bins detect *far* more peaks — mostly near-noise), bottom
# row is normalised density (the shape shifts right as bins grow). At **1×1 the
# median SNR is ~5**, right on the threshold with a huge near-noise spike; by
# **5×5 the median is ~18** and the near-noise pile is gone. (Recreated from
# `peak_intensity_snr_histograms.py`.)

# %%
# ── Detected-peak SNR distribution by bin size (recreated) ───────────────────
# Reads the multi-bin peaks catalogs; these only exist for Scan_0203 in the in-repo
# TakaProject. Any bin whose file is absent is skipped so the cell still runs.
_SNR_LAB = BASE / "TakaTest" / "TakaProject" / "Labels" / "Scan_0203"
_SNR_DET = "5x5_tophat_band_adaptive_snr"
_SNR_BINS = [1, 2, 3, 4, 5]
_SNR_THR, _NEAR = 4.0, 6.0
_SNR_COLORS = {1: "#3b6fb6", 2: "#46a0a0", 3: "#6aab4d", 4: "#e0902b", 5: "#c4432b"}


def _load_snr(n):
    p = _SNR_LAB / f"{_SNR_DET}_peaks_{n}x{n}.json"
    if not p.exists():
        return None
    d = json.load(open(p))
    s = np.array([pk.get("snr", np.nan)
                  for plist in d["peaks_by_bin"].values() for pk in plist], float)
    return s[np.isfinite(s) & (s > 0)]


_snr_data = {n: _load_snr(n) for n in _SNR_BINS}
_snr_present = [n for n in _SNR_BINS if _snr_data[n] is not None and _snr_data[n].size]

if _snr_present:
    _alls = np.concatenate([_snr_data[n] for n in _snr_present])
    _lo = max(_SNR_THR * 0.9, np.percentile(_alls, 0.5))
    _hi = np.percentile(_alls, 99.8)
    _edges = np.logspace(np.log10(_lo), np.log10(_hi), 50)

    fig, axes = plt.subplots(1, len(_snr_present), figsize=(3.0 * len(_snr_present), 3.4),
                             sharex=True, squeeze=False)
    for col, n in enumerate(_snr_present):
        s = _snr_data[n]
        med = np.median(s)
        ax = axes[0][col]
        ax.hist(s, bins=_edges, density=False, color=_SNR_COLORS[n],
                alpha=0.85, edgecolor="white", linewidth=0.3)
        ax.set_xscale("log")
        ax.set_title(f"{n}x{n}\nmed SNR={med:.1f}")
        if col == 0:
            ax.set_ylabel("count")
        ax.set_xlabel("SNR")
    fig.suptitle("Detected-peak SNR distribution by bin size: Scan_0203", y=1.0)
    fig.tight_layout()
    fig.savefig(BASE / "snr_hist_by_binsize.png", dpi=140, bbox_inches="tight")
    plt.show()
    print("wrote", (BASE / "snr_hist_by_binsize.png").resolve())
    print("median SNR by bin:", {f"{n}x{n}": round(float(np.median(_snr_data[n])), 1)
                                  for n in _snr_present})
else:
    print("No multi-bin peaks catalogs found for the SNR-distribution figure.")
    print(f"Expected {_SNR_DET}_peaks_NxN.json under {_SNR_LAB}")

# %% [markdown]
# ## Algorithm 2 — Contouring: turning scattered detections into one grain
#
# **Contouring is the `peaks → shapes` step** (`ShapeAlgorithms/gaussian.py`). Each
# spatial position yields raw per-frame Bragg detections — many of them isolated
# noise. The algorithm:
#
# 1. **Union-Find linking.** Two detections in 8-neighbour positions are merged if
#    they hit the **same detector spot within `link_tolerance = 5 px`**. A grain is
#    one connected component of that graph — its `spatial_extent` is the set of
#    positions it covers, i.e. the **contour interior**.
# 2. **Gaussian-profile filter.** A component is *kept* only if its cross-position
#    intensity has a bright, **monotonic** centre (intensity falls with distance
#    from the peak). Flat or incoherent blobs are dropped.
#
# The quantifiable payoff is below: of Scan 179's detections, **6,311 are isolated
# single-position hits that the filter rejects as noise**, while the survivors form
# clean, physically real grains. The example is one of the **largest** grains in
# the whole dataset.

# %%
# ── Contour example figure ──────────────────────────────────────────────
# The single largest (001) grain in Scan 179: draw a contour over its real
# detection positions, and scatter a sample of the *rejected* detections to show
# the contour wraps a coherent grain and excludes the scattered "weird points".
cat179, tot179, nr179, nc179 = (load_shapes("179") + (0, 0))[:4] \
    if load_shapes("179")[0] else (None, 0, 0, 0)


def _rc(extent):
    """['r_c', ...] -> (rows, cols) int arrays."""
    rc = [k.split("_") for k in extent if "_" in k]
    return (np.array([int(a) for a, _ in rc]),
            np.array([int(b) for _, b in rc]))


if cat179 is not None:
    f001 = [f for f in cat179["kept"] if f["reflection"] == "(001)"]
    grain = max(f001, key=lambda f: f["n_bins"])
    gr_rows, gr_cols = _rc(grain["spatial_extent"])

    # Pick the slice row (shared with the SNR-slice figure below): among the widest
    # rows, the one with the most *isolated* interior holes (a gap flanked by
    # detections on both sides) — those are the gaps smoothing genuinely bridges,
    # so the slice shows a clean Gaussian drop-off with bridged holes rather than a
    # big dead run. Stored in SLICE_ROW for reuse by the slice figure.
    _ext = {(int(r), int(c)) for r, c in zip(gr_rows, gr_cols)}
    _rows = sorted({r for r, _ in _ext})
    _span = {r: (max(c for rr, c in _ext if rr == r) - min(c for rr, c in _ext if rr == r) + 1)
             for r in _rows}
    _maxspan = max(_span.values())

    def _isolated_holes(r):
        return sum(1 for c in range(min(c for rr, c in _ext if rr == r) + 1,
                                    max(c for rr, c in _ext if rr == r))
                   if (r, c) not in _ext and (r, c - 1) in _ext and (r, c + 1) in _ext)

    SLICE_ROW = max((r for r in _rows if _span[r] >= 0.9 * _maxspan),
                    key=lambda r: (_isolated_holes(r), _span[r]))

    # Rejected single-position detections (the "weird points"): sample for clarity.
    rej = [f for f in cat179["filtered"] if f.get("n_bins", 1) == 1]
    rr, rc_ = [], []
    for f in rej:
        rows, cols = _rc(f.get("spatial_extent", []))
        if rows.size:
            rr.append(rows[0]); rc_.append(cols[0])
    rr, rc_ = np.array(rr), np.array(rc_)
    step = max(1, len(rr) // 800)          # thin to ~800 dots so the plot reads
    rr, rc_ = rr[::step], rc_[::step]

    # Occupancy grid for the grain, smoothed, then contoured.
    occ = np.zeros((nr179, nc179))
    occ[gr_rows, gr_cols] = 1.0
    smooth = gaussian_filter(occ, sigma=1.2) if gaussian_filter else occ

    fig, ax = plt.subplots(figsize=(9.5, 8.5))
    # rejected noise first (background)
    if rr.size:
        ax.scatter(rc_, rr, s=8, color="#bbbbbb", alpha=0.55, marker="x",
                   label=f"rejected detections (noise) — {len(rej):,} single-position hits")
    # the grain's real detection positions
    ax.scatter(gr_cols, gr_rows, s=10, color="#c0392b", alpha=0.65,
               label=f"grain detections — {grain['n_bins']:,} ")
    # the contour that wraps them
    cs = ax.contour(smooth, levels=[0.15], colors="#1a3fa0", linewidths=2.4)
    ax.contourf(smooth, levels=[0.15, 1.1], colors=["#4f86c6"], alpha=0.18)
    ax.plot([], [], color="#1a3fa0", lw=2.4, label="grain contour (kept shape)")
    # red line marking where the SNR slice below is taken (pairs with the slice figure)
    ax.axhline(SLICE_ROW, color="#d62728", lw=2.2, ls="-",
               label=f"SNR slice (row {SLICE_ROW}) →")

    ax.set_xlim(-1, nc179); ax.set_ylim(nr179, -1)   # image orientation
    ax.set_aspect("equal")
    ax.set_xlabel("scan column (x position)")
    ax.set_ylabel("scan row (y position)")
    ax.set_title(
        f"Scan 179 · largest (001) grain: {grain['n_bins']:,} = "
        f"{100*grain['n_bins']/tot179:.1f}% of the scan · "
        f"χ = {grain['chi_deg']:.0f}°",
        fontsize=10.5 * FONT)

    fig.tight_layout()
    fig.savefig(BASE / "contour_example.png", dpi=130, bbox_inches="tight")
    plt.show()
    print("wrote", (BASE / "contour_example.png").resolve())
    print(f"\nContour example (Scan 179, largest (001) grain):")
    print(f"  footprint      : {grain['n_bins']:,} positions "
          f"({100*grain['n_bins']/tot179:.1f}% of {tot179})")
    print(f"  Gaussian filter: {grain['reason']}")
    print(f"  orientation χ  : {grain['chi_deg']:.1f}°   χ-FWHM {grain.get('chi_fwhm'):.3f}°   "
          f"Δ2θ-FWHM {grain.get('tth_fwhm'):.4f}°")
    print(f"  rejected noise : {len(rej):,} single-position detections excluded by the contour")
else:
    print("Scan 179 catalog not found — cannot draw the contour example.")
    print("Point PROJECTS['179'] at a tree with Labels/Scan_0179/gaussian_shapes_3x3.json.")

# %% [markdown]
# ## Algorithm 2 (cont.) — the Gaussian profile along one slice
#
# This is a **1-D cut through the same grain**, along the red line drawn on the
# contour above (display them side by side). For every position on that scan row we
# plot the detected Bragg **SNR** (log axis). The kept-shape test
# (`check_gaussian_profile`) asks exactly this: does signal rise to a bright centre
# and **fall off monotonically** toward the edges? It does — a clean Gaussian
# profile. Two things the contour step relies on are visible:
#
# - **Drop-off to threshold.** SNR falls from its bright centre down to the
#   detection threshold (SNR = 4); where the smoothed profile crosses that line is
#   the **contour edge**. Positions beyond it have no in-grain detection.
# - **Smoothing bridges holes.** A few interior positions have no detection (gaps in
#   the raster); smoothing spans them so the grain stays one continuous contour
#   instead of shattering at every missing pixel.

# %%
# ── SNR slice through the same grain (pairs with the red line on the contour) ──
import glob
from scipy.ndimage import gaussian_filter1d

SNR_THRESHOLD = 4.0   # detector's detection cut-off (see CLAUDE.md); = the contour level


def _peaks_by_bin_179():
    """Per-bin peak list for Scan 179 at 3x3 (has per-bin SNR), or None."""
    cand = sorted(glob.glob(f"{PROJECTS['179']}/Labels/Scan_0179/*_peaks_{BIN}x{BIN}.json"))
    cand = [c for c in cand if "rawgrid" not in c and "territory" not in c]
    if not cand:
        return None
    d = json.load(open(cand[0]))
    return d.get("peaks_by_bin", d)


pbb = _peaks_by_bin_179() if cat179 is not None else None
if pbb is not None and "grain" in dir():
    dx, dy, ref = grain["detector_x"], grain["detector_y"], grain["reflection"]
    ext = {(int(r), int(c)) for r, c in zip(*_rc(grain["spatial_extent"]))}
    cols_in = sorted(c for r, c in ext if r == SLICE_ROW)
    c0, c1 = cols_in[0], cols_in[-1]
    xs = list(range(c0 - 4, c1 + 5))          # pad so we see the drop below threshold outside

    def snr_at(r, c, win=12):
        """Best (001) detection SNR near the grain's detector spot in bin (r,c)."""
        best = None
        for p in pbb.get(f"{r}_{c}", []):
            if p.get("label") != ref:
                continue
            if ((p["x"] - dx) ** 2 + (p["y"] - dy) ** 2) ** 0.5 <= win:
                s = p.get("snr", 0.0)
                best = s if best is None else max(best, s)
        return best

    raw = [snr_at(SLICE_ROW, c) for c in xs]                       # None where no detection
    # interior hole = column between the grain's ends but not in the grain (a raster gap);
    # exterior = beyond the grain's span (truly outside the contour).
    interior = [c0 <= c <= c1 for c in xs]
    detected = np.array([s if s else np.nan for s in raw], float)
    # smoothed profile: treat missing as 0, smooth, floor for the log axis
    filled = np.array([s if s else 0.0 for s in raw], float)
    smoothed = np.clip(gaussian_filter1d(filled, sigma=1.0), 1.0, None)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_yscale("log")
    # detected SNR at in-grain positions
    m = np.isfinite(detected)
    ax.scatter(np.array(xs)[m], detected[m], s=42, color="#c0392b", zorder=5,
               label="detected SNR (in grain)")
    # holes inside the grain + positions outside it (no detection) drawn at the floor
    holes = [x for x, s, inr in zip(xs, raw, interior) if inr and not s]
    outside = [x for x, s, inr in zip(xs, raw, interior) if (not inr) and not s]
    if holes:
        ax.scatter(holes, [SNR_THRESHOLD * 0.55] * len(holes), s=60, facecolors="none",
                   edgecolors="#c0392b", linewidths=1.6, zorder=5,
                   label=f"holes bridged by smoothing (×{len(holes)})")
    if outside:
        ax.scatter(outside, [SNR_THRESHOLD * 0.55] * len(outside), s=26, color="#bbbbbb",
                   marker="x", zorder=4, label="outside grain (no detection)")
    # the smoothed Gaussian profile + threshold / contour edges
    ax.plot(xs, smoothed, color="#1a3fa0", lw=2.4, label="smoothed profile (Gaussian)")
    ax.axhline(SNR_THRESHOLD, color="#555", ls="--", lw=1.3)
    above = np.array(xs)[smoothed >= SNR_THRESHOLD]
    if above.size:
        ax.axvspan(above.min(), above.max(), color="#4f86c6", alpha=0.12,
                   label="inside contour (profile ≥ threshold)")

    ax.set_ylabel("peak SNR (log)")
    ax.set_xticks([])                    # drop the x position-index ticks
    fig.tight_layout()
    fig.savefig(BASE / "contour_slice.png", dpi=130, bbox_inches="tight")
    plt.show()
    print("wrote", (BASE / "contour_slice.png").resolve())
    print(f"slice row {SLICE_ROW}: cols {c0}-{c1}, peak SNR {np.nanmax(detected):.0f}, "
          f"{len(holes)} interior holes bridged")
else:
    print("Peaks catalog for Scan 179 not found — cannot draw the SNR slice.")
    print("Expected Labels/Scan_0179/<detector>_peaks_3x3.json (has per-bin SNR).")

# %% [markdown]
# ## Conclusions & Next Steps
#
# **The pipeline is fast enough to make a whole beamtime tractable.** Each nano-XRD
# scan is *thousands* of detector frames (Scan_0203 alone is 25,170), so the
# **Scan_0179–0226 campaign is of order 10⁶ detector images** — far beyond what
# manual peak-picking can keep up with. By collapsing N×N neighbouring frames
# before detection, **binning shrinks the map the detector must process ~N²-fold**
# (a 3×3 grid holds ~9× fewer positions than 1×1), turning a scan into a
# quick-lookup device map and letting a whole project be screened for trends in
# minutes rather than days. Running headless over the entire 179–226 set, the
# automated grain-finder was able to triage every scan and flag **all but 6 as
# beam-damage-degraded** — leaving a clean set of trusted devices to draw
# conclusions from without hand-inspecting each one.
#
# **Binning + contouring reliably establishes differentiated grain contours.** The
# SNR-∝-N gain from binning lifts weak grains over threshold, and the Union-Find +
# Gaussian-profile contouring separates them into distinct, physically real grains —
# resolving a single (001) domain up to **1,122 positions (15.7 % of a scan)** while
# rejecting **6,311** isolated noise detections. That fidelity is what makes the
# downstream comparisons trustworthy:
#
# - **Conclusion 1 — Grain-boundary passivation preserves (001) texture (clean).**
#   Turning GB *off* (group B → C, Cl+DI held fixed) collapses the largest-grain
#   **001:111 size ratio from ≈ 3.4 to ≈ 0.9** — without passivation, competing
#   (111) grains grow as large as (001).
# - **Conclusion 2 — 5 % Cl + DI consolidates (001) coverage (confounded).**
#   Adding 5 % Cl + DI (group A → B) drives the **001:111 summed-area ratio
#   ≈ 2.9 → ≈ 4.6** — fewer, larger (001) domains. DI and Cl change together, so
#   this is a combined effect.
#
# **Next steps — broaden the device set.** The automated pipeline made screening
# 48 scans cheap, so the natural extension is *more devices of different types*:
# 1. Run the same grain-finding over **additional device architectures / chemistries**
#    to test whether the GB and Cl+DI trends generalise beyond this batch.
# 2. Add a controlled **GB on/off pair at fixed Cl + DI** and a **Cl-only vs
#    DI-only split** to de-confound Conclusion 2.
# 3. Correlate grain maps with the ME7 **XRF** elemental maps already wired into
#    `xrd-app`, and extend to the θ-resolved **rocking series** for strain / mosaicity.

# %%
print("Poster board complete. Figures in:", BASE)
for f in ["chi_hist.png",
          "conclusion1_GB.png", "conclusion1_GB_violin.png",
          "conclusion2_5%Cl.png", "conclusion2_5%Cl_violin.png",
          "binning_explainer.png", "snr_hist_by_binsize.png",
          "contour_example.png", "contour_slice.png"]:
    p = BASE / f
    print(f"  {'✓' if p.exists() else '✗'} {f}")

# %%
