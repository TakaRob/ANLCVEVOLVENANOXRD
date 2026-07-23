# %% [markdown]
# # Focus scans: azimuthal (χ) area distribution + feature summary table
#
# Uses the **gaussian-fit shapes** at **3×3 binning** for the six focus scans.
# Per scan: `Labels/<scan>/gaussian_shapes_3x3.json` (kept shapes) and
# `Metadata/<scan>/grid_mapping_3x3.json` (grid size). Data is loaded once; the
# χ histogram and the summary table both read it.
#
# Run as `# %%` cells (VS Code / Jupyter / Spyder), or `python feature_summary_table.py`.

# %%
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.signal import find_peaks

BIN = 3            # binning
ALGO = "gaussian"  # shape algorithm (the gaussian-profile fit)

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

# scan -> project root (each tree has Labels/ and Metadata/)
PROJECTS = {
    "179": "/home/takaji/179-201",
    "182": "/home/takaji/179-201",
    "203": "/home/takaji/rocking_203_214",
    "207": "/home/takaji/rocking_203_214",
    "215": "/home/takaji/215-226",
    "218": "/home/takaji/215-226",
}


def load_scan(scan, root):
    sd = f"Scan_0{scan}"
    kept = json.load(open(f"{root}/Labels/{sd}/{ALGO}_shapes_{BIN}x{BIN}.json"))["kept"]
    grid = json.load(open(f"{root}/Metadata/{sd}/grid_mapping_{BIN}x{BIN}.json"))
    return kept, grid["n_bin_rows"] * grid["n_bin_cols"]


# DATA[scan] = (kept shapes list, total grid bins)
DATA = {s: load_scan(s, r) for s, r in PROJECTS.items()}
print({s: (len(k), tot) for s, (k, tot) in DATA.items()})

# %% [markdown]
# ## Azimuthal (χ) area histogram with Gaussian KDE overlay
#
# Same distribution the **Orientation Map** shows: features binned by `chi_deg`,
# **weighted by area** (`n_bins`). Bars = area histogram; red curve = circular
# Gaussian KDE (app's `cluster_features_by_chi`): `Σ wᵢ·exp(−½((Δχ)/bw)²)` on a
# 1° wrapped grid.
#
# **Centering (`CENTER=True`)**: χ is circular, so a cluster on the ±180 seam
# splits to both edges. We cut the circle at its **largest empty gap** so data is
# contiguous; the two extreme features become the axis min/max (app's
# `_circular_frame`). Tick labels are wrapped back to real χ.
#
# **Knobs:** `KDE_BANDWIDTH_DEG` (σ in degrees, app default 5°), `HIST_BIN_DEG`
# (bar width), `REFS` (restrict reflections), `CENTER`.

# %%
# ── knobs ───────────────────────────────────────────────────────────
KDE_BANDWIDTH_DEG = 5.0    # <<< Gaussian KDE bandwidth (sigma), in degrees
HIST_BIN_DEG      = 5.0    # azimuthal histogram bar width, in degrees
REFS              = ["(111)"]   # e.g. ["(001)"] or ["(001)", "(111)"]; None = all
CENTER            = True   # cut the circle at the largest gap so data is contiguous
# ────────────────────────────────────────────────────────────────────


def _wrap180(a):
    """Wrap angle(s) into [-180, 180)."""
    return (np.asarray(a, float) + 180.0) % 360.0 - 180.0


def chi_area(kept, refs=None):
    """(chi_deg, area=n_bins) arrays for features with a defined chi."""
    chi, w = [], []
    for f in kept:
        c = f.get("chi_deg")
        if c is None:
            continue
        if refs and f.get("reflection") not in refs:
            continue
        chi.append(c)
        w.append(float(f.get("n_bins") or len(f.get("intensity_profile") or {}) or 1))
    return _wrap180(chi), np.asarray(w, float)


def kde_wrapped(chi, w, bandwidth, grid):
    """Circular Gaussian KDE over chi (degrees), as the orientation map builds it.
    Wrapped Δχ makes it correct on any grid, incl. a centered range past ±180."""
    kde = np.zeros_like(grid, dtype=float)
    for c, wi in zip(chi, w):
        diff = (grid - c + 180.0) % 360.0 - 180.0
        kde += wi * np.exp(-0.5 * (diff / bandwidth) ** 2)
    return kde


def circular_frame(chi):
    """Port of orientation._circular_frame: cut χ at the largest empty gap so all
    data is contiguous. Returns (map fn, lo, hi); the two farthest-apart points
    become lo and hi. Points below the cut lift by +360."""
    v = np.sort(np.asarray(chi, float))
    if v.size < 2:
        c = float(v[0]) if v.size else 0.0
        return (lambda x: np.asarray(x, float)), c, c
    gaps = np.diff(v)
    wrap_gap = 360.0 - (v[-1] - v[0])
    if gaps.size == 0 or wrap_gap >= gaps.max():
        return (lambda x: np.asarray(x, float)), float(v[0]), float(v[-1])
    k = int(np.argmax(gaps))          # widest interior gap is between v[k], v[k+1]
    thr = (v[k] + v[k + 1]) / 2.0     # everything below the gap lifts by +360
    mapfn = lambda x: np.where(np.asarray(x, float) < thr,
                               np.asarray(x, float) + 360.0, np.asarray(x, float))
    return mapfn, float(v[k + 1]), float(v[k] + 360.0)


def plot_chi_hist(ax, scan, bandwidth=None, bin_deg=None, refs=None, center=None):
    bandwidth = KDE_BANDWIDTH_DEG if bandwidth is None else bandwidth
    bin_deg   = HIST_BIN_DEG      if bin_deg   is None else bin_deg
    refs      = REFS              if refs      is None else refs
    center    = CENTER            if center    is None else center
    kept, _ = DATA[scan]
    chi, w = chi_area(kept, refs)

    if center and chi.size:
        mapfn, lo, hi = circular_frame(chi)
    else:
        mapfn, lo, hi = (lambda x: np.asarray(x, float)), -180.0, 180.0
    mchi = mapfn(chi)
    pad = bin_deg

    edges = np.arange(lo, hi + bin_deg, bin_deg)
    hist, _ = np.histogram(mchi, bins=edges, weights=w)
    centers = (edges[:-1] + edges[1:]) / 2
    ax.bar(centers, hist, width=bin_deg, align="center",
           color="#7aa8d2", edgecolor="none", alpha=0.85, label="area (n_bins)")

    grid = np.arange(lo, hi + 1.0, 1.0)
    kde = kde_wrapped(chi, w, bandwidth, grid)
    scale = bin_deg / (bandwidth * np.sqrt(2 * np.pi))
    ax.plot(grid, kde * scale, color="#c0392b", lw=1.8,
            label=f"Gaussian KDE (bw={bandwidth:g}°)")

    rlabel = "all reflections" if not refs else ", ".join(refs)
    ax.set_title(f"Scan {scan}: {rlabel}")
    ax.set_xlim(lo - pad, hi + pad)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{((x + 180) % 360 - 180):.0f}"))
    ax.set_xlabel("χ (°)")
    ax.set_ylabel("summed area (bins)")
    ax.legend(fontsize=8 * FONT, loc="upper right")


fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for ax, scan in zip(axes.ravel(), PROJECTS):
    plot_chi_hist(ax, scan)
fig.suptitle(f"Azimuthal (χ) area distribution: {ALGO} shapes {BIN}×{BIN}, "
             f"KDE bandwidth = {KDE_BANDWIDTH_DEG:g}° (centered on largest gap)", fontsize=13 * FONT)
fig.tight_layout()
fig.savefig("chi_hist.png", dpi=120)
plt.show()

# %% [markdown]
# ## Dominant χ cluster per reflection
#
# Splits a reflection's features into χ clusters at the KDE valleys (same method
# as the Orientation Map, area-weighted), picks the **most populous** cluster
# (largest total area = tallest KDE peak), and reports:
# - **Peak χ**: the χ at the *tip of that cluster's Gaussian* (KDE argmax).
# - **Cluster % Area**: union of that cluster's feature bins ÷ the scan's grid.

# %%
def chi_clusters(feats, bandwidth):
    """Area-weighted χ clusters split at KDE valleys (port of
    orientation.cluster_features_by_chi). Returns [{features, area}]."""
    items = [(_wrap180([f["chi_deg"]])[0], f) for f in feats if f.get("chi_deg") is not None]
    if not items:
        return []
    items.sort(key=lambda t: t[0])
    chis = np.array([t[0] for t in items])
    ws = np.array([float(f.get("n_bins") or 1) for _, f in items])
    if len(items) < 3:
        return [{"features": [f for _, f in items], "area": float(ws.sum())}]
    grid = np.linspace(-180, 179, 360)
    kde = kde_wrapped(chis, ws, bandwidth, grid)
    pad = max(4, int(bandwidth * 2))
    ext = np.concatenate([kde[-pad:], kde, kde[:pad]])
    vidx, _ = find_peaks(-ext, distance=max(4, int(bandwidth * 1.5)), prominence=0.3 * kde.max())
    vidx = vidx - pad
    vidx = vidx[(vidx >= 0) & (vidx < 360)]
    if len(vidx) < 2 or kde.max() == 0:
        return [{"features": [f for _, f in items], "area": float(ws.sum())}]
    vnorm = np.sort((grid[vidx] + 180) % 360)
    nseg = len(vnorm)
    groups = defaultdict(list)
    for c, f in items:
        cn = (c + 180) % 360
        idx = int(np.searchsorted(vnorm, cn, side="right")) % nseg
        groups[idx].append(f)
    return [{"features": g, "area": sum(float(f.get("n_bins") or 1) for f in g)}
            for g in groups.values() if g]


def dominant_cluster(feats, total_bins, bandwidth=None):
    """Most populous (max area) χ cluster -> (peak_chi, union_pct_of_grid)."""
    bandwidth = KDE_BANDWIDTH_DEG if bandwidth is None else bandwidth
    cls = chi_clusters(feats, bandwidth)
    if not cls:
        return (np.nan, 0.0)
    dom = max(cls, key=lambda c: c["area"])
    chi, w = chi_area(dom["features"])
    grid = np.arange(-180, 180, 0.5)
    peak = float(grid[int(np.argmax(kde_wrapped(chi, w, bandwidth, grid)))])
    union = set()
    for f in dom["features"]:
        union.update(f.get("spatial_extent", []))
    return (peak, 100.0 * len(union) / total_bins)

# %% [markdown]
# ## Feature summary table
#
# One row per scan.
# - **"001" / "111"**: number of kept shapes of that reflection.
# - **001/111 Union %**: grid bins touched by ≥1 shape of that reflection
#   (set-union of `spatial_extent`) ÷ grid. Real spatial coverage, overlaps counted once.
# - **001/111 Sum %**: Σ of each shape's footprint (`len(spatial_extent)`) ÷ grid.
#   Overlaps counted repeatedly, so Sum ≥ Union; the gap measures how much shapes stack.
# - **Largest 001 / 111 Size**: max `n_bins` of a single shape; **Percent Area** = that ÷ the scan's 3×3 grid.
# - **001/111 Peak χ**: tip χ of the most populous χ cluster of that reflection.
# - **001/111 Cluster % Area**: union area of that dominant cluster ÷ grid.

# %%
def coverage(feats, total_bins):
    """(union %, sum %) of the grid for a set of features.

    union = bins touched by ≥1 feature (set-union of `spatial_extent`, dedup);
    sum   = Σ per-feature footprint (`len(spatial_extent)`), overlaps repeated.
    """
    union = set()
    total = 0
    for f in feats:
        ext = f.get("spatial_extent", [])
        union.update(ext)
        total += len(ext)
    return 100.0 * len(union) / total_bins, 100.0 * total / total_bins


def scan_row(scan):
    kept, total_bins = DATA[scan]
    counts = Counter(f["reflection"] for f in kept)

    def largest(ref):
        sizes = [f["n_bins"] for f in kept if f["reflection"] == ref]
        return max(sizes) if sizes else 0

    f001 = [f for f in kept if f["reflection"] == "(001)"]
    f111 = [f for f in kept if f["reflection"] == "(111)"]
    u001, s001 = coverage(f001, total_bins)
    u111, s111 = coverage(f111, total_bins)
    peak001, uarea001 = dominant_cluster(f001, total_bins)
    peak111, uarea111 = dominant_cluster(f111, total_bins)
    l001, l111 = largest("(001)"), largest("(111)")

    return {
        "Scan": scan,
        '"001"': counts.get("(001)", 0),
        '"111"': counts.get("(111)", 0),
        "001 Union %": u001,
        "001 Sum %": s001,
        "111 Union %": u111,
        "111 Sum %": s111,
        "Largest 001 Size": l001,
        "Percent Area (001)": 100 * l001 / total_bins,
        "Largest 111 Size": l111,
        "Percent Area (111)": 100 * l111 / total_bins,
        "001 Peak χ": peak001,
        "001 Cluster % Area": uarea001,
        "111 Peak χ": peak111,
        "111 Cluster % Area": uarea111,
        "grid bins": total_bins,
        "total shapes": len(kept),
    }


df = pd.DataFrame([scan_row(s) for s in PROJECTS]).set_index("Scan")
cols = ['"001"', '"111"', "001 Union %", "001 Sum %", "111 Union %", "111 Sum %",
        "Largest 001 Size", "Percent Area (001)",
        "Largest 111 Size", "Percent Area (111)",
        "001 Peak χ", "001 Cluster % Area", "111 Peak χ", "111 Cluster % Area"]
pct = ["001 Union %", "001 Sum %", "111 Union %", "111 Sum %",
       "Percent Area (001)", "Percent Area (111)",
       "001 Cluster % Area", "111 Cluster % Area"]
deg = ["001 Peak χ", "111 Peak χ"]
fmt = {c: "{:.1f}%" for c in pct}
fmt.update({c: "{:.0f}°" for c in deg})
try:
    display(df[cols].style.format(fmt))   # noqa: F821  (Jupyter)
except NameError:
    print(df[cols].round(1).to_string())

# %%
def safe_to_csv(frame, path):
    """Write a CSV, but don't abort the run if the file is locked open (Excel /
    OneDrive on this box hold a write lock): warn and keep going so figures below
    still generate."""
    p = Path(path)
    try:
        frame.to_csv(p)
        print("wrote", p.resolve())
    except PermissionError:
        print(f"!! could not write {p.name} (open in Excel/OneDrive?): skipping, "
              f"close it and re-run this cell")


safe_to_csv(df[cols].round(1), "feature_summary_table.csv")

# %% [markdown]
# # Treatment-group comparison: GB and Cl+DI effects on (001) vs (111)
#
# The six trusted focus scans fall into **three treatment groups, two scans each**.
# Other scans in each project degraded too fast (beam damage) to trust, so this
# stays at **n = 2 per group**: read the per-scan dots drawn on every bar, not
# just the group-mean height.
#
# | Group | Scans | DI | Cl | **GB** |
# |---|---|---|---|---|
# | **A** | 179, 182 | No |:   | **Yes** |
# | **B** | 203, 207 | Yes | 5% | **Yes** |
# | **C** | 215, 218 | Yes | 5% | **No**  |
#
# - **B vs C** isolates the **grain-boundary (GB) treatment**: only GB changes,
#   Cl+DI held fixed. This is the clean, single-variable contrast.
# - **A vs B** changes **both DI and Cl** at once: confounded; read it as a
#   combined "Cl+DI" effect, not attributable to either alone.
#
# ### Conclusion 1: GB treatment suppresses (111) relative to (001)  *(clean, B vs C)*
# Turning GB **off** (B→C) leaves (001) roughly flat but grows (111) sharply: the
# largest-grain **001:111 size ratio collapses from ~3.4 (GB on) to ~0.9 (GB off)**
# and the (111) largest-grain area fraction roughly doubles. Interpretation: GB
# passivation preserves the preferred (001) texture; without it, competing (111)
# grains are free to nucleate and grow as large as the (001) ones. The effect is
# consistent across *both* scans in each group (not an outlier).
#
# ### Conclusion 2: Cl+DI consolidates (001) coverage  *(confounded, A vs B)*
# Adding 5% Cl + DI (A→B) drops total feature counts but **raises (001) dominance
# per area**: (001) spatial coverage (Union %) and the **001:111 Sum-area ratio
# (~2.9 → ~4.6)** both climb. Fewer, but more area-consolidated (001) domains.
# Cannot be split into a Cl-only vs DI-only effect from these scans.
#
# **Prediction for future study:** (i) a dedicated GB on/off pair at fixed Cl+DI
# should reproduce the 001:111 largest-grain collapse; (ii) a Cl-only vs DI-only
# split (both with GB on) is needed to de-confound Conclusion 2.

# %%
# ── build the treatment-group table (means + per-scan 001:111 ratios) ──
TREATMENT   = {"179": "A", "182": "A", "203": "B", "207": "B", "215": "C", "218": "C"}
GROUP_ORDER = ["A", "B", "C"]
GROUP_LABEL = {"A": "A\nNo_DI · Yes_GB",
               "B": "B\n5%_DI · Yes_GB",
               "C": "C\n5%_DI · No_GB"}
GROUP_COLOR = {"A": "#9bbcd8", "B": "#4f86c6", "C": "#c0392b"}  # A/B share hue (GB on), C red (GB off)

g = df.copy()
g["Group"] = [TREATMENT[s] for s in g.index]

# per-scan 001:111 ratios (mean-of-ratios is what the group bars average)
g["r_count"]   = g['"001"']            / g['"111"']
g["r_union"]   = g["001 Union %"]      / g["111 Union %"]
g["r_sum"]     = g["001 Sum %"]        / g["111 Sum %"]
g["r_largest"] = g["Largest 001 Size"] / g["Largest 111 Size"]


def mean_shape_size(scan, ref):
    """Average footprint (n_bins) of a single shape of `ref` in `scan`."""
    kept, _ = DATA[scan]
    sizes = [f["n_bins"] for f in kept if f["reflection"] == ref]
    return float(np.mean(sizes)) if sizes else 0.0


g["Mean 001 Size"] = [mean_shape_size(s, "(001)") for s in g.index]
g["Mean 111 Size"] = [mean_shape_size(s, "(111)") for s in g.index]

grp_cols = ['"001"', '"111"', "001 Union %", "111 Union %", "001 Sum %", "111 Sum %",
            "Largest 001 Size", "Largest 111 Size", "Percent Area (001)", "Percent Area (111)",
            "Mean 001 Size", "Mean 111 Size",
            "r_count", "r_union", "r_sum", "r_largest"]
group_means = g.groupby("Group")[grp_cols].mean().reindex(GROUP_ORDER)
print("Treatment-group means (n = 2 scans each):")
print(group_means.round(2).to_string())
safe_to_csv(group_means.round(2), "treatment_group_means.csv")


def _scan_dots(ax, xpos, values, jitter=0.07):
    """Overlay the individual scan values on a group bar (honesty for n=2)."""
    values = np.asarray(values, float)
    xs = np.linspace(-jitter, jitter, len(values)) if len(values) > 1 else np.array([0.0])
    ax.scatter(xpos + xs, values, s=34, color="#222", zorder=6,
               edgecolor="white", linewidth=0.7)

# %% [markdown]
# ## Conclusion 1 figure: GB suppresses (111)  (largest-grain ratio + area share)

# %%
x = np.arange(len(GROUP_ORDER))
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Panel A: largest-grain 001:111 size ratio, per group
r_lg = group_means["r_largest"].values
ax1.bar(x, r_lg, width=0.6, color=[GROUP_COLOR[k] for k in GROUP_ORDER], alpha=0.9)
for xi, grp in zip(x, GROUP_ORDER):
    _scan_dots(ax1, xi, g.loc[g.Group == grp, "r_largest"].values)
ax1.axhline(1.0, ls="--", color="#555", lw=1)
ax1.annotate("", xy=(2, r_lg[2] + 0.15), xytext=(1, r_lg[1] - 0.15),
             arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.6))
ax1.text(1.46, 3.7, "GB off →\n111 grows",
         fontsize=8.5 * FONT, color="#c0392b", ha="left", va="center")
ax1.set_xticks(x); ax1.set_xticklabels([GROUP_LABEL[k] for k in GROUP_ORDER], fontsize=8.5 * FONT)
ax1.set_ylabel("largest-grain size ratio  (001/111)")
ax1.set_title("Largest-grain 001:111 ratio\n≈3.4 with GB → ≈0.9 without (B→C)", fontsize=10.5 * FONT)

# Panel B: largest-grain area share, (001) vs (111), grouped
w = 0.38
a001 = group_means["Percent Area (001)"].values
a111 = group_means["Percent Area (111)"].values
ax2.bar(x - w / 2, a001, w, color="#4f86c6", label="(001)")
ax2.bar(x + w / 2, a111, w, color="#e08e45", label="(111)")
for xi, grp in zip(x, GROUP_ORDER):
    _scan_dots(ax2, xi - w / 2, g.loc[g.Group == grp, "Percent Area (001)"].values)
    _scan_dots(ax2, xi + w / 2, g.loc[g.Group == grp, "Percent Area (111)"].values)
# arrow over the (111) bars: B -> C, where the 111 area roughly doubles
_top2 = max(a001.max(), a111.max())
ax2.set_ylim(0, _top2 * 1.35)                # headroom so the arrow/label clear the bars/dots
ax2.annotate("", xy=(2 + w / 2, a111[2] + _top2 * 0.05),
             xytext=(1 + w / 2, a111[1] + _top2 * 0.05),
             arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.6))
ax2.text(1.5 + w / 2, _top2 * 1.20, "Largest 111 doubles",
         fontsize=8.5 * FONT, color="#c0392b", ha="center", va="center",
         bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))
ax2.set_xticks(x); ax2.set_xticklabels([GROUP_LABEL[k] for k in GROUP_ORDER], fontsize=8.5 * FONT)
ax2.set_ylabel("largest-grain area (% of scan)")
ax2.set_title("Largest-grain area share\n111 ~doubles when GB removed (B→C)", fontsize=10.5 * FONT)
ax2.legend(loc="upper left", fontsize=9 * FONT)

fig1.suptitle("Conclusion 1: Glove Box fabrication suppresses (111) vs (001)", fontsize=12 * FONT)
fig1.tight_layout()
fig1.savefig("conclusion1_GB.png", dpi=130)
plt.show()
print("wrote", Path("conclusion1_GB.png").resolve())

# %% [markdown]
# ## Conclusion 2 figure: Cl+DI consolidates (001)  (Sum ratio + 001 coverage)

# %%
fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(13, 5))

# Panel A: 001:111 Sum-area ratio, per group (A→B is the effect; C≈A)
r_sum = group_means["r_sum"].values
ax3.bar(x, r_sum, width=0.6, color=[GROUP_COLOR[k] for k in GROUP_ORDER], alpha=0.9)
for xi, grp in zip(x, GROUP_ORDER):
    _scan_dots(ax3, xi, g.loc[g.Group == grp, "r_sum"].values)
ax3.set_ylim(0, r_sum.max() * 1.35)          # headroom so the label clears the bars/dots
ax3.annotate("", xy=(1, r_sum[1] + 0.15), xytext=(0, r_sum[0] - 0.15),
             arrowprops=dict(arrowstyle="->", color="#1a6b3a", lw=1.6))
ax3.text(0.5, r_sum.max() * 1.20, "5% Cl →\n001 grows",
         fontsize=8.5 * FONT, color="#1a6b3a", ha="center", va="center",
         bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))
ax3.set_xticks(x); ax3.set_xticklabels([GROUP_LABEL[k] for k in GROUP_ORDER], fontsize=8.5 * FONT)
ax3.set_ylabel("area ratio  (001/111)")
ax3.set_title("001:111 area ratio\n≈2.9 → ≈4.6 with 5% Cl (A→B)", fontsize=10.5 * FONT)

# Panel B: average shape size, (001) vs (111), per group: 001 grows, 111 doesn't
w = 0.38
m001 = group_means["Mean 001 Size"].values
m111 = group_means["Mean 111 Size"].values
ax4.bar(x - w / 2, m001, w, color="#4f86c6", label="(001)")
ax4.bar(x + w / 2, m111, w, color="#e08e45", label="(111)")
for xi, grp in zip(x, GROUP_ORDER):
    _scan_dots(ax4, xi - w / 2, g.loc[g.Group == grp, "Mean 001 Size"].values)
    _scan_dots(ax4, xi + w / 2, g.loc[g.Group == grp, "Mean 111 Size"].values)
_top4 = max(m001.max(), m111.max())
ax4.set_ylim(0, _top4 * 1.35)                # headroom so the label clears the bars/dots
ax4.annotate("", xy=(1 - w / 2, m001[1] + 3), xytext=(0 - w / 2, m001[0] + 3),
             arrowprops=dict(arrowstyle="->", color="#1a6b3a", lw=1.6))
ax4.text(0.5 - w / 2, _top4 * 1.20, "001 grows", fontsize=8.5 * FONT,
         color="#1a6b3a", ha="center", va="center",
         bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))
ax4.set_xticks(x); ax4.set_xticklabels([GROUP_LABEL[k] for k in GROUP_ORDER], fontsize=8.5 * FONT)
ax4.set_ylabel("average shape size (n_bins)")
ax4.set_title("Average grain size\n001 rises with 5% Cl (A→B); 111 does not", fontsize=10.5 * FONT)
ax4.legend(loc="upper right", fontsize=9 * FONT)

fig2.suptitle("Conclusion 2: 5% Cl consolidates (001) coverage", fontsize=12 * FONT)
fig2.tight_layout()
fig2.savefig("conclusion2_5%Cl.png", dpi=130)
plt.show()
print("wrote", Path("conclusion2_5%Cl.png").resolve())

# %% [markdown]
# # Conclusions as overlaid half-violins: does the grain-size distribution shift?
#
# A focused redesign of the size-distribution comparison. Grain **footprint is on
# the x-axis (log)**, and each group is drawn as a **one-sided (half) violin** —
# density rising from a common baseline — so two distributions can be **overlaid**
# and read directly against each other. Each conclusion overlays its two groups,
# the **same reflection on itself**: (001)-on-(001) (top panel) and
# (111)-on-(111) (bottom panel). Solid vline = median, dashed = mean (both
# labelled in real footprint); each curve's legend carries its grain count *n*.
#
# - **Conclusion 1 (GB, B vs C)** — drop A. Turning GB *off* (B→C) leaves the (001)
#   bulk about where it is but pushes the **(111) upper tail out** (the largest
#   (111) grains appear).
# - **Conclusion 2 (Cl + DI, A vs B)** — drop C. Adding 5% Cl + DI (A→B) shifts the
#   **(001) tail outward** while the **(111) distribution retreats** to smaller grains.

# %%
# ── overlaid half-violins: footprint on x (log), one pair of groups per figure ─
from scipy.stats import gaussian_kde

GROUP_SCANS = {gp: [s for s in PROJECTS if TREATMENT[s] == gp] for gp in GROUP_ORDER}
XT = np.array([2, 3, 5, 10, 20, 50, 100, 200, 500, 1000])


def _grain_sizes(scan, ref):
    """Footprint (n_bins) of every kept shape of `ref` in `scan`."""
    kept, _ = DATA[scan]
    return np.array([f["n_bins"] for f in kept if f["reflection"] == ref], float)


def _pool(gp, ref):
    """Footprints of `ref` pooled over both scans of group `gp`."""
    return np.concatenate([_grain_sizes(s, ref) for s in GROUP_SCANS[gp]])


def _half_violin(ax, sizes, color, label, grid, lab_y):
    """Upward half-violin of log10(footprint), peak-normalised; median/mean vlines."""
    d = np.log10(sizes)
    y = gaussian_kde(d)(grid)
    y = y / y.max()                                        # peak-normalise so shapes compare
    ax.fill_between(grid, 0, y, color=color, alpha=0.38, lw=0)
    ax.plot(grid, y, color=color, lw=1.8, label=f"{label}   (n={sizes.size})")
    med, mean = float(np.median(sizes)), float(sizes.mean())
    ax.vlines(np.log10(med),  0, 1.02, color=color, lw=2.2)               # median: solid
    ax.vlines(np.log10(mean), 0, 1.02, color=color, lw=1.4, ls="--")      # mean: dashed
    ax.text(np.log10(med),  lab_y,        f"M {med:.0f}",   color=color,
            fontsize=6.5 * FONT, ha="center", va="bottom")
    ax.text(np.log10(mean), lab_y + 0.11, f"μ {mean:.0f}", color=color,
            fontsize=6.5 * FONT, ha="center", va="bottom")
    return med, mean


def overlay_fig(pair, pair_color, pair_label, suptitle, outname, annotate):
    """Two-panel (001/111) overlay of `pair`'s half-violins; `annotate(axes)` adds arrows."""
    grid = np.linspace(np.log10(1.8), np.log10(1300), 400)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8.5), sharex=True)
    for ax, ref in zip(axes, ["(001)", "(111)"]):
        for i, gp in enumerate(pair):
            _half_violin(ax, _pool(gp, ref), pair_color[gp], pair_label[gp], grid,
                         lab_y=1.06 + 0.24 * i)
        ax.set_ylim(0, 1.6)
        ax.set_yticks([])
        ax.set_ylabel(f"{ref}\ndensity", fontsize=10.5 * FONT)
        ax.legend(loc="center right", fontsize=8 * FONT, framealpha=0.9)
    annotate(axes)
    axes[-1].set_xticks(np.log10(XT)); axes[-1].set_xticklabels([str(v) for v in XT])
    axes[-1].set_xlim(np.log10(1.8), np.log10(1300))
    axes[-1].set_xlabel("grain footprint  (binned positions, log)")
    fig.suptitle(suptitle, fontsize=12 * FONT)
    fig.tight_layout()
    fig.savefig(outname, dpi=130)
    plt.show()
    print("wrote", Path(outname).resolve())


# %% [markdown]
# ## Conclusion 1 (overlaid half-violins): GB off pushes the (111) tail out (B vs C)

# %%
def _annot_c1(axes):
    axes[0].text(np.log10(15), 0.58, "(001): similar\n(medians 5–6)", color="#555",
                 fontsize=8 * FONT, ha="center", va="center")
    # (111): the honest reading — bulk is *not* larger with GB off. B (GB on) holds
    # more mid-size (111); C (GB off) only wins at the single largest grain.
    axes[1].text(np.log10(35), 0.74, "(111) medians equal (4);\nGB-on (B) holds more mid-size (111)",
                 color="#2f6fb3", fontsize=7.5 * FONT, ha="center", va="center")
    axes[1].annotate("", xy=(np.log10(450), 0.14), xytext=(np.log10(170), 0.09),
                     arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.6))
    axes[1].text(np.log10(430), 0.30, "only C's single\nlargest grain is bigger", color="#c0392b",
                 fontsize=7.5 * FONT, ha="center", va="center")


overlay_fig(["B", "C"], {"B": "#2f6fb3", "C": "#c0392b"},
            {"B": "B · Yes_GB (baseline)", "C": "C · No_GB"},
            "Glove Box on vs off (B vs C): (001) similar, (111) bulk unchanged",
            "conclusion1_GB_violin.png", _annot_c1)

# %% [markdown]
# ## Conclusion 2 (overlaid half-violins): 5% Cl + DI shifts (001) out, (111) retreats (A vs B)

# %%
def _annot_c2(axes):
    axes[0].annotate("", xy=(np.log10(430), 0.34), xytext=(np.log10(110), 0.20),
                     arrowprops=dict(arrowstyle="->", color="#1a6b3a", lw=1.8))
    axes[0].text(np.log10(300), 0.50, "5% Cl + DI →\n(001) upper tail rises", color="#1a6b3a",
                 fontsize=8 * FONT, ha="center", va="center")
    axes[1].annotate("", xy=(np.log10(6), 0.34), xytext=(np.log10(60), 0.34),
                     arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.8))
    axes[1].text(np.log10(45), 0.50, "(111) retreats\nto smaller grains", color="#c0392b",
                 fontsize=8 * FONT, ha="center", va="center")


overlay_fig(["A", "B"], {"A": "#9a9a9a", "B": "#2f6fb3"},
            {"A": "A · No_DI (baseline)", "B": "B · 5% Cl + DI"},
            "5% Cl + DI (A → B): the (001) upper tail rises, (111) retreats",
            "conclusion2_5%Cl_violin.png", _annot_c2)

# %%
