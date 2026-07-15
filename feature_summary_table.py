# %% [markdown]
# # Focus scans — azimuthal (χ) area distribution + feature summary table
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
# **Centering (`CENTER=True`)** — χ is circular, so a cluster on the ±180 seam
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
REFS              = None   # e.g. ["(001)"] or ["(001)", "(111)"]; None = all
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
    ax.set_title(f"Scan {scan} — {rlabel}")
    ax.set_xlim(lo - pad, hi + pad)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{((x + 180) % 360 - 180):.0f}"))
    ax.set_xlabel("χ (°)")
    ax.set_ylabel("summed area (bins)")
    ax.legend(fontsize=8, loc="upper right")


fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for ax, scan in zip(axes.ravel(), PROJECTS):
    plot_chi_hist(ax, scan)
fig.suptitle(f"Azimuthal (χ) area distribution — {ALGO} shapes {BIN}×{BIN}, "
             f"KDE bandwidth = {KDE_BANDWIDTH_DEG:g}° (centered on largest gap)", fontsize=13)
fig.tight_layout()
fig.savefig("chi_hist.png", dpi=120)
plt.show()

# %% [markdown]
# ## Dominant χ cluster per reflection
#
# Splits a reflection's features into χ clusters at the KDE valleys (same method
# as the Orientation Map, area-weighted), picks the **most populous** cluster
# (largest total area = tallest KDE peak), and reports:
# - **Peak χ** — the χ at the *tip of that cluster's Gaussian* (KDE argmax).
# - **Cluster % Area** — union of that cluster's feature bins ÷ the scan's grid.

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
# - **"001" / "111"** — number of kept shapes of that reflection.
# - **001/111 Union %** — grid bins touched by ≥1 shape of that reflection
#   (set-union of `spatial_extent`) ÷ grid. Real spatial coverage, overlaps counted once.
# - **001/111 Sum %** — Σ of each shape's footprint (`len(spatial_extent)`) ÷ grid.
#   Overlaps counted repeatedly, so Sum ≥ Union; the gap measures how much shapes stack.
# - **Largest 001 / 111 Size** — max `n_bins` of a single shape; **Percent Area** = that ÷ the scan's 3×3 grid.
# - **001/111 Peak χ** — tip χ of the most populous χ cluster of that reflection.
# - **001/111 Cluster % Area** — union area of that dominant cluster ÷ grid.

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
out = Path("feature_summary_table.csv")
df[cols].round(1).to_csv(out)
print("wrote", out.resolve())
