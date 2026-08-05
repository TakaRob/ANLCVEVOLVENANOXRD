"""Visualize the existing processed ME7 and XRD products from the 2026-2 Luo data."""

# %% [markdown]
# # Processed XRF and XRD visualizations
#
# This notebook demonstrates what the existing files in `processed/` are useful
# for. It intentionally omits a standalone position/trajectory plot. Positions
# are used only to place XRF and XRD intensity values in sample space.
#
# Products shown:
#
# 1. Br K-alpha intensity distribution and spatial map.
# 2. Ten point-wise XRD ROI intensity maps for one scan.
# 3. Br-vs-XRD correlations at identical acquisition points.
# 4. Detector-ROI center-of-mass diagnostics.
# 5. Interpolated maps from `crystal_roi_dict.h5`.
# 6. Scan-series XRD ROI intensity trends across scans 24-44.

# %% Configuration
from pathlib import Path

DATA_ROOT = Path("/mnt/z/isn/2026-2/2026-2-Luo/xrd_scripts")
PROCESSED = DATA_ROOT / "processed"
SCAN = 24
ROI_TO_INSPECT = 0
N_XRD_ROIS = 10

# Percentiles suppress isolated hot pixels in map color scales without changing
# the underlying data used for statistics and correlations.
MAP_PERCENTILES = (1.0, 99.0)
SCATTER_SAMPLE = 15000

# %% Imports and helpers
import json
import re

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from IPython.display import display
from matplotlib.colors import LogNorm
from scipy.stats import spearmanr

ROI_FILE_RE = re.compile(r"Scan_(\d+)_xrd_roi_(\d+)\.h5$")


def read_nxdata(path):
    """Read datasets and attributes from one processed `entry/data` group."""
    with h5py.File(path, "r") as handle:
        group = handle["entry/data"]
        datasets = {name: np.asarray(group[name][:]) for name in group.keys()}
        attrs = {name: group.attrs[name] for name in group.attrs.keys()}
    return datasets, attrs


def finite_limits(values, percentiles=MAP_PERCENTILES, positive=False):
    """Robust plotting limits from finite values."""
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(values)
    if positive:
        mask &= values > 0
    selected = values[mask]
    if selected.size == 0:
        return (1.0, 2.0) if positive else (0.0, 1.0)
    lo, hi = np.percentile(selected, percentiles)
    if hi <= lo:
        hi = lo + max(abs(lo) * 1e-9, 1e-12)
    return float(lo), float(hi)


def scatter_map(ax, x, y, values, title, log=False, size=3):
    """Plot measured intensity at its sample coordinate."""
    if log:
        vmin, vmax = finite_limits(values, positive=True)
        artist = ax.scatter(x, y, c=values, s=size, cmap="magma",
                            norm=LogNorm(vmin=max(vmin, 1e-12), vmax=vmax))
    else:
        vmin, vmax = finite_limits(values)
        artist = ax.scatter(x, y, c=values, s=size, cmap="viridis",
                            vmin=vmin, vmax=vmax)
    ax.set(title=title, xlabel="X position (um)", ylabel="Y position (um)")
    ax.set_aspect("equal")
    return artist


def apply_y_offset(y, scan):
    """Apply the theta-indexed Y correction used by the prefilter notebook."""
    offset_file = DATA_ROOT / "position_offset.json"
    master_file = DATA_ROOT / f"Scan_{scan:04d}.h5"
    if not offset_file.exists() or not master_file.exists():
        return np.asarray(y, dtype=float), np.nan, 0.0
    with h5py.File(master_file, "r") as handle:
        theta = float(np.nanmean(
            handle["entry/instrument/bluesky/streams/baseline/sample_theta/value"][:]
        ))
    with offset_file.open() as stream:
        offsets = json.load(stream)
    angles = np.asarray(offsets["theta"], dtype=float)
    y_offsets = np.asarray(offsets["y_offset"], dtype=float)
    order = np.argsort(angles)
    angles, y_offsets = angles[order], y_offsets[order]
    nearest = int(np.argmin(np.abs(angles - theta)))
    offset = (float(y_offsets[nearest]) if abs(angles[nearest] - theta) <= 0.01
              else float(np.interp(theta, angles, y_offsets)))
    return np.asarray(y, dtype=float) + offset, theta, offset


def roi_label(index, attrs, tth_map):
    """Label an XRD detector ROI by index and approximate 2-theta range."""
    y0, y1 = int(attrs["roi_y_start"]), int(attrs["roi_y_end"])
    x0, x1 = int(attrs["roi_x_start"]), int(attrs["roi_x_end"])
    region = np.asarray(tth_map[y0:y1, x0:x1], dtype=float)
    finite = region[np.isfinite(region)]
    if finite.size == 0:
        return f"ROI {index}"
    return f"ROI {index}\n{finite.min():.2f}-{finite.max():.2f} deg 2theta"

# %% Load scan 24 point-wise products
position_path = PROCESSED / "SOCKETSERVER" / f"Scan_{SCAN:04d}_position.h5"
br_path = PROCESSED / "ME7" / f"Scan_{SCAN:04d}_Br ka.h5"
xrd_paths = [
    PROCESSED / "XRD" / f"Scan_{SCAN:04d}_xrd_roi_{index}.h5"
    for index in range(N_XRD_ROIS)
]

required = [position_path, br_path, *xrd_paths]
missing = [path for path in required if not path.exists()]
if missing:
    raise FileNotFoundError("Missing processed products:\n" + "\n".join(map(str, missing)))

positions, _ = read_nxdata(position_path)
br, br_attrs = read_nxdata(br_path)
xrd_products = [read_nxdata(path) for path in xrd_paths]

n = min(
    len(positions["X_Position"]), len(positions["Y_Position"]),
    len(br["Intensity"]), *(len(product[0]["Intensity"]) for product in xrd_products),
)
x_position = positions["X_Position"][:n].astype(float)
y_position, sample_theta, y_offset = apply_y_offset(
    positions["Y_Position"][:n], SCAN
)
br_intensity = br["Intensity"][:n].astype(float)
xrd_intensity = np.vstack([
    product[0]["Intensity"][:n].astype(float) for product in xrd_products
])

tth_path = DATA_ROOT / "tth.tiff"
tth_map = tifffile.imread(tth_path) if tth_path.exists() else None
xrd_labels = [
    roi_label(index, product[1], tth_map) if tth_map is not None else f"ROI {index}"
    for index, product in enumerate(xrd_products)
]

print(f"Scan {SCAN}: {n:,} aligned position/Br/XRD points")
print(f"sample theta={sample_theta:.6g} deg, applied Y offset={y_offset:g}")
print(f"Br detector ROI: channels {br_attrs['roi_y_start']}:{br_attrs['roi_y_end']}, "
      f"MCA pixels {br_attrs['roi_x_start']}:{br_attrs['roi_x_end']}")
for label in xrd_labels:
    print(label.replace("\n", ", "))

# %% [markdown]
# ## 1. Br XRF: distribution and sample map
#
# The histogram helps choose an XRF intensity threshold. The map shows where Br
# counts occur on the sample. This uses the measured coordinates, but it is an XRF
# signal visualization rather than a standalone position plot.

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
positive_br = br_intensity[br_intensity > 0]
axes[0].hist(positive_br, bins=120, log=True, color="#a64b2a", alpha=0.85)
axes[0].set(title="Br K-alpha point intensity", xlabel="integrated Br counts",
            ylabel="scan points")
if positive_br.size:
    for percentile in (50, 90, 95, 99):
        value = np.percentile(positive_br, percentile)
        axes[0].axvline(value, linestyle=":", linewidth=0.8,
                        label=f"p{percentile}={value:.3g}")
    axes[0].legend(fontsize=8)
artist = scatter_map(axes[1], x_position, y_position, br_intensity,
                     "Br K-alpha intensity map", log=True)
fig.colorbar(artist, ax=axes[1], label="integrated Br counts")
plt.show()

# %% [markdown]
# ## 2. XRD ROI maps
#
# Each panel is integrated diffraction intensity from one fixed detector rectangle.
# The approximate 2-theta range is calculated from `tth.tiff`. Bright sample-space
# regions identify where that diffraction band is strong and can be followed back
# to raw XRD frames for full-pattern analysis.

# %%
fig, axes = plt.subplots(2, 5, figsize=(18, 7.5), constrained_layout=True)
for index, ax in enumerate(axes.flat):
    artist = scatter_map(ax, x_position, y_position, xrd_intensity[index],
                         xrd_labels[index], log=True, size=2)
    fig.colorbar(artist, ax=ax, fraction=0.046, label="XRD ROI counts")
plt.show()

# %% [markdown]
# ## 3. Br-XRD correlation
#
# This asks whether Br-rich points also have stronger diffraction in each detector
# ROI. Spearman correlation captures monotonic trends without assuming a linear
# count scale. The scatter panels are deterministically subsampled for readability;
# correlations use all finite points.

# %%
correlation_rows = []
for index in range(N_XRD_ROIS):
    mask = np.isfinite(br_intensity) & np.isfinite(xrd_intensity[index])
    rho, p_value = spearmanr(br_intensity[mask], xrd_intensity[index, mask])
    correlation_rows.append({
        "roi": index, "label": xrd_labels[index].replace("\n", ", "),
        "spearman_rho": rho, "p_value": p_value,
    })
correlations = pd.DataFrame(correlation_rows).set_index("roi")
display(correlations)

fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
colors = ["#b84a3a" if value < 0 else "#2878a6" for value in correlations["spearman_rho"]]
ax.bar(np.arange(N_XRD_ROIS), correlations["spearman_rho"], color=colors)
ax.axhline(0, color="black", linewidth=0.8)
ax.set(xticks=np.arange(N_XRD_ROIS), xlabel="XRD detector ROI",
       ylabel="Spearman rho", title="Br intensity vs XRD ROI intensity")
plt.show()

rng = np.random.default_rng(0)
sample_count = min(SCATTER_SAMPLE, n)
sample = np.sort(rng.choice(n, size=sample_count, replace=False))
fig, axes = plt.subplots(2, 5, figsize=(17, 7), constrained_layout=True)
for index, ax in enumerate(axes.flat):
    ax.hexbin(br_intensity[sample], xrd_intensity[index, sample], gridsize=45,
              bins="log", mincnt=1, cmap="cividis")
    ax.set(title=f"ROI {index}: rho={correlations.loc[index, 'spearman_rho']:.2f}",
           xlabel="Br counts", ylabel="XRD ROI counts")
plt.show()

# %% [markdown]
# ## 4. Detector center-of-mass diagnostics
#
# These are not sample positions. For XRD they show where intensity lies inside a
# detector rectangle. A compact cloud indicates a stable diffraction spot; drift,
# splitting, or multiple clouds can indicate peak motion or multiple contributions.

# %%
index = int(ROI_TO_INSPECT)
if not 0 <= index < N_XRD_ROIS:
    raise ValueError(f"ROI_TO_INSPECT must be in [0, {N_XRD_ROIS - 1}]")
product, attrs = xrd_products[index]
com_x = product["COM_X"][:n].astype(float)
com_y = product["COM_Y"][:n].astype(float)
intensity = product["Intensity"][:n].astype(float)
valid = np.isfinite(com_x) & np.isfinite(com_y) & (intensity > 0)
valid_indices = np.flatnonzero(valid)
if valid_indices.size > SCATTER_SAMPLE:
    valid_indices = np.sort(rng.choice(valid_indices, SCATTER_SAMPLE, replace=False))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
vmin, vmax = finite_limits(intensity[valid_indices], positive=True)
artist = axes[0].scatter(
    com_x[valid_indices], com_y[valid_indices], c=intensity[valid_indices], s=5,
    cmap="magma", norm=LogNorm(vmin=max(vmin, 1e-12), vmax=vmax),
)
axes[0].set(
    title=f"ROI {index} detector centroid",
    xlabel="COM_X (detector column)", ylabel="COM_Y (detector row)",
    xlim=(attrs["roi_x_start"], attrs["roi_x_end"]),
    ylim=(attrs["roi_y_start"], attrs["roi_y_end"]),
)
fig.colorbar(artist, ax=axes[0], label="XRD ROI counts")

br_com_x = br["COM_X"][:n].astype(float)
br_valid = np.isfinite(br_com_x) & (br_intensity > 0)
axes[1].hist(br_com_x[br_valid], bins=80, color="#a64b2a")
axes[1].set(
    title="Br spectral centroid within saved ROI",
    xlabel="COM_X (MCA pixel)", ylabel="scan points",
    xlim=(br_attrs["roi_x_start"], br_attrs["roi_x_end"]),
)
plt.show()

# %% [markdown]
# ## 5. Meshed XRD maps
#
# `crystal_roi_dict.h5` stores interpolated regular-grid maps. These are convenient
# for publication figures, overlays, and animations. Use the point-wise products
# above for exact thresholds and correlations because interpolation smooths data.

# %%
mesh_path = PROCESSED / "crystal_roi_dict.h5"
if not mesh_path.exists():
    raise FileNotFoundError(mesh_path)

with h5py.File(mesh_path, "r") as handle:
    scan_group = handle[f"scan_{SCAN:04d}"]
    mesh_maps = {
        index: tuple(np.asarray(scan_group[f"roi_{index}"][name][:], dtype=float)
                     for name in ("X", "Y", "Z"))
        for index in range(N_XRD_ROIS)
    }

fig, axes = plt.subplots(2, 5, figsize=(18, 7.5), constrained_layout=True)
for index, ax in enumerate(axes.flat):
    mesh_x, mesh_y, mesh_z = mesh_maps[index]
    vmin, vmax = finite_limits(mesh_z, positive=True)
    artist = ax.pcolormesh(mesh_x, mesh_y, mesh_z, shading="auto", cmap="magma",
                           norm=LogNorm(vmin=max(vmin, 1e-12), vmax=vmax))
    ax.set(title=xrd_labels[index], xlabel="X position", ylabel="Y position")
    ax.set_aspect("equal")
    fig.colorbar(artist, ax=ax, fraction=0.046, label="interpolated XRD counts")
plt.show()

# %% [markdown]
# ## 6. Scan-series XRD trends
#
# The point-wise XRD ROI products cover scans 24-44. This summarizes each scan by
# median and upper-decile intensity. It is useful for identifying which scan/angle
# makes a diffraction band strongest before opening raw detector data. The scan
# number is used when a local scan master file is unavailable for theta metadata.

# %%
series_rows = []
for path in sorted((PROCESSED / "XRD").glob("Scan_*_xrd_roi_*.h5")):
    match = ROI_FILE_RE.match(path.name)
    if not match:
        continue
    scan, roi = map(int, match.groups())
    with h5py.File(path, "r") as handle:
        values = np.asarray(handle["entry/data/Intensity"][:], dtype=float)
    finite = values[np.isfinite(values)]
    series_rows.append({
        "scan": scan, "roi": roi,
        "median": float(np.median(finite)),
        "p90": float(np.percentile(finite, 90)),
        "total": float(np.sum(finite)),
    })
series = pd.DataFrame(series_rows)

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
for roi in sorted(series["roi"].unique()):
    rows = series[series["roi"] == roi].sort_values("scan")
    axes[0].plot(rows["scan"], rows["median"], marker="o", ms=3, label=f"ROI {roi}")
    axes[1].plot(rows["scan"], rows["p90"], marker="o", ms=3, label=f"ROI {roi}")
axes[0].set(title="Scan-series median XRD ROI intensity", ylabel="median counts")
axes[1].set(title="Scan-series upper-decile XRD ROI intensity",
            xlabel="scan number", ylabel="90th-percentile counts")
for ax in axes:
    ax.set_yscale("symlog", linthresh=1)
    ax.grid(alpha=0.2)
axes[0].legend(ncol=5, fontsize=8)
plt.show()

# %% Compact numerical summary
summary = pd.DataFrame({
    "roi": np.arange(N_XRD_ROIS),
    "label": [label.replace("\n", ", ") for label in xrd_labels],
    "scan24_median": np.median(xrd_intensity, axis=1),
    "scan24_p90": np.percentile(xrd_intensity, 90, axis=1),
    "br_spearman_rho": correlations["spearman_rho"].to_numpy(),
})
display(summary)
