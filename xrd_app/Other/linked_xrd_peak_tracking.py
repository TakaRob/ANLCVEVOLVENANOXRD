"""Explore cropped XRF-to-XRD links and track detector peaks through sample space."""

# %% [markdown]
# # Linked XRD frame exploration and peak tracking
#
# This notebook loads the link table written by `xrf_prefilter_xrd.py`. It does
# not copy raw detector images. It can:
#
# 1. inspect the cropped sample positions and available spatial bin widths;
# 2. load an individual linked XRD frame;
# 3. sum a reproducible sample of frames to identify detector peaks;
# 4. sum frames in spatial bins inside a small sample-space ROI; and
# 5. track each selected detector peak by intensity and center of mass (COM).
#
# The COM tracking follows `processed_data_visualizations.py`, but computes the
# same diagnostics directly from linked raw XRD frames. Raw-data binning is
# intentionally opt-in because each detector frame is large.

# %% Configuration
from pathlib import Path

DATA_ROOT = Path("/mnt/z/isn/2026-2/2026-2-Luo/xrd_scripts")
SCAN = 24
LINK_TABLE_H5 = (
    DATA_ROOT / "processed" / "xrf_prefilter"
    / f"Scan_{SCAN:04d}_xrf_xrd_links.h5"
)
MATERIAL = "Br"

# Row in the material-filtered link table to show as an individual detector frame.
FRAME_TO_SHOW = 0

# Physical sample-space bin widths to compare. These are in the same units as X/Y.
BIN_WIDTHS = (0.10, 0.25, 0.50, 1.00)
BIN_WIDTH = 0.25

# Spatial bin to preview as a summed detector image. None selects the bin that
# contains the maximum XRF-intensity position; otherwise use a key such as "3_4".
BINNED_IMAGE_BIN_KEY = None

# Limit expensive raw-frame reads to a small physical region. Set to None to use
# the complete cropped material map (usually too expensive for initial testing).
# Format: (x_min, x_max, y_min, y_max)
SAMPLE_ROI = None

# Build a detector sum from evenly spaced linked frames to choose peak locations.
DETECTOR_SUM_SAMPLE = 1000

# Choose detector peaks manually as [(x, y), ...], or leave empty for automatic
# candidates from the sampled detector sum. Detector x is column; y is row.
PEAK_CENTERS = []
MAX_AUTO_PEAKS = 12
AUTO_PEAK_SENSITIVITY = 6.0
AUTO_PEAK_MIN_DISTANCE = 12

# A fixed detector window follows each peak. The local maximum may move inside
# this window; COM is measured in a smaller window around that local maximum.
TRACK_RADIUS = 12
COM_RADIUS = 4

# Heavy step. First inspect the frame, detector sum, and ROI/bin occupancy with
# this False. Set True to run the xrd-app CLI engine for SAMPLE_ROI.
RUN_SPATIAL_BINNING = False
MAX_FRAMES = 5000
TRACKING_OUTPUT = (
    DATA_ROOT / "processed" / "xrf_prefilter"
    / f"Scan_{SCAN:04d}_{MATERIAL}_xrd_peak_tracking.h5"
)

# %% Imports and helpers
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from click.testing import CliRunner
from IPython.display import display
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
from scipy import ndimage

H5_DATASET = "entry/data/data"


def decode_strings(values):
    """Decode an HDF5 string dataset to ordinary Python strings."""
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def load_link_table(path):
    """Load the column-oriented HDF5 linker written by the prefilter notebook."""
    if not path.exists():
        raise FileNotFoundError(
            f"Link table not found: {path}\nRun the final xrf_prefilter_xrd.py cell first."
        )
    with h5py.File(path, "r") as handle:
        group = handle["links"]
        table = pd.DataFrame({
            "Material": decode_strings(group["Material"][:]),
            "Point": np.asarray(group["Point"][:], dtype=np.int64),
            "X": np.asarray(group["X"][:], dtype=float),
            "Y": np.asarray(group["Y"][:], dtype=float),
            "XRF Intensity": np.asarray(group["XRF Intensity"][:], dtype=float),
            "XRD File Link": decode_strings(group["XRD File Link"][:]),
            "XRD Frame Index": np.asarray(group["XRD Frame Index"][:], dtype=np.int64),
            "Global Frame Index": np.asarray(group["Global Frame Index"][:], dtype=np.int64),
        })
        attrs = dict(handle.attrs)
    return table, attrs


def read_linked_frame(row, region=None):
    """Read one linked detector frame, optionally restricted to [y0:y1, x0:x1]."""
    with h5py.File(row["XRD File Link"], "r") as handle:
        dataset = handle[H5_DATASET]
        frame_index = int(row["XRD Frame Index"])
        if region is None:
            return dataset[frame_index].astype(np.float32)
        y0, y1, x0, x1 = region
        return dataset[frame_index, y0:y1, x0:x1].astype(np.float32)


def robust_positive_limits(values, percentiles=(2.0, 99.8)):
    """Return finite positive display limits for logarithmic images."""
    selected = np.asarray(values, dtype=float)
    selected = selected[np.isfinite(selected) & (selected > 0)]
    if not selected.size:
        return 1.0, 2.0
    lo, hi = np.percentile(selected, percentiles)
    return max(float(lo), 1e-9), max(float(hi), float(lo) + 1e-9)


def assign_spatial_bins(table, width, origin=None):
    """Assign physical X/Y coordinates to square spatial bins."""
    if width <= 0:
        raise ValueError("Spatial bin width must be positive")
    x0, y0 = origin or (float(table["X"].min()), float(table["Y"].min()))
    assigned = table.copy()
    assigned["bin_col"] = np.floor((assigned["X"] - x0) / width).astype(int)
    assigned["bin_row"] = np.floor((assigned["Y"] - y0) / width).astype(int)
    assigned["bin_key"] = (
        assigned["bin_row"].astype(str) + "_" + assigned["bin_col"].astype(str)
    )
    return assigned, (x0, y0)


def detect_peak_centers(image, sensitivity, min_distance, max_peaks):
    """Detect separated local maxima after broad-background subtraction."""
    background = ndimage.gaussian_filter(image.astype(float), sigma=10.0)
    cleaned = np.clip(image - background, 0, None)
    finite = cleaned[np.isfinite(cleaned)]
    median = float(np.median(finite))
    mad = 1.4826 * float(np.median(np.abs(finite - median)))
    threshold = median + sensitivity * max(mad, 1e-9)
    radius = max(1, int(min_distance))
    maxima = cleaned == ndimage.maximum_filter(cleaned, size=2 * radius + 1)
    ys, xs = np.nonzero(maxima & (cleaned > threshold))
    order = np.argsort(cleaned[ys, xs])[::-1][:max_peaks]
    return [(int(xs[index]), int(ys[index])) for index in order]


def sum_linked_rows(rows):
    """Sum linked frames while opening each source HDF5 file only once."""
    summed = None
    for path, file_rows in rows.groupby("XRD File Link", sort=False):
        with h5py.File(path, "r") as handle:
            dataset = handle[H5_DATASET]
            for frame_index in file_rows["XRD Frame Index"].astype(int):
                frame = dataset[frame_index].astype(np.float64)
                summed = frame if summed is None else summed + frame
    return summed


def measure_peak(image, seed_x, seed_y, track_radius, com_radius):
    """Find a seed's local maximum and measure background-subtracted COM."""
    height, width = image.shape
    x0, x1 = max(0, seed_x - track_radius), min(width, seed_x + track_radius + 1)
    y0, y1 = max(0, seed_y - track_radius), min(height, seed_y + track_radius + 1)
    search = image[y0:y1, x0:x1].astype(float)
    if not search.size:
        return np.nan, np.nan, np.nan, np.nan
    local_y, local_x = np.unravel_index(np.nanargmax(search), search.shape)
    peak_x, peak_y = x0 + int(local_x), y0 + int(local_y)

    cx0, cx1 = max(0, peak_x - com_radius), min(width, peak_x + com_radius + 1)
    cy0, cy1 = max(0, peak_y - com_radius), min(height, peak_y + com_radius + 1)
    patch = image[cy0:cy1, cx0:cx1].astype(float)
    background = float(np.percentile(search, 25))
    weights = np.clip(patch - background, 0, None)
    intensity = float(weights.sum())
    if intensity <= 0:
        return float(peak_x), float(peak_y), 0.0, float(image[peak_y, peak_x])
    yy, xx = np.mgrid[cy0:cy1, cx0:cx1]
    com_x = float((weights * xx).sum() / intensity)
    com_y = float((weights * yy).sum() / intensity)
    return com_x, com_y, intensity, float(image[peak_y, peak_x])


# %% Load and select one material
links, link_attrs = load_link_table(LINK_TABLE_H5)
materials = sorted(links["Material"].unique())
if MATERIAL not in materials:
    raise KeyError(f"MATERIAL {MATERIAL!r} is not available; choose from {materials}")
material_links = links.loc[links["Material"] == MATERIAL].reset_index(drop=True)
if SAMPLE_ROI is not None:
    x_min, x_max, y_min, y_max = SAMPLE_ROI
    roi_links = material_links.loc[
        material_links["X"].between(x_min, x_max)
        & material_links["Y"].between(y_min, y_max)
    ].copy()
else:
    roi_links = material_links.copy()
if roi_links.empty:
    raise ValueError("SAMPLE_ROI contains no retained linked positions")

print(f"Link table: {LINK_TABLE_H5}")
print(f"Materials: {materials}")
print(f"{MATERIAL}: {len(material_links):,} linked frames")
print(f"Selected sample ROI: {len(roi_links):,} linked frames")
display(roi_links.head())

# %% Cropped sample map and candidate spatial bin sizes
fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
artist = ax.scatter(
    material_links["X"], material_links["Y"],
    c=material_links["XRF Intensity"], s=3, cmap="viridis",
)
if SAMPLE_ROI is not None:
    x_min, x_max, y_min, y_max = SAMPLE_ROI
    ax.add_patch(Rectangle(
        (x_min, y_min), x_max - x_min, y_max - y_min,
        fill=False, edgecolor="red", linewidth=1.5,
    ))
ax.set(
    title=f"{MATERIAL}: cropped XRF-to-XRD positions",
    xlabel="X position", ylabel="corrected Y position",
)
ax.set_aspect("equal")
fig.colorbar(artist, ax=ax, label=f"{MATERIAL} XRF intensity")
plt.show()

bin_rows = []
origin = (float(roi_links["X"].min()), float(roi_links["Y"].min()))
for width in BIN_WIDTHS:
    assigned, _ = assign_spatial_bins(roi_links, width, origin=origin)
    occupancy = assigned.groupby("bin_key").size()
    bin_rows.append({
        "bin_width": width,
        "occupied_bins": int(occupancy.size),
        "median_frames_per_bin": float(occupancy.median()),
        "maximum_frames_per_bin": int(occupancy.max()),
    })
bin_summary = pd.DataFrame(bin_rows)
display(bin_summary)

# %% Load one individual linked XRD frame
if not 0 <= FRAME_TO_SHOW < len(material_links):
    raise IndexError(f"FRAME_TO_SHOW must be between 0 and {len(material_links) - 1}")
frame_row = material_links.iloc[FRAME_TO_SHOW]
frame = read_linked_frame(frame_row)
vmin, vmax = robust_positive_limits(frame)
fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
image_artist = ax.imshow(
    np.clip(frame, vmin, None), origin="upper", cmap="magma",
    norm=LogNorm(vmin=vmin, vmax=vmax),
)
ax.set(
    title=(f"{MATERIAL} linked frame {FRAME_TO_SHOW}: "
           f"X={frame_row['X']:.3f}, Y={frame_row['Y']:.3f}"),
    xlabel="detector x / column", ylabel="detector y / row",
)
fig.colorbar(image_artist, ax=ax, label="detector counts")
plt.show()
print(f"XRD source: {frame_row['XRD File Link']}")
print(f"Frame index inside source: {int(frame_row['XRD Frame Index'])}")

# %% Sampled detector sum and candidate peaks
sample_count = min(int(DETECTOR_SUM_SAMPLE), len(roi_links))
sample_indices = np.linspace(0, len(roi_links) - 1, sample_count, dtype=int)
sum_rows = roi_links.iloc[np.unique(sample_indices)]
detector_sum = sum_linked_rows(sum_rows)
if detector_sum is None:
    raise ValueError("No linked detector frames were available for the sampled sum")

peak_centers = list(PEAK_CENTERS) or detect_peak_centers(
    detector_sum,
    sensitivity=AUTO_PEAK_SENSITIVITY,
    min_distance=AUTO_PEAK_MIN_DISTANCE,
    max_peaks=MAX_AUTO_PEAKS,
)
print(f"Peak centers (x, y): {peak_centers}")

vmin, vmax = robust_positive_limits(detector_sum)
fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
sum_artist = ax.imshow(
    np.clip(detector_sum, vmin, None), origin="upper", cmap="inferno",
    norm=LogNorm(vmin=vmin, vmax=vmax),
)
for peak_id, (peak_x, peak_y) in enumerate(peak_centers, 1):
    ax.add_patch(Rectangle(
        (peak_x - TRACK_RADIUS, peak_y - TRACK_RADIUS),
        2 * TRACK_RADIUS + 1, 2 * TRACK_RADIUS + 1,
        fill=False, edgecolor="cyan", linewidth=1,
    ))
    ax.text(peak_x, peak_y, f" {peak_id}", color="cyan", va="bottom")
ax.set(
    title=f"{MATERIAL}: sum of {len(sum_rows)} linked frames and tracking windows",
    xlabel="detector x / column", ylabel="detector y / row",
)
fig.colorbar(sum_artist, ax=ax, label="summed detector counts")
plt.show()

# %% Spatially bin the selected sample ROI (heavy, opt-in)
binned_links, bin_origin = assign_spatial_bins(roi_links, BIN_WIDTH, origin=origin)
occupancy = binned_links.groupby("bin_key").size().sort_values(ascending=False)
print(f"{BIN_WIDTH:g}-unit bins: {len(occupancy):,} occupied")
print(occupancy.describe())

# Preview the summed XRD detector image for the spatial bin at peak XRF intensity.
peak_xrf_row = binned_links.loc[binned_links["XRF Intensity"].idxmax()]
preview_bin_key = (
    str(BINNED_IMAGE_BIN_KEY)
    if BINNED_IMAGE_BIN_KEY is not None
    else str(peak_xrf_row["bin_key"])
)
preview_rows = binned_links.loc[binned_links["bin_key"] == preview_bin_key]
if preview_rows.empty:
    raise KeyError(
        f"BINNED_IMAGE_BIN_KEY {preview_bin_key!r} is not occupied; "
        f"choose from {sorted(occupancy.index)}"
    )
preview_peak_xrf_row = preview_rows.loc[preview_rows["XRF Intensity"].idxmax()]
binned_detector_sum = sum_linked_rows(preview_rows)
vmin, vmax = robust_positive_limits(binned_detector_sum)
fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
binned_artist = ax.imshow(
    np.clip(binned_detector_sum, vmin, None), origin="upper", cmap="inferno",
    norm=LogNorm(vmin=vmin, vmax=vmax),
)
ax.set(
    title=(f"{MATERIAL}: bin {preview_bin_key}, sum of {len(preview_rows)} frames\n"
           f"Bin XRF-peak position: X={preview_peak_xrf_row['X']:.3f}, "
           f"Y={preview_peak_xrf_row['Y']:.3f}"),
    xlabel="detector x / column", ylabel="detector y / row",
)
fig.colorbar(binned_artist, ax=ax, label="summed detector counts")
plt.show()
display(preview_rows[["Point", "X", "Y", "XRF Intensity"]])

# The notebook invokes the same Click command as the terminal, so all raw-frame
# binning and tracking behavior stays in xrd_app.core.linked_xrd.
if RUN_SPATIAL_BINNING:
    from xrd_app.cli import main

    command = [
        "linked-xrd-track",
        "--links", str(LINK_TABLE_H5),
        "--material", MATERIAL,
        "--bin-width", str(BIN_WIDTH),
        "--detector-sum-sample", str(DETECTOR_SUM_SAMPLE),
        "--track-radius", str(TRACK_RADIUS),
        "--com-radius", str(COM_RADIUS),
        "--max-frames", str(MAX_FRAMES),
        "--output", str(TRACKING_OUTPUT),
    ]
    if SAMPLE_ROI is not None:
        command.extend(["--sample-roi", *(str(value) for value in SAMPLE_ROI)])
    for peak_x, peak_y in peak_centers:
        command.extend(["--peak", str(peak_x), str(peak_y)])
    result = CliRunner().invoke(main, command, catch_exceptions=False)
    print(result.output)
    if result.exit_code:
        raise RuntimeError(f"linked-xrd-track failed with exit code {result.exit_code}")
else:
    print("Tracking skipped. Set RUN_SPATIAL_BINNING=True after choosing SAMPLE_ROI.")

# %% Load the CLI-generated tracking result
tracking = pd.DataFrame()
if TRACKING_OUTPUT.exists():
    from xrd_app.core import linked_xrd

    saved_tracking = linked_xrd.load_result(TRACKING_OUTPUT)
    tracking = saved_tracking["tracking"]
    peak_centers = [tuple(center) for center in saved_tracking["peak_centers"]]
    display(tracking.head(20))
elif RUN_SPATIAL_BINNING:
    raise FileNotFoundError(TRACKING_OUTPUT)

# %% Visualize each peak's intensity and detector motion through the scan
if not tracking.empty:
    fig, axes = plt.subplots(
        len(peak_centers), 3, figsize=(16, 4.6 * len(peak_centers)),
        squeeze=False, constrained_layout=True,
    )
    for row_index, peak_id in enumerate(range(1, len(peak_centers) + 1)):
        points = tracking.loc[tracking["peak_id"] == peak_id]
        positive = points["intensity"] > 0
        intensity_artist = axes[row_index, 0].scatter(
            points["sample_x"], points["sample_y"], c=points["intensity"],
            s=18, cmap="magma",
        )
        axes[row_index, 0].set_title(f"Peak {peak_id}: integrated intensity")
        fig.colorbar(intensity_artist, ax=axes[row_index, 0], label="background-subtracted counts")

        shift_artist = axes[row_index, 1].quiver(
            points.loc[positive, "sample_x"], points.loc[positive, "sample_y"],
            points.loc[positive, "shift_x"], points.loc[positive, "shift_y"],
            points.loc[positive, "intensity"], cmap="viridis", angles="xy",
            scale_units="xy", scale=1,
        )
        axes[row_index, 1].set_title(f"Peak {peak_id}: detector COM shift")
        fig.colorbar(shift_artist, ax=axes[row_index, 1], label="intensity")

        axes[row_index, 2].scatter(
            points.loc[positive, "com_x"], points.loc[positive, "com_y"],
            c=points.loc[positive, "intensity"], s=18, cmap="viridis",
        )
        axes[row_index, 2].plot(
            points["seed_x"].iloc[0], points["seed_y"].iloc[0],
            marker="+", color="red", markersize=12,
        )
        axes[row_index, 2].set_title(f"Peak {peak_id}: detector centroid cloud")
        axes[row_index, 2].set_xlabel("COM x / detector column")
        axes[row_index, 2].set_ylabel("COM y / detector row")

        for axis in axes[row_index, :2]:
            axis.set_xlabel("sample X")
            axis.set_ylabel("sample Y")
            axis.set_aspect("equal")
    plt.show()
else:
    print("No tracking table yet; run the spatial-binning cell with RUN_SPATIAL_BINNING=True.")

# %% Compact peak-motion summary
if not tracking.empty:
    tracking_summary = tracking.groupby("peak_id").agg(
        seed_x=("seed_x", "first"),
        seed_y=("seed_y", "first"),
        spatial_bins=("bin_key", "size"),
        median_intensity=("intensity", "median"),
        maximum_intensity=("intensity", "max"),
        com_x_min=("com_x", "min"),
        com_x_max=("com_x", "max"),
        com_y_min=("com_y", "min"),
        com_y_max=("com_y", "max"),
    )
    tracking_summary["com_x_span"] = (
        tracking_summary["com_x_max"] - tracking_summary["com_x_min"]
    )
    tracking_summary["com_y_span"] = (
        tracking_summary["com_y_max"] - tracking_summary["com_y_min"]
    )
    display(tracking_summary)

# %%
