"""Scan 24 first pass: filter raw XRD links using calibrated ME7 XRF ROIs."""

# %% [markdown]
# # Scan 24 XRF-guided XRD prefilter
#
# This workflow follows the conventions in the 2026-2 Luo `xrd_scripts`:
#
# - raw data: `Raw/Scan_0024/{ME7,XRD}/scan_0024_NNNNN.h5`
# - positions: `processed/SOCKETSERVER/Scan_0024_position.h5`
# - Y correction: interpolate `position_offset.json` at the recorded sample theta
# - one numbered ME7 file corresponds to the same-numbered XRD file
# - points within each file correspond by local frame index
#
# Only narrow XRF slices are read when building the table. XRD detector images
# are not read or copied; each selected point stores an XRD file/frame link.

# %% Configuration
from pathlib import Path

DATA_ROOT = Path("/mnt/z/isn/2026-2/2026-2-Luo/xrd_scripts")
SCAN = 24
POSITION_OFFSET_FILE = DATA_ROOT / "position_offset.json"
SCAN_MASTER_FILE = DATA_ROOT / f"Scan_{SCAN:04d}.h5"

# Match the existing mictools ROI `Roi(0, 6, ..., name=...)`: detector channels
# 0 through 5 are summed. Set True only if deadtime-corrected counts are desired;
# the existing processed Br ROI appears to use uncorrected ROI sums.
CHANNELS = list(range(6))
DEADTIME_CORRECTION = False

# Supplied CCD calibration, returning keV for an MCA pixel.
ENERGY_CALIBRATION = {
    "quadratic_kev": 5.263744e-7,
    "linear_kev": 8.41967e-3,
    "offset_kev": 1.136032,
}

# Initial physical ROIs. Br preserves the existing script's exact 1190:1238
# integration range. In and Pb start at +/-0.15 keV around their tabulated lines
# and should be checked against the plotted spectrum before filtering.
XRF_ROIS = {
    "Br": {"line_kev": 11.924, "pixel_range": (1190, 1238),
           "intensity_bounds": (None, None)},
    "In": {"line_kev": 3.287, "energy_range_kev": (3.137, 3.437),
           "intensity_bounds": (None, None)},
    "Pb": {"line_kev": 10.551, "energy_range_kev": (10.401, 10.701),
           "intensity_bounds": (None, None)},
}

# Subsample both scan lines and points for the overview spectrum. Set both to 1
# for the exact whole-scan sum; the defaults are a fast first look over /mnt/z.
SPECTRUM_FILE_STRIDE = 10
SPECTRUM_POINT_STRIDE = 10
OUTPUT_CSV = DATA_ROOT / "processed" / "xrf_prefilter" / "Scan_0024_xrf_xrd_links.csv"

# %% Imports
import json
import re

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from matplotlib.widgets import SpanSelector

H5_DATASET = "entry/data/data"
DT_FACTOR = "entry/instrument/NDAttributes/CHAN{channel1}DTFactor"
FILE_PATTERN = re.compile(r"scan_(\d+)_(\d+)\.h5$")


def pixel_to_kev(pixel):
    """Apply the supplied quadratic CCD calibration."""
    pixel = np.asarray(pixel, dtype=float)
    cal = ENERGY_CALIBRATION
    return (
        cal["quadratic_kev"] * pixel**2
        + cal["linear_kev"] * pixel
        + cal["offset_kev"]
    )


def kev_to_pixel(energy_kev):
    """Invert the monotonic positive branch of the quadratic calibration."""
    cal = ENERGY_CALIBRATION
    a = cal["quadratic_kev"]
    b = cal["linear_kev"]
    c = cal["offset_kev"] - np.asarray(energy_kev, dtype=float)
    discriminant = b**2 - 4 * a * c
    if np.any(discriminant < 0):
        raise ValueError("Requested energy is outside the calibration domain")
    return (-b + np.sqrt(discriminant)) / (2 * a)


def energy_range_to_pixels(energy_range_kev):
    """Convert a calibrated energy interval to an inclusive-exclusive slice."""
    lo_kev, hi_kev = sorted(map(float, energy_range_kev))
    lo = max(0, int(np.floor(kev_to_pixel(lo_kev))))
    hi = min(4096, int(np.ceil(kev_to_pixel(hi_kev))))
    if hi <= lo:
        raise ValueError(f"Empty energy ROI {energy_range_kev}")
    return lo, hi


def numbered_files(folder):
    """Return `{file_number: path}` and reject ambiguous duplicate numbers."""
    found = {}
    for path in sorted(folder.glob(f"scan_{SCAN:04d}_*.h5")):
        match = FILE_PATTERN.match(path.name)
        if not match:
            continue
        number = int(match.group(2))
        if number in found:
            raise ValueError(f"Duplicate file number {number} in {folder}")
        found[number] = path
    return found


for roi in XRF_ROIS.values():
    if "pixel_range" not in roi:
        roi["pixel_range"] = energy_range_to_pixels(roi["energy_range_kev"])
    lo, hi = roi["pixel_range"]
    roi["energy_range_kev"] = (float(pixel_to_kev(lo)), float(pixel_to_kev(hi)))

# %% Discover and validate scan 24 inputs
scan_dir = DATA_ROOT / "Raw" / f"Scan_{SCAN:04d}"
me7_files = numbered_files(scan_dir / "ME7")
xrd_files = numbered_files(scan_dir / "XRD")
position_file = DATA_ROOT / "processed" / "SOCKETSERVER" / f"Scan_{SCAN:04d}_position.h5"

if not me7_files:
    raise FileNotFoundError(f"No ME7 files in {scan_dir / 'ME7'}")
if not xrd_files:
    raise FileNotFoundError(f"No XRD files in {scan_dir / 'XRD'}")
if not position_file.exists():
    raise FileNotFoundError(f"No processed position file: {position_file}")
if not POSITION_OFFSET_FILE.exists():
    raise FileNotFoundError(f"No position offset file: {POSITION_OFFSET_FILE}")
if not SCAN_MASTER_FILE.exists():
    raise FileNotFoundError(f"No scan master file: {SCAN_MASTER_FILE}")

common_numbers = sorted(me7_files.keys() & xrd_files.keys())
if not common_numbers:
    raise ValueError("ME7 and XRD have no matching numbered files")

print(f"Data root:           {DATA_ROOT}")
print(f"Matched ME7/XRD:     {len(common_numbers)} files")
print(f"ME7-only numbers:    {sorted(me7_files.keys() - xrd_files.keys())}")
print(f"XRD-only numbers:    {sorted(xrd_files.keys() - me7_files.keys())}")
for material, roi in XRF_ROIS.items():
    print(
        f"{material:>2}: {roi['pixel_range'][0]}:{roi['pixel_range'][1]} pixels, "
        f"{roi['energy_range_kev'][0]:.3f}:{roi['energy_range_kev'][1]:.3f} keV"
    )

# %% Build the exact sequential ME7/XRD/position registration
with h5py.File(position_file, "r") as handle:
    position_group = handle["entry/data"]
    x_position = np.asarray(position_group["X_Position"][:], dtype=float)
    y_position_raw = np.asarray(position_group["Y_Position"][:], dtype=float)

with h5py.File(SCAN_MASTER_FILE, "r") as handle:
    theta_values = np.asarray(
        handle["entry/instrument/bluesky/streams/baseline/sample_theta/value"][:],
        dtype=float,
    )
scan_theta_deg = float(np.nanmean(theta_values))

with POSITION_OFFSET_FILE.open() as stream:
    position_offset = json.load(stream)
offset_theta = np.asarray(position_offset["theta"], dtype=float)
offset_y = np.asarray(position_offset["y_offset"], dtype=float)
if offset_theta.shape != offset_y.shape or offset_theta.size == 0:
    raise ValueError("position_offset.json theta and y_offset must be non-empty equal arrays")
order = np.argsort(offset_theta)
offset_theta = offset_theta[order]
offset_y = offset_y[order]
nearest = int(np.argmin(np.abs(offset_theta - scan_theta_deg)))
if abs(offset_theta[nearest] - scan_theta_deg) <= 0.01:
    y_offset = float(offset_y[nearest])
else:
    y_offset = float(np.interp(scan_theta_deg, offset_theta, offset_y))
y_position = y_position_raw + y_offset
print(f"Sample theta:        {scan_theta_deg:.6f} deg")
print(f"Applied Y offset:    {y_offset:.3f} (corrected Y = raw Y + offset)")

records = []
global_start = 0
for file_number in common_numbers:
    me7_file = me7_files[file_number]
    xrd_file = xrd_files[file_number]
    with h5py.File(me7_file, "r") as handle:
        me7_count = int(handle[H5_DATASET].shape[0])
    with h5py.File(xrd_file, "r") as handle:
        xrd_data = handle[H5_DATASET]
        xrd_count = int(xrd_data.shape[0]) if xrd_data.ndim == 3 else 1
    mapped_count = min(me7_count, xrd_count)
    records.append({
        "file_number": file_number,
        "me7_file": me7_file,
        "xrd_file": xrd_file,
        "me7_count": me7_count,
        "xrd_count": xrd_count,
        "mapped_count": mapped_count,
        "global_start": global_start,
    })
    global_start += xrd_count

total_xrd_frames = sum(record["xrd_count"] for record in records)
total_mapped_frames = sum(record["mapped_count"] for record in records)
if len(x_position) < total_xrd_frames or len(y_position_raw) < total_xrd_frames:
    raise ValueError(
        f"Position array has {len(x_position)} points but XRD has {total_xrd_frames} frames"
    )

unmatched_me7 = sum(record["me7_count"] - record["mapped_count"] for record in records)
unmatched_xrd = sum(record["xrd_count"] - record["mapped_count"] for record in records)
print(f"XRD frames:          {total_xrd_frames:,}")
print(f"Position points:     {len(x_position):,}")
print(f"Mapped XRF/XRD:      {total_mapped_frames:,}")
print(f"Unmatched ME7/XRD:   {unmatched_me7}/{unmatched_xrd}")

# %% Calibrated overview spectrum
# This is deliberately subsampled by scan line for a fast first pass on /mnt/z.
overview = np.zeros(4096, dtype=np.float64)
sampled_records = records[::max(1, int(SPECTRUM_FILE_STRIDE))]
for record in sampled_records:
    with h5py.File(record["me7_file"], "r") as handle:
        data = handle[H5_DATASET]
        point_slice = slice(None, None, max(1, int(SPECTRUM_POINT_STRIDE)))
        for channel in CHANNELS:
            spectra = data[point_slice, channel, :].astype(np.float64)
            if DEADTIME_CORRECTION:
                factor_path = DT_FACTOR.format(channel1=channel + 1)
                if factor_path in handle:
                    factors = np.asarray(handle[factor_path][point_slice], dtype=float)
                    spectra *= factors[:, None]
            overview += spectra.sum(axis=0)

energy_axis_kev = pixel_to_kev(np.arange(overview.size))
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(energy_axis_kev, overview, linewidth=0.8)
ax.set(
    title=(f"Scan {SCAN}: ME7 spectrum, every {SPECTRUM_FILE_STRIDE}th line "
           f"and {SPECTRUM_POINT_STRIDE}th point"),
    xlabel="calibrated energy (keV)", ylabel="summed counts",
    xlim=(1.2, 15.0),
)
ax.set_yscale("log")
for material, roi in XRF_ROIS.items():
    lo, hi = roi["energy_range_kev"]
    ax.axvspan(lo, hi, alpha=0.15, label=material)
    ax.axvline(roi["line_kev"], linewidth=0.8, linestyle=":")
ax.legend()
ax.grid(alpha=0.2)
plt.show()

# %% Optional interactive ROI adjustment in calibrated energy
# Set ACTIVE_MATERIAL and drag across the desired peak. The callback records both
# calibrated energy and the MCA pixel slice used by the raw-data pass.
ACTIVE_MATERIAL = "In"

if ACTIVE_MATERIAL not in XRF_ROIS:
    raise KeyError(f"Unknown material {ACTIVE_MATERIAL!r}")

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(energy_axis_kev, overview, linewidth=0.8)
ax.set_yscale("log")
ax.set_xlim(1.2, 15.0)
ax.set(title=f"Drag across the {ACTIVE_MATERIAL} peak", xlabel="energy (keV)")


def save_selected_roi(energy_min, energy_max):
    energy_range = tuple(sorted((float(energy_min), float(energy_max))))
    pixel_range = energy_range_to_pixels(energy_range)
    XRF_ROIS[ACTIVE_MATERIAL]["energy_range_kev"] = energy_range
    XRF_ROIS[ACTIVE_MATERIAL]["pixel_range"] = pixel_range
    print(
        f"{ACTIVE_MATERIAL}: {energy_range[0]:.4f}:{energy_range[1]:.4f} keV, "
        f"pixels {pixel_range[0]}:{pixel_range[1]}"
    )


roi_selector = SpanSelector(
    ax, save_selected_roi, "horizontal", useblit=True,
    props={"alpha": 0.3, "facecolor": "tab:orange"}, interactive=True,
)
plt.show()

# %% Efficient point-wise ROI integration
def read_roi_intensities(record):
    """Read only the selected MCA slices for one ME7 scan line."""
    count = record["mapped_count"]
    values = {material: np.zeros(count, dtype=np.float64) for material in XRF_ROIS}
    with h5py.File(record["me7_file"], "r") as handle:
        data = handle[H5_DATASET]
        for channel in CHANNELS:
            factors = None
            if DEADTIME_CORRECTION:
                factor_path = DT_FACTOR.format(channel1=channel + 1)
                if factor_path in handle:
                    factors = np.asarray(handle[factor_path][:count], dtype=float)
            for material, roi in XRF_ROIS.items():
                lo, hi = roi["pixel_range"]
                roi_counts = data[:count, channel, lo:hi].sum(axis=1, dtype=np.float64)
                if factors is not None:
                    roi_counts *= factors
                values[material] += roi_counts
    return values


intensity_chunks = {material: [] for material in XRF_ROIS}
for record in records:
    for material, values in read_roi_intensities(record).items():
        intensity_chunks[material].append(values)

fig, axes = plt.subplots(1, len(XRF_ROIS), figsize=(14, 3.5), constrained_layout=True)
for ax, (material, chunks) in zip(np.atleast_1d(axes), intensity_chunks.items()):
    values = np.concatenate(chunks)
    ax.hist(values, bins=100, log=True)
    ax.set(title=material, xlabel="XRF ROI counts", ylabel="scan points")
    print(
        f"{material}: min={values.min():.3g}, median={np.median(values):.3g}, "
        f"p95={np.percentile(values, 95):.3g}, max={values.max():.3g}"
    )
plt.show()

# %% Set intensity bounds after inspecting the histograms, then build links
# Example: XRF_ROIS["Br"]["intensity_bounds"] = (5000, None)
rows = []
point_number = {material: 0 for material in XRF_ROIS}
for record in records:
    values_by_material = read_roi_intensities(record)
    start = record["global_start"]
    for material, intensities in values_by_material.items():
        lower, upper = XRF_ROIS[material]["intensity_bounds"]
        keep = np.ones(len(intensities), dtype=bool)
        if lower is not None:
            keep &= intensities >= lower
        if upper is not None:
            keep &= intensities <= upper
        for local_index in np.flatnonzero(keep):
            global_index = start + int(local_index)
            rows.append({
                "material": material,
                "point_index": point_number[material],
                "x_position": float(x_position[global_index]),
                "y_position": float(y_position[global_index]),
                "y_position_raw": float(y_position_raw[global_index]),
                "y_position_offset": y_offset,
                "sample_theta_deg": scan_theta_deg,
                "xrf_intensity": float(intensities[local_index]),
                "xrf_roi_lo_kev": XRF_ROIS[material]["energy_range_kev"][0],
                "xrf_roi_hi_kev": XRF_ROIS[material]["energy_range_kev"][1],
                "xrf_file": str(record["me7_file"]),
                "xrf_frame_index": int(local_index),
                "xrd_file": str(record["xrd_file"]),
                "xrd_frame_index": int(local_index),
                "global_frame_index": global_index,
            })
            point_number[material] += 1

columns = [
    "material", "point_index", "x_position", "y_position", "y_position_raw",
    "y_position_offset", "sample_theta_deg", "xrf_intensity",
    "xrf_roi_lo_kev", "xrf_roi_hi_kev", "xrf_file", "xrf_frame_index",
    "xrd_file", "xrd_frame_index", "global_frame_index",
]
filtered_points = pd.DataFrame(rows, columns=columns).set_index(
    ["material", "point_index"]
).sort_index()

print(filtered_points.groupby(level="material").size().rename("retained_points"))
display(filtered_points.head(20))

# %% Save links only; no XRD image data is copied
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
filtered_points.to_csv(OUTPUT_CSV)
print(f"Wrote {len(filtered_points):,} links to {OUTPUT_CSV}")

# %% Registration and physics check
fig, axes = plt.subplots(1, len(XRF_ROIS), figsize=(14, 4), constrained_layout=True)
for ax, material in zip(np.atleast_1d(axes), XRF_ROIS):
    try:
        points = filtered_points.loc[material]
    except KeyError:
        ax.set_title(f"{material}: no retained points")
        ax.set_axis_off()
        continue
    image = ax.scatter(
        points["x_position"], points["y_position"],
        c=points["xrf_intensity"], s=3, cmap="viridis",
    )
    ax.set(title=material, xlabel="X position (um)", ylabel="Y position (um)")
    ax.set_aspect("equal")
    fig.colorbar(image, ax=ax, label="XRF ROI counts")
plt.show()
